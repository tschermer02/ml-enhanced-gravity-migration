"""Benchmark E02 model, gravity, loss, and training-step components."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gravity_migration_ml.ml.e02_dataset import load_e02_sample  # noqa: E402
from gravity_migration_ml.ml.e02_losses import structural_gradient_loss  # noqa: E402
from gravity_migration_ml.ml.e02_physics import gravity_forward_tf  # noqa: E402
from gravity_migration_ml.ml.e02_trainer import E02Trainer  # noqa: E402
from gravity_migration_ml.ml.unet3d import build_unet3d  # noqa: E402

DATA_ROOT = ROOT / "outputs" / "ml" / "training" / "E01" / "data"
GRAVITY_ROOT = SRC / "data" / "dataset" / "forward_models"


def synchronize(value: object) -> None:
    """Materialize returned tensors so GPU work finishes before stopping timer."""
    if isinstance(value, tf.Tensor):
        value.numpy()
    elif isinstance(value, dict):
        for item in value.values():
            synchronize(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            synchronize(item)


def memory_info() -> dict[str, float] | None:
    """Return TensorFlow GPU allocator current/peak bytes where supported."""
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
    except (ValueError, RuntimeError):
        return None
    return {key: value / (1024.0 ** 2) for key, value in info.items()}


def benchmark_case(name: str, function: Callable[[], object], *, warmup: int,
                   iterations: int, trace: Callable[[], object] | None = None) -> dict[str, float | None]:
    """Report explicit trace time, first execution, and synchronized steady state."""
    trace_seconds: float | None = None
    if trace is not None:
        started = time.perf_counter()
        trace()
        trace_seconds = time.perf_counter() - started

    started = time.perf_counter()
    first_result = function()
    synchronize(first_result)
    first_execution_seconds = time.perf_counter() - started

    for _ in range(warmup):
        synchronize(function())

    before_memory = memory_info()
    timings = []
    for _ in range(iterations):
        started = time.perf_counter()
        result = function()
        synchronize(result)
        timings.append(time.perf_counter() - started)
    after_memory = memory_info()

    average = float(np.mean(timings))
    result = {
        "trace_seconds": trace_seconds,
        "first_execution_seconds": first_execution_seconds,
        "steady_seconds_per_step": average,
        "steady_samples_per_second": 1.0 / average,
        "gpu_memory_before_mib": before_memory.get("current") if before_memory else None,
        "gpu_memory_peak_before_mib": before_memory.get("peak") if before_memory else None,
        "gpu_memory_after_mib": after_memory.get("current") if after_memory else None,
        "gpu_memory_peak_after_mib": after_memory.get("peak") if after_memory else None,
    }
    print(f"{name}: {json.dumps(result)}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--gravity-root", type=Path, default=GRAVITY_ROOT)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--sample-id", default="sample_000000")
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup must be >= 0 and --iterations must be >= 1")

    data_root = args.data_root if args.data_root.is_absolute() else ROOT / args.data_root
    gravity_root = args.gravity_root if args.gravity_root.is_absolute() else ROOT / args.gravity_root
    migration_path = data_root / args.split / f"{args.sample_id}.npz"
    migration_input, targets = load_e02_sample(migration_path, gravity_root)
    inputs = tf.convert_to_tensor(np.repeat(migration_input[None, ...], args.batch_size, axis=0))
    density = tf.convert_to_tensor(np.repeat(targets["density"][None, ...], args.batch_size, axis=0))
    gravity = tf.convert_to_tensor(np.repeat(targets["gravity"][None, ...], args.batch_size, axis=0))
    variable = tf.Variable(density)

    network = build_unet3d()
    weights = {"lambda_model": 1.0, "lambda_phys": 0.1,
               "lambda_struct": 0.1, "lambda_reg": 0.0}
    trainer = E02Trainer(network, weights, gravity_normalization_mgal=3.5)
    trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), jit_compile=False)
    trainer.optimizer.build(network.trainable_variables)

    @tf.function(jit_compile=False, reduce_retracing=True)
    def unet_forward() -> tf.Tensor:
        return network(inputs, training=False)

    @tf.function(jit_compile=False, reduce_retracing=True)
    def unet_forward_backward() -> tf.Tensor:
        with tf.GradientTape() as tape:
            prediction = network(inputs, training=True)
            loss = tf.reduce_mean(tf.square(prediction - density))
        gradients = tape.gradient(loss, network.trainable_variables)
        return tf.add_n([tf.reduce_sum(gradient) for gradient in gradients if gradient is not None])

    @tf.function(jit_compile=False, reduce_retracing=True)
    def gravity_with_gradient() -> tf.Tensor:
        with tf.GradientTape() as tape:
            tape.watch(variable)
            prediction = gravity_forward_tf(variable)
            loss = tf.reduce_mean(tf.square(prediction))
        gradient = tape.gradient(loss, variable)
        return tf.reduce_sum(gradient)

    @tf.function(jit_compile=False, reduce_retracing=True)
    def structural_only() -> tf.Tensor:
        prediction = network(inputs, training=False)
        return structural_gradient_loss(density, prediction)

    train_step = tf.function(trainer.train_step, jit_compile=False, reduce_retracing=True)
    step_data = (inputs, {"density": density, "gravity": gravity})

    cases = [
        ("unet_forward", unet_forward, lambda: unet_forward.get_concrete_function()),
        ("unet_forward_backward", unet_forward_backward,
         lambda: unet_forward_backward.get_concrete_function()),
        ("gravity_forward", lambda: gravity_forward_tf(variable),
         lambda: gravity_forward_tf.get_concrete_function(tf.TensorSpec(variable.shape, tf.float32))),
        ("gravity_forward_gradient", gravity_with_gradient,
         lambda: gravity_with_gradient.get_concrete_function()),
        ("structural_loss", structural_only,
         lambda: structural_only.get_concrete_function()),
        ("complete_e02_train_step", lambda: train_step(step_data),
         lambda: train_step.get_concrete_function(step_data)),
    ]
    device_list = tf.config.list_physical_devices("GPU")
    print(json.dumps({
        "tensorflow_version": tf.__version__,
        "jit_compile": False,
        "physical_gpus": [device.name for device in device_list],
        "logical_gpus": [device.name for device in tf.config.list_logical_devices("GPU")],
        "batch_size": args.batch_size,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "gravity_kernel_constant_shape": [24, 256, 256],
        "gravity_kernel_complex64_mib": 24 * 256 * 256 * 8 / (1024.0 ** 2),
    }, indent=2), flush=True)

    results = {}
    for name, function, trace in cases:
        try:
            results[name] = benchmark_case(
                name, function, warmup=args.warmup,
                iterations=args.iterations, trace=trace,
            )
        except (RuntimeError, tf.errors.OpError) as error:
            print(f"Benchmark failed for {name}: {error}", flush=True)
            raise

    if args.output is not None:
        output_path = args.output if args.output.is_absolute() else ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Benchmark results saved: {output_path}")


if __name__ == "__main__":
    main()

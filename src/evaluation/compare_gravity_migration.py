"""Compare the Python migration operator with the original MATLAB implementation."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from physics.gravity_migration import (  # noqa: E402
    CELL_SIZE,
    NX,
    NY,
    NZ,
    OBS_X,
    OBS_Y,
    calculate_migration,
)

MATLAB_DIR = SRC / "physics" / "MATLAB" / "GGmigration_2014"
MATLAB_SOURCE = MATLAB_DIR / "gg_migration2014.m"
MATLAB_RUNNER = MATLAB_DIR / "run_migration_comparison.m"


def sample_files(dataset: Path) -> list[Path]:
    """Find NPZ cases in a split directory or dataset root."""
    if any(dataset.glob("*.npz")):
        return sorted(dataset.glob("*.npz"))
    for split in ("validation", "val", "test"):
        split_dir = dataset / split
        files = sorted(split_dir.glob("*.npz"))
        if files:
            return files
    return []


def write_matlab_inputs(work_dir: Path, samples: list[tuple[str, np.ndarray]]) -> None:
    """Write inputs in the layout expected by gg_migration2014.m."""
    grid_x, grid_y = np.meshgrid(OBS_X, OBS_Y)
    receiver_rows = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    model_x = (np.arange(NX) + 0.5) * CELL_SIZE
    model_y = (np.arange(NY) + 0.5) * CELL_SIZE
    domain_x, domain_y = np.meshgrid(model_x, model_y)
    np.savetxt(work_dir / "domain.dat",
               np.column_stack((domain_x.ravel(), domain_y.ravel())), fmt="%.17g")

    case_ids = []
    for sample_id, gravity in samples:
        case_name = f"case_{sample_id}.dat"
        case_ids.append(case_name)
        rows = np.column_stack((
            receiver_rows,
            np.zeros(receiver_rows.shape[0]),
            np.zeros(receiver_rows.shape[0]),
            gravity.ravel(order="C"),
        ))
        np.savetxt(work_dir / case_name, rows, fmt="%.17g")

    (work_dir / "case_ids.txt").write_text("\n".join(case_ids) + "\n", encoding="ascii")
    (work_dir / "current_case.txt").write_text(case_ids[0], encoding="ascii")
    (work_dir / "input_gg_mig.m").write_text(
        "filename = strtrim(fileread('current_case.txt'));\n"
        "migid = [1];\n"
        "migjoint = [];\n"
        "modelid = 1;\n"
        "domainfile = 'domain.dat';\n"
        "xstep = 10;\n"
        "ystep = 10;\n"
        "mz = 24;\n"
        "zmin = 0;\n"
        "zmax = 240;\n"
        "delta_x = 10;\n"
        "delta_y = 10;\n"
        "restrictid = 0;\n"
        "ro_min = 0;\n"
        "ro_max = 1;\n",
        encoding="ascii",
    )


def run_matlab(samples: list[tuple[str, np.ndarray]], matlab: str) -> dict[str, np.ndarray]:
    """Run the original MATLAB function in an isolated staging directory."""
    matlab_executable = shutil.which(matlab)
    if matlab_executable is None and Path(matlab).is_file():
        matlab_executable = str(Path(matlab).resolve())
    if matlab_executable is None:
        raise FileNotFoundError(
            f"MATLAB executable not found: {matlab!r}. Pass --matlab with its full path."
        )

    with tempfile.TemporaryDirectory(prefix="gravity_migration_matlab_") as temporary:
        work_dir = Path(temporary)
        matlab_source = MATLAB_SOURCE.read_text(encoding="utf-8")
        ascii_save = "save(outname,'dms','-ascii');"
        precise_ascii_save = "save(outname,'dms','-ascii','-double');"
        if ascii_save not in matlab_source:
            raise ValueError("Could not locate MATLAB migration output statement")
        (work_dir / "gg_migration.m").write_text(
            matlab_source.replace(ascii_save, precise_ascii_save), encoding="utf-8"
        )
        shutil.copy2(MATLAB_RUNNER, work_dir / "run_migration_comparison.m")
        write_matlab_inputs(work_dir, samples)
        matlab_dir = work_dir.as_posix().replace("'", "''")
        command = f"cd('{matlab_dir}'); run_migration_comparison"
        subprocess.run([matlab_executable, "-batch", command], check=True)

        results = {}
        for sample_id, _ in samples:
            matlab_data = np.loadtxt(work_dir / f"den_case_{sample_id}.dat", ndmin=2)
            expected_rows = NX * NY * NZ
            if matlab_data.shape != (expected_rows, 4):
                raise ValueError(
                    f"{sample_id}: MATLAB result shape {matlab_data.shape}; "
                    f"expected ({expected_rows}, 4)"
                )
            results[sample_id] = matlab_data[:, 3].reshape(NZ, NY, NX)
        return results


def metrics(python_result: np.ndarray, matlab_result: np.ndarray,
            atol: float, rtol: float) -> dict[str, object]:
    difference = python_result - matlab_result
    matlab_norm = float(np.linalg.norm(matlab_result))
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "max_abs_error": float(np.max(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference) / matlab_norm) if matlab_norm else 0.0,
        "passed": bool(np.allclose(python_result, matlab_result, atol=atol, rtol=rtol)),
    }


def save_comparison_plot(output: Path, label: str, sample_id: str,
                         python_result: np.ndarray, matlab_result: np.ndarray,
                         rmse: float) -> None:
    """Save the center-depth slice from both operators and their difference."""
    depth_idx = NZ // 2
    python_slice = python_result[depth_idx]
    matlab_slice = matlab_result[depth_idx]
    difference = python_slice - matlab_slice
    value_min = min(float(python_slice.min()), float(matlab_slice.min()))
    value_max = max(float(python_slice.max()), float(matlab_slice.max()))
    error_limit = max(float(np.max(np.abs(difference))), 1e-15)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, image, title, cmap, vmin, vmax in (
        (axes[0], python_slice, "Python migration", "viridis", value_min, value_max),
        (axes[1], matlab_slice, "MATLAB migration", "viridis", value_min, value_max),
        (axes[2], difference, "Python - MATLAB", "RdBu_r", -error_limit, error_limit),
    ):
        plot = axis.imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set(title=title, xlabel="X cell", ylabel="Y cell")
        fig.colorbar(plot, ax=axis)
    fig.suptitle(f"{label}: {sample_id} | center-depth RMSE={rmse:.3e}")
    fig.savefig(output / f"{label}_{sample_id}.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/synthetic/single_prism"))
    parser.add_argument("--output", type=Path,
                        default=Path("analysis_outputs/gravity_migration_validation"))
    parser.add_argument("--matlab", default="matlab", help="MATLAB executable or full path")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--rtol", type=float, default=1e-9)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    files = sample_files(args.dataset)
    if args.limit is not None:
        files = files[:args.limit]
    if not files:
        raise FileNotFoundError(f"No NPZ samples found under {args.dataset}")

    samples = []
    for path in files:
        with np.load(path) as archive:
            gravity = np.asarray(archive["gravity"], dtype=np.float64)
        if gravity.shape != (len(OBS_Y), len(OBS_X)) or not np.isfinite(gravity).all():
            raise ValueError(f"{path.name}: gravity must be finite with shape (81, 81)")
        samples.append((path.stem, gravity))

    matlab_results = run_matlab(samples, args.matlab)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    visual = []
    for index, (sample_id, gravity) in enumerate(samples, 1):
        python_result = calculate_migration(gravity)
        matlab_result = matlab_results[sample_id]
        row = {"sample_id": sample_id, **metrics(python_result, matlab_result, args.atol, args.rtol)}
        rows.append(row)
        visual.append((row["rmse"], sample_id, python_result, matlab_result))
        print(f"Compared {index}/{len(samples)}: {sample_id}; "
              f"max diff {row['max_abs_error']:.3e}; pass={row['passed']}", flush=True)

    with (args.output / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "sample_count": len(rows),
        "passed_count": sum(row["passed"] for row in rows),
        "all_passed": all(row["passed"] for row in rows),
        "atol": args.atol,
        "rtol": args.rtol,
        "mean_rmse": float(np.mean([row["rmse"] for row in rows])),
        "maximum_absolute_error": float(max(row["max_abs_error"] for row in rows)),
    }
    (args.output / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    for label, case in (("best", min(visual)), ("worst", max(visual))):
        rmse, sample_id, python_result, matlab_result = case
        save_comparison_plot(args.output, label, sample_id,
                             python_result, matlab_result, rmse)

    print(json.dumps(summary, indent=2))
    if not summary["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

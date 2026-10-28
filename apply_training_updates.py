"""Update existing experiment training scripts to the consolidated src modules.

Run from the repository root:
    python apply_training_updates.py

This tool backs up each modified file to .training_backup/ and does not remove
experiment files. Review the generated diff before training.
"""
from pathlib import Path
import re
import shutil
import ast

ROOT = Path.cwd()
TRAINING = ROOT / "src" / "training"
BACKUP = ROOT / ".training_backup"

# Local compatibility names preserve the existing experiment objectives and
# plotting callbacks; the implementations live in the unified losses module.
COMMON = """
from src.data.dataset import make_dataset, sample_paths
from src.models.unet3d import build_unet3d
from src.losses.losses import (
    ordinary_mse, balanced_density_mse, balanced_huber_loss,
    soft_tversky_loss, vertical_profile_loss, make_loss,
)

TVERSKY_WEIGHT = 0.1
TVERSKY_ALPHA = 0.7
TVERSKY_BETA = 0.3
DENSITY_THRESHOLD = 0.1
OCCUPANCY_TEMPERATURE = 0.025
DEPTH_WEIGHT = 0.02
THRESHOLD = DENSITY_THRESHOLD
TEMP = OCCUPANCY_TEMPERATURE

e03_combined_loss = make_loss(weights={"balanced_mse": 1, "tversky": TVERSKY_WEIGHT})
e05_combined_loss = make_loss(weights={"balanced_mse": 1, "tversky": TVERSKY_WEIGHT, "depth": DEPTH_WEIGHT})
signed_e06_loss = make_loss(weights={"balanced_mse": 1, "tversky": TVERSKY_WEIGHT, "depth": DEPTH_WEIGHT}, signed=True)

def signed_balanced_density_mse(y_true, y_pred):
    return balanced_density_mse(y_true, y_pred, signed=True)

def signed_tversky(y_true, y_pred):
    return soft_tversky_loss(y_true, y_pred, signed=True, subtract_baseline=True)

def signed_vertical_profile(y_true, y_pred):
    return vertical_profile_loss(y_true, y_pred, signed=True)
"""
# Removing import statements as complete AST spans is safer than regex on
# multiline imports, and leaves unrelated imports untouched.
def replace_imports(source):
    tree = ast.parse(source)
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and (
            node.module.startswith("gravity_migration_ml.ml.")
        ):
            spans.append((node.lineno, node.end_lineno))
    lines = source.splitlines(keepends=True)
    for first, last in reversed(sorted(spans)):
        lines[first-1:last] = []
    return "".join(lines)

def patch_file(path):
    original = path.read_text(encoding="utf8")
    revised = replace_imports(original)
    # From src/training, the repository root is parents[2].
    revised = re.sub(
        r'Path\(__file__\)\.resolve\(\)\.parents\[1\]',
        'Path(__file__).resolve().parents[2]', revised
    )
    # Existing module inserts sometimes refer to the removed src/scripts tree;
    # benign when it does, but no longer needed.
    revised = revised.replace("ROOT/'scripts'", "ROOT/'src'")
    # Import the consolidated modules and bind familiar objective names.
    marker = "\n# BEGIN UNIFIED TRAINING IMPORTS\n"
    revised = revised.replace("from __future__ import annotations",
        "from __future__ import annotations" + marker + COMMON + "\n# END UNIFIED TRAINING IMPORTS\n", 1)
    # Signed loader is now the same loader; replace function calls.
    revised = revised.replace("signed_dataset(", "make_dataset(")
    ast.parse(revised, filename=str(path))
    if revised != original:
        dest = BACKUP / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        path.write_text(revised, encoding="utf8")
    return revised != original

def main():
    if not TRAINING.is_dir():
        raise SystemExit("Run from the repository root (src/training not found)")
    files = sorted(TRAINING.glob("train_unet*.py"))
    for path in files:
        print(("Updated: " if patch_file(path) else "Unchanged: ") + str(path.relative_to(ROOT)))
    trainer = TRAINING / "e02_trainer.py"
    if trainer.exists():
        original = trainer.read_text(encoding="utf8")
        old = "from gravity_migration_ml.ml.e02_losses import e02_loss_components"
        if old in original:
            compat = '''from src.losses.losses import loss_components
from src.physics.gravity_forward_tf import gravity_forward_tf


def e02_loss_components(density_true, density_predicted, gravity_observed,
                        *, lambda_model=1.0, lambda_phys=0.1,
                        lambda_struct=0.1, lambda_reg=0.0,
                        huber_delta=0.1, gravity_normalization_mgal=3.5,
                        cell_spacing_m=10.0):
    terms = loss_components(
        density_true, density_predicted,
        weights={"huber": lambda_model, "physics": lambda_phys,
                 "gradient": lambda_struct},
        huber_delta=huber_delta, cell_spacing_m=cell_spacing_m,
        gravity_observed=gravity_observed,
        forward_model=gravity_forward_tf,
        gravity_normalization=gravity_normalization_mgal,
    )
    return {"total_loss": terms["total"],
            "model_loss": terms.get("huber", tf.constant(0.0)),
            "physics_loss": terms.get("physics", tf.constant(0.0)),
            "struct_loss": terms.get("gradient", tf.constant(0.0)),
            "regularization_loss": tf.constant(0.0),
            "gravity_predicted": gravity_forward_tf(density_predicted)}
'''
            revised = original.replace(old, compat)
            ast.parse(revised, filename=str(trainer))
            dest = BACKUP / trainer.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trainer, dest)
            trainer.write_text(revised, encoding="utf8")
            print("Updated: src/training/e02_trainer.py")
    print("\nBackups saved in .training_backup/")
    print("Check git diff and run one-epoch smoke tests before deleting any old modules.")

if __name__ == "__main__":
    main()

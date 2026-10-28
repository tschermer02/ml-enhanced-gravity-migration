"""Apply visualization consolidation using original local source, preserving functions.

Run once from the repository root: python apply_visualization_refactor.py
Keeps backups in .visualization_backup/ and leaves old scripts available for review.
"""
from pathlib import Path
import ast
import shutil

root = Path.cwd()
folder = root / "src" / "visualization"
backup = root / ".visualization_backup"
if not (folder / "export_three_panel_cross_sections.py").exists():
    raise SystemExit("Run from repository root with the original visualization files present.")

for filename in ("plotting.py", "plotly_3d.py", "export_three_panel_cross_sections.py",
                 "visualization.py", "compare_dataset_models.py", "export_prediction_gifs.py"):
    source = folder / filename
    if source.exists():
        target = backup / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

# Extract plotting functions, preserving original algorithms and constants.
cross = (folder / "export_three_panel_cross_sections.py").read_text(encoding="utf8")
tree = ast.parse(cross)
needed = {"_name", "_load", "_components", "_slices", "_figure", "export_one"}
functions = [cross.splitlines(keepends=True)[node.lineno-1:node.end_lineno]
             for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in needed]
if len(functions) != len(needed):
    raise RuntimeError("Cross-section functions changed; review source before merging.")

plot_path = folder / "plotting.py"
plot = plot_path.read_text(encoding="utf8")
if "def export_one(" not in plot:
    additions = ("\n# Cross-section comparison tools (from export_three_panel_cross_sections.py)\n"
                 "from scipy import ndimage\n"
                 "from matplotlib.colors import TwoSlopeNorm\n"
                 "SHAPE = (24, 64, 64)\n"
                 "DENSITY_THRESHOLD = 0.1\n"
                 "COLUMNS = ('Synthetic Model', 'Migration', 'Inversion')\n"
                 "ROWS = ('XY', 'XZ', 'YZ')\n\n")
    additions += "\n\n".join("".join(f) for f in functions) + "\n"
    plot_path.write_text(plot + additions, encoding="utf8")

prism = (folder / "visualization.py").read_text(encoding="utf8")
tree = ast.parse(prism)
needed_prism = {"_faces", "plot_prism_3d"}
prism_funcs = ["".join(prism.splitlines(keepends=True)[n.lineno-1:n.end_lineno])
               for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in needed_prism]
if len(prism_funcs) != 2:
    raise RuntimeError("Prism functions changed; review before merging.")
path3d = folder / "plotly_3d.py"
code3d = path3d.read_text(encoding="utf8")
if "def plot_prism_3d(" not in code3d:
    code3d += ("\n# Matplotlib prism overview (from visualization.py)\n"
               "import matplotlib.pyplot as plt\n"
               "from mpl_toolkits.mplot3d.art3d import Poly3DCollection\n\n"
               + "\n\n".join(prism_funcs) + "\n")
    path3d.write_text(code3d, encoding="utf8")

src = Path(__file__).resolve().parent / "export_figures.py"
dest = folder / "export_figures.py"
shutil.copy2(src, dest)
for p in (plot_path, path3d, dest):
    ast.parse(p.read_text(encoding="utf8"), filename=str(p))
print("Visualization modules updated. Old scripts were not deleted.")
print("Review git diff; then smoke-test --type cross_sections, gif, dataset.")

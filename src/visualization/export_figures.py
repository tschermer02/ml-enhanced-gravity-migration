"""Export dataset comparisons, prediction cross-sections, and 3D GIFs.

Run from repository root:
 python -m src.visualization.export_figures --name E07 --type cross_sections --limit 2
 python -m src.visualization.export_figures --name E07 --type gif --samples 1 2
 python -m src.visualization.export_figures --name E07 --type dataset --split train --limit 2
 python -m src.visualization.export_figures --all --type cross_sections --limit 1
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from src.visualization.plotting import plot_volume_comparison, export_one
from src.visualization.plotly_3d import plot_sample_3d

ROOT = Path(__file__).resolve().parents[2]


def sample_name(text):
    stem = str(text).strip().removesuffix(".npz")
    if stem.startswith("sample_"):
        stem = stem[7:]
    return f"sample_{int(stem):06d}" if stem.isdigit() else stem


def prediction_folders(base, name, split, all_runs):
    if all_runs:
        return sorted(base.glob(f"**/predictions/{split}"))
    root = base / name
    folders = sorted(root.glob(f"**/predictions/{split}"))
    if not folders:
        raise FileNotFoundError(f"No prediction folders beneath {root}")
    return folders


def selected_files(folder, samples, limit):
    files = sorted(folder.glob("*.npz"))
    if samples:
        wanted = {sample_name(s) for s in samples}
        files = [f for f in files if f.stem in wanted]
    return files[:limit] if limit is not None else files


def export_dataset(args):
    root = ROOT / "outputs" / "ml" / "training" / args.dataset / "data"
    folder = root / args.split
    for path in selected_files(folder, args.samples, args.limit):
        with np.load(path) as data:
            truth = np.asarray(data["density"], dtype=np.float32)
            migration = np.asarray(data["migration"], dtype=np.float32)
        relative = path.parent.relative_to(folder)
        destination = ROOT / "outputs" / "ml" / "predict" / args.name / relative / "figures" / args.split / f"{path.stem}_dataset.png"
        if destination.exists() and not args.overwrite:
            continue
        plot_volume_comparison(truth, migration, destination,
                               estimate_label="Migration", title=path.stem)
        print(f"Saved {destination}", flush=True)


def export_prediction(args):
    base = ROOT / "outputs" / "ml" / "predict"
    for folder in prediction_folders(base, args.name, args.split, args.all):
        for path in selected_files(folder, args.samples, args.limit):
            if args.type == "cross_sections":
                outputs, bodies = export_one(
                    path, base, args.split, args.threshold, args.connectivity,
                    args.min_body_voxels, args.migration_source, args.dpi,
                    args.overwrite, not args.no_per_body,
                )
                print(f"{path.stem}: {bodies} bodies, {len(outputs)} figures", flush=True)
            else:
                with np.load(path) as data:
                    truth = np.asarray(data["density"], dtype=np.float32).squeeze()
                    prediction = np.asarray(data["prediction"], dtype=np.float32).squeeze()
                out = folder.parent.parent / "figures" / args.split / f"{path.stem}_3d.gif"
                if out.exists() and not args.overwrite:
                    continue
                plot_sample_3d(
                    truth, None, out, reconstruction=prediction,
                    reconstruction_threshold=args.reconstruction_threshold,
                    make_gif=True, gif_frames=args.frames,
                    gif_duration_seconds=args.duration,
                )
                print(f"Saved {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", help="Experiment name or nested path, e.g. E07/multiple/noise_05")
    parser.add_argument("--all", action="store_true", help="Find all prediction folders")
    parser.add_argument("--dataset", help="Dataset name for --type dataset (defaults to --name)")
    parser.add_argument("--type", choices=("cross_sections", "gif", "dataset"), required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--samples", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--connectivity", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--min-body-voxels", type=int, default=8)
    parser.add_argument("--migration-source", choices=("normalized", "raw"), default="normalized")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--no-per-body", action="store_true")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--duration", type=float, default=8.)
    parser.add_argument("--reconstruction-threshold", type=float, default=0.25)
    args = parser.parse_args()
    if not args.name and not args.all:
        parser.error("Use --name or --all")
    if args.type == "dataset":
        if args.all:
            parser.error("--all is only for prediction figures")
        args.dataset = args.dataset or args.name
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.frames < 2 or args.duration <= 0 or not (0 < args.reconstruction_threshold < 1):
        parser.error("Invalid GIF settings")
    if args.type == "dataset":
        export_dataset(args)
    else:
        export_prediction(args)


if __name__ == "__main__":
    main()

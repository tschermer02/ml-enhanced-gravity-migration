"""Patch existing src/physics/gravity_consistency.py to add opt-in E11 mode.
Run from repository root: python update_gravity_consistency.py
Preserves the existing module verbatim except its __main__ dispatcher.
"""
from pathlib import Path
p=Path('src/physics/gravity_consistency.py')
if not p.exists(): raise SystemExit('Run from repository root')
s=p.read_text()
if 'def main_e11(' in s:
    print('E11 gravity consistency extension already installed');raise SystemExit(0)
anchor="if __name__ == '__main__':"
# Most modules use double quotes; accept both.
import re
pattern=r'if __name__\s*==\s*[\'\"]__main__[\'\"]\s*:\s*main\(\)\s*$'
if not re.search(pattern,s):
    raise SystemExit('Could not locate expected main() guard: inspect file, no changes made')
addition='''

# ---- E11 optional: height-aware anisotropic Gzz validation -----------------
def main_e11():
    """Validate E11 saved synthetic predictions against their gravity data."""
    import json
    import csv
    import argparse
    from collections import defaultdict
    from src.physics.gravity_forward import calculate_gravity, observation_coordinates

    parser = argparse.ArgumentParser(description="E11 physics-consistency reports")
    parser.add_argument("--name", required=True)
    parser.add_argument("--predictions-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("test", "validation", "train"), default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    source = args.predictions_root or ROOT / "outputs/ml/predict" / args.name
    out = args.output or source / "gravity_consistency" / args.split
    source, out = Path(source), Path(out)
    files = sorted(source.glob(f"**/predictions/{args.split}/*.npz"))
    if args.limit is not None:
        files = files[:args.limit]
    if not files:
        raise FileNotFoundError(f"No predictions in {source} for split {args.split}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    residuals = []
    first_by_geometry = {}
    for i, path in enumerate(files, 1):
        with np.load(path, allow_pickle=False) as archive:
            required = {"prediction", "gravity", "dx_m", "dy_m", "dz_m", "observation_height_m"}
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"{path}: missing {sorted(missing)}")
            rho = np.asarray(archive["prediction"], dtype=np.float32).squeeze()
            obs = np.asarray(archive["gravity"], dtype=np.float64).squeeze()
            dx, dy, dz = (float(archive[k]) for k in ("dx_m", "dy_m", "dz_m"))
            height = float(archive["observation_height_m"])
            geom = str(archive["geometry"].item()) if "geometry" in archive else "unknown"
            noise = float(archive["noise_percent"]) if "noise_percent" in archive else 0.0
            component = str(archive["observation_component"].item()).lower() if "observation_component" in archive else "gzz"
        if component != "gzz":
            raise ValueError(f"E11 mode requires Gzz; {path} has {component}")
        if rho.shape != (24, 64, 64) or obs.shape != (81, 81):
            raise ValueError(f"Invalid E11 shapes: {rho.shape}, {obs.shape}")
        modeled = calculate_gravity(rho, component="gzz", dx_m=dx, dy_m=dy,
                                    dz_m=dz, observation_height_m=height)
        diff = obs - modeled
        rmse = float(np.sqrt(np.mean(diff * diff)))
        mae = float(np.mean(np.abs(diff)))
        correlation = (float(np.corrcoef(obs.ravel(), modeled.ravel())[0, 1])
                       if np.std(obs) > 0 and np.std(modeled) > 0 else None)
        rows.append(dict(sample=path.stem, geometry=geom, noise_percent=noise,
                         observation_height_m=height, rmse_e=rmse, mae_e=mae,
                         bias_e=float(np.mean(diff)), correlation=correlation))
        residuals.append(diff.ravel())
        first_by_geometry.setdefault(geom, (obs, modeled, diff, dx, dy, height))
        print(f"{i}/{len(files)} {geom} at {height:g}m Gzz RMSE={rmse:.4g} E", flush=True)

    with (out / "gravity_metrics_per_sample.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    def stats(items):
        return dict(samples=len(items), mean_rmse_e=float(np.mean([r["rmse_e"] for r in items])),
                    mean_mae_e=float(np.mean([r["mae_e"] for r in items])))

    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["geometry"], r["noise_percent"], r["observation_height_m"])].append(r)
    summary = dict(overall=stats(rows), by_geometry_noise_height={
        f"{g}|noise_{n:g}|height_{h:g}m":stats(v) for (g,n,h),v in sorted(groups.items())})
    (out / "gravity_summary.json").write_text(json.dumps(summary,indent=2))

    for geometry, (obs, modeled, diff, dx, dy, height) in first_by_geometry.items():
        ox, oy = observation_coordinates(dx, dy)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
        bound=max(np.max(np.abs(obs)), np.max(np.abs(modeled)), 1e-9)
        for ax, data, title, limit in zip(axes, (obs, modeled, diff),
                ("Observed Gzz", "Forward modeled Gzz", "Residual Gzz"),
                (bound, bound, max(np.max(np.abs(diff)), 1e-9))):
            im=ax.imshow(data, origin="lower", cmap="RdBu_r", vmin=-limit,vmax=limit,
                         extent=(ox[0]/1000,ox[-1]/1000,oy[0]/1000,oy[-1]/1000))
            ax.set(xlabel="X (km)",ylabel="Y (km)",title=title)
            fig.colorbar(im,ax=ax,shrink=.8,label="Gzz (Eötvös)")
        fig.suptitle(f"{geometry} | receiver height {height:g} m")
        fig.savefig(out / f"gravity_comparison_{geometry}.png",dpi=args.dpi)
        plt.close(fig)
    print("Saved E11 gravity reports:", out)


'''
s=re.sub(pattern,addition+'\nif __name__ == "__main__":\n    import sys\n    if "--e11" in sys.argv:\n        sys.argv.remove("--e11")\n        main_e11()\n    else:\n        main()\n',s)
compile(s,str(p),'exec')
p.write_text(s)
print('Updated:',p)

# E12 — Thunderbird-geometry gravity inversion (isolated sandbox)

All files live inside `src/E12/`. This is a **new** self-contained experiment family. Nothing in `src/physics/`, `src/data/`, `src/training/`, `src/prediction/`, or `src/evaluation/` is changed. Run all commands from the repository root. Output files are written in new `outputs/ml/training/D12_*`, `outputs/ml/training/E12*`, `outputs/ml/predict/E12*`, and `outputs/ml/field/Thunderbird_E12` directories.

## Physics and survey assumptions

- Receiver grid **81×81**, **100 m pitch**, coordinates 0–8000 m in both horizontal axes.
- Density grid **24×64×64**, **125×125×41.6667 m** cells, total **8×8×1 km**. Receiver domain matches the density domain (no margins).
- The physical receiver locations and density-cell centers differ. An isolated FFT forward operator computes the exact cell-centered point-mass Gzz kernel using a staggered common 25 m computational lattice. No interpolation of density or Gzz is used in the synthetic forward or migration operators.
- Data are Gzz in **Eötvös**, density contrast in **g/cm³**, heights specified above a *flat model top* in meters.
- Jorgensen & Zhdanov (2021), *Minerals*, 11, 366, §6.2, report **100 m nominal airborne flight height** and **250 m flight-line spacing** for the McFaulds Lake survey. These are *survey specifications*, not proof that the supplied `Tbird_gravity.dat` uses a constant observation datum. The paper reports **50 m horizontal cells** and logarithmic vertical layers for its case-study inversion (§6.3); E12 is an intentionally different experiment retaining the ML model dimensions.
- Field channel 7 is *assumed* to be Gzz in Eötvös. Verify the file's channel/unit conventions separately. Source X/Y coordinate ranges are preserved without stretching.
- The field `--height` defaults to 100 m but does not correct for actual topography or terrain clearance. No E12 paper-based guarantee is made about individual observation heights.
- Field-data inference is *strict*: if any of the 81×81 receiver nodes is outside the measured convex hull, it **stops**, leaving the coverage map; the input is **never** filled silently. On a survey with incomplete corners, E12 synthetic training can still proceed, but this strict field inference will not proceed until a scientifically justified coverage policy is chosen.

## 1. Install from repository root

Extract the archive into the repository root. It should create only `src/E12/` files. The repository already contains `src` package. You need `numpy`, `scipy`, `matplotlib`, `plotly`, and `tensorflow` (for training/inference) in the `gravity-tf` environment. No pytest needed.

## 2. Validate physics BEFORE data generation or training

```bash
python -m unittest src.E12.tests.test_physics -v
```

Checks NumPy vs direct point-mass response at three heights, sign correctness, exact discrete unweighted adjoint, TF parity at three heights, gradient existence, and gradient finite differences. **Do not train physics-constrained models until the TensorFlow checks pass.**

## 3. Explore Thunderbird input without any inversion

```bash
python -m src.E12.field --prepare-only
```

Saved to `outputs/ml/field/Thunderbird_E12/`:
- `observed_Gzz_original_vs_81.png`
- `interpolation_coverage_map.png`
- `Thunderbird_Gzz_81x81.npz` (outside-hull nodes remain NaN)
- `preprocessing_metadata.json` (true X/Y bounds, original physical spacing, coverage, zero stretch factors)

Linear interpolation within convex hull does not invent measurements, but interpolated grid values are **not** independent observations. Nearest-station distance diagnostics are provided to assess gaps between 250 m flight lines. Original source gravity amplitudes are not rescaled.

## 4. Generate datasets (run each once)

Run a short smoke dataset using reduced counts FIRST, with separate output names/config? A smoke run with default D12_10 creates a configuration; delete the smoke outputs *before* generating the full dataset, or run directly at full size after tests. Config guards refuse silent overwrites/mismatched counts.

Full reproducible E12 datasets (2400 train, 240 validation, 480 test; geometry/noise balanced):

```bash
python -m src.E12.generate --dataset D12_07 --train 2400 --validation 240 --test 480
python -m src.E12.generate --dataset D12_08 --train 2400 --validation 240 --test 480
python -m src.E12.generate --dataset D12_09 --train 2400 --validation 240 --test 480
python -m src.E12.generate --dataset D12_10 --train 2400 --validation 240 --test 480
```

D12_07: compact/multiple/dipping/salt, h=0; D12_08: add broad intrusion, h=0; D12_09: four geometries, h=0/50/100/150/200; D12_10: five geometries, variable h. All train with signed density and 0/5/10/15% Gaussian noise (sigma as percentage of maximum absolute clean gravity). Five-category counts must be divisible by five; four-category counts by four. Generated sample NPZ contains the true density, clean/noisy Gzz, migration, normalized input, observation height and physical metadata.

**Compute note:** FFT on the independent two-grid geometry is more expensive than E11's aligned-grid FFT. Generate one dataset at a time and monitor resource use.

## 5. Train E12BASE through E12F

```bash
python -m src.E12.train --experiment E12BASE --epochs 100 --batch-size 2
python -m src.E12.train --experiment E12A --epochs 100 --batch-size 2
python -m src.E12.train --experiment E12B --epochs 100 --batch-size 2
python -m src.E12.train --experiment E12C --epochs 100 --batch-size 2
python -m src.E12.train --experiment E12D --epochs 100 --batch-size 2
python -m src.E12.train --experiment E12E --epochs 100 --batch-size 2
python -m src.E12.train --experiment E12F --epochs 100 --batch-size 2
```

Training checkpoints: `outputs/ml/training/<EXPERIMENT>/checkpoints/best_unet3d.keras`. History: `logs/training_history.csv`. Do not run E12C/E/E12F before TF tests. The E12F loss uses balanced Huber 1, Tversky 0.02, vertical 0.005, physics 0.01; other physics models use 1/0.1/0.02/0.01. Adam 1e-3, early stopping 15, reduce LR 5, seed 42, 100 epochs, batch size 2.

## 6. Predict and evaluate synthetic test set

```bash
python -m src.E12.evaluate --experiment E12D --split test
python -m src.E12.evaluate --experiment E12E --split test
python -m src.E12.evaluate --experiment E12F --split test
```

Outputs `outputs/ml/predict/E12*/test_metrics.csv` and `test_summary.json`, containing IoU/Dice, signed overlap, sign, density MAE/RMSE, Gzz RMSE/correlation and geometry breakdown. This E12 evaluator is a new implementation and should not be assumed numerically identical to E11's evaluation definitions without validating metric conventions.

## 7. Optional field inference (requires full observation grid coverage)

```bash
python -m src.E12.field --experiment E12F --height 100
```

If coverage is incomplete, the script *raises an error rather than manufacturing missing field observations*. When complete, it produces NPZ arrays, observed/predicted/residual Gzz figures, migration and density cross sections, and interactive 3-D Plotly isosurfaces.

## Experiment matrix

| Experiment | Dataset | Geometries | Heights (m) | Physics | Tversky | Vertical |
|---|---|---:|---|---:|---:|---:|
| E12BASE | D12_07 | 4 | 0 | 0 | 0.1 | 0.02 |
| E12A | D12_08 | 5 | 0 | 0 | 0.1 | 0.02 |
| E12B | D12_09 | 4 | 0/50/100/150/200 | 0 | 0.1 | 0.02 |
| E12C | D12_07 | 4 | 0 | 0.01 | 0.1 | 0.02 |
| E12D | D12_10 | 5 | 0/50/100/150/200 | 0 | 0.1 | 0.02 |
| E12E | D12_10 | 5 | 0/50/100/150/200 | 0.01 | 0.1 | 0.02 |
| E12F | D12_10 | 5 | 0/50/100/150/200 | 0.01 | 0.02 | 0.005 |

## Interpretation

This is a controlled E11→E12 **physical geometry** experiment, retaining its synthetic geometry rules and loss variants while removing the outer receiver margin and using different receiver/model spacings. The synthetic-to-field geological domain gap is not solved by correct coordinate scaling. Do not claim recovered E12 density is ground-truth Thunderbird geology.

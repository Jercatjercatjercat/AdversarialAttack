# Checked example runs

The project was checked end to end on CPU, with seed 42 and the default 15 epochs.
Saved models, curves, input gradients, forecasts and attacks are in `outputs/`.

| Dataset | Test RMSE | Predict-no-change RMSE | FGSM RMSE (0.5 K limit) | PGD RMSE (0.5 K limit) |
|---|---:|---:|---:|---:|
| Synthetic | 0.113974 K | 1.605180 K | 0.4627 K | 0.4788 K |
| NCEP tutorial air temperature | 4.116012 K | 4.257857 K | 4.5045 K | 4.5059 K |

These attacks held model weights and future targets fixed. Maximum input changes
were checked against their limits. Both input-gradient scripts produced finite,
nonzero gradients with shape `[1,1,16,16]`.

Original pipeline suite: **8 passed**. The expanded suite now has **20 passing tests**. Checks cover exact six-hour pairing, disjoint
split timestamps, train-only normalization, NetCDF/Zarr agreement, nonzero CNN input
gradients, synthetic reproducibility, area weights and bounded attacks. The local
NetCDF/Zarr fixtures are fabricated, not actual ERA5 data. One third-party NetCDF
import warning about NumPy binary sizes was emitted; both file-format checks passed.
`python -m pip check` reported no broken requirements.

The tutorial is real reanalysis data, but **not ERA5 T2m**. Real ERA5 runs are now
available as described below; original tutorial results above remain preserved.

The isolated `.venv` was created because the pre-existing global Python environment
had conflicting OpenMP libraries. NumPy and xarray version bounds in requirements
keep the tested Zarr 2 setup compatible.


## Larger real ERA5 comparison

Completed three model seeds (42, 43, 44), 15 epochs each, with the unchanged
three-layer CNN. Train: 2020 (1,463 pairs); validation: 2021 (1,459); test: 2022
(1,459). Mean/std came only from each run's training inputs. Every condition used
the same test weather and the same checkpoint within each model seed.

Results below are mean forecast RMSE across trained models. Random trials were
averaged within each model first. Each random condition used three trials.

| Condition | Linf: max cell bound 0.5 K | L2: RMS bound 0.5 K |
|---|---:|---:|
| Clean CNN | 2.494 K | 2.494 K |
| Random changes | 2.510 K | 2.509 K |
| One-step gradient attack | 2.783 K | 2.862 K |
| PGD | 2.784 K | 2.868 K |

Predicting no change gives 2.624 K on this same 2022 test set. All three CNNs
beat that baseline, though only modestly. Random changes have much less effect
than gradient-directed changes here. PGD only slightly improves on a single step;
this result does not establish that PGD is always substantially stronger.

L2 radius is 8 K for the 0.5 K RMS bound on 256 cells. PGD's largest cell change
averaged over model seeds was 4.169 K under L2, versus 0.5 K under Linf. Thus the
larger L2 error is accompanied by more concentrated input changes, not evidence
that equal maximum-cell constraints produced different errors. Squared temperature
change is not physical atmospheric energy.

Saved in `outputs/experiments/era5_2020_2022/`: `summary.csv`, `all_results.csv`,
`clean_models.csv`, configs, and model/plot folders for each seed. The table is
a descriptive comparison, not a statistical significance claim. Test weather
differs from the January-only experiment, so their absolute RMSE values should
not be used as a controlled comparison of training-data quantity.

Table values are rounded directly from the saved summary to three decimal places.
See [EXPERIMENTS.md](EXPERIMENTS.md) for worked norm-budget examples and result
interpretation, and [OUTPUTS.md](OUTPUTS.md) for a guide to the saved folders and
the reason for repeating training with three seeds.

Verification: 20 tests passed; zero-budget legacy FGSM and PGD commands reproduced
the clean synthetic RMSE exactly. Tests cover sample-wise L2 projection, full-budget
reproducible random noise, zero gradients, frozen model weights, measured physical
change sizes, and unchanged evaluation weather when the training window changes.
Comparison figures were inspected for labels, scales and legibility.

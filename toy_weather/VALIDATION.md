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

`python -m pytest -q`: **8 passed**. Checks cover exact six-hour pairing, disjoint
split timestamps, train-only normalization, NetCDF/Zarr agreement, nonzero CNN input
gradients, synthetic reproducibility, area weights and bounded attacks. The local
NetCDF/Zarr fixtures are fabricated, not actual ERA5 data. One third-party NetCDF
import warning about NumPy binary sizes was emitted; both file-format checks passed.
`python -m pip check` reported no broken requirements.

The tutorial is real reanalysis data, but **not ERA5 T2m**. Actual ERA5 forecasting
has not been measured because no local ERA5 file was supplied. Use the README's
local-file preprocessing command when that file is available.

The isolated `.venv` was created because the pre-existing global Python environment
had conflicting OpenMP libraries. NumPy and xarray version bounds in requirements
keep the tested Zarr 2 setup compatible.

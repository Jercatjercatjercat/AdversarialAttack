# Toy weather: second iteration

This iteration forecasts five weather maps six hours into the future with one
small CNN. It builds on the completed temperature-only experiment in
[archive/first_iteration/](archive/first_iteration/), whose code and results remain
preserved. The purpose is to improve the clean forecasting baseline and keep the
input-gradient/attack experiment understandable. More layers alone do not guarantee
better forecasts; evaluation compares every output with predicting no change.

## The weather represented by each channel

| Short name | Variable | Stored physical units |
|---|---|---|
| `t2m` | Two-metre temperature | Kelvin (K) |
| `u10` | East–west wind component at ten metres | metres per second (m/s) |
| `v10` | North–south wind component at ten metres | m/s |
| `mslp` | Mean sea-level pressure | hectopascals (hPa) |
| `q850` | Specific humidity at 850 hPa: water-vapour mass per mass of moist air | grams per kilogram (g/kg) |

Positive U points east, positive V north. Specific humidity is not relative humidity
and is not a percentage. 850 hPa specifies an atmospheric pressure level; its height
varies and it is not a measurement two metres above ground.

The proposed dew-point field was checked in the public coarse ERA5 archive. Its
metadata exists, but the sampled 2015 maps contain only missing values. This run
uses populated humidity at 850 hPa instead. The preprocessing rejects missing
maps rather than silently filling them. Pressure is converted from Pa to hPa and
humidity from kg/kg to g/kg before preparation. Temperature pictures use Celsius;
errors in Kelvin and Celsius have the same numerical size.

The source is the public [WeatherBench ERA5 archive](https://weatherbench2.readthedocs.io/en/latest/data-guide.html).
It is already regridded to 1.5-degree spacing and sampled every six hours. We select
45–60°N, 10°W–10°E, then interpolate to 16×16. Interpolation changes grid size; it
does not create finer observations. This is a regional, simplified experiment,
not a complete global atmospheric state. The local NetCDF/Zarr alternative expects
`time`, `latitude`, `longitude`; the variable names in `config.py`; and `level=850`
for `specific_humidity`, with explicit units. A suitable file must cover the crop.

## What enters and leaves the CNN

Each example contains weather six hours ago, weather now, and the true weather
six hours ahead. The model receives the two input states only. The true future is
used to calculate the loss, not as an input.

```text
Channel 0..4:  t2m, u10, v10, mslp, q850 at t−6h
Channel 5..9:  t2m, u10, v10, mslp, q850 at t
Channel 10..13: sine/cosine of UTC time of day and time of year

Input:                     [B,14,16,16]
Conv 3×3 + ReLU:            [B,64,16,16]
Three Conv 3×3 + ReLU:      [B,64,16,16]
Add earlier feature maps:  [B,64,16,16]
Final Conv 3×3:             [B, 5,16,16] predicted changes
Add current weather:       [B, 5,16,16] predicted future
```

B is the number of examples processed together. Internal channels are learned
features; they are not 64 weather variables. Each 3×3 convolution looks at nearby
cells. One-cell padding repeats crop-edge values and preserves spatial size. ReLU
sets negative internal features to zero, introducing nonlinear processing.

A skip connection adds information from an earlier stage. The internal skip helps
features pass through the network. The final skip adds the current weather to the
predicted correction. The output layer starts with zero weights, so the initial
forecast is exactly “predict no change.” Training then learns corrections. A
correction can be negative; the final layer has no ReLU. Five convolution layers
and 64 features give about 122,000 parameters, still small enough to inspect locally.
The model does not use dropout and produces one deterministic forecast.

Clock features use known calendar time, not future measurements. Two input states
provide recent trends; they are not an autoregressive rollout. Every forecast here
is a single six-hour step.

## Dates, scaling and fair evaluation

The default preparation uses 2015–2020 for training, 2021 for validation and 2022
for testing. Raw states are split first. Complete (t−6h,t,t+6h) groups are built
within each split, requiring exact six-hour intervals. No state crosses a split.

Each variable gets its own mean and population standard deviation, calculated
from the unique TRAINING INPUT states only. Repeated overlapping histories do not
count the same state twice. Those statistics are reused for both input times, future
targets, validation and testing:

```text
normalized variable = (physical variable − its training mean) / its training std
physical variable   = normalized variable × its training std + its training mean
```

Training minimizes the mean of the five normalized MSEs (mean squared errors).
This gives equal coefficients to variables after scaling; it is a modelling choice,
not proof that all physical errors have equal importance. Validation uses that
same combined loss to choose a checkpoint. A combined optimum need not be the
best checkpoint for every variable individually. We report physical MSE/RMSE
separately for each output and its own no-change baseline. RMSE is the square root
of MSE and has the variable's original units. Scores pool squared errors over all
examples/cells before taking the root. These are unweighted crop scores.

2022 was already inspected in the first iteration. It is an EXPLORATORY test
comparison, not a fresh final test. The source ends on 10 January 2023, so it does
not provide a full 2023 test year. A future final study needs additional unseen
weather from a suitable source. The scripts permit different split boundaries
when preparing such a file, but the coverage must actually exist.

## Running the pipeline

From `toy_weather/`:

```bash
# This checkout currently has a local .venv alias to the archived environment.
# A new checkout can create its own environment instead:
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Download and prepare real ERA5. This needs internet but no CDS credentials.
python -m data.prepare_era5

# Or prepare an appropriate local climate file:
python -m data.prepare_era5 --input /path/to/era5.nc --output local_data/my_data.npz

python train.py --dataset era5
python evaluate.py --dataset era5
python check_input_gradient.py
```

Default prepared data is `local_data/era5_five_variables.npz`, saved once as physical
maps rather than duplicated pairs. Default results are `outputs/era5/`. Training
uses up to 40 epochs, batch size 64, Adam with initial learning rate 0.001, seed 42,
and CPU. Validation plateaus reduce the learning rate; ten consecutive epochs
without improvement stop training. The best validation epoch is saved, which need
not be the final epoch. `--device mps`/`cuda` are available when supported by PyTorch.
The checked local environment runs on CPU.

Preparation and training refuse to overwrite existing results. Repeat runs use a
new output path, for example `python train.py --output-dir outputs/repeat_43 --seed 43`.
Evaluation and gradient commands use `--checkpoint-dir` and `--data` for alternate
runs. Checkpoints record a fingerprint of the dataset so a different prepared file
cannot silently be substituted. No test score chooses the learning rate or epoch.

## Looking at saved files

`best_model.pt` contains learned weights, model settings, normalization and dataset
identity. `normalization.json` records the per-variable scaling, `data_metadata.json`
records dates/source/channels, and `training_config.json` records training choices.
`history.json`, `training_summary.json` and `training_curves.png` show learning and
checkpoint selection. `test_metrics.json` reports separate physical-unit scores.
`test_example_*.png` shows current weather, true future, predicted future and error
for each variable. Example dates are spaced across the test period rather than
selected for low error.

`input_gradient.json` and `input_gradient.png` record loss sensitivity to normalized
input weather. The script freezes weights, enables gradients on x, computes the
forecast loss and calls backward. Gradient values are not temperatures or applied
input changes. To convert a weather-channel derivative to a physical-input
derivative, divide it by that variable's training standard deviation.

## Controlled changes and attacks

One model design supports simpler settings so changes can be introduced separately.
For example, this temperature-only comparison has the original three-convolution,
32-feature architecture with direct future prediction:

```bash
python train.py --variables t2m --history 1 --no-time-features \
  --direct-output --width 32 --depth 3 --output-dir outputs/t2m_reference
```

`--history 1`, `--no-time-features`, `--direct-output`, `--depth` and `--width` expose
the proposed changes explicitly. Matching data/dates and changing one setting at
a time is needed to attribute an improvement to a specific action. Comparing the
final model with the archived run alone changes several factors simultaneously.

The input-attack bridge remains available:

```bash
python compare_attacks.py --variable t2m --budget 0.1 0.5
```

This compares random noise, a one-step gradient attack and ten-step PGD under Linf
and L2 rules. It changes only the named variable's CURRENT map. Earlier weather,
other variables, clock features, targets and weights stay fixed. The objective is
to increase the mean normalized loss over all five outputs; each output's physical
RMSE is reported. Cross-variable effects can therefore be observed.

`--budget` has the named variable's units: Linf caps every cell by that value; L2
caps input RMS by that value. On 16×16 maps L2 radius = 16 × RMS bound. These do not
have equal largest-cell allowances. Budgets do not establish physical plausibility.
A wind budget uses m/s, a pressure budget hPa, and a humidity budget g/kg. There is
no unexplained norm combining different physical units. Random conditions have
three trials; rows are saved individually, not selected for greatest damage.

PGD keeps the highest-loss candidate per example, includes the clean and one-step
candidates, and has no random restarts. Attack results are separate from clean
forecast results. The model stays fixed during all changes. The second iteration
has not implemented physically coupled changes, extreme-event selection, training
data poisoning or multiple future forecast steps.

## Files and checks

`data/prepare_era5.py` reads/crops physical maps; `data/era5.py` pairs and scales them.
`models/residual_cnn.py` contains the one configurable forecasting network.
`train.py`, `evaluate.py` and `check_input_gradient.py` are the learning, testing and
sensitivity entry points. `metrics.py`, `plotting.py` and `runtime.py` provide shared
scoring, pictures and checkpoint loading. `perturbations.py` contains input-changing
functions; `compare_attacks.py` runs comparisons without training another model.

```bash
python -m pytest -q
```

The tests use clearly labelled fabricated fields. They check exact time pairing,
split isolation, training-only normalization, units, residual initialization,
input gradients, pooled scores and bounded attacks. Scientific ERA5 findings live
in [EXPERIMENTS.md](EXPERIMENTS.md), with published-paper context in
[the reading collection](../PaperReadings/README.md).

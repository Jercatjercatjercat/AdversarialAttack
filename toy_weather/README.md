# Toy weather forecasting and input attacks

This project teaches one idea: give a small neural network a temperature map and
ask it to predict the next map. Then slightly change the input to see whether its
forecast becomes worse. The aim is understanding, not competitive forecasts.

Explanations use everyday language. Comments describe tensor shapes, each operation,
and whether it belongs to weather processing or general machine learning.

## Setup

Run these commands from `toy_weather/`. Use a separate environment so this project
has its own Python libraries (Python 3.10 or newer).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

CPU is the default. Each main script also accepts `--device mps` or `--device cuda`
if your PyTorch installation supports that device. No ERA5 credentials are needed
for the synthetic experiments.

## Experiment 1: synthetic temperature forecasting

Goal: understand maps, tensors, training and spatial prediction.

```bash
python train.py --dataset synthetic
python evaluate.py --dataset synthetic
```

Defaults: 4,000 training pairs, 500 validation pairs, 500 test pairs, 16×16 maps,
15 epochs, batch size 64, learning rate 0.001. Seeds make the examples reproducible.
Use `--epochs 3` for a shorter first run; use `--size 32` to try a larger synthetic grid.

Every starting map is made independently. We smooth random noise with a Gaussian
filter, add two warm/cold blobs, then give the map a spatial standard deviation of
4 K and a mean near 285 K. This is weather-like texture, not realistic weather physics.

The future depends on its own starting map:

```text
future = 0.85 × shifted input + 0.15 × blurred input + small noise
```

The shift moves the map one cell east. Blur softens temperature differences. Noise
has standard deviation 0.1 K. The edges join, like a repeating tiled map. The
synthetic time step is artificial: it has no physically calibrated duration.
The synthetic CNN uses circular padding to match these joined edges.

Training learns from the training pairs. Validation uses different pairs to choose
which epoch's weights to save. Test measures that selected model on unseen pairs.
All splits use the same generation rule but independent examples. The future map
within a pair is NOT independent of the starting map.

`outputs/synthetic/example_pair.png` shows the starting map, future map and their
difference. A positive difference means that grid cell warmed during the toy step.
Evaluation plots the input, true future, predicted future, and prediction minus
truth. All temperature panels use Celsius; differences use K (1 K difference is
also 1°C difference). Grid rows/columns are not a geographic map projection.

## The model, in pictures

```text
Input temperature          [B,  1, H, W]
       ↓ Conv 3×3, padding 1
       ↓ ReLU              [B, 32, H, W]
       ↓ Conv 3×3, padding 1
       ↓ ReLU              [B, 32, H, W]
       ↓ Conv 3×3, padding 1
Predicted temperature      [B,  1, H, W]
```

B means number of examples in a batch. H and W are map height and width. The 32
internal channels are learned feature maps, not 32 physical weather variables.
A filter looks at a 3×3 neighbourhood and combines its values with learned weights.
It can learn smoothing, edges, or other patterns. Padding supplies a one-cell border
so the map stays the same size: `16 + 2×1 − 3 + 1 = 16` at stride 1.
ReLU replaces negative feature values with zero. Repeating convolution and ReLU is
a traditional CNN design. Three convolution layers see a 7×7 input neighbourhood.
There is no final ReLU because normalized temperature may be negative.
For a local ERA5 crop, replicated edge values provide the border; this is still a
simplification, since weather outside the crop is unknown to the model.

We normalize temperatures using one scalar mean and standard deviation from
**training inputs only**, then apply those same numbers to every input and target:

```text
normalized = (temperature_K − training_mean_K) / training_std_K
Kelvin     = normalized × training_std_K + training_mean_K
Celsius    = Kelvin − 273.15
```

`data/era5.py` provides `normalize` and `denormalize` for both tensors and arrays.
Using one scalar keeps normalization and physical attack limits easy to follow.

Training uses MSE: average squared error. It penalizes large mistakes strongly.
RMSE is the square root of MSE. Unlike MSE (K² in physical units), RMSE is in K, so
it is easier to interpret. All test pixels and examples contribute to one global
MSE before taking its square root. We do not average batch RMSE values.
Evaluation also reports **persistence**: predict the future equals the input. A
useful forecaster should improve on that simple baseline.

Saved files include `best_model.pt`, `normalization.json`, `history.json`,
`training_curves.png`, `test_metrics.json` and example plots. Checkpoint selection
uses validation RMSE only; test data never selects an epoch.

## Experiment 2: input gradients on synthetic weather

Goal: verify that changing the input can affect forecast loss.

```bash
python check_input_gradient.py
```

The script loads the saved model, freezes its weights, and runs one test example:

```python
x = x.clone().detach().requires_grad_(True)
prediction = model(x)
loss = criterion(prediction, y)
loss.backward()
gradient = x.grad
```

`x.grad` has shape `[1, 1, 16, 16]`. It tells us how sensitive the loss is to each
input cell. Positive means a small increase locally raises loss; negative means a
small decrease locally raises loss. Large magnitude means greater sensitivity.
This script changes nothing: it only prints and plots the gradient. The default
loss is normalized MSE. Its gradient with respect to Kelvin input is the normalized
input gradient divided by training standard deviation. Both are plotted.

## Experiment 3: ERA5 T2m forecasting

Goal: use the same pipeline on real atmospheric data.

### Existing local ERA5 subset

Real ERA5 temperature maps for January 2020 over a small European region are
already saved in `local_data/era5_t2m_2020_01.nc`. The one-off download script has
been removed. These are six-hourly, regridded ERA5 maps from the public WeatherBench
archive used by Otter, not original full-resolution data or NCEP tutorial data.

Prepared 16×16 pairs are already available in `outputs/era5/prepared.npz`.
Run from `toy_weather/` with the environment activated:

```bash
python train.py --dataset era5
python evaluate.py --dataset era5 --latitude-weighted
python check_input_gradient.py --dataset era5
python attack.py --dataset era5 --method pgd --epsilon-k 0.5
```

To recreate the prepared pairs from the saved weather maps:

```bash
python -m data.preprocessing --input local_data/era5_t2m_2020_01.nc \
  --bounds 45 60 -10 10 --output outputs/era5/prepared.npz
```

One month is a small pipeline demonstration; use longer periods for meaningful
weather experiments. The saved file and training work offline. See the
[WeatherBench data guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html)
for the source description. The manual local-file route below works for new data.

ERA5 is a historical weather reconstruction combining observations with a weather
model. A downloaded subset might look like:

```text
Coordinates: time[T], latitude[H], longitude[W]
Variable:    t2m[T, H, W] in Kelvin
```

Each timestamp holds a map. This project takes only 2-metre temperature and predicts
it six hours later. It does not download full ERA5 or manage credentials.
Obtain an hourly T2m subset as NetCDF/Zarr yourself, then:

```bash
python -m data.preprocessing --input /path/to/era5.nc \
  --bounds 45 60 -10 10 --output outputs/era5/prepared.npz
python train.py --dataset era5
python evaluate.py --dataset era5 --latitude-weighted
```

Bounds are south, north, west, east in degrees. Use a region actually covered by
your file. NetCDF `.nc` and Zarr `.zarr` paths are supported. Names can be overridden:

```bash
python -m data.preprocessing --input /path/to/era5.nc \
  --variable t2m --time valid_time --latitude latitude --longitude longitude \
  --bounds 45 60 -10 10
```

Requirements and processing:

1. One-dimensional time, latitude and longitude coordinates on a regular grid.
   Extra singleton dimensions are removed. Select a single level/member yourself
   if an extra dimension has several values. Unrecognized units, duplicates and
   missing temperatures are rejected instead of silently repaired.
2. Convert explicitly labelled Celsius data to Kelvin when necessary. Sort latitude
   and time; convert longitudes to −180..180. Date-line-crossing crops are excluded.
3. Crop with neighbouring source cells retained at the boundary, then linearly
   interpolate onto a regular 16×16 grid. This is a simple resize, not a sophisticated
   conservative regrid. No values outside the source area are extrapolated.
4. Keep exact UTC 00/06/12/18 timestamps. Split the timeline into 70% training,
   15% validation and 15% test BEFORE making pairs.
5. Within each split, make only exact `t → t+6h` pairs. Missing timestamps cause
   affected pairs to be skipped. Pairs never cross a split boundary, so no raw state
   appears in more than one split.
6. Compute mean/std from training inputs only. Apply them unchanged to all pairs.
   Save coordinates, source label and input/target timestamps with the arrays.

Neighbouring pairs may share states within a split. This is fine; splitting pairs
randomly across train/test would leak those states and is not done here. Interpolation
uses only each map's spatial neighbours; it does not fit to future statistics.

Use a reasonably long period; a few days are a code demonstration, not a meaningful
weather study. Temperature-only forecasting omits winds, clouds, sunlight and other
causes of temperature change. Consequently six-hour forecasts may not beat persistence.

Optional latitude-weighted RMSE uses weights proportional to `cos(latitude)`.
On a regular latitude/longitude grid, cells near the poles cover less area. These
weights stop those cells counting as much as larger cells near the equator. The
weights are normalized to have mean one. This is area weighting, not an attack.

### Real-data fallback without ERA5 credentials

Xarray offers an NCEP tutorial air-temperature dataset. **It is not ERA5 T2m** and
must not be described as an ERA5 experiment. The command below downloads it on
request, labels its source, and uses its six-hour maps with the same code:

```bash
python -m data.preprocessing --tutorial --output outputs/tutorial/prepared.npz
python train.py --dataset era5 --data outputs/tutorial/prepared.npz --output-dir outputs/tutorial
python evaluate.py --dataset era5 --output-dir outputs/tutorial --latitude-weighted
```

Here `--dataset era5` chooses the shared real-file loader; terminal output and saved
metadata identify the actual tutorial source. Internet access is required only for
this optional download. The main local-file pipeline works offline.

## Experiment 4: input gradients on real weather

Goal: check that exactly the same idea works on real data.

```bash
python check_input_gradient.py --dataset era5
# For the tutorial checkpoint instead:
python check_input_gradient.py --dataset era5 --output-dir outputs/tutorial
```

The model has the same architecture, but is trained separately for each dataset.
The script reads the same normalization statistics and prepared data used in training.
Do not overwrite a prepared file with a different dataset after training a checkpoint.

## Experiment 5: actual FGSM and PGD attacks

Only start this after the preceding forecasts and gradient checks work.

```bash
python attack.py --dataset synthetic --method fgsm --epsilon-k 0.1 0.5 1.0
python attack.py --dataset synthetic --method pgd --epsilon-k 0.1 0.5 1.0 --steps 10
python attack.py --dataset era5 --method fgsm --epsilon-k 0.1 0.5 1.0
python attack.py --dataset era5 --method pgd --epsilon-k 0.1 0.5 1.0 --steps 10
```

For the tutorial, add `--output-dir outputs/tutorial` to the real-data commands.

FGSM makes one change in the gradient's uphill direction. PGD makes repeated small
changes and clips them so each pixel remains within its limit of the original input.
PGD starts from the clean input and retains the strongest candidate per sample,
including the clean input. Random restarts are left for later. An attack is an attempt
to increase loss; a single FGSM step is not guaranteed to succeed on every example.

`--epsilon-k 0.1` means every input cell may change by at most 0.1 K. It does not
mean the forecast error must increase by 0.1 K. The scripts convert this limit to
normalized units by dividing by training standard deviation. These are L-infinity
limits: they bound the largest absolute pixel change. There is no extra global
Kelvin clamp. We keep the real future target and model weights fixed.

Each run prints clean RMSE, attacked RMSE and maximum actual input change. PNGs show
the original/changed inputs, perturbation, true future and both forecasts. JSON files
record results for all test pairs. Use epsilon 0 to check that clean and attacked
results agree. We use real targets to construct these attacks, an idealized evaluation
setting; operational attackers may not know the future target. These pixel changes
need not be physically consistent weather states. This is an educational vulnerability
check, not a realistic atmospheric intervention.

## Connection to larger weather models

Toy model: `T_now → model → T_next`.
Larger model: `past weather state → model → future weather state`.
That larger state may include temperature, humidity, winds, geopotential, pressure
levels and surface variables. This toy has the same differentiable input-to-forecast
structure, but none of the scale or scientific completeness of a full model.

During training, we change model weights to reduce loss. During attacks, we freeze
those weights and calculate `∇x L`: how input changes affect loss. We then change
the input to increase loss. Our experiments are one-step forecasts. Feeding predictions
back into the model for many steps (autoregressive forecasting) is not implemented.

## Files and checks

```text
data/synthetic.py        Make smooth toy maps and their future states
data/era5.py             Normalization and prepared-file loading
data/preprocessing.py    Xarray crop, six-hour pairing and chronological split
models/simple_cnn.py     Three convolution layers
config.py                Defaults and output paths
runtime.py               Shared data/checkpoint setup
metrics.py               MSE, RMSE, persistence and area weighting
plotting.py              Fields, curves and gradients
train.py                 Learn model weights
evaluate.py              Measure untouched test performance
check_input_gradient.py  Verify input differentiation
attacks.py               FGSM and PGD functions
attack.py                Run attacks and compare forecasts
tests/test_pipeline.py   Leakage, exact leads, file formats and attack limits
```

```bash
python -m pytest -q
```

Tests use fabricated local weather files to check the machinery; those fixtures are
not evidence of real ERA5 forecasting performance. They check disjoint split states,
exact six-hour leads despite missing times, train-only normalization, NetCDF/Zarr
agreement, gradients, and attack limits.

References: [ECMWF ERA5 overview](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5),
[ERA5 data documentation](https://confluence.ecmwf.int/pages/viewpage.action?pageId=185081924),
and [Xarray tutorial](https://tutorial.xarray.dev/overview/xarray-in-45-min).

See [VALIDATION.md](VALIDATION.md) for the completed example runs and their results.
`requirements-tested.txt` records the exact environment used for those runs; it can
be installed instead of `requirements.txt` when reproducing them on a compatible
Python/platform. The main requirements file permits compatible version ranges.

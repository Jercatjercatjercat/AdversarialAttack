# Second iteration: forecasting setup and evidence

The second iteration tests a five-layer CNN that predicts corrections to current
weather. It receives two recent states plus clock features and predicts temperature,
two wind components, sea-level pressure and humidity at 850 hPa six hours ahead.
The original temperature-only project is preserved in
[archive/first_iteration/EXPERIMENTS.md](archive/first_iteration/EXPERIMENTS.md).

## What changes from the first iteration

The default network has five 3×3 convolutions, 64 internal feature channels, an
internal feature skip and a current-weather skip. The final output layer is
initialized to zero, so it begins as the no-change forecast and learns corrections.
There is no dropout. Inputs are [B,14,16,16]; outputs [B,5,16,16]. The extra inputs
include weather six hours ago and known UTC time-of-day/year features.

These changes add both information and model capacity. A gain in the final model
cannot by itself identify which change caused it. The training flags allow separate
comparisons on the same data. Increasing the number of training years also changes
the experiment; an old-versus-new score is not a controlled architecture comparison.

## Data and evaluation design

Source: WeatherBench's public 1.5-degree, six-hourly ERA5 archive. Crop: 45–60°N,
10°W–10°E, linearly interpolated to 16×16. The final variable order is t2m, u10,
v10, mslp, q850. Dew point was proposed, but the sampled 2015 coarse-source maps
were missing despite the variable's metadata being present. Valid humidity at
850 hPa replaces it; the two quantities are not physically equivalent.

Dates: training 2015–2020, validation 2021, exploratory testing 2022. Each example
uses t−6h, t and t+6h entirely inside its split. Input states can overlap between
nearby examples within one split, but cannot cross into another split. Each variable's
normalization is fitted only on unique training input states, then reused for inputs,
targets and every split. Pressure is in hPa, humidity g/kg, wind m/s and temperature K.

The training objective is the average normalized MSE over variables, examples
and cells. Validation uses the same loss for checkpoint selection and learning-rate
scheduling. Physical-unit RMSEs are reported per variable; their units are not
mixed. Predicting no change is scored on the same examples and cells. These scores
are unweighted; they do not use latitude weighting or extreme-event selection.

2022 was already inspected in the first iteration, so these are exploratory results.
A genuinely unseen final period remains necessary for stronger conclusions. The
source stops on 10 January 2023 and cannot supply a full 2023 test year.

## Verification record

The second-iteration correctness suite checks exact pairing, disjoint split states,
training-input-only statistics, reversible scaling, channel ordering, unit conversion,
clock features, exact initial persistence, frozen-weight input gradients, global
RMSE aggregation and sample-wise attack bounds. Its maps are fabricated and are
not scientific ERA5 evidence. A temporary two-epoch fixture run also exercised
training, evaluation, gradient plotting and random/one-step/PGD command-line paths.
No fabricated fixture scores are included in the weather results.

## Attack interpretation

The comparison script freezes the model and changes one named variable's CURRENT
map only. The history, other variables, clocks and true future remain unchanged.
Its objective is increased average normalized forecast loss across all five outputs.
The random baseline uses no gradient; one-step and PGD attacks do. Every output's
physical RMSE is recorded, including effects on variables whose inputs were unchanged.

Linf caps each changed cell. L2 caps the combined length of that change map, with
radius 16 times the requested RMS bound on this grid. Equal RMS bounds do not give
equal maximum-cell allowances. The budget has the chosen variable's units and is
not physical atmospheric energy. An increased combined loss need not increase
every output's error individually. No physical coupling constraints or random PGD
restarts are implemented.

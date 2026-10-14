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

Preparation excludes whole states with missing crop values, with counts recorded
per month; it never fills pixels or pairs across an irregular time gap.

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
No fabricated fixture scores are included in the weather results. All 15 checks
passed. The local NetCDF import emitted the existing third-party NumPy binary-size
warning, but NetCDF and Zarr returned identical prepared values. `pip check` found
no broken dependencies. The new 2020–2022 T2m maps were compared with the archived
raw ERA5 subset after the same interpolation: maximum difference was exactly 0 K.

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

## Prepared data record

The completed dataset contains 11,688 physical states, shaped [11688,5,16,16].
All 96 month blocks passed the finite-value checks: no states were excluded.
There are 8,766 training examples, 1,458 validation examples and 1,458 test examples.
Default two-state normalization uses 8,767 unique training input states. The
prepared file is about 51.7 MiB and supports offline training. Download caches
are ignored; the prepared weather, weights and results are tracked.

## Reference model

`outputs/t2m_reference/` is the original three-convolution, 32-feature structure,
trained here on the same calendar splits with only current temperature. It has
9,857 parameters and no clock/history/skip inputs. It uses the same eligible
forecast cases as the main model, including the triple-validity rule, even though
it receives only the current state. Its scaling uses the current training states
that are actually inputs. This is a simpler reference, not a second improved model.

Both runs use seed 42, batch size 64, maximum 40 epochs, Adam at initial lr 0.001,
validation-selected checkpoints and the same plateau schedule. Changing information,
capacity and output representation together is a whole-system comparison; it does
not isolate the benefit of each design choice. This iteration has one seed per
condition, so no claim about variability across training seeds is made.

## Completed clean results, 14 October 2026

Both runs completed 40 epochs and selected epoch 40 using validation. The main
model has 121,797 parameters. Its CPU training took about 403 seconds, with the
reference run partly concurrent. The test set contains 1,458 six-hour forecasts.
Lower RMSE is better; temperature differences of 1 K and 1 °C have equal size.

| Variable | CNN test RMSE | No-change test RMSE | Reduction in RMSE |
|---|---:|---:|---:|
| t2m (K) | 0.788 | 2.625 | 70.0% |
| u10 (m/s) | 1.063 | 2.118 | 49.8% |
| v10 (m/s) | 1.138 | 2.356 | 51.7% |
| mslp (hPa) | 1.021 | 2.555 | 60.0% |
| q850 (g/kg) | 0.540 | 0.950 | 43.1% |

The temperature-only reference has test RMSE **2.377 K** on
the same cases. The main model's **0.788 K** is about
**66.8% lower**. All five outputs beat their no-change
baselines in this run. This establishes a stronger clean forecasting baseline
than the first iteration; it does not show which added information or architectural
change caused the gain, or how results vary across new seeds/years.

The archived first-iteration CNN mean was 2.494 K versus persistence 2.624 K,
with three seeds, only 2020 training, 15 epochs and 1,459 test pairs. Those are
contextual scores, not the matched reference above. One test pair is omitted here
because a complete earlier input state is required at the beginning of the split.

Scores are in `outputs/era5/test_metrics.json`, `forecast_comparison.csv`,
`temperature_comparison.json` and `outputs/t2m_reference/test_metrics.json`.

## Completed gradient and attack checks

The final model produced a finite, nonzero input gradient shaped [1,14,16,16].
The test-sample gradient minimum was −0.0107381, maximum 0.0156226 and mean
−5.3813e−7, all derivatives of normalized MSE with respect to normalized inputs.
The gradient is a sensitivity field, not an applied weather change.

The default full-test comparison changed only current T2m, with three random
trials, one-step gradient ascent, and ten-step PGD. It used two budgets and both
Linf/L2 rules. All recorded input-change sizes met their constraints within
float32 tolerance. The loss target is the five-output normalized MSE, not
temperature error alone.

At budget 0.5 K, the following temperature RMSEs illustrate the effect. Random
scores average the three trials; there is only one trained main model.

| Method | Linf: every cell ≤0.5 K | L2: input RMS ≤0.5 K |
|---|---:|---:|
| random | 0.896 K | 0.896 K |
| one_step | 1.074 K | 1.115 K |
| pgd | 1.078 K | 1.125 K |

Clean temperature RMSE is 0.788 K. Deliberately chosen input changes cause more
forecast error than random changes in these conditions. Changing temperature
also affects the predicted winds, pressure and humidity; those separate scores
are recorded in `attack_comparison/results.csv` and `summary.csv`. Because PGD
maximises combined loss, it need not increase each individual variable's error
more than a one-step attack. L2 permits larger individual changes than Linf at
the matched RMS setting, so the two columns do not have equal maximum-cell limits.

This is an attack demonstration on the new model, not a completed study of
physical plausibility, extreme events, multi-step error growth or Otter robustness.
A claim that the new model is more robust than the old model would also require
matched attack objectives and access to the same input times/variables. Better
clean accuracy alone does not establish adversarial robustness.

## Attack settings extension

The subsequent [step-count/norm experiment](ATTACK_EXPERIMENTS.md) adds MI-FGSM,
L0 top-k and L1 projection, with validation-selected steps/sizes and restart counts.
It targets temperature forecast error rather than the combined output loss above.
The new results live in `attack_tuning_steps/` and `tuned_attacks_steps/`; they do
not replace the older tables. Their interpretation includes the different norm
budgets, the separate amplitude cap, validation subset and selection rule.

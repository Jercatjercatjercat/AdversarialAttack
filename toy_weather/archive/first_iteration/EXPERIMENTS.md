# Toy-weather experiments: setup, findings and limits

This project asks two questions: can a small neural network predict a future
temperature map, and can deliberately chosen changes to its input make that
prediction worse? We keep the model simple so the effect of the input changes is
easy to understand. The saved results below are completed runs, not proposed work.

## 1. What goes into the model?

A map is a table of temperatures. Each example contains two maps:

- **Input x:** temperature now.
- **Target y:** the true temperature at the next time.
- **Prediction:** the future map produced by the neural network.
- **Changed input:** the original input plus changes chosen randomly or by an attack.
  It is an alternative input to the model, not a new observation of the true weather.

For ERA5, the next time is six hours later. ERA5 is a historical reconstruction of
weather that combines measurements with a physical weather model. Our subset comes
from the public [WeatherBench archive](https://weatherbench2.readthedocs.io/en/latest/data-guide.html).
We select a European region, 45–60°N and 10°W–10°E, and resize each map to 16×16.
We use only **T2m**, meaning air temperature two metres above the ground.

The **CNN**, or convolutional neural network, applies learned small filters to
nearby cells. It has three convolution layers: one temperature map becomes 32
learned feature maps, then 32 processed feature maps, then one future map. These
32 maps are different learned views of temperature, not different weather variables.
Between the first two convolutions, **ReLU** replaces negative feature values with zero.
The output has the same 16×16 size as the input.

A batch has shape `[B, 1, 16, 16]`: B example pairs, one temperature variable,
16 rows and 16 columns. A **batch** is the group processed for one weight update.
An **epoch** is one pass through all training pairs. These runs use batch size 64,
15 epochs and learning rate 0.001; the learning rate controls the size of weight updates.

## 2. Training and evaluation use different dates

| Purpose | Dates | Six-hour input/target pairs | What we use it for |
|---|---|---:|---|
| Training | 2020 | 1,463 | Learn weights |
| Validation | 2021 | 1,459 | Choose the saved model |
| Testing | 2022 | 1,459 | Measure the selected model on unseen weather |

We divide the timeline before creating pairs. A pair never crosses a division
boundary, and no weather state appears in two divisions. **Information leakage**
means allowing evaluation data to influence learning or model selection; we avoid
it when preparing data and choosing a training epoch.

**Normalization** puts temperatures on a manageable numerical scale:
`normalized = (temperature_K - training_mean_K) / training_std_K`.
The mean and standard deviation describe the centre and spread of TRAINING INPUTS
only. The same two numbers are used for inputs and targets in every division.
For this prepared dataset, they are about 284.144 K and 5.222 K. To convert back,
use `temperature_K = normalized * training_std_K + training_mean_K`.

Training reduces **MSE**, mean squared error: average the squared differences
between predictions and targets. We report **RMSE**, its square root, in Kelvin
(K). Lower is better. A difference of 1 K is the same size as a difference of 1°C;
absolute Kelvin and Celsius temperatures have different starting points.
We combine squared errors across all test cells and samples before taking the
square root, rather than averaging separate batch RMSE values.

A **checkpoint** is a saved set of learned weights with the information needed to
load it. We save the checkpoint with the lowest validation RMSE, which need not be
the final epoch. Attacks use that saved checkpoint with its weights **frozen**:
the weights stay fixed while we change inputs.

## 3. Why three CNNs?

A **seed** is a number controlling a reproducible sequence of random choices.
We train three copies of the SAME three-layer CNN, with seeds 42, 43 and 44.
The starting weights and shuffled batch orders differ; the weather, normalization,
architecture and training settings do not. We do not combine their predictions.

| Training seed | Clean CNN test RMSE | Predict-no-change test RMSE |
|---|---:|---:|
| 42 | 2.487 K | 2.624 K |
| 43 | 2.493 K | 2.624 K |
| 44 | 2.502 K | 2.624 K |
| Mean across CNNs | 2.494 K | 2.624 K |

Repeating training checks whether a finding depends on one particular starting
point. These are three trained models, not three independent test datasets:
all use the same 2022 weather. The no-change score is identical because that
method has no learned weights.

## 4. Two different comparison baselines

A **baseline** is a simple reference method against which we compare results.

- **Forecast baseline — persistence:** predict that the future temperature equals
  the original current temperature. We also call this “predict no change.”
- **Input-change baseline — random noise:** change the CNN's input without using
  the model, gradient or future target to choose the direction. This checks whether
  deliberate choices cause more damage than random changes of comparable size.

In the current experiment, the clean CNN beats persistence by about 5% in RMSE.
That is a modest advantage, not strong forecasting performance.

The older January-only ERA5 run, in `outputs/era5/`, has CNN RMSE 1.434 K versus
persistence 1.311 K: that CNN lost to the simple baseline. The larger experiment
must be judged against its OWN persistence score. January and whole-year results
use different test weather, so comparing 1.434 K directly with 2.494 K does not
isolate the effect of getting more training data.

## 5. What an attack does

An input **gradient** tells us how forecast loss responds to small changes in each
input cell. Computing it through the model is called **backpropagation**.
During training, gradients guide weight changes that reduce error. During our
attacks, input gradients guide temperature changes that increase error.

Our attacks can inspect the trained model and the true future during evaluation.
The true future stays fixed. We try to increase error across the whole predicted
map; we do not require a particular wrong forecast, such as hiding a heatwave.

| Method | How it changes the input |
|---|---|
| Random, Linf | Independently choose +r or -r at every cell |
| Random, L2 | Draw random values, then rescale each map to the full shared budget |
| FGSM, Linf | One gradient step: increase or decrease each cell according to its gradient sign |
| One-step L2 | One gradient step: divide the gradient by its L2 length, then scale to the budget |
| PGD | Ten gradient steps, bringing changes back inside the allowed budget after each step |

**FGSM** stands for Fast Gradient Sign Method. **PGD** stands for Projected Gradient
Descent; here we move uphill to INCREASE forecast error. “Projected” means bringing
an over-budget change back within the limit. The Linf version clips individual
changes; the L2 version reduces the length of the change map.

PGD starts from the original input. It retains the highest-loss candidate separately
for each example, including the clean and one-step candidates. There are no **random
restarts**, meaning we do not retry PGD from several random initial changes.
The L2 single-step method is labelled `one_step_l2` in result tables, because it
uses gradient magnitude as well as sign.

Each random condition has three noise trials. They use the full budget, and we
average their scores rather than choose the most damaging trial. We reuse the
same noise draws across model comparisons. These trials do not train extra CNNs.

## 6. How much input change is allowed?

Let **delta** be changed input minus original input, expressed in Kelvin.
A **norm** measures the size of a collection of numbers. We use two different
norms to limit the 256 changes in each 16×16 map:

- **Linf**, also written L∞ and pronounced “L infinity,” is `max(abs(delta))`.
  It limits the largest individual cell change.
- **L2** is `sqrt(sum(delta**2))`. All cells share a budget, with large individual
  changes using it up faster because they are squared.
- **Input RMS**, root mean square input change, is `sqrt(mean(delta**2))`.
  It describes the overall size of the input changes; it is not their simple
  average absolute value.

The **CLI**, command-line interface, is the commands typed in your terminal.
The setting `--budget-k r` means a Linf limit of r K or an input RMS limit of r K
for L2. “Radius” means allowed change size, not geographical distance.

For 256 cells:

```text
input RMS = sqrt(sum(delta**2) / 256)
          = L2 / sqrt(256)
          = L2 / 16

Therefore an input RMS bound of r needs an L2 radius of 16r.
For r = 0.5 K, the L2 radius is 8 K.
For a different grid, replace 16 with sqrt(height * width).
```

| Example input changes | L2 size | Input RMS | Linf limit 0.5 K allows it? | L2 radius 8 K allows it? |
|---|---:|---:|---|---|
| All 256 cells change by 0.5 K | 8 K | 0.5 K | Yes | Yes |
| One cell changes by 8 K; the other 255 do not change | 8 K | 0.5 K | No | Yes |

The second row illustrates an allowed possibility, not a typical attack.
Every change allowed by this Linf rule fits within the corresponding L2 rule,
but L2 also permits concentrated larger changes. Matching input RMS limits does
NOT match maximum-cell limits. Neither rule guarantees physically consistent weather.
Squared temperature change is a size measure, not physical atmospheric energy.

**Input RMS measures the disturbance. Forecast RMSE measures the prediction error.**
They answer different questions even though both are expressed in Kelvin here.
Limits are enforced separately for each example, not shared across a batch.
We report the largest cell change and RMS input change to make this distinction visible.

The attack code works in normalized units. Dividing a physical norm radius by the
training standard deviation converts it to the normalized radius. For several
weather variables, each variable would need its own physical units and limits.

## 7. Current attack findings

The following are mean forecast RMSE values across the three CNNs. Random trials
are averaged within each CNN first. Scores come from `summary.csv`, rounded
directly to three decimals. The clean score is 2.494 K throughout.

| Method | Linf, r=0.1 K | L2, r=0.1 K | Linf, r=0.5 K | L2, r=0.5 K |
|---|---:|---:|---:|---:|
| Random changes | 2.495 K | 2.495 K | 2.510 K | 2.509 K |
| One-step gradient attack | 2.549 K | 2.566 K | 2.783 K | 2.862 K |
| PGD | 2.549 K | 2.566 K | 2.784 K | 2.868 K |

At r=0.5 K:

1. **The direction matters.** Random changes increase forecast RMSE by about 0.6%.
   PGD increases it by about 12% under Linf and 15% under L2.
2. **PGD adds little over one step here.** The extra steps give only a small increase
   in error in these runs. This does not establish that PGD is unnecessary for other
   models, budgets or starting points.
3. **L2 allows larger local changes.** For L2 PGD, the average of the three runs'
   largest cell changes over the test set is 4.17 K, while input RMS is 0.5 K.
   That maximum is not the typical cell change. Linf PGD caps every cell at 0.5 K.
   The larger L2 forecast error therefore comes with a more permissive local-change
   rule, not the same maximum-cell constraint.
4. **Attacks remove the clean advantage relative to clean persistence.** Directed
   attack scores exceed the 2.624 K no-change reference. That reference uses ORIGINAL
   inputs; we have not evaluated persistence on the changed inputs in this table.
5. **Larger budgets cause more damage in these runs.** Compare r=0.1 with r=0.5
   for the same method and norm. This observation alone does not determine which
   budgets are plausible real measurement errors.

The supported finding is that these toy CNNs are more sensitive to deliberately
chosen changes than to random changes of comparable overall size. The clean
forecasting advantage remains modest. These scores do not prove vulnerability of
Otter or another operational model.

All reported attack scores use ordinary, unweighted full-map error. Optional
**latitude weighting** gives cells weights proportional to the cosine of latitude,
because regular latitude/longitude cells cover less area near the poles. We have
not used that option in this three-model table.

All forecasts here are one step. **Autoregressive forecasting** means feeding a
prediction back into the model to predict the following step; that is not implemented
in this toy pipeline.

### Earlier demonstrations, kept separately

The original runs used one trained model each. Their attack scores below use a
Linf limit of 0.5 K per cell. They did not include the newer random-noise/L2
comparison, so they are not part of the three-model average above.

| Earlier dataset | Clean CNN RMSE | No-change RMSE | FGSM RMSE | PGD RMSE |
|---|---:|---:|---:|---:|
| Synthetic, made-up temperature | 0.114 K | 1.605 K | 0.463 K | 0.479 K |
| NCEP tutorial air temperature, NOT ERA5 | 4.116 K | 4.258 K | 4.504 K | 4.506 K |

The synthetic result shows that the CNN learns our deliberately simple shift/blur
rule. Real weather has causes that rule does not represent, so success there does
not imply equally good ERA5 forecasting. The tutorial is another historical
weather reconstruction supplied through xarray's example-data collection, not the
ERA5 two-metre-temperature dataset. Its clean advantage is modest too.
Source files are the `test_metrics.json`, `fgsm_metrics.json` and
`pgd_metrics.json` files in `outputs/synthetic/` and `outputs/tutorial/`.

## 8. Where the evidence lives

The current folder is `outputs/experiments/era5_2020_2022/`:

- `clean_models.csv`: clean CNN and persistence scores for each training seed.
- `summary.csv` / `summary.json`: condition averages and standard deviations
  across trained models. Standard deviation describes their spread; it does not
  prove that small differences are statistically meaningful.
- `all_results.csv`: every model, condition and individual random trial.
- `experiment_config.json`: settings used for the whole comparison.
- `seed_42/`, `seed_43/`, `seed_44/`: each checkpoint, clean evaluation,
  learning curves, input gradient and `attack_comparison/` results.

CSV stores tables; JSON stores labelled values. Configuration files also record
**hashes**, file fingerprints identifying the data and weights actually used.
Each example picture shows one case; the numerical scores cover all 1,459 test pairs.
See [OUTPUTS.md](OUTPUTS.md) for the individual filenames.

## 9. Commands and code roles

Run from `toy_weather/` after `source .venv/bin/activate`. The prepared file
`local_data/era5_2020_2022/prepared.npz` is already saved, so no download is needed.

```bash
# Repeat training and comparisons in a NEW folder.
python run_experiments.py --dataset era5 \
  --data local_data/era5_2020_2022/prepared.npz \
  --output-root outputs/experiments/my_repeat

# Compare attacks on the existing seed-42 model, without training it again.
# Use another results folder to keep earlier findings intact.
python compare_attacks.py --dataset era5 \
  --checkpoint-dir outputs/experiments/era5_2020_2022/seed_42 \
  --results-dir outputs/experiments/era5_2020_2022/seed_42/my_repeat \
  --budget-k 0.1 0.5
```

`run_experiments.py` refuses an existing output root. The preparation script
`data/prepare_era5_experiment.py` caches public ERA5 and builds the date divisions;
it refuses to replace an existing prepared file. Shorter training windows can use
`--train-end`, with a different `--output`, while keeping validation/test dates fixed.

| File | Role |
|---|---|
| `train.py` | Learn weights and select the validation checkpoint |
| `evaluate.py` | Measure clean forecasts and persistence |
| `check_input_gradient.py` | Compute input sensitivity without changing the input |
| `inspect_weights.py` | Print saved synthetic weights and one learned filter |
| `perturbations.py` | Random changes, one-step attacks and PGD |
| `compare_attacks.py` | Compare conditions on one frozen checkpoint |
| `run_experiments.py` | Repeat training and comparisons for several seeds |
| `attacks.py` / `attack.py` | Small bridges keeping earlier imports/commands working |

## 10. Next steps, not completed results

Moving to Otter is reasonable now that the input-gradient and comparison pipeline
works. First obtain trained weights, prepare one complete input, run a clean forecast,
and verify a temperature-input gradient. See [OTTER_PREPARATION.md](../../../OTTER_PREPARATION.md)
for the training approach and local hardware limits.

Optional improvements to the toy forecaster should be changed one at a time:
check whether training has finished learning, predict a correction to the current
map, add clock information, or provide the previous temperature map. None of these
changes is included in the results above.

If this already inspected test year influences new design choices, treat those
comparisons as exploration. Use validation to choose settings and a fresh unseen
period for final evaluation. Extreme-event thresholds should also be chosen without
looking at final test outcomes. Additional weather variables, deeper CNNs, attacks
on extreme events and multi-step forecasts remain future experiments.

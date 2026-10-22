# Longer forecasts and longer-window attacks

## What changes in this experiment

The same saved five-variable CNN predicts six hours ahead. We feed its predictions
back as weather inputs and repeat, reaching **120 hours (five days)**. At the second
call, the input is the original current state plus the first prediction; later calls
use two predicted states. Only the known time/season channels are refreshed.
Future ERA5 maps are comparison targets, never replacement input weather.

We do **not retrain** the CNN. It was trained on one-step forecasts, so growing
long-range error can come from its own accumulated mistakes. We compare attacked
and clean forecasts at every lead to separate this drift from additional attack damage.
Otter's rollout fine-tuning instead updates model weights through a gradually
lengthened 24–72h forecast loss. We have not applied that training stage to this CNN;
our different attack windows update starting inputs while the weights stay fixed.

## Which inputs and loss are attacked?

- Frozen checkpoint: `outputs/era5/best_model.pt`.
- Change only **current T2m**, input channel 5; leave the older state, other
  variables, time features, true future and CNN weights fixed.
- Same **L∞ limit of ±0.1 K per cell**, for every method and window. Random
  changes use the same per-cell magnitude, with three seeded trials.
- Independently optimise **6, 24 and 72-hour loss windows** from each original
  input. Each objective averages temperature squared error over **every six-hour
  forecast inside the window**, not just its final map.
- Evaluate each resulting starting input through **120 hours**, without adding
  further disturbances. Earlier predictions retain their gradient connections
  when constructing attacks.

For example, the 24-hour objective averages errors at +6, +12, +18 and +24 hours:

```text
initial changed input → +6h prediction → +12h → +18h → +24h
                         error          error   error   error
                             average of these four errors
```

The gradient says how changing the initial temperature affects this whole average.
FGSM makes one sign-gradient update; PGD repeatedly updates and projects back
inside the same input limit. A “one-step attack” means one optimisation update;
it can still differentiate through a multi-step forecast. Errors are normalised
using the existing training statistics during optimisation; plots report Kelvin.
For this single variable, scaling by its fixed standard deviation does not change
which input maximises the objective.

The reference paper, *Atmospheric Predictability Beyond 30 Days*, **reduces**
forecast error by optimising initial states through progressively longer forecasts.
Here we **increase** error, optimise each window separately, and compare how long
its damage persists. This is an adaptation of its idea, not a reproduction of its
GraphCast results or its progressive optimisation schedule.

## Comparison and saved files

Run from `toy_weather/` with the existing environment:

```bash
python compare_rollouts.py \
  --results-dir outputs/era5/rollout_windows \
  --loss-windows 6 24 72 --horizon-hours 120 \
  --validation-samples 32 --test-samples 0 \
  --attack-steps 10 20 --step-factors 1 2
```

The results folder must be new. `--test-samples 0` uses all eligible cases.
For a small practice run use a different folder and `--test-samples 16`.
`--norm l2` supports the existing RMS interpretation of `--budget`; the completed
comparison here uses L∞ only.

PGD settings are selected separately for each window on **32 spaced validation
cases from 2021**, before evaluating 2022. Candidates use 10/20 updates and step
size factors 1/2. Choose the cheapest setting retaining 99% of the best observed
validation loss increase. The comparison is matched in **input limits**, not in
computational cost: a 72-hour gradient passes through 12 CNN calls, versus one
for a six-hour gradient.

| File in `outputs/era5/rollout_windows/` | Contents |
|---|---|
| `clean_rollout_errors.png` | Clean CNN and no-change RMSE for all five variables |
| `one_step_loss_windows.png` | FGSM: total temperature error and extra error over clean |
| `pgd_loss_windows.png` | PGD: same comparison; stars mark the loss-window endpoints |
| `attack_window_comparison.png` | FGSM/PGD error at 120h for each optimisation window |
| `attack_window_extra_error.png` | Same comparison after subtracting clean 120h error |
| `lead_time_metrics.csv` | Physical RMSE, increase over clean, fraction of cases made worse, at every lead |
| `attack_duration.csv` | Descriptive duration measures at a predeclared +0.05 K threshold |
| `per_case_errors.npz` | Case dates, per-case MSE `[case,20,5]`, starting perturbations `[case,16,16]` |
| `validation_sweep.*`, `selected_settings.json` | Validation-only PGD choices and checkpoint/data fingerprints |
| `conditions.json`, `run_config.json`, `summary.json` | Actual disturbance sizes, costs, settings and limitations |

The original one-step attack comparison is drawn separately in
`outputs/era5/one_step_attack_comparison.png` from its existing saved results.
It uses 1,458 cases; a five-day trajectory needs more future coverage, so the
new comparison has fewer cases. Do not interpret their small clean-score difference
as a changed checkpoint.

The no-change baseline repeats the **original unperturbed state at t** at every
lead: it never refreshes with future observations. Its temperature error zigzags
as future temperatures move away from and back towards the starting temperature,
consistent with the daily temperature cycle. At +6/+12/+18/+24h its RMSE is
2.637/3.593/2.949/1.858 K. An independent calculation directly from physical ERA5
maps reproduced all saved temperature baseline errors within rounding tolerance;
`persistence_check.json` records this check. The grey baseline's zigzag is not
the CNN prediction curve.

## How to interpret attack duration

“Validity” here means **extra forecast error over the clean rollout**, not that
an attacked forecast is physically valid. The chosen descriptive threshold is
**attacked temperature RMSE − clean temperature RMSE ≥ 0.05 K**.
We report the first/last lead above it and, separately, how long it stays above
continuously from +6h. A late return above the threshold is not continuous damage.
Still above at 120h means the end of the effect was not observed in this experiment.
A negative difference means the attack accidentally improved that later forecast.

This threshold is not a statistical significance test or an atmospheric
predictability limit. Results pool cell/case squared errors before taking the
square root. Random-trial mean pools trial MSEs too; all three trials are retained.

## Completed run — 21 October 2026

**1,439 complete five-day test cases**, initialised from 1 January 2022 06:00 UTC
through 26 December 2022 18:00 UTC. Every method uses these same cases. Targets
remain within 2022, and incomplete six-hour sequences are excluded.

Temperature RMSE (K), lower is better:

| Condition | +6h | +24h | +72h | +120h |
|---|---:|---:|---:|---:|
| Clean CNN | 0.789 | 1.567 | 3.295 | 4.269 |
| No change | 2.637 | 1.858 | 2.736 | 2.984 |
| Random (3 trials) | 0.794 | 1.567 | 3.294 | 4.269 |
| FGSM, 6h loss | 0.856 | 1.588 | 3.306 | 4.277 |
| FGSM, 24h loss | 0.830 | 1.648 | 3.354 | 4.313 |
| FGSM, 72h loss | 0.806 | 1.618 | 3.531 | 4.495 |
| PGD, 6h loss | 0.856 | 1.588 | 3.306 | 4.277 |
| PGD, 24h loss | 0.830 | 1.648 | 3.355 | 4.312 |
| PGD, 72h loss | 0.805 | 1.618 | 3.548 | 4.512 |

The **six-hour-loss attack is strongest early and fades quickly**. The
**72-hour-loss attack sacrifices immediate damage but leaves more later damage**:
PGD adds **0.244 K** at 120h, versus **0.008 K** for the six-hour objective.
FGSM shows the same broad pattern; PGD adds a modest further increase for the
longest window. Random changes have very little average effect here.

At the predeclared +0.05 K extra-RMSE threshold:

| Attack loss window | FGSM leads above threshold | PGD leads above threshold |
|---|---|---|
| 6h | +6h only | +6h only |
| 24h | +12h through +78h | +12h through +78h |
| 72h | +24h through +120h | +24h through +120h |

These intervals cover every sampled six-hour lead inside their endpoints. The
72h attack is **still above the threshold at the end**, 48 hours beyond its loss
window; its eventual end is unknown. The 24/72h attacks do not cross the threshold
at +6h, so their “continuous from start” CSV value is zero despite later damage.

**The clean CNN itself loses long-range skill.** At 120h its temperature RMSE is
4.269 K versus 2.984 K for no change. It also loses to no change at 72h. Strong
six-hour performance therefore does not establish good multi-day forecasting.
The attack results describe this fixed model, including its drift; they are not
results for a well-trained medium-range weather model.

Validation selected 10 PGD updates for all windows, with step sizes **0.01, 0.01,
0.02 K** for 6/24/72h respectively, one attempt starting from clean input. Nominal
CNN steps differentiated per case are **10, 40, 120** for PGD, and **1, 4, 12**
for FGSM; scoring calls cost extra. Test attack-plus-rollout times on this CPU
were about 29/81/217 seconds for PGD and 14/21/37 seconds for FGSM. These are
separately tuned conditions, not an experiment isolating window length from every
other attack setting. The FGSM step magnitude stays 0.1 K in all three windows.

Actual maximum cell changes were 0.10000004 K (float32 rounding); PGD input RMS
changes were about 0.100/0.099/0.097 K. All norm and separate cell-cap checks passed.
Checkpoint and prepared-data fingerprints stayed unchanged. **30 tests passed**,
including full-trajectory gradients, opposite short/long objectives, exact time
pairing and split isolation. The existing third-party NetCDF/NumPy warning remains.

## Limits

One checkpoint, one region, one input variable and overlapping 2022 cases make
this an exploratory comparison. The same previously inspected test year is not a
new final holdout. True future ERA5 states are available to the attacker here,
so this is a retrospective white-box test, not a real-time attack demonstration.
Bounded temperature changes alone do not prove physical plausibility. Long-window
attacks cost more, and one-step training may make later trajectories unreliable.
No claim about a physical five-day predictability limit follows from these curves.

The subsequent [looser-bound experiment](LOOSER_BOUND_EXPERIMENTS.md) repeats
both FGSM and PGD at a validation-selected ±2 K input limit. Its plots live in
`outputs/era5/rollout_windows_2K/`; the original ±0.1 K plots above remain intact.

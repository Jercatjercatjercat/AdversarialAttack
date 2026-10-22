# Looser input limits — 21 October 2026

The original ±0.1 K comparison barely separated FGSM, PGD and MI-FGSM.
This follow-up checks whether allowing larger initial temperature changes makes
repeated updates more useful. The CNN weights, data and normalization stay fixed.
Only the **current input T2m map** changes; the older weather, other variables,
calendar features and true future observations stay unchanged.

## Choosing the limit without consulting test results

A saved pilot uses 32 spaced **2021 validation** examples. It tries every
predeclared limit: **±0.25, ±0.5, ±1 and ±2 K**, retaining every result.
It compares FGSM with 40-update PGD/MI-FGSM for the six-hour objective, and FGSM
with 20-update PGD for the 72-hour average-loss objective.

Before the pilot, we defined a visibility rule: choose the smallest tested limit
where PGD exceeds FGSM by at least **0.005 K** in six-hour RMSE and **0.01 K** in
72-hour window RMSE. These are display criteria, not significance tests or physical
plausibility criteria. If none qualifies, the code explicitly labels that failure.

| Per-cell limit | Six-hour PGD − FGSM (K) | 72-hour window PGD − FGSM (K) |
|---|---:|---:|
| ±0.25 K | 0.000202 | 0.038142 |
| ±0.5 K | 0.001259 | 0.132541 |
| ±1 K | 0.004931 | 0.368187 |
| ±2 K | 0.021306 | 0.935537 |

The rule selected **±2 K**. It is **20 times the original ±0.1 K allowance**;
it should not be described as a slightly larger or physically justified attack.
A much smaller increase already separates longer-window attacks on validation.
The shared ±2 K setting is used for both full-test comparisons to keep their input
rules consistent. The cutoffs are a transparent educational choice, not a natural
boundary: ±1 K narrowly misses the chosen six-hour threshold.

`outputs/era5/looser_budget_pilot/` contains the protocol written before the pilot,
all scores, validation indices, selected budget and model/data fingerprints.
`budget_sensitivity.png` displays every pilot result; `select_looser_budget.py`
implements the rule, and its tests reject test-derived selections.
No 2022 test score chooses the limit. This remains an exploratory study because
2022 was inspected in earlier experiments.

## 1. Fixed update counts

Use all **1,458** original six-hour test cases. Compare random noise (three trials),
FGSM once, and PGD/MI-FGSM at **5, 10, 20 and 40 updates**, with one clean-start
attempt. Both repeated attacks use **step size = 2 / update count K**:
0.4, 0.2, 0.1 and 0.05 K respectively. Momentum is 1 for MI-FGSM.

The ±2 K norm bound and separate cell cap are both explicit; an old 1 K cell cap
must not silently restrict this experiment. Projection always limits the total
change from the original input, not just the latest update.

Results: `outputs/era5/fixed_step_attacks_2K/`. `step_comparison.png` shows the
same overall-scale/zoomed-scale comparison used for the original ±0.1 K run.

Completed six-hour test RMSE: **FGSM = 2.399102 K**.

| Updates | PGD RMSE (K) | MI-FGSM RMSE (K) |
|---:|---:|---:|
| 5 | 2.422305 | 2.411383 |
| 10 | 2.426208 | 2.413306 |
| 20 | 2.428464 | 2.414545 |
| 40 | 2.429732 | 2.415166 |

PGD at 40 updates adds **0.030630 K** beyond FGSM, compared with **0.000069 K**
at the original ±0.1 K bound. The gap is now visible but remains modest (about
1.3% of FGSM's attacked RMSE). MI-FGSM adds **0.016064 K**, and again does not
outperform ordinary PGD. Random-trial pooled RMSE is **1.881556 K**, versus
**0.788087 K** for clean forecasts. Random changes also become much more damaging
when the allowed magnitude is enlarged.

## 2. Different forecast-loss windows

The original `rollout_windows/one_step_loss_windows.png` is **FGSM**, not PGD.
The new experiment includes **both methods**, each with 6/24/72-hour loss windows,
evaluated through **120 hours** on the same **1,439 complete test cases**.

Each window averages temperature MSE over all its six-hour leads. Backpropagation
passes through the whole predicted trajectory to the initial input. FGSM makes
one input update; PGD makes repeated input updates. Neither retrains the CNN nor
adds new attacks later in the forecast. Calendar features advance, but future
observed weather never replaces predicted input weather.

PGD settings are chosen separately per window using 32 validation trajectories,
trying 10/20 updates and step factors 1/2. Choose the cheapest condition retaining
99% of the best observed validation MSE increase. Settings are saved before test
scoring. Different windows therefore have different computational costs.

Results: `outputs/era5/rollout_windows_2K/`:

- `one_step_loss_windows.png`: FGSM total error and extra error above clean.
- `pgd_loss_windows.png`: PGD, using the same axes and input limit.
- `method_loss_window_comparison.png`: FGSM/PGD extra errors side by side, with a shared vertical scale.
- `pgd_minus_fgsm.png`: directly subtract FGSM RMSE from PGD RMSE at every lead, separately for each window.
- `per_case_errors.npz`, `lead_time_metrics.csv`: underlying per-case errors and pooled physical-unit scores.
- `window_objective_scores.csv`: error pooled over each attack's own loss window, plus its 120h score.

Higher error means a stronger attack. A positive PGD-minus-FGSM curve means PGD
caused more error at that lead. PGD optimises the window average, so it need not
beat FGSM at every individual lead or beyond its optimisation window.

## Completed longer-window findings

Temperature error pooled over each attack's **own loss window**, not just its
last forecast (K; higher means stronger attack):

| Loss window | Clean | FGSM | PGD | PGD updates | Step size (K) |
|---|---:|---:|---:|---:|---:|
| 6h | 0.789 | 2.400 | 2.446 | 10 | 0.4 |
| 24h | 1.236 | 2.719 | 2.916 | 20 | 0.2 |
| 72h | 2.296 | 4.069 | 4.923 | 20 | 0.2 |

The additional PGD effect is much larger for longer-window objectives:
**+0.046 / +0.197 / +0.854 K** over FGSM for 6/24/72h windows respectively.
The paired fraction of cases with higher PGD window MSE is approximately
**82.4% / 100% / 99.9%**; these are descriptive case fractions, not significance
tests, and cases overlap in time. No claim that PGD wins every individual case or
forecast lead follows from a higher pooled score.

At **120 hours**, beyond all three optimisation windows:

| Attack loss window | FGSM RMSE (K) | PGD RMSE (K) | PGD − FGSM (K) |
|---|---:|---:|---:|
| 6h | 4.818 | 4.842 | 0.024 |
| 24h | 5.638 | 5.995 | 0.357 |
| 72h | 6.912 | 9.022 | 2.109 |

For the 72h objective, PGD's extra effect over FGSM is **2.109 K** at five days.
The input limit still applies only to the starting temperature: it does not cap
later forecast error. Clean five-day RMSE is **4.269 K**, compared with **2.984 K**
for predicting no change. This substantial pre-existing model drift remains a
limitation of the experiment.

At ±2 K, every tested condition (including random noise) remains above the old
+0.05 K extra-error threshold through 120h. The endpoint of its effect is not
observed here. That small descriptive threshold no longer distinguishes these
stronger disturbances well; the direct FGSM/PGD difference curves are more useful.

The fixed-step and longer-window comparisons deliberately use different PGD
settings: the former fixes α=budget/updates, while the latter independently tunes
steps/factors on validation. Their six-hour PGD scores also use slightly different
test-case sets (1,458 versus 1,439). They must not be treated as duplicate runs
with identical parameters.

All measured perturbations obeyed the ±2 K limits within float32 tolerance.
The audit confirmed identical case dates and clean forecasts relative to the old
runs, unchanged weights/data fingerprints, and finite errors throughout. The
correctness suite has **32 passing tests**, with the existing third-party
NetCDF/NumPy warning. `paired_method_comparison.csv` records the direct method gaps;
`completed_test_summary.json` collects the completed comparisons.

## Reproduce

From `toy_weather/`; use new results folders for repeats:

```bash
python select_looser_budget.py
# Read selected_budget.json before setting the following --budget/--cell-cap.
python compare_attacks.py --results-dir outputs/era5/fixed_step_attacks_2K \
  --variable t2m --objective t2m --budget 2 --norm linf \
  --methods random one_step pgd mi_fgsm --steps 5 10 20 40 \
  --step-factor 1 --restarts 1 --cell-cap 2 --random-trials 3 --split test
python plot_attack_steps.py --results-dir outputs/era5/fixed_step_attacks_2K
python compare_rollouts.py --results-dir outputs/era5/rollout_windows_2K \
  --budget 2 --cell-cap 2 --loss-windows 6 24 72 --horizon-hours 120 \
  --validation-samples 32 --test-samples 0 --attack-steps 10 20 --step-factors 1 2
python summarize_looser_bounds.py
```

The last command checks that the new and original runs use identical test cases
and clean forecasts, verifies saved fingerprints/bounds, and writes the paired
window summary and budget-selection references. It does not run more attacks or
choose any settings from test scores.

## Interpretation limits

Larger limits enlarge the set of allowed inputs; they do not guarantee that every
particular attack will improve or that momentum will outperform ordinary PGD.
Step sizes scale with the budget in these comparisons. The budget pilot supports
a growing method gap under this protocol on validation, not a general
claim across weather models. The CNN is trained for one-step forecasts and already
loses to predicting no change at long leads without any attack. Known future
observations are available to construct these retrospective white-box attacks.
The old ±0.1 K files remain preserved for direct comparison.

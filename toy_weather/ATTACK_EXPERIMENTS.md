# Choosing attack steps and comparing input restrictions

This extension attacks the already trained second-iteration CNN. Its weights,
weather dataset and true future states remain fixed. The input change is applied
only to CURRENT temperature. Earlier weather, other variables and clock features
are unchanged. The new sweep maximizes temperature forecast MSE, whereas the older
`attack_comparison/` maximized the combined five-variable MSE. Their attack scores
therefore do not isolate the effect of adding momentum or more iterations.

## What is configurable

FGSM, labelled `one_step`, always makes one full-budget gradient update under
Linf. `--steps` cannot turn it into an iterative method. Under L2 the one-step
method uses a unit-L2 gradient; under L1 it uses the strongest available cell.
These are not all called FGSM because the directions differ.

PGD allows a chosen number of updates, step size and attempts. The first attempt
starts with clean weather. Additional attempts start with random feasible changes,
and each example keeps its highest-loss candidate across all attempts. Thus
`--restarts 3` means THREE total attempts, including the clean start, rather than
three random attempts plus a clean one. Random starts are seeded consistently.

MI-FGSM keeps a running memory of L1-normalized gradients and moves using the sign
of that memory. Its default momentum coefficient is 1. This implementation uses
Linf, a forecast loss and explicit budget projection. With one update it reduces
to FGSM; a selected one-update setting is not evidence that accumulated momentum
improves attacks. The sweep includes longer runs to examine that possibility.

`top_k` is an educational L0 attack. Each update changes the candidate map in a
gradient direction, then keeps only the k largest accumulated changes. Other cells
are reset to ORIGINAL weather, so repeatedly choosing cells never accumulates more
than k nonzero changes. A separate amplitude cap limits each selected cell. This
is a hard-thresholding heuristic, not a proof of the optimal k locations. Random
L0 chooses random locations with the same count/amplitude limits.

## What each budget means

For one 16×16 temperature map there are 256 entries. The `--budget` number uses
Kelvin differences, but its interpretation depends on the restriction:

| Restriction | Budget meaning | At budget 0.1 K |
|---|---|---|
| Linf | Largest absolute cell change | Every cell changes by at most 0.1 K |
| L2 | Root mean square (RMS) input change | L2 radius is 16×0.1 = 1.6 K |
| L1 | Mean absolute input change | L1 radius is 256×0.1 = 25.6 K |
| L0 | Amplitude per selected cell; k separately limits count | At most k cells, each changing by at most 0.1 K |

L0 is a nonzero-entry count, conventionally called a norm. It does not bound
amplitude on its own. The sweep additionally sets a 1 K cell cap for all norms.
This cap is tighter than an unconstrained L1/L2 concentration in a single cell;
for Linf/L0 at budget 0.1 K the existing smaller amplitude limit already applies.

Equal numerical budgets are NOT equal disturbance allowances. Four cells changed
by 0.1 K have RMS at most 0.0125 K. L1 allows larger local values than Linf and can
produce a larger RMS. Differences in attacked RMSE must be interpreted alongside
changed-cell count, max cell change, mean absolute change, RMS and sample-wise norms.
No norm or amplitude cap ensures a physically consistent atmospheric state.

L1 projection uses soft thresholding to enforce total absolute magnitude. It does
not enforce an exact count. Its gradient step selects the strongest available cell;
a cell already at its cap in the outward direction is skipped so other cells can
compete. More updates may be needed to distribute the available budget. This is
one chosen projected-ascent heuristic, not an exhaustive L1 attack evaluation.

## Worked example: what the gradient, step and radius represent

The fixed trained network is g(i, w), where i is an input weather example and w
is its saved weights. The true future o is a fixed reference, not the network's
prediction. The forecast is g(i, w), and its error is the mean of
(o - g(i, w)) squared across the selected forecast cells. Training changes w to
reduce error. Our attack holds w and o fixed and changes an allowed part of i to
increase error. Equivalently, it optimises a change delta in g(i_original + delta, w).
The original recorded weather remains untouched; only a working copy changes.

Sensitivity means how a small input change affects the loss locally. For example,
a derivative +2 at a cell means that a sufficiently small +0.01 change in that
input coordinate is predicted to increase loss by about 0.02, keeping all other
coordinates fixed. This is a first-order approximation, not a guarantee for large
changes. In our code the coordinates and MSE are normalized. A derivative of loss
is not the same as a derivative of an individual forecast value.
`requires_grad_(True)` permits gradient calculation; it does not itself calculate
a gradient or change the input. `torch.autograd.grad` calculates it.

For one small 2-by-2 example, these arrays illustrate the selected input channel:

```text
original:       candidate:       delta = candidate - original:
[[280, 281],    [[280.3, 281.4],  [[0.3, 0.4],
 [282, 283]]     [282.0, 283.0]]   [0.0, 0.0]]
```

The numbers above are physical temperatures in Kelvin and differences in Kelvin.
Delta is the accumulated change from the ORIGINAL input, not only the latest step.
The gradient is another map of the same shape, but contains loss derivatives,
not temperatures. Suppose the gradient is:

```text
gradient = [[3, 4], [0, 0]]
L2 length = sqrt(3^2 + 4^2) = 5
direction = gradient / 5 = [[0.6, 0.8], [0, 0]]
```

The direction has L2 length one. Dividing removes overall magnitude while keeping
the relative 3-to-4 preference. A step size alpha = 0.5 produces a step
[[0.3, 0.4], [0, 0]], whose L2 length is 0.5. The radius R is a separate limit
on the TOTAL accumulated delta. If R = 0.4, this candidate exceeds the limit,
so L2 projection multiplies all changes by 0.4/0.5 = 0.8:

```text
projected delta = [[0.24, 0.32], [0, 0]]
L2 length = sqrt(0.24^2 + 0.32^2) = 0.4
```

If delta already fits, projection leaves it unchanged. The next PGD step
recalculates the gradient at the changed input, adds another step, and enforces
the radius again. Alpha controls one move; R controls distance from the original.
These examples assume physical coordinates for clarity. The implementation divides
physical limits and step sizes by the selected variable's training standard
deviation before calling `perturbations.py`, which operates in normalized units.

Linf projection is different. `delta.clamp(-radius, radius)` limits EACH cell
independently. With radius 0.4, [[0.3, 0.4], [0, 0]] is allowed by Linf, even
though its L2 size is 0.5. The same variable name `radius` describes different
geometries in the two branches; equal numbers do not impose the same restriction.

For the actual 16-by-16 map, 256 cells are changed at most. L2 size is the square
root of the SUM of squared changes. RMS size is the square root of their MEAN.
Consequently RMS = L2/16. Our command-line `--budget 0.1` means L2 radius 1.6 K
under L2 but per-cell radius 0.1 K under Linf. Changing all 256 cells by 0.1 K
uses L2 size 1.6 K. A single 1.6 K change would have the same L2 size, but our
additional 1 K cell cap forbids it. Linf 0.1 K would forbid any cell above 0.1 K.

## Which candidates are retained, and which dates are attacked

Keeping the highest-loss candidate happens INSIDE one attack method, for one
example and one fixed setting. For example, a PGD run might encounter losses
0.10 (clean), 0.14 (step 1), 0.13 (step 2) and 0.16 (step 3). It returns the
step-3 input. If later steps score worse, the earlier best input is retained.
Additional starts compete within that same run. The code compares per-example
MSE on the selected objective; for one variable and fixed scaling, its ordering
matches RMSE. It does not merge FGSM, PGD, momentum and random results into one
winner. Their separate scores remain in the results and plots. Random baselines
are measured directly rather than retained only when they beat the clean input.

The completed `tuned_attacks_steps` comparison attacks ALL 1,458 eligible examples
in the 2022 test period, not only a few displayed dates. Each example contains
weather six hours ago, weather now and a true future six hours ahead. All three
timestamps must lie in 2022. The first example uses 1 January 00:00 and 06:00 UTC
to forecast 12:00; the last uses 31 December 06:00 and 12:00 to forecast 18:00.
Current times therefore run every six hours from 1 January 06:00 to 31 December
12:00 UTC. For EACH example, a fresh disturbance is optimised on its current
temperature map (input channel 5 in the default 14-channel layout). Its older
temperature map, other variables, clock channels and future target stay fixed.
The true temperature at t+6h is used to calculate attack loss. This is a
retrospective white-box evaluation with a known future reference, not a claim
that an operational attacker already knows tomorrow's weather. Changes to one
example are not carried into the next example. This is not an attack on one
continuous year-long autoregressive forecast.

The attack-settings sweep used 32 evenly spaced 2021 validation examples. Choosing
update counts on validation is separate from optimising the disturbance for each
2022 test example under those fixed settings. Test RMSE pools squared errors over
all tested cases and cells before taking the square root; a few displayed examples
do not constitute the whole test evaluation.

## How step sizes and step counts are compared

An explicit step factor sets:

```text
step size = factor × norm radius / number of updates
```

For Linf, budget 0.1 K, ten updates and factor 1 give a 0.01 K update before
projection. Under L2/L1 the step is in that norm's units, not a change of that
size at every cell. Under L0 it uses the per-cell amplitude budget. The factors
0.5, 1 and 2 cover smaller and larger updates. Changing the number of steps here
also changes step size; curves compare complete settings, not a single fixed-size
trajectory truncated at different times.

The validation sweep tries 1, 5, 10, 20 and 40 updates, factors 0.5/1/2 and one or
three attempts. It separately groups each method, norm, budget and k. Random trials
are not selected for being particularly damaging.

By default, the selector chooses the LEAST gradient effort that retains at least
99% of the greatest observed INCREASE in validation MSE. This is an efficiency
convention, not 99% attack success or a certified guarantee. For example, if clean
MSE is 1 and the strongest observed attacked MSE is 2, the cutoff is 1.99. A cheap
setting with MSE 1.995 qualifies. `--gain-fraction 1` instead selects the exact best
observed score. Cost is gradient-update count, with deterministic tie-breaking;
recorded wall time can vary with machine load.

A plateau is evidence about this grid of settings and these cases. It does not
prove that no stronger attack exists. Settings include hashes of the weights/data.
The test command rejects selections that did not come from validation or that
belong to different weights/data. It never selects a setting using test RMSE.

## Commands and saved evidence

Run from `toy_weather/` with the existing Python environment:

```bash
# New validation sweep; the default uses 64 evenly spaced validation examples.
python tune_attacks.py --results-dir outputs/era5/my_attack_tuning

# Apply those fixed settings to the FULL 2022 test period.
python compare_attacks.py \
  --settings outputs/era5/my_attack_tuning/selected_settings.json \
  --results-dir outputs/era5/my_tuned_test

# Vary the allowed sparse-cell count on validation, rather than selecting it on test.
python tune_attacks.py --norm l0 --k 1 4 16 \
  --results-dir outputs/era5/my_sparse_tuning
```

`--max-samples 0` uses the complete validation period. Results folders refuse
replacement. The historical comparison command is still supported: omitting
`--step-factor` retains the old PGD schedule, with its initial full one-step
candidate. Explicit factors in the new sweep use a constant step size per run.

The completed `outputs/era5/attack_tuning_steps/` contains 165 conditions on 32
FIXED, evenly spaced 2021 validation cases: the five step counts, three factors,
two attempt counts, four norms, random baselines and one-step comparisons. L0 uses
k=4. The saved plot shows the curves, clean error and selected effort. The exact
strongest settings were also retained separately; the near-best settings were
chosen from the same validation rows before the new test evaluation, without
rerunning or selecting attacks on test data.

- `results.csv` / `results.json`: each validation condition and measured changes.
- `step_sweep.png`: error versus update count.
- `selected_settings.json`: near-best settings, frozen for test evaluation.
- `strongest_validation_settings.json`: original strict-maximum choices for comparison.

`outputs/era5/tuned_attacks_steps/` contains results after applying those fixed
near-best settings to all 1,458 test examples. Its `comparison_config.json` records
the selection fingerprint, weights/data identities, cases and objective. These
2022 comparisons remain exploratory because we had inspected that year earlier.

The first sweep is deliberately small. It tests one budget (0.1 K), one model
seed and a validation subset. It does not establish optimal step counts for all
budgets, variables, weather events or models. Input-change bounds are checked on
the full test set with float32 rounding tolerances: cell tolerance 1e−5 in the
variable's units; L1 tolerance scales with cell count and L2 with its square root.
The tolerances do not change the requested mathematical budgets; measured sizes
and tolerances are recorded so the checks can be inspected.

## Completed findings, 14 October 2026

The near-best validation selections chose one update/one attempt for PGD-Linf,
MI-FGSM-Linf, PGD-L2 and top-k with k=4. L1 selected 40 updates, factor 2 and one
attempt. The strict maximum and 99%-gain choices differ in some groups, so both
are saved. This does not establish that one update is sufficient at larger budgets,
or that L1 has converged: its strongest score occurred at the largest tested count.
The selected one-update MI-FGSM is equivalent to FGSM, not an observed momentum gain.

Full 2022 test results with clean temperature RMSE 0.788 K:

| Restriction | Selected updates | Random RMSE (K) | Directed RMSE (K) | Largest cell change (K) | Input RMS (K) |
|---|---:|---:|---:|---:|---:|
| linf | 1 | 0.793 | 0.855 | 0.100 | 0.1000 |
| l2 | 1 | 0.793 | 0.880 | 1.000 | 0.1000 |
| l1 | 40 | 0.795 | 1.027 | 1.000 | 0.3108 |
| l0 | 1 | 0.788 | 0.793 | 0.100 | 0.0125 |

Random scores average three trials. Directed means PGD except for L0, which uses
top-k. L1's larger error accompanies a much larger RMS and permitted local change
than Linf. L0's small effect accompanies a much smaller RMS. The table does not
rank the four norms under equal disturbance sizes. All input bounds, sparse counts
and caps passed full-test checks with the recorded floating-point tolerances.

A separate sparse-count validation sweep (`outputs/era5/sparse_count_tuning/`)
compared k=1,4,16, update counts 1,10,40 and factors 1,2, with one attempt on the
same 32 validation cases. It selected one update for each count. The corresponding
fixed-setting full-test evaluation is in `outputs/era5/sparse_count_test/`.
Each cell has the SAME 0.1 K amplitude cap; allowing more cells grants a larger
total disturbance, so increased error is not a comparison at equal total size.

| Allowed cells k | Random test RMSE (K) | Top-k test RMSE (K) |
|---|---:|---:|
| 1 | 0.7881 | 0.7896 |
| 4 | 0.7882 | 0.7933 |
| 16 | 0.7884 | 0.8028 |

The extended correctness suite has 25 passing tests. New checks include L1/L0
projection, accumulated sparse counts, frozen weights, zero gradients/budgets,
FGSM's one-update semantics, zero-momentum equivalence, random starts, the near-best
effort selector and validation-to-test handoff. A selection labelled as test-derived
is rejected. The existing third-party NetCDF/NumPy import warning remains, while
the NetCDF/Zarr value-agreement test passes. No CNN retraining was needed.

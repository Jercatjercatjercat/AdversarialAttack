# Comparing attacks while keeping the forecasting model simple

The central question is: do small, deliberately chosen input changes damage
forecasts more than equally sized random changes? The default CNN stays at three
layers. More layers and additional weather variables are future side experiments.

## Two different baselines

The forecasting baseline is **persistence**: predict that temperature six hours
later equals its current value. It does not learn weights.

The perturbation baseline is **random noise**: change the model's input in random
directions without consulting its gradient or target. It tests whether the attack's
choice of direction matters, rather than any disturbance simply hurting the model.

The old January ERA5 experiment had CNN RMSE 1.434 K versus persistence 1.311 K.
Lower is better: that CNN was slightly worse than the simple unchanged-map forecast.
An attack could still hurt it, but that result alone would not demonstrate the
vulnerability of a useful forecaster. The larger dataset must be assessed afresh.
Do not compare January error directly with whole-year error as if test data were equal.

## Fixed dates, not fixed training weights

The larger experiment uses actual ERA5 via WeatherBench:

| Purpose | Dates | Six-hour pairs |
|---|---|---:|
| Training | 2020 | 1,463 |
| Validation | 2021 | 1,459 |
| Testing | 2022 | 1,459 |

"Fixed" means keep evaluation dates identical across experiments. If training grows
from January to all of 2020, validation is still 2021 and testing is still 2022.
No weather state is shared across split boundaries. Training windows can change,
but normalization always comes from that run's training inputs only. Input and
future target use the same training mean/std. Maps are 16×16 and only T2m is used.

Train models with seeds 42, 43 and 44 to see whether results depend on initialization
and shuffled batch order. They use exactly the same prepared data. For synthetic
experiments, `--data-seed 42` fixes examples independently of the model's `--seed`.
An epoch is still a pass through all training pairs; each attack uses the best
validation checkpoint, frozen after training.

B in `[B,1,16,16]` means pairs in the current batch. It is not the dataset size.
Input x is the starting map; target y is the real map six hours later. Neither the
weights nor y changes during an attack.

## Budget definitions and fair comparisons

Let delta be changed input minus original input, expressed in Kelvin.
For each sample, flatten its map into a vector of 256 numbers.

- **Linf** limits `max(abs(delta))`. Every cell stays within the limit.
- **L2** limits `sqrt(sum(delta**2))`. All cells share a total budget; individual
  changes may be larger. This is the Frobenius norm of the map, not a spectral norm.
- **Input RMS** is `sqrt(mean(delta**2)) = L2 / sqrt(256)`.

The CLI uses `--budget-k r`. For Linf the radius is r. For L2 the radius is 16r,
so the input RMS bound is r. For r=0.5 K, the L2 radius is 8 K. This is a size
convention for readable comparison, not proof that both allowed sets are identical:
the Linf set is contained inside the corresponding L2 set. L2 can concentrate
changes into fewer cells. A 0.5 K RMS bound does NOT impose a 0.5 K individual cap.

Every result records largest cell change, RMS input change and largest per-sample
L2 size. RMS input change is calculated over all cells and test samples. Attack
constraints are checked separately for every sample. The norm is unweighted here;
latitude-weighted error remains an optional forecasting metric. Squared temperature
change is energy-like as a mathematical size measure, not physical atmospheric energy.

The temperature-only training standard deviation converts physical budgets to
normalized units. Future multi-variable experiments will need explicit scales and
units per variable rather than adding Kelvin and m/s blindly.

## Methods

| Method | How it chooses input changes |
|---|---|
| Random, Linf | Independent random +/- r changes at all cells |
| Random, L2 | Random Gaussian direction, scaled to the full L2 radius |
| FGSM, Linf | One step using each input-gradient sign |
| One-step L2 | One step using a unit-length input gradient |
| PGD | Repeated gradient steps projected inside the chosen norm budget |

Random trials use the full budget and three recorded seeds. Their results are
averaged, not selected for maximum damage. The same noise seeds are used across
model comparisons. PGD starts from the clean input and retains the strongest
candidate per sample, including clean and single-step inputs. PGD does not have
random restarts. These are untargeted attacks increasing full-map MSE, not attacks
specifically designed to hide a heatwave or damage one selected region.

## Structure and commands

```text
train.py                         Train the unchanged three-layer CNN
evaluate.py                      Check clean forecasts and persistence
check_input_gradient.py          Verify differentiability
perturbations.py                 Input-changing functions, including random noise
compare_attacks.py               Run all methods/norms on one frozen checkpoint
run_experiments.py               Train multiple model seeds and collect comparisons
data/preprocessing.py            Crop, pair, split, normalize local weather files
data/prepare_era5_experiment.py   Cache larger public ERA5 and prepare fixed dates
attacks.py / attack.py           Thin bridges for older examples and commands
```

From `toy_weather/` with `.venv` activated:

```bash
# This is already prepared in the current workspace.
python -m data.prepare_era5_experiment

# Full repeat: use a fresh output folder to preserve the existing completed runs.
python run_experiments.py --dataset era5 \
  --data local_data/era5_2020_2022/prepared.npz \
  --output-root outputs/experiments/my_repeat

# Just compare the existing seed-42 model; no retraining needed.
python compare_attacks.py --dataset era5 \
  --checkpoint-dir outputs/experiments/era5_2020_2022/seed_42 \
  --budget-k 0.1 0.5
```

Preparation refuses to replace an existing prepared file. Reuse it directly or
choose another `--output`. To reduce training to January while preserving full-year
validation/test periods:

```bash
python -m data.prepare_era5_experiment --train-end 2020-02-01 \
  --output local_data/era5_2020_2022/january_training.npz
```

Re-training uses a new folder and computes new training statistics. Test conditions
for that model still use the exact same 2022 weather values.

Each model folder stores its own checkpoint, data identity, training configuration,
curves, clean metrics, gradients and `attack_comparison/`. The comparison folder has
CSV/JSON results, configuration, maps and `comparison.png` (one panel per norm).
Different configurations require different result folders, preventing accidental
mixing. Prepared-file and checkpoint hashes identify the actual files used.

At the top experiment folder:

- `clean_models.csv`: CNN versus persistence for each model seed.
- `all_results.csv`: every random trial and attack condition for every model.
- `summary.csv` / `summary.json`: mean and standard deviation across model seeds.
- `experiment_config.json`: settings used to run the whole experiment.

Random repeats are averaged within each model before aggregating model seeds.
This avoids pretending three noise draws from one model are three trained models.
The reported spread is descriptive variation, not a statistical significance test.

## What comes later

After interpreting the noise/attack and norm results, we can compare extremes with
ordinary temperatures using held-out dates and thresholds defined without test
leakage. Adding winds or changing CNN depth should each be a separate comparison,
so changes in attack outcomes are easy to attribute. These extensions are not part
of the current implementation.

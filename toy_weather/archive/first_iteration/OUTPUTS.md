# A guide to the saved outputs

`outputs/` holds the results of running the project: trained weights, numerical
scores and pictures. A folder is a record of an experiment, not another source-code
module. Earlier demonstrations are preserved alongside the current comparison.

## Which folder should I look at?

```text
outputs/
├── synthetic/                     Original made-up weather demonstration
├── era5/                          Original January 2020 ERA5 demonstration
├── tutorial/                      NCEP real-data fallback, NOT ERA5
├── experiments/
│   ├── era5_2020_2022/             Current larger ERA5 comparison
│   │   ├── clean_models.csv
│   │   ├── all_results.csv
│   │   ├── summary.csv / summary.json
│   │   ├── experiment_config.json
│   │   ├── seed_42/
│   │   ├── seed_43/
│   │   └── seed_44/
│   └── legacy_smoke/              Checks that the old attack commands still work
└── .matplotlib/                   Plotting-library cache, not experiment results
```

Start with **`experiments/era5_2020_2022/`** for the current findings. The training
weather is 2020, validation is 2021, and testing is 2022. Its prepared data is at
`local_data/era5_2020_2022/prepared.npz`, outside `outputs/`. Older `era5/` and
`tutorial/` folders keep their prepared arrays inside the output folder instead.
The raw downloaded weather maps are in `local_data/`; `tutorial/tutorial_cache/`
holds the downloaded tutorial data.

The January model in `era5/` has RMSE 1.434 K versus 1.311 K for predicting no
change, so it lost to that simple baseline. The current larger experiment has mean
RMSE 2.494 K versus 2.624 K for no change, so it wins modestly. The absolute errors
cannot be compared directly between these folders: their test dates differ.

## Why are there three CNNs?

`seed_42/`, `seed_43/` and `seed_44/` contain three separately trained copies of
the **same three-layer CNN**. A seed is a number controlling reproducible random
choices. Changing it changes the starting weights and shuffled training order.

All copies use the same weather pairs, normalization, model design and training
settings. Each trains for 15 epochs and saves its best validation checkpoint.
Training can end with slightly different weights even when everything else matches.

| Folder | Clean test RMSE | No-change test RMSE |
|---|---:|---:|
| `seed_42/` | 2.487 K | 2.624 K |
| `seed_43/` | 2.493 K | 2.624 K |
| `seed_44/` | 2.502 K | 2.624 K |

Repeating training lets us check that an attack finding occurs across different
starting weights. These are not different depths, not one CNN per attack, and not
three forecasts combined into one prediction. We calculate each model's scores
separately and then average the scores. All models share the same test weather, so
three training runs are not three independent test datasets.

The **three random-noise trials** are a separate repetition. They use different
noise draws on a fixed CNN; they do not train more models. We average those trials
within each CNN before averaging across CNNs. Noise draws are matched across CNNs.

## Files shared across the current experiment

| File | What it tells you |
|---|---|
| `clean_models.csv` | Each CNN's untouched forecast error beside the no-change baseline. Read this first. |
| `summary.csv` / `summary.json` | Average results and variation across the three trained models for each method, norm and budget. These two files contain the same summary in different formats. |
| `all_results.csv` | Detailed rows for each model and attack condition, including individual random-noise trials and measured input changes. |
| `experiment_config.json` | Whole-experiment settings: dataset, model seeds, epochs, budgets, PGD steps and noise repeats. |

CSV is a table format; JSON stores labelled values. The reported standard deviation
describes variation between the three trained models. It is not a confidence interval
or a test proving that a small difference matters statistically.

## Inside one CNN folder

Take `experiments/era5_2020_2022/seed_42/` as an example:

| File | Meaning |
|---|---|
| `best_model.pt` | Saved learned weights and supporting information. It is the checkpoint with the lowest validation RMSE, which need not be the final epoch. |
| `training_config.json` | The settings used to train this copy, including seed, batch size, learning rate and epochs. |
| `normalization.json` | Training-input temperature mean and standard deviation, used for both inputs and targets. |
| `data_metadata.json` | Data source, dates, pair counts, grid information and prepared-file identity. |
| `history.json` | Training loss and validation RMSE recorded after each epoch. |
| `training_curves.png` | A picture of those learning histories. Validation helps choose the checkpoint; this is not test performance. |
| `test_metrics.json` | Clean test MSE/RMSE and the no-change baseline. Physical RMSE is `rmse_k`; normalized RMSE has a different scale. |
| `example_pair.png` | An example input, true future and future-minus-input change. |
| `test_example_0.png` | A clean test example: input, true future, CNN forecast and forecast error. |
| `input_gradient.png` | How forecast loss responds to input changes, shown for normalized and Kelvin inputs. This picture alone is not an applied attack. |
| `attack_comparison/` | Scores and pictures after actually changing inputs to this frozen CNN. |

The earlier demonstration folders have many of the same files. They also contain
older `fgsm_metrics.json`, `pgd_metrics.json` and attack pictures. Those belong to
the older runs, not the new three-model summary. Older runs may lack
`training_config.json` because that file was added later.

## Inside `attack_comparison/`

| File | Meaning |
|---|---|
| `results.csv` / `results.json` | This CNN's scores for every method, norm, budget and random trial. |
| `comparison_config.json` | Attack settings and checkpoint/data identifiers. File hashes are fingerprints that identify which saved files were used. |
| `comparison.png` | Forecast RMSE versus input budget, with separate Linf and L2 panels. Random-noise lines average trials; reference lines show clean CNN and clean persistence. |
| `random_linf_0.5K.png`, etc. | One example of the original input, changed input, input change, true future, clean forecast and changed-input forecast. |

Names identify the method, norm and `--budget-k` value. For example,
`pgd_l2_0.5K.png` means PGD with an input RMS bound of 0.5 K; it does **not** mean
every cell changed by at most 0.5 K. On 256 cells, its L2 radius is 8 K.
`fgsm_l2_...png` is the current filename for the L2 single-step method, labelled
`one_step_l2` in the numerical results. The filename does not mean it uses FGSM's
sign direction under L2.

Example pictures show one sample; numerical scores use the whole test set. Random
pictures show the first trial, not an average picture across all trials.

Important result columns:

- `clean_rmse_k`: forecast error before changing the input.
- `changed_rmse_k`: forecast error after changing the input, against the SAME true future.
- `rmse_increase_k`: changed RMSE minus clean RMSE.
- `budget_k`: requested max-cell bound for Linf or input RMS bound for L2.
- `constraint_radius_k`: the actual norm limit, e.g. 0.5 for Linf or 8 for L2 on this grid.
- `max_cell_change_k`: largest absolute cell change across the evaluated test set.
- `rms_input_change_k`: overall size of input changes, calculated across cells and test samples.
- `max_sample_l2_k`: largest L2 change size of any individual test map.
- `persistence_rmse_k`: the no-change predictor on ORIGINAL inputs; it is not an attacked persistence score.

Limits are enforced separately for every test map. See [EXPERIMENTS.md](EXPERIMENTS.md)
for worked budget examples and interpretation of the current results.

## Suggested reading order

1. Open `clean_models.csv`: does the CNN beat predicting no change?
2. Open `summary.csv`: how do random changes and directed attacks differ?
3. Open `seed_42/attack_comparison/comparison.png`: see those comparisons plotted for one CNN.
4. Open `seed_42/test_example_0.png`, then `seed_42/attack_comparison/pgd_linf_0.5K.png`:
   distinguish the observed input, true future, prediction and changed input.
5. Use `all_results.csv` and the configuration files when you need the exact settings
   or want to compare runs systematically.

`legacy_smoke/` is a small compatibility check: attacks with zero budget reproduce
the clean synthetic score. It is not another main ERA5 experiment.

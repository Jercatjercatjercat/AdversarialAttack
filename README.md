# Weather forecasting input attacks

This workspace studies whether small changes to weather inputs can make an AI
forecast worse. We first use a small, understandable CNN, then prepare to study
the trained Otter weather model. Attacks change model inputs while keeping its
learned weights fixed.

| Location | What to read or run |
|---|---|
| [toy_weather/README.md](toy_weather/README.md) | Setup and the learning sequence |
| [toy_weather/EXPERIMENTS.md](toy_weather/EXPERIMENTS.md) | Completed experiment setup, explained terms, numerical findings and limits |
| [toy_weather/OUTPUTS.md](toy_weather/OUTPUTS.md) | Guide to models, plots and result tables |
| [OTTER_PREPARATION.md](OTTER_PREPARATION.md) | Otter's training approach, local hardware assessment and first practical milestone |
| [PaperReadings/README.md](PaperReadings/README.md) | Which suggested papers are present and how each relates to this project |
| [otter-main/README.md](otter-main/README.md) | Original Otter source instructions |

The current larger toy ERA5 results are in
`toy_weather/outputs/experiments/era5_2020_2022/`. They use the same three-layer CNN
trained three times, not three different architectures. The clean mean RMSE is
2.494 K versus 2.624 K for predicting no change. Directed input attacks cause more
error than random changes in these runs; this has not yet been demonstrated on Otter.

## Workspace cleanup, 11 October 2026

- Removed `otter-main.zip` after verifying that all 79 archived files were identical
  to the extracted source. The source itself remains in `otter-main/`.
- Removed generated Python bytecode and plotting/test caches. The existing ignore
  rules prevent these temporary files from being added to future commits.
- Renamed `toy_weather/irrelevant_testings.py` to `toy_weather/inspect_weights.py`;
  its useful learned-weight inspection remains available.
- Renamed `PaperReadings/ML_WP.pdf` to `PaperReadings/AardvarkWeather.pdf` without
  modifying its content, and moved the existing variable/metric reference sheet to
  `PaperReadings/WeatherVariablesAndMetrics.pdf`.
- Preserved the trained models, weather subsets, experiment history, annotated
  papers and `planning_form_xm272.pdf`. Earlier runs have different data or purposes
  and are not duplicates of the current experiment.

`.venv/` is the local Python environment and is not uploaded. The ignored
`.workspace-backup-20261007/` preserves the earlier workspace, including unique
documents; it is not part of the active project or GitHub checkout. GitHub contains
the meaningful source, documentation, saved small weather subsets and results.

The compatibility files `toy_weather/attack.py` and `toy_weather/attacks.py` remain
because older documented commands/imports use them. New comparisons start from
`compare_attacks.py`; the input-changing functions are in `perturbations.py`.

## Where to start reading

All five distinct recommended works are now in `PaperReadings/`; the predictability
study's separate journal-version PDF is still absent. Start with the newer attack
survey's Section II and selected methods in Section IV.A. Then read Otter's setup,
training description and Appendix B.3, followed by the predictability study's methods.
The [reading guide](PaperReadings/README.md) links each local PDF and gives a short
first-session plan. [OTTER_PREPARATION.md](OTTER_PREPARATION.md) separates evidence
from inference about predicting changes and explains Otter's repeated-forecast training.

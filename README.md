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
The [paper overview](PaperReadings/README.md) links each local PDF and explains
what each work covers and how it connects to our experiments. [OTTER_PREPARATION.md](OTTER_PREPARATION.md) separates evidence
from inference about predicting changes and explains Otter's repeated-forecast training.

The [reading log](PaperReadings/README.md#reading-log-notes-and-questions-as-we-go)
now records the first survey questions, corrections and next reading. Its
[survey notes](PaperReadings/notes/HowDeepLearningSeesTheWorld.md) include weather
definitions, proposed metrics, defense bookmarks and the paper's attack catalogue.

The main [Otter PDF](PaperReadings/Otter.pdf) is now the June 2026 preprint
(arXiv:2606.26421v1), covering deterministic and probabilistic forecasting. The
older annotated ICLR paper is kept in [the archive](PaperReadings/archive/Otter_ICLR2026.pdf).
Our [Otter reading notes](PaperReadings/notes/Otter.md) explain the architecture,
evaluation and differences between the two versions.

For probabilistic forecasting, one route starts from a deterministic model's trained
weights and continues training using CRPS, a score for possible forecasts against
the observed future. Monte Carlo dropout then runs the resulting model repeatedly
on the same weather input, with different random masks on internal features. Each
run produces a new forecast map; the collection represents possible futures. The
weights stay fixed during these runs. The paper also trains probabilistic models
from scratch. Read [the sampling clarification](PaperReadings/notes/Otter.md#follow-up-does-monte-carlo-sample-the-deterministic-output)
before Section 4.2. Replacing the PDF has not changed the saved Otter source code
or made its newer probabilistic experiments ready to run locally.

Latest reading checkpoint: [Otter Section 3.2](PaperReadings/notes/Otter.md#section-32-checkpoint-the-otter-base-architecture)
explains how stacked weather inputs become patch features and move through a Swin
U-Net. It also distinguishes dropout inside the network from changes to observed
inputs, and random sampling from CRPS training. Next read Section 4.1's controlled
design comparisons.

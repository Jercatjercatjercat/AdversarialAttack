# A foundation model for the Earth system: Aurora

Read and notes: 20 October 2026.

## Model and novelty

- **Architecture:** 3D Perceiver encoder → 3D Swin Transformer U-Net processor → 3D Perceiver decoder. The third dimension represents atmospheric vertical structure, not forecast time.
- **Forecast type:** deterministic—one forecast per input, not sampled possible futures.
- **Input stage:** processed gridded products such as ERA5 and CAMS, not raw satellite or station measurements. These products incorporate observations into estimated states. Predictions feed back as inputs for longer forecasts. Not an adversarial study.
- **Flexible representation:** the encoder accommodates different variables and pressure levels; the decoder reconstructs requested physical fields.
- **Foundation-model approach:** pretrain on over one million hours of diverse Earth-system data, then fine-tune for weather, air quality, ocean waves and cyclone tracking. The contribution combines broad training, flexible inputs and task adaptation—not simply the U-Net.

## Relevance to our project

- **Extreme-weather evaluation:** thresholded RMSE scores selected intensity levels; thresholds use local ERA5 training-year means and standard deviations. Compare clean and attacked forecasts on the same selected cases/cells.
- **Multiple variables:** define budgets in meaningful units and check relationships between variables; strong clean forecasts alone do not demonstrate robustness.
- **Repeated forecasts:** rollout fine-tuning offers a comparison with Otter; investigate how attack effects accumulate over lead time.

## Questions and bookmarks

- Do attacks harm ordinary and extreme conditions differently? Do they transfer between task-specific versions?
- **Figure 1:** architecture and pretraining; **High-resolution weather forecasting / Figure 5:** Storm Ciarán; **Methods → Verification metrics:** thresholded RMSE.

## Additional variables and attack interpretation

- **Beyond standard weather fields:** atmospheric pollutants (e.g., ozone, NO2, CO and particulate matter) and ocean-wave height, period and direction. Specialist physical systems already predict these; Aurora brings several tasks into one pretrained model family, not necessarily one checkpoint predicting everything together.
- **Extreme-weather attacks:** not studied in this paper. Borrow thresholded RMSE and storm intensity/track evaluations to test damage, concealment or displacement. Define extreme cases from reference weather and hold evaluation locations fixed across clean/attacked forecasts.

# Research so far

- Synthetic and ERA5 forecasting.
- Three-layer and five-layer CNNs.
- Five weather variables and input history.
- Input gradients with frozen weights.
- FGSM, PGD, MI-FGSM and sparse attacks.
- L0, L1, L2 and L∞ limits.
- Random-noise and no-change baselines.
- Attack steps, sizes and restarts.
- Validation selection and test evaluation.
- Weather and adversarial paper readings.
- Otter architecture and training.

# To do

- Compare attacks at matched limits — our CNN results; *How Deep Learning Sees the World*.
- Test regional and smooth attacks — *FABLE*; *Forecasting Fails*.
- Target a region/outcome: fabricate, conceal or redirect extreme-weather forecasts — *How Deep Learning Sees the World*; *Explainability and Adversarial Vulnerability*, §5.2.3; *Adversarial Observations*, §4.
- Compare longer-range and one-step 6-hour error — *Atmospheric Predictability Beyond 30 Days*.
- Inspect perturbation patterns with Otter — *Atmospheric Predictability Beyond 30 Days*.
- Test attack transfer between models — *Atmospheric Predictability Beyond 30 Days*; *Boosting Adversarial Attacks with Momentum*.
- Define physical-plausibility criteria — *Atmospheric Predictability Beyond 30 Days*; *Forecasting Fails*.
- Investigate IE/NE alongside MSE/ACC — *Atmospheric Predictability Beyond 30 Days*.
- Attack probabilistic forecasts using CRPS; separate accuracy and diversity terms — *Otter Weather*.

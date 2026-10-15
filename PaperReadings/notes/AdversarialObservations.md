# Adversarial Observations in Weather Forecasting

[Source](https://arxiv.org/abs/2504.15942)

Checked against the local April 2025 v1 PDF on 15 October 2026. This is an attack study on GenCast, a probabilistic forecaster that generates different possible future weather maps using diffusion. Diffusion generates a map by repeatedly removing noise; autoregression feeds a predicted state back in to forecast farther ahead.

## What is changed, and how

The experiments change the two initial, assimilated weather grids from ERA5. The threat model motivates these changes through compromised observations, especially satellites. Section 3.1 nevertheless explicitly gives the attacker access to the assimilated grid. A raw satellite radiance or station reading is not the same object as an ERA5 grid entry, and the paper should not be presented as a demonstrated attack through a complete operational observation-to-assimilation pipeline.

The attack differentiates through a cheaper approximation of GenCast's sampling and repeated forecasting. Algorithm 1 approximates diffusion using a few sampled noise levels; Algorithm 2 updates input perturbations using momentum and a changing step size. Projection removes each variable's mean perturbation and bounds its standard deviation. This differs from limiting every cell separately with Linf. The experiments use 50 optimisation steps and evaluate effects on five-member GenCast ensembles.

The objectives include fabricating extreme forecasts and concealing genuine events. An attack can therefore push a selected regional wind, rainfall or temperature outcome toward a chosen result, rather than simply increasing whole-map RMSE.

## Connection to the CNN experiments

Our CNN uses a processed ERA5 input too, but produces a deterministic six-hour forecast. Regional objectives, changes spread across several future steps, and per-variable disturbance statistics are possible comparisons. Diffusion approximation would only become relevant after moving to a probabilistic model with that sampling process.

## Questions to resolve while reading

- Which experiments restrict changed grid entries, and how is the reported measurement fraction counted?
- What evidence connects the allowed grid changes to changes achievable through one real observing platform?
- Do attacks remain effective across fresh random forecasts rather than only the random draws used to construct them?
- How much does the cheap approximation differ from the full GenCast forecast during attack optimisation?

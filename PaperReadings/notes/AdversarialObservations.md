# Adversarial Observations in Weather Forecasting

Read and notes: 16 October 2026; April 2025 v1.

## Setting and threat model

- **Probabilistic GenCast:** autoregressive diffusion forecasting. Globally shared observations and conflicting institutional interests motivate security concerns; satellites supply most assimilation data.
- **Actual attack input:** both initial assimilated ERA5 states, Xt−1 and Xt. Satellite compromise motivates the threat model, but the experiments do not demonstrate tampering through an operational satellite-to-assimilation pipeline.
- **White-box:** weights and gradients are accessible; the authors argue this is reasonable for openly released models.
- **Small changes:** zero-mean perturbations with per-variable standard deviation ≤ ε. This is not a per-cell ±ε limit. Experiments set ε relative to an estimated background-error standard deviation (up to 0.25%); the one-satellite interpretation is an approximation, not a demonstrated capability.

## Diffusion and gradient approximation — Sections 2–3

- Given two weather states, sample a noisy future candidate and denoise it through a noise-level schedule, conditioned on those states. Training learns denoising at sampled noise levels; the schedule is not recomputed from the two states at each step.
- Differentiating through full diffusion and autoregressive sampling is expensive. The attack uses a short stochastic approximation: sample multiple noise levels and denoise through them, rather than relying on a single sampled level.
- Backpropagate through this approximation to change both initial states, using momentum, scheduled steps and projection. Model weights stay fixed. The setup uses 50 attack updates, two approximation steps, five ensemble members and a two-day target horizon (four 12-hour forecast steps). Results span geographically varied targets.

## Main takeaway: targeted extreme-weather forecasts

The objective combines **S**, a region selector, with **V**, a variable-specific objective. It need not be distance to a complete target map.

| Goal | Forecast objective |
|---|---|
| Fabricate | Increase regional wind, temperature or precipitation toward extreme values. |
| Conceal | Reduce predicted precipitation or temperature during a real extreme event. |
| Redirect | Reduce wind at the original landfall and increase it at a chosen alternative location. |

These change predicted weather, not the actual event. They motivate targeted regional attacks beyond whole-map RMSE.

## Detection — Section 5

- Idealised model: independent Gaussian background and attack noise; variances add: σ²total = σ²background + ε².
- A chi-square variance test checks for an increase, assuming known, constant background variance. Small increases can be difficult to detect with finite samples.
- This tests one statistical detector under stated assumptions, not undetectability against all quality-control methods; optimised perturbations need not truly be independent Gaussian noise.

## Bookmarks

- **Section 3.2 / Algorithms 1–2:** attacking autoregressive diffusion models.
- **Sections 4.2–4.3:** targeted fabrication, concealment and redirected storm forecasts.
- **Section 5:** statistical detection assumptions.

# Forecasting Fails: Unveiling Evasion Attacks in Weather Prediction Models

[Source](https://arxiv.org/abs/2512.08832)

Checked against the local December 2025 v1 PDF on 15 October 2026. The experiments attack FourCastNet using ERA5 initial weather fields.

## What is changed, and how

The attack changes processed gridded weather inputs, not raw observing-platform messages. FourCastNet is a deterministic forecasting model. The paper also averages forecasts produced from perturbed initial conditions; that ensemble experiment does not make the underlying network a probabilistic diffusion model.

Weather Adaptive Adversarial Perturbation Optimization (WAAPO) changes initial weather fields to influence later forecasts. Its objective combines the desired forecast effect with penalties. Its constraints include which variable channels may change, a geographical mask, per-variable amplitude limits and spatial smoothness. Gradient-based updates are followed by constraint enforcement; Algorithm 1 gives the procedure. Channel sparsity means changing only selected variables, which is different from our L0 constraint that counts changed cells.

The experiments include targeted temperature changes and examine how changes propagate through forecasts. Smoothness limits abrupt differences between neighbouring perturbations. Smoothness and small amplitude are useful checks, but do not establish that winds, temperature, humidity and pressure jointly satisfy atmospheric physics.

## Connection to the CNN experiments

This is a close match for adding a geographical mask and comparing rough changes against smooth changes at explicitly reported sizes. A regional or later-forecast objective would extend our current whole-map, one-step temperature-error experiments.

## Questions to resolve while reading

- How are the amplitude and smoothness limits chosen in physical units?
- Which quantities are hard constraints and which are penalties that can be traded against attack success?
- Does the attack alter just the initial state or also later inputs?
- How much forecast damage remains when maximum cell change and RMS change are matched to random noise?

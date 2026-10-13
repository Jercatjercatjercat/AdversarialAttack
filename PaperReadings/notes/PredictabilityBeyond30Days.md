# Atmospheric Predictability Beyond 30 Days: reading notes

## What does one epoch mean here?

Checked on 13 October 2026 against the annotated local preprint's Optimization
section, algorithm steps 1–5. Here, an epoch is one input-optimisation iteration,
not one pass through a training dataset. GraphCast's trained weights remain fixed.

One iteration starts from the current pair of input weather states (t and t−6h),
rolls the model forward over the current optimisation window, compares the forecasts
with ERA5 at every six-hour step, differentiates the loss with respect to both
input states, and updates those inputs using Adam to reduce error.

For a two-day window, this requires eight six-hour forecast steps before one
input update. Repeating 100 epochs means repeating that optimisation cycle 100
times; it does not mean forecasting 100 consecutive days or retraining GraphCast.

The window begins at two days and expands by three days at a time. Each stage
starts from the inputs optimised at the preceding stage. The paper uses 100
iterations for windows shorter than 10 days, 50 between 10 and 20 days, and 25
for windows of 20 days or more.

Contrast with our CNN training: an epoch is one pass through all training pairs,
usually containing many batches and weight updates. Here the authors optimise
one case's initial weather, rather than train weights on many cases. Both use
gradients, but the object being changed is different. Our adversarial attacks
also change inputs with fixed weights, aiming to increase rather than reduce error.

Source: [annotated local preprint](../PredictabilityBeyond30Days.pdf), Optimization
section; [paper reference](https://arxiv.org/abs/2504.20238).

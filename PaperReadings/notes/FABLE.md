# FABLE: A Localized, Targeted Adversarial Attack on Weather Forecasting Models

[Source](https://arxiv.org/abs/2505.12167)

Checked against the local April 2026 v2 PDF on 15 October 2026. The original preprint appeared in 2025. The saved version includes Jack Gunn in its author list, in addition to the authors in Rich's original citation.

## What is changed, and how

FABLE attacks numerical forecasts from CLCRN and an adapted FourCastNet. The experiments use gridded NLDAS and ERA5 data. These are processed atmospheric data products, not direct edits to raw satellite signals or individual station messages. In the ERA5 experiments, FourCastNet is adapted to use a history of one target variable, including temperature or incident solar radiation. This is not automatically an attack on the unchanged full operational FourCastNet configuration.

The reported forecasting task is deterministic: a given input produces a forecast array, rather than a sampled distribution of futures. FABLE first specifies a desired forecast change in a selected place and time. It represents the input history using a three-dimensional wavelet transform, which separates patterns across time and space into components of different scales. It adjusts those components with Adam, reconstructs the input, and clips input changes to an Linf limit. Its objective balances reaching the desired forecast with keeping the transformed input close to the original. Section IV and Algorithm 1 describe this procedure.

Locality of the desired forecast change and locality of the allowed input changes are separate choices. A targeted output region does not, by itself, prove that only that same input region is changed. Likewise, closeness and pattern-based scores are evidence about particular properties, not proof that all atmospheric relationships remain valid.

## Connection to the CNN experiments

A first comparison could target a chosen temperature region, then compare ordinary PGD with changes represented at different spatial scales. The simplest implementation would start with our deterministic CNN and explicitly state where changes are permitted. Reproducing FABLE itself would also require its time-history representation and full objective.

## Questions to resolve while reading

- Which locations may change in the input, and which locations are supposed to change in the output?
- How do the wavelet penalties affect small-scale versus broad changes?
- Are attack methods compared at matched input-change limits and computational effort?
- Which conclusions depend on the adapted single-variable FourCastNet setup?

# Otter Weather: reading notes

Read and notes: 13 and 16 October 2026. Current reading: June 2026 preprint.

## Purpose and architecture

- Reduce the GPU cost of training competitive weather models, making training and adaptation accessible to smaller research groups.
- A 2D Swin Transformer in a U-Net layout; shifted attention windows, RoPE position information and skip connections. Predicts a six-hour correction added to the latest weather state.
- Processed ERA5 inputs; training years 1979–2019, evaluation year 2020, at 1.5° resolution. The paper does not test adversarial robustness.

## Inputs

- 6 atmospheric variables × 13 pressure levels + 4 surface variables + 5 static features = **87 channels**. Only 82 are changing weather fields.
- Atmospheric: temperature, geopotential, specific humidity, vertical velocity and U/V winds. Surface: T2m, mean sea-level pressure and U/V winds at 10 metres.
- With four context states, conceptual shape: **[B, 4, 121, 240, 87]**. B is batch size; the four times are t−18h, t−12h, t−6h and t, including the current state.
- Time and channel dimensions are combined for the 2D network: **[B, 121, 240, 348]** before added time embeddings. The 121×240 grid has 1.5° spacing. Four context frames are specified in the saved base configuration; Appendix B describes general context length L.
- Variable-specific normalization uses training statistics. Static features are inputs, not weather variables to forecast or perturb by default.

## Training routes

| Route | Initial training | Rollout fine-tuning |
|---|---|---|
| Deterministic | One-step, latitude/pressure-weighted RMSE; 20 epochs | RMSE over repeated forecasts; horizon grows from 24h (4 steps) to 72h (12 steps). |
| Probabilistic from deterministic weights | One-step CRPS fine-tuning for 5 epochs; MC dropout supplies different forecast samples | CRPS over repeated forecasts, using the same horizon curriculum. |
| Probabilistic from scratch | One-step CRPS training for 15 epochs | CRPS rollout fine-tuning. |

Muon updates the main matrix weights. Rollout training addresses errors caused by feeding predictions back as inputs. During training, model weights change; this differs from *Beyond 30 Days*, which changes initial conditions with weights fixed.

## Probabilistic output and CRPS

MC dropout produces different complete forecast maps from the same input by changing internal dropout masks. CRPS trains the sampled predictions jointly; it does not itself create randomness. The probabilistic ensemble mean need not equal the original deterministic forecast.

| Fair CRPS term | Meaning |
|---|---|
| Mean absolute error of members against the true future | Encourages accurate predictions. |
| Subtracted, scaled pairwise distance between members | Rewards spread, balanced against the first term; unlimited diversity is not the goal. |

Deterministic evaluation uses RMSE; probabilistic evaluation uses CRPS. A probabilistic attack could increase total CRPS, with both terms reported to distinguish accuracy damage from changes in spread. Fresh dropout samples are needed to check that an attack works beyond its optimisation draws.

## Follow-up: output representation, training relationship and paper versions

- The archived ICLR paper covers deterministic Otter; the newer preprint adds probabilistic variants and revised design choices. Replacing the PDF does not update local code or weights.
- Forecasting is deterministic with dropout disabled; probabilistic inference samples maps with dropout active. Dropout changes internal features, not raw weather observations.

## Bookmarks

- Section 3.2 / Figure 2: architecture; Section 4.2 / Equation 1: probabilistic forecasts and CRPS.
- Appendix B.1: input representation; B.3.5: training routes and rollout curriculum.
- First practical bridge: load weights, verify clean forecasts and input gradients, then test attacks with explicit physical-unit limits.

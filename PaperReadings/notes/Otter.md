# Otter Weather: reading notes

Read and notes: 13 and 16 October 2026; newer June 2026 paper.

- **Purpose:** lower GPU training costs for wider access.
- **Architecture:** Swin Transformer in a U-Net layout; predicts a six-hour correction added to the latest state.
- **Data:** processed ERA5; train 1979–2019, evaluate 2020.
- **Input:** [B, 4, 121, 240, 87]; batch × four states (t−18h through t) × 1.5° grid × channels. Four states follow the saved base configuration.
- **Channels:** 6 atmospheric variables × 13 levels + 4 surface variables + 5 static features = 87. Only 82 are changing weather fields.

## Training

- **Deterministic:** one-step weighted RMSE → rollout RMSE.
- **Probabilistic:** deterministic weights → one-step CRPS with MC dropout → rollout CRPS. CRPS training from scratch is also tested.
- **Rollouts:** gradually extend from 24h to 72h to address accumulated forecast errors.
- **MC dropout:** different internal masks generate possible future maps; CRPS trains their accuracy and spread.

## Attack idea

- Increase CRPS; inspect the accuracy term and subtracted diversity term separately. More spread alone need not worsen CRPS. Test with fresh dropout samples.

## Bookmarks

- Section 3.2 / Figure 2: architecture; 4.2 / Equation 1: CRPS; Appendix B: inputs and training.

## Follow-up: output representation, training relationship and paper versions

- Archived ICLR paper: deterministic. Newer paper: both forecast types. Local code is not automatically updated with the paper.

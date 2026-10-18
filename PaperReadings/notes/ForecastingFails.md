# Forecasting Fails: Unveiling Evasion Attacks in Weather Prediction Models

Read and notes: 18 October 2026; December 2025 v1.

## Setting and goal

- White-box WAAPO attack on deterministic FourCastNet using processed, gridded ERA5 inputs. Figure 1 gives the high-level attack pipeline.
- **Objective:** change initial conditions to push forecasts toward a chosen extreme-weather outcome. **Constraints:** restrict affected channels/region and encourage realistic magnitudes and smoothness.
- The tested ensemble mean stays close to the control under small Gaussian input noise; this does not establish robustness to every random disturbance or individual ensemble member. Crafted changes cause larger effects.

## Method and novelty

- Solve for the initial disturbance δ with model weights fixed: minimise target-forecast squared error plus **L∞ and total-variation penalties**.
- **Three-term loss:** target matching + magnitude penalty + spatial roughness penalty. In the equations, the latter two apply to forecast states along the trajectory, not simply to δ. They are soft penalties, not hard guarantees.
- **Hard restrictions:** spatial mask and selected-channel mask after each update. Channel sparsity limits variables, not the number of changed cells.
- Implementation uses Adam, gradient clipping and learning-rate scheduling. Smoothness encourages gradual transitions; it does not prove compliance with atmospheric physics or undetectability.
- Localised patches induce changes, but the paper notes difficulty matching the target temperature profile exactly.

## Target choice and project connection

- An example makes a **24-hour forecast resemble a state from 120 hours (five days)**. This differs from attacking the error of a five-day forecast.
- Compare input masks, selected variables and trajectory-smoothness penalties on our CNN. Test future-state-derived targets separately from longer forecast horizons.

## Bookmarks

- **Figure 1:** overview; **Algorithm 1:** loss, gradient updates and masks.
- **Experimental Discussion:** five-day target, optimisation settings and limits of patch-based targeting.

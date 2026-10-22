# FABLE: A Localized, Targeted Adversarial Attack on Weather Forecasting Models

Read and notes: 17–18 October 2026; April 2026 v2.

## Setting and goal

- White-box attacks on deterministic CLCRN and adapted FourCastNet using processed, gridded NLDAS/ERA5 data.
- **Localized target:** make forecasts match desired values at selected places/times while preserving forecasts elsewhere. Input changes need not be confined to the target region.
- Relevant to regional attacks and pattern-preserving disturbances; the reported realism measures do not establish complete physical consistency.

- **Key novelty:** replace a direct input-change penalty, **target-forecast RMSE + λ ‖X − X′‖2**, with a weighted wavelet-change penalty, **target-forecast RMSE + λ Σf ωf ‖Cf − C′f‖2**. This lets different spatial/temporal frequency bands carry different penalties.

## Evaluation — Section III

| Criterion | Measure |
|---|---|
| Faithfulness | In-AE: error against the desired forecast inside the target; Out-AE: unwanted change outside it. |
| Closeness | L1 distance between original and attacked inputs. |
| Geospatio-temporal realisticness | Changes in spatial autocorrelation (Moran's I) and temporal autocorrelation. |

## Wavelet attack — Section IV

- **Core idea:** input history → 3D Haar wavelet coefficients → optimise coefficients with Adam → inverse transform → clip input changes → forecast. Gradients pass through reconstruction and the forecaster.
- L/H represent low/high-frequency components along time and two spatial axes; HHH captures fine-scale variation in all three.
- Objective: **target-forecast RMSE + λ Σf ωf ‖Cf − C′f‖2**, subject to **‖X′ − X‖∞ ≤ ε**. The penalty measures coefficient CHANGES, not coefficients themselves.
- The target forecast keeps original predictions outside the desired region. The RMSE term therefore encourages both the intended local effect and preservation elsewhere.
- Band weights control where changes are cheaper. Lower penalties on high-frequency bands allow more change there; exact weights are experimental choices, not a universal hierarchy. Figure 6 tests alternatives, including excluding LLL from the penalty in configurations 1–4.
- Wavelets organise spatial/temporal changes; clipping enforces the amplitude bound. Wavelet penalties do not themselves prohibit changes outside a geographical region.

## Small wavelet example

- 1D Haar: [2, 4] → L = 6/√2, H = −2/√2; inverse transform restores [2, 4]. FABLE applies the idea across time and two spatial axes.
- LLL captures broad structure; HHH captures fine differences along all three axes. Larger penalty weights discourage changes in a band; smaller weights permit them. This encourages closeness and pattern preservation, rather than proving physical consistency.

## Project plan and bookmarks

- First minimise target-forecast MSE/RMSE directly over input changes under an explicit norm bound; then compare with FABLE at matched limits.
- Start with our CNN; extend to Otter after checking input histories, weights and gradients.
- **Section III:** goal, closeness, faithfulness, Moran's I and temporal realism. **Section IV / Algorithm 1:** replication; **Figure 6:** band-weight choices.

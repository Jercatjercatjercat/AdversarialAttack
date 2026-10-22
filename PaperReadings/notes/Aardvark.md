# End-to-end data-driven weather prediction: Aardvark

Read and notes: 16 October 2026.

## Main idea

- Replace conventional assimilation, forecasting and local post-processing with learned modules—not skip state estimation. Competitive results against specified baselines do not show universal superiority over NWP.
- **Deterministic pipeline:** observations → encoder → initial gridded state → processor → 24-hour forecast, repeated autoregressively → decoder → local station predictions.
- Modular ERA5 pretraining is followed by observational training and joint fine-tuning for selected downstream tasks.
- **Architecture:** ViT encoder and processor; lightweight convolutional decoder.

## Inputs and outputs

- **Inputs:** four main satellite-data sources, plus stations, ships and radiosondes; temporal and topographic features. Remote-sensing inputs are pre-gridded, so “raw observations” does not mean no preprocessing.
- **Global outputs:** U/V winds, humidity, geopotential and temperature at 200/500/700/850 hPa; 10-m U/V winds, T2m and mean sea-level pressure.
- **Local outputs:** station T2m and 10-m wind speed.

## Attack questions

- **State reconstruction:** can allowed observation changes make the encoder reproduce a chosen adversarial weather state? Some states may be unreachable.
- **End-to-end attack:** which observation changes most damage forecasts through the encoder → processor → decoder, with all weights fixed?
- The modules already form a cascade and support joint training; an attack still needs verified input gradients and realistic observation constraints.

## Later research directions

- **Two-stage attack:** attack the processor's initial state, then search for bounded observation changes whose encoded state approximates that target. The encoder is not generally invertible; many observation sets may match, or the target may be unreachable.
- **Direct joint attack:** optimise forecast error through the frozen encoder and processor, optionally the station decoder; compare with the two-stage method.
- **Restricted observations:** select a few stations, one instrument type or one platform; measure required changes relative to uncertainty, not raw gradient size alone.
- **Availability and persistence:** test missing/delayed reports and consistent biases across observation times, where supported by the input pipeline.
- **Stage comparison:** compare direct-state attacks with observation-level attacks to see which disturbances the encoder suppresses or amplifies. An encoded-state change that is small on average may still matter to the forecast.
- **Presentation placement:** keep these as later discussion questions, after the basic state-input attacks. These are proposed experiments, not results from the paper.

## Bookmarks

- Figure 1: pipeline and inputs; Extended Data Tables 1–2: variables and sources.
- Methods: module training and joint fine-tuning. Potential bridge from our ERA5-state attacks to observation-level attacks.

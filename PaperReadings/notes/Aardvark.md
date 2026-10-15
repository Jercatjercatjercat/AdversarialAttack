# End-to-end data-driven weather prediction

[Source](https://www.nature.com/articles/s41586-025-08897-0)

Aardvark describes a system that starts from observations, estimates a weather state, forecasts its evolution and produces local station forecasts. The reported setup produces deterministic forecasts. It is a forecasting paper, not an attack study.

Its observation inputs are important for locating an attack within the wider pipeline. Raw measurements, an estimated gridded state and a predicted future are different objects. Our current CNN experiment changes the estimated ERA5 state supplied to a forecaster; it does not change raw station readings or satellite measurements.

## Questions to develop while reading

- Which observation types are used, and what processing occurs before they enter the network?
- Would changing an observation have the same effect as changing a reconstructed weather-state cell?
- Which stages could reduce or amplify a disturbance?

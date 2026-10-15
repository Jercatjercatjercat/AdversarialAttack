# A foundation model for the Earth system

[Source](https://www.nature.com/articles/s41586-025-09005-y)

Aurora is a forecasting foundation model: broad initial training supports later adaptation to several Earth-system tasks. The reported weather setup uses processed gridded atmospheric states and deterministic forecasts. The paper is not an adversarial attack study.

Its multi-variable inputs, extreme-weather examples and verification methods connect to our CNN project. Thresholded RMSE can assess selected unusually intense conditions rather than averaging every location together. A strong clean forecast alone does not demonstrate resistance to attacks.

## Questions to develop while reading

- How are extreme cases and thresholds defined without using the test set to choose them?
- Which forecast variables and physical-unit errors matter most for a selected event?
- What is lost when a global multi-level state is reduced to our small crop?

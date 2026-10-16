# How Deep Learning Sees the World: reading notes

Read and first notes: 12 October 2026. Shortened: 15 October 2026.

- **Model setting:** mainly deterministic image classifiers; the attacks primarily target deterministic predictions. The attack procedure itself can still use randomness.
- **Input stage:** classifiers typically receive processed image tensors (e.g., resized and normalized), rather than raw sensor signals. Some surveyed attacks also operate through physical objects or patches.

The survey mainly concerns image classification: a perturbed image can make a network misclassify an object that a human still identifies correctly.

- **Weather equivalent:** change initial conditions within stated physical-unit limits to increase forecast RMSE or produce a chosen regional outcome. A norm limit alone does not ensure physical plausibility.

## Keywords and limits

- **Untargeted:** increase forecast error, such as RMSE.
- **Targeted:** seek a specified forecast change, such as +2°C at a chosen location.
- **White-box / black-box:** access to model gradients / attacks using queries or transfer without those gradients.
- **L0 / L1 / L2 / L∞:** changed-entry count / total absolute change / square root of summed squared changes / largest cell change.
- **Detectability:** sparse L0 changes or concentrated L2 changes can stand out, but visibility also depends on budget, spatial pattern and data scale; no norm gives a universal ranking.

## White-box methods — Section IV.A

- **Main weather candidates:** FGSM (one step), BIM (repeated clipped sign steps), PGD (repeated steps with projection). Forecast loss replaces classification loss.
- **Other surveyed methods:** L-BFGS, JSM/JSMA, DeepFool, ILCM, C&W, GAAS, AdvGAN, MI-FGSM, SIAA, SmoothFool, AdvCam, FIA and MGAA.
- Classifier results motivate experiments; they do not establish weather effectiveness. Class-specific objectives may need redesigning.

## Bookmarks

- **Black-box, Section IV.C:** transfer and query-based attacks, including NES, FGBB and Square Attack.
- **Defenses, Section V and parts of VI:** adversarial training, detection and purification; evaluate clean performance and attacks that account for the defense.
- **CNN versus ViT, Section VI:** architecture, patch and attention effects; SAGA, PNA/PatchOut, ATA and Patch-fool. No blanket robustness ranking.
- **Metrics, Section VIII:** possible weather adaptations include attack-success rate and the smallest found budget causing a specified failure; retain clean/attacked RMSE and disturbance sizes.

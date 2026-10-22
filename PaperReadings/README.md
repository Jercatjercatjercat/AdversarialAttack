# Papers behind the weather-input attack project

This folder brings together two sides of the project: how AI models forecast
weather, and how deliberate changes to their inputs can affect those forecasts.
The toy experiments connect these ideas on a small temperature grid. The longer-term
question is whether similar input changes affect a larger model such as Otter,
including forecasts with several variables, repeated forecast steps and extreme events.

The collection contains eleven distinct research works: the five originally recommended,
Otter, the momentum-attack paper and four weather-specific works suggested by Rich.
There are twelve research PDFs because an earlier Otter paper is also archived,
plus a separate weather-variable reference sheet. The six original links represent
five works because the predictability paper has both a preprint and a journal
publication. The local PDFs include existing personal annotations. Detailed
questions and corrections are stored in `notes/` and linked below.

## How Deep Learning Sees the World: attacks and defenses

[Local PDF](<How Deep Learning Sees the World: A Survey on Adversarial Attacks & Defenses.pdf>) ·
[2023 preprint](https://arxiv.org/abs/2305.10862) ·
[Our reading notes](notes/HowDeepLearningSeesTheWorld.md)

This survey explains how changes to an input can cause a neural network to make
an unwanted prediction. It covers attacks with different levels of model access,
ways to restrict the size of an input change, defenses, and attacks on Vision
Transformers—networks that process spatial patches using attention. Much of the
survey concerns images and classification, where the output is a label such as
“cat” or “dog.” Our weather model instead predicts numerical temperatures, so an
attack is judged by its effect on forecast error or a specified forecast outcome.

Section II provides the background vocabulary. Section IV describes attack
methods, including gradient-based methods related to our FGSM and PGD experiments.
Section V covers defenses, Section VI discusses attention and patch-based models,
and Section VIII addresses evaluation. These parts connect respectively to our
input-change limits, attack comparisons, eventual Transformer experiments and
questions about how to measure success.

The connection to Otter does not depend on an attack being designed originally
for a CNN. FGSM and PGD use gradients and can also be applied to a differentiable
Transformer. A separate question is whether an input change chosen for one model
also affects another model; this is called **transfer**. The survey provides
possible methods and evaluation questions, rather than evidence that a particular
weather model is vulnerable.

Our notes also record important corrections to the image analogy. A weather
attack does not have to be visually invisible, and an input-change limit alone
does not guarantee physical plausibility. L0 counts changed entries, L2 limits
the combined size of changes, and Linf limits each entry separately. Their effects
need interpretation in weather units as well as numerical scores. The defense
material remains useful for later work, when the attack itself is well understood.

## Boosting Adversarial Attacks with Momentum: remembering earlier directions

[Local PDF](<Boosting Adversarial Attacks with Momentum.pdf>) ·
[2018 preprint, v3](https://arxiv.org/abs/1710.06081) ·
[Our reading notes](notes/BoostingAdversarialAttacksWithMomentum.md)

Dong and colleagues introduce MI-FGSM, Momentum Iterative Fast Gradient Sign
Method. Instead of using only the latest input gradient, the attack keeps a running
memory of normalised gradient directions and takes steps using its sign. Section
3.1 and Algorithm 1 describe this update; Section 3.3 extends the idea to L2 and
targeted attacks. The paper examines image classification and reports improved
transfer to other classifiers, not experiments on weather forecasts.

This connects to our proposed comparison of FGSM, PGD and momentum-based attacks
on the frozen CNN. Forecast loss would replace the classification objective,
with the same weather examples and physical budgets across methods. Momentum's
gradient normalisation uses an L1 magnitude, but this is not an L1 restriction on
the final disturbance. The [toy-weather attack sweep](../toy_weather/ATTACK_EXPERIMENTS.md) now tests
these ideas on a small validation subset. It does not establish benefits for
transfer to another model.

The accompanying notes also distinguish momentum from a sparse attack, which
changes only a few cells. Gradient thresholding or top-k selection can choose
locations; an iterative method must constrain the accumulated change, not merely
choose k new locations on each step. L0 count limits need a separate cell-amplitude
cap. These sparse methods are our educational adaptations, now implemented in the
toy-weather code; they are not methods evaluated by this momentum paper.

## Otter Weather: the forecasting model we may eventually attack

[Main local PDF](<Otter Weather: Skillful and Computationally Efficient Medium-Range Weather Forecasting.pdf>) ·
[June 2026 preprint](https://arxiv.org/abs/2606.26421) ·
[Our reading notes](notes/Otter.md) ·
[Local preparation and code discussion](../OTTER_PREPARATION.md)

The main PDF is **Otter Weather: Skillful and Computationally Efficient
Medium-Range Weather Forecasting**, arXiv:2606.26421v1. It describes a family of
models intended to make strong weather forecasts with a comparatively small
training budget. Otter receives recent global weather states containing several
variables and atmospheric levels, then predicts the next state. Repeating this
operation produces forecasts farther into the future.

Section 3.1 describes the forecasting task and evaluation. Section 3.2 and
Figure 2 explain the architecture: weather information is combined into channels,
converted into patch features, and processed by a Swin Transformer in a U-Net
layout. The network reduces spatial resolution to process broader patterns and
then restores it, using earlier features to retain detail. Section 4.1 compares
design choices and their forecast-skill and computational costs. These parts
connect our temperature-only CNN to a larger system while preserving the same
basic input-to-forecast relationship.

Section 4.2 extends Otter to probabilistic forecasting, where the output is a
collection of possible future maps. In one training route, a deterministic model
is trained further using CRPS, a score that assesses possible forecasts against
the observed future. Monte Carlo dropout creates different maps by running the
model repeatedly with different random masks on internal features. Randomness
creates variation; CRPS training helps make that variation useful. The paper
also describes probabilistic training from scratch and combinations of trained
checkpoints. A single finished deterministic map is not itself a distribution
from which these possible futures can be sampled.

For our project, the deterministic model offers the more direct first bridge:
fixed trained weights, a weather input, a forecast loss and a gradient with respect
to that input. Probabilistic forecasting adds a further question about how attacks
should be evaluated when repeated predictions differ. Its random internal dropout
is distinct from deliberately changing the observed input in an attack.

The main PDF was replaced on 13 October 2026. The earlier **Otter Weather: Highly
Skillful Medium-Range Weather Forecasting on a Single GPU**, the deterministic
ICLR 2026 paper, is preserved with its annotations in
[archive/Otter Weather: Highly Skillful Medium-Range Weather Forecasting on a Single GPU.pdf](<archive/Otter Weather: Highly Skillful Medium-Range Weather Forecasting on a Single GPU.pdf>). Our saved `otter-main/`
source code has not been updated merely by replacing the paper. The
[version comparison](notes/Otter.md#follow-up-output-representation-training-relationship-and-paper-versions)
separates the papers' claims from what the local code currently supports.

## Atmospheric Predictability Beyond 30 Days: changing the starting weather

[Annotated local preprint](<Atmospheric Predictability Beyond 30 Days with Machine Learning.pdf>) ·
[Preprint reference](https://arxiv.org/abs/2504.20238) ·
[Journal publication](https://journals.ametsoc.org/view/journals/aies/aop/AIES-D-26-0009.1/AIES-D-26-0009.1.xml) ·
[Our reading notes](notes/PredictabilityBeyond30Days.md)

This paper is the closest methodological connection to our input-gradient work.
The authors keep a trained GraphCast model fixed and adjust its starting weather
states to reduce future forecast error. Forecasts are fed back into the model
repeatedly, and the error is differentiated through those forecast steps back to
the initial inputs. The optimisation window begins short and gradually expands,
using the previously improved inputs as the starting point for each longer window.
The study also tests whether inputs chosen for GraphCast help Pangu-Weather.

The Optimization subsection explains what is changed and how the gradient is
used. Here, one “epoch” means one forecast, loss, gradient and input-update cycle.
It does not mean a pass through a dataset to train GraphCast's weights. For a
two-day window, eight six-hour forecast steps occur before one input update.

Our attacks share the idea of differentiating with respect to inputs through a
fixed forecaster, but aim to increase error. The paper's input optimisation is
not automatically the same procedure as our constrained FGSM or PGD attacks. It
motivates comparisons between error-reducing and error-increasing changes, effects
across longer forecasts, and transfer between models. Because known future weather
is used during optimisation, the improvements are retrospective; they do not
show an operational method that knows the future in advance.

The local file is the annotated arXiv v2 preprint from May/June 2026. A separate
journal PDF is not saved. The two online references describe the same study,
although version differences matter when quoting particular results.

## Aurora: several variables, different tasks and extreme weather

[Local PDF](<A foundation model for the Earth system.pdf>) ·
[Nature publication](https://www.nature.com/articles/s41586-025-09005-y)

**A foundation model for the Earth system** introduces Aurora, a model trained
broadly on diverse Earth-system data and then adapted to particular tasks. This
broad initial training followed by task-specific adaptation is what “foundation
model” means here. Its applications include weather, tropical cyclone tracks,
air quality and ocean waves. The model uses an encoder, a processor and a decoder
to accommodate different variables and atmospheric levels.

The section **“Aurora: an Earth system foundation model”** and Figure 1 describe
this shared model and its task-specific adaptations. **“High-resolution weather
forecasting”** and Figure 5 combine broad forecast comparisons with the Storm
Ciarán example. **“Tropical cyclone tracking”** and Figure 4 show another way to
judge a forecast: the error in a storm's predicted position. Position error,
storm intensity error and temperature-map error answer different questions.

The **Methods → “Verification metrics”** subsection is particularly relevant to
our proposed extreme-weather experiments. Alongside general forecast scores, it
describes thresholded RMSE: an error score calculated at grid locations meeting
specified intensity thresholds. Those thresholds use ERA5 training-year means
and standard deviations calculated separately at each location. This gives us
an example of assessing unusually severe conditions instead of allowing ordinary
conditions to dominate a whole-map average. A single storm case remains an
illustration, rather than proof of performance on every extreme event.

For our project, Aurora contributes ideas for multi-variable inputs and separate
ordinary-weather and extreme-weather evaluations. A fair clean-versus-attacked
comparison would use the same selected cases and locations. Multiple variables
also need meaningful limits in their own units; Kelvin and metres per second
cannot simply be combined into an unexplained physical budget. Aurora's paper
reports forecast performance, not adversarial robustness, so its results do not
establish resistance to our attacks.

The Methods subsections on pretraining, short-lead-time fine-tuning and rollout
fine-tuning also provide a comparison with Otter's training. Aurora's memory-saving
rollout training does not differentiate through every step in the same way as
the initial-condition optimisation in the predictability study.

## Aardvark: the wider path from observations to forecasts

[Local PDF](<End-to-end data-driven weather prediction.pdf>) ·
[Nature publication](https://www.nature.com/articles/s41586-025-08897-0)

**End-to-end data-driven weather prediction** describes Aardvark, a pipeline that
starts with observations, estimates the current atmospheric state, forecasts
future global states and produces local station forecasts. Its model overview
and Figure 1 show these stages: an encoder estimates the state, a processor
forecasts its evolution, and a decoder produces local outputs.

This matters because an attack can enter the forecasting process at different
places. Our ERA5 experiment changes an already reconstructed weather state at
the forecaster's input. Changing a raw station or satellite observation before
state estimation would be a different experiment. Aardvark makes that distinction
visible and suggests later questions about whether state estimation reduces or
amplifies an observation change, and whether effects at particular stations differ
from the average effect across a map. It describes the wider forecasting pipeline,
rather than supplying an FGSM or PGD implementation. The local PDF was previously
named `ML_WP.pdf`.

## Adversarial Attacks and Defences: the foundational vocabulary

[Local PDF](<Adversarial Attacks and Defences: A Survey.pdf>) ·
[2018 preprint](https://arxiv.org/abs/1810.00069)

This earlier survey organises attacks by what the attacker knows, what can be
changed and what outcome is intended. A white-box attacker can inspect model
calculations and gradients; a black-box attacker has more limited access. An
untargeted attack seeks an unwanted result without choosing one particular wrong
output, while a targeted attack aims at a specified result. Evasion changes inputs
to an already trained model; poisoning changes its training data.

These distinctions describe our current experiments precisely: we inspect gradients,
change test-time weather inputs and aim to increase forecast error. The model's
weights and the true future remain fixed, with the future available as a reference
during evaluation. The survey largely concerns classification, so our numerical
forecast scores replace wrong-label counts. It complements the newer survey as a
source of basic terminology, rather than a complete account of later methods.

## Weather-specific adversarial studies suggested by Rich

All four suggested works are present locally, checked on 15 October 2026. The
saved FABLE paper is April 2026 v2, extending the original 2025 preprint; its author
list includes Jack Gunn. The ECMWF-hosted report lists Mahendran, Schmidt and
Benn-Maksimovic in that order. Filenames now follow the titles printed in the PDFs,
including subtitles. Renaming preserves the PDF contents and annotations.

### Adversarial Observations in Weather Forecasting

[Local PDF](<Adversarial Observations in Weather Forecasting.pdf>) ·
[Source](https://arxiv.org/abs/2504.15942) ·
[Notes and questions](notes/AdversarialObservations.md)

Imgrund, Eisenhofer and Rieck attack GenCast, a probabilistic model that samples
possible future weather maps. Their method adjusts initial weather inputs using
gradients through a cheaper approximation of the diffusion forecasting process,
with momentum and constraints on perturbation statistics. The objectives include
fabricating and concealing extreme events. Although the motivation concerns
compromised observations, the demonstrated attack changes assimilated ERA5 grids.
This distinction matters: a change to an ERA5 cell is not automatically a change
that could be produced by editing a particular satellite measurement before assimilation.

### FABLE: A Localized, Targeted Adversarial Attack on Weather Forecasting Models

[Local PDF](<FABLE: A Localized, Targeted Adversarial Attack on Weather Forecasting Models.pdf>) ·
[Source](https://arxiv.org/abs/2505.12167) · [Notes and questions](notes/FABLE.md)

FABLE aims to produce chosen forecast changes at selected places and times. It
changes wavelet components, which describe patterns at different spatial and
temporal scales, and reconstructs an input bounded by an Linf limit. The experiments
use deterministic CLCRN and adapted FourCastNet models with gridded NLDAS and ERA5
data. A chosen output region and an allowed input region are separate parts of
an attack. The work connects to regional targets and comparisons of disturbance
patterns in our CNN; its plausibility measures should not be described as proof
of complete physical consistency.

### Forecasting Fails: Unveiling Evasion Attacks in Weather Prediction Models

[Local PDF](<Forecasting Fails: Unveiling Evasion Attacks in Weather Prediction Models.pdf>) ·
[Source](https://arxiv.org/abs/2512.08832) · [Notes and questions](notes/ForecastingFails.md)

Arif and colleagues attack FourCastNet through processed ERA5 initial fields.
Their WAAPO method combines gradient updates with selected-variable, geographical,
amplitude and smoothness restrictions. The underlying forecaster is deterministic;
averaging forecasts from different perturbed initial conditions is a separate
ensemble experiment. This is directly relevant to comparing unrestricted cell
patterns with geographically limited or smooth changes in our CNN. Changing only
some variable channels is different from limiting the total number of changed cells.

### Explainability and Adversarial Vulnerability in AI-Based Weather Forecasting Systems

[Local PDF](<Explainability and Adversarial Vulnerability in AI-Based Weather Forecasting Systems.pdf>) ·
[ECMWF source](https://www.ecmwf.int/en/elibrary/81717-explainability-and-adversarial-vulnerability-ai-based-weather-forecasting-systems) ·
[Notes and questions](notes/ExplainabilityAndAdversarialVulnerability.md)

This February 2026 report discusses explanation methods, attack assumptions and
trust in weather systems. It reviews existing work, including the GenCast attack,
rather than presenting a new attack experiment. It helps distinguish a gradient
picture, which shows sensitivity, from an actual experiment demonstrating forecast
damage. Its discussions cover several model types and input stages, so it has no
single deterministic/probabilistic label or single attacked dataset.

## The presentation and its connection to the CNN experiments

The planned presentation brings together a brief understanding of each paper,
the evidence supporting its claims and the questions it leaves open. Each paper
can be described through the same set of distinctions:

| Part of the account | What it establishes |
|---|---|
| Main idea | The problem studied and the paper's contribution, in everyday terms. |
| Forecasting model and output | Which model is used; whether it returns one forecast or samples possible futures. |
| Input actually changed | Raw observation, processed weather state such as ERA5, or no attack experiment. |
| Attack objective | General forecast damage, a chosen regional outcome, or no attack objective. |
| Attack procedure and limits | How changes are chosen; allowed variables, locations, sizes and computational effort. |
| Evidence and limitations | What the reported experiments support and what remains untested. |
| Open questions and CNN connection | Questions raised by the reading and a small experiment that could examine them. |

A deterministic forecaster returns the same map for the same input and fixed
settings. A probabilistic forecaster represents multiple possible futures, often
through sampled maps. A group of deterministic forecasts made from different
initial conditions should be described separately from randomness inside the model.
ERA5 is a reanalysis: an estimated atmospheric state reconstructed from observations
and a physical model. It is not a folder of raw satellite or station readings.

| Work | Input or attack location | Forecast type / role |
|---|---|---|
| How Deep Learning Sees the World | Multiple image-attack settings; no single weather experiment | Survey; no single weather forecast type |
| Adversarial Attacks and Defences | Multiple attack settings; no single weather experiment | Survey; basic attack vocabulary |
| Boosting Adversarial Attacks with Momentum | Image classifier inputs | Classification attack, not weather forecasting |
| Otter, newer paper | Processed atmospheric states; no attack study | Deterministic and probabilistic variants |
| Otter, archived paper | Processed atmospheric states; no attack study | Deterministic |
| Atmospheric Predictability Beyond 30 Days with Machine Learning | Initial processed weather states; changes reduce error | Deterministic GraphCast; transfer to Pangu-Weather |
| A foundation model for the Earth system | Processed Earth-system states; no attack study | Reported weather setup is deterministic |
| End-to-end data-driven weather prediction | Observation-based forecasting pipeline; no attack study | Reported setup is deterministic |
| Adversarial Observations in Weather Forecasting | Assimilated ERA5 grids; observation compromise is the motivation | Probabilistic GenCast |
| FABLE | Processed NLDAS and ERA5 input histories | Deterministic CLCRN and adapted FourCastNet |
| Forecasting Fails | Processed ERA5 initial states | Deterministic FourCastNet; perturbed-input ensembles also examined |
| Explainability and Adversarial Vulnerability | Review of several input stages and attack settings | Report, not one new model experiment |

The completed CNN work supplies a small practical comparison: clean forecasts,
random changes, one-step gradient attacks, PGD, momentum updates and sparse changes.
Validation data select attack settings before evaluation on the test period.
[Attack experiments](../toy_weather/ATTACK_EXPERIMENTS.md) record the actual results;
a paper-inspired proposal is kept separate from an experiment already completed.
Possible extensions include a chosen geographical target, a mask restricting input
changes, smooth disturbances and attacks evaluated over repeated forecast steps.
These remain proposed extensions unless the experiment record says otherwise.

## Reading notes as the discussion develops

Notes explain the subject directly, using descriptive prose. They distinguish
paper claims, checked interpretations, unresolved questions and proposed CNN
experiments. New questions and their explanations belong in the relevant paper's
notes, with version or section references where they matter. Open questions below
are starting points for reading, not claims that every paper has already been read.

| Paper | Notes |
|---|---|
| How Deep Learning Sees the World | [Attack methods, definitions and weather adaptations](notes/HowDeepLearningSeesTheWorld.md) |
| Adversarial Attacks and Defences | [Attack assumptions and vocabulary](notes/AdversarialAttacksAndDefences.md) |
| Boosting Adversarial Attacks with Momentum | [Momentum and sparse attacks](notes/BoostingAdversarialAttacksWithMomentum.md) |
| Otter, both versions | [Architecture, training and forecast uncertainty](notes/Otter.md) |
| Atmospheric Predictability Beyond 30 Days with Machine Learning | [Input optimisation](notes/PredictabilityBeyond30Days.md) |
| A foundation model for the Earth system | [Aurora and extreme-weather evaluation](notes/Aurora.md) |
| End-to-end data-driven weather prediction | [Aardvark and observation-based forecasting](notes/Aardvark.md) |
| Adversarial Observations in Weather Forecasting | [GenCast attack, input stage and open questions](notes/AdversarialObservations.md) |
| FABLE | [Targeted forecasts, wavelets and open questions](notes/FABLE.md) |
| Forecasting Fails | [Location, smoothness and variable constraints](notes/ForecastingFails.md) |
| Explainability and Adversarial Vulnerability | [Sensitivity, attacks and operational trust](notes/ExplainabilityAndAdversarialVulnerability.md) |

[WeatherVariablesAndMetrics.pdf](WeatherVariablesAndMetrics.pdf) is a reference
sheet rather than a research paper, so its existing descriptive filename remains.
All four works Rich listed are present. This check establishes completeness against
that list, not against every weather-adversarial paper in the wider literature.

## PhysMetrics.Weather: An Evaluation Framework for Physical Consistency in ML Weather Models

[Local PDF](<PhysMetrics.Weather- An Evaluation Framework for Physical Consistency in ML Weather Models.pdf>) ·
[Paper](https://arxiv.org/abs/2606.10642) · [Reading notes](notes/PhysMetricsWeather.md)

An evaluation toolkit for existing forecasts: mass/energy budgets, spatial-scale
energy and balances between atmospheric variables. It complements RMSE and may
help compare physical consistency before and after our attacks. Some diagnostics
require global coverage and vertical variables absent from the current toy CNN.
Reading started 22 October 2026.

Variable coverage: [PhysMetrics vs Otter](figures/physmetrics_otter_venn.png) ·
[PhysMetrics vs Aardvark](figures/physmetrics_aardvark_venn.png).
The diagrams compare core diagnostic fields with gridded forecast fields;
vertical coverage and derived surface pressure still affect which checks apply.

![PhysMetrics versus Otter: variable coverage](figures/physmetrics_otter_venn.png)

![PhysMetrics versus Aardvark: variable coverage](figures/physmetrics_aardvark_venn.png)

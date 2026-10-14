# Papers behind the weather-input attack project

This folder brings together two sides of the project: how AI models forecast
weather, and how deliberate changes to their inputs can affect those forecasts.
The toy experiments connect these ideas on a small temperature grid. The longer-term
question is whether similar input changes affect a larger model such as Otter,
including forecasts with several variables, repeated forecast steps and extreme events.

The collection contains the five distinct works originally recommended, plus
Otter, the momentum-attack paper and a weather-variable reference sheet. The six original links represent
five works because the predictability paper has both a preprint and a journal
publication. The local PDFs include existing personal annotations. Detailed
questions and corrections are stored in `notes/` and linked below.

## How Deep Learning Sees the World: attacks and defenses

[Local PDF](<How Deep Learning Sees the World- A Survey on Adversarial Attacks & Defenses.pdf>) ·
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

[Local PDF](BoostingAdversarialAttacksWithMomentum.pdf) ·
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

[Main local PDF](Otter.pdf) ·
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
[archive/Otter_ICLR2026.pdf](archive/Otter_ICLR2026.pdf). Our saved `otter-main/`
source code has not been updated merely by replacing the paper. The
[version comparison](notes/Otter.md#follow-up-output-representation-training-relationship-and-paper-versions)
separates the papers' claims from what the local code currently supports.

## Atmospheric Predictability Beyond 30 Days: changing the starting weather

[Annotated local preprint](PredictabilityBeyond30Days.pdf) ·
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

[Local PDF](AardvarkWeather.pdf) ·
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

[Local PDF](<Adversarial Attacks and Defences- A Survey.pdf>) ·
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

## Reading log: notes and questions as we go

The detailed notes preserve the explanations developed while reading, including
corrections and distinctions between published experiments and our proposed work.
The newer survey notes cover weather adaptations, attack limits, method families,
evaluation and defenses. The Otter notes cover architecture, evaluation, paper
versions, dropout and probabilistic forecasting through the Section 3.2 checkpoint.
The predictability notes explain input optimisation and its use of “epoch.”
Otter reading is currently paused; Aurora connects most directly to the proposed
multi-variable and extreme-weather experiments.

| Paper | Detailed notes |
|---|---|
| How Deep Learning Sees the World | [Questions, corrections and attack catalogue](notes/HowDeepLearningSeesTheWorld.md) |
| Boosting Adversarial Attacks with Momentum | [Momentum updates and sparse-attack distinctions](notes/BoostingAdversarialAttacksWithMomentum.md) |
| Otter Weather | [Architecture, evaluation, versions and probabilistic output](notes/Otter.md) |
| Atmospheric Predictability Beyond 30 Days | [Input updates and optimisation epochs](notes/PredictabilityBeyond30Days.md) |

[WeatherVariablesAndMetrics.pdf](WeatherVariablesAndMetrics.pdf) is an additional
reference sheet for weather-variable names and metrics, not a research paper.
Together, the collection explains the attack methods, the forecasting models,
the role of initial conditions and the evaluation choices behind our project.

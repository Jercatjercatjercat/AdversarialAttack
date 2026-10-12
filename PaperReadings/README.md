# Reading guide for the weather-input attack project

Inventory updated on 11 October 2026 after checking the three newly downloaded
PDFs' text, titles and article identifiers, not just their filenames. All five
distinct suggested works are now saved. Existing personal annotations are preserved.
The six suggested links describe **five distinct works**: the first two are
the preprint and journal publication of the same predictability study.

## What is already saved?

| Suggested reference | Local status |
|---|---|
| [arXiv:2504.20238 — Atmospheric Predictability Beyond 30 Days with Machine Learning](https://arxiv.org/abs/2504.20238) | Present: [PredictabilityBeyond30Days.pdf](PredictabilityBeyond30Days.pdf), annotated arXiv v2, dated May/June 2026 |
| [AMS: AIES-D-26-0009.1 — Atmospheric Predictability beyond 30 Days with Machine Learning](https://journals.ametsoc.org/view/journals/aies/aop/AIES-D-26-0009.1/AIES-D-26-0009.1.xml) | Same study is present as the preprint above; a separate journal-version PDF is not saved |
| [Nature: s41586-025-08897-0 — End-to-end data-driven weather prediction](https://www.nature.com/articles/s41586-025-08897-0) | Present: [AardvarkWeather.pdf](AardvarkWeather.pdf), previously named `ML_WP.pdf` |
| [Nature: s41586-025-09005-y — A foundation model for the Earth system](https://www.nature.com/articles/s41586-025-09005-y) | Present: [Aurora paper](<A foundation model for the Earth system.pdf>), published Nature version, May 2025 |
| [arXiv:1810.00069 — Adversarial Attacks and Defences: A Survey](https://arxiv.org/abs/1810.00069) | Present: [2018 survey](<Adversarial Attacks and Defences- A Survey.pdf>), arXiv v1 |
| [arXiv:2305.10862 — How Deep Learning Sees the World: A Survey on Adversarial Attacks & Defenses](https://arxiv.org/abs/2305.10862) | Present: [newer survey](<How Deep Learning Sees the World- A Survey on Adversarial Attacks & Defenses.pdf>), arXiv v1, May 2023 |

Other saved material: [Otter.pdf](Otter.pdf), the deterministic ICLR 2026 Otter
paper, and [WeatherVariablesAndMetrics.pdf](WeatherVariablesAndMetrics.pdf), our
existing reference sheet moved here from the workspace root. The reference sheet
is not one of the six requested articles. A separate AMS journal-version PDF of
the predictability study is still not saved; the annotated preprint covers that work.

## Start here: one short reading session

Start with [How Deep Learning Sees the World](<How Deep Learning Sees the World- A Survey on Adversarial Attacks & Defenses.pdf>).
Read **Section II, Background for Adversarial Attacks**, then the FGSM, iterative
attack and PGD passages in **Section IV.A, White-box Settings**. Aim for a first
30–45 minute pass, not the whole survey. Skip the long method catalogue and defenses
until the basic input-change idea is clear. The 2018 survey can be a second
explanation of unfamiliar terms rather than another complete reading assignment.

Connect the reading to our code with four questions:

1. What stays fixed? The trained weights and the true future temperature.
2. What changes? The input temperature map, within a stated limit.
3. What does the gradient tell us? Which small input changes increase forecast loss.
4. What does projection do? Bring each attempted change back inside its budget.

The survey mostly predicts image labels. Our model predicts temperatures, so a
wrong-class objective becomes increased numerical forecast error. Our random-noise
comparison asks whether choosing the direction using the gradient matters.

Next read **Otter Sections 2 and 3, the Rollout Fine-tuning paragraph, and Appendix
B.3** alongside [the preparation note](../OTTER_PREPARATION.md). Then read the
predictability study's **Section 2, Data and Methods**, focusing on what is fixed,
what is optimized and how future error enters the calculation.

## Reading log: notes and questions as we go

Use this README as the index; keep detailed, corrected notes with each paper.
Each update distinguishes the user's interpretation, checked claims, proposed
weather applications and questions still open. Existing PDF annotations remain intact.

| Date | Paper | Progress and next step |
|---|---|---|
| 12 October 2026 | How Deep Learning Sees the World | First selections/questions reviewed; [corrected notes and complete attack catalogue](notes/HowDeepLearningSeesTheWorld.md). Read Otter's setup/training next. |

Main conclusions from the first notes:

- Weather attacks need explicit physical limits and an error/target objective;
  visual invisibility and high-confidence wrong classes are not universal requirements.
- L0 counts changed entries, L2 shares a magnitude budget and Linf caps each entry.
  A norm alone does not determine visibility, spatial smoothness or physical consistency.
- Untargeted and targeted objectives apply to numerical forecasts; class-confidence
  reduction needs probabilities/uncertainty, which the current CNN does not output.
- FGSM/PGD can attack differentiable Transformers directly. Reusing a CNN's chosen
  changes on Otter is a separate transfer experiment and may fail.
- Bookmark Section V's defenses and Section VI's attention/patch studies for later.
- Adapt Section VIII's evaluation questions to RMSE and clearly defined success
  thresholds. Destruction rate concerns attacks losing effectiveness after a
  transformation; it is not an input-disturbance-size measure.

The detailed notes list the general, universal, black-box and Transformer-oriented
attacks by section, with acronyms explained and defenses kept out of the attack list.

## Why each work matters

### 1–2. Predictability beyond 30 days: the closest methodological connection

The study freezes a trained GraphCast weather model and changes its initial weather
state using gradients to REDUCE future forecast error. Our attacks change inputs
to INCREASE error. Both differentiate through a fixed forecaster. The study also
tests whether inputs optimized for GraphCast help a different model, Pangu-Weather.
It uses known future weather during optimization, so its retrospective improvements
are not evidence of an operational method that knows the future in advance.
Sources: [preprint](https://arxiv.org/abs/2504.20238) and
[journal publication](https://journals.ametsoc.org/view/journals/aies/aop/AIES-D-26-0009.1/AIES-D-26-0009.1.xml).

For our project, read the methods with three questions: what stays fixed, what gets
changed, and which future times contribute to the error? This suggests experiments
comparing error-increasing and error-reducing input changes, then checking how their
effects grow across repeated forecast steps. Later, **transfer** would mean applying
changes chosen for one model to another model without choosing them again.
Its input-optimization procedure is not automatically the same as our constrained
FGSM/PGD attacks. Use the journal version for final citation when possible; retain
the annotated preprint for study and check differences before quoting results.

### 3. Aardvark: identify where an input attack enters the weather pipeline

Aardvark takes observations, estimates the current atmospheric state, forecasts
future global states, and produces local station forecasts. Its encoder turns
observations into a state; its processor forecasts; its decoder produces local
outputs. This is a broader pipeline than supplying an existing ERA5 state to Otter.
Source: [End-to-end data-driven weather prediction](https://www.nature.com/articles/s41586-025-08897-0).

This distinction helps us define our experiment precisely: we currently change a
reconstructed temperature map at the forecasting model's input. We are not modifying
raw satellite or station observations before state estimation. Read Figure 1 and
the model overview first. Later, this paper can motivate studying whether state
estimation reduces or amplifies an observation change, and measuring effects at
particular stations rather than only across a map. It is useful context, not an
FGSM/PGD implementation guide.

### 4. Aurora: move toward several variables and extreme-event evaluation

A **foundation model** is pretrained broadly and then adapted to more specific
tasks. Aurora learns from diverse Earth-system data and uses an encoder, processor
and decoder to handle different variables and atmospheric levels. The paper includes
weather, tropical-cyclone, air-quality and wave applications. Source:
[A foundation model for the Earth system](https://www.nature.com/articles/s41586-025-09005-y).

For our project, focus on the input/output structure and evaluation, rather than
trying to reproduce its large training setup. It suggests investigating which input
variable matters most and whether attacks affect severe conditions more than ordinary
weather. Each variable needs its own units and attack limits: a temperature change
in Kelvin cannot be added directly to a wind change in metres per second when defining
a physical budget. This paper is a source of experiment ideas, not evidence that
Aurora or Otter has the vulnerabilities observed in our toy CNN.

### 5. The 2018 survey: learn the attack vocabulary

This survey organizes attack scenarios and defenses. Source:
[Adversarial Attacks and Defences: A Survey](https://arxiv.org/abs/1810.00069).

Use it to describe what the attacker knows and can change. A **white-box** attack
can inspect model calculations and gradients; a **black-box** attack has more limited
access. An **untargeted** attack increases error without demanding one particular
wrong output; a **targeted** attack aims for a specified wrong result. An **evasion**
attack changes inputs to an already trained model; **poisoning** changes its training
data. Our current experiment is white-box, untargeted evasion with known future
targets during evaluation.

Read the attack definitions and FGSM/iterative-method discussion first. Much of
the survey concerns classification, which chooses a label; our forecaster instead
predicts numbers. We measure increased forecast RMSE, not classification accuracy.
It is foundational background rather than a complete account of today's methods.

### 6. The newer survey: broaden attack and evaluation choices

This survey covers attacks grouped by attacker capabilities, defenses and attacks
on Vision Transformers, models that process spatial patches using attention.
Source: [How Deep Learning Sees the World](https://arxiv.org/abs/2305.10862).

This helps when moving from a CNN to Otter's Transformer architecture. It can guide
later comparisons of attacks with different access, disturbances that work across
models, and defenses. A defense must be tested against attacks designed for that
defended pipeline: a weak attack result alone does not show genuine resistance.
Read the attacker-capability and evaluation sections selectively. Image examples
are useful intuition, but weather limits should be stated in physical units and
evaluated as forecast errors; an image's visually imperceptible change is not a
definition of plausible atmospheric uncertainty.

## Suggested order and an achievable reading goal

1. Read the newer survey's background and selected white-box methods as described
   above. Use the 2018 survey to clarify terminology when needed.
2. Read the saved Otter paper's setup, training method and Appendix B.3 alongside
   [OTTER_PREPARATION.md](../OTTER_PREPARATION.md).
3. Read the predictability study's methods carefully: it is closest to differentiating
   with respect to weather inputs. Treat the two links as versions of one study.
4. Read Aurora's overview and evaluation for multi-variable/extreme-weather ideas.
5. Read Aardvark's pipeline figure to place our state-input experiment in the wider
   observation-to-forecast process.

For each work, write five short notes: input, predicted output, fixed quantities,
changed quantities, and how error is measured. Separate what the authors actually
tested from experiments we propose. No paper needs to be completely mastered before
we try one clean Otter forecast and one input-gradient check.

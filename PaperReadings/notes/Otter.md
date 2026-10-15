# Reading notes: Otter Weather

Architecture questions recorded on 13 October 2026. These notes refer to the saved
[deterministic ICLR 2026 paper](<../archive/Otter Weather: Highly Skillful Medium-Range Weather Forecasting on a Single GPU.pdf>), especially Figure 2 and Section 3.2,
and the [local model configuration](../../otter-main/otter/experiment/config/base.yaml).
On 13 October 2026, the [June 2026 preprint](<../Otter Weather: Skillful and Computationally Efficient Medium-Range Weather Forecasting.pdf>) replaced the main reading
PDF. It includes probabilistic variants. Earlier architecture notes used the
archived ICLR paper; the local source code remains unchanged.

## Section 2 checkpoint: Related Work

Reviewed on 13 October 2026. Section 2 explains how Otter fits into existing
research, rather than giving its complete input/output or training procedure.
Correction to earlier guidance: the saved paper's forecasting task is in Section
3.1, and its architecture is in Section 3.2. An earlier preparation note incorrectly
attributed the correction-prediction description to Section 2; that reference is fixed.

### What to take away from Section 2.1

1. **Several different designs can forecast weather.** Pangu-Weather uses
   Transformers adapted to Earth's spatial structure; GraphCast uses connected
   points and learned information passing; FourCastNet uses wave/frequency-based
   processing; ClimaX and Aurora are broadly pretrained models adapted to tasks.
   Memorizing every cited model is unnecessary. Recognize the families and the
   different ways they represent spatial information.
2. **Otter's central question concerns forecasting quality versus training cost.**
   The authors argue that standard components, efficiently implemented on GPUs,
   can compete with more specialized designs at lower cost. This is their research
   argument, not proof that specialized architectures never help or that physics
   is irrelevant. Their experiments later in the paper supply the evidence.
3. **Training, running and adapting a model have different costs.** Training learns
   weights; inference runs a trained model to produce a forecast; fine-tuning
   continues learning from existing weights for another task or training stage.
   A model can run forecasts quickly yet remain expensive to train or adapt.
   Efficiency claims need their hardware, data, resolution and forecast task stated.

An **inductive bias** is a design assumption that makes some patterns easier for a
model to learn. For example, building Earth's geometry into its connections tells
it something about space before training. “Fewer weather-specific assumptions” does
not mean Otter receives no location information or can ignore the structure of weather.

### What to take away from Section 2.2

The authors investigate techniques developed in image and language modelling.
At this stage, recognize their roles rather than memorizing equations:

- RoPE supplies position-related information inside attention calculations.
- Muon is an optimizer, the rule that updates learned weights.
- Regularization includes methods such as dropout and weight decay intended to
  reduce overfitting, where training improves more than performance on unseen data.
- Other candidates change the processing blocks, connections or initial training.

The key word is **ablation**: compare controlled variants, often removing or replacing
one component, to estimate whether that choice helps under matched conditions.
Being listed as an investigated technique does not mean it is used in the final
model or improved performance. Look for those results later. Also, lower forecast
error does not automatically imply lower input-attack sensitivity.

### Relevance to our attack project

Section 2 gives model context and the reason Otter is attractive to smaller research
groups. It does not establish resistance to FGSM/PGD. We would still have to load
trained weights, verify clean forecasts and input gradients, and measure attacks
with explicit budgets. Fast inference alone does not determine how affordable
backpropagation or repeated attack steps will be.

A sufficient first-pass summary is: “Otter asks whether a relatively standard
spatial Transformer, combined with carefully tested training choices, can produce
competitive forecasts at lower training cost. Other forecasters use different
spatial designs. This section motivates that comparison; it does not establish
adversarial robustness.”

Next read Section 3.1 to record the actual inputs, forecast lead time, variables,
training/test dates, resolution and evaluation measures. Then use Figure 2 and
Section 3.2 to follow how those inputs become the predicted future.

## Evaluation protocol: benchmark, forecast type and forecast time

Question recorded on 13 October 2026. The quoted passage covers deterministic and
probabilistic Otter variants. For the latter, these notes use the
[expanded Otter paper, Section 3.1](https://arxiv.org/html/2606.26421v1#S3.SS1).
The archived ICLR PDF focuses on deterministic Otter; do not assume the versions
have identical evaluation windows or every extension is in the local code.

### Benchmark versus baseline versus metric

A **benchmark** is a shared test setup: the weather cases, reference future data,
spatial grid, forecast times and rules for calculating scores. Like an exam, it
provides common questions and marking rules so models can be compared. WeatherBench
2 supplies weather data, reference forecasts and evaluation tools. The paper specifies
which part of that setup it uses; the framework permits several configurations.
Source: [WeatherBench evaluation guide](https://weatherbench2.readthedocs.io/en/latest/evaluation.html).

- **Benchmark:** the test setup, such as a defined WeatherBench 2 comparison.
- **Baseline:** a reference method/model, such as predicting no change or an
  established physical weather forecasting system.
- **Metric:** a numerical scoring rule, such as RMSE.

The benchmark is not a single model, a single score, or a training algorithm.
Here, **1.5°** describes latitude/longitude grid spacing for evaluation, not
temperature accuracy. Models with finer native grids can have forecasts converted
to the common grid for comparison. The true-future reference is usually ERA5,
a reconstruction of weather rather than perfect direct observations everywhere.

### Deterministic and probabilistic forecasts get separate grading rules

| Type | What one input produces | Example at one location | What to score |
|---|---|---|---|
| Deterministic | One predicted weather state | “Tomorrow at noon: 23°C” | Point-forecast error, such as latitude-weighted RMSE |
| Probabilistic | A predicted distribution, often represented by multiple possible weather states | Possible forecasts 21, 22, 23, 24 and 25°C | Distribution quality, such as CRPS |

An **ensemble** is a collection of forecast outcomes. It represents the model's
possible futures; it is not a collection of true futures. A probabilistic model may
generate these outcomes using randomness without training a separate network for
every outcome. Different trained networks can also be combined into an ensemble.

Deterministic does not mean correct, certain, or trained without random choices.
It describes the single forecast output under a fixed inference setup. Probabilistic
does not mean adding arbitrary noise makes a forecast useful: its uncertainty must
be evaluated against what actually occurs across many cases.

**CRPS**, Continuous Ranked Probability Score, compares a forecast distribution
with the realized future value. Lower is better. It evaluates where the outcomes
are located and how spread out they are, rather than only the ensemble mean.
Two ensembles with the same mean but different spreads can receive different scores.
The underlying probability-scoring formula can wait until a later reading session.

Otter's evaluation separates point-forecast models, scored using RMSE, from the
probabilistic variants, scored using CRPS. The variants can share a backbone but
use different training objectives and output-generation procedures. This is not
another train/validation/test split. It is a separation by the kind of prediction
being assessed. RMSE and CRPS values are not interchangeable scores.

Our toy CNN is deterministic. Its three training seeds are evaluated separately
and their scores averaged; we have not used them to produce and score a probabilistic
forecast distribution.

### Follow-up: “I thought I just output a map”

Yes: one individual forecast is still a map. The distinction is one predicted
map versus a collection of possible maps for the SAME starting case and SAME
future time. It is not a distinction between a map and a single temperature number.

```text
Deterministic:
one input history → one map for tomorrow at noon

Probabilistic:
one input history → possible map 1 for tomorrow at noon
                  → possible map 2 for tomorrow at noon
                  → possible map 3 for tomorrow at noon
                  → ...
```

For one example and temperature only, a conceptual tensor layout is `[1,H,W]`
for a deterministic forecast versus `[M,1,H,W]` for M probabilistic ensemble members.
For a batch, those become `[B,1,H,W]` and `[B,M,1,H,W]`. These are teaching layouts,
not the exact dimension order of the Otter package. Otter predicts many fields.

Look at one grid cell across five possible maps: the predicted temperatures could
be 21, 22, 23, 24 and 25°C. Those values describe uncertainty at that cell. Each
map also predicts all the OTHER cells, forming a complete spatial forecast.
By contrast, maps at +6h, +12h and +18h from one forecast trajectory are different
times; having several time outputs does not itself make a model probabilistic.

For the probabilistic Otter variant, random internal dropout masks generate different
forecasts from the same supplied input history. **Dropout** temporarily sets
selected internal feature values to zero. **Monte Carlo dropout** means making
repeated predictions with independently drawn masks. The paper trains/fine-tunes
this setup with CRPS; merely switching randomness on in our MSE-trained CNN would
not establish that its spread is useful forecast uncertainty. Otter also combines
different fine-tuned checkpoints in some deep-ensemble variants. Source:
[expanded paper, Section 4.2](https://arxiv.org/html/2606.26421v1#S4.SS2).

The two evaluation procedures can use the same actual future map y:

1. **Deterministic:** compare the one predicted temperature with y at each cell,
   combine squared errors and take the square root to obtain RMSE.
2. **Probabilistic:** at each cell, collect the ensemble's predicted temperatures,
   score that predicted distribution against the one realized value in y using
   CRPS, then aggregate over cells, dates and the specified variables/lead times.
   CRPS here is a per-variable, per-location distribution score; it does not on
   its own evaluate every aspect of joint spatial consistency between cells.

There is still only ONE realized future map per case. It is not necessary to know
several true futures to score an ensemble; uncertainty quality is assessed over
many forecast cases. Averaging the predicted maps first and calculating RMSE can
assess the ensemble mean, but removes the spread information needed for a
probabilistic assessment. Generating uncertainty within the trained forecaster is
also different from an adversarial attack that deliberately corrupts weather inputs.

### Follow-up: expectation, CRPS and what is built into Otter

The probabilistic interpretation is close: at a given location and for a given
variable, values across possible forecast maps represent a predicted distribution.
One ordinary numeric map contains one value per location/variable; it does not
automatically contain a full nontrivial probability distribution at every cell.
In the sampled representation, the distribution is represented across ensemble
members. The model generates complete fields; this does not mean drawing unrelated
values independently for every cell.

Two corrections:

1. **A deterministic model does not necessarily generate a distribution and then
   average it.** It can directly learn one output value per cell. For an ideal
   predictor trained to minimize expected MSE, that value is the expected future
   given the input. Our learned toy CNN is an estimate, not a guarantee of this
   exact optimum. This MSE result is not a blanket guarantee for every loss or
   deterministic model. Otter's deterministic training uses weighted RMSE.
2. **CRPS is a scoring/training rule, not the forecast output.** The probabilistic
   model generates forecast samples. Training/evaluation compares their distribution
   with the known future using CRPS. At forecast time the actual future is not yet
   known, so that forecast's observed CRPS cannot be calculated yet. An ensemble
   mean map can also be scored with RMSE, but that score ignores uncertainty.

The expanded Otter paper includes CRPS-trained probabilistic variants that reuse
the backbone, with MC dropout or tested alternatives for generating different
members. That is an implemented research variant described in the paper, not an
automatic distribution calculation performed by every deterministic checkpoint.
Source: [expanded paper, Section 4.2](https://arxiv.org/html/2606.26421v1#S4.SS2).

The LOCAL saved code has a more limited default setup:

- [base.yaml](../../otter-main/otter/experiment/config/base.yaml) selects `Delta`:
  one forecast value per entry. Here `Delta` means a point distribution, not the
  six-hour weather correction. Its `sample()` returns the same stored prediction.
- [distributions.py](../../otter-main/otter/models/distributions.py) also has a
  Gaussian option that predicts a mean and standard deviation, a spread parameter.
  This alternative is not the same representation as the paper's MC-dropout ensemble.
- [weatherbench_probabilistic.py](../../otter-main/otter/evaluate/weatherbench_probabilistic.py)
  includes an ensemble CRPS evaluator and an ensemble-mean RMSE evaluator. It
  expects predictions with an ensemble dimension called `realization`.
- The default [save_forecasts.py](../../otter-main/otter/evaluate/save_forecasts.py)
  disables dropout through `model.eval()` and saves `prd_dist.mean`, without an
  ensemble axis. The `.mean` field name is a shared interface; with `Delta` it is
  simply the deterministic prediction, not an average computed from multiple maps.
- I did not find the paper's complete MC-dropout/CRPS training recipe in the
  saved configurations or training-loss module. It would need a compatible
  checkpoint/code path before we could run that published probabilistic variant.

Thus: probabilistic functionality is described in the expanded Otter paper, and
some probability/evaluation building blocks exist locally, but our saved default
configuration is deterministic. A CRPS evaluation script alone does not turn
single-map output into a useful forecast distribution.

### Follow-up: output representation, training relationship and paper versions

“Expanded paper” in earlier answers meant the separate later preprint
[Otter Weather: Skillful and Computationally Efficient Medium-Range Weather Forecasting](https://arxiv.org/abs/2606.26421),
submitted on 24 June 2026. At the time of those answers it was an online reference.
It is now the local [Otter Weather: Skillful and Computationally Efficient Medium-Range Weather Forecasting.pdf](<../Otter Weather: Skillful and Computationally Efficient Medium-Range Weather Forecasting.pdf>), replaced on 13 October 2026. Earlier
answers should have named the source explicitly before introducing its additional variants.

| Work | What it covers |
|---|---|
| Archived [ICLR paper](<../archive/Otter Weather: Highly Skillful Medium-Range Weather Forecasting on a Single GPU.pdf>): **Otter Weather: Highly Skillful Medium-Range Weather Forecasting on a Single GPU**, ICLR 2026 | Deterministic Otter, Swin U-Net architecture, training/component comparisons, rollout fine-tuning and a larger patch-size-one variant called Otter-PS1 |
| Main [Otter Weather: Skillful and Computationally Efficient Medium-Range Weather Forecasting.pdf](<../Otter Weather: Skillful and Computationally Efficient Medium-Range Weather Forecasting.pdf>): **arXiv:2606.26421v1**, June 2026 | The same core Swin U-Net approach, updated deterministic results, probabilistic CRPS variants, checkpoint ensembles, scaled Otter-XL variants and an additional acoustic-wave experiment |

The older paper cites probabilistic research but does not present the later paper's
Otter CRPS experiments. Both versions are now saved, with the newer preprint as
the main reading copy and the original annotated PDF in the archive.

#### What a probabilistic output looks like

For one input case and one future time, it is a collection of complete predicted
weather states. A teaching layout is `[M,121,240,82]`: M ensemble members, 121
latitude rows, 240 longitude columns and 82 changing weather fields. This omits
batch/time axes and is not a promise about exact package dimension order.
Each member contains temperature, wind, humidity and other predicted maps.

Looking at one location's T2m across members could give the following hypothetical
values, not extracted paper results:

| Ensemble member | Temperature at the same location and future time |
|---|---:|
| 1 | 21.4°C |
| 2 | 22.0°C |
| 3 | 23.1°C |
| 4 | 22.6°C |
| 5 | 24.0°C |

Each number above belongs to a different full weather prediction. Collectively
they represent sampled uncertainty at that location. They are not five different
forecast times or five known future outcomes. One can summarize them with a mean,
spread or fraction exceeding a selected temperature threshold.

The later paper's Appendix C.4 uses five ensemble members for its main comparisons,
ten for model selection, and nine in a deep-ensemble setup consisting of three
models generating three members each. Member count is an evaluation/sampling choice,
not the number of weather variables or a universal fixed output-channel count.
Source: [expanded paper, Appendix C.4](https://arxiv.org/html/2606.26421v1).

#### Conversion means additional training, not converting a map

The paper's fine-tuned probabilistic route starts with its deterministic pretrained
backbone. It preserves the main network layout, generates varied predictions through
internal randomness such as MC dropout, and continues updating weights using CRPS.
It then performs rollout fine-tuning. Figure 7 and Appendix B.3.5 describe five
epochs of one-step CRPS fine-tuning after deterministic pretraining.

```text
Deterministic pretraining: learn one-step forecast weights
  ├─ Rollout fine-tuning → deterministic forecasting model
  └─ CRPS fine-tuning with random sampling
       → Rollout fine-tuning → probabilistic forecasting model

Alternative also tested:
Random starting weights → CRPS training → Rollout fine-tuning → probabilistic model
```

The preferred MC-dropout approach can keep the same learned-layer layout and
per-sample map dimensions while producing varied outputs. The paper also tests
another noise-injection mechanism with additional parameters. Continued training
changes weights, so the probabilistic ensemble mean need not equal the original
deterministic forecast. Merely changing the evaluation metric or enabling random
dropout on an unchanged deterministic checkpoint is not the trained conversion.
Source: [Section 4.2 and training details](https://arxiv.org/html/2606.26421v1#S4.SS2).

Thus “same model” needs qualification: same architectural family/backbone idea,
but separate trained weight sets and inference procedures. The original and later
papers also have configuration differences; for example, the saved original base
uses absolute spatial embeddings, while the later base table lists spherical
embeddings alongside RoPE. These are different ways to supply location information.
Do not assume that a checkpoint from one version can be loaded with another
version's configuration, or that Otter-PS1 and Otter-XL are identical trained models.

For our first attack milestone, the original deterministic model remains sufficient:
one trained checkpoint, one verified forecast and one input gradient. Probabilistic
variants add sampling and uncertainty evaluation and can be a separate later study.

### Rollout fine-tuning is another, independent distinction

**RFT**, rollout fine-tuning, continues training while feeding earlier predictions
back as later inputs. It tries to reduce the difference between training on real
inputs for one step and forecasting on one's own imperfect predictions repeatedly.
Both deterministic and probabilistic models can undergo RFT. RFT does not by itself
turn a deterministic forecast into a probabilistic one.

The paper records RFT versus non-RFT because that training can affect scores.
Architectural component comparisons without RFT and comparisons with external
models after RFT answer different questions. In the expanded paper, non-RFT
component comparisons average 6–24h RMSE; the main deterministic external-model
comparison uses 24h, with longer results in appendices. The probabilistic evaluation
uses CRPS up to ten days. Check each figure's caption rather than assuming every
table uses identical horizons.

### A 24-hour comparison is not a 24-hour Otter model step

A **lead time** is how far ahead the forecast is made. A **horizon** is the maximum
lead time being considered. Comparing at increments of 24h means comparing weather
for +24h, +48h, +72h and so on where those leads are included in that experiment.

| Future being compared | Otter with a 6h step | ArchesWeather with a 24h step |
|---|---|---|
| +24h | Four repeated model steps | One model step |
| +48h | Eight repeated model steps | Two model steps |

Both models must start from the same forecast initialization time and be checked
against the same future timestamp. Daily comparison times do not change Otter's
internal six-hour step or compare one model's six-hour score with another's daily score.

### What are the headline variables?

They are selected representative weather quantities used to summarize performance,
not every input variable or learned feature channel. Pressure levels identify
atmospheric layers using air pressure; hPa is the pressure unit hectopascal.

| Label | Meaning |
|---|---|
| Z500 | Geopotential/height-related field at 500 hPa |
| Q700 | Specific humidity, amount of water vapour per mass of moist air, at 700 hPa |
| T850 | Temperature at 850 hPa |
| U850 / V850 | Eastward / northward wind components at 850 hPa |
| T2m | Temperature two metres above the surface |
| SP | In the checked Otter evaluation code, the label for mean sea-level pressure |
| U10m / V10m | Eastward / northward wind components ten metres above the surface |

The SP mapping is confirmed in [parse_wb_data.py](../../otter-main/otter/eval_utils/parse_wb_data.py).
Do not assume that label always denotes the same pressure quantity in every dataset.

Different variables have different units. The paper's aggregate skill score first
compares each variable's metric with the corresponding reference model's metric,
then averages those relative improvements. It does not simply add Kelvin RMSE
to wind-speed or pressure RMSE. See the expanded paper's Appendix C.1.

Reading goal: for each results figure, identify forecast type, score, RFT status,
lead times, variables and reference model before interpreting which model is better.

## What is the underlying architecture?

Otter uses a **2D Swin Transformer arranged as a U-Net**. This describes two
different aspects: U-Net is the overall layout; Swin Transformer describes the
processing blocks inside it. The architecture is how calculations are connected,
not the loss function or optimizer used to train their weights.

At a high level:

```text
Four recent weather states + time/location information
                        ↓
Learned patch features: numerical descriptions of small map regions
                        ↓
Encoder: process features while reducing spatial resolution
                        ↓
Bottleneck: process the coarsest internal representation
                        ↓
Decoder: increase spatial resolution and combine earlier features
                        ↓
Weather correction map
                        ↓ add the latest weather state
Predicted weather six hours ahead
```

The four states contain current weather and weather 6, 12 and 18 hours earlier.
Their variables are stacked as input channels. A strided convolution, a learned
filter applied at spaced locations, converts the maps into patch features called
**tokens**. These tokens are vectors of learned numbers, not additional weather variables.

In the encoder, the internal maps become smaller. The decoder restores resolution.
**Skip connections** pass earlier, finer-scale features to matching decoder stages,
helping retain spatial detail. That decreasing-then-increasing arrangement gives
the U-Net its name. It is not the same as the final addition of the current weather
state to the forecast correction; both provide direct routes for information, but
at different places in the model.

The actual implementation is in [transformer.py](../../otter-main/otter/models/architectures/transformer.py)
and [forecasting_model.py](../../otter-main/otter/models/forecasting_model.py).
“Constant-width” in Section 3.2 means the learned feature-vector length stays the
same across stages; it does not mean spatial map width stays unchanged.

## What does Swin attention do?

**Attention** learns how strongly to combine information from different tokens,
with those combinations depending on the input. Ordinary ViT attention can connect
patches across the whole map. **Swin**, short for shifted windows, normally performs
attention within local groups of patches, then shifts those groups between blocks
so information can cross the previous group boundaries. Repeated blocks and coarser
representations let information travel beyond one original local window.

For weather, think of processing a detailed local view and a broader view together.
This is intuition about information flow, not a claim that a particular attention
weight identifies a physical cause. Source:
[Swin Transformer](https://arxiv.org/abs/2103.14030).

## U-Cast, U-Net and ViT are different names

- **U-Cast** is a separate probabilistic weather forecaster built on a standard
  U-Net. Probabilistic means it produces possible forecast outcomes to describe
  uncertainty, rather than only one forecast. It is not Otter's name for its ViT.
  Source: [U-Cast paper](https://arxiv.org/abs/2604.09041).
- **U-Net** is an encoder–decoder layout with connections between corresponding
  scales. It was introduced with convolutional blocks, but the layout can also be
  used with Transformer blocks, as in Otter. Source:
  [original U-Net paper](https://arxiv.org/abs/1505.04597).
- **ViT**, vision transformer, is a Transformer operating on spatial patch tokens.
  Swin is a related hierarchical vision-Transformer design. Source:
  [original ViT paper](https://arxiv.org/abs/2010.11929).

U-Cast is absent from the archived ICLR paper but discussed in the new main
Otter preprint, Sections 2 and 4.2, as a separate weather model. It is not the
name of Otter’s architecture.

## Common designs for maps and weather

| Design | Main idea | Connection to this project |
|---|---|---|
| CNN | Learned filters mix information from nearby grid cells | Our toy model |
| U-Net | Reduce resolution, process broad features, restore resolution with earlier detail | Otter's overall layout; also used by U-Cast |
| ViT | Turn spatial patches into tokens and mix them using attention | A vision-Transformer family, not a whole weather forecasting pipeline |
| Swin Transformer | Mix tokens within shifted local windows and at several scales | Otter's processing blocks |
| GNN, graph neural network | Pass learned information between connected points on a graph | GraphCast is a weather example |
| FNO, Fourier neural operator | Learn field-to-field mappings using large-scale wave components in Fourier space | Another design for spatial physical-system prediction |

These are not mutually exclusive categories: a model can combine convolutions,
Transformer blocks, a U-Net layout and skip connections. Likewise, predicting a
correction is an output choice that several architectures can use. It does not
define a separate architecture or guarantee forecasting performance.

Further sources: [GraphCast](https://arxiv.org/abs/2212.12794),
[Fourier neural operator](https://arxiv.org/abs/2010.08895).
Our [toy CNN](../../toy_weather/archive/first_iteration/models/simple_cnn.py) is the local small example.

Next reading goal: follow Figure 2 and identify input, patch features, encoder,
bottleneck, decoder and the final forecast. Leave attention equations and optimizer
details for later questions. The [preparation note](../../OTTER_PREPARATION.md)
already records the correction-prediction and repeated-forecast training explanations.

## Follow-up: does Monte Carlo sample the deterministic output?

Question recorded on 13 October 2026, after replacing the main PDF.

**Close, but the starting point is the trained model, not its finished forecast map.**
Section 4.2 and Appendix B.3.5 describe this route:

1. Train a deterministic model to produce one forecast.
2. Continue training its weights using CRPS, which scores a collection of possible
   forecasts against the observed future. This is called fine-tuning.
3. At forecast time, run the resulting probabilistic model repeatedly on the same
   input weather. MC (Monte Carlo) dropout uses different random masks to switch
   off some internal features on each run. Each run produces a complete new map.

For example, three maps might predict 21, 22 and 24 °C at one location and the same
future time. These are illustrative values, not results from our model. The spread
represents the model's uncertainty; their average is an ensemble-mean forecast.
A single deterministic prediction of 22 °C does not tell us how wide that spread
should be, so sampling cannot be defined from that value alone.

Weights change during fine-tuning, but stay fixed while generating the repeated
forecasts. The randomness comes from the internal dropout masks. The paper also
trains probabilistic models directly from scratch using CRPS, so deterministic
pretraining is one route, not a requirement. Its deep-ensemble variant additionally
combines several trained checkpoints.

Replacing the paper does not install this training/sampling procedure in our saved
Otter code. See the implementation distinction above before running experiments.

Evidence: [new local preprint](<../Otter Weather: Skillful and Computationally Efficient Medium-Range Weather Forecasting.pdf>), Section 4.2 and Appendix B.3.5;
[online version](https://arxiv.org/html/2606.26421v1#S4.SS2).

### Small distinction to remember

- **Observation:** the weather that actually happened; this is the reference used
  to score forecasts.
- **Deterministic prediction:** one model-generated future map.
- **Probabilistic prediction:** several model-generated maps for the same future
  time, representing possible outcomes.
- **Perturbed input:** the starting weather we deliberately change in an attack.

MC dropout's random internal masks generate possible forecasts. They are not
adversarial changes to the input. In a later attack experiment, we would keep the
trained weights fixed and deliberately change the input to worsen a clearly
defined forecast score. How to handle randomness in that attack is still an
experimental choice; it has not been implemented for Otter here.

## Section 3.2 checkpoint: the Otter base architecture

Checked against the new main PDF on 13 October 2026, Section 3.2 and Figure 2.

1. **Stack the input information.** Weather variables and recent history are
   combined as channels on the same spatial grid. Time embeddings are extra
   numerical features telling the model about time. They are not measured weather.
2. **Convert patches into learned features.** A convolution turns small spatial
   patches into tokens, meaning vectors of learned numbers. The base width
   D = 1536 counts internal features per token, not weather variables.
3. **Use Swin attention inside a U-Net.** Attention lets nearby patch features
   exchange information. Shifted windows change the neighbourhoods between blocks.
   The encoder reduces spatial resolution, the bottleneck processes broad patterns,
   and the decoder restores resolution using earlier detail through skip connections.
4. **Keep the design fairly general.** Otter stacks variables rather than building
   separate specialised encoders for surface and pressure-level data. It keeps the
   same feature width across U-Net stages. This is a deliberate efficiency choice.
5. **Supply position information and simplify boundary handling.** RoPE encodes
   relative spatial position in attention. Removing attention masks permits
   communication across wrapped grid boundaries, including the latitude edges.
   The paper calls this a torus topology: a computational choice, not a claim that
   the Earth is doughnut-shaped. Positional information helps the network distinguish
   locations. Section 4.1 tests the skill/compute trade-off.
6. **Train and regularise the deterministic backbone.** The loss is latitude- and
   pressure-weighted RMSE. Latitude weighting accounts for differing grid-cell
   area; pressure weighting sets the importance of atmospheric levels. Muon is the
   optimiser used to update weights. Gradient accumulation combines gradients from
   several smaller batches before one update. Weight decay discourages large
   weights; dropout zeros selected internal features; stochastic depth skips whole
   residual blocks during training. These aim to reduce over-reliance and improve
   training behaviour, rather than generate adversarial examples.

Terms to recognise without memorising their formulas: RoPE (position encoding),
SwiGLU (the nonlinear feature-processing component), attention heads (parallel
attention calculations), and Muon (the optimiser). The stated base settings include
D = 1536, 16 attention heads, blocks per stage [2, 8, 4], dropout 0.1 and weight
decay 0.15. These are settings of this model, not universal rules.

**CRPS clarification from the preceding questions:** Randomness makes repeated
forecasts differ; CRPS trains the collection to be useful against observations.
CRPS alone does not make a model random. Identical ensemble members reduce the
fair CRPS formula to absolute error at each evaluated value. Random dropout alone
is likewise not evidence of well-calibrated uncertainty. Section 3.2 describes
the deterministic backbone; the probabilistic training is explained in Section 4.2.

For our attack project, identify the observed input, the internal features and the
forecast output. An attack deliberately changes the input with fixed trained
weights. Dropout changes internal activity; it does not replace input temperature
readings. The next reading step is Section 4.1: compare which design changes
improve forecast skill and how much computation they cost.

Source: [main Otter PDF](<../Otter Weather: Skillful and Computationally Efficient Medium-Range Weather Forecasting.pdf>), Section 3.2, Figure 2 and Section 4.2.

# Reading notes: How Deep Learning Sees the World

First notes recorded on 12 October 2026. These explain the main concepts, checked interpretations and proposed weather applications.
They do not claim that the additional attacks, metrics or defenses have been
implemented. Continue adding questions and dated notes here as reading progresses.

Sources: the saved [May 2023 arXiv v1 PDF](<../How Deep Learning Sees the World: A Survey on Adversarial Attacks & Defenses.pdf>)
and its [searchable text](https://arxiv.org/html/2305.10862v1).
Section numbers below refer to that version. This is a computer-vision survey,
not a weather-robustness study or a complete catalogue of methods developed since 2023.

## What to keep from this first reading

Highest priority: distinguish an attack's objective, allowed input changes and model
access; separate those choices from the model architecture; evaluate both forecast
damage and input-change size. Bookmark defenses for later. Keep architecture-specific
and black-box methods as future directions rather than trying every method now.

## 1. Invisible changes and confident mistakes: useful examples, not universal rules

The introduction describes image changes that people barely notice but that change
the predicted class. Keep the idea of deliberately exploiting model sensitivity.
Do not require every adversarial attack to be invisible: attacks can also use visible
patches or other explicitly allowed changes. Visibility is one possible constraint.

Also, a wrong class receiving the highest score does not mean the model became
more confident than before. For example, a prediction can change from a correct
class at 99% to a wrong class at 55%. It is still wrong. Increased confidence in
a wrong answer is a possible objective or outcome, not a necessary condition.

### Weather equivalent for our project

An input attack deliberately changes a supplied weather state within stated limits
to increase forecast error or produce a specified wrong forecast. Visual similarity
of coloured temperature maps is not a reliable definition of a small disturbance.
Use physical units, the affected region/variables/times and explicit limits.

For our current example:

```text
Observed/reconstructed temperature input x → clean prediction → compare with true future y
Changed input x + delta                   → changed prediction → compare with SAME y

Keep the trained weights fixed. Choose delta deliberately to worsen the forecast.
```

For instance, allow at most 0.1 K change to each input temperature cell, then measure
whether forecast RMSE increases. RMSE is the square root of average squared error.
The numerical limit is an experimental choice; it is not automatically justified
as realistic ERA5 uncertainty. To study realism, separately justify disturbance
sizes and constraints using measurement/state-estimation uncertainty, spatial
smoothness and relationships between variables. A norm bound alone cannot ensure
that a modified state is physically consistent.

This is corruption of the input supplied to the model, not a claim that the actual
atmosphere changed while its true future stayed identical. A calculated attack may
fail on an example; measure its effect rather than assuming success.

## 2. Norms: what they limit, and what they do not

The survey's limits and ours share the same idea. Apply the size measure to the
CHANGE delta, not to the original weather field or forecast error.

| Limit | Everyday interpretation | Additional point |
|---|---|---|
| L0 | Count how many entries were changed | It does not limit the size of those changes. Add an amplitude cap separately if needed. |
| L1 | Add the absolute changes | A shared absolute-change budget. Now implemented in the second-iteration attack sweep. |
| L2 | Square the changes, add them, take the square root | A shared budget that charges larger changes more strongly. |
| Linf / L∞ | Take the largest absolute change | Caps every entry individually; does not require spatial smoothness. |

**L0 is commonly called a norm, but it is technically a nonzero-entry count, not
a mathematical norm.** Its unit is a count. With multiple variables/times, specify
whether a counted item is one scalar entry, one grid location, or a whole patch.
For p >= 1, the general formula is `Lp = (sum(abs(delta)**p))**(1/p)`.
L2 is a member of that family; Linf is its limit as p grows. L0 is treated separately.

The useful interpretation of “larger p emphasizes large errors” is that large
entries contribute more strongly RELATIVE TO small entries inside that calculation.
It does not mean all numerical norm values grow as p grows. For `[1,1,1,1]`,
L1=4, L2=2 and Linf=1. Nor does a smaller p automatically give small changes a
larger absolute contribution. L0 only asks whether an entry changed at all.

For a concrete concentration example, `[1,1,1,1]` and `[2,0,0,0]` both have L2=2,
but their largest changes are 1 and 2. A shared L2 budget can therefore permit a
larger local spike. The allowed change pattern still depends on the attack objective
and other constraints; the norm does not determine the pattern by itself.

### Detectability is not ordered universally by the norm

Do not retain the survey's Linf-versus-L2/L0 visibility ranking as a general rule.
The budget, spatial pattern, data scale and detection procedure all matter. A
small L2 change can be hard to notice; a large Linf change can be obvious. A Linf
bound may allow rapid cell-to-cell oscillations, so bounded does not mean smooth.
This distinction is especially important for weather, where physical consistency
and plausible uncertainty matter more than a colour plot's appearance.

Our current `--budget-k r` caps each cell by r for Linf, but caps input RMS by r
for L2. On 256 cells, the L2 radius is 16r. Those allowed sets differ. Continue
reporting BOTH largest cell change and input RMS. See the worked examples in
[EXPERIMENTS.md](../../toy_weather/archive/first_iteration/EXPERIMENTS.md).

## 3. The goals carry over, but class labels do not

| Survey goal | Weather interpretation | Applies to the current model? |
|---|---|---|
| Confidence reduction | Alter reported uncertainty or reduce an event probability | Not directly: our deterministic CNN returns temperatures, not class probabilities or an uncertainty estimate. A probabilistic weather model would need an explicit probability/uncertainty objective. |
| Untargeted | Increase error without demanding a particular wrong temperature pattern | Yes: our current attacks increase forecast MSE. |
| Targeted | Push forecasts toward a specified wrong value, field or event outcome | A possible extension, e.g. underpredict heat in a chosen region. Not implemented yet. |

Example targeted objective: make a region's forecast 2 K too cold at a chosen lead
time, within the input budget. The target and success criterion must be specified
before evaluating the attack. A thresholded “heatwave / no heatwave” task could also
use classification metrics, but that would be an added event task, not our present
continuous temperature forecasting task. Merely changing a forecast is not enough;
the result must worsen error or satisfy the specified attack objective.

One implementation detail worth noting: the survey's FGSM equation uses a minus
sign with a desired target label. Minimizing loss toward a chosen wrong target
differs from our untargeted attack, which uses a plus sign to increase loss against
the true future. The sign depends on what the loss and objective mean.
Likewise, PGD's full minimum-over-weights/maximum-over-inputs formula describes
adversarial training. An attack on a frozen model uses only the input-changing
part; it does not update the forecaster's weights.

## 4. CNN methods, Transformers and transfer are different questions

**DNN** means deep neural network; both CNNs and Transformers belong to that broad
family. A **ViT**, vision transformer, processes spatial patches using attention,
learned comparisons that determine which information to combine. Otter uses a
Swin Transformer with local shifted attention windows and predicts weather fields;
it is not the survey's simple image classifier with a class-output head.

FGSM and PGD operate on input gradients. They can attack a differentiable Transformer
directly, with the forecast loss replacing the classification loss. Their basic
method is not restricted to CNNs. This does not imply every classification-specific
attack can be used unchanged: methods based on class scores or class boundaries
need a suitable weather objective.

**Direct attack:** calculate a change using Otter's own gradient.
**Transfer attack:** calculate a change for another model, then reuse it on Otter
without choosing the change again. A directly calculated Otter attack does not
require a CNN-generated change to transfer. Cross-model effects must be measured
with compatible data and variables; our cropped toy input is not directly an Otter input.

Section VI reports mixed results and discusses fair comparisons of training
settings, as well as patch/attention-specific attacks. Do not conclude that all
Transformers are safer than all CNNs. Architecture alone does not settle robustness.
The primary [ViT attack study](https://arxiv.org/abs/2104.02610) also tests several
ordinary gradient attacks and develops SAGA; that supports their applicability beyond CNNs.

## 5. Black-box access: two useful routes

Black-box means the target model's internal weights/gradients are unavailable.
It does not automatically mean the attacker knows no training data or model family.
Record exactly which outputs are available and how many model calls are allowed.

1. **Transfer:** attack an accessible substitute model and try the resulting input
   on the target. The target's gradient is never inspected; success is not guaranteed.
2. **Query-based search:** try inputs and use returned scores or forecasts to guide
   further changes. One option estimates how the loss changes by trying small
   increases/decreases; another searches without estimating gradients. A query is
   one call to the model. Forecast models can make this expensive.

If the API returns a whole temperature map, we can calculate forecast loss from
that map and a known target during evaluation. If it returns only an alert, much
less information is available. Full-output access is not the same as class-label-only
access. Our current comparisons are white-box, meaning model gradients are available.

## 6. Defenses: bookmark Section V and the defense parts of VI

Keep this as future work. The survey groups defenses into six broad areas:
adversarial training, modified training, supplementary networks, changed architectures,
validation/detection, and purification. Purification means processing the input to
try to remove harmful changes. There are many individual methods within those groups.

The most understandable first weather defense experiment would compare ordinary
training with training that includes attacked inputs. Another candidate is carefully
chosen input smoothing or quality checks. Smoothing can also erase real weather
structures, so evaluate clean forecasting as well as attacked forecasting.

Attacks must account for the defended pipeline. An **adaptive attack** is designed
with knowledge of the defense. Merely making gradients difficult to use can give
false confidence without making the system resistant to other attacks; see
[Obfuscated Gradients](https://proceedings.mlr.press/v80/athalye18a.html).
Defensive distillation or an attention-related defense is not automatically effective
just because it appears in the survey. The paper's “temperature scaling” concerns
classification scores, not changing meteorological temperature.

## 7. Section VIII metrics: borrow the question, adapt the definition

The survey's four main measures are accuracy, fooling rate, average robustness and
destruction rate. Their original definitions concern classification. The following
weather counterparts are PROPOSALS, not additional metrics already saved by our code.

| Survey measure | A useful weather counterpart | Important distinction |
|---|---|---|
| Clean / robust accuracy | Clean / attacked RMSE in K; optionally MAE, mean absolute error | Exact equality to a temperature target is not a useful correct/incorrect label. Report error rather than image-class accuracy. |
| Fooling rate | Fraction of cases meeting a predefined attack-success condition | For example, per-case RMSE increases by at least a stated amount, or a correctly forecast event is hidden. Any tiny change in a continuous prediction is not automatically success. |
| Average robustness | Smallest FOUND input budget that reaches a predefined forecast-failure threshold | Search across budgets; report failures to find one. This estimates attack resistance and does not prove a minimum perturbation or certified guarantee. |
| Destruction rate | Fraction of previously successful attacks made unsuccessful by a specified input transformation/defense | Measures loss of attack effectiveness after processing, not the numerical size of the disturbance. |

Destruction rate needs care: the survey's introductory sentence loosely associates
it with perturbation size, but its detailed definition concerns transformations.
The [original paper, Section 3.1](https://arxiv.org/html/1607.02533v4), explicitly
counts successful adversarial images whose correct classification is restored after
transformation. Use that interpretation, not “how much input was destroyed.”
For a weather adaptation, define success first, then count attacks that lose it.
If there are no initially successful attacks, the rate is undefined, not zero.

Do not copy average robustness's `change_L2 / input_L2` directly for absolute
temperature: the input norm changes when switching between Kelvin and Celsius,
even though the physical disturbance does not. Physical input budgets or explicit
training-based scales are more interpretable. Different weather variables also
need separate units/scales.

For a first expanded results table, keep our existing clean RMSE, attacked RMSE,
RMSE increase, largest cell change and input RMS. Add per-case error-change summaries
or a thresholded success rate later. An illustrative criterion might be “RMSE rises
by at least 0.5 K”; that number is an example, not a justified default or implemented
setting. Define criteria using the research question/validation before examining
final test outcomes. When evaluating event flips, report performance on cases the
clean forecast already got wrong as well as the clean-correct subset.

For defenses, compare the undefended clean, undefended attacked, defended clean,
and defended attacked results. Include attacks that account for the defense; a
transformation undoing an old attack is not enough to establish general robustness.

## 8. Attack catalogue from this survey

Scope: named attacks/families discussed in Section IV and the attack parts of VI,
plus the named occlusion attack introduced during Section V's defense discussion.
This is not every method in the bibliography or a claim that all methods apply
unchanged to weather. Several families overlap; they are not independent categories.
Only random noise, FGSM/L2 single-step and PGD are implemented in our toy project.

### General attacks, Section IV.A

| Name | Basic idea |
|---|---|
| L-BFGS, Limited-memory Broyden–Fletcher–Goldfarb–Shanno | Solve for a low-distortion wrong-class input using numerical optimization. |
| FGSM, Fast Gradient Sign Method | One input step using gradient signs. |
| JSM / commonly JSMA, Jacobian-based Saliency Map Attack | Use derivatives to select influential input entries. The survey calls it JSM. |
| DeepFool | Iteratively seek a small change crossing a classification boundary. |
| BIM, Basic Iterative Method / I-FGSM, Iterative FGSM | Repeat small sign-gradient steps while limiting changes. |
| ILCM, Iterative Least-likely Class Method | Target the class initially assigned the lowest score. |
| C&W, Carlini–Wagner: L0, L2 and Linf variants | Optimize wrong-class objectives with different disturbance restrictions. |
| GAAS, Gradient Aligned Adversarial Subspace | Explore multiple damaging directions using gradients. |
| PGD, Projected Gradient Descent | Repeat input-gradient steps and project back inside the budget. |
| AdvGAN | Train a generator network to produce damaging input changes. |
| MI-FGSM, Momentum Iterative FGSM | Combine recent gradient directions across iterations. |
| SIAA, Sparse and Imperceivable Adversarial Attacks | Construct sparse changes with local image-dependent restrictions. |
| SmoothFool / SF | Seek smoother classification-boundary-crossing changes. |
| AdvCam, Adversarial Camouflage | Craft image changes resembling plausible physical camouflage. |
| FIA, Feature Importance-aware Attack | Use important internal features to guide transferable changes. |
| MGAA, Meta Gradient Adversarial Attack | Use gradients from several models to improve cross-model effects. |

### Universal, black-box and attack suites, Sections IV.B–D

| Name/family | Basic idea |
|---|---|
| UAPs, Universal Adversarial Perturbations | Find one change that damages many inputs. Universal refers to reuse across inputs, not guaranteed success across all models. |
| UAN, Universal Adversarial Networks | Generate broadly reusable perturbations with a network. |
| Ilyas et al.'s query-limited attack using NES, Natural Evolutionary Strategies | Estimate gradient direction from model queries. |
| Ilyas et al.'s partial-information and label-only variants | Search with only restricted scores or returned labels. These are access settings/variants, not two new acronyms. |
| FGBB, Feature-Guided Black-Box | Guide query-based search using extracted image features. |
| Square Attack | Use queries and randomly placed square-shaped updates without model gradients. |
| AutoAttack / Auto-Attack | Combine attacks into an evaluation suite rather than one new perturbation rule. |
| APGD-CE, Auto-PGD with cross-entropy | PGD variant using classification loss and automatic step-size adjustment. |
| APGD-DLR, Auto-PGD with difference-of-logits-ratio loss | PGD variant using a classification-score-based objective. |
| FAB, Fast Adaptive Boundary | Search for small classification-boundary-crossing changes; the suite uses a targeted variant. |

AutoAttack combines APGD-CE, APGD-DLR, targeted FAB and Square Attack. These
classification losses/boundaries cannot simply be applied unchanged to temperature
regression. “Automatic step size” does not mean the method never has settings.

### Attacks discussed for ViTs, Section VI

| Name | What is special about it |
|---|---|
| SAGA, Self-Attention blended Gradient Attack | Combine gradient information to attack CNNs and Transformers. |
| PNA, Pay No Attention | Ignore selected attention-gradient paths when crafting transferable changes. |
| PatchOut | Use random subsets of patches during attack construction. |
| ATA, Architecture-oriented Transferable Attacking | Use patch/embedding sensitivity across Transformer architectures. |
| Patch-fool | Attack selected patches with attention-aware objectives. |
| Patch-wise perturbation experiments by Gu et al. | A study of patch attacks; the survey does not supply another distinct attack acronym here. |

Section VI also tests FGSM, PGD, momentum iterative methods (MIM/MI-FGSM) and
C&W on Transformers. MIM denotes a momentum-iterative family and overlaps MI-FGSM
terminology; do not assume every named label is an independent method.
Grad-CAM is an explanation tool, not an attack. PatchVeto, Smoothed ViTs,
Derandomized Smoothing, Temperature Scaling and VeinGuard are defenses, not attacks.

### Additional named attack in the defense discussion

Section V.A mentions ROA, Rectangular Occlusion Attack, as part of evaluating/training
Defense against Occlusion Attacks (DOA). Occlusion covers or replaces part of an input.
“Oracle-invariant” and “oracle-sensitive” elsewhere in Section V describe whether
a reference/human interpretation changes; they are not standalone named algorithms.

## 9. Next reading and open project decisions

Read [Otter Weather: Skillful and Computationally Efficient Medium-Range Weather Forecasting.pdf](<../Otter Weather: Skillful and Computationally Efficient Medium-Range Weather Forecasting.pdf>) next: Sections 2–3, the Rollout Fine-tuning paragraph
and Appendix B.3. Write down the input fields/history, output fields/time, loss,
normalization and repeated-forecast training. Use [OTTER_PREPARATION.md](../../OTTER_PREPARATION.md)
for the checked explanation. Then read the predictability study's methods to compare
input changes that increase error with changes that reduce it.

Questions to keep open for future experiments:

- What physical evidence should justify temperature budgets and spatial restrictions?
- Should success mean increased whole-map RMSE, a regional cold bias, or a missed event?
- Which input fields/history times may change, and what model access is assumed?
- How does attack damage grow across repeated forecasts?
- Can a defense improve attacked forecasts without worsening clean forecasts?

These decisions are more useful now than implementing every method in the catalogue.

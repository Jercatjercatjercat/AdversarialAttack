# Preparing to move from toy weather to Otter

The toy project has established the input-gradient and attack comparison idea.
Our next practical milestone is one correct forecast from a trained Otter model,
followed by one temperature-input gradient. This document describes preparation;
Otter has not been trained or run in this workspace.

## How Otter learns

The local [Otter paper](PaperReadings/Otter.pdf) is the deterministic ICLR 2026
paper, “Otter Weather: Highly Skillful Medium-Range Weather Forecasting on a
Single GPU.” These details refer to that paper and the saved `otter-main/` code,
not automatically to every later Otter model variant.

1. **Give it four recent global weather states.** Each state is six hours apart:
   current weather and weather 6, 12 and 18 hours earlier. The grid has 121 latitude
   rows and 240 longitude columns, at about 1.5° spacing. A weather state includes
   temperature, moisture and winds at several heights, as well as surface variables.
   The default input has 87 fields: 82 changing weather fields and five fixed
   geographical fields, including land/sea and terrain information. Pressure levels
   are standard air-pressure surfaces used to describe different atmospheric heights.
2. **Learn a six-hour correction.** Rather than recreating the future from scratch,
   Otter predicts a change and adds it to the latest state. It also receives clock
   information. A Swin Transformer processes map patches using attention: learned
   comparisons that decide which nearby information to use. Shifting its local
   windows between layers allows information to pass between neighbouring regions.
3. **Update weights using forecast error.** The initial stage trains one six-hour
   step. The default configuration uses 20 epochs, meaning 20 passes through the
   training examples, with ERA5 training dates covering 1979–2019. The loss is
   weighted root mean squared error: cells and variables do not all count equally.
   Latitude weights reflect grid-cell area; other weights account for variable
   scales and pressure levels. The Muon optimizer is the rule used to update large
   matrix weights (tables of numbers); the implementation also uses AdamW for other parameters. It
   accumulates gradients from four small batches before a weight update, giving an
   effective batch of 16 while limiting memory use. The learning rate, or update
   size, starts small, increases briefly and then decreases.
4. **Continue training on its own repeated predictions.** A later stage called
   rollout fine-tuning feeds predictions back as inputs and trains forecasts up to
   three days ahead. This addresses errors accumulating across repeated steps.
   At evaluation, the same repetition can generate longer forecasts. This is
   autoregressive forecasting; our current toy experiments only predict one step.

Sources: [official Otter repository](https://github.com/cambridge-mlg/otter),
[base configuration](otter-main/otter/experiment/config/base.yaml),
[rollout configuration](otter-main/otter/experiment/config/rollout_finetuning.yaml),
and [forecasting model](otter-main/otter/models/forecasting_model.py).

The input tensor uses a different order from the toy CNN:
`[batch, latitude, longitude, history_time, field]`. For the base configuration,
one complete input is `[1, 121, 240, 4, 87]`. The history is stored latest first.
One-step output has shape `[1, 121, 240, 1, 82]`. We must match the checkpoint's
field ordering, normalization and configuration, not just its tensor dimensions.

## Why predict a change? Evidence versus inference

The confirmed design is `predicted future = latest state + learned correction`.
Section 2 of the saved paper describes predicting a difference from the latest
state. In [single_time_forward](otter-main/otter/models/forecasting_model.py), the
code takes the latest state's corresponding fields and adds them to the network's
output. The calculation uses normalized fields internally before converting the
forecast back to physical units.

My interpretation is that this gives learning a useful starting structure: carry
the current map forward and learn the adjustments. For example, if temperature is
280 K now and 282 K six hours later, the correction is +2 K. A direct-output model
must also learn how to preserve parts of the input that barely change. With an
explicit addition, that copying route already exists. If the learned correction
is zero, the forecast is persistence, or “predict no change.” This does not mean
Otter actually initializes its correction at zero; I have not established that.

Evidence supporting this interpretation, rather than proving it:

- The paper and code establish that Otter uses the addition described above.
- In our ERA5 test set, carrying the current temperature forward gives RMSE
  2.624 K without learning, compared with clean CNN mean RMSE 2.494 K. Persistence
  is therefore an important reference that the learned corrections should improve
  on; these toy scores do not measure Otter's benefit from predicting changes.
- The identity `future - current = change` makes the two forecasting targets
  mathematically related. Choosing the correction changes how the model represents
  the task; it does not add information or guarantee better optimization.

I did not find a controlled comparison in the saved Otter paper that changes ONLY
direct-output prediction versus correction prediction. Such a comparison, with
matched data and training settings, would be needed to establish its causal benefit.
Normalization already removes the large Kelvin offset, so “2 is smaller than 282”
alone is not evidence that the correction will be easier to learn. The more useful
intuition is the explicit route for carrying the current state forward.

## What specifically addresses repeated-forecast errors?

This is explicitly discussed and trained, not left unmentioned. A model trained
only on clean observed inputs later receives its own imperfect predictions when
forecasting repeatedly. That change in inputs can make errors accumulate.

Otter uses **rollout fine-tuning**, continued weight training on a chain of its
own predictions:

```text
Real input history → prediction at +6h
Updated history including that prediction → prediction at +12h
Updated history including those predictions → prediction at +18h → ...

Compare predictions at the included times with true future states.
Use that combined error to update the model weights.
```

The earlier predictions are used as later inputs rather than being replaced by
the true future states. The real future states still provide the targets for
calculating error. The code averages the per-time losses across included times
and backpropagates through the forecast calculation.

Appendix B.3 reports a **curriculum**, meaning the training task becomes harder
gradually: start at 24 hours, or four six-hour steps, and extend to 72 hours, or
twelve steps. It reports one further training epoch using data from 2007 onward
for the base model. The saved configuration likewise sets the initial horizon to
four steps and the maximum to twelve, with a small learning rate of 0.000002.
The model weights are updated in this stage; it is not an input attack.

The purpose is to reduce accumulated error, not guarantee that it disappears.
The paper also notes that this training can make long-range forecasts overly
smooth, so it limits the training horizon to three days. The rollout training
configuration and the final checkpoint must be recorded when comparing attacks.

Evidence: the saved [paper](PaperReadings/Otter.pdf), Section 3's Rollout
Fine-tuning paragraph and Appendix B.3;
[rollout training settings](otter-main/otter/experiment/config/rollout_finetuning.yaml);
[history update loop](otter-main/otter/models/forecasting_model.py);
and [loss averaging](otter-main/otter/models/losses/loss_fn.py).

## Can we reproduce full training on this Mac?

The machine checked on 11 October 2026 has an **Apple M4 Max with 36 GiB of shared
memory**. Shared memory is used by the CPU, GPU and other applications together.
It is not equivalent to the same amount of dedicated NVIDIA GPU memory.

The local paper reports training in under three A100-equivalent days. An A100 is
a high-performance NVIDIA GPU; an A100-day means 24 hours of equivalent GPU work,
not 24 hours on any computer. The authors used several types of NVIDIA GPU in
single-GPU runs and report **46.5 GB peak GPU memory** for the base training setup
(Appendix B.3). This already exceeds this Mac's total shared-memory capacity.

The saved forecast evaluation script explicitly uses CUDA, NVIDIA's GPU system,
and the attention code uses the `xformers` attention library. Apple's GPU uses
PyTorch's MPS backend, a different execution system. The existing code is therefore
not a ready-to-run Mac training pipeline. Reducing batch size alone does not settle
model memory, attention compatibility or training speed.

| Work | Assessment here |
|---|---|
| Read code, inspect shapes, prepare a few ERA5 examples | Suitable locally |
| Run the existing toy CNN and attack experiments | Already demonstrated locally |
| Train a much smaller Otter-style model | A possible separate exercise after testing compatible attention and memory use; not a reproduction of the published model |
| Run a pretrained full Otter forecast/input gradient | Needs a checkpoint and a measured memory/compatibility check; not yet demonstrated on this Mac |
| Reproduce the published full training | Not practical on this Mac as configured; plan suitable NVIDIA GPU access |

The official full-resolution data preparation workflow also describes about 660 GB
for its larger dataset. The local filesystem currently reports about 647 GiB free,
but units differ and conversion/temporary files need additional space. We have not
downloaded that archive. A few complete global cases suffice for a first forecast;
the whole multi-decade training dataset is not required for pretrained-model attacks.
See the [official data preparation instructions](https://github.com/cambridge-mlg/otter#readme).

## First practical experiment

1. Obtain a **checkpoint**, the trained model weights, plus its exact configuration.
   No `.pt`, `.pth`, `.ckpt` or `.safetensors` checkpoint was found in the local
   Otter folder. The official README describes training and evaluation but does
   not provide a direct pretrained checkpoint download. Availability still needs
   to be established; random weights cannot test a trained forecaster's robustness.
2. Prepare one complete four-time input and its true six-hour future. Our saved
   European temperature crop cannot replace the full input expected by the checkpoint.
3. Measure the clean forecast and compare temperature error with predicting no
   change on the same case. The published results do not guarantee our new setup
   is correct or that every case improves on persistence.
4. Freeze the weights, keep gradient tracking enabled, and calculate how temperature
   input changes affect temperature forecast error. The existing forecast-saving
   script disables gradients, so it cannot be used unchanged for this check.
5. Change only the latest input's T2m, keeping other fields, older frames and clock
   information fixed. Compare random noise, FGSM and PGD with a maximum-cell
   limit in Kelvin. Extend to more cases, L2 budgets and longer lead times after
   the one-step setup is verified.

Attacks need intermediate calculations retained for differentiation, so their memory
use can exceed a forecast run that disables gradients. We should measure it rather
than infer it from inference speed. Here “robustness” means how much forecast error
changes under a clearly specified input disturbance; it does not establish that
the disturbed inputs are physically possible weather states.

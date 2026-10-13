# Reading notes: Otter Weather

Architecture questions recorded on 13 October 2026. These notes refer to the saved
[deterministic ICLR 2026 paper](../Otter.pdf), especially Figure 2 and Section 3.2,
and the [local model configuration](../../otter-main/otter/experiment/config/base.yaml).
Later Otter publications also discuss probabilistic variants; keep the version explicit.

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

I did not find the name U-Cast in the saved Otter PDF. A newer Otter paper discusses
it as another weather model. If a particular passage says something different,
check which paper version and exact term are being read.

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
Our [toy CNN](../../toy_weather/models/simple_cnn.py) is the local small example.

Next reading goal: follow Figure 2 and identify input, patch features, encoder,
bottleneck, decoder and the final forecast. Leave attention equations and optimizer
details for later questions. The [preparation note](../../OTTER_PREPARATION.md)
already records the correction-prediction and repeated-forecast training explanations.

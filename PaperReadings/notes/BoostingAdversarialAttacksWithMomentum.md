# Boosting Adversarial Attacks with Momentum

Notes: 14 and 20 October 2026; March 2018 v3.

## Method

- **Momentum memory**, not autoregressive weather forecasting: combine the previous accumulator with the current normalized input gradient.

```text
g = μg + gradient / sum(abs(gradient))
x = x + α sign(g)
```

- L1 normalization balances gradient contributions; the main attack uses an **L∞ disturbance budget**, with α = ε / steps.
- μ controls memory: 0 gives ordinary iterative sign updates; 1 accumulates without decay. The paper commonly uses μ = 1, so memory need not decay.

## Important findings

- **Transferability:** attacks are constructed white-box and tested on unseen models without their gradients. Iterating more can overfit the source model; momentum improves transfer in the reported classifier experiments.
- **Ensemble attacks:** combine several source models; the paper compares combination strategies and favours combining logits. Weather needs its own forecast-loss formulation.
- **Extensions:** targeted objectives and L2 momentum attacks. Classification results do not guarantee weather robustness findings.

## Our experiments

- MI-FGSM is implemented with forecast loss and fixed CNN weights. The selected one-step run equals FGSM, so it demonstrates no momentum advantage.
- Sparse top-k attacks are a separate idea, not this paper's contribution.

## Bookmarks

- Section 3.1 / Algorithm 1: update; Section 3.2 / Algorithm 2: ensembles; Section 3.3: extensions.

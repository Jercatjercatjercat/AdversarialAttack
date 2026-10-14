# Boosting Adversarial Attacks with Momentum

Notes recorded on 14 October 2026. The saved paper is arXiv:1710.06081v3,
22 March 2018, by Dong and colleagues.

[Local PDF](../BoostingAdversarialAttacksWithMomentum.pdf) ·
[Paper reference](https://arxiv.org/abs/1710.06081)

## The idea behind MI-FGSM

FGSM makes one gradient-sign update. An ordinary iterative sign attack recalculates
the gradient at each step and uses the latest direction. Momentum Iterative Fast
Gradient Sign Method (MI-FGSM) keeps a running memory of gradient directions, so
one new direction does not immediately erase what earlier steps suggested.

Section 3.1, Algorithm 1 and Equations 6–7 describe the method. Starting with
momentum g=0, each step calculates the loss gradient at the changed input, divides
it by its L1 magnitude, combines it with the previous momentum, and updates the
input using the sign of that combined value:

```text
g = mu * g + gradient / sum(abs(gradient))
changed_input = changed_input + step_size * sign(g)
```

The decay factor mu controls the memory. Zero removes the memory; one accumulates
normalised directions without decay. A zero gradient needs safe handling when
implementing the division. The L1 division normalises each gradient's contribution;
it does NOT make MI-FGSM an L1-constrained attack. The main algorithm constrains
the final disturbance with Linf and uses step size epsilon / number_of_steps.
A weather implementation would explicitly enforce the physical input budget.

The authors report improved transfer between image classifiers. Momentum does not
guarantee a stronger attack on every model or weather case. Section 3.3 also gives
an L2 momentum extension, using a unit-L2 momentum direction rather than its sign.

For our forecast adaptation, the classification loss would be replaced by a stated
forecast-error objective. Trained weights and true future maps stay fixed while
inputs change. The first understandable comparison would alter current T2m only,
with the same limits, test examples and forecast objective as FGSM/PGD. MI-FGSM is now implemented in the second iteration, with the
[attack-setting experiment](../../toy_weather/ATTACK_EXPERIMENTS.md) documenting
its validation sweep and forecast-loss adaptation.

## Sparse attacks and thresholding: a separate proposed experiment

Sparse describes the disturbance: only a few input entries are changed. It is not
one unique algorithm and is not the same idea as momentum. For current T2m there
are 256 cells; an L0 budget of k=4 permits at most four changed temperature entries.
L0 is a nonzero-entry count, conventionally called a norm. It needs a separate
amplitude cap, such as 0.5 K per changed cell. Otherwise four cells could change
by arbitrarily large amounts.

A proposed educational approach calculates gradients, chooses k sensitive cells,
updates them and retains at most k nonzero changes, subject to the amplitude cap.
Choosing cells by absolute gradient is a local heuristic, not a guarantee that the
chosen set gives the best finite attack. A random-k-cell baseline with the same
amplitude cap separates deliberate location choice from sparse random noise.

Thresholding is one way to obtain sparsity. A fixed threshold on gradient magnitude
keeps cells exceeding that value, but the number retained varies by example.
Top-k selection instead keeps exactly k candidates (unless fewer are nonzero),
so it directly enforces a changed-cell-count budget.

For an iterative L0 attack, the total accumulated change must stay within k cells.
Choosing k new cells on each step could eventually change many more than k. One
feasible projection is to clip candidate changes to the per-cell cap and keep only
the k largest absolute changes, resetting every other entry to its ORIGINAL value.
This is hard thresholding of the change map, distinct from selecting cells using
the gradient. It provides a feasible sparse candidate, not a globally optimal attack.

L1 projection can also zero small entries using soft thresholding; an L1 budget
controls total absolute magnitude, not an exact number of changed cells. These L0/L1 educational methods are now implemented; their
[setup and exploratory evidence](../../toy_weather/ATTACK_EXPERIMENTS.md) distinguish
our heuristics from the classification methods in the paper.

"""Demeter recipe with three corrections to the released one.

Measured against the released recipe on the prepared 67/11 soybean split:

1. **Colour units.** The prepared samples store RGB in [0, 1], but the inherited
   Pointcept colour transforms are written for [0, 255]. `ChromaticJitter` added
   noise with std 9.39 on top of a signal with std 0.185 (50x the signal), and at
   test time `NormalizeColor` left a near-constant -0.997 (std 0.001). Three of
   the nine input channels were noise while training and a constant while
   testing. `color_scale=255.0` hands those transforms the range they expect.

2. **Rotation axis: withdrawn.** An earlier version of this file rotated about z,
   on the evidence that every released plant came out with its main stem on +Z.
   The released checkpoint settled it the other way -- it scores 0.97 on +X input
   against 0.90 on +Z -- so the recipe's rotation about x was right all along and
   `prepare_data.py` was the thing that had drifted. That is fixed at the source
   now (`--stem-axis x`, the default), and this file inherits config.py's
   augmentation unchanged.

3. **Boundary loss.** `dist_head` is a bare `nn.Linear(..., 1)`, so its output is
   a logit; the released recipe clipped it to [0, 1] and regressed it with MSE,
   which zeroes the gradient wherever the logit leaves [0, 1]. Trained as a logit
   with binary cross-entropy against the same soft target instead.

Everything else -- architecture, optimizer, schedule, grid size, loss weights --
is inherited unchanged, so a run of this against `config.py` isolates the three
changes. The released checkpoint is not compatible with fix 1.
"""

_base_ = ["./config.py"]

grid_size = 0.02

# Fix 3: the boundary head is a logit, so train it with cross-entropy.
#
# Swapping the loss is not scale-neutral. The released weight of 15 was tuned for
# MSE, whose term starts at mean(target^2) * 15 = 0.62 against a semantic
# cross-entropy of ln(5) = 1.61. Binary cross-entropy starts at ln(2) = 0.69
# whatever the target, so the same weight would start at 10.4 -- 17x the released
# boundary term and 6.5x the semantic term, which flattens the semantic head (a
# 2-epoch trial at weight 15 reached 0.058 accuracy). A weight of 1.0 restores
# roughly the released balance; 0.89 would match it exactly.
#
# ``dist_prior`` seeds the head's final bias at logit(mean target). Without it a
# 100-epoch run left the head at a constant logit ~0.1 (sigmoid 0.526) with
# correlation -0.001 against the target: AdamW moves a parameter by about the
# learning rate per step, so 1600 steps at a 5e-4 peak cannot travel the -2.07
# that the target's own mean requires. Raising the loss weight does not help,
# because Adam is invariant to gradient scale.
model = dict(dist_loss="bce", dist_loss_weight=1.0, dist_prior=0.1124)

data = dict(
    train=dict(color_scale=255.0),   # fix 1
    val=None,
    test=dict(color_scale=255.0),   # fix 1
)

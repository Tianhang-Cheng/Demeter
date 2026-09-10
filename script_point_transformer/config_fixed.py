"""Demeter recipe with three corrections to the released one.

Measured against the released recipe on the prepared 67/11 soybean split:

1. **Colour units.** The prepared samples store RGB in [0, 1], but the inherited
   Pointcept colour transforms are written for [0, 255]. `ChromaticJitter` added
   noise with std 9.39 on top of a signal with std 0.185 (50x the signal), and at
   test time `NormalizeColor` left a near-constant -0.997 (std 0.001). Three of
   the nine input channels were noise while training and a constant while
   testing. `color_scale=255.0` hands those transforms the range they expect.

2. **Rotation axis.** Preprocessing aligns every main stem to +Z (verified on all
   78 released plants), so the released +-180 degree rotation about x tipped
   plants sideways or upside-down, while the azimuth the data really is invariant
   to got only +-2.8 degrees. The big rotation moves to z and the small jitter to
   x and y, which is the convention the transform list was inherited under.

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
    train=dict(
        color_scale=255.0,  # fix 1
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
            # Fix 2: the main stem is along +Z, so z carries the free azimuth and
            # x / y only get the small jitter that models a slightly tilted scan.
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
    ),
    val=None,
    test=dict(
        color_scale=255.0,  # fix 1
        test_cfg=dict(
            # Fix 2: average test-time rotations about the plant's own up axis.
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[angle], axis="z",
                      center=[0, 0, 0], p=1)]
                for angle in (0, 1 / 2, 1, 3 / 2)
            ] + [
                [dict(type="RandomRotateTargetAngle", angle=[angle], axis="z",
                      center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[scale, scale])]
                for scale in (0.95, 1.05) for angle in (0, 1 / 2, 1, 3 / 2)
            ] + [
                [dict(type="RandomFlip", p=1)],
            ],
        ),
    ),
)

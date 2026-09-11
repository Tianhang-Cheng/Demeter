"""config_fixed plus a PointGroup-style offset head, for testing only.

The boundary-score route to instances saturates: on the held-out plants DBSCAN
after deleting junction points reaches coverage 0.466, a kNN graph cut reaches
0.511, and substituting a *perfect* boundary score reaches only 0.537. The limit
is that organs touch, so no post-hoc grouping of xyz separates them.

This adds a third head predicting the vector from each point to its own organ's
centroid, so instances can be clustered in the shifted space where touching
organs pull apart. Not committed: it is an experiment, and it has not been shown
to beat the graph cut.

Ordering trap: ``ShufflePoint`` permutes coord/segment/instance but *not*
``instance_centroid``, so ``PlantInstanceCentroid`` has to run after it or the
target silently misaligns with the coordinates. ``GridSample`` also drops
``instance`` unless it is named in ``keys``.
"""

_base_ = ["./config_fixed.py"]

grid_size = 0.02

model = dict(
    backbone=dict(predict_offset=True),
    # The target is centroid - coord in unit-radius coordinates, so an organ-sized
    # vector is around 0.1-0.5 and the summed L1 lands near the semantic
    # cross-entropy at weight 1.
    offset_loss_weight=1.0,
    # Direction matters more than magnitude here: the L1-only head reproduced the
    # target's length (0.108 against 0.119) but its 0.082 error was 61% of the
    # 0.133 median gap between neighbouring organ centroids, so organs did not
    # separate in the shifted space. Oracle offsets reach coverage 0.972 there.
    offset_direction_weight=1.0,
)

data = dict(
    train=dict(
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
            dict(type="RandomRotate", angle=[-1, 1], axis="x", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="z", p=0.5),
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
                keys=("coord", "color", "normal", "segment", "instance"),
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ShufflePoint"),
            # After the shuffle, and after every transform that moves coord.
            dict(type="PlantInstanceCentroid"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment", "instance", "instance_centroid"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
    ),
    val=None,
)

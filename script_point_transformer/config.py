"""Demeter training recipe; all paths can be overridden by run.py."""

_base_ = ["../third_party/PointTransformer_V3/Pointcept/configs/soybean3d/custom3.py"]

seed = 2025
evaluate = False
num_worker = 4
save_path = "outputs/point_transformer/soybean"
data_root = "data/point_transformer/soybean"
test_split = "test"
data = dict(
    train=dict(data_root=data_root, split="train"),
    val=None,
    test=dict(data_root=data_root, split="test"),
)

# No validation set or checkpoint selection on test data. Test the final epoch.
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=True),
]

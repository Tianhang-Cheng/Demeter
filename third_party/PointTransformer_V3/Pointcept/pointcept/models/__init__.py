import warnings as _warnings

from .builder import build_model
from .default import DefaultSegmentor, DefaultClassifier


def _optional_import(module_name: str) -> None:
    """Import an optional backbone/head submodule, tolerating missing deps.

    Some backbones (e.g. ``sparse_unet`` needs ``spconv``, ``point_group`` needs
    ``pointgroup_ops``) rely on heavy CUDA extensions that may not be installed.
    Importing this package must still register the models that *are* available
    (e.g. ``DefaultCustom`` + ``point_transformer_v2``); a single missing
    dependency should not abort the whole ``pointcept.models`` import and leave
    the model registry empty.

    Args:
        module_name: Submodule name relative to ``pointcept.models`` to import.

    Returns:
        None. Registration happens as an import side effect.
    """
    import importlib

    try:
        importlib.import_module(f".{module_name}", __name__)
    except ImportError as exc:  # missing optional dependency
        _warnings.warn(
            f"pointcept.models: skipping optional backbone '{module_name}' "
            f"({exc}). Install its dependency if you need it.",
            stacklevel=2,
        )


# Backbones (optional ones are imported defensively)
_optional_import("sparse_unet")  # requires spconv
_optional_import("point_transformer")
from .point_transformer_v2 import *  # required by PT-v2m2-custom / DefaultCustom

# from .stratified_transformer import *
# from .spvcnn import *
# from .octformer import *
# from .swin3d import *

# Semantic Segmentation
_optional_import("context_aware_classifier")

# Instance Segmentation
_optional_import("point_group")  # requires pointgroup_ops

# Pretraining
_optional_import("masked_scene_contrast")
_optional_import("point_prompt_training")

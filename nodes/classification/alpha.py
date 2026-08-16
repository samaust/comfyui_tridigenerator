import comfy.model_management as model_management

from ...utils.classification.alpha import add_classification_alpha
from ...utils.cameras.intrinsics import validate_vipe_intrinsics


class TDGenClassificationAlpha:
    """Replace mesh alpha classifications from an optional mask sequence."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "meshes": ("PYTORCH3D_MESHES",),
                "height": ("INT", {"min": 1, "step": 1}),
                "width": ("INT", {"min": 1, "step": 1}),
                "intrinsics": ("VIPE_INTRINSICS",),
            },
            "optional": {"masks": ("MASK",)},
        }

    RETURN_TYPES = ("PYTORCH3D_MESHES", "INT", "INT", "VIPE_INTRINSICS", "IMAGE")
    RETURN_NAMES = ("meshes", "height", "width", "intrinsics", "previews")
    FUNCTION = "run"
    CATEGORY = "TDGen/Classification/Alpha"
    DESCRIPTION = "Set per-vertex alpha classification from masks."

    def run(self, meshes, height, width, intrinsics, masks=None):
        validate_vipe_intrinsics(intrinsics, len(meshes))
        classified_meshes, previews = add_classification_alpha(
            meshes=meshes,
            masks=masks,
            height=height,
            width=width,
            device=model_management.get_torch_device(),
        )
        return (classified_meshes, height, width, intrinsics, previews)

import comfy.model_management as model_management

from ...utils.classification.alpha import add_classification_alpha


class TDGenClassificationAlpha:
    """Replace mesh alpha classifications from an optional mask sequence."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "meshes": ("PYTORCH3D_MESHES",),
                "height": ("INT", {"min": 1, "step": 1}),
                "width": ("INT", {"min": 1, "step": 1}),
                "fov_deg": (
                    "INT",
                    {"default": 74, "min": 1, "max": 120, "step": 1},
                ),
            },
            "optional": {"masks": ("MASK",)},
        }

    RETURN_TYPES = ("PYTORCH3D_MESHES", "INT", "INT", "INT", "IMAGE")
    RETURN_NAMES = ("meshes", "height", "width", "fov_deg", "previews")
    FUNCTION = "run"
    CATEGORY = "TDGen/Classification/Alpha"
    DESCRIPTION = "Set per-vertex alpha classification from masks."

    def run(self, meshes, height, width, fov_deg, masks=None):
        classified_meshes, previews = add_classification_alpha(
            meshes=meshes,
            masks=masks,
            height=height,
            width=width,
            device=model_management.get_torch_device(),
        )
        return (classified_meshes, height, width, fov_deg, previews)

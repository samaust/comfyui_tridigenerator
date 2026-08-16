import comfy.model_management as model_management

from ...utils.classification.visible import add_classification_visible


class TDGenClassificationVisible:
    """Classify mesh vertices by face orientation relative to the camera."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "meshes": ("PYTORCH3D_MESHES",),
                "height": ("INT", {"min": 1, "step": 1}),
                "width": ("INT", {"min": 1, "step": 1}),
                "intrinsics": ("VIPE_INTRINSICS",),
                "znear": ("FLOAT", {"default": 0.1, "min": 0.001, "step": 0.001}),
                "zfar": ("FLOAT", {"default": 100.0, "min": 0.002, "step": 0.001}),
                "threshold": ("FLOAT", {"default": 0.1, "min": -1.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("PYTORCH3D_MESHES", "INT", "INT", "VIPE_INTRINSICS", "IMAGE")
    RETURN_NAMES = ("meshes", "height", "width", "intrinsics", "previews")
    FUNCTION = "run"
    CATEGORY = "TDGen/Classification/Visible"
    DESCRIPTION = "Set per-vertex visibility classification from mesh face angles."

    def run(self, meshes, height, width, intrinsics, znear, zfar, threshold):
        classified_meshes, previews = add_classification_visible(
            meshes=meshes,
            height=height,
            width=width,
            intrinsics=intrinsics,
            znear=znear,
            zfar=zfar,
            threshold=threshold,
            device=model_management.get_torch_device(),
        )
        return (classified_meshes, height, width, intrinsics, previews)

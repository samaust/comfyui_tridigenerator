import comfy.model_management as model_management

from ...utils.meshes.generation import create_meshes
from ...utils.cameras.intrinsics import validate_vipe_intrinsics


class TDGenMeshesCreateMeshes:
    """Create a sequence of PyTorch3D meshes from a point cloud."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pcd": ("PCD",),
                "height": ("INT", {"min": 2, "step": 1}),
                "width": ("INT", {"min": 2, "step": 1}),
                "intrinsics": ("VIPE_INTRINSICS",),
            }
        }

    RETURN_TYPES = ("PYTORCH3D_MESHES", "INT", "INT", "VIPE_INTRINSICS")
    RETURN_NAMES = ("meshes", "height", "width", "intrinsics")
    FUNCTION = "run"
    CATEGORY = "TDGen/Meshes/Generation"
    DESCRIPTION = "Create grid meshes with RGB, alpha, and visibility vertex features."

    def run(self, pcd, height, width, intrinsics):
        validate_vipe_intrinsics(intrinsics, len(pcd[0]))
        meshes = create_meshes(
            pcd=pcd,
            height=height,
            width=width,
            device=model_management.get_torch_device(),
        )
        return (meshes, height, width, intrinsics)

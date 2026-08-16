import comfy.model_management as model_management

from ...utils.meshes.generation import create_meshes


class TDGenMeshesCreateMeshes:
    """Create a sequence of PyTorch3D meshes from a point cloud."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pcd": ("PCD",),
                "height": ("INT", {"min": 2, "step": 1}),
                "width": ("INT", {"min": 2, "step": 1}),
                "fov_deg": (
                    "INT",
                    {
                        "default": 74,
                        "min": 1,
                        "max": 120,
                        "step": 1,
                        "tooltip": "Horizontal field of view in degrees",
                    },
                ),
            }
        }

    RETURN_TYPES = ("PYTORCH3D_MESHES", "INT", "INT", "INT")
    RETURN_NAMES = ("meshes", "height", "width", "fov_deg")
    FUNCTION = "run"
    CATEGORY = "TDGen/Meshes/Generation"
    DESCRIPTION = "Create grid meshes with RGB, alpha, and visibility vertex features."

    def run(self, pcd, height, width, fov_deg):
        meshes = create_meshes(
            pcd=pcd,
            height=height,
            width=width,
            device=model_management.get_torch_device(),
        )
        return (meshes, height, width, fov_deg)

import comfy.model_management as model_management

from ...utils.rendering.textured_mesh import render_textured_mesh


class TDGenRenderTexturedMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "masks": ("MASK",),
            "depths": ("IMAGE",),
            "json_data": ("STRING", {"forceInput": True}),
            "displacement_x": (
                "FLOAT",
                {"default": 0.0, "min": -100000.0, "max": 100000.0, "step": 0.01},
            ),
            "displacement_y": (
                "FLOAT",
                {"default": 0.0, "min": -100000.0, "max": 100000.0, "step": 0.01},
            ),
            "displacement_z": (
                "FLOAT",
                {"default": 0.0, "min": -100000.0, "max": 100000.0, "step": 0.01},
            ),
            "yaw": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
            "pitch": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
            "roll": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "TDGen/Rendering"
    DESCRIPTION = "Render ViPE RGB-D frames as pose-aware textured meshes using per-frame pinhole intrinsics."

    def run(self, images, masks, depths, json_data, displacement_x, displacement_y, displacement_z, yaw, pitch, roll):
        rendered = render_textured_mesh(
            images,
            masks,
            depths,
            json_data,
            displacement=(displacement_x, displacement_y, displacement_z),
            rotation=(yaw, pitch, roll),
            device=model_management.get_torch_device(),
        )
        return (rendered,)

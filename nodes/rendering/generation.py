import comfy.model_management as model_management

from ...utils.rendering.textured_mesh import CAMERA_ORIGINS, render_textured_mesh


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
                {"default": 0.0, "min": -100000.0, "max": 100000.0, "step": 0.001},
            ),
            "displacement_y": (
                "FLOAT",
                {"default": 0.0, "min": -100000.0, "max": 100000.0, "step": 0.001},
            ),
            "displacement_z": (
                "FLOAT",
                {"default": 0.0, "min": -100000.0, "max": 100000.0, "step": 0.001},
            ),
            "yaw": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
            "pitch": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
            "roll": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
            "camera_origin": (
                CAMERA_ORIGINS,
                {
                    "default": "vipe_world",
                    "tooltip": "Fix the virtual camera at the first frame or ViPE world, or follow the current frame.",
                },
            ),
            "bg_color": (
                "STRING",
                {
                    "default": "#000000",
                    "tooltip": "Background color as a six-digit hex code, for example #FFFFFF.",
                },
            ),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "TDGen/Rendering"
    DESCRIPTION = "Render ViPE RGB-D meshes from a fixed origin or each frame's estimated camera pose."

    def run(
        self,
        images,
        masks,
        depths,
        json_data,
        displacement_x,
        displacement_y,
        displacement_z,
        yaw,
        pitch,
        roll,
        camera_origin="vipe_world",
        bg_color="#000000",
    ):
        rendered = render_textured_mesh(
            images,
            masks,
            depths,
            json_data,
            displacement=(displacement_x, displacement_y, displacement_z),
            rotation=(yaw, pitch, roll),
            device=model_management.get_torch_device(),
            camera_origin=camera_origin,
            bg_color=bg_color,
        )
        return (rendered,)

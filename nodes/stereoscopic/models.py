import torch
import comfy.model_management as model_management

from ...utils.stereoscopic.models import load_stereocrafter_pipeline


class TDGenStereoscopicLoadModelSvdStereoCrafter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt": (
                    ["stabilityai/stable-video-diffusion-img2vid-xt-1-1"],
                    {"default": "stabilityai/stable-video-diffusion-img2vid-xt-1-1"},
                ),
                "unet": (["TencentARC/StereoCrafter"], {"default": "TencentARC/StereoCrafter"}),
            },
            "optional": {"precision": (["fp16"], {"default": "fp16"})},
        }

    RETURN_TYPES = ("SVD_STEREOCRAFTER_PIPELINE",)
    RETURN_NAMES = ("svdsc_pipeline",)
    FUNCTION = "run"
    CATEGORY = "TDGen/Stereoscopic/Models"
    DESCRIPTION = "Download and load the SVD StereoCrafter pipeline."

    def run(self, ckpt, unet, precision="fp16"):
        if precision != "fp16":
            raise ValueError("only fp16 StereoCrafter pipelines are supported")
        pipeline = load_stereocrafter_pipeline(
            checkpoint=ckpt,
            unet=unet,
            device=model_management.get_torch_device(),
            dtype=torch.float16,
        )
        return (pipeline,)

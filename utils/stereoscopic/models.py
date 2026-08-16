import torch
from stereocrafter.inference.inpainting import load_models

from ..datatypes import Device


def load_stereocrafter_pipeline(
    checkpoint: str,
    unet: str,
    device: Device,
    dtype: torch.dtype,
):
    if dtype != torch.float16:
        raise ValueError("the StereoCrafter python_package loader supports fp16 only")
    pipeline = load_models(
        pre_trained_path=checkpoint,
        unet_path=unet,
    )
    return pipeline.to(device)

from contextlib import nullcontext

import numpy as np
import torch

from ..datatypes import Device


def _images_to_uint8(images: torch.Tensor) -> np.ndarray:
    if images.ndim != 4 or images.shape[-1] not in (3, 4):
        raise ValueError("images must have shape [frames, height, width, 3|4]")
    images = images[..., :3].detach().cpu().clamp(0.0, 1.0)
    return (images.numpy() * 255.0).round().astype(np.uint8)


def dav2_infer_metric_depths(
    images,
    model,
    input_size=518,
    dtype=torch.float32,
    device: Device = "cpu",
) -> torch.Tensor:
    """Estimate metric depth for each input image."""
    images_uint8 = _images_to_uint8(images)
    device_type = device.type if isinstance(device, torch.device) else str(device)
    autocast = (
        torch.autocast(device_type=device_type, dtype=dtype)
        if dtype != torch.float32 and device_type != "cpu"
        else nullcontext()
    )
    with autocast:
        depths = [model.infer_image(image, input_size) for image in images_uint8]
    if not depths:
        raise ValueError("images must contain at least one frame")
    return torch.from_numpy(np.stack(depths).astype(np.float32, copy=False))


def vda_infer_norm_inv_depths(
    images,
    model,
    enable_autocast=False,
    input_size=518,
    device: Device = "cpu",
) -> torch.Tensor:
    """Estimate normalized inverse depth for an image sequence."""
    images_uint8 = _images_to_uint8(images)
    device_type = device.type if isinstance(device, torch.device) else str(device)
    depths, _ = model.infer_video_depth(
        images_uint8,
        None,
        input_size=input_size,
        device=device_type,
        fp32=not enable_autocast,
    )
    return torch.from_numpy(np.asarray(depths, dtype=np.float32))

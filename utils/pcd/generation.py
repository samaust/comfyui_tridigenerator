import numpy as np
import torch

from ..cameras.intrinsics import validate_vipe_intrinsics


def compute_verts(
    images: torch.Tensor,
    metric_depths: torch.Tensor,
    intrinsics: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a CPU point-cloud sequence using a negative-Z forward convention."""
    if images.ndim != 4 or images.shape[-1] not in (3, 4):
        raise ValueError("images must have shape [frames, height, width, 3|4]")
    if metric_depths.ndim == 4 and metric_depths.shape[-1] == 1:
        metric_depths = metric_depths[..., 0]
    if metric_depths.ndim != 3:
        raise ValueError("metric_depths must have shape [frames, height, width]")
    if tuple(images.shape[:3]) != tuple(metric_depths.shape):
        raise ValueError("images and metric_depths must share frame, height, and width")
    values = validate_vipe_intrinsics(intrinsics, images.shape[0])
    colors = images.detach().cpu().to(torch.float32)
    depths = metric_depths.detach().cpu().to(torch.float32)
    if not torch.isfinite(depths).all() or (depths <= 0).any():
        raise ValueError("metric_depths must contain finite positive values")

    frame_count, height, width = depths.shape
    y_pixels, x_pixels = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    fx, fy, cx, cy = values.unbind(dim=1)
    x = (x_pixels.unsqueeze(0) - cx[:, None, None]) * depths / fx[:, None, None]
    y = -(y_pixels.unsqueeze(0) - cy[:, None, None]) * depths / fy[:, None, None]
    z = -depths
    positions = torch.stack((x, y, z), dim=-1).reshape(frame_count, -1, 3)
    return positions.numpy(), colors.reshape(frame_count, -1, colors.shape[-1]).numpy()

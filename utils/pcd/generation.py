import numpy as np
import torch


def compute_verts(
    images: torch.Tensor,
    metric_depths: torch.Tensor,
    fovx_deg: int,
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
    if not 0 < fovx_deg < 180:
        raise ValueError("fovx_deg must be between 0 and 180 degrees")

    colors = images.detach().cpu().numpy().astype(np.float32, copy=False)
    depths = metric_depths.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(depths).all() or np.any(depths <= 0):
        raise ValueError("metric_depths must contain finite positive values")

    _, height, width = depths.shape
    horizontal_half_angle = np.deg2rad(fovx_deg) / 2.0
    vertical_half_angle = np.arctan(
        (height / 2.0) * np.tan(horizontal_half_angle) / (width / 2.0)
    )
    horizontal_angles = np.linspace(
        -horizontal_half_angle, horizontal_half_angle, width, dtype=np.float32
    )
    vertical_angles = np.linspace(
        vertical_half_angle, -vertical_half_angle, height, dtype=np.float32
    )
    tan_x, tan_y = np.meshgrid(
        np.tan(horizontal_angles), np.tan(vertical_angles), indexing="xy"
    )

    x = depths * tan_x[None, ...]
    y = depths * tan_y[None, ...]
    z = -depths
    positions = np.stack((x, y, z), axis=-1).reshape(depths.shape[0], -1, 3)
    colors = colors.reshape(colors.shape[0], -1, colors.shape[-1])
    return positions.astype(np.float32, copy=False), colors.astype(np.float32, copy=False)

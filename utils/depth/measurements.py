import numpy as np
import torch

from ..cameras.intrinsics import validate_vipe_intrinsics


def measure_distances(
    metric_depths: torch.Tensor,
    intrinsics: dict,
    coordinates: np.ndarray,
) -> list[list[float]]:
    """Unproject pixel pairs from the first depth frame and measure 3D distances."""
    if metric_depths.ndim == 4 and metric_depths.shape[-1] == 1:
        metric_depths = metric_depths[..., 0]
    if metric_depths.ndim != 3 or metric_depths.shape[0] == 0:
        raise ValueError("metric_depths must have shape [frames, height, width]")

    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (2, 2):
        raise ValueError("coordinates must have shape [pairs, 2, 2]")
    if coordinates.shape[0] == 0:
        raise ValueError("at least one coordinate pair is required")

    depth = metric_depths[0].detach().cpu().numpy().astype(np.float64, copy=False)
    height, width = depth.shape
    values = validate_vipe_intrinsics(intrinsics, metric_depths.shape[0])
    fx, fy, cx, cy = values[0].to(torch.float64).tolist()

    results = []
    for pair in coordinates:
        points = []
        for x_value, y_value in pair:
            if not float(x_value).is_integer() or not float(y_value).is_integer():
                raise ValueError("coordinates must contain integer pixel positions")
            x, y = int(x_value), int(y_value)
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"coordinate ({x}, {y}) is outside the depth image")
            z = float(depth[y, x])
            if not np.isfinite(z) or z <= 0:
                raise ValueError(f"depth at ({x}, {y}) must be finite and positive")
            points.append(((x - cx) * z / fx, (cy - y) * z / fy, -z))

        start = np.asarray(points[0], dtype=np.float64)
        end = np.asarray(points[1], dtype=np.float64)
        delta = np.abs(end - start)
        results.append([*start.tolist(), *end.tolist(), *delta.tolist(), float(np.linalg.norm(delta))])
    return results

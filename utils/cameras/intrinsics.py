from typing import Any

import torch


def validate_vipe_intrinsics(intrinsics: dict[str, Any], frame_count: int | None = None) -> torch.Tensor:
    if not isinstance(intrinsics, dict):
        raise ValueError("intrinsics must be a VIPE_INTRINSICS value")
    values = intrinsics.get("values")
    indices = intrinsics.get("frame_indices")
    models = intrinsics.get("camera_models")
    if not isinstance(values, torch.Tensor) or values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("VIPE_INTRINSICS values must have shape [N,4]")
    values = values.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(values).all() or (values[:, :2] <= 0).any():
        raise ValueError("VIPE_INTRINSICS must contain finite values and positive focal lengths")
    expected = list(range(values.shape[0]))
    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().tolist()
    if list(indices if indices is not None else []) != expected:
        raise ValueError("VIPE_INTRINSICS frame indices must be contiguous and start at zero")
    if list(models if models is not None else []) != ["pinhole"] * values.shape[0]:
        raise ValueError("only pinhole VIPE_INTRINSICS are supported")
    if frame_count is not None and values.shape[0] != frame_count:
        raise ValueError("intrinsics frame count does not match the image sequence")
    return values


def slice_vipe_intrinsics(intrinsics: dict[str, Any], index: int) -> dict[str, Any]:
    values = validate_vipe_intrinsics(intrinsics)
    if index < 0 or index >= values.shape[0]:
        raise ValueError("intrinsics does not contain the selected frame")
    return {
        "values": values[index:index + 1],
        "frame_indices": torch.arange(1, dtype=torch.int64),
        "camera_models": ("pinhole",),
    }

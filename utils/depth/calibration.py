import torch


def torch_scale_and_shift_full(prediction, target, mask=None) -> tuple[float, float]:
    prediction = prediction.float()
    target = target.float().to(prediction.device)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if prediction.numel() == 0:
        raise ValueError("prediction and target must not be empty")
    if mask is None:
        mask = torch.ones_like(target)
    else:
        mask = mask.float().to(target.device)
        if mask.shape != target.shape:
            raise ValueError("mask must have the same shape as prediction")

    a_00 = torch.sum(mask * prediction * prediction)
    a_01 = torch.sum(mask * prediction)
    a_11 = torch.sum(mask)
    b_0 = torch.sum(mask * prediction * target)
    b_1 = torch.sum(mask * target)
    determinant = a_00 * a_11 - a_01 * a_01
    if not torch.isfinite(determinant) or determinant.abs() <= 1e-6:
        raise ValueError("cannot calibrate scale and shift from degenerate inputs")

    scale = (a_11 * b_0 - a_01 * b_1) / determinant
    shift = (-a_01 * b_0 + a_00 * b_1) / determinant
    return float(scale.item()), float(shift.item())


def torch_calibrate_scale_shift(metric_depths, norm_inv_depths) -> tuple[float, float]:
    if metric_depths.ndim not in (3, 4) or norm_inv_depths.ndim not in (3, 4):
        raise ValueError("depth inputs must be image batches")
    if tuple(metric_depths.shape[1:3]) != tuple(norm_inv_depths.shape[1:3]):
        raise ValueError("depth inputs must have the same spatial dimensions")
    frame_count = min(32, metric_depths.shape[0], norm_inv_depths.shape[0])
    if frame_count == 0:
        raise ValueError("depth inputs must contain at least one frame")
    metric = metric_depths[:frame_count].float()
    normalized = norm_inv_depths[:frame_count].float().to(metric.device)
    if torch.any(metric <= 0) or not torch.isfinite(metric).all():
        raise ValueError("metric depths must be finite and positive")
    return torch_scale_and_shift_full(normalized.reshape(-1), (1.0 / metric).reshape(-1))


def torch_calibrate_scale_shift_manual(
    norm_inv_depths,
    min_depth,
    max_depth,
) -> tuple[float, float]:
    if min_depth <= 0 or max_depth <= min_depth:
        raise ValueError("max_depth must be greater than positive min_depth")
    if norm_inv_depths.numel() == 0 or not torch.isfinite(norm_inv_depths).all():
        raise ValueError("normalized inverse depths must be finite and non-empty")
    x1 = norm_inv_depths.min()
    x2 = norm_inv_depths.max()
    if torch.isclose(x1, x2):
        raise ValueError("normalized inverse depths must contain a non-zero range")
    scale = (1.0 / max_depth - 1.0 / min_depth) / (x1 - x2)
    shift = 1.0 / max_depth - x1 * scale
    return float(scale.item()), float(shift.item())


def torch_estimate_metric_depths(
    norm_inv_depths: torch.Tensor,
    scale: float,
    shift: float,
    max_depth: float,
) -> torch.Tensor:
    if max_depth <= 0:
        raise ValueError("max_depth must be positive")
    denominator = norm_inv_depths.float() * scale + shift
    metric_depths = torch.reciprocal(denominator)
    invalid = (~torch.isfinite(metric_depths)) | (metric_depths <= 0)
    return torch.where(invalid, torch.as_tensor(max_depth, device=metric_depths.device), metric_depths)

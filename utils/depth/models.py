from contextlib import nullcontext

import torch
from depth_anything_v2.dpt import DepthAnythingV2
from video_depth_anything.video_depth import VideoDepthAnything

try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
except ImportError:
    init_empty_weights = None
    set_module_tensor_to_device = None


METRIC_MODEL_CONFIGS = {
    "depth_anything_v2_metric_hypersim_vitl_fp32.safetensors": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "max_depth": 20,
        "is_metric": True,
    }
}

VIDEO_MODEL_CONFIGS = {
    "video_depth_anything_vits.safetensors": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "video_depth_anything_vitl.safetensors": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}


def resolve_dtype(precision: str, model_name: str) -> torch.dtype:
    if precision == "auto":
        return torch.float16 if "fp16" in model_name else torch.float32
    dtypes = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    try:
        return dtypes[precision]
    except KeyError as error:
        raise ValueError(f"unsupported precision: {precision}") from error


def _load_state(model, state_dict, device, dtype):
    if init_empty_weights is not None and set_module_tensor_to_device is not None:
        for key, value in state_dict.items():
            set_module_tensor_to_device(
                model, key, device=device, dtype=dtype, value=value
            )
    else:
        model.load_state_dict(state_dict)
        model.to(device=device, dtype=dtype)
    model.eval()
    return model


def load_metric_depth_model(model_name, state_dict, device, dtype):
    try:
        config = METRIC_MODEL_CONFIGS[model_name]
    except KeyError as error:
        raise ValueError(f"unsupported metric depth model: {model_name}") from error
    context = init_empty_weights() if init_empty_weights is not None else nullcontext()
    with context:
        model = DepthAnythingV2(**config)
    return _load_state(model, state_dict, device, dtype)


def load_video_depth_model(model_name, state_dict, device, dtype):
    try:
        config = VIDEO_MODEL_CONFIGS[model_name]
    except KeyError as error:
        raise ValueError(f"unsupported video depth model: {model_name}") from error
    context = init_empty_weights() if init_empty_weights is not None else nullcontext()
    with context:
        model = VideoDepthAnything(**config)
    return _load_state(model, state_dict, device, dtype)

from pathlib import Path

import comfy.model_management as model_management
import folder_paths
from comfy.utils import load_torch_file

from ...utils.depth.models import (
    METRIC_MODEL_CONFIGS,
    VIDEO_MODEL_CONFIGS,
    load_metric_depth_model,
    load_video_depth_model,
    resolve_dtype,
)


def _model_path(folder_name: str, model_name: str) -> str:
    path = Path(folder_paths.models_dir, folder_name, model_name)
    if not path.is_file():
        raise FileNotFoundError(f"model file not found: {path}")
    return str(path)


class TDGenDepthLoadModelDepthAnythingV2Metric:
    @classmethod
    def INPUT_TYPES(cls):
        names = list(METRIC_MODEL_CONFIGS)
        return {
            "required": {"model": (names, {"default": names[0]})},
            "optional": {
                "precision": (["auto", "bf16", "fp16", "fp32"], {"default": "auto"})
            },
        }

    RETURN_TYPES = ("DAV2METRICMODEL",)
    RETURN_NAMES = ("dav2m_model",)
    FUNCTION = "run"
    CATEGORY = "TDGen/Depth/Models"
    DESCRIPTION = "Load a Depth Anything V2 metric model."

    def run(self, model, precision="auto"):
        device = model_management.get_torch_device()
        dtype = resolve_dtype(precision, model)
        state_dict = load_torch_file(_model_path("depthanything", model))
        loaded = load_metric_depth_model(model, state_dict, device, dtype)
        return ({"model": loaded, "dtype": dtype},)


class TDGenDepthLoadModelVideoDepthAnything:
    @classmethod
    def INPUT_TYPES(cls):
        names = list(VIDEO_MODEL_CONFIGS)
        return {
            "required": {"model": (names, {"default": names[0]})},
            "optional": {
                "precision": (["auto", "bf16", "fp16", "fp32"], {"default": "auto"})
            },
        }

    RETURN_TYPES = ("VDAMODEL",)
    RETURN_NAMES = ("vda_model",)
    FUNCTION = "run"
    CATEGORY = "TDGen/Depth/Models"
    DESCRIPTION = "Load a Video Depth Anything model."

    def run(self, model, precision="auto"):
        device = model_management.get_torch_device()
        dtype = resolve_dtype(precision, model)
        state_dict = load_torch_file(_model_path("video_depth_anything", model))
        loaded = load_video_depth_model(model, state_dict, device, dtype)
        return ({"model": loaded, "dtype": dtype},)

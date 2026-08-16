import unittest
from unittest.mock import Mock, patch
import sys
import types

try:
    import torch

    depth_package = types.ModuleType("depth_anything_v2")
    depth_package.__path__ = []
    depth_module = types.ModuleType("depth_anything_v2.dpt")
    depth_module.DepthAnythingV2 = Mock
    sys.modules.setdefault("depth_anything_v2", depth_package)
    sys.modules.setdefault("depth_anything_v2.dpt", depth_module)

    video_package = types.ModuleType("video_depth_anything")
    video_package.__path__ = []
    video_module = types.ModuleType("video_depth_anything.video_depth")
    video_module.VideoDepthAnything = Mock
    sys.modules.setdefault("video_depth_anything", video_package)
    sys.modules.setdefault("video_depth_anything.video_depth", video_module)

    stereo_package = types.ModuleType("stereocrafter")
    stereo_package.__path__ = []
    stereo_inference = types.ModuleType("stereocrafter.inference")
    stereo_inference.__path__ = []
    stereo_inpainting = types.ModuleType("stereocrafter.inference.inpainting")
    stereo_inpainting.load_models = Mock()
    stereo_inpainting.tiled_inpaint = Mock()
    sys.modules.setdefault("stereocrafter", stereo_package)
    sys.modules.setdefault("stereocrafter.inference", stereo_inference)
    sys.modules.setdefault("stereocrafter.inference.inpainting", stereo_inpainting)

    from utils.depth import models as depth_models
    from utils.stereoscopic import models as stereoscopic_models
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "Torch and model packages are required")
class ModelUtilityTests(unittest.TestCase):
    def test_precision_resolution(self):
        self.assertIs(depth_models.resolve_dtype("fp16", "model"), torch.float16)
        self.assertIs(depth_models.resolve_dtype("auto", "model_fp16.safetensors"), torch.float16)
        self.assertIs(depth_models.resolve_dtype("auto", "model_fp32.safetensors"), torch.float32)

    def test_non_accelerate_model_loading_moves_and_evaluates(self):
        model = Mock()
        with (
            patch.object(depth_models, "init_empty_weights", None),
            patch.object(depth_models, "set_module_tensor_to_device", None),
            patch.object(depth_models, "DepthAnythingV2", return_value=model),
        ):
            loaded = depth_models.load_metric_depth_model(
                next(iter(depth_models.METRIC_MODEL_CONFIGS)),
                {"weight": torch.tensor([1.0])},
                "cpu",
                torch.float32,
            )
        self.assertIs(loaded, model)
        model.load_state_dict.assert_called_once()
        model.to.assert_called_once_with(device="cpu", dtype=torch.float32)
        model.eval.assert_called_once()

    def test_stereocrafter_wrapper_forwards_arguments(self):
        pipeline = Mock()
        pipeline.to.return_value = pipeline
        with patch.object(stereoscopic_models, "load_models", return_value=pipeline) as loader:
            result = stereoscopic_models.load_stereocrafter_pipeline(
                "checkpoint", "unet", "cpu", torch.float16
            )
        self.assertIs(result, pipeline)
        loader.assert_called_once_with(
            pre_trained_path="checkpoint",
            unet_path="unet",
        )
        pipeline.to.assert_called_once_with("cpu")


if __name__ == "__main__":
    unittest.main()

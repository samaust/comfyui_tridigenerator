from ...utils.depth.calibration import (
    torch_calibrate_scale_shift,
    torch_calibrate_scale_shift_manual,
    torch_estimate_metric_depths
)


class TDGenDepthCalibrateConstantsAutomatic:
    """
    A ComfyUI node to estimate scale and shift calibration values from 
    32 frames of inverse metric depth and normalized inverse depth
    """
    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the node.
        """
        return {
            "required": {
                # Data
                "metric_depths": ("IMAGE",),
                "norm_inv_depths": ("IMAGE",),
            }
        }

    RETURN_TYPES = (
        "FLOAT",
        "FLOAT",
    )
    RETURN_NAMES = (
        "scale",
        "shift",
    )
    FUNCTION = "run"
    CATEGORY = "TDGen/Depth/Calibration"
    DESCRIPTION = "Estimate scale and shift calibration values."

    def run(
        self,
        metric_depths,
        norm_inv_depths,
    ):
        # Process data
        scale, shift = torch_calibrate_scale_shift(
            metric_depths,
            norm_inv_depths
        )

        return (scale, shift,)


class TDGenDepthCalibrateConstantsManual:
    """
    A ComfyUI node to estimate scale and shift calibration values from
    manually input min and max depths
    """
    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the node.
        """
        return {
            "required": {
                # Data
                "norm_inv_depths": ("IMAGE",),
                "min_depth": ("FLOAT", {"default": 0.5, "min": 0.1, "step": 0.001}),
                "max_depth": ("FLOAT", {"default": 2.5, "min": 0.1, "max": 20000.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = (
        "FLOAT",
        "FLOAT",
    )
    RETURN_NAMES = (
        "scale",
        "shift",
    )
    FUNCTION = "run"
    CATEGORY = "TDGen/Depth/Calibration"
    DESCRIPTION = "Estimate scale and shift calibration values."

    def run(
        self,
        norm_inv_depths,
        min_depth,
        max_depth
    ):
        # Process data
        scale, shift = torch_calibrate_scale_shift_manual(
            norm_inv_depths,
            min_depth,
            max_depth
        )

        return (scale, shift,)


class TDGenDepthCalibrateMetric:
    """
    A ComfyUI node to estimate scale and shift calibration values from
    32 frames of inverse metric depth metric depth using model Video Depth Anything

    Cannot be previewed because the output is outside the range 0.0 to 1.0
    """
    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the node.
        """
        return {
            "required": {
                # Data
                "norm_inv_depths": ("IMAGE",),
                # Calibration constants
                "scale": ("FLOAT", {"default": 1.0, "step": 0.000001}),
                "shift": ("FLOAT", {"default": 0.0, "step": 0.000001}),
                # Settings
                "max_depth": ("FLOAT", {"default": 100.0, "min": 0.001, "step": 0.001, "tooltip": "Max depth from camera (m)"}),
                "world_scale": ("FLOAT", {"default": 1.0, "min": 0.001, "step": 0.001}),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
    )
    RETURN_NAMES = (
        "metric_depths",
    )
    FUNCTION = "run"
    CATEGORY = "TDGen/Depth/Calibration"
    DESCRIPTION = "Estimate metric depth."

    def run(
        self,
        norm_inv_depths,
        scale,
        shift,
        max_depth,
        world_scale
    ):
        metric_depths = torch_estimate_metric_depths(
            norm_inv_depths,
            scale,
            shift,
            max_depth
        )

        if world_scale != 1.0:
            metric_depths = metric_depths*world_scale

        return (metric_depths,)

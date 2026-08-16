from ...utils.pcd.generation import compute_verts
from ...utils.cameras.intrinsics import slice_vipe_intrinsics, validate_vipe_intrinsics


class TDGenPcdCreatePcd:
    """Create a point-cloud sequence from images and metric depth."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "metric_depths": ("IMAGE",),
                "index": ("INT", {"default": -1, "min": -1, "step": 1}),
                "intrinsics": ("VIPE_INTRINSICS",),
            }
        }

    RETURN_TYPES = ("PCD", "INT", "INT", "VIPE_INTRINSICS")
    RETURN_NAMES = ("pcd", "height", "width", "intrinsics")
    FUNCTION = "run"
    CATEGORY = "TDGen/PCD/Generation"
    DESCRIPTION = "Create point-cloud positions and colors from images and metric depth."

    def run(self, images, metric_depths, index, intrinsics):
        if images.ndim != 4:
            raise ValueError("images must have shape [frames, height, width, channels]")
        frame_count = images.shape[0]
        if frame_count == 0:
            raise ValueError("images must contain at least one frame")
        validate_vipe_intrinsics(intrinsics, frame_count)
        depth_shape = metric_depths.shape[:-1] if metric_depths.ndim == 4 and metric_depths.shape[-1] == 1 else metric_depths.shape
        if tuple(depth_shape) != tuple(images.shape[:3]):
            raise ValueError("metric_depths must match the image frame count and resolution")
        if index < -1 or index >= frame_count:
            raise ValueError(f"index must be -1 or between 0 and {frame_count - 1}")

        if index == -1:
            selected_images = images
            selected_depths = metric_depths
        else:
            if metric_depths.shape[0] <= index:
                raise ValueError("metric_depths does not contain the selected frame")
            selected_images = images[index:index + 1]
            selected_depths = metric_depths[index:index + 1]
            intrinsics = slice_vipe_intrinsics(intrinsics, index)

        pcd = compute_verts(selected_images, selected_depths, intrinsics)
        return (pcd, images.shape[1], images.shape[2], intrinsics)

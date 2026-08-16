from ...utils.pcd.generation import compute_verts


class TDGenPcdCreatePcd:
    """Create a point-cloud sequence from images and metric depth."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "metric_depths": ("IMAGE",),
                "index": ("INT", {"default": -1, "min": -1, "step": 1}),
                "fov_deg": (
                    "INT",
                    {
                        "default": 74,
                        "min": 1,
                        "max": 120,
                        "step": 1,
                        "tooltip": "Horizontal field of view in degrees",
                    },
                ),
            }
        }

    RETURN_TYPES = ("PCD", "INT", "INT", "INT")
    RETURN_NAMES = ("pcd", "height", "width", "fov_deg")
    FUNCTION = "run"
    CATEGORY = "TDGen/PCD/Generation"
    DESCRIPTION = "Create point-cloud positions and colors from images and metric depth."

    def run(self, images, metric_depths, index, fov_deg):
        if images.ndim != 4:
            raise ValueError("images must have shape [frames, height, width, channels]")
        frame_count = images.shape[0]
        if frame_count == 0:
            raise ValueError("images must contain at least one frame")
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

        pcd = compute_verts(selected_images, selected_depths, fov_deg)
        return (pcd, images.shape[1], images.shape[2], fov_deg)

from ....utils.stereoscopic.generation.inpaint import inpaint_stereoscopic_video


class TDGenStereoscopicTiledInpaint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_warped": ("IMAGE",),
                "frames_mask": ("IMAGE",),
                "svdsc_pipeline": ("SVD_STEREOCRAFTER_PIPELINE",),
                "resize_input": ("BOOLEAN", {"default": True}),
                "resize_height": ("INT", {"default": 1920, "min": 128, "max": 8192, "step": 128}),
                "resize_width": ("INT", {"default": 1024, "min": 128, "max": 8192, "step": 128}),
                "frame_rate": ("INT", {"default": 16, "min": 1, "step": 1}),
                "frames_chunk": ("INT", {"default": 25, "min": 2, "max": 10000}),
                "overlap": ("INT", {"default": 3, "min": 0, "max": 256}),
                "tile_num": ("INT", {"default": 2, "min": 1, "max": 256}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "frames_warped", "frames_mask")
    FUNCTION = "run"
    CATEGORY = "TDGen/Stereoscopic/Generation"
    DESCRIPTION = "Inpaint stereoscopic video frames using tiled StereoCrafter inference."

    def run(
        self,
        frames_warped,
        frames_mask,
        svdsc_pipeline,
        resize_input,
        resize_height,
        resize_width,
        frame_rate,
        frames_chunk,
        overlap,
        tile_num,
        steps,
    ):
        if overlap >= frames_chunk:
            raise ValueError("overlap must be smaller than frames_chunk")
        resize_size = (resize_height, resize_width) if resize_input else None
        images = inpaint_stereoscopic_video(
            frames_warped=frames_warped,
            frames_mask=frames_mask,
            pipeline=svdsc_pipeline,
            resize_size=resize_size,
            fps=frame_rate,
            frames_chunk=frames_chunk,
            overlap=overlap,
            tile_num=tile_num,
            num_inference_steps=steps,
        )
        return (images, frames_warped, frames_mask)

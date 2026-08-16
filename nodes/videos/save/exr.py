from pathlib import Path

import folder_paths

from ....utils.videos.save.exr import save_video_exr


class TDGenSaveVideoExr:
    """
    A ComfyUI node to save a video to an OpenEXR multi-part file
    """
    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the node.
        """
        return {
            "required": {
                # Data
                "images": ("IMAGE",),
                "bit_depth": (["32", "16"],),
                "compression": (["none", "zip", "zips", "rle", "pxr24", "b44", "b44a", "dwaa", "dwab"],{"default": "zip"}),
                "filename_prefix": ("STRING", {"default": "TDGen"}),
            }
        }

    RETURN_TYPES = (
        "STRING",
    )
    RETURN_NAMES = (
        "video_path",
    )
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "TDGen/Videos/Save"
    DESCRIPTION = "Save a video to a .exr file."

    def run(
        self,
        images,
        bit_depth,
        compression,
        filename_prefix
    ):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory()
        )
        file = f"{filename}_{counter:05}_.exr"
        save_video_path = str(Path(full_output_folder) / file)

        # Save video
        save_video_exr(images, save_video_path, bit_depth, compression)

        return (save_video_path,)

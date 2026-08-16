from pathlib import Path

import folder_paths

from ....utils.pcd.save.las import save_las


class TDGenPcdSaveLas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "meshes": ("PYTORCH3D_MESHES",),
                "height": ("INT", {"min": 1, "step": 1}),
                "width": ("INT", {"min": 1, "step": 1}),
                "filename_prefix": ("STRING", {"default": "TDGen"}),
                "file_format": (["las", "laz"], {"default": "las"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("save_path",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "TDGen/PCD/Save"
    DESCRIPTION = "Save an incoming mesh sequence to one LAS or LAZ file."

    def run(self, meshes, height, width, filename_prefix, file_format):
        full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
        )
        path = str(Path(full_output_folder) / f"{filename}_{counter:05}_.{file_format}")
        save_las(meshes, height, width, path, file_format)
        return (path,)

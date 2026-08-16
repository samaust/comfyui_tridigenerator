from pathlib import Path

import folder_paths

from ....utils.meshes.save.ply import save_mesh_ply


class TDGenMeshesSavePly:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "meshes": ("PYTORCH3D_MESHES",),
                "filename_prefix": ("STRING", {"default": "TDGen"}),
            }
        }

    RETURN_TYPES = ("STRINGS",)
    RETURN_NAMES = ("save_meshes_paths",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "TDGen/Meshes/Save"
    DESCRIPTION = "Save every incoming mesh as an RGB PLY file."

    def run(self, meshes, filename_prefix):
        if not meshes:
            raise ValueError("meshes must not be empty")
        save_paths = []
        for mesh in meshes:
            full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path(
                filename_prefix,
                folder_paths.get_output_directory(),
            )
            path = str(Path(full_output_folder) / f"{filename}_{counter:05}_.ply")
            save_paths.append(save_mesh_ply(mesh, path))
        return (save_paths,)

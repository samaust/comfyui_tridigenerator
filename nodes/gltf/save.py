from pathlib import Path

import folder_paths

from ...utils.gltf.save import pretty_print_gltf_json, save_gltf


class TDGenGltfSaveGltf:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gltf": ("GLTF",),
                "filename_prefix": ("STRING", {"default": "TDGen"}),
                "save_output": ("BOOLEAN", {"default": True}),
                "print_gltf_json": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("gltf_file_path", "gltf_json")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "TDGen/glTF/Save"
    DESCRIPTION = "Save a glTF point-cloud sequence."

    def run(self, gltf, filename_prefix, save_output=True, print_gltf_json=False):
        output_file_path = ""
        if save_output:
            full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path(
                filename_prefix,
                folder_paths.get_output_directory(),
            )
            output_file_path = str(Path(full_output_folder) / f"{filename}_{counter:05}_.gltf")
            save_gltf(gltf, output_file_path)
        gltf_json = pretty_print_gltf_json(gltf) if print_gltf_json else ""
        return (output_file_path, gltf_json)

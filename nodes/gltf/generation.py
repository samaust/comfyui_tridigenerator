from ...utils.gltf.generation import create_gltf


class TDGenGltfCreateGltf:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pcd": ("PCD",)}}

    RETURN_TYPES = ("GLTF",)
    RETURN_NAMES = ("gltf",)
    FUNCTION = "run"
    CATEGORY = "TDGen/glTF/Generation"
    DESCRIPTION = "Create an in-memory glTF point-cloud sequence."

    def run(self, pcd):
        positions, colors = pcd
        return (create_gltf(positions, colors),)

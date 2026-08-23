import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock

try:
    import numpy  # noqa: F401
    import torch  # noqa: F401
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]


def package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def install_external_stubs():
    comfy = package("comfy")
    comfy_management = types.ModuleType("comfy.model_management")
    comfy_management.get_torch_device = Mock(return_value="cpu")
    comfy_utils = types.ModuleType("comfy.utils")
    comfy_utils.load_torch_file = Mock()
    sys.modules.update(
        {
            "comfy": comfy,
            "comfy.model_management": comfy_management,
            "comfy.utils": comfy_utils,
            "folder_paths": types.ModuleType("folder_paths"),
        }
    )

    pytorch3d = package("pytorch3d")
    renderer = types.ModuleType("pytorch3d.renderer")
    renderer.PerspectiveCameras = Mock
    renderer.MeshRasterizer = Mock
    renderer.RasterizationSettings = Mock
    renderer.TexturesVertex = Mock
    renderer.look_at_view_transform = Mock
    structures = types.ModuleType("pytorch3d.structures")
    structures.Meshes = Mock
    ops = types.ModuleType("pytorch3d.ops")
    ops.interpolate_face_attributes = Mock
    io_module = types.ModuleType("pytorch3d.io")
    io_module.IO = Mock
    sys.modules.update(
        {
            "pytorch3d": pytorch3d,
            "pytorch3d.renderer": renderer,
            "pytorch3d.structures": structures,
            "pytorch3d.ops": ops,
            "pytorch3d.io": io_module,
        }
    )

    oiio = types.ModuleType("OpenImageIO")
    oiio.HALF = object()
    oiio.FLOAT = object()
    sys.modules["OpenImageIO"] = oiio
    pdal = types.ModuleType("pdal")
    pdal.Writer = Mock()
    sys.modules["pdal"] = pdal

    pygltflib = package("pygltflib")
    pygltflib.GLTF2 = Mock
    v2 = package("pygltflib.v2")
    schema = types.ModuleType("pygltflib.v2.schema")
    for name in (
        "Accessor",
        "Attributes",
        "Buffer",
        "BufferView",
        "Material",
        "MaterialPbrMetallicRoughness",
        "Mesh",
        "MeshPrimitive",
        "Node",
        "Scene",
    ):
        setattr(schema, name, Mock)
    for name in (
        "Accessor_componentType",
        "BufferView_target",
        "Material_alphaMode",
        "Mesh_primitive_mode",
    ):
        setattr(schema, name, Mock())
    schema.DATA_URI_HEADER = "data:application/octet-stream;base64,"
    sys.modules.update(
        {
            "pygltflib": pygltflib,
            "pygltflib.v2": v2,
            "pygltflib.v2.schema": schema,
        }
    )

    depth = package("depth_anything_v2")
    depth_dpt = types.ModuleType("depth_anything_v2.dpt")
    depth_dpt.DepthAnythingV2 = Mock
    video = package("video_depth_anything")
    video_depth = types.ModuleType("video_depth_anything.video_depth")
    video_depth.VideoDepthAnything = Mock
    stereo = package("stereocrafter")
    stereo_inference = package("stereocrafter.inference")
    stereo_inpainting = types.ModuleType("stereocrafter.inference.inpainting")
    stereo_inpainting.load_models = Mock()
    stereo_inpainting.tiled_inpaint = Mock()
    sys.modules.update(
        {
            "depth_anything_v2": depth,
            "depth_anything_v2.dpt": depth_dpt,
            "video_depth_anything": video,
            "video_depth_anything.video_depth": video_depth,
            "stereocrafter": stereo,
            "stereocrafter.inference": stereo_inference,
            "stereocrafter.inference.inpainting": stereo_inpainting,
        }
    )


@unittest.skipUnless(torch is not None, "NumPy and Torch are required")
class PackageImportTests(unittest.TestCase):
    def test_root_package_exports_all_registered_nodes(self):
        install_external_stubs()
        module_name = "_tdgen_import_test"
        spec = importlib.util.spec_from_file_location(
            module_name,
            ROOT / "__init__.py",
            submodule_search_locations=[str(ROOT)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        self.assertEqual(len(module.NODE_CLASS_MAPPINGS), 21)
        self.assertEqual(
            set(module.NODE_CLASS_MAPPINGS),
            set(module.NODE_DISPLAY_NAME_MAPPINGS),
        )
        self.assertEqual(module.WEB_DIRECTORY, "./web")

        renderer_inputs = module.NODE_CLASS_MAPPINGS["TDGenRenderTexturedMesh"].INPUT_TYPES()["required"]
        for name in ("displacement_x", "displacement_y", "displacement_z", "yaw", "pitch", "roll"):
            options = renderer_inputs[name][1]
            self.assertLess(options["min"], 0, name)
            self.assertGreater(options["max"], 0, name)
        self.assertEqual(
            renderer_inputs["camera_origin"][0],
            ["first_frame", "vipe_world", "current_frame"],
        )
        self.assertEqual(renderer_inputs["camera_origin"][1]["default"], "vipe_world")
        self.assertEqual(renderer_inputs["bg_color"][0], "STRING")
        self.assertEqual(renderer_inputs["bg_color"][1]["default"], "#000000")
        self.assertEqual(list(renderer_inputs)[-1], "bg_color")


if __name__ == "__main__":
    unittest.main()

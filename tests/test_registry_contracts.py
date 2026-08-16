import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_KEYS = {
    "TDGenDepthLoadModelDepthAnythingV2Metric",
    "TDGenDepthLoadModelVideoDepthAnything",
    "TDGenDepthEstimateDepthAnythingV2Metric",
    "TDGenDepthEstimateVideoDepthAnything",
    "TDGenDepthCalibrateConstantsAutomatic",
    "TDGenDepthCalibrateConstantsManual",
    "TDGenDepthCalibrateMetric",
    "TDGenDepthMeasureDistance",
    "TDGenPcdCreatePcd",
    "TDGenPcdSaveLas",
    "TDGenGltfCreateGltf",
    "TDGenGltfSaveGltf",
    "TDGenMeshesCreateMeshes",
    "TDGenMeshesSavePly",
    "TDGenClassificationAlpha",
    "TDGenClassificationVisible",
    "TDGenRenderTexturedMesh",
    "TDGenStereoscopicLoadModelSvdStereoCrafter",
    "TDGenStereoscopicTiledInpaint",
    "TDGenSaveVideoExr",
    "TDGenLoadVideoExr",
}
OUTPUT_NODES = {
    "TDGenPcdSaveLas",
    "TDGenGltfSaveGltf",
    "TDGenMeshesSavePly",
    "TDGenSaveVideoExr",
}


def assigned_dict_keys(tree, variable_name):
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == variable_name for target in node.targets):
            continue
        return {key.value for key in node.value.keys if isinstance(key, ast.Constant)}
    raise AssertionError(f"{variable_name} was not found")


class RegistryContractTests(unittest.TestCase):
    def test_mapping_keys_are_complete_and_symmetric(self):
        tree = ast.parse((ROOT / "nodes" / "nodes.py").read_text())
        class_keys = assigned_dict_keys(tree, "NODE_CLASS_MAPPINGS")
        display_keys = assigned_dict_keys(tree, "NODE_DISPLAY_NAME_MAPPINGS")
        self.assertEqual(class_keys, EXPECTED_KEYS)
        self.assertEqual(display_keys, EXPECTED_KEYS)
        self.assertNotIn("TridiGeneratorPcdEstimation", class_keys)

    def test_registered_nodes_do_not_expose_fov(self):
        for path in (ROOT / "nodes").rglob("*.py"):
            self.assertNotIn("fov", path.read_text().lower(), str(path))

    def test_pytorch3d_is_not_imported_while_registering_nodes(self):
        for path in (ROOT / "utils").rglob("*.py"):
            if path.name == "pytorch3d.py":
                continue
            tree = ast.parse(path.read_text())
            for statement in tree.body:
                if isinstance(statement, ast.Import):
                    modules = [alias.name for alias in statement.names]
                elif isinstance(statement, ast.ImportFrom) and statement.level == 0:
                    modules = [statement.module or ""]
                else:
                    continue
                self.assertFalse(
                    any(module == "pytorch3d" or module.startswith("pytorch3d.") for module in modules),
                    f"{path} imports PyTorch3D at module scope",
                )

    def test_every_node_declares_required_metadata(self):
        classes = {}
        output_nodes = set()
        for path in (ROOT / "nodes").rglob("*.py"):
            if path.name in {"__init__.py", "nodes.py"}:
                continue
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if not isinstance(node, ast.ClassDef) or not node.name.startswith("TDGen"):
                    continue
                assigned = {
                    target.id
                    for statement in node.body
                    if isinstance(statement, ast.Assign)
                    for target in statement.targets
                    if isinstance(target, ast.Name)
                }
                methods = {
                    statement.name
                    for statement in node.body
                    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                classes[node.name] = assigned | methods
                if "OUTPUT_NODE" in assigned:
                    output_nodes.add(node.name)

        self.assertEqual(set(classes), EXPECTED_KEYS)
        required = {"INPUT_TYPES", "RETURN_TYPES", "RETURN_NAMES", "FUNCTION", "CATEGORY", "DESCRIPTION"}
        for name, fields in classes.items():
            self.assertTrue(required <= fields, f"{name} is missing {required - fields}")
        self.assertEqual(output_nodes, OUTPUT_NODES)

    def test_pipeline_contracts_are_visible_in_node_sources(self):
        pcd_source = (ROOT / "nodes" / "pcd" / "generation.py").read_text()
        mesh_source = (ROOT / "nodes" / "meshes" / "generation.py").read_text()
        gltf_source = (ROOT / "nodes" / "gltf" / "generation.py").read_text()
        self.assertIn('"index":', pcd_source)
        self.assertNotIn('"index":', mesh_source)
        self.assertIn('"pcd": ("PCD",)', gltf_source)


if __name__ == "__main__":
    unittest.main()

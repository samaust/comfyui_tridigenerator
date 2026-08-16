import importlib.util
from pathlib import Path
import sys
import types
import unittest

try:
    import numpy as np
    import torch

    from utils.depth.calibration import (
        torch_calibrate_scale_shift_manual,
        torch_estimate_metric_depths,
    )
    from utils.depth.measurements import measure_distances
    from utils.pcd.generation import compute_verts
except ImportError:
    np = None
    torch = None


ROOT = Path(__file__).resolve().parents[1]


def load_pcd_node_module():
    root_name = "_tdgen_node_test"
    package_paths = {
        root_name: ROOT,
        f"{root_name}.nodes": ROOT / "nodes",
        f"{root_name}.nodes.pcd": ROOT / "nodes" / "pcd",
        f"{root_name}.utils": ROOT / "utils",
        f"{root_name}.utils.cameras": ROOT / "utils" / "cameras",
        f"{root_name}.utils.pcd": ROOT / "utils" / "pcd",
    }
    for name, path in package_paths.items():
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package

    utility_name = f"{root_name}.utils.pcd.generation"
    utility_spec = importlib.util.spec_from_file_location(
        utility_name, ROOT / "utils" / "pcd" / "generation.py"
    )
    utility_module = importlib.util.module_from_spec(utility_spec)
    sys.modules[utility_name] = utility_module
    utility_spec.loader.exec_module(utility_module)

    node_name = f"{root_name}.nodes.pcd.generation"
    node_spec = importlib.util.spec_from_file_location(
        node_name, ROOT / "nodes" / "pcd" / "generation.py"
    )
    node_module = importlib.util.module_from_spec(node_spec)
    sys.modules[node_name] = node_module
    node_spec.loader.exec_module(node_module)
    return node_module


@unittest.skipUnless(np is not None and torch is not None, "NumPy and Torch are required")
class GeometryUtilityTests(unittest.TestCase):
    @staticmethod
    def intrinsics(count=1):
        return {
            "values": torch.tensor([[2.0, 4.0, 0.25, 0.75]]).repeat(count, 1),
            "frame_indices": torch.arange(count),
            "camera_models": ("pinhole",) * count,
        }

    def test_point_cloud_shape_dtype_and_negative_z(self):
        images = torch.tensor(
            [[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]]]
        )
        depths = torch.full((1, 2, 2), 2.0)
        positions, colors = compute_verts(images, depths, self.intrinsics())
        self.assertEqual(positions.shape, (1, 4, 3))
        self.assertEqual(colors.shape, (1, 4, 3))
        self.assertEqual(positions.dtype, np.float32)
        self.assertEqual(colors.dtype, np.float32)
        np.testing.assert_allclose(positions[..., 2], -2.0)
        np.testing.assert_allclose(positions[0, 0], [-0.25, 0.375, -2.0])

    def test_point_cloud_rejects_invalid_depth(self):
        with self.assertRaises(ValueError):
            compute_verts(torch.ones((1, 2, 2, 3)), torch.zeros((1, 2, 2)), self.intrinsics())

    def test_pcd_node_selects_all_or_one_frame_once(self):
        node_class = load_pcd_node_module().TDGenPcdCreatePcd
        images = torch.ones((3, 2, 2, 3))
        depths = torch.ones((3, 2, 2))
        intrinsics = self.intrinsics(3)
        all_pcd = node_class().run(images, depths, -1, intrinsics)[0]
        one_pcd = node_class().run(images, depths, 1, intrinsics)[0]
        self.assertEqual(all_pcd[0].shape[0], 3)
        self.assertEqual(one_pcd[0].shape[0], 1)
        with self.assertRaises(ValueError):
            node_class().run(images, depths, 3, intrinsics)

    def test_distance_measurement_for_center_ray(self):
        depths = torch.full((1, 2, 2), 2.0)
        pairs = np.asarray([[[1, 1], [0, 1]]], dtype=np.float64)
        result = measure_distances(depths, self.intrinsics(), pairs)[0]
        self.assertAlmostEqual(result[9], 1.0)
        self.assertAlmostEqual(result[2], -2.0)
        self.assertAlmostEqual(result[5], -2.0)

    def test_manual_calibration_returns_python_floats(self):
        normalized = torch.tensor([[[0.0, 1.0]]])
        scale, shift = torch_calibrate_scale_shift_manual(normalized, 1.0, 2.0)
        self.assertIsInstance(scale, float)
        self.assertIsInstance(shift, float)
        metric = torch_estimate_metric_depths(normalized, scale, shift, 10.0)
        self.assertTrue(torch.isfinite(metric).all())


if __name__ == "__main__":
    unittest.main()

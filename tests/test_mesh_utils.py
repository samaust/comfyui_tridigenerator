from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

try:
    import numpy as np
    import torch

    from utils.classification.alpha import add_classification_alpha
    from utils.meshes.generation import create_meshes
    from utils.meshes.save.ply import save_mesh_ply
    from utils.pcd.save import las as las_utils
except ImportError:
    np = None
    torch = None


@unittest.skipUnless(np is not None and torch is not None, "PyTorch3D and PDAL are required")
class MeshUtilityTests(unittest.TestCase):
    def setUp(self):
        positions = np.asarray(
            [[[-1, 1, -2], [1, 1, -2], [-1, -1, -2], [1, -1, -2]]],
            dtype=np.float32,
        )
        colors = np.ones((1, 4, 3), dtype=np.float32)
        self.meshes = create_meshes((positions, colors), 2, 2, "cpu")

    def test_mesh_features_are_fixed_and_alpha_is_replaced(self):
        features = self.meshes[0].textures.verts_features_list()[0]
        self.assertEqual(tuple(features.shape), (4, 5))
        torch.testing.assert_close(features[:, 3:], torch.ones((4, 2)))

        masks = torch.tensor([[[0.0, 0.25], [0.5, 1.0]]])
        classified, previews = add_classification_alpha(self.meshes, masks, 2, 2, "cpu")
        output_features = classified[0].textures.verts_features_list()[0]
        torch.testing.assert_close(output_features[:, 3], masks.reshape(-1))
        torch.testing.assert_close(output_features[:, 4], torch.ones(4))
        self.assertEqual(tuple(previews.shape), (1, 2, 2, 3))

    def test_ply_writer_strips_extra_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "mesh.ply")
            self.assertEqual(save_mesh_ply(self.meshes[0], path), path)
            self.assertTrue(Path(path).is_file())

    def test_las_stream_uses_uint16_frame_ids_and_classification(self):
        class FakePipeline:
            def __init__(self):
                self.inputs = None
                self.frames = []

            def execute_streaming(self, chunk_size):
                buffer, callback = self.inputs[0]
                count = 0
                while True:
                    loaded = callback()
                    if loaded == 0:
                        return count
                    self.frames.append(buffer.copy())
                    count += loaded

        fake_pipeline = FakePipeline()
        writer = Mock()
        writer.pipeline.return_value = fake_pipeline
        with patch.object(las_utils.pdal.Writer, "las", return_value=writer):
            count = las_utils.save_las(
                self.meshes * 2,
                2,
                2,
                "unused.las",
                "las",
            )
        self.assertEqual(count, 8)
        self.assertEqual(fake_pipeline.frames[0]["PointSourceId"].dtype, np.dtype("uint16"))
        self.assertTrue(np.all(fake_pipeline.frames[0]["PointSourceId"] == 0))
        self.assertTrue(np.all(fake_pipeline.frames[1]["PointSourceId"] == 1))
        self.assertTrue(np.all(fake_pipeline.frames[0]["Classification"] == 3))


if __name__ == "__main__":
    unittest.main()

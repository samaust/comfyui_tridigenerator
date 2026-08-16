from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
    import torch

    from utils.gltf.generation import create_gltf
    from utils.videos.load.exr import load_video_exr
    from utils.videos.save.exr import save_video_exr
except ImportError:
    np = None
    torch = None


@unittest.skipUnless(np is not None and torch is not None, "Media dependencies are required")
class MediaUtilityTests(unittest.TestCase):
    def test_gltf_accepts_rgb_and_rgba(self):
        positions = np.zeros((1, 4, 3), dtype=np.float32)
        for channels in (3, 4):
            colors = np.ones((1, 4, channels), dtype=np.float32)
            gltf = create_gltf(positions, colors)
            self.assertEqual(gltf.accessors[1].type, "VEC4")
            self.assertEqual(gltf.accessors[1].count, 4)

    def test_exr_round_trip_supported_channel_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            for channels in (1, 3, 4):
                shape = (2, 4, 4) if channels == 1 else (2, 4, 4, channels)
                images = torch.rand(shape, dtype=torch.float32)
                path = str(Path(directory) / f"video-{channels}.exr")
                save_video_exr(images, path, "32", "zip")
                loaded = load_video_exr(path)
                expected_shape = (2, 4, 4, 1) if channels == 1 else shape
                self.assertEqual(tuple(loaded.shape), expected_shape)
                self.assertEqual(loaded.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()

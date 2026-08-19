import unittest

try:
    import torch

    from utils.rendering.textured_mesh import _camera_poses_relative_to_first
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "Torch is required")
class RenderingUtilityTests(unittest.TestCase):
    def test_camera_poses_are_rebased_into_nonidentity_first_camera(self):
        first = torch.tensor(
            [
                [0.0, -1.0, 0.0, 4.0],
                [1.0, 0.0, 0.0, -2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        first_to_second = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.5],
                [0.0, 0.0, -1.0, 0.25],
                [0.0, 1.0, 0.0, -0.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        poses = torch.stack((first, first @ first_to_second))
        conversion = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))

        relative = _camera_poses_relative_to_first(poses, conversion)

        expected_second = conversion @ first_to_second @ torch.linalg.inv(conversion)
        torch.testing.assert_close(relative[0], torch.eye(4))
        torch.testing.assert_close(relative[1], expected_second)


if __name__ == "__main__":
    unittest.main()

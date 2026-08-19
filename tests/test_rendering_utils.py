import unittest

try:
    import torch

    from utils.rendering.textured_mesh import _camera_poses_for_origin, _camera_view_for_frame
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

        relative = _camera_poses_for_origin(poses, conversion, "first_frame")

        expected_second = conversion @ first_to_second @ torch.linalg.inv(conversion)
        torch.testing.assert_close(relative[0], torch.eye(4))
        torch.testing.assert_close(relative[1], expected_second)

    def test_camera_poses_can_keep_vipe_world_origin(self):
        poses = torch.tensor(
            [
                [
                    [0.0, -1.0, 0.0, 4.0],
                    [1.0, 0.0, 0.0, -2.0],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ]
        )
        conversion = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))

        converted = _camera_poses_for_origin(poses, conversion, "vipe_world")

        expected = conversion @ poses @ torch.linalg.inv(conversion)
        torch.testing.assert_close(converted, expected)

    def test_current_frame_uses_first_frame_relative_mesh_poses(self):
        first = torch.tensor(
            [
                [0.0, -1.0, 0.0, 4.0],
                [1.0, 0.0, 0.0, -2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        motion = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.5],
                [0.0, 0.0, -1.0, 0.25],
                [0.0, 1.0, 0.0, -0.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        poses = torch.stack((first, first @ motion))
        conversion = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))

        current = _camera_poses_for_origin(poses, conversion, "current_frame")
        anchored = _camera_poses_for_origin(poses, conversion, "first_frame")

        torch.testing.assert_close(current, anchored)

    def test_current_frame_camera_cancels_mesh_pose_with_identity_controls(self):
        mesh_poses = torch.stack((torch.eye(4), self._camera_motion()))
        controls = torch.eye(4)

        for mesh_pose in mesh_poses:
            view = _camera_view_for_frame(mesh_pose, controls, "current_frame")
            torch.testing.assert_close(view @ mesh_pose, torch.eye(4))

    def test_current_frame_camera_applies_controls_in_local_axes(self):
        mesh_pose = self._camera_motion()
        controls = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.5],
                [0.0, 1.0, 0.0, -0.25],
                [-1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        view = _camera_view_for_frame(mesh_pose, controls, "current_frame")

        torch.testing.assert_close(view @ mesh_pose, torch.linalg.inv(controls))

    def test_camera_poses_reject_unknown_origin(self):
        poses = torch.eye(4).unsqueeze(0)
        conversion = torch.eye(4)

        with self.assertRaisesRegex(ValueError, "camera_origin"):
            _camera_poses_for_origin(poses, conversion, "unknown")

    @staticmethod
    def _camera_motion():
        return torch.tensor(
            [
                [0.0, -1.0, 0.0, 2.0],
                [1.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )


if __name__ == "__main__":
    unittest.main()

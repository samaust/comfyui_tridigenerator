import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch

    from utils.rendering.textured_mesh import (
        _camera_poses_for_origin,
        _camera_view_for_frame,
        _parse_hex_color,
        render_textured_mesh,
    )
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "Torch is required")
class RenderingUtilityTests(unittest.TestCase):
    def test_render_mask_selects_only_pixels_without_rasterized_faces(self):
        images, masks, depths, metadata = self._render_inputs(mask_value=1.0)
        pix_to_face = torch.tensor([[[[0], [-1]], [[1], [-1]]]])
        fake_pytorch3d = self._fake_pytorch3d(pix_to_face)

        with patch("utils.rendering.textured_mesh.require_pytorch3d", return_value=fake_pytorch3d):
            rendered, unrendered_mask = render_textured_mesh(
                images,
                masks,
                depths,
                metadata,
                bg_color="#336699",
            )

        expected_mask = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
        expected_background = torch.tensor([0x33 / 255, 0x66 / 255, 0x99 / 255])
        torch.testing.assert_close(unrendered_mask, expected_mask)
        torch.testing.assert_close(rendered[0, 0, 1], expected_background)
        torch.testing.assert_close(rendered[0, 1, 1], expected_background)
        torch.testing.assert_close(rendered[0, 0, 0], torch.full((3,), 0.25))
        self.assertEqual(tuple(rendered.shape), (1, 2, 2, 3))
        self.assertEqual(tuple(unrendered_mask.shape), (1, 2, 2))
        self.assertEqual(unrendered_mask.dtype, torch.float32)
        self.assertEqual(unrendered_mask.device.type, "cpu")

    def test_frame_without_renderable_faces_has_background_and_full_mask(self):
        images, masks, depths, metadata = self._render_inputs(mask_value=0.0)

        with patch("utils.rendering.textured_mesh.require_pytorch3d", return_value=SimpleNamespace()):
            rendered, unrendered_mask = render_textured_mesh(
                images,
                masks,
                depths,
                metadata,
                bg_color="#336699",
            )

        expected_background = torch.tensor([0x33 / 255, 0x66 / 255, 0x99 / 255])
        torch.testing.assert_close(rendered, expected_background.expand(1, 2, 2, 3))
        torch.testing.assert_close(unrendered_mask, torch.ones((1, 2, 2)))

    def test_hex_background_color_is_converted_to_normalized_rgb(self):
        color = _parse_hex_color("#00ff80")

        self.assertEqual(color[:2], (0.0, 1.0))
        self.assertAlmostEqual(color[2], 128 / 255)

    def test_hex_background_color_rejects_invalid_values(self):
        for value in ("000000", "#fff", "#1234567", "#GG0000", "# 00000", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "bg_color"):
                    _parse_hex_color(value)

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

    @staticmethod
    def _render_inputs(mask_value):
        metadata = {
            "schema": "comfyui_vipe_metadata",
            "timing": {"frame_count": 1},
            "dimensions": {"height": 2, "width": 2},
            "camera_to_world": {
                "frame_indices": [0],
                "matrix_layout": "row_major",
                "convention": "opencv_x_right_y_down_z_forward",
                "matrices": [torch.eye(4).reshape(-1).tolist()],
            },
            "intrinsics": {
                "frame_indices": [0],
                "layout": ["fx", "fy", "cx", "cy"],
                "camera_models": ["pinhole"],
                "values": [[1.0, 1.0, 0.0, 0.0]],
            },
        }
        return (
            torch.zeros((1, 2, 2, 3)),
            torch.full((1, 2, 2), mask_value),
            torch.ones((1, 2, 2)),
            json.dumps(metadata),
        )

    @staticmethod
    def _fake_pytorch3d(pix_to_face):
        class FakeRasterizer:
            def __init__(self, **kwargs):
                pass

            def __call__(self, mesh):
                return SimpleNamespace(
                    pix_to_face=pix_to_face,
                    bary_coords=torch.zeros((1, 2, 2, 1, 3)),
                )

        return SimpleNamespace(
            Meshes=lambda **kwargs: object(),
            PerspectiveCameras=lambda **kwargs: object(),
            MeshRasterizer=FakeRasterizer,
            RasterizationSettings=lambda **kwargs: object(),
            interpolate_face_attributes=lambda *args: torch.full((1, 2, 2, 1, 3), 0.25),
        )


if __name__ == "__main__":
    unittest.main()

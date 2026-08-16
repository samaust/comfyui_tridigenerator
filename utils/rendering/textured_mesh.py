import json
import math

import torch

from ..datatypes import Device
from ..pytorch3d import require_pytorch3d


def _validated_metadata(json_data: str, frame_count: int, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        metadata = json.loads(json_data)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("json_data must be valid ViPE metadata JSON") from exc
    if not isinstance(metadata, dict) or metadata.get("schema") != "comfyui_vipe_metadata":
        raise ValueError("json_data must use the comfyui_vipe_metadata schema")
    timing = metadata.get("timing", {})
    dimensions = metadata.get("dimensions", {})
    if timing.get("frame_count") != frame_count or dimensions != {"height": height, "width": width}:
        raise ValueError("metadata frame count or resolution does not match the inputs")
    pose_data = metadata.get("camera_to_world", {})
    intr_data = metadata.get("intrinsics", {})
    expected = list(range(frame_count))
    if pose_data.get("frame_indices") != expected or intr_data.get("frame_indices") != expected:
        raise ValueError("metadata frame indices must be contiguous and synchronized")
    if pose_data.get("matrix_layout") != "row_major":
        raise ValueError("camera poses must be row-major")
    if pose_data.get("convention") != "opencv_x_right_y_down_z_forward":
        raise ValueError("camera poses must use the documented OpenCV convention")
    if intr_data.get("layout") != ["fx", "fy", "cx", "cy"]:
        raise ValueError("metadata must contain [fx,fy,cx,cy] intrinsics")
    if intr_data.get("camera_models") != ["pinhole"] * frame_count:
        raise ValueError("only pinhole camera models are supported")
    poses = torch.as_tensor(pose_data.get("matrices"), dtype=torch.float64)
    intrinsics = torch.as_tensor(intr_data.get("values"), dtype=torch.float64)
    if poses.shape != (frame_count, 16) or intrinsics.shape != (frame_count, 4):
        raise ValueError("metadata poses or intrinsics have invalid shapes")
    poses = poses.reshape(frame_count, 4, 4)
    if not torch.isfinite(poses).all() or not torch.isfinite(intrinsics).all() or (intrinsics[:, :2] <= 0).any():
        raise ValueError("metadata poses and intrinsics must be finite with positive focal lengths")
    expected_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64).expand(frame_count, -1)
    rotations = poses[:, :3, :3]
    identity = torch.eye(3, dtype=torch.float64).expand(frame_count, -1, -1)
    if not torch.allclose(poses[:, 3], expected_bottom, atol=1e-5, rtol=0):
        raise ValueError("camera poses must be rigid homogeneous transforms")
    if not torch.allclose(rotations.transpose(1, 2) @ rotations, identity, atol=1e-4, rtol=0):
        raise ValueError("camera pose rotations must be orthonormal")
    if not torch.allclose(torch.linalg.det(rotations), torch.ones(frame_count, dtype=torch.float64), atol=1e-4, rtol=0):
        raise ValueError("camera pose rotations must have determinant one")
    return poses.to(torch.float32), intrinsics.to(torch.float32)


def _axis_rotation(axis: str, degrees: float, device: Device) -> torch.Tensor:
    angle = math.radians(float(degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    if axis == "x":
        values = ((1, 0, 0), (0, cosine, -sine), (0, sine, cosine))
    elif axis == "y":
        values = ((cosine, 0, sine), (0, 1, 0), (-sine, 0, cosine))
    else:
        values = ((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1))
    return torch.tensor(values, dtype=torch.float32, device=device)


def _view_control(displacement: tuple[float, float, float], rotation: tuple[float, float, float], device: Device) -> torch.Tensor:
    yaw, pitch, roll = rotation
    transform = torch.eye(4, dtype=torch.float32, device=device)
    transform[:3, :3] = (
        _axis_rotation("y", yaw, device)
        @ _axis_rotation("x", pitch, device)
        @ _axis_rotation("z", roll, device)
    )
    transform[:3, 3] = torch.tensor(displacement, dtype=torch.float32, device=device)
    return transform


def _grid_faces(height: int, width: int, valid: torch.Tensor) -> torch.Tensor:
    indices = torch.arange(height * width, device=valid.device).reshape(height, width)
    top_left = indices[:-1, :-1]
    bottom_left = indices[1:, :-1]
    bottom_right = indices[1:, 1:]
    top_right = indices[:-1, 1:]
    first = torch.stack((top_left, bottom_left, bottom_right), dim=-1).reshape(-1, 3)
    second = torch.stack((top_left, bottom_right, top_right), dim=-1).reshape(-1, 3)
    faces = torch.cat((first, second), dim=0)
    return faces[valid.reshape(-1)[faces].all(dim=1)]


def render_textured_mesh(
    images: torch.Tensor,
    masks: torch.Tensor,
    depths: torch.Tensor,
    json_data: str,
    displacement: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    device: Device = "cpu",
) -> torch.Tensor:
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("images must have shape [N,H,W,3]")
    if masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]
    if depths.ndim == 4 and depths.shape[-1] == 1:
        depths = depths[..., 0]
    if masks.ndim != 3 or depths.ndim != 3 or tuple(images.shape[:3]) != tuple(masks.shape) or tuple(masks.shape) != tuple(depths.shape):
        raise ValueError("images, masks, and depths must have synchronized frame counts and resolutions")
    frame_count, height, width = images.shape[:3]
    if frame_count == 0 or height < 2 or width < 2:
        raise ValueError("the renderer requires non-empty images at least 2x2 pixels")
    pytorch3d = require_pytorch3d()
    poses, intrinsics = _validated_metadata(json_data, frame_count, height, width)
    images = images.detach().to(device=device, dtype=torch.float32)
    masks = masks.detach().to(device=device)
    depths = depths.detach().to(device=device, dtype=torch.float32)
    poses = poses.to(device)
    intrinsics = intrinsics.to(device)
    if not torch.isfinite(images).all():
        raise ValueError("images must contain finite values")

    conversion = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device=device))
    relative = conversion @ poses @ torch.linalg.inv(poses[0]) @ torch.linalg.inv(conversion)
    view = torch.linalg.inv(_view_control(displacement, rotation, device))
    y_pixels, x_pixels = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    output = []
    for index in range(frame_count):
        depth = depths[index]
        valid = torch.isfinite(depth) & (depth > 0) & (masks[index] > 0.5)
        safe_depth = torch.where(torch.isfinite(depth) & (depth > 0), depth, torch.ones_like(depth))
        fx, fy, cx, cy = intrinsics[index]
        local = torch.stack(
            (
                (x_pixels - cx) * safe_depth / fx,
                -(y_pixels - cy) * safe_depth / fy,
                -safe_depth,
                torch.ones_like(depth),
            ),
            dim=-1,
        ).reshape(-1, 4)
        world = (relative[index] @ local.transpose(0, 1)).transpose(0, 1)
        camera_gl = (view @ world.transpose(0, 1)).transpose(0, 1)[:, :3]
        verts = camera_gl * torch.tensor([-1.0, 1.0, -1.0], device=device)
        faces = _grid_faces(height, width, valid)
        if faces.numel() > 0:
            faces = faces[(verts[:, 2] > 1e-6)[faces].all(dim=1)]
        if faces.numel() == 0:
            output.append(torch.zeros((height, width, 3), device=device))
            continue
        mesh = pytorch3d.Meshes(verts=[verts], faces=[faces])
        camera = pytorch3d.PerspectiveCameras(
            focal_length=intrinsics[index:index + 1, :2],
            principal_point=intrinsics[index:index + 1, 2:],
            image_size=((height, width),),
            in_ndc=False,
            device=device,
        )
        fragments = pytorch3d.MeshRasterizer(
            cameras=camera,
            raster_settings=pytorch3d.RasterizationSettings(
                image_size=(height, width), blur_radius=0.0, faces_per_pixel=1, perspective_correct=True
            ),
        )(mesh)
        face_colors = images[index].reshape(-1, 3)[faces]
        pixels = pytorch3d.interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, face_colors
        )[0, :, :, 0]
        pixels[fragments.pix_to_face[0, :, :, 0] < 0] = 0
        output.append(pixels.clamp(0, 1))
    return torch.stack(output).detach().cpu().to(torch.float32)

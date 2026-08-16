import numpy as np
import torch
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes

from ..datatypes import Device, PointCloud


def _grid_faces(height: int, width: int, device: Device) -> torch.Tensor:
    if height < 2 or width < 2:
        raise ValueError("mesh height and width must both be at least 2")
    indices = torch.arange(height * width, device=device).reshape(height, width)
    top_left = indices[:-1, :-1].reshape(-1)
    bottom_left = indices[1:, :-1].reshape(-1)
    bottom_right = indices[1:, 1:].reshape(-1)
    top_right = indices[:-1, 1:].reshape(-1)
    first = torch.stack((top_left, bottom_left, bottom_right), dim=1)
    second = torch.stack((top_left, bottom_right, top_right), dim=1)
    return torch.cat((first, second), dim=0).to(dtype=torch.int64)


def create_meshes(
    pcd: PointCloud,
    height: int,
    width: int,
    device: Device = "cpu",
) -> list[Meshes]:
    """Convert a point-cloud sequence to meshes with fixed RGBA/visibility features."""
    positions, colors = pcd
    positions = np.asarray(positions, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("PCD positions must have shape [frames, vertices, 3]")
    if colors.ndim != 3 or colors.shape[-1] not in (3, 4):
        raise ValueError("PCD colors must have shape [frames, vertices, 3|4]")
    if positions.shape[:2] != colors.shape[:2]:
        raise ValueError("PCD positions and colors must share frame and vertex counts")
    if positions.shape[1] != height * width:
        raise ValueError("height * width must equal the PCD vertex count")

    faces = _grid_faces(height, width, device)
    meshes = []
    for frame_positions, frame_colors in zip(positions, colors):
        verts = torch.as_tensor(frame_positions, dtype=torch.float32, device=device)
        rgb_or_rgba = torch.as_tensor(frame_colors, dtype=torch.float32, device=device)
        if rgb_or_rgba.shape[1] == 3:
            alpha = torch.ones((rgb_or_rgba.shape[0], 1), device=device)
            rgba = torch.cat((rgb_or_rgba, alpha), dim=1)
        else:
            rgba = rgb_or_rgba
        visible = torch.ones((rgba.shape[0], 1), device=device)
        features = torch.cat((rgba, visible), dim=1).unsqueeze(0)
        meshes.append(
            Meshes(
                verts=[verts],
                faces=[faces.clone()],
                textures=TexturesVertex(verts_features=features),
            )
        )
    return meshes

from __future__ import annotations

from typing import Any

import torch

from ..datatypes import Device
from ..pytorch3d import require_pytorch3d


def _mesh_features(mesh: Any, device: Device) -> torch.Tensor:
    if mesh.textures is None or not hasattr(mesh.textures, "verts_features_list"):
        raise ValueError("mesh must contain vertex features")
    features = mesh.textures.verts_features_list()[0].to(device)
    if features.ndim != 2 or features.shape[1] != 5:
        raise ValueError("mesh features must use [R, G, B, alpha, visible]")
    return features


def add_classification_alpha(
    meshes: list[Any],
    masks: torch.Tensor | None,
    height: int,
    width: int,
    device: Device = "cpu",
) -> tuple[list[Any], torch.Tensor]:
    """Replace the alpha feature channel and return ComfyUI preview images."""
    if not meshes:
        raise ValueError("meshes must not be empty")
    if masks is not None:
        if masks.ndim == 4 and masks.shape[-1] == 1:
            masks = masks[..., 0]
        if masks.ndim != 3 or tuple(masks.shape) != (len(meshes), height, width):
            raise ValueError("masks must have shape [meshes, height, width]")

    pytorch3d = require_pytorch3d()
    classified_meshes = []
    previews = []
    for index, mesh in enumerate(meshes):
        features = _mesh_features(mesh, device).clone()
        alpha = (
            masks[index].to(device=device, dtype=features.dtype).reshape(-1, 1)
            if masks is not None
            else torch.ones((features.shape[0], 1), device=device, dtype=features.dtype)
        )
        if alpha.shape[0] != features.shape[0]:
            raise ValueError("mask pixel count must equal mesh vertex count")
        features[:, 3:4] = alpha.clamp(0.0, 1.0)
        verts = mesh.verts_list()[0].to(device)
        faces = mesh.faces_list()[0].to(device)
        classified_meshes.append(
            pytorch3d.Meshes(
                verts=[verts],
                faces=[faces],
                textures=pytorch3d.TexturesVertex(verts_features=features.unsqueeze(0)),
            )
        )
        preview = features[:, 3].reshape(height, width, 1).repeat(1, 1, 3)
        previews.append(preview.detach().cpu())

    return classified_meshes, torch.stack(previews).to(dtype=torch.float32)

from __future__ import annotations

from typing import Any

import torch

from ..cameras.intrinsics import validate_vipe_intrinsics
from ..datatypes import Device
from ..pytorch3d import require_pytorch3d


def classify_visible(
    mesh: Any,
    camera: Any,
    threshold: float,
    device: Device,
) -> torch.Tensor:
    """Classify a vertex as visible only when all adjacent faces face the camera."""
    verts = mesh.verts_packed().to(device)
    faces = mesh.faces_packed().to(device)
    normals = mesh.faces_normals_packed().to(device)
    if verts.numel() == 0 or faces.numel() == 0:
        raise ValueError("mesh must contain vertices and faces")

    centroids = (verts[faces[:, 0]] + verts[faces[:, 1]] + verts[faces[:, 2]]) / 3.0
    camera_center = camera.get_camera_center().to(device)
    to_camera = torch.nn.functional.normalize(camera_center - centroids, dim=1)
    face_visible = ((normals * to_camera).sum(dim=1) > threshold).float()
    vertex_indices = faces.reshape(-1, 1)
    vertex_visible = torch.ones((verts.shape[0], 1), device=device)
    return vertex_visible.scatter_reduce(
        dim=0,
        index=vertex_indices,
        src=face_visible.repeat_interleave(3).reshape(-1, 1),
        reduce="amin",
        include_self=False,
    )


def add_classification_visible(
    meshes: list[Any],
    height: int,
    width: int,
    intrinsics: dict,
    znear: float,
    zfar: float,
    threshold: float,
    device: Device = "cpu",
) -> tuple[list[Any], torch.Tensor]:
    """Replace the visibility feature channel and return ComfyUI previews."""
    if not meshes:
        raise ValueError("meshes must not be empty")
    if height <= 0 or width <= 0 or znear <= 0 or zfar <= znear:
        raise ValueError("invalid image dimensions or camera clipping range")

    pytorch3d = require_pytorch3d()
    values = validate_vipe_intrinsics(intrinsics, len(meshes)).to(device)
    rotation, translation = pytorch3d.look_at_view_transform(
        eye=((0, 0, 0),), at=((0, 0, -1),), up=((0, 1, 0),), device=device
    )
    classified_meshes = []
    previews = []
    for index, mesh in enumerate(meshes):
        camera = pytorch3d.PerspectiveCameras(
            focal_length=values[index:index + 1, :2],
            principal_point=values[index:index + 1, 2:],
            image_size=((height, width),),
            in_ndc=False,
            R=rotation,
            T=translation,
            device=device,
        )
        if mesh.textures is None or not hasattr(mesh.textures, "verts_features_list"):
            raise ValueError("mesh must contain vertex features")
        features = mesh.textures.verts_features_list()[0].to(device).clone()
        if features.ndim != 2 or features.shape[1] != 5:
            raise ValueError("mesh features must use [R, G, B, alpha, visible]")
        visible = classify_visible(mesh, camera, threshold, device)
        if visible.shape[0] != features.shape[0] or visible.shape[0] != height * width:
            raise ValueError("mesh vertex count must equal height * width")
        features[:, 4:5] = visible
        verts = mesh.verts_list()[0].to(device)
        faces = mesh.faces_list()[0].to(device)
        classified_meshes.append(
            pytorch3d.Meshes(
                verts=[verts],
                faces=[faces],
                textures=pytorch3d.TexturesVertex(verts_features=features.unsqueeze(0)),
            )
        )
        preview = visible.reshape(height, width, 1).repeat(1, 1, 3)
        previews.append(preview.detach().cpu())

    return classified_meshes, torch.stack(previews).to(dtype=torch.float32)

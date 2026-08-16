from typing import Any

from ...pytorch3d import require_pytorch3d


def save_mesh_ply(mesh: Any, path: str) -> str:
    """Save a mesh as PLY after reducing fixed vertex features to RGB."""
    if mesh.textures is None or not hasattr(mesh.textures, "verts_features_list"):
        raise ValueError("mesh must contain vertex features")
    features = mesh.textures.verts_features_list()[0]
    if features.ndim != 2 or features.shape[1] < 3:
        raise ValueError("mesh vertex features must contain RGB channels")
    pytorch3d = require_pytorch3d()
    mesh_rgb = pytorch3d.Meshes(
        verts=[mesh.verts_list()[0].detach().cpu()],
        faces=[mesh.faces_list()[0].detach().cpu()],
        textures=pytorch3d.TexturesVertex(
            verts_features=features[:, :3].detach().cpu().unsqueeze(0)
        ),
    )
    pytorch3d.IO().save_mesh(mesh_rgb, path)
    return path

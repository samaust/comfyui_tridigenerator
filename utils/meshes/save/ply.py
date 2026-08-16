from pytorch3d.io import IO
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes


def save_mesh_ply(mesh: Meshes, path: str) -> str:
    """Save a mesh as PLY after reducing fixed vertex features to RGB."""
    if mesh.textures is None or not hasattr(mesh.textures, "verts_features_list"):
        raise ValueError("mesh must contain vertex features")
    features = mesh.textures.verts_features_list()[0]
    if features.ndim != 2 or features.shape[1] < 3:
        raise ValueError("mesh vertex features must contain RGB channels")
    mesh_rgb = Meshes(
        verts=[mesh.verts_list()[0].detach().cpu()],
        faces=[mesh.faces_list()[0].detach().cpu()],
        textures=TexturesVertex(verts_features=features[:, :3].detach().cpu().unsqueeze(0)),
    )
    IO().save_mesh(mesh_rgb, path)
    return path

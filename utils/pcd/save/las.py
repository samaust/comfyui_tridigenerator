import datetime

import numpy as np
import pdal


def _mesh_arrays(mesh, expected_vertices: int):
    verts = mesh.verts_list()[0].detach().cpu().numpy()
    if verts.shape != (expected_vertices, 3):
        raise ValueError("mesh vertex count must equal height * width")
    if mesh.textures is None or not hasattr(mesh.textures, "verts_features_list"):
        raise ValueError("mesh must contain vertex features")
    features = mesh.textures.verts_features_list()[0].detach().cpu().numpy()
    if features.shape != (expected_vertices, 5):
        raise ValueError("mesh features must use [R, G, B, alpha, visible]")
    if not np.isfinite(verts).all() or not np.isfinite(features).all():
        raise ValueError("mesh data must contain finite values")
    return verts, features


def save_las(meshes, height: int, width: int, save_path: str, file_format: str) -> int:
    """Save a mesh sequence to one LAS/LAZ stream and return the point count."""
    if not meshes:
        raise ValueError("meshes must not be empty")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if file_format not in ("las", "laz"):
        raise ValueError("file_format must be las or laz")
    if len(meshes) > np.iinfo(np.uint16).max + 1:
        raise ValueError("LAS PointSourceId supports at most 65536 frames")

    chunk_size = height * width
    mesh_iterator = iter(enumerate(meshes))
    output = np.zeros(
        chunk_size,
        dtype=[
            ("X", "f8"),
            ("Y", "f8"),
            ("Z", "f8"),
            ("Classification", "u1"),
            ("PointSourceId", "u2"),
            ("Red", "u2"),
            ("Green", "u2"),
            ("Blue", "u2"),
        ],
    )

    def load_next_chunk():
        try:
            frame_index, mesh = next(mesh_iterator)
        except StopIteration:
            return 0
        verts, features = _mesh_arrays(mesh, chunk_size)
        colors = np.rint(np.clip(features[:, :3], 0.0, 1.0) * 65535).astype(np.uint16)
        classified_visible = (features[:, 3] >= 0.5) & (features[:, 4] >= 0.5)

        output["X"] = verts[:, 0]
        output["Y"] = verts[:, 1]
        output["Z"] = verts[:, 2]
        output["Classification"] = np.where(classified_visible, 3, 7).astype(np.uint8)
        output["PointSourceId"] = np.uint16(frame_index)
        output["Red"] = colors[:, 0]
        output["Green"] = colors[:, 1]
        output["Blue"] = colors[:, 2]
        return chunk_size

    today = datetime.date.today()
    pipeline = pdal.Writer.las(
        filename=save_path,
        creation_doy=today.timetuple().tm_yday,
        creation_year=today.year,
        dataformat_id=2,
        compression="true" if file_format == "laz" else "false",
        scale_x=0.000001,
        scale_y=0.000001,
        scale_z=0.000001,
        offset_x=0,
        offset_y=0,
        offset_z=0,
    ).pipeline()
    pipeline.inputs = [(output, load_next_chunk)]
    return pipeline.execute_streaming(chunk_size)

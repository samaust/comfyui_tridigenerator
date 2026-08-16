import base64

import numpy as np

# Using a fork of pygltflib
# https://github.com/samaust/pygltflib refactor branch
from pygltflib.v2.schema import (
    Accessor,
    Accessor_componentType,
    Attributes,
    Buffer,
    BufferView,
    BufferView_target,
    DATA_URI_HEADER,
    Material,
    Material_alphaMode,
    MaterialPbrMetallicRoughness,
    Mesh,
    MeshPrimitive,
    Mesh_primitive_mode,
    Node,
    Scene,
)
from pygltflib import GLTF2

from ..images.colors import apply_gamma_correction


def get_uri_from_blob(blob):
    data = base64.b64encode(blob).decode('utf-8')
    uri = f'{DATA_URI_HEADER}{data}'

    return uri


def create_gltf(positions, colors) -> GLTF2:
    """
    Saves a point cloud with vertex colors to a GLTF file.

    Parameters:
    - in_positions (numpy ndarray): ndarray of points positions.
    - in_colors (numpy ndarray): ndarray of points colors
    - save_dir (str) : absolute path to save directory
    """
    frames_positions = np.asarray(positions, dtype=np.float32)
    frames_colors_input = np.asarray(colors, dtype=np.float32)
    if frames_positions.ndim != 3 or frames_positions.shape[-1] != 3:
        raise ValueError("positions must have shape [frames, vertices, 3]")
    if frames_colors_input.ndim != 3 or frames_colors_input.shape[-1] not in (3, 4):
        raise ValueError("colors must have shape [frames, vertices, 3|4]")
    if frames_positions.shape[:2] != frames_colors_input.shape[:2]:
        raise ValueError("positions and colors must share frame and vertex counts")
    if frames_positions.shape[0] == 0 or frames_positions.shape[1] == 0:
        raise ValueError("positions and colors must not be empty")
    if not np.isfinite(frames_positions).all() or not np.isfinite(frames_colors_input).all():
        raise ValueError("positions and colors must contain finite values")
    if frames_colors_input.shape[-1] == 3:
        alpha = np.ones((*frames_colors_input.shape[:2], 1), dtype=np.float32)
        frames_colors_input = np.concatenate((frames_colors_input, alpha), axis=-1)

    frames_colors_original = np.rint(
        np.clip(frames_colors_input, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    frames_colors = []
    for frame_colors_original in frames_colors_original:
        frames_colors.append(apply_gamma_correction(frame_colors_original, gamma=2.0))
    frames_colors = np.stack(frames_colors)

    # Get binary blobs data
    positions_binary_blob = frames_positions.tobytes()
    colors_binary_blob = frames_colors.tobytes()

    # Caculate buffers byteLength
    positions_buffer_byteLength = len(positions_binary_blob)
    colors_buffer_byteLength = len(colors_binary_blob)

    # Create buffers to store point and color data
    position_buffer = Buffer(
        uri=get_uri_from_blob(positions_binary_blob),
        byteLength=positions_buffer_byteLength
    )  # 4 bytes per float32
    color_buffer = Buffer(
        uri=get_uri_from_blob(colors_binary_blob),
        byteLength=colors_buffer_byteLength
    )  # 1 byte per uint8

    scenes = []
    bufferViews = []
    accessors = []
    meshes = []
    nodes = []
    for i, (frame_positions, frame_colors) in enumerate(zip(frames_positions, frames_colors)):
        # Calculate positions count and byteLength
        positions_count = len(frame_positions)
        positions_bufferView_byteLength = len(frame_positions.flatten()) * 4  # np.float32

        # Calculate colors count
        colors_count = len(frame_colors)
        colors_bufferView_byteLength = len(frame_colors.flatten())  # np.uint8

        # Create buffer views
        point_buffer_view = BufferView(
            buffer=0,
            byteOffset=i*positions_bufferView_byteLength,
            byteLength=positions_bufferView_byteLength,
            target=BufferView_target.ARRAY_BUFFER.value
        )
        color_buffer_view = BufferView(
            buffer=1,
            byteOffset=i*colors_bufferView_byteLength,
            byteLength=colors_bufferView_byteLength,
            target=BufferView_target.ARRAY_BUFFER.value
        )
        bufferViews.append(point_buffer_view)
        bufferViews.append(color_buffer_view)

        # Create accessor for point cloud positions
        positions_max = np.max(frame_positions, 0).tolist()
        positions_min = np.min(frame_positions, 0).tolist()
        position_accessor = Accessor(
            bufferView=i*2,
            byteOffset=0,
            componentType=Accessor_componentType.FLOAT.value,
            count=positions_count,
            type="VEC3",
            max=positions_max,
            min=positions_min
        )
        # Create accessor for point cloud colors
        color_accessor = Accessor(
            bufferView=i*2+1,
            byteOffset=0,
            componentType=Accessor_componentType.UNSIGNED_BYTE.value,
            normalized=True,
            count=colors_count,
            type="VEC4"  # RGBA
        )
        accessors.append(position_accessor)
        accessors.append(color_accessor)

        # Create mesh primitives for points and colors
        attributes = Attributes(
            POSITION=i*2,
            COLOR_0=i*2+1
        )
        primitive = MeshPrimitive(
            attributes=attributes,
            indices=None,
            mode=Mesh_primitive_mode.POINTS.value,
            material=0
        )
        # Create the mesh object
        meshes.append(Mesh(
            name="Object_{}".format(i),
            primitives=[primitive]
        ))

        # Create the node object
        nodes.append(Node(mesh=i))

        # Create the scene
        scenes.append(Scene(nodes=[i]))

    # Create a simple material using the vertex colors
    material = Material(
        name="vertexColorMaterial",
        doubleSided=None,
        pbrMetallicRoughness=MaterialPbrMetallicRoughness(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=None,
            roughnessFactor=None
        ),
        alphaMode=Material_alphaMode.MASK.value,
        alphaCutoff=0.5,
        emissiveFactor=None,
        extensions={"KHR_materials_unlit": {}},  # Optional: Use unlit material (no lighting)
    )
    extensionsUsed = ["KHR_materials_unlit"]

    # Create the GLTF object
    gltf = GLTF2()

    # Add everything to the GLTF object
    gltf.extensionsUsed = extensionsUsed
    gltf.buffers = [position_buffer, color_buffer]
    gltf.bufferViews = bufferViews
    gltf.accessors = accessors
    gltf.materials = [material]  # Add the unlit material
    gltf.meshes = meshes
    gltf.nodes = nodes
    gltf.scenes = scenes
    gltf.scene = 0

    return gltf

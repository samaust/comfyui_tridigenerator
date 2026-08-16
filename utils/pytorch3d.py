"""Runtime loading and diagnostics for PyTorch3D's compiled extension."""

from dataclasses import dataclass
from functools import lru_cache
import sys
from typing import Any

import torch


_PYTORCH3D_REVISION = "d9839a9"


@dataclass(frozen=True)
class PyTorch3D:
    IO: Any
    Meshes: Any
    MeshRasterizer: Any
    PerspectiveCameras: Any
    RasterizationSettings: Any
    TexturesVertex: Any
    interpolate_face_attributes: Any
    look_at_view_transform: Any


@lru_cache(maxsize=1)
def require_pytorch3d() -> PyTorch3D:
    """Load PyTorch3D lazily and turn native ABI failures into actionable errors."""
    try:
        # Importing _C explicitly detects extensions compiled for a different Torch ABI.
        from pytorch3d import _C  # noqa: F401
        from pytorch3d.io import IO
        from pytorch3d.ops import interpolate_face_attributes
        from pytorch3d.renderer import (
            MeshRasterizer,
            PerspectiveCameras,
            RasterizationSettings,
            TexturesVertex,
            look_at_view_transform,
        )
        from pytorch3d.structures import Meshes
    except (ImportError, OSError) as exc:
        command = (
            f'{sys.executable} -m pip install --force-reinstall --no-deps '
            f'--no-build-isolation --no-cache-dir '
            f'"git+https://github.com/facebookresearch/pytorch3d.git@{_PYTORCH3D_REVISION}"'
        )
        raise RuntimeError(
            "PyTorch3D could not load its compiled extension against the active "
            f"Torch {torch.__version__}. Rebuild PyTorch3D after installing or "
            f"updating Torch, then restart ComfyUI. Run: {command}"
        ) from exc

    return PyTorch3D(
        IO=IO,
        Meshes=Meshes,
        MeshRasterizer=MeshRasterizer,
        PerspectiveCameras=PerspectiveCameras,
        RasterizationSettings=RasterizationSettings,
        TexturesVertex=TexturesVertex,
        interpolate_face_attributes=interpolate_face_attributes,
        look_at_view_transform=look_at_view_transform,
    )

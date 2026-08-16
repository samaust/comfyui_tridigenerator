# Depth
from .depth.models import (
    TDGenDepthLoadModelDepthAnythingV2Metric,
    TDGenDepthLoadModelVideoDepthAnything
)
from .depth.generation import (
    TDGenDepthEstimateDepthAnythingV2Metric,
    TDGenDepthEstimateVideoDepthAnything,
)
from .depth.calibration import (
    TDGenDepthCalibrateConstantsAutomatic,
    TDGenDepthCalibrateConstantsManual,
    TDGenDepthCalibrateMetric
)
from .depth.measurements import (
    TDGenDepthMeasureDistance
)

# PCD
from .pcd.generation import (
    TDGenPcdCreatePcd
)
from .pcd.save.las import (
    TDGenPcdSaveLas
)

# glTF
from .gltf.generation import (
    TDGenGltfCreateGltf
)
from .gltf.save import (
    TDGenGltfSaveGltf
)

# Meshes generation
from .meshes.generation import (
    TDGenMeshesCreateMeshes
)
# Meshes save
from .meshes.save.ply import (
    TDGenMeshesSavePly
)

# Classification
from .classification.alpha import TDGenClassificationAlpha
from .classification.visible import TDGenClassificationVisible

# Stereoscopic
from .stereoscopic.models import (
    TDGenStereoscopicLoadModelSvdStereoCrafter
)
from .stereoscopic.generation.inpaint import (
    TDGenStereoscopicTiledInpaint
)

# Videos
from .videos.save.exr import (
    TDGenSaveVideoExr
)
from .videos.load.exr import (
    TDGenLoadVideoExr
)

NODE_CLASS_MAPPINGS = {
    # Depth models
    "TDGenDepthLoadModelDepthAnythingV2Metric": TDGenDepthLoadModelDepthAnythingV2Metric,
    "TDGenDepthLoadModelVideoDepthAnything": TDGenDepthLoadModelVideoDepthAnything,
    # Depth estimation
    "TDGenDepthEstimateDepthAnythingV2Metric": TDGenDepthEstimateDepthAnythingV2Metric,
    "TDGenDepthEstimateVideoDepthAnything": TDGenDepthEstimateVideoDepthAnything,
    # Depth calibration
    "TDGenDepthCalibrateConstantsAutomatic": TDGenDepthCalibrateConstantsAutomatic,
    "TDGenDepthCalibrateConstantsManual": TDGenDepthCalibrateConstantsManual,
    "TDGenDepthCalibrateMetric": TDGenDepthCalibrateMetric,
    # Depth measurements
    "TDGenDepthMeasureDistance": TDGenDepthMeasureDistance,

    # PCD generation
    "TDGenPcdCreatePcd": TDGenPcdCreatePcd,
    # PCD save
    "TDGenPcdSaveLas": TDGenPcdSaveLas,

    # glTF
    "TDGenGltfCreateGltf": TDGenGltfCreateGltf,
    "TDGenGltfSaveGltf": TDGenGltfSaveGltf,

    # Meshes generation
    "TDGenMeshesCreateMeshes": TDGenMeshesCreateMeshes,
    # Meshes save
    "TDGenMeshesSavePly": TDGenMeshesSavePly,

    # Classification
    "TDGenClassificationAlpha": TDGenClassificationAlpha,
    "TDGenClassificationVisible": TDGenClassificationVisible,

    # Stereoscopic models
    "TDGenStereoscopicLoadModelSvdStereoCrafter": TDGenStereoscopicLoadModelSvdStereoCrafter,
    # Stereoscopic inpaint
    "TDGenStereoscopicTiledInpaint": TDGenStereoscopicTiledInpaint,

    # Videos
    "TDGenSaveVideoExr": TDGenSaveVideoExr,
    "TDGenLoadVideoExr": TDGenLoadVideoExr,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    # Depth models
    "TDGenDepthLoadModelDepthAnythingV2Metric": "TDGen Depth - Load Model DepthAnythingV2 Metric",
    "TDGenDepthLoadModelVideoDepthAnything": "TDGen Depth - Load Model VideoDepthAnything",
    # Depth estimation
    "TDGenDepthEstimateDepthAnythingV2Metric": "TDGen Depth - Estimate DepthAnythingV2 Metric",
    "TDGenDepthEstimateVideoDepthAnything": "TDGen Depth - Estimate VideoDepthAnything",
    # Depth calibration
    "TDGenDepthCalibrateConstantsAutomatic": "TDGen Depth - Calibrate constants - Automatic",
    "TDGenDepthCalibrateConstantsManual": "TDGen Depth - Calibrate constants - Manual",
    "TDGenDepthCalibrateMetric": "TDGen Depth - Calibrate Metric",
    # Measure
    "TDGenDepthMeasureDistance": "TDGen Depth - Measure Distance",

    # PCD generation
    "TDGenPcdCreatePcd": "TDGen PCD - Create PCD",
    # PCD save
    "TDGenPcdSaveLas": "TDGen PCD - Save PCD",

    # glTF
    "TDGenGltfCreateGltf": "TDGen glTF - Create glTF",
    "TDGenGltfSaveGltf": "TDGen glTF - Save glTF",

    # Meshes generation
    "TDGenMeshesCreateMeshes": "TDGen Meshes - Create Meshes",
    # Meshes save
    "TDGenMeshesSavePly": "TDGen Meshes - Save Ply",

    # Classification
    "TDGenClassificationAlpha": "TDGen Classification - Alpha",
    "TDGenClassificationVisible": "TDGen Classification - Visible",

    # Stereoscopic models
    "TDGenStereoscopicLoadModelSvdStereoCrafter": "TDGen Stereoscopic - (Down)load Model Svd StereoCrafter",
    # Stereoscopic inpaint
    "TDGenStereoscopicTiledInpaint": "TDGen Stereoscopic - Tiled Inpaint",

    # Videos
    "TDGenSaveVideoExr": "TDGen Video - Save Exr",
    "TDGenLoadVideoExr": "TDGen Video - Load Exr",
}

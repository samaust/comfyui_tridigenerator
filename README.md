# ComfyUI-TridiGenerator

ComfyUI nodes for turning image sequences and metric depth into colored point clouds, meshes, glTF, LAS/LAZ, and OpenEXR video sequences. Experimental browser assets are included for future AR/VR viewing work.

## Architecture

ComfyUI-facing classes live under `nodes/`. They declare inputs and outputs, resolve ComfyUI paths and devices, and call reusable implementations under the matching themed `utils/` directory. `nodes/nodes.py` owns the node registry exported by the root `__init__.py`.

The `PCD` value passed between nodes contains two CPU `float32` NumPy arrays:

- positions: `[frames, height * width, 3]`
- colors: `[frames, height * width, 3]` or `[frames, height * width, 4]`

Generated meshes use five vertex feature channels in this order: red, green, blue, alpha, and visibility. Alpha and visibility default to `1` and can be replaced independently by classification nodes.

## Installation

ComfyUI custom nodes run from the `custom_nodes` directory. Clone this repository there, activate the Python environment used by ComfyUI, and install its requirements:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/samaust/ComfyUI-TridiGenerator.git
cd ComfyUI-TridiGenerator
python3 -m pip install -r requirements.txt
```

Install PyTorch3D using the separate step below, then restart ComfyUI and check
its console for import failures.

### Native and CUDA dependencies

PyTorch3D contains a native extension and must be built against the exact Torch
installation used by ComfyUI. Install it only after `requirements.txt` (and
after every future Torch upgrade):

```bash
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"
python3 -m pip uninstall -y pytorch3d
MAX_JOBS=4 python3 -m pip install --force-reinstall --no-deps --no-build-isolation --no-cache-dir -r requirements-pytorch3d.txt
python3 -c "import torch; from pytorch3d import _C; print('PyTorch3D extension loaded with', torch.__version__)"
```

An error mentioning an undefined C++ symbol (including
`materialize_cow_storage`) means the installed PyTorch3D extension was compiled
for a different Torch ABI. Re-running ordinary `requirements.txt` installation
does not replace an already satisfied native package; uninstall and rebuild it
with the commands above. `--no-deps` prevents the PyTorch3D installation step
from replacing ComfyUI's Torch build.

PDAL and its Python bindings require native PDAL libraries. On Ubuntu, install the required compiler, GDAL, GeoTIFF, LASzip, TIFF, SQLite, cURL, and PROJ development packages before installing `pdal`.

StereoCrafter model loading downloads Hugging Face model data and requires a CUDA environment compatible with its Torch dependencies.

### Depth models

Place supported safetensors files under the ComfyUI model directory:

- `models/depthanything/depth_anything_v2_metric_hypersim_vitl_fp32.safetensors`
- `models/video_depth_anything/video_depth_anything_vitl.safetensors`
- `models/video_depth_anything/video_depth_anything_vits.safetensors`

## Nodes

| Group | Node | Purpose |
| --- | --- | --- |
| Depth / Models | `TDGenDepthLoadModelDepthAnythingV2Metric` | Load a metric Depth Anything V2 model |
| Depth / Models | `TDGenDepthLoadModelVideoDepthAnything` | Load a Video Depth Anything model |
| Depth / Generation | `TDGenDepthEstimateDepthAnythingV2Metric` | Estimate metric depth |
| Depth / Generation | `TDGenDepthEstimateVideoDepthAnything` | Estimate normalized inverse depth |
| Depth / Calibration | `TDGenDepthCalibrateConstantsAutomatic` | Fit scale and shift from metric and normalized inverse depth |
| Depth / Calibration | `TDGenDepthCalibrateConstantsManual` | Derive scale and shift from a manual depth range |
| Depth / Calibration | `TDGenDepthCalibrateMetric` | Convert normalized inverse depth to metric depth |
| Depth / Measurements | `TDGenDepthMeasureDistance` | Measure paired pixels in the first metric-depth frame |
| PCD / Generation | `TDGenPcdCreatePcd` | Create a point-cloud sequence and apply optional frame selection |
| PCD / Save | `TDGenPcdSaveLas` | Save a mesh sequence to one LAS or LAZ file |
| glTF / Generation | `TDGenGltfCreateGltf` | Convert a `PCD` sequence to glTF |
| glTF / Save | `TDGenGltfSaveGltf` | Save glTF with a unique ComfyUI output path |
| Meshes / Generation | `TDGenMeshesCreateMeshes` | Create grid meshes with fixed vertex features |
| Meshes / Save | `TDGenMeshesSavePly` | Save every incoming mesh as an RGB PLY file |
| Classification | `TDGenClassificationAlpha` | Replace per-vertex alpha from masks |
| Classification | `TDGenClassificationVisible` | Replace per-vertex visibility from face angles |
| Rendering | `TDGenRenderTexturedMesh` | Render synchronized ViPE RGB-D data with poses and per-frame pinhole intrinsics |
| Stereoscopic / Models | `TDGenStereoscopicLoadModelSvdStereoCrafter` | Load the SVD StereoCrafter pipeline |
| Stereoscopic / Generation | `TDGenStereoscopicTiledInpaint` | Inpaint warped stereoscopic frames |
| Videos / Save | `TDGenSaveVideoExr` | Save one-, three-, or four-channel EXR video sequences |
| Videos / Load | `TDGenLoadVideoExr` | Load EXR video sequences as ComfyUI images |

## Modular workflows

The point-cloud path is:

```text
depth model -> depth estimation -> calibration -> PCD creation
```

From the `PCD` output, use either branch:

```text
PCD -> glTF creation -> glTF save
PCD -> mesh creation -> alpha/visibility classification -> PLY or LAS/LAZ save
```

`index` is applied only by `TDGenPcdCreatePcd`: `-1` processes every frame and a non-negative index creates a one-frame `PCD`. Downstream nodes always process the collection they receive.

The OpenEXR load/save nodes and StereoCrafter nodes can be used independently of the point-cloud chain.

The ViPE render path is:

```text
VipeExtractData -> VipeSelectMasks -> TDGenRenderTexturedMesh
```

Connect the extracted images, metric depths, and JSON metadata directly, plus the selected binary mask. The renderer validates that all streams and metadata describe the same frames and resolution.

## Current limitations and future work

- glTF embeds point and color buffers and is practical only for relatively short sequences.
- Geometry nodes consume per-frame `VIPE_INTRINSICS` from `comfyui_vipe`; field-of-view approximation is intentionally unsupported.
- `TDGenRenderTexturedMesh` renders synchronized RGB, masks, metric depth, and ViPE metadata using per-frame pinhole intrinsics. It rebases every camera-to-world pose into the first camera's coordinate system, where the displacement controls are measured in metres and rotation uses yaw (Y), pitch (X), then roll (Z), in degrees.
- Stereoscopic mesh rendering and browser/AR/VR integration remain experimental and are not registered as ComfyUI nodes.
- Model inference, PyTorch3D, StereoCrafter, and PDAL workflows require their full runtime dependencies and suitable hardware.

## Development validation

```bash
python3 -m compileall -q __init__.py nodes utils tests
python3 -m unittest discover -s tests -v
git diff --check
```

Runtime changes should also be checked by starting ComfyUI and exercising the affected workflow.

## Acknowledgements

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything)
- [StereoCrafter](https://github.com/TencentARC/StereoCrafter)
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d)

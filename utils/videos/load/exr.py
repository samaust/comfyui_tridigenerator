import numpy as np
import OpenImageIO as oiio
import torch


def load_video_exr(path: str) -> torch.Tensor:
    """Load a multipart OpenEXR video as a CPU float32 tensor."""
    image_input = oiio.ImageInput.open(path)
    if not image_input:
        raise RuntimeError(f"failed to open EXR video: {oiio.geterror()}")
    frames = []
    try:
        index = 0
        while image_input.seek_subimage(index, 0):
            spec = image_input.spec()
            frame = image_input.read_image(spec.format)
            if frame is None:
                raise RuntimeError(
                    f"failed to read EXR subimage {index}: {image_input.geterror()}"
                )
            frames.append(frame)
            index += 1
    finally:
        image_input.close()

    if not frames:
        raise RuntimeError("EXR video contains no subimages")
    video = np.stack(frames).astype(np.float32, copy=False)
    return torch.from_numpy(video)

import numpy as np
import OpenImageIO as oiio


CHANNEL_NAMES = {
    1: ("Z",),
    3: ("R", "G", "B"),
    4: ("R", "G", "B", "A"),
}


def save_video_exr(images, path: str, bit_depth: str, compression: str) -> None:
    """Save a tensor as a multipart OpenEXR video sequence."""
    if images.ndim == 3:
        images = images.unsqueeze(-1)
    if images.ndim != 4:
        raise ValueError("images must have shape [frames, height, width, channels]")
    if images.shape[0] == 0 or images.shape[1] == 0 or images.shape[2] == 0:
        raise ValueError("images must not be empty")

    channel_count = images.shape[3]
    try:
        channel_names = CHANNEL_NAMES[channel_count]
    except KeyError as error:
        raise ValueError("EXR videos support 1, 3, or 4 channels") from error

    if bit_depth == "16":
        images_np = images.detach().cpu().numpy().astype(np.float16)
        pixel_type = oiio.HALF
    elif bit_depth == "32":
        images_np = images.detach().cpu().numpy().astype(np.float32)
        pixel_type = oiio.FLOAT
    else:
        raise ValueError("bit_depth must be 16 or 32")

    frame_count, height, width, _ = images_np.shape
    specs = []
    for index in range(frame_count):
        spec = oiio.ImageSpec(width, height, channel_count, pixel_type)
        spec.attribute("Software", "comfyui_tridigenerator")
        spec.attribute("compression", compression)
        spec.channelnames = channel_names
        if channel_count == 1:
            spec.z_channel = 0
        spec.attribute("oiio:subimagename", f"{index:05}")
        specs.append(spec)

    output = oiio.ImageOutput.create(path)
    if output is None:
        raise RuntimeError(f"failed to create EXR output: {oiio.geterror()}")
    try:
        for index, frame in enumerate(images_np):
            opened = (
                output.open(path, tuple(specs))
                if index == 0
                else output.open(path, specs[index], "AppendSubimage")
            )
            if not opened:
                raise RuntimeError(f"failed to open EXR subimage {index}: {output.geterror()}")
            if not output.write_image(frame):
                raise RuntimeError(f"failed to write EXR subimage {index}: {output.geterror()}")
    finally:
        output.close()

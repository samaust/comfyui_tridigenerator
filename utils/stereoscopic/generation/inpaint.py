from stereocrafter.inference.inpainting import tiled_inpaint


def inpaint_stereoscopic_video(
    frames_warped,
    frames_mask,
    pipeline,
    resize_size,
    fps,
    frames_chunk,
    overlap,
    tile_num,
    num_inference_steps,
):
    return tiled_inpaint(
        frames_warped=frames_warped,
        frames_mask=frames_mask,
        pipeline=pipeline,
        resize_size=resize_size,
        fps=fps,
        frames_chunk=frames_chunk,
        overlap=overlap,
        tile_num=tile_num,
        num_inference_steps=num_inference_steps,
    )

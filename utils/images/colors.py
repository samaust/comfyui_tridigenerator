import numpy as np


def apply_gamma_correction(image: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """Apply gamma correction to RGB and preserve any additional channels."""
    image = np.asarray(image)
    if image.ndim < 2 or image.shape[-1] < 3:
        raise ValueError("image must have at least three color channels")
    rgb = image[..., :3]
    rgb_corrected = np.power(rgb / 255.0, gamma) * 255.0
    if image.shape[-1] == 3:
        return rgb_corrected.astype(np.uint8)
    return np.concatenate((rgb_corrected, image[..., 3:]), axis=-1).astype(np.uint8)

import numpy as np


def horizontal_to_vertical_fov(
    horizontal_fov_deg: float,
    image_width: int,
    image_height: int,
) -> float:
    """Convert a horizontal field of view to its vertical equivalent."""
    if not 0.0 < horizontal_fov_deg < 180.0:
        raise ValueError("horizontal_fov_deg must be between 0 and 180 degrees")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    horizontal_fov_rad = np.deg2rad(horizontal_fov_deg)
    vertical_fov_rad = 2.0 * np.arctan(
        np.tan(horizontal_fov_rad / 2.0) * image_height / image_width
    )
    return float(np.rad2deg(vertical_fov_rad))


def compute_camera_matrix(
    fov_horizontal_deg: float | None,
    fov_vertical_deg: float | None,
    image_width: int,
    image_height: int
) -> np.ndarray:
    """Compute pinhole camera intrinsics from either horizontal or vertical FOV."""
    if fov_horizontal_deg is None and fov_vertical_deg is None:
        raise ValueError("at least one field of view must be provided")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    if fov_horizontal_deg is not None:
        if not 0.0 < fov_horizontal_deg < 180.0:
            raise ValueError("horizontal field of view must be between 0 and 180 degrees")
        fov_horizontal_rad = np.deg2rad(fov_horizontal_deg)
        fx = image_width / (2 * np.tan(fov_horizontal_rad / 2))

    if fov_vertical_deg is not None:
        if not 0.0 < fov_vertical_deg < 180.0:
            raise ValueError("vertical field of view must be between 0 and 180 degrees")
        fov_vertical_rad = np.deg2rad(fov_vertical_deg)
        fy = image_height / (2 * np.tan(fov_vertical_rad / 2))

    if fov_vertical_deg is None:
        fy = fx

    if fov_horizontal_deg is None:
        fx = fy

    cx = image_width / 2
    cy = image_height / 2
    camera_matrix = np.array([[fx,  0, cx],
                              [ 0, fy, cy],
                              [ 0,  0,  1]], dtype=np.float64)

    return camera_matrix

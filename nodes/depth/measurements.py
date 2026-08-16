import ast
import json

import numpy as np

from ...utils.depth.measurements import measure_distances


def _parse_coordinates(value: str, name: str) -> list[tuple[int, int]]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        try:
            decoded = ast.literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"{name} must be a JSON list of x/y objects") from error
    if not isinstance(decoded, list) or not decoded:
        raise ValueError(f"{name} must contain at least one coordinate")
    coordinates = []
    for item in decoded:
        if not isinstance(item, dict) or "x" not in item or "y" not in item:
            raise ValueError(f"{name} entries must contain x and y")
        coordinates.append((item["x"], item["y"]))
    return coordinates


class TDGenDepthMeasureDistance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "metric_depths": ("IMAGE",),
                "intrinsics": ("VIPE_INTRINSICS",),
                "coordinates_positive": ("STRING", {"forceInput": True}),
                "coordinates_negative": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("distance_m",)
    FUNCTION = "run"
    CATEGORY = "TDGen/Depth/Measurements"
    DESCRIPTION = "Measure 3D distances between paired pixels in the first depth frame."

    def run(self, metric_depths, intrinsics, coordinates_positive, coordinates_negative):
        starts = _parse_coordinates(coordinates_positive, "coordinates_positive")
        ends = _parse_coordinates(coordinates_negative, "coordinates_negative")
        if len(starts) != len(ends):
            raise ValueError("positive and negative coordinate lists must have equal length")
        coordinate_pairs = np.asarray(list(zip(starts, ends)), dtype=np.float64)
        distances = measure_distances(metric_depths, intrinsics, coordinate_pairs)

        payload = {}
        for index, (pair, distance) in enumerate(zip(coordinate_pairs.tolist(), distances)):
            payload[f"point_{index}"] = {
                "s": {"x": pair[0][0], "y": pair[0][1]},
                "e": {"x": pair[1][0], "y": pair[1][1]},
                "sp": {"x": distance[0], "y": distance[1], "z": distance[2]},
                "ep": {"x": distance[3], "y": distance[4], "z": distance[5]},
                "d": {
                    "x": distance[6],
                    "y": distance[7],
                    "z": distance[8],
                    "total": distance[9],
                },
            }
        return (json.dumps(payload),)

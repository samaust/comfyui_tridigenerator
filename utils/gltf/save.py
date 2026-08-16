import json


def save_gltf(gltf, path: str) -> str:
    """Save a glTF object to an explicit output path."""
    gltf.save(path)
    return path


def pretty_print_gltf_json(gltf) -> str:
    """Return formatted glTF JSON."""
    gltf_json = gltf.gltf_to_json()
    json_object = json.loads(gltf_json)
    json_formatted_str = json.dumps(json_object, indent=2)

    return json_formatted_str

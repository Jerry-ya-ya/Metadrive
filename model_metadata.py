import json
from pathlib import Path


def saved_model_path(path_value):
    path = Path(path_value)
    return path if path.suffix.lower() == ".zip" else Path(f"{path}.zip")


def metadata_path(path_value):
    return saved_model_path(path_value).with_suffix(".metadata.json")


def load_model_metadata(path_value):
    path = metadata_path(path_value)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Model metadata must contain a JSON object: {path}")
    return data


def save_model_metadata(path_value, **values):
    path = metadata_path(path_value)
    data = load_model_metadata(path_value)
    data.update({key: value for key, value in values.items() if value is not None})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
    return path


def resolve_model_map(path_value, requested_map=None, *, model=None, default_map="SC"):
    if requested_map is not None and str(requested_map).strip():
        return str(requested_map).strip()

    metadata_map = load_model_metadata(path_value).get("map")
    if metadata_map is not None and str(metadata_map).strip():
        return str(metadata_map).strip()

    model_map = getattr(model, "metadrive_map", None) if model is not None else None
    if model_map is not None and str(model_map).strip():
        return str(model_map).strip()

    return str(default_map).strip()

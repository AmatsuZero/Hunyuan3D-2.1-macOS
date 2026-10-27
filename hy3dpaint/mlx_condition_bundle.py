"""Validated Torch-free weights for the paint outer conditioning path."""

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile

import mlx.core as mx


CONDITION_BUNDLE_SCHEMA = "hy3d-paint-condition-mlx-v1"
CONDITION_SHAPES = {
    "image_proj_model_dino.proj.weight": (4096, 1536),
    "image_proj_model_dino.proj.bias": (4096,),
    "image_proj_model_dino.norm.weight": (1024,),
    "image_proj_model_dino.norm.bias": (1024,),
    "learned_text_clip_albedo": (77, 1024),
    "learned_text_clip_mr": (77, 1024),
    "learned_text_clip_ref": (77, 1024),
}


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_sha(value, name):
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"invalid {name}")


def _check_arrays(arrays):
    if set(arrays) != set(CONDITION_SHAPES):
        raise ValueError("condition keys mismatch")
    for name, expected_shape in CONDITION_SHAPES.items():
        value = arrays[name]
        if not isinstance(value, mx.array) or tuple(value.shape) != expected_shape:
            raise ValueError(f"{name} shape mismatch")
        if value.dtype != mx.float16:
            raise ValueError(f"{name} dtype mismatch")


def save_bundle(arrays, source_sha256, config_sha256, bundle_path, manifest_path):
    """Publish exactly the seven checkpoint condition tensors to a new bundle."""
    bundle_path, manifest_path = Path(bundle_path), Path(manifest_path)
    if bundle_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite condition bundle or manifest")
    _check_sha(source_sha256, "source_sha256")
    _check_sha(config_sha256, "config_sha256")
    _check_arrays(arrays)
    metadata = {
        "schema": CONDITION_BUNDLE_SCHEMA,
        "source_sha256": source_sha256,
        "config_sha256": config_sha256,
        "mlx_version": version("mlx"),
        "dtype": "float16",
        "parameter_count": str(len(CONDITION_SHAPES)),
    }
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{bundle_path.name}.",
                                         suffix=".safetensors", dir=bundle_path.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
        mx.save_safetensors(str(temporary), arrays, metadata=metadata)
        os.link(temporary, bundle_path)
        temporary.unlink()
        temporary = None
        manifest = {**metadata, "bundle_sha256": _sha256_file(bundle_path),
                    "parameter_shapes": {name: list(shape) for name, shape in CONDITION_SHAPES.items()}}
        with tempfile.NamedTemporaryFile(prefix=f".{manifest_path.name}.", suffix=".tmp",
                                         dir=manifest_path.parent, mode="w", encoding="utf-8",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.link(temporary, manifest_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_bundle(path, expected_source_sha256, expected_config_sha256):
    """Validate the exact paint condition checkpoint before MLX inference."""
    _check_sha(expected_source_sha256, "source_sha256")
    _check_sha(expected_config_sha256, "config_sha256")
    arrays, metadata = mx.load(str(path), return_metadata=True)
    if metadata.get("schema") != CONDITION_BUNDLE_SCHEMA:
        raise ValueError("condition bundle schema mismatch")
    if metadata.get("source_sha256") != expected_source_sha256:
        raise ValueError("source_sha256 mismatch")
    if metadata.get("config_sha256") != expected_config_sha256:
        raise ValueError("config_sha256 mismatch")
    if metadata.get("mlx_version") != version("mlx"):
        raise ValueError("mlx_version mismatch")
    if metadata.get("dtype") != "float16" or metadata.get("parameter_count") != "7":
        raise ValueError("condition metadata mismatch")
    _check_arrays(arrays)
    return arrays, metadata

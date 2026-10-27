#!/usr/bin/env python3
"""Convert the inner UNet checkpoint to a Torch-free MLX safetensors bundle.

Torch is intentionally imported only by this offline conversion entry point.
The produced bundle is consumed by ``hy3dpaint.mlx_weight_bundle``.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import mlx.core as mx
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hy3dpaint"))

from mlx_unet import HunyuanInnerUNet  # noqa: E402
from mlx_utils import load_weights_from_pytorch  # noqa: E402
from mlx_weight_bundle import build_metadata  # noqa: E402


EXCLUDED_INNER_KEYS = {
    "image_proj_model_dino.norm.bias",
    "image_proj_model_dino.norm.weight",
    "image_proj_model_dino.proj.bias",
    "image_proj_model_dino.proj.weight",
    "learned_text_clip_albedo",
    "learned_text_clip_mr",
    "learned_text_clip_ref",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flatten_tree(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            yield from flatten_tree(child, child_prefix)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from flatten_tree(child, f"{prefix}.{index}")
    else:
        yield prefix, value


def audit_source_coverage(source_keys: set[str], mapped_inner_keys: set[str]) -> dict:
    """Ensure every inner-UNet source key is mapped or explicitly documented."""
    inner_keys = {key.removeprefix("unet.") for key in source_keys if key.startswith("unet.")}
    mapped = {key.removeprefix("unet.") for key in mapped_inner_keys}
    excluded = sorted(inner_keys & EXCLUDED_INNER_KEYS)
    unmapped = sorted(inner_keys - mapped - EXCLUDED_INNER_KEYS)
    if unmapped:
        raise ValueError(f"unmapped inner checkpoint keys: {', '.join(unmapped)}")
    return {
        "source_inner_keys": len(inner_keys),
        "mapped_inner_keys": len(inner_keys & mapped),
        "excluded_inner_keys": excluded,
        "unet_dual_keys": sum(key.startswith("unet_dual.") for key in source_keys),
    }


def write_bundle(model, source_checkpoint: Path, config_path: Path, bundle_path: Path,
                 manifest_path: Path, source_keys: set[str], mapped_inner_keys: set[str]) -> dict:
    """Write the safetensors file first and publish its manifest only after success."""
    source_checkpoint = Path(source_checkpoint)
    config_path = Path(config_path)
    bundle_path = Path(bundle_path)
    manifest_path = Path(manifest_path)
    if bundle_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite an existing MLX bundle or manifest")
    if not source_checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {source_checkpoint}")
    if not config_path.is_file():
        raise ValueError(f"config does not exist: {config_path}")
    coverage = audit_source_coverage(source_keys, mapped_inner_keys)
    parameters = dict(flatten_tree(model.parameters()))
    if not parameters or not all(isinstance(value, mx.array) for value in parameters.values()):
        raise ValueError("model parameters are not a flat MLX array set")
    dtypes = {str(value.dtype).removeprefix("mlx.core.") for value in parameters.values()}
    if len(dtypes) != 1:
        raise ValueError(f"mixed model dtypes: {sorted(dtypes)}")
    dtype = dtypes.pop()
    if dtype not in {"float16", "float32"}:
        raise ValueError(f"unsupported MLX bundle dtype: {dtype}")
    metadata = build_metadata(sha256_file(source_checkpoint), dtype, len(parameters))
    metadata["config_sha256"] = sha256_file(config_path)
    manifest = {
        **metadata,
        "source_checkpoint": str(source_checkpoint),
        "config_path": str(config_path),
        "source_key_count": len(source_keys),
        "mapped_key_count": len(mapped_inner_keys),
        "parameter_names": sorted(parameters),
        "coverage": coverage,
    }
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_bundle = None
    temporary_manifest = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{bundle_path.name}.", suffix=".safetensors", dir=bundle_path.parent, delete=False
        ) as stream:
            temporary_bundle = Path(stream.name)
        mx.save_safetensors(str(temporary_bundle), parameters, metadata=metadata)
        os.replace(temporary_bundle, bundle_path)
        temporary_bundle = None
        manifest["bundle_sha256"] = sha256_file(bundle_path)
        with tempfile.NamedTemporaryFile(
            prefix=f".{manifest_path.name}.", suffix=".tmp", dir=manifest_path.parent,
            mode="w", encoding="utf-8", delete=False
        ) as stream:
            temporary_manifest = Path(stream.name)
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_manifest, manifest_path)
        temporary_manifest = None
    finally:
        for temporary in (temporary_bundle, temporary_manifest):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    if args.dtype != "float16":
        raise SystemExit("only native float16 conversion is currently supported")
    mx.set_default_device(mx.cpu)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model = HunyuanInnerUNet()
    mapped = load_weights_from_pytorch(model, str(args.checkpoint), return_used_keys=True)
    manifest = write_bundle(model, args.checkpoint, args.config, args.output, args.manifest,
                            set(source.keys()), mapped)
    print(json.dumps({key: manifest[key] for key in
                      ("schema", "source_sha256", "config_sha256", "parameter_count",
                       "source_key_count", "mapped_key_count", "coverage")}, indent=2))


if __name__ == "__main__":
    main()

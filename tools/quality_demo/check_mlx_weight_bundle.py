#!/usr/bin/env python3
"""Strictly load an MLX inner-UNet bundle in a Torch-forbidden process."""

import argparse
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import time


def verify_manifest(bundle: Path, manifest_path: Path) -> dict:
    """Verify bundle bytes, parameter names, and count against its manifest."""
    import mlx.core as mx

    manifest = json.loads(Path(manifest_path).read_text())
    sha = hashlib.sha256()
    with Path(bundle).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    digest = sha.hexdigest()
    if manifest.get("bundle_sha256") != digest:
        raise ValueError("bundle_sha256 mismatch")
    arrays, _ = mx.load(str(bundle), return_metadata=True)
    names = sorted(arrays)
    if names != sorted(manifest.get("parameter_names", [])):
        raise ValueError("parameter_names mismatch")
    if len(names) != int(manifest.get("parameter_count", "-1")):
        raise ValueError("parameter_count mismatch")
    return manifest


class _ForbidTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"Torch import forbidden by R1 smoke: {fullname}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        help="defaults to manifest.json beside the bundle")
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--dtype", default="float16", choices=("float16", "float32"))
    args = parser.parse_args()
    sys.meta_path.insert(0, _ForbidTorch())
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from hy3dpaint.mlx_unet import HunyuanInnerUNet
    from hy3dpaint.mlx_weight_bundle import load_model_weights

    manifest = verify_manifest(args.bundle, args.manifest or args.bundle.with_name("manifest.json"))
    started = time.perf_counter()
    model = load_model_weights(
        HunyuanInnerUNet(), args.bundle,
        expected_source_sha256=args.source_sha256,
        expected_config_sha256=args.config_sha256,
        expected_dtype=args.dtype,
    )
    torch_loaded = any(name == "torch" or name.startswith("torch.") for name in sys.modules)
    if torch_loaded:
        raise RuntimeError("Torch appeared in sys.modules")
    print(json.dumps({
        "status": "passed",
        "torch_imported": False,
        "parameter_count": sum(1 for _ in _flatten(model.parameters())),
        "bundle": str(args.bundle.resolve()),
        "bundle_sha256_verified": manifest["bundle_sha256"],
        "conv_in_shape": list(model.conv_in.weight.shape),
        "up0_upsample_shape": list(model.up_block_0.upsample.conv.weight.shape),
        "load_s": time.perf_counter() - started,
    }, indent=2))


def _flatten(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _flatten(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _flatten(child)
    else:
        yield value


if __name__ == "__main__":
    main()

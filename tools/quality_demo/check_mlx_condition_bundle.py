#!/usr/bin/env python3
"""Validate a paint condition bundle in a process that forbids Torch."""

import argparse
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys


class _ForbidTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"Torch import forbidden: {fullname}")
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    sys.meta_path.insert(0, _ForbidTorch())
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import mlx.core as mx
    from hy3dpaint.mlx_condition_bundle import load_bundle

    mx.set_default_device(mx.cpu)
    arrays, metadata = load_bundle(args.bundle, args.source_sha256, args.config_sha256)
    manifest_path = args.manifest or args.bundle.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    digest = hashlib.sha256()
    with args.bundle.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if manifest.get("bundle_sha256") != digest.hexdigest():
        raise ValueError("bundle_sha256 mismatch")
    if any(manifest.get(key) != value for key, value in metadata.items()):
        raise ValueError("manifest metadata mismatch")
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        raise RuntimeError("Torch appeared in sys.modules")
    print(json.dumps({
        "status": "passed",
        "torch_imported": False,
        "schema": metadata["schema"],
        "parameter_count": len(arrays),
        "bundle_sha256_verified": digest.hexdigest(),
        "keys": sorted(arrays),
        "dtype": metadata["dtype"],
    }, indent=2))


if __name__ == "__main__":
    main()

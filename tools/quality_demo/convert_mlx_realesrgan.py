#!/usr/bin/env python3
"""Offline Torch conversion of the RealESRGAN x4plus RRDB checkpoint."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hy3dpaint.mlx_realesrgan import MLXRRDBNet, load_weights_from_pytorch, save_bundle


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_checkpoint(checkpoint: Path, bundle: Path, manifest: Path, *,
                       num_feat=64, num_block=23, num_grow_ch=32, dtype="float16"):
    """Convert once; publish a manifest only after all weights are saved."""
    checkpoint, bundle, manifest = Path(checkpoint), Path(bundle), Path(manifest)
    if bundle.exists() or manifest.exists():
        raise FileExistsError("refusing to overwrite a RealESRGAN bundle or manifest")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if dtype not in {"float16", "float32"}:
        raise ValueError("unsupported bundle dtype")
    mx.set_default_device(mx.cpu)
    source_hash = sha256_file(checkpoint)
    model = MLXRRDBNet(num_feat=num_feat, num_block=num_block, num_grow_ch=num_grow_ch)
    load_weights_from_pytorch(model, checkpoint, dtype=getattr(mx, dtype))
    metadata = save_bundle(model, bundle, source_sha256=source_hash, dtype=dtype)
    report = {
        **metadata,
        "source_checkpoint": str(checkpoint.resolve()),
        "bundle": str(bundle.resolve()),
        "bundle_sha256": sha256_file(bundle),
        "num_feat": num_feat,
        "num_block": num_block,
        "num_grow_ch": num_grow_ch,
        "scale": 4,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{manifest.name}.", suffix=".tmp",
                                         dir=manifest.parent, mode="w", encoding="utf-8",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, manifest)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    report = convert_checkpoint(args.checkpoint, args.bundle, args.manifest, dtype=args.dtype)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

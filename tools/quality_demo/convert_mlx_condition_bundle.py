#!/usr/bin/env python3
"""Extract the seven paint outer-condition tensors from a trusted checkpoint."""

import argparse
import hashlib
from pathlib import Path
import sys

import mlx.core as mx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hy3dpaint.mlx_condition_bundle import CONDITION_SHAPES, save_bundle  # noqa: E402


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(checkpoint_path, config_path, bundle_path, manifest_path):
    if Path(bundle_path).exists() or Path(manifest_path).exists():
        raise FileExistsError("refusing to overwrite condition bundle or manifest")
    if not Path(checkpoint_path).is_file() or not Path(config_path).is_file():
        raise FileNotFoundError("checkpoint and config must exist")
    mx.set_default_device(mx.cpu)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    source_keys = {"unet." + name for name in CONDITION_SHAPES}
    actual_keys = {name for name in checkpoint
                   if name.startswith("unet.image_proj_model_dino.")
                   or name.startswith("unet.learned_text_clip_")}
    if actual_keys != source_keys:
        raise ValueError(f"condition source keys mismatch: missing={sorted(source_keys - actual_keys)}, "
                         f"extra={sorted(actual_keys - source_keys)}")
    arrays = {name: mx.array(checkpoint["unet." + name].detach().numpy().copy())
              for name in CONDITION_SHAPES}
    save_bundle(arrays, _sha256_file(Path(checkpoint_path)),
                _sha256_file(Path(config_path)), bundle_path, manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    convert(args.checkpoint, args.config, args.output, args.manifest)
    print(f"wrote {args.output} and {args.manifest}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare the checkpoint DINO projector on Torch/MPS and MLX/GPU."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hy3dpaint.mlx_dino_projector import MLXDinoProjector  # noqa: E402


KEY_PREFIX = "unet.image_proj_model_dino."
WEIGHT_NAMES = ("proj.weight", "proj.bias", "norm.weight", "norm.bias")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing report")
    if bool(args.config) != bool(args.bundle):
        parser.error("--config and --bundle must be supplied together")

    import torch
    import torch.nn.functional as functional
    import mlx.core as mx

    features = np.load(args.features, allow_pickle=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    weights = {name: checkpoint[KEY_PREFIX + name].detach().numpy().copy()
               for name in WEIGHT_NAMES}
    del checkpoint
    input_dim = weights["proj.weight"].shape[1]
    output_dim = weights["norm.weight"].shape[0]
    extra_tokens = weights["proj.weight"].shape[0] // output_dim
    if features.ndim != 3 or features.shape[-1] != input_dim:
        raise ValueError("DINO fixture does not match projector input")

    torch_input = torch.from_numpy(features.copy()).to("mps")
    torch_weights = {name: torch.from_numpy(value).to("mps") for name, value in weights.items()}
    with torch.inference_mode():
        projected = functional.linear(torch_input.reshape(-1, input_dim),
                                      torch_weights["proj.weight"], torch_weights["proj.bias"])
        projected = projected.reshape(-1, extra_tokens, output_dim)
        projected = functional.layer_norm(projected, (output_dim,),
                                          torch_weights["norm.weight"], torch_weights["norm.bias"])
        reference = projected.reshape(features.shape[0], -1, output_dim).cpu().numpy()
    del projected, torch_weights, torch_input
    torch.mps.empty_cache()

    mx.set_default_device(mx.gpu)
    candidate = MLXDinoProjector(input_dim, output_dim, extra_tokens)
    if args.bundle:
        from hy3dpaint.mlx_condition_bundle import load_bundle
        source_sha = sha256_file(args.checkpoint)
        config_sha = sha256_file(args.config)
        bundle_arrays, _ = load_bundle(args.bundle, source_sha, config_sha)
        candidate.load_weights([(name, bundle_arrays["image_proj_model_dino." + name])
                                for name in WEIGHT_NAMES], strict=True)
    else:
        candidate.load_weights([(name, mx.array(value)) for name, value in weights.items()], strict=True)
    actual_mx = candidate(mx.array(features))
    mx.eval(actual_mx)
    actual = np.array(actual_mx)
    difference = actual.astype(np.float32) - reference.astype(np.float32)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "features": str(args.features.resolve()),
        "bundle": str(args.bundle.resolve()) if args.bundle else None,
        "features_sha256": sha256_file(args.features),
        "input_shape": list(features.shape),
        "output_shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "torch_device": "mps",
        "mlx_device": "gpu",
        "mean_abs": float(np.abs(difference).mean()),
        "max_abs": float(np.abs(difference).max()),
        "nrmse": float(np.linalg.norm(difference.ravel().astype(np.float64)) /
                       np.linalg.norm(reference.ravel().astype(np.float64))),
        "finite": bool(np.isfinite(actual).all()),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

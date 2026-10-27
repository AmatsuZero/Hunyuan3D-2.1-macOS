#!/usr/bin/env python3
"""Replay the captured first paint generation pass with the MLX inner UNet."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hy3dpaint.mlx_unet import HunyuanInnerUNet  # noqa: E402
from hy3dpaint.mlx_weight_bundle import load_model_weights  # noqa: E402


def replay(capture, bundle, source_sha256, config_sha256, device):
    capture = Path(capture)
    record = (capture / "capture.json").read_bytes()
    metadata = json.loads(record)
    contract = metadata["cache_contract"]
    if len(contract["condition_layers"]) != 16 or len(contract["position_entries"]) != 4:
        raise ValueError("incomplete inner first-step cache capture")
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    model = load_model_weights(HunyuanInnerUNet(), bundle,
                               expected_source_sha256=source_sha256,
                               expected_config_sha256=config_sha256,
                               expected_dtype="float16")

    def load(name):
        return mx.array(np.load(capture / f"{name}.npy", allow_pickle=False))

    cache = {name: load("condition_embed_" + name.replace(".", "_"))
             for name in contract["condition_layers"]}
    positions = {
        int(sequence): {
            "voxel_indices": load("position_voxel_indices_" + sequence),
            "voxel_resolution": contract["position_resolutions"][sequence],
        }
        for sequence in contract["position_entries"]
    }
    sample = load("inner_sample")
    timestep = load("inner_timestep").reshape(1).astype(mx.float16)
    encoder = load("generation_encoder_hidden_states")
    dino = load("inner_dino_hidden_states")
    ref_scale = load("inner_ref_scale")
    mva_scale = float(np.load(capture / "inner_mva_scale.npy", allow_pickle=False))
    num_in_batch = int(np.load(capture / "inner_num_in_batch.npy", allow_pickle=False))
    expected = np.load(capture / "inner_output.npy", allow_pickle=False).astype(np.float32)
    started = time.perf_counter()
    output = model(sample, timestep, encoder, num_in_batch=num_in_batch,
                   condition_embed_dict=cache, dino_hidden_states=dino,
                   position_voxel_indices=positions, ref_scale=ref_scale,
                   mva_scale=mva_scale)
    mx.eval(output)
    elapsed = time.perf_counter() - started
    actual = np.array(output).astype(np.float32)
    if actual.shape != expected.shape:
        raise ValueError(f"inner output shape mismatch: {actual.shape} != {expected.shape}")
    difference = actual - expected
    return {
        "capture": str(capture.resolve()),
        "capture_sha256": hashlib.sha256(record).hexdigest(),
        "bundle": str(Path(bundle).resolve()),
        "device": device,
        "shape": list(actual.shape),
        "finite": bool(np.isfinite(actual).all()),
        "mean_abs": float(np.abs(difference).mean()),
        "p99_abs": float(np.percentile(np.abs(difference), 99)),
        "max_abs": float(np.abs(difference).max()),
        "nrmse": float(np.linalg.norm(difference.ravel().astype(np.float64)) /
                       np.linalg.norm(expected.ravel().astype(np.float64))),
        "elapsed_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing report")
    result = replay(args.capture, args.bundle, args.source_sha256, args.config_sha256, args.device)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

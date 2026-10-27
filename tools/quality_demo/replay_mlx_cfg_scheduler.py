#!/usr/bin/env python3
"""Replay real paint CFG and UniPC updates without Torch."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hy3dpaint.mlx_cfg import combine_classifier_free  # noqa: E402
from hy3dpaint.mlx_unipc_scheduler import MLXUniPCScheduler  # noqa: E402


def metrics(actual, expected):
    difference = actual.astype(np.float32) - expected.astype(np.float32)
    return {
        "mean_abs": float(np.abs(difference).mean()),
        "max_abs": float(np.abs(difference).max()),
        "nrmse": float(np.linalg.norm(difference.ravel().astype(np.float64)) /
                       np.linalg.norm(expected.ravel().astype(np.float64))),
        "finite": bool(np.isfinite(actual).all()),
    }


def replay(capture, device):
    capture = Path(capture)
    report = json.loads((capture / "capture.json").read_text())
    schedule = report["scheduler"]
    mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
    scheduler = MLXUniPCScheduler.from_config(schedule["config"])
    scheduler.set_timesteps(len(schedule["timesteps"]))
    if scheduler.timesteps.tolist() != schedule["timesteps"]:
        raise ValueError("captured timesteps differ from MLX schedule")
    azims = report["cameras"]["azims"]
    guidance_scale = float(report["guidance_scale"])
    pbr_views = 2
    results = []
    for index, timestep in enumerate(schedule["timesteps"]):
        pre_cfg = np.load(capture / f"noise_pred_{index}.npy", allow_pickle=False)
        expected_cfg = np.load(capture / f"scheduler_model_output_{index}.npy", allow_pickle=False)
        sample = np.load(capture / f"scheduler_sample_{index}.npy", allow_pickle=False)
        expected_latent = np.load(capture / f"latents_{index}.npy", allow_pickle=False)
        cfg = combine_classifier_free(mx.array(pre_cfg), guidance_scale, azims, pbr_views)
        latent = scheduler.step(cfg, timestep, mx.array(sample))
        mx.eval(cfg, latent)
        results.append({
            "step": index,
            "cfg": metrics(np.array(cfg), expected_cfg),
            "scheduler": metrics(np.array(latent), expected_latent),
        })
    return {
        "capture": str(capture.resolve()),
        "capture_sha256": hashlib.sha256((capture / "capture.json").read_bytes()).hexdigest(),
        "device": device,
        "steps": len(results),
        "guidance_scale": guidance_scale,
        "results": results,
        "max_cfg_nrmse": max(item["cfg"]["nrmse"] for item in results),
        "max_scheduler_nrmse": max(item["scheduler"]["nrmse"] for item in results),
        "final_scheduler_nrmse": results[-1]["scheduler"]["nrmse"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing report")
    result = replay(args.capture, args.device)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()

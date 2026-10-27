#!/usr/bin/env python3
"""Replay a fixed-condition UniPC capture through the Torch-free MLX scheduler."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hy3dpaint.mlx_unipc_scheduler import MLXUniPCScheduler  # noqa: E402


def _metrics(actual, expected):
    difference = actual.astype(np.float32) - expected.astype(np.float32)
    return {
        "mean_abs": float(np.abs(difference).mean()),
        "max_abs": float(np.abs(difference).max()),
        "nrmse": float(np.linalg.norm(difference.ravel().astype(np.float64)) /
                       np.linalg.norm(expected.ravel().astype(np.float64))),
        "finite": bool(np.isfinite(actual).all()),
    }


def replay(capture: Path, device: str):
    report = json.loads((capture / "capture.json").read_text())
    schedule = report["scheduler"]
    mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
    teacher = MLXUniPCScheduler.from_config(schedule["config"])
    chained = MLXUniPCScheduler.from_config(schedule["config"])
    teacher.set_timesteps(len(schedule["timesteps"]))
    chained.set_timesteps(len(schedule["timesteps"]))
    if teacher.timesteps.tolist() != schedule["timesteps"]:
        raise ValueError("captured timesteps differ from MLX schedule")
    if not np.allclose(teacher.sigmas, schedule["sigmas"], rtol=1e-6, atol=1e-7):
        raise ValueError("captured sigmas differ from MLX schedule")

    results = []
    chained_sample = None
    for index, timestep in enumerate(schedule["timesteps"]):
        prediction = np.load(capture / f"scheduler_model_output_{index}.npy", allow_pickle=False)
        sample = np.load(capture / f"scheduler_sample_{index}.npy", allow_pickle=False)
        expected = np.load(capture / f"latents_{index}.npy", allow_pickle=False)
        captured_timestep = int(np.load(capture / f"scheduler_timestep_{index}.npy", allow_pickle=False))
        if captured_timestep != timestep:
            raise ValueError(f"timestep mismatch at step {index}")
        if prediction.shape != sample.shape or expected.shape != sample.shape:
            raise ValueError(f"tensor shape mismatch at step {index}")
        if chained_sample is None:
            chained_sample = mx.array(sample)
        teacher_result = teacher.step(mx.array(prediction), timestep, mx.array(sample))
        chained_sample = chained.step(mx.array(prediction), timestep, chained_sample)
        mx.eval(teacher_result, chained_sample)
        results.append({
            "step": index,
            "teacher_forced": _metrics(np.array(teacher_result), expected),
            "chained": _metrics(np.array(chained_sample), expected),
        })
    return {
        "capture": str(capture.resolve()),
        "capture_sha256": hashlib.sha256((capture / "capture.json").read_bytes()).hexdigest(),
        "device": device,
        "steps": len(results),
        "dtype": str(prediction.dtype),
        "results": results,
        "max_teacher_nrmse": max(item["teacher_forced"]["nrmse"] for item in results),
        "final_chained_nrmse": results[-1]["chained"]["nrmse"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing report")
    result = replay(args.capture, args.device)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Offline localization of occluded512 CFG/scheduler/latent drift.

This consumes archived resident traces only; it never imports Torch/MLX or
starts inference. The image-to-latent windows are the coarse /8 mappings from
the view-mapping handoff, while image 10 is retained as a counterexample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REGIONS = {
    5: (slice(38, 42), slice(25, 27)),
    11: (slice(41, 43), slice(28, 30)),
    10: (slice(42, 45), slice(34, 37)),
}


def load(root: Path, kind: str, step: int) -> np.ndarray:
    path = root / "resident-trace" / f"resident_{kind}_{step}.npy"
    return np.load(path, allow_pickle=False).astype(np.float32)


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    difference = actual - expected
    return {
        "mean_abs": float(np.abs(difference).mean()),
        "max_abs": float(np.abs(difference).max()),
        "rms": float(np.sqrt(np.mean(difference * difference))),
        "nrmse": float(np.linalg.norm(difference.ravel()) / np.linalg.norm(expected.ravel())),
    }


def analyze(reference: Path, candidate: Path) -> dict:
    reference_capture = json.loads((reference / "capture.json").read_text())
    candidate_capture = json.loads((candidate / "capture.json").read_text())
    for key in ("cameras", "guidance_scale"):
        if reference_capture[key] != candidate_capture[key]:
            raise ValueError(f"capture {key} differs between reference and candidate")
    for key in ("class", "timesteps", "sigmas"):
        if reference_capture["scheduler"][key] != candidate_capture["scheduler"][key]:
            raise ValueError(f"scheduler {key} differs between reference and candidate")
    for capture in (reference_capture, candidate_capture):
        capture["scheduler"]["config"]["_use_default_values"].sort()
    if reference_capture["scheduler"]["config"] != candidate_capture["scheduler"]["config"]:
        raise ValueError("scheduler config differs between reference and candidate")
    if not np.array_equal(
        np.load(reference / "initial-latents.npy", allow_pickle=False),
        np.load(candidate / "initial-latents.npy", allow_pickle=False),
    ):
        raise ValueError("initial latents differ between reference and candidate")

    result = {
        "reference": str(reference.resolve()),
        "candidate": str(candidate.resolve()),
        "input_sha256": {},
        "regions": {
            str(image): {"y": [ys.start, ys.stop - 1], "x": [xs.start, xs.stop - 1]}
            for image, (ys, xs) in REGIONS.items()
        },
        "rows": [],
    }
    for root, label in ((reference, "reference"), (candidate, "candidate")):
        for name in ("initial-latents.npy", "capture.json"):
            result["input_sha256"][f"{label}/{name}"] = hashlib.sha256(
                (root / name).read_bytes()
            ).hexdigest()

    for step in range(15):
        candidate_noise, reference_noise = (
            load(root, "noise_pred", step) for root in (candidate, reference)
        )
        candidate_cfg, reference_cfg = (
            load(root, "scheduler_model_output", step) for root in (candidate, reference)
        )
        candidate_latents, reference_latents = (
            load(root, "latents", step) for root in (candidate, reference)
        )
        for image, (ys, xs) in REGIONS.items():
            branch = [
                metrics(
                    candidate_noise[branch_index * 12 + image, :, ys, xs],
                    reference_noise[branch_index * 12 + image, :, ys, xs],
                )
                for branch_index in range(3)
            ]
            cfg = metrics(candidate_cfg[image, :, ys, xs], reference_cfg[image, :, ys, xs])
            latent = metrics(
                candidate_latents[image, :, ys, xs], reference_latents[image, :, ys, xs]
            )
            # The captured three-branch formula simplifies to
            # (1-k)*uncond + k*full; reference cancels except rounding.
            k = 6 if image in (5, 11) else 3
            predicted = k * (
                candidate_noise[24 + image, :, ys, xs]
                - reference_noise[24 + image, :, ys, xs]
            ) + (1 - k) * (
                candidate_noise[image, :, ys, xs]
                - reference_noise[image, :, ys, xs]
            )
            actual = candidate_cfg[image, :, ys, xs] - reference_cfg[image, :, ys, xs]
            residual = actual - predicted
            result["rows"].append(
                {
                    "step": step,
                    "image": image,
                    "noise_pred": branch,
                    "cfg": cfg,
                    "latent": latent,
                    "cfg_linear_prediction_rms": float(np.sqrt(np.mean(predicted * predicted))),
                    "cfg_residual_rms": float(np.sqrt(np.mean(residual * residual))),
                }
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    result = analyze(args.reference, args.candidate)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()

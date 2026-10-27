#!/usr/bin/env python3
"""Measure serial Torch/MPS RealESRGAN enhance on one image."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "hy3dpaint"))
from utils.torchvision_fix import apply_fix  # noqa: E402
apply_fix()
from utils.image_super_utils import imageSuperNet  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or args.runs < 2:
        parser.error("output must not exist and runs must be at least 2")
    started = time.perf_counter()
    model = imageSuperNet(SimpleNamespace(realesrgan_ckpt_path=str(args.checkpoint)))
    load_s = time.perf_counter() - started
    image = Image.open(args.input).convert("RGB")
    samples = []
    for _ in range(args.runs):
        started = time.perf_counter()
        result = np.asarray(model(image))
        samples.append({
            "elapsed_s": time.perf_counter() - started,
            "sha256": hashlib.sha256(result.tobytes()).hexdigest(),
        })
    report = {"load_s": load_s, "samples": samples, "shape": list(result.shape)}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure serial RealESRGAN MLX forward plus RGB conversion on one image."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import mlx.core as mx
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hy3dpaint.mlx_realesrgan import MLXRRDBNet, enhance_rgb, load_bundle  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or args.runs < 2:
        parser.error("output must not exist and runs must be at least 2")
    manifest = json.loads(args.manifest.read_text())
    mx.set_default_device(mx.gpu)
    started = time.perf_counter()
    model = load_bundle(MLXRRDBNet(), args.bundle,
                        expected_source_sha256=manifest["source_sha256"],
                        expected_dtype=manifest["dtype"])
    load_s = time.perf_counter() - started
    image = np.asarray(Image.open(args.input).convert("RGB"))
    samples = []
    for _ in range(args.runs):
        started = time.perf_counter()
        result = enhance_rgb(model, image)
        samples.append({
            "elapsed_s": time.perf_counter() - started,
            "sha256": hashlib.sha256(result.tobytes()).hexdigest(),
        })
    report = {
        "load_s": load_s,
        "samples": samples,
        "shape": list(result.shape),
        "mlx_peak_gib": mx.get_peak_memory() / 1024**3,
        "torch_imported": any(name == "torch" or name.startswith("torch.") for name in sys.modules),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

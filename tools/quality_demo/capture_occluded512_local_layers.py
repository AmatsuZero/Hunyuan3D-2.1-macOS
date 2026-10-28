"""Capture selected MLX inner-UNet patches in the frozen occluded512 run.

Run one backend per process, with a fresh output directory. This diagnostic
wraps the existing fixed-condition runner and never changes default dispatch.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


STEPS = (3, 6, 10, 14)
POINTS = (
    "conv_in", "down_0", "down_1", "down_2", "down_3", "mid",
    "up_1.2.attn", "up_2.2.attn", "up_3.2.attn",
)
REGIONS = {5: (38, 42, 25, 27), 11: (41, 43, 28, 30),
           10: (42, 45, 34, 37)}
EVIDENCE = Path("/Users/daubert/Github/hy3d-upgrade-evidence")
FIXTURES = EVIDENCE / "20261028/r3-reference-fp32-resident-01/frozen-inputs/occluded512/fixtures"
NOISE = EVIDENCE / "20261027/color-diagnosis/initial-noise-torch251-seed0.npy"
BUNDLE = EVIDENCE / "20261028/r3-condition-bundle-01/conditions.safetensors"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch(value, image, height, width):
    y0, y1, x0, x1 = REGIONS[image]
    sy0 = max(0, y0 * height // 64 - 1)
    sy1 = min(height, (y1 * height + 63) // 64 + 1)
    sx0 = max(0, x0 * width // 64 - 1)
    sx1 = min(width, (x1 * width + 63) // 64 + 1)
    # MLX internal feature maps use NHWC; retain all three CFG branches.
    return np.stack([
        np.asarray(value[branch * 12 + image, sy0:sy1, sx0:sx1, :])
        for branch in range(3)
    ])


def capture(args):
    from probe_fixed_conditions import add_checkout_paths, main as run_fixed

    root = Path(__file__).resolve().parents[2]
    add_checkout_paths(root)
    from mlx_utils import MLXDispatchUNet

    expected = {
        "HUNYUAN_MLX_FIRST_STEP": "1", "HUNYUAN_MLX_RESIDENT_LOOP": "1",
        "HUNYUAN_MLX_POSITION_INDICES": "1", "HUNYUAN_MLX_REFERENCE_CACHE": "1",
        "HUNYUAN_MLX_FP32": "0",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
    }
    expected["HUNYUAN_MLX_REFERENCE_FORWARD"] = "1" if args.backend == "mlx" else "0"
    expected["HUNYUAN_MLX_REFERENCE_FP32"] = "1" if args.backend == "mlx" else "0"
    expected["HUNYUAN_MLX_DISABLE_WINOGRAD"] = "1" if args.backend == "mlx" else "0"
    for key, value in expected.items():
        if os.environ.get(key) != value:
            raise ValueError(f"{key} must be {value}")
    if os.environ.get("HUNYUAN_MLX_DINO_PROJECTOR_BUNDLE") != str(BUNDLE):
        raise ValueError("DINO bundle differs from frozen comparison")
    if os.environ.get("HUNYUAN_MLX_RESIDENT_TRACE_DIR"):
        raise ValueError("disable full resident trace for local layer capture")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.output_dir.parent.is_dir():
        raise FileNotFoundError(args.output_dir.parent)

    original = MLXDispatchUNet.forward_mlx
    counter = {}

    def forward_mlx(self, sample, timestep, debug_trace=None):
        index = counter.get(id(self), 0) + 1
        counter[id(self)] = index
        if index not in STEPS:
            return original(self, sample, timestep, debug_trace=debug_trace)
        model = self._mlx_model
        local = args.output_dir / "local-layers"
        local.mkdir(parents=True, exist_ok=True)

        def trace(name, value):
            if name not in POINTS:
                return
            if value.ndim != 4 or value.shape[0] != 36:
                raise ValueError(f"unexpected {name} shape {value.shape}")
            for image in REGIONS:
                selected = patch(value, image, value.shape[1], value.shape[2])
                path = local / f"step{index}_{name}_image{image}.npy"
                np.save(path, selected, allow_pickle=False)

        for block_index in range(4):
            block = getattr(model, f"up_block_{block_index}")
            block._reference_debug_trace = trace
            if block_index:
                block._reference_trace_index = block_index

        try:
            return original(self, sample, timestep, debug_trace=trace)
        finally:
            for block_index in range(4):
                getattr(model, f"up_block_{block_index}")._reference_debug_trace = None

    MLXDispatchUNet.forward_mlx = forward_mlx
    prior_argv = sys.argv
    sys.argv = [str(root / "tools/quality_demo/probe_fixed_conditions.py"),
                "--stage", "capture", "--fixtures", str(FIXTURES),
                "--output-dir", str(args.output_dir), "--steps", "15",
                "--stop-after", "0", "--capture-cache", "--initial-noise", str(NOISE),
                "--guidance-scale", "3", "--reference-scale", "1", "--mva-scale", "1",
                "--mlx", "--mlx-cache-limit-mib", "1024"]
    try:
        run_fixed()
    finally:
        sys.argv = prior_argv
        MLXDispatchUNet.forward_mlx = original
    finalize(args)


def finalize(args):
    if (args.output_dir / "local-layers.json").exists():
        raise FileExistsError(args.output_dir / "local-layers.json")
    local = args.output_dir / "local-layers"
    expected_files = {
        f"step{step}_{name}_image{image}.npy"
        for step in STEPS for name in POINTS for image in REGIONS
    }
    actual_files = {path.name for path in local.glob("*.npy")}
    if actual_files != expected_files:
        raise RuntimeError(f"local trace missing={expected_files - actual_files}, "
                           f"unexpected={actual_files - expected_files}")
    for path in local.glob("*.npy"):
        value = np.load(path, allow_pickle=False)
        if value.shape[0] != 3 or not np.isfinite(value).all():
            raise ValueError(f"invalid local patch: {path}")
    if json.loads((args.output_dir / "capture.json").read_text())["generated_images"] != 12:
        raise ValueError("incomplete image capture")
    report = {"backend": args.backend, "steps": STEPS, "points": POINTS,
              "regions": REGIONS, "capture_sha256": digest(args.output_dir / "capture.json"),
              "records": {path.name: digest(path) for path in sorted(
                  (args.output_dir / "local-layers").glob("*.npy"))}}
    (args.output_dir / "local-layers.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"local_layers": str(args.output_dir / "local-layers.json"),
                      "sha256": digest(args.output_dir / "local-layers.json")}))


def compare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    from analyze_occluded512_view5 import metrics

    roots = (args.reference, args.candidate)
    meta = [json.loads((root / "local-layers.json").read_text()) for root in roots]
    if (meta[0]["backend"], meta[1]["backend"]) != ("torch", "mlx"):
        raise ValueError("expected Torch-reference then MLX-reference captures")
    for key in ("steps", "points", "regions"):
        if meta[0][key] != meta[1][key]:
            raise ValueError(f"{key} differs")
    archive = (EVIDENCE / "20261028/r3-reference-fp32-resident-01/occluded512/torch-reference",
               EVIDENCE / "20261029/r3-reference-winograd-optin-20261029/occluded512-replay")
    image_exact = {}
    for index, (root, old) in enumerate(zip(roots, archive)):
        image_exact[meta[index]["backend"]] = all(
            digest(root / f"generated_{i}.png") == digest(old / f"generated_{i}.png")
            for i in range(12)
        )
    if not all(image_exact.values()):
        raise ValueError(f"instrumented outputs differ from archive: {image_exact}")
    boundary = {}
    exact_inputs = ("initial-latents.npy", "inner_sample.npy",
                    "inner_dino_hidden_states.npy", "generation_encoder_hidden_states.npy",
                    "reference_sample.npy", "reference_encoder_hidden_states.npy")
    for name in exact_inputs:
        left = np.load(roots[0] / name, allow_pickle=False)
        right = np.load(roots[1] / name, allow_pickle=False)
        if not np.array_equal(left, right):
            raise ValueError(f"boundary input differs: {name}")
    for name in ("reference_write.npy", "noise_pred_0.npy"):
        left = np.load(roots[0] / name, allow_pickle=False)
        right = np.load(roots[1] / name, allow_pickle=False)
        boundary[name] = metrics(right.astype(np.float32), left.astype(np.float32))
    cache_rows = []
    for path in sorted(roots[0].glob("condition_embed_*.npy")):
        right_path = roots[1] / path.name
        if not right_path.is_file():
            raise ValueError(f"missing candidate cache: {path.name}")
        left = np.load(path, allow_pickle=False)
        right = np.load(right_path, allow_pickle=False)
        cache_rows.append({"name": path.name,
                           **metrics(right.astype(np.float32), left.astype(np.float32))})
    rows = []
    for step in STEPS:
        for point in POINTS:
            for image in REGIONS:
                name = f"step{step}_{point}_image{image}.npy"
                left, right = (np.load(root / "local-layers" / name, allow_pickle=False)
                               for root in roots)
                if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
                    raise ValueError(f"shape or finite check failed: {name}")
                rows.append({"step": step, "point": point, "image": image,
                             "metrics_by_branch": [metrics(right[b].astype(np.float32),
                                                            left[b].astype(np.float32))
                                                   for b in range(3)]})
    report = {"reference": str(args.reference), "candidate": str(args.candidate),
              "image_exact": image_exact, "boundary_exact_inputs": list(exact_inputs),
              "reference_boundary": boundary, "condition_cache": cache_rows,
              "local_capture_sha256": [digest(
                  root / "local-layers.json") for root in roots], "rows": rows}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture_parser = sub.add_parser("capture")
    capture_parser.add_argument("--backend", choices=("torch", "mlx"), required=True)
    capture_parser.add_argument("--output-dir", type=Path, required=True)
    finalize_parser = sub.add_parser("finalize")
    finalize_parser.add_argument("--backend", choices=("torch", "mlx"), required=True)
    finalize_parser.add_argument("--output-dir", type=Path, required=True)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--reference", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "capture":
        capture(args)
    elif args.command == "finalize":
        finalize(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()

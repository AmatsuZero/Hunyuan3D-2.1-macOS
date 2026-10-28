"""Replay captured dual reference inputs through the opt-in MLX UNet."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "hy3dpaint"))

from mlx_unet import HunyuanInnerUNet  # noqa: E402
from mlx_utils import load_reference_weights_from_pytorch  # noqa: E402


LAYERS = (
    "down_0_0_0", "down_0_1_0", "down_1_0_0", "down_1_1_0",
    "down_2_0_0", "down_2_1_0", "mid_0_0", "up_1_0_0",
    "up_1_1_0", "up_1_2_0", "up_2_0_0", "up_2_1_0",
    "up_2_2_0", "up_3_0_0", "up_3_1_0", "up_3_2_0",
)


def metrics(expected, actual):
    if expected.shape != actual.shape:
        raise ValueError(f"shape differs: {expected.shape} vs {actual.shape}")
    left = expected.astype(np.float64)
    right = actual.astype(np.float64)
    difference = np.abs(left - right)
    return {
        "shape": list(expected.shape),
        "nrmse": float(np.linalg.norm(difference) / (np.linalg.norm(left) + 1e-12)),
        "mean_abs": float(difference.mean()),
        "max_abs": float(difference.max()),
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--torch-reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")

    for name in ("reference_sample", "reference_timestep",
                 "reference_encoder_hidden_states"):
        reference_path = args.torch_reference / f"{name}.npy"
        if not reference_path.exists():
            parser.error(f"Torch reference input missing: {reference_path}")
        if not np.array_equal(
            np.load(args.capture / f"{name}.npy"), np.load(reference_path)
        ):
            parser.error(f"reference input differs: {name}")

    sample = mx.array(np.load(args.capture / "reference_sample.npy"))
    timestep = mx.array([int(np.load(args.capture / "reference_timestep.npy"))],
                        dtype=sample.dtype)
    encoder = mx.array(np.load(args.capture / "reference_encoder_hidden_states.npy"))
    model = HunyuanInnerUNet(input_channels=4)
    mapped = load_reference_weights_from_pytorch(model, str(args.checkpoint))
    mx.eval(model.parameters())
    cache = {}
    output = model(
        sample, timestep, encoder, num_in_batch=1,
        condition_embed_dict={}, dino_hidden_states=None,
        mode="w", reference_cache=cache,
    )
    mx.eval([output, *cache.values()])
    if set(cache) != set(LAYERS):
        raise ValueError(f"unexpected reference layers: {sorted(cache)}")
    report = {
        "capture": str(args.capture.resolve()),
        "torch_reference": str(args.torch_reference.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "inputs_exact": True,
        "mapped_weights": mapped,
        "layers": {},
    }
    for name in LAYERS:
        expected = np.load(args.torch_reference / f"condition_embed_{name}.npy")
        report["layers"][name] = metrics(expected, np.array(cache[name]))
    report["reference_write"] = metrics(
        np.load(args.torch_reference / "reference_write.npy"), np.array(output)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""CPU float32 oracle for captured reference time-embedding boundaries.

No GPU imports or model forward. Replays each backend's own frozen input to
separate upstream drift from local linear/SiLU arithmetic. A fresh output
directory and matching input/checkpoint hashes are required.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.embeddings import get_timestep_embedding


LINEARS = {
    "time_embedding.linear_1": "time_embedding.linear_1",
    "time_embedding.linear_2": "time_embedding.linear_2",
    "down_0_resnet_0.time_emb_proj": "down_blocks.0.resnets.0.time_emb_proj",
}
ACTIVATIONS = {
    "time_embedding.silu": ("time_embedding.linear_1", "time_embedding.linear_2.input"),
    "down_0_resnet_0.silu": ("down_0_resnet_0.temb", "down_0_resnet_0.time_emb_proj.input"),
}


def metrics(reference, actual):
    left, right = np.asarray(reference), np.asarray(actual)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("non-finite oracle input/output")
    delta = right.astype(np.float64) - left.astype(np.float64)
    return {
        "shape": list(left.shape), "exact": bool(np.array_equal(left, right)),
        "nrmse": float(np.linalg.norm(delta) / (np.linalg.norm(left.astype(np.float64)) + 1e-12)),
        "max_abs": float(np.abs(delta).max()), "mean_abs": float(np.abs(delta).mean()),
    }


def run(args):
    roots = {"torch": args.torch_trace, "mlx": args.mlx_trace}
    metadata = {name: json.loads((root / "trace.json").read_text()) for name, root in roots.items()}
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if metadata["torch"]["input_sha256"] != metadata["mlx"]["input_sha256"]:
        raise ValueError("trace inputs differ")
    if any(meta["checkpoint_sha256"] != checkpoint_hash for meta in metadata.values()):
        raise ValueError("trace checkpoint differs from oracle")
    if metadata["torch"].get("torch_precision") != "fp32" or metadata["mlx"].get("precision") != "fp32":
        raise ValueError("oracle requires two float32 traces")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    def load(root, name):
        array = np.load(root / f"{name}.npy", allow_pickle=False)
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"{name}: expected finite float32")
        return array

    report = {"checkpoint_sha256": checkpoint_hash,
              "input_sha256": metadata["torch"]["input_sha256"],
              "torch_version": torch.__version__, "device": "cpu", "backends": {},
              "cross_backend": {}}
    names = [f"{name}{suffix}" for name in LINEARS for suffix in (".input", "")]
    names.append("down_0_resnet_0.temb")
    for name in names:
        report["cross_backend"][name] = metrics(load(roots["torch"], name), load(roots["mlx"], name))
    with torch.no_grad():
        for backend, root in roots.items():
            rows = {}
            for name, key in LINEARS.items():
                x = torch.from_numpy(load(root, f"{name}.input"))
                weight = state[f"unet_dual.{key}.weight"].float()
                bias = state[f"unet_dual.{key}.bias"].float()
                expected = F.linear(x, weight, bias).numpy()
                np.save(args.output / f"{backend}.{name}.cpu.npy", expected, allow_pickle=False)
                rows[name] = metrics(expected, load(root, name))
                rows[name]["weight_shape_out_in"] = list(weight.shape)
            for name, (source, target) in ACTIVATIONS.items():
                expected = F.silu(torch.from_numpy(load(root, source))).numpy()
                rows[name] = metrics(expected, load(root, target))
            rows["temb_identity"] = metrics(load(root, "time_embedding.linear_2"), load(root, "down_0_resnet_0.temb"))
            capture = Path(metadata[backend]["capture"])
            timestep_path = capture / "reference_timestep.npy"
            if hashlib.sha256(timestep_path.read_bytes()).hexdigest() != metadata[backend]["input_sha256"]["reference_timestep"]:
                raise ValueError("frozen timestep hash differs")
            timestep = int(np.load(timestep_path, allow_pickle=False))
            x = load(root, "time_embedding.linear_1.input")
            projected = get_timestep_embedding(torch.full((x.shape[0],), timestep), x.shape[1],
                                              flip_sin_to_cos=True, downscale_freq_shift=0).numpy()
            rows["time_projection"] = metrics(projected, x)
            report["backends"][backend] = rows
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("torch-trace", "mlx-trace", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = run(args)
    (args.output / "oracle.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

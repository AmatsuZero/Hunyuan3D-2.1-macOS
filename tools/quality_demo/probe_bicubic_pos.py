"""Compare MPS bicubic interpolation for DINOv2 position embeddings.

DINOv2 interpolates its 37x37 patch position grid down to 16x16 with
F.interpolate(mode="bicubic"). Torch 2.5.1 has no MPS bicubic kernel and
silently falls back to CPU, while 2.14.1 has one. This probe measures the
resulting difference against a float64 CPU reference.
Diagnostic only.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def _early_fallback(argv):
    for index, value in enumerate(argv):
        if value == "--fallback":
            return argv[index + 1]
        if value.startswith("--fallback="):
            return value.split("=", 1)[1]
    return None


_FALLBACK = _early_fallback(sys.argv)
if _FALLBACK is not None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = _FALLBACK

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def metrics(name, value, reference, results):
    array = value.detach().float().cpu().numpy().astype(np.float64)
    base = reference.detach().float().cpu().numpy().astype(np.float64)
    norm = float(np.linalg.norm(base))
    entry = {"name": name, "shape": list(array.shape),
             "absmax": float(np.abs(array).max()),
             "nonfinite": int((~np.isfinite(array)).sum())}
    if norm:
        entry["nrmse_vs_cpu"] = float(np.linalg.norm(array - base) / norm)
    results.append(entry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dim", type=int, default=1536)
    parser.add_argument("--source", type=int, default=37)
    parser.add_argument("--target", type=int, default=16)
    parser.add_argument("--fallback", choices=("0", "1"), default="0")
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(0)
    grid = torch.randn(1, args.source, args.source, args.dim, generator=generator, dtype=torch.float32)
    source = grid.permute(0, 3, 1, 2)

    with torch.no_grad():
        reference = F.interpolate(source.to(torch.float32), size=(args.target, args.target),
                                  mode="bicubic", align_corners=False).to(torch.float16)
        mps_result = F.interpolate(source.to("mps").to(torch.float16), size=(args.target, args.target),
                                   mode="bicubic", align_corners=False)
        torch.mps.synchronize()
        cpu_half = F.interpolate(source.to(torch.float16), size=(args.target, args.target),
                                 mode="bicubic", align_corners=False)

    results = []
    metrics("mps_bicubic_vs_cpu_fp32", mps_result, reference, results)
    metrics("cpu_half_bicubic_vs_cpu_fp32", cpu_half, reference, results)

    payload = {"torch": torch.__version__, "fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
               "dim": args.dim, "source": args.source, "target": args.target, "results": results}
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

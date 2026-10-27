"""Isolate which DINOv2 operator differs on MPS between Torch releases.

Runs the real operator set of one DINOv2 encoder layer on MPS and on CPU in
the same process, and reports the difference against a CPU reference.
Diagnostic only.
"""
import argparse
import copy
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


def compare(name, mps_value, cpu_value, results):
    mps64 = mps_value.detach().float().cpu().numpy().astype(np.float64)
    cpu64 = cpu_value.detach().float().cpu().numpy().astype(np.float64)
    mps_norm = float(np.linalg.norm(mps64))
    cpu_norm = float(np.linalg.norm(cpu64))
    entry = {"name": name, "shape": list(mps64.shape),
             "mps_absmax": float(np.abs(mps64).max()),
             "cpu_absmax": float(np.abs(cpu64).max()),
             "mps_nonfinite": int((~np.isfinite(mps64)).sum())}
    if cpu_norm:
        entry["nrmse_vs_cpu"] = float(np.linalg.norm(mps64 - cpu64) / cpu_norm)
    if mps_norm and cpu_norm:
        entry["cosine_vs_cpu"] = float((mps64 * cpu64).sum() / (mps_norm * cpu_norm))
    results.append(entry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=257)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--fallback", choices=("0", "1"), default="0")
    args = parser.parse_args()

    results = []
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(args.batch, args.seq, args.hidden, generator=generator, dtype=torch.float32).to(torch.float16)
    mlp_input = torch.randn(args.batch, args.seq, 4 * args.hidden, generator=generator,
                           dtype=torch.float32).to(torch.float16)

    def build(module_class, *shape):
        module = module_class(*shape).to(torch.float16)
        with torch.no_grad():
            module.weight.normal_(0.0, 0.02, generator=generator)
            module.bias.zero_()
        return module

    layernorm = build(torch.nn.LayerNorm, args.hidden)
    qkv = build(torch.nn.Linear, args.hidden, 3 * args.hidden)
    proj = build(torch.nn.Linear, args.hidden, args.hidden)
    mlp_fc1 = build(torch.nn.Linear, args.hidden, 4 * args.hidden)
    mlp_fc2 = build(torch.nn.Linear, 4 * args.hidden, args.hidden)

    head_dim = args.hidden // args.heads
    query = torch.randn(args.batch, args.heads, args.seq, head_dim, generator=generator, dtype=torch.float32).to(torch.float16)
    key = torch.randn(args.batch, args.heads, args.seq, head_dim, generator=generator, dtype=torch.float32).to(torch.float16)
    value = torch.randn(args.batch, args.heads, args.seq, head_dim, generator=generator, dtype=torch.float32).to(torch.float16)

    modules_cpu = {"layernorm": layernorm, "linear_qkv": qkv, "linear_proj": proj,
                   "linear_mlp_fc1": mlp_fc1, "linear_mlp_fc2": mlp_fc2}
    modules_mps = {name: copy.deepcopy(module).to("mps") for name, module in modules_cpu.items()}

    with torch.no_grad():
        for name, module in modules_cpu.items():
            source = mlp_input if name == "linear_mlp_fc2" else hidden
            cpu_result = module(source)
            mps_result = modules_mps[name](source.to("mps"))
            torch.mps.synchronize()
            compare(name, mps_result, cpu_result, results)

        cpu_result = F.gelu(hidden)
        mps_result = F.gelu(hidden.to("mps"))
        torch.mps.synchronize()
        compare("gelu", mps_result, cpu_result, results)

        cpu_result = F.silu(hidden)
        mps_result = F.silu(hidden.to("mps"))
        torch.mps.synchronize()
        compare("silu", mps_result, cpu_result, results)

        cpu_result = F.scaled_dot_product_attention(query, key, value)
        mps_result = F.scaled_dot_product_attention(query.to("mps"), key.to("mps"), value.to("mps"))
        torch.mps.synchronize()
        compare("sdpa", mps_result, cpu_result, results)

        cpu_result = torch.softmax(query, dim=-1)
        mps_result = torch.softmax(query.to("mps"), dim=-1)
        torch.mps.synchronize()
        compare("softmax", mps_result, cpu_result, results)

        cpu_result = hidden + hidden
        mps_result = hidden.to("mps") + hidden.to("mps")
        torch.mps.synchronize()
        compare("add", mps_result, cpu_result, results)

        cpu_result = hidden * hidden
        mps_result = hidden.to("mps") * hidden.to("mps")
        torch.mps.synchronize()
        compare("mul", mps_result, cpu_result, results)

    payload = {"torch": torch.__version__, "fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
               "batch": args.batch, "seq": args.seq, "hidden": args.hidden, "heads": args.heads,
               "results": results}
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

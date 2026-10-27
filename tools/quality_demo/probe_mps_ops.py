"""Bounded operator parity probes. CPU is only the small reference oracle."""
import json
import os
from pathlib import Path
import sys

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
    raise RuntimeError("unset PYTORCH_ENABLE_MPS_FALLBACK before running probes")

import torch
import torch.nn.functional as F


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    torch.manual_seed(20261027)
    results = []

    def check(name, fn, inputs, dtype):
        cpu = [x.to(dtype) if x.is_floating_point() else x for x in inputs]
        mps = [x.to("mps") for x in cpu]
        row = {"name": name, "dtype": str(dtype), "status": "failed"}
        try:
            # CPU antialiased bicubic has no Half kernel. Use a float32 oracle
            # on the same already-rounded inputs, rather than skipping MPS.
            oracle = [x.float() if x.is_floating_point() else x for x in cpu]
            expected = fn(*oracle)
            row["oracle_dtype"] = "torch.float32"
            actual = fn(*mps)
            torch.mps.synchronize()
            assert actual.device.type == "mps"
            assert torch.isfinite(actual).all()
            actual_cpu = actual.cpu()
            row["max_abs_error"] = (actual_cpu.float() - expected.float()).abs().max().item()
            tolerance = (1e-4, 1e-5) if dtype == torch.float32 else (1e-2, 2e-3)
            torch.testing.assert_close(actual_cpu.float(), expected, rtol=tolerance[0], atol=tolerance[1])
            row["status"] = "passed"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        results.append(row)

    for dtype in (torch.float32, torch.float16):
        image = torch.randn(1, 3, 31, 29)
        for aa in (False, True):
            check(f"bicubic_antialias_{aa}", lambda x: F.interpolate(
                x, size=(17, 19), mode="bicubic", align_corners=False, antialias=aa), [image], dtype)
        q, k, v = [torch.randn(2, 4, 24, 16) for _ in range(3)]
        mask = torch.ones(24, 24, dtype=torch.bool).tril()
        for noncontiguous in (False, True):
            for masked in (False, True):
                def attention(a, b, c, m):
                    if noncontiguous:
                        a, b, c = [x.transpose(1, 2) for x in (a, b, c)]
                    return F.scaled_dot_product_attention(a, b, c, attn_mask=m if masked else None)
                size = 4 if noncontiguous else 24
                mask = torch.ones(size, size, dtype=torch.bool).tril()
                check(f"sdpa_noncontig_{noncontiguous}_masked_{masked}", attention, [q, k, v, mask], dtype)
        source = torch.randn(131072, 4)
        indices = torch.randint(0, source.shape[0], (20000,))
        check("bounded_index_gather", lambda x, idx: x[idx], [source, indices], dtype)
        target = torch.zeros(128, 4)
        index = torch.arange(128).repeat(2)
        values = torch.randn(256, 4)
        check("index_add", lambda x, i, v: x.index_add(0, i, v), [target, index, values], dtype)
    report = {"torch": torch.__version__, "torch_file": torch.__file__,
              "fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"), "results": results}
    Path(sys.argv[1]).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if any(r["status"] != "passed" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

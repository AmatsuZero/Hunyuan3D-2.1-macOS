"""Isolate the DINOv2 embedding components on MPS.

Splits the real Dinov2Embeddings forward into patch convolution, position
encoding interpolation and the final sum, and reports each against a CPU
reference computed from a CPU copy of the same weights. Diagnostic only.
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
    parser.add_argument("--checkout", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dino-dir", default="/Users/daubert/Github/hunyuan3d-artifacts/dinov2-giant")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--fallback", choices=("0", "1"), default="0")
    args = parser.parse_args()

    checkout = Path(args.checkout).resolve()
    paint = checkout / "hy3dpaint"
    for path in (str(paint / "custom_rasterizer"), str(paint), str(checkout)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from PIL import Image
    from hunyuanpaintpbr.unet.modules import Dino_v2

    results = []
    dino = Dino_v2(args.dino_dir).to(torch.float16).to(args.device)
    reference = Image.open(Path(args.fixtures) / "reference.png").convert("RGB")

    processor = dino.dino_processor
    pixels = {}
    original_processor = processor

    def processor_call(*call_args, **call_kwargs):
        result = original_processor(*call_args, **call_kwargs)
        pixels["values"] = result.pixel_values
        return result

    dino.dino_processor = processor_call
    with torch.no_grad():
        dino(reference)
    dino.dino_processor = original_processor
    pixel_values = pixels["values"]

    device_module = dino.dino_v2.embeddings
    cpu_module = copy.deepcopy(device_module).to("cpu")
    with torch.no_grad():
        target_dtype = device_module.patch_embeddings.projection.weight.dtype
        cpu_pixels = pixel_values.to("cpu", dtype=target_dtype)
        mps_pixels = pixel_values.to(args.device, dtype=target_dtype)

        cpu_conv = cpu_module.patch_embeddings.projection(cpu_pixels).flatten(2).transpose(1, 2)
        mps_conv = device_module.patch_embeddings.projection(mps_pixels).flatten(2).transpose(1, 2)
        if args.device == "mps":
            torch.mps.synchronize()
            compare("patch_conv", mps_conv, cpu_conv, results)

        dim = cpu_conv.shape[-1]
        patch_pos = device_module.position_embeddings[:, 1:]
        source_positions = int((device_module.position_embeddings.shape[1] - 1) ** 0.5)
        target_positions = int(cpu_conv.shape[1] ** 0.5)
        grid = patch_pos.reshape(1, source_positions, source_positions, dim).permute(0, 3, 1, 2)

        cpu_interp = F.interpolate(grid.to("cpu").to(torch.float32),
                                   size=(target_positions, target_positions), mode="bicubic",
                                   align_corners=False).to(torch.float16)
        mps_interp = F.interpolate(grid.to(args.device).to(torch.float32),
                                   size=(target_positions, target_positions), mode="bicubic",
                                   align_corners=False).to(torch.float16)
        if args.device == "mps":
            torch.mps.synchronize()
            compare("pos_interp_bicubic", mps_interp, cpu_interp, results)

        cpu_cls = cpu_module.cls_token.expand(1, -1, -1)
        mps_cls = device_module.cls_token.expand(1, -1, -1)
        class_pos = device_module.position_embeddings[:, :1]

        cpu_interp_tokens = cpu_interp.permute(0, 2, 3, 1).reshape(1, -1, dim)
        mps_interp_tokens = mps_interp.permute(0, 2, 3, 1).reshape(1, -1, dim)
        cpu_sum = torch.cat((cpu_cls, cpu_conv), dim=1) + torch.cat((class_pos.to("cpu"), cpu_interp_tokens), dim=1)
        mps_sum = torch.cat((mps_cls, mps_conv), dim=1) + torch.cat((class_pos.to(args.device), mps_interp_tokens), dim=1)
        if args.device == "mps":
            torch.mps.synchronize()
            compare("embeddings_sum", mps_sum, cpu_sum, results)

    payload = {"torch": torch.__version__, "device": args.device,
               "fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"), "results": results}
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

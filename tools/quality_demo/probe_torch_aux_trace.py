"""Trace one real-size VAE decode or RealESRGAN forward, without mesh writes.

Run under xctrace Metal System Trace with the os_signpost instrument. Timings
include per-operation synchronization and are not performance measurements.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("vae.decode", "super_resolution"), required=True)
    parser.add_argument("--input", type=Path, required=True,
                        help="final scaled diffusion latents .npy, or a generated 512x512 RGB PNG")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--model-dir", type=Path, default=Path(
        "/Users/daubert/Github/hunyuan3d-artifacts/Hunyuan3D-2.1-modelscope/hunyuan3d-paintpbr-v2-1"))
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        parser.error("fallback must be disabled before Torch import")
    if not args.input.is_file():
        parser.error("input does not exist")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out = args.output_dir.resolve()

    import numpy as np
    import torch
    from PIL import Image
    from probe_fixed_conditions import add_checkout_paths, trace_scope, _runtime_object_source

    add_checkout_paths(args.checkout)
    from utils.execution_audit import ExecutionAudit

    audit = ExecutionAudit(required=[args.stage], event_path=out / "events.jsonl")
    audit.require_mps()
    report = {"stage": args.stage, "torch": torch.__version__, "torch_file": torch.__file__,
              "fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
              "input": str(args.input.resolve()),
              "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "status": "failed", "trace_stage": args.stage}
    original = None
    try:
        if args.stage == "vae.decode":
            from diffusers import AutoencoderKL

            array = np.load(args.input, allow_pickle=False)
            if array.shape != (12, 4, 64, 64) or array.dtype != np.float16 or not np.isfinite(array).all():
                raise ValueError("decode fixture must be finite float16 (12,4,64,64)")
            model = AutoencoderKL.from_pretrained(args.model_dir, subfolder="vae",
                torch_dtype=torch.float16, local_files_only=True, use_safetensors=False).eval().to("mps")
            model.requires_grad_(False)
            # Match HunyuanPaintPipeline's decode, with scaling before profiling.
            tensor = torch.from_numpy(array).to("mps") / model.config.scaling_factor
            report["scaling_factor"] = model.config.scaling_factor
            report["slicing"] = model.use_slicing
            report["tiling"] = model.use_tiling
            method = "decode"
        else:
            from utils.torchvision_fix import apply_fix
            apply_fix()
            from utils.image_super_utils import imageSuperNet

            image = Image.open(args.input).convert("RGB")
            if image.size != (512, 512):
                raise ValueError("super-resolution fixture must be 512x512")
            config = SimpleNamespace(realesrgan_ckpt_path=str(args.checkout / "ckpt/RealESRGAN_x4plus.pth"))
            super_model = imageSuperNet(config)
            model = super_model.upsampler.model
            method = "forward"

        report["model_source"] = _runtime_object_source(args.stage, model)
        original = getattr(model, method)

        def profiled(*pos, **kw):
            with trace_scope(args.stage, args.stage, report):
                return original(*pos, **kw)

        # Audit wraps profiling: metadata checks and their CPU work stay outside.
        setattr(model, method, profiled)
        audit.wrap(args.stage, model, method)
        with torch.no_grad():
            if args.stage == "vae.decode":
                result = model.decode(tensor, return_dict=False)[0].cpu().numpy()
            else:
                result = np.asarray(super_model(image))
        if not np.isfinite(result).all():
            raise ValueError("nonfinite model output")
        audit.validate()
        np.save(out / "output.npy", result)
        report["output"] = {"shape": list(result.shape), "dtype": str(result.dtype),
                            "sha256": hashlib.sha256(result.tobytes()).hexdigest()}
        report["status"] = "passed_pending_trace_correlation"
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        audit.close()
        if original is not None:
            setattr(model, method, original)
        audit.save(out / "audit.json", **report)
        (out / "trace.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

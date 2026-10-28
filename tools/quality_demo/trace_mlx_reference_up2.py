"""Trace frozen Torch/MLX reference forwards around the up1/up2 boundary.

Run each backend in a separate process. Each output directory must be fresh.
This diagnostic does not generate images or change the dispatch path.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
PAINT = PROJECT / "hy3dpaint"
sys.path.insert(0, str(PAINT))

POINTS = (
    "conv_in", "down_0", "down_1", "down_2", "down_3", "mid",
    "up_0.0.concat", "up_0.0.resnet", "up_0.1.concat", "up_0.1.resnet",
    "up_0.2.concat", "up_0.2.resnet", "up_0.upsample_input",
    "up_0.upsample_nearest", "up_0.upsample", "up_0.output",
    "up_1.0.concat", "up_1.0.resnet", "up_1.0.attn",
    "up_1.1.concat", "up_1.1.resnet", "up_1.1.attn",
    "up_1.2.concat", "up_1.2.resnet", "up_1.2.attn",
    "up_1.upsample_input", "up_1.upsample_nearest", "up_1.upsample",
    "up_2.0.concat", "up_2.0.resnet", "up_2_0.attn_input",
    "up_2_0.norm", "up_2_0.proj_in", "up_2_0_0.norm1",
    "up_1_1_0.norm1", "up_1_1_0.attn1", "up_1_1_0.norm2",
    "up_1_1_0.attn2", "up_1_1_0.norm3", "up_1_1_0.ff",
    "up_1_1_0.ff_proj", "up_1_1_0.ff_geglu", "up_1_1_0.output",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_input(capture):
    names = ("reference_sample", "reference_timestep", "reference_encoder_hidden_states")
    paths = {name: capture / f"{name}.npy" for name in names}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths, {name: np.load(path, allow_pickle=False) for name, path in paths.items()}


def save(output, name, value, *, nhwc=False):
    if name not in POINTS:
        raise ValueError(f"unexpected trace point: {name}")
    if isinstance(value, (tuple, list)):
        value = value[0]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    else:
        value = np.array(value)
    if nhwc and value.ndim == 4:
        value = value.transpose(0, 3, 1, 2)
    np.save(output / f"{name}.npy", value, allow_pickle=False)


def run_torch(args, inputs):
    import torch
    from hunyuanpaintpbr.pipeline import HunyuanPaintPipeline

    if torch.backends.mps.is_available() is False:
        raise RuntimeError("MPS is unavailable")
    pipe = HunyuanPaintPipeline.from_pretrained(str(args.model_dir), torch_dtype=torch.float16)
    dual = pipe.unet.unet_dual.to("mps").eval()
    if args.torch_fp32:
        dual = dual.float()
    hooks = []

    def output_hook(name):
        def hook(_module, _inputs, result):
            save(args.output, name, result)
        return hook

    def input_hook(name):
        def hook(_module, values):
            save(args.output, name, values[0])
        return hook

    hooks.append(dual.conv_in.register_forward_hook(output_hook("conv_in")))
    for index, block in enumerate(dual.down_blocks):
        hooks.append(block.register_forward_hook(output_hook(f"down_{index}")))
    hooks.append(dual.mid_block.register_forward_hook(output_hook("mid")))

    for block_index in (0, 1, 2):
        block = dual.up_blocks[block_index]
        indices = (0, 1, 2) if block_index in (0, 1) else (0,)
        for index in indices:
            prefix = f"up_{block_index}.{index}"
            hooks.append(block.resnets[index].register_forward_pre_hook(input_hook(f"{prefix}.concat")))
            hooks.append(block.resnets[index].register_forward_hook(output_hook(f"{prefix}.resnet")))
            if block_index == 1:
                hooks.append(block.attentions[index].register_forward_hook(output_hook(f"{prefix}.attn")))
                if index == 1:
                    transformer = block.attentions[index].transformer_blocks[0]
                    for name in ("norm1", "attn1", "norm2", "attn2", "norm3", "ff"):
                        hooks.append(getattr(transformer, name).register_forward_hook(
                            output_hook(f"up_1_1_0.{name}")
                        ))
                    hooks.append(transformer.ff.net[0].proj.register_forward_hook(
                        output_hook("up_1_1_0.ff_proj")
                    ))
                    hooks.append(transformer.ff.net[0].register_forward_hook(
                        output_hook("up_1_1_0.ff_geglu")
                    ))
                    hooks.append(transformer.register_forward_hook(
                        output_hook("up_1_1_0.output")
                    ))
            elif block_index == 2:
                attn = block.attentions[index]
                hooks.append(attn.register_forward_pre_hook(input_hook("up_2_0.attn_input")))
                # The MLX trace records the post-GroupNorm NHWC tensor after
                # flattening. Capture the exact sequence fed to proj_in on Torch.
                hooks.append(attn.proj_in.register_forward_pre_hook(input_hook("up_2_0.norm")))
                hooks.append(attn.proj_in.register_forward_hook(output_hook("up_2_0.proj_in")))
                hooks.append(attn.transformer_blocks[0].norm1.register_forward_hook(
                    output_hook("up_2_0_0.norm1")
                ))
        if block_index in (0, 1):
            if block_index == 0:
                hooks.append(block.register_forward_hook(output_hook("up_0.output")))
            upsample_prefix = f"up_{block_index}"
            hooks.append(block.upsamplers[0].register_forward_pre_hook(
                input_hook(f"{upsample_prefix}.upsample_input")
            ))
            hooks.append(block.upsamplers[0].conv.register_forward_pre_hook(
                input_hook(f"{upsample_prefix}.upsample_nearest")
            ))
            hooks.append(block.upsamplers[0].register_forward_hook(
                output_hook(f"{upsample_prefix}.upsample")
            ))

    cache = {}
    try:
        with torch.no_grad():
            result = dual(
                torch.from_numpy(inputs["reference_sample"]).to(
                    device="mps", dtype=torch.float32 if args.torch_fp32 else torch.float16
                ),
                int(inputs["reference_timestep"]),
                encoder_hidden_states=torch.from_numpy(
                    inputs["reference_encoder_hidden_states"]
                ).to(
                    device="mps", dtype=torch.float32 if args.torch_fp32 else torch.float16
                ),
                return_dict=False,
                cross_attention_kwargs={
                    "mode": "w", "num_in_batch": 1, "condition_embed_dict": cache,
                },
            )
        torch.mps.synchronize()
    finally:
        for hook in hooks:
            hook.remove()
    return {
        "runtime_class": f"{type(dual).__module__}.{type(dual).__name__}",
        "torch_version": torch.__version__,
        "torch_precision": "fp32" if args.torch_fp32 else "fp16",
        "cache_layers": len(cache),
        "reference_write_shape": list(result[0].shape),
    }


def run_mlx(args, inputs):
    import mlx.core as mx
    from mlx_unet import HunyuanInnerUNet
    from mlx_utils import _tree_map_to_f32, load_reference_weights_from_pytorch

    mx.set_cache_limit(1024 ** 3)
    model = HunyuanInnerUNet(input_channels=4, transformer_norm_eps=args.transformer_norm_eps)
    mapped = load_reference_weights_from_pytorch(model, str(args.checkpoint))
    model.update(_tree_map_to_f32(model.parameters()))
    mx.eval(model.parameters())

    def trace(name, value):
        if name not in POINTS:
            return
        mx.eval(value)
        save(args.output, name, value, nhwc=True)

    model._reference_debug_trace = trace
    model.up_block_0._reference_debug_trace = trace
    for index in (1, 2):
        block = getattr(model, f"up_block_{index}")
        block._reference_debug_trace = trace
        block._reference_trace_index = index
    model.up_block_1.attentions[1]._reference_debug_trace = trace
    model.up_block_1.attentions[1].transformer_blocks[0]._reference_debug_trace = trace
    attn = model.up_block_2.attentions[0]
    attn._reference_debug_trace = trace
    attn.transformer_blocks[0]._reference_debug_trace = trace
    sample = mx.array(inputs["reference_sample"]).astype(mx.float32)
    encoder = mx.array(inputs["reference_encoder_hidden_states"]).astype(mx.float32)
    timestep = mx.array([int(inputs["reference_timestep"])], dtype=mx.float32)
    cache = {}
    result = model(sample, timestep, encoder, num_in_batch=1,
                   condition_embed_dict={}, dino_hidden_states=None,
                   mode="w", reference_cache=cache, debug_trace=trace)
    mx.eval([result, *cache.values()])
    return {"mapped_weights": mapped, "cache_layers": len(cache),
            "reference_write_shape": list(result.shape), "precision": "fp32",
            "transformer_norm_eps": args.transformer_norm_eps}


def compare(args):
    torch_meta = json.loads((args.torch_trace / "trace.json").read_text())
    mlx_meta = json.loads((args.mlx_trace / "trace.json").read_text())
    if torch_meta["input_sha256"] != mlx_meta["input_sha256"]:
        raise ValueError("trace inputs differ")
    if torch_meta["checkpoint_sha256"] != mlx_meta["checkpoint_sha256"]:
        raise ValueError("trace checkpoints differ")
    rows = {}
    for name in POINTS:
        left = np.load(args.torch_trace / f"{name}.npy", allow_pickle=False)
        right = np.load(args.mlx_trace / f"{name}.npy", allow_pickle=False)
        if left.shape != right.shape:
            raise ValueError(f"{name}: shape {left.shape} != {right.shape}")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"{name}: non-finite trace value")
        difference = left.astype(np.float64) - right.astype(np.float64)
        rows[name] = {
            "shape": list(left.shape),
            "torch_dtype": str(left.dtype), "mlx_dtype": str(right.dtype),
            "nrmse": float(np.linalg.norm(difference) / (np.linalg.norm(left.astype(np.float64)) + 1e-12)),
            "mean_abs": float(np.abs(difference).mean()),
            "max_abs": float(np.abs(difference).max()),
        }
    return {"torch_trace": str(args.torch_trace.resolve()),
            "mlx_trace": str(args.mlx_trace.resolve()),
            "input_sha256": torch_meta["input_sha256"],
            "checkpoint_sha256": torch_meta["checkpoint_sha256"], "points": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "mlx", "compare"), required=True)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path(
        "/Users/daubert/Github/hunyuan3d-artifacts/Hunyuan3D-2.1-modelscope/hunyuan3d-paintpbr-v2-1"
    ))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--torch-fp32", action="store_true",
                        help="run the Torch reference model and inputs in float32")
    parser.add_argument("--transformer-norm-eps", type=float, default=1e-6,
                        help="MLX reference Transformer2D GroupNorm epsilon")
    parser.add_argument("--torch-trace", type=Path)
    parser.add_argument("--mlx-trace", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.backend == "compare":
        report = compare(args)
    else:
        if args.capture is None:
            parser.error("--capture is required for forward trace")
        paths, inputs = capture_input(args.capture)
        if args.backend == "mlx" and args.checkpoint is None:
            args.checkpoint = args.model_dir / "unet/diffusion_pytorch_model.bin"
        checkpoint_path = (args.checkpoint if args.backend == "mlx"
                           else args.model_dir / "unet/diffusion_pytorch_model.bin")
        report = {
            "backend": args.backend,
            "capture": str(args.capture.resolve()),
            "input_sha256": {name: sha256(path) for name, path in paths.items()},
            "checkpoint_sha256": sha256(checkpoint_path),
        }
        result = run_torch(args, inputs) if args.backend == "torch" else run_mlx(args, inputs)
        report.update(result)
        missing = [name for name in POINTS if not (args.output / f"{name}.npy").is_file()]
        if missing:
            raise RuntimeError(f"missing trace points: {missing}")
    (args.output / "trace.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

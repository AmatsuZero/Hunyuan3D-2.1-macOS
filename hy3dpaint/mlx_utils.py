"""MLX dispatch wrapper and weight loader for Hunyuan3D inner UNet.

Provides MLXDispatchUNet: a drop-in replacement for the inner UNet that
runs generation passes (steps 2-15) on MLX for ~3-5x speedup over PyTorch MPS.

Usage in textureGenPipeline.py:
    from mlx_utils import setup_mlx_dispatch
    setup_mlx_dispatch(self.models["multiview_model"].pipeline)

Note: mx.eval() is MLX's graph evaluation function — it forces lazy computation
to complete on the Metal GPU. It is NOT Python's built-in eval().
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

import mlx.core as mx
import mlx.nn as mlx_nn

try:
    from .mlx_unet import HunyuanInnerUNet
except ImportError:
    from mlx_unet import HunyuanInnerUNet


# ──────────────────────────────────────────────────────────────────────────────
# Weight Loading
# ──────────────────────────────────────────────────────────────────────────────

def _pt_to_mx(tensor: torch.Tensor) -> mx.array:
    """Convert PyTorch tensor to MLX array, preserving dtype."""
    return mx.array(tensor.cpu().numpy())


def _conv_weight_pt_to_mlx(w: torch.Tensor) -> mx.array:
    """Convert Conv2d weight from PyTorch OIHW to MLX OHWI."""
    return mx.array(w.cpu().permute(0, 2, 3, 1).numpy())


def _load_upsample_weights(upsample, prefix: str, state_dict: dict):
    """Load a Diffusers upsampler Conv2d into the MLX OHWI layout."""
    upsample.conv.weight = _conv_weight_pt_to_mlx(state_dict[f"{prefix}.weight"])
    upsample.conv.bias = _pt_to_mx(state_dict[f"{prefix}.bias"])


def _load_attention_weights(block, prefix: str, state_dict: dict, dim: int):
    """Load weights for a Basic2p5DTransformerBlock from PyTorch state dict."""
    p = prefix

    def get(key):
        return _pt_to_mx(state_dict[key])

    # Self-attention (albedo) — under transformer.attn1.*
    block.attn1_to_q.weight = get(f"{p}.transformer.attn1.to_q.weight")
    block.attn1_to_k.weight = get(f"{p}.transformer.attn1.to_k.weight")
    block.attn1_to_v.weight = get(f"{p}.transformer.attn1.to_v.weight")
    block.attn1_to_out.weight = get(f"{p}.transformer.attn1.to_out.0.weight")
    block.attn1_to_out.bias = get(f"{p}.transformer.attn1.to_out.0.bias")

    # Self-attention (MR) — under transformer.attn1.processor.*
    block.attn1_to_q_mr.weight = get(f"{p}.transformer.attn1.processor.to_q_mr.weight")
    block.attn1_to_k_mr.weight = get(f"{p}.transformer.attn1.processor.to_k_mr.weight")
    block.attn1_to_v_mr.weight = get(f"{p}.transformer.attn1.processor.to_v_mr.weight")
    block.attn1_to_out_mr.weight = get(f"{p}.transformer.attn1.processor.to_out_mr.0.weight")
    block.attn1_to_out_mr.bias = get(f"{p}.transformer.attn1.processor.to_out_mr.0.bias")

    # Reference attention
    block.attn_ref_to_q.weight = get(f"{p}.attn_refview.to_q.weight")
    block.attn_ref_to_k.weight = get(f"{p}.attn_refview.to_k.weight")
    block.attn_ref_to_v.weight = get(f"{p}.attn_refview.to_v.weight")
    block.attn_ref_to_out.weight = get(f"{p}.attn_refview.to_out.0.weight")
    block.attn_ref_to_out.bias = get(f"{p}.attn_refview.to_out.0.bias")
    block.attn_ref_to_v_mr.weight = get(f"{p}.attn_refview.processor.to_v_mr.weight")
    block.attn_ref_to_out_mr.weight = get(f"{p}.attn_refview.processor.to_out_mr.0.weight")
    block.attn_ref_to_out_mr.bias = get(f"{p}.attn_refview.processor.to_out_mr.0.bias")

    # Multiview attention
    block.attn_mv_to_q.weight = get(f"{p}.attn_multiview.to_q.weight")
    block.attn_mv_to_k.weight = get(f"{p}.attn_multiview.to_k.weight")
    block.attn_mv_to_v.weight = get(f"{p}.attn_multiview.to_v.weight")
    block.attn_mv_to_out.weight = get(f"{p}.attn_multiview.to_out.0.weight")
    block.attn_mv_to_out.bias = get(f"{p}.attn_multiview.to_out.0.bias")

    # Text cross-attention
    block.attn2_to_q.weight = get(f"{p}.transformer.attn2.to_q.weight")
    block.attn2_to_k.weight = get(f"{p}.transformer.attn2.to_k.weight")
    block.attn2_to_v.weight = get(f"{p}.transformer.attn2.to_v.weight")
    block.attn2_to_out.weight = get(f"{p}.transformer.attn2.to_out.0.weight")
    block.attn2_to_out.bias = get(f"{p}.transformer.attn2.to_out.0.bias")

    # DINO cross-attention
    block.attn_dino_to_q.weight = get(f"{p}.attn_dino.to_q.weight")
    block.attn_dino_to_k.weight = get(f"{p}.attn_dino.to_k.weight")
    block.attn_dino_to_v.weight = get(f"{p}.attn_dino.to_v.weight")
    block.attn_dino_to_out.weight = get(f"{p}.attn_dino.to_out.0.weight")
    block.attn_dino_to_out.bias = get(f"{p}.attn_dino.to_out.0.bias")

    # Norms
    block.norm1.weight = get(f"{p}.transformer.norm1.weight")
    block.norm1.bias = get(f"{p}.transformer.norm1.bias")
    block.norm2.weight = get(f"{p}.transformer.norm2.weight")
    block.norm2.bias = get(f"{p}.transformer.norm2.bias")
    block.norm3.weight = get(f"{p}.transformer.norm3.weight")
    block.norm3.bias = get(f"{p}.transformer.norm3.bias")

    # FFN (GEGLU)
    block.ff_proj.weight = get(f"{p}.transformer.ff.net.0.proj.weight")
    block.ff_proj.bias = get(f"{p}.transformer.ff.net.0.proj.bias")
    block.ff_out.weight = get(f"{p}.transformer.ff.net.2.weight")
    block.ff_out.bias = get(f"{p}.transformer.ff.net.2.bias")


def _load_resnet_weights(resnet, prefix: str, state_dict: dict):
    """Load ResnetBlock2D weights."""
    def get(key):
        return _pt_to_mx(state_dict[key])
    def get_conv(key):
        return _conv_weight_pt_to_mlx(state_dict[key])

    resnet.norm1.weight = get(f"{prefix}.norm1.weight")
    resnet.norm1.bias = get(f"{prefix}.norm1.bias")
    resnet.conv1.weight = get_conv(f"{prefix}.conv1.weight")
    resnet.conv1.bias = get(f"{prefix}.conv1.bias")
    resnet.time_emb_proj.weight = get(f"{prefix}.time_emb_proj.weight")
    resnet.time_emb_proj.bias = get(f"{prefix}.time_emb_proj.bias")
    resnet.norm2.weight = get(f"{prefix}.norm2.weight")
    resnet.norm2.bias = get(f"{prefix}.norm2.bias")
    resnet.conv2.weight = get_conv(f"{prefix}.conv2.weight")
    resnet.conv2.bias = get(f"{prefix}.conv2.bias")

    shortcut_key = f"{prefix}.conv_shortcut.weight"
    if shortcut_key in state_dict:
        resnet.conv_shortcut.weight = get_conv(shortcut_key)
        resnet.conv_shortcut.bias = get(f"{prefix}.conv_shortcut.bias")


def _load_transformer2d_weights(transformer2d, prefix: str, state_dict: dict):
    """Load Transformer2D (norm + proj_in + blocks + proj_out)."""
    def get(key):
        return _pt_to_mx(state_dict[key])

    transformer2d.norm.weight = get(f"{prefix}.norm.weight")
    transformer2d.norm.bias = get(f"{prefix}.norm.bias")
    transformer2d.proj_in.weight = get(f"{prefix}.proj_in.weight")
    transformer2d.proj_in.bias = get(f"{prefix}.proj_in.bias")
    transformer2d.proj_out.weight = get(f"{prefix}.proj_out.weight")
    transformer2d.proj_out.bias = get(f"{prefix}.proj_out.bias")

    for i, block in enumerate(transformer2d.transformer_blocks):
        _load_attention_weights(
            block, f"{prefix}.transformer_blocks.{i}", state_dict, block.dim
        )


class _TrackedStateDict(dict):
    """Record checkpoint keys actually consumed by the explicit MLX mapping."""

    def __init__(self, values):
        super().__init__(values)
        self.used_keys = set()

    def __getitem__(self, key):
        self.used_keys.add(key)
        return super().__getitem__(key)


def load_weights_from_pytorch(model: HunyuanInnerUNet, ckpt_path: str,
                              return_used_keys: bool = False):
    """Load all weights from PyTorch checkpoint into MLX model."""
    print(f"  Loading PyTorch checkpoint: {ckpt_path}")
    t0 = time.time()
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    print(f"  Checkpoint loaded in {time.time() - t0:.1f}s ({len(state_dict)} keys)")

    def get(key):
        return _pt_to_mx(state_dict[key])
    def get_conv(key):
        return _conv_weight_pt_to_mlx(state_dict[key])

    # Strip "unet." prefix if present
    if any(k.startswith("unet.") for k in state_dict):
        state_dict = {k.replace("unet.", "", 1): v for k, v in state_dict.items()}
    if return_used_keys:
        state_dict = _TrackedStateDict(state_dict)

    t0 = time.time()

    # Time embedding
    model.time_embedding.linear_1.weight = get("time_embedding.linear_1.weight")
    model.time_embedding.linear_1.bias = get("time_embedding.linear_1.bias")
    model.time_embedding.linear_2.weight = get("time_embedding.linear_2.weight")
    model.time_embedding.linear_2.bias = get("time_embedding.linear_2.bias")

    # Conv in/out
    model.conv_in.weight = get_conv("conv_in.weight")
    model.conv_in.bias = get("conv_in.bias")
    model.conv_norm_out.weight = get("conv_norm_out.weight")
    model.conv_norm_out.bias = get("conv_norm_out.bias")
    model.conv_out.weight = get_conv("conv_out.weight")
    model.conv_out.bias = get("conv_out.bias")

    # Down blocks
    for block_idx, block in enumerate([model.down_block_0, model.down_block_1,
                                        model.down_block_2]):
        prefix = f"down_blocks.{block_idx}"
        for i, resnet in enumerate(block.resnets):
            _load_resnet_weights(resnet, f"{prefix}.resnets.{i}", state_dict)
        for i, attn in enumerate(block.attentions):
            _load_transformer2d_weights(attn, f"{prefix}.attentions.{i}", state_dict)
        if block.has_downsample:
            ds_prefix = f"{prefix}.downsamplers.0.conv"
            block.downsample.conv.weight = get_conv(f"{ds_prefix}.weight")
            block.downsample.conv.bias = get(f"{ds_prefix}.bias")

    # Down block 3 (no attention)
    for i, resnet in enumerate(model.down_block_3.resnets):
        _load_resnet_weights(resnet, f"down_blocks.3.resnets.{i}", state_dict)

    # Mid block
    _load_resnet_weights(model.mid_resnet_0, "mid_block.resnets.0", state_dict)
    _load_resnet_weights(model.mid_resnet_1, "mid_block.resnets.1", state_dict)
    _load_transformer2d_weights(model.mid_attn, "mid_block.attentions.0", state_dict)

    # Up blocks
    # up_block_0: UpBlock2D (no attention)
    for i, resnet in enumerate(model.up_block_0.resnets):
        _load_resnet_weights(resnet, f"up_blocks.0.resnets.{i}", state_dict)
    if model.up_block_0.has_upsample:
        _load_upsample_weights(
            model.up_block_0.upsample,
            "up_blocks.0.upsamplers.0.conv",
            state_dict,
        )

    # up_blocks 1-3: CrossAttnUpBlock2D
    for block_idx, block in enumerate([model.up_block_1, model.up_block_2,
                                        model.up_block_3], start=1):
        prefix = f"up_blocks.{block_idx}"
        for i, resnet in enumerate(block.resnets):
            _load_resnet_weights(resnet, f"{prefix}.resnets.{i}", state_dict)
        for i, attn in enumerate(block.attentions):
            _load_transformer2d_weights(attn, f"{prefix}.attentions.{i}", state_dict)
        if block.has_upsample:
            _load_upsample_weights(block.upsample, f"{prefix}.upsamplers.0.conv", state_dict)

    print(f"  Weights mapped in {time.time() - t0:.1f}s")
    if return_used_keys:
        return state_dict.used_keys


# ──────────────────────────────────────────────────────────────────────────────
# Dispatch Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class MLXDispatchUNet(torch.nn.Module):
    """Drop-in replacement for inner UNet that dispatches gen passes to MLX.

    Default: Step 1 PyTorch, steps 2-15 MLX. With the explicit
    HUNYUAN_MLX_FIRST_STEP=1 opt-in, the outer Torch cache is consumed and
    step 1 also runs on MLX.
    """

    def __init__(self, original_unet, mlx_model: HunyuanInnerUNet, use_fp32: bool = False):
        super().__init__()
        self._original = original_unet
        self._mlx_model = mlx_model
        self._use_fp32 = use_fp32
        self._mlx_dtype = mx.float32 if use_fp32 else mx.float16
        self._gen_call_count = 0
        self._cached_mlx = None  # Pre-converted MLX arrays

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._original, name)

    def forward(self, sample, timestep, encoder_hidden_states, *args, **kwargs):
        cross_attention_kwargs = kwargs.get("cross_attention_kwargs", {}) or {}
        mode = cross_attention_kwargs.get("mode", "")

        # Reference write pass — always PyTorch
        if "w" in mode:
            return self._original(sample, timestep, encoder_hidden_states, *args, **kwargs)

        self._gen_call_count += 1

        # First gen pass — outer UNet already populated the request cache.
        if self._gen_call_count == 1:
            self._cache_constant_inputs(encoder_hidden_states, kwargs)
            if os.environ.get("HUNYUAN_MLX_FIRST_STEP") == "1":
                if not self._cached_mlx.get("condition_embed_dict"):
                    raise RuntimeError("HUNYUAN_MLX_FIRST_STEP requires outer reference cache")
                if os.environ.get("HUNYUAN_MLX_RESIDENT_LOOP") == "1":
                    return self._mlx_forward_raw(sample, timestep)
                return self._mlx_forward(sample, timestep)
            return self._original(sample, timestep, encoder_hidden_states, *args, **kwargs)

        # Steps 2-15 — MLX
        if os.environ.get("HUNYUAN_MLX_COMPARE"):
            return self._compare_forward(sample, timestep, encoder_hidden_states, args, kwargs)
        started = time.perf_counter()
        result = self._mlx_forward(sample, timestep)
        if os.environ.get("HUNYUAN_MLX_TIMING"):
            print(f"  [MLX] Step {self._gen_call_count}: elapsed={time.perf_counter() - started:.3f}s "
                  f"active={mx.get_active_memory() / 1024**3:.3f}GiB "
                  f"cache={mx.get_cache_memory() / 1024**3:.3f}GiB "
                  f"peak={mx.get_peak_memory() / 1024**3:.3f}GiB", flush=True)
        return result

    def _compare_forward(self, sample, timestep, encoder_hidden_states, args, kwargs):
        """Run both PyTorch and MLX, compare outputs numerically."""
        # Print input stats
        s = sample.detach().cpu().float().numpy()
        print(f"\n  [MLX-COMPARE] Input sample: range=[{s.min():.4f}, {s.max():.4f}], "
              f"mean={s.mean():.4f}, dtype={sample.dtype}, shape={list(sample.shape)}")
        print(f"  [MLX-COMPARE] Timestep: {timestep.item():.1f}")

        trace = {}
        hooks = []
        trace_enabled = bool(os.environ.get("HUNYUAN_MLX_TRACE"))
        trace_dir = os.environ.get("HUNYUAN_MLX_TRACE_DIR") if trace_enabled else None
        if trace_enabled:
            def save_torch(name):
                def hook(_module, _inputs, result):
                    value = result[0] if isinstance(result, tuple) else result
                    if name in {"down_0_mda", "down_0_ref", "down_0_mva"}:
                        value = value.reshape(-1, 4096, value.shape[-1])
                    trace[name] = value.detach().cpu().float().numpy()
                return hook

            def save_torch_input(name):
                def hook(_module, inputs, _result):
                    trace[name] = inputs[0].detach().cpu().float().numpy()
                return hook

            hooks.append(self._original.conv_in.register_forward_hook(save_torch("conv_in")))
            hooks.append(self._original.down_blocks[0].resnets[0].register_forward_hook(save_torch("down_0_resnet_0")))
            first_transformer = self._original.down_blocks[0].attentions[0].transformer_blocks[0]
            hooks.append(first_transformer.attn1.register_forward_hook(save_torch("down_0_mda")))
            hooks.append(first_transformer.attn_refview.register_forward_hook(save_torch("down_0_ref")))
            hooks.append(first_transformer.attn_refview.to_q.register_forward_hook(save_torch("down_0_ref_q")))
            hooks.append(first_transformer.attn_refview.to_k.register_forward_hook(save_torch("down_0_ref_k")))
            hooks.append(first_transformer.attn_refview.to_v.register_forward_hook(save_torch("down_0_ref_v")))
            hooks.append(first_transformer.attn_refview.processor.to_v_mr.register_forward_hook(save_torch("down_0_ref_v_mr")))
            hooks.append(first_transformer.attn_refview.to_out[0].register_forward_hook(
                save_torch("down_0_ref_out_a")))
            hooks.append(first_transformer.attn_refview.processor.to_out_mr[0].register_forward_hook(
                save_torch("down_0_ref_out_mr")))
            hooks.append(first_transformer.attn_refview.to_out[0].register_forward_hook(
                save_torch_input("down_0_ref_attn_a")))
            hooks.append(first_transformer.attn_refview.processor.to_out_mr[0].register_forward_hook(
                save_torch_input("down_0_ref_attn_mr")))
            norm_q = getattr(first_transformer.attn_refview, "norm_q", None)
            norm_k = getattr(first_transformer.attn_refview, "norm_k", None)
            print(f"    [REF-SDPA] Torch norms: norm_q={type(norm_q).__name__ if norm_q is not None else None} "
                  f"norm_k={type(norm_k).__name__ if norm_k is not None else None}")
            if norm_q is not None:
                hooks.append(norm_q.register_forward_hook(save_torch("down_0_ref_q_norm")))
            if norm_k is not None:
                hooks.append(norm_k.register_forward_hook(save_torch("down_0_ref_k_norm")))
            hooks.append(first_transformer.attn_multiview.register_forward_hook(save_torch("down_0_mva")))
            hooks.append(self._original.down_blocks[0].attentions[0].register_forward_hook(save_torch("down_0_attn_0")))
            hooks.append(self._original.down_blocks[0].resnets[1].register_forward_hook(save_torch("down_0_resnet_1")))
            hooks.append(self._original.down_blocks[0].attentions[1].register_forward_hook(save_torch("down_0_attn_1")))
            hooks.append(self._original.down_blocks[0].register_forward_hook(save_torch("down_0")))

        # PyTorch MPS
        try:
            pt_out = self._original(sample, timestep, encoder_hidden_states, *args, **kwargs)
        finally:
            for hook in hooks:
                hook.remove()
        pt_arr = pt_out[0].detach().cpu().float().numpy()

        # MLX
        mlx_trace = {}
        def save_mlx(name, value):
            mx.eval(value)
            mlx_trace[name] = np.array(value.transpose(0, 3, 1, 2) if value.ndim == 4 else value).astype(np.float32)

        mlx_started = time.perf_counter()
        mlx_out = self._mlx_forward(sample, timestep, debug_trace=save_mlx if trace else None)
        if os.environ.get("HUNYUAN_MLX_TIMING"):
            print(f"  [MLX] Step {self._gen_call_count}: elapsed={time.perf_counter() - mlx_started:.3f}s "
                  f"active={mx.get_active_memory() / 1024**3:.3f}GiB "
                  f"cache={mx.get_cache_memory() / 1024**3:.3f}GiB "
                  f"peak={mx.get_peak_memory() / 1024**3:.3f}GiB", flush=True)
        mlx_arr = mlx_out[0].detach().cpu().float().numpy()

        if trace_dir:
            destination = Path(trace_dir) / f"step-{self._gen_call_count}"
            destination.mkdir(parents=True, exist_ok=False)
            for source, records in (("torch", trace), ("mlx", mlx_trace)):
                for name, value in records.items():
                    if name.startswith("down_0_ref") or name in {"conv_in", "down_0"}:
                        np.save(destination / f"{source}_{name}.npy", value, allow_pickle=False)

        for name, pt_value in trace.items():
            mlx_value = mlx_trace.get(name)
            if mlx_value is None:
                print(f"    [MLX-TRACE] {name}: missing MLX trace")
                continue
            if pt_value.shape != mlx_value.shape:
                print(f"    [MLX-TRACE] {name}: shape mismatch torch={pt_value.shape} mlx={mlx_value.shape}")
                continue
            difference = pt_value.astype(np.float64) - mlx_value.astype(np.float64)
            norm = np.linalg.norm(pt_value.astype(np.float64))
            print(f"    [MLX-TRACE] {name}: nrmse={np.linalg.norm(difference) / norm:.6f} "
                  f"mean_abs={np.abs(difference).mean():.6f} max_abs={np.abs(difference).max():.6f}")

        # Compare
        abs_diff = np.abs(pt_arr - mlx_arr)
        rel_diff = abs_diff / (np.abs(pt_arr) + 1e-8)
        print(f"\n  [MLX-COMPARE] Step {self._gen_call_count}:")
        print(f"    PT  range: [{pt_arr.min():.4f}, {pt_arr.max():.4f}], mean={pt_arr.mean():.4f}, std={pt_arr.std():.4f}")
        print(f"    MLX range: [{mlx_arr.min():.4f}, {mlx_arr.max():.4f}], mean={mlx_arr.mean():.4f}, std={mlx_arr.std():.4f}")
        print(f"    Abs diff:  mean={abs_diff.mean():.6f}, max={abs_diff.max():.6f}, p99={np.percentile(abs_diff, 99):.6f}")
        print(f"    Rel diff:  mean={rel_diff.mean():.6f}, max={rel_diff.max():.6f}, p99={np.percentile(rel_diff, 99):.6f}")

        # Use PyTorch result (ground truth)
        return pt_out

    def _cache_constant_inputs(self, encoder_hidden_states, kwargs):
        """Convert constant tensors to MLX arrays (cached for steps 2-15)."""
        cross_attn = kwargs.get("cross_attention_kwargs") or {}
        dtype = self._mlx_dtype

        def to_mlx(tensor):
            return _pt_to_mx(tensor).astype(dtype)

        cached = {}
        cached["encoder_hs"] = to_mlx(encoder_hidden_states)
        cached["num_in_batch"] = cross_attn.get("num_in_batch", 6)
        # ref_scale/mva_scale may be float, scalar tensor, or multi-element tensor
        ref_scale = cross_attn.get("ref_scale", 1.0)
        mva_scale = cross_attn.get("mva_scale", 1.0)
        if isinstance(ref_scale, torch.Tensor):
            cached["ref_scale"] = to_mlx(ref_scale) if ref_scale.numel() > 1 else ref_scale.item()
        else:
            cached["ref_scale"] = float(ref_scale)
        if isinstance(mva_scale, torch.Tensor):
            cached["mva_scale"] = to_mlx(mva_scale) if mva_scale.numel() > 1 else mva_scale.item()
        else:
            cached["mva_scale"] = float(mva_scale)

        # DINO hidden states
        dino = cross_attn.get("dino_hidden_states")
        if dino is not None:
            cached["dino_hs"] = to_mlx(dino)

        # Condition embed dict
        cond_dict = cross_attn.get("condition_embed_dict", {})
        cached["condition_embed_dict"] = {
            name: to_mlx(tensor) for name, tensor in cond_dict.items()
        }

        # Position voxel indices (int32, no upcast needed)
        pvox = cross_attn.get("position_voxel_indices", {})
        cached["position_voxel_indices"] = {}
        for key, entry in pvox.items():
            cached["position_voxel_indices"][key] = {
                "voxel_indices": _pt_to_mx(entry["voxel_indices"]),
                "voxel_resolution": entry["voxel_resolution"],
            }

        self._cached_mlx = cached
        precision = "fp32" if self._use_fp32 else "fp16"
        print(f"  [MLX] Cached {len(cached)} inputs for steps 2-15 ({precision})")

    def _mlx_forward(self, sample, timestep, debug_trace=None):
        """Run MLX model and convert output back to PyTorch."""
        output = self._mlx_forward_raw(sample, timestep, debug_trace=debug_trace)

        # Convert back to PyTorch on MPS
        output_pt = torch.from_numpy(np.array(output)).to(
            dtype=sample.dtype, device=sample.device
        )

        # Return as tuple (matching inner UNet return format)
        return (output_pt,)

    def _mlx_forward_raw(self, sample, timestep, debug_trace=None):
        """Run the cached MLX model and keep its output in MLX."""
        c = self._cached_mlx
        dtype = self._mlx_dtype
        sample_mlx = _pt_to_mx(sample).astype(dtype)
        timestep_mlx = mx.array([timestep.item()]).astype(dtype)
        output = self._mlx_model(
            sample_mlx,
            timestep_mlx,
            c["encoder_hs"],
            num_in_batch=c["num_in_batch"],
            condition_embed_dict=c["condition_embed_dict"],
            dino_hidden_states=c.get("dino_hs"),
            position_voxel_indices=c["position_voxel_indices"],
            ref_scale=c["ref_scale"],
            mva_scale=c["mva_scale"],
            debug_trace=debug_trace,
        )
        mx.eval(output)  # noqa: S307 — this is MLX's graph eval, not Python eval
        return output

    def forward_mlx(self, sample, timestep, debug_trace=None):
        """Run the cached inner model entirely with MLX arrays.

        This is used by the opt-in resident denoise loop after the outer Torch
        call has populated the request-local reference and position caches.
        Unlike ``_mlx_forward``, it does not cross the Torch boundary or wrap
        the result in a Diffusers tuple.
        """
        if self._cached_mlx is None:
            raise RuntimeError("MLX cache is not initialized; run the outer first step first")
        c = self._cached_mlx
        output = self._mlx_model(
            sample,
            timestep,
            c["encoder_hs"],
            num_in_batch=c["num_in_batch"],
            condition_embed_dict=c["condition_embed_dict"],
            dino_hidden_states=c.get("dino_hs"),
            position_voxel_indices=c["position_voxel_indices"],
            ref_scale=c["ref_scale"],
            mva_scale=c["mva_scale"],
            debug_trace=debug_trace,
        )
        mx.eval(output)
        return output

    def reset(self):
        """Reset for new denoising run."""
        self._gen_call_count = 0
        self._cached_mlx = None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tree_map_to_f16(tree):
    """Recursively convert all mx.arrays in a nested dict/list to float16."""
    if isinstance(tree, dict):
        return {k: _tree_map_to_f16(v) for k, v in tree.items()}
    elif isinstance(tree, list):
        return [_tree_map_to_f16(v) for v in tree]
    elif isinstance(tree, mx.array):
        return tree.astype(mx.float16)
    else:
        return tree


def _tree_map_to_f32(tree):
    """Recursively convert all mx.arrays in a nested dict/list to float32."""
    if isinstance(tree, dict):
        return {k: _tree_map_to_f32(v) for k, v in tree.items()}
    elif isinstance(tree, list):
        return [_tree_map_to_f32(v) for v in tree]
    elif isinstance(tree, mx.array):
        return tree.astype(mx.float32)
    else:
        return tree


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────

def _load_inner_unet_weights(model: HunyuanInnerUNet, ckpt_path: str | None) -> str:
    """Select a checked MLX bundle only when explicitly configured."""
    bundle_path = os.environ.get("HUNYUAN_MLX_INNER_UNET_BUNDLE")
    if bundle_path:
        source_sha = os.environ.get("HUNYUAN_MLX_INNER_UNET_SOURCE_SHA256")
        config_sha = os.environ.get("HUNYUAN_MLX_INNER_UNET_CONFIG_SHA256")
        if not source_sha or not config_sha:
            raise ValueError("bundle selection requires source and config SHA256")
        try:
            from .mlx_weight_bundle import load_model_weights
        except ImportError:
            from mlx_weight_bundle import load_model_weights
        load_model_weights(
            model, bundle_path,
            expected_source_sha256=source_sha,
            expected_config_sha256=config_sha,
            expected_dtype="float16",
        )
        return "bundle"
    if ckpt_path is None:
        raise FileNotFoundError("inner UNet checkpoint not found")
    load_weights_from_pytorch(model, ckpt_path)
    return "checkpoint"


def setup_mlx_dispatch(pipeline) -> Optional[MLXDispatchUNet]:
    """Replace pipeline's inner UNet with MLX dispatch wrapper.

    Args:
        pipeline: HunyuanPaintPipeline instance (pipeline.unet is outer UNet)

    Returns:
        MLXDispatchUNet instance, or None if setup fails
    """
    # Find checkpoint path
    ckpt_path = None
    for path in [
        os.environ.get("HUNYUAN_PAINT_UNET_CKPT", ""),
        "/Users/daubert/Github/hunyuan3d-artifacts/Hunyuan3D-2.1-modelscope/"
        "hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin",
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--tencent--Hunyuan3D-2.1/"
            "snapshots/0b94677654c57bb9a6b6845cd7b704ccf551d327/"
            "hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin"
        ),
    ]:
        if os.path.exists(path):
            ckpt_path = path
            break

    if ckpt_path is None and not os.environ.get("HUNYUAN_MLX_INNER_UNET_BUNDLE"):
        print("  [MLX] Checkpoint not found, falling back to PyTorch")
        return None

    print("  [MLX] Building MLX inner UNet...")
    t0 = time.time()
    mlx_model = HunyuanInnerUNet()
    print(f"  [MLX] Model built in {time.time() - t0:.1f}s")

    weight_source = _load_inner_unet_weights(mlx_model, ckpt_path)
    print(f"  [MLX] Inner UNet weights loaded from {weight_source}")

    # Precision: fp16 (default) or fp32 (HUNYUAN_MLX_FP32=1)
    # The GEGLU bug was the real quality issue, not precision. fp16 should work fine.
    use_fp32 = os.environ.get("HUNYUAN_MLX_FP32", "0") == "1"
    if use_fp32:
        print("  [MLX] Converting to float32 (HUNYUAN_MLX_FP32=1)...")
        params = mlx_model.parameters()
        params_f32 = _tree_map_to_f32(params)
        mlx_model.update(params_f32)
    else:
        print("  [MLX] Using fp16 weights (native checkpoint precision)")
    # Force materialization of all weight arrays on Metal
    mx.eval(mlx_model.parameters())  # noqa: S307 — MLX graph eval, not Python eval
    print(f"  [MLX] Model ready")

    # Replace inner UNet
    outer_unet = pipeline.unet
    original_inner = outer_unet.unet
    dispatch = MLXDispatchUNet(original_inner, mlx_model, use_fp32=use_fp32)
    try:
        from .mlx_dino_torch_adapter import install_mlx_dino_projector
    except ImportError:
        from mlx_dino_torch_adapter import install_mlx_dino_projector
    projector_installed = install_mlx_dino_projector(outer_unet, ckpt_path)
    outer_unet.unet = dispatch

    print(f"  [MLX] Inner UNet replaced with MLX dispatch wrapper")
    if projector_installed:
        print("  [MLX] Outer DINO projector replaced with bundle-backed MLX adapter")
    if os.environ.get("HUNYUAN_MLX_FIRST_STEP") == "1":
        print(f"  [MLX] Steps 1-15: MLX Metal (opt-in first-step mode)")
    else:
        print(f"  [MLX] Step 1: PyTorch MPS, Steps 2+: MLX Metal")

    return dispatch

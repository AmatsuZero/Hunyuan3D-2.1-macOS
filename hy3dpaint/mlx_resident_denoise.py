"""Keep paint latents in MLX from the first scheduler update through denoising."""

import os
from pathlib import Path
import mlx.core as mx
import numpy as np
import torch

try:
    from .mlx_cfg import combine_classifier_free
    from .mlx_unipc_scheduler import MLXUniPCScheduler
except ImportError:
    from mlx_cfg import combine_classifier_free
    from mlx_unipc_scheduler import MLXUniPCScheduler


def _as_mlx(tensor, dtype=None):
    if isinstance(tensor, mx.array):
        return tensor if dtype is None else tensor.astype(dtype)
    value = mx.array(tensor.detach().cpu().numpy())
    return value if dtype is None else value.astype(dtype)


def run_mlx_denoise(
    dispatch,
    initial_latents,
    first_noise_pred,
    *,
    timesteps,
    scheduler_config,
    embeds_normal,
    embeds_position,
    guidance_scale,
    camera_azims,
    pbr_views,
    views,
    interrupted=None,
):
    """Run CFG/UniPC in MLX and call the cached inner UNet without step bridges.

    The first outer UNet call, including reference-write and cache production,
    happens in Torch before this function. The returned latent crosses back to
    Torch once so the existing VAE and image postprocessing can consume it.
    """
    if first_noise_pred.shape[0] != initial_latents.shape[0] * 3:
        raise ValueError("first prediction must contain three CFG branches")
    if initial_latents.shape[0] % (pbr_views * views):
        raise ValueError("latent batch does not match PBR materials and views")
    if len(camera_azims) != views:
        raise ValueError("camera azimuth count does not match views")
    if embeds_normal is None or embeds_position is None:
        raise ValueError("normal and position embeddings are required")
    latents = _as_mlx(initial_latents)
    if isinstance(first_noise_pred, mx.array):
        if first_noise_pred.dtype != latents.dtype:
            raise ValueError("first prediction and latent dtype differ")
    elif first_noise_pred.dtype != initial_latents.dtype:
        raise ValueError("first prediction and latent dtype differ")

    schedule = [int(value) for value in timesteps]
    scheduler = MLXUniPCScheduler.from_config(scheduler_config)
    scheduler.set_timesteps(len(schedule))
    if scheduler.timesteps.tolist() != schedule:
        raise ValueError("Torch and MLX timesteps differ")

    trace_dir = os.environ.get("HUNYUAN_MLX_RESIDENT_TRACE_DIR")
    if trace_dir:
        trace_path = Path(trace_dir)
        trace_path.mkdir(parents=True, exist_ok=False)

        def trace(name, value):
            mx.eval(value)
            np.save(trace_path / f"{name}.npy", np.array(value), allow_pickle=False)
    else:
        trace = None

    normal = _as_mlx(embeds_normal, latents.dtype)
    position = _as_mlx(embeds_position, latents.dtype)
    expected_embed_shape = (initial_latents.shape[0] // (pbr_views * views) * 3,
                            views, initial_latents.shape[1], *initial_latents.shape[-2:])
    if normal.shape != expected_embed_shape or position.shape != expected_embed_shape:
        raise ValueError("normal/position embedding shape does not match the latent batch")

    first_prediction = _as_mlx(first_noise_pred)
    guided = combine_classifier_free(first_prediction, guidance_scale, camera_azims, pbr_views)
    latents = scheduler.step(guided, schedule[0], latents)
    mx.eval(latents)
    if trace is not None:
        trace("resident_noise_pred_0", first_prediction)
        trace("resident_scheduler_model_output_0", guided)
        trace("resident_latents_0", latents)

    batch = initial_latents.shape[0] // (pbr_views * views)
    channels, height, width = initial_latents.shape[1:]
    normal = mx.repeat(normal[:, None], pbr_views, axis=1)
    position = mx.repeat(position[:, None], pbr_views, axis=1)
    for index, timestep in enumerate(schedule[1:], start=1):
        if interrupted is not None and interrupted():
            raise RuntimeError("MLX resident loop interrupted")
        branch_latents = mx.concatenate([latents] * 3, axis=0)
        branch_latents = branch_latents.reshape(3 * batch, pbr_views, views,
                                                 channels, height, width)
        sample = mx.concatenate([branch_latents, normal, position], axis=3)
        sample = sample.reshape(3 * batch * pbr_views * views,
                                channels + normal.shape[3] + position.shape[3],
                                height, width)
        prediction = dispatch.forward_mlx(sample, mx.array([timestep]))
        if prediction.shape != first_prediction.shape or prediction.dtype != latents.dtype:
            raise ValueError("MLX inner prediction shape or dtype differs from first step")
        guided = combine_classifier_free(prediction, guidance_scale, camera_azims, pbr_views)
        latents = scheduler.step(guided, timestep, latents)
        mx.eval(latents)
        if trace is not None:
            trace(f"resident_noise_pred_{index}", prediction)
            trace(f"resident_scheduler_model_output_{index}", guided)
            trace(f"resident_latents_{index}", latents)

    return torch.from_numpy(np.array(latents)).to(
        device=initial_latents.device, dtype=initial_latents.dtype
    )

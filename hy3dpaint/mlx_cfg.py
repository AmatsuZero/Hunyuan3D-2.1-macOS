"""Torch-free three-branch classifier-free guidance for paint generation."""

import numpy as np
import mlx.core as mx


def _camera_scale(azim):
    if 0 <= azim < 90:
        return float(azim) / 90.0 + 1.0
    if 90 <= azim < 330:
        return 2.0
    return -float(azim) / 90.0 + 5.0


def combine_classifier_free(noise_pred, guidance_scale, camera_azims, pbr_views):
    """Apply the exact uncond/ref/full guidance formula used by the paint pipeline."""
    if noise_pred.ndim < 1 or noise_pred.shape[0] % 3:
        raise ValueError("noise prediction must contain three CFG branches")
    batch = noise_pred.shape[0] // 3
    if pbr_views < 1 or batch % pbr_views:
        raise ValueError("noise prediction batch is not divisible by pbr_views")
    views = batch // pbr_views
    if len(camera_azims) != views:
        raise ValueError("camera azimuth count does not match generation views")
    scales = np.tile(np.asarray([_camera_scale(value) for value in camera_azims], dtype=np.float32), pbr_views)
    view_scale = mx.array(scales).astype(noise_pred.dtype).reshape(
        batch, *([1] * (noise_pred.ndim - 1))
    )
    uncond, ref, full = mx.split(noise_pred, 3, axis=0)
    result = uncond + guidance_scale * view_scale * (ref - uncond)
    result = result + guidance_scale * view_scale * (full - ref)
    return result.astype(noise_pred.dtype)

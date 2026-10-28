"""The opt-in paint loop keeps the denoising tail in MLX."""

import json
from pathlib import Path
import unittest

from diffusers import UniPCMultistepScheduler
import mlx.core as mx
import numpy as np
import torch


CONFIG = Path(__file__).resolve().parent / "fixtures" / "paint_unipc_config.json"


class _ConstantMLXDispatch:
    def __init__(self):
        self.calls = []

    def forward_mlx(self, sample, timestep):
        self.calls.append((sample, timestep))
        assert isinstance(sample, mx.array)
        assert isinstance(timestep, mx.array)
        assert sample.shape == (6, 12, 2, 2)
        return mx.zeros((6, 4, 2, 2), dtype=sample.dtype)


class MLXResidentDenoiseTests(unittest.TestCase):
    def setUp(self):
        mx.set_default_device(mx.cpu)
        self.config = json.loads(CONFIG.read_text())

    def test_two_steps_match_torch_and_call_inner_with_mlx_arrays(self):
        from hy3dpaint.mlx_resident_denoise import run_mlx_denoise

        torch_scheduler = UniPCMultistepScheduler.from_config(self.config)
        torch_scheduler.set_timesteps(2, device="cpu")
        initial = torch.full((2, 4, 2, 2), 0.2, dtype=torch.float32)
        first_prediction = torch.zeros((6, 4, 2, 2), dtype=torch.float32)
        normal = torch.ones((3, 1, 4, 2, 2), dtype=torch.float16)
        position = torch.ones_like(normal) * 2
        dispatch = _ConstantMLXDispatch()

        actual = run_mlx_denoise(
            dispatch, initial, first_prediction,
            timesteps=torch_scheduler.timesteps,
            scheduler_config=dict(torch_scheduler.config),
            embeds_normal=normal, embeds_position=position,
            guidance_scale=3.0, camera_azims=[0], pbr_views=2, views=1,
        )

        expected = initial
        for timestep in torch_scheduler.timesteps:
            expected = torch_scheduler.step(torch.zeros_like(expected), timestep, expected).prev_sample
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=5e-4, atol=5e-4)
        self.assertEqual(len(dispatch.calls), 1)
        sample, timestep = dispatch.calls[0]
        np.testing.assert_array_equal(np.array(sample[:, 4:8]), 1)
        np.testing.assert_array_equal(np.array(sample[:, 8:12]), 2)
        self.assertEqual(int(np.array(timestep)[0]), int(torch_scheduler.timesteps[1]))

    def test_mlx_first_prediction_stays_in_mlx(self):
        from hy3dpaint.mlx_resident_denoise import run_mlx_denoise

        scheduler = UniPCMultistepScheduler.from_config(self.config)
        scheduler.set_timesteps(1, device="cpu")
        initial = torch.full((2, 4, 2, 2), 0.2, dtype=torch.float16)
        prediction = mx.zeros((6, 4, 2, 2), dtype=mx.float16)
        embeds = torch.ones((3, 1, 4, 2, 2), dtype=torch.float16)
        result = run_mlx_denoise(
            _ConstantMLXDispatch(), initial, prediction,
            timesteps=scheduler.timesteps, scheduler_config=dict(scheduler.config),
            embeds_normal=embeds, embeds_position=embeds,
            guidance_scale=3.0, camera_azims=[0], pbr_views=2, views=1,
        )
        self.assertEqual(result.dtype, initial.dtype)


if __name__ == "__main__":
    unittest.main()

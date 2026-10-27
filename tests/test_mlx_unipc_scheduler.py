"""Match the paint pipeline's UniPC latent updates without Torch at runtime."""

import json
from pathlib import Path
import unittest

import mlx.core as mx
import numpy as np
import torch
from diffusers import UniPCMultistepScheduler


CONFIG = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "paint_unipc_config.json"


class MLXUniPCSchedulerTests(unittest.TestCase):
    def setUp(self):
        mx.set_default_device(mx.cpu)
        self.config = json.loads(CONFIG.read_text())

    def test_fifteen_step_v_prediction_matches_diffusers(self):
        from hy3dpaint.mlx_unipc_scheduler import MLXUniPCScheduler

        reference = UniPCMultistepScheduler.from_config(self.config, timestep_spacing="trailing")
        reference.set_timesteps(15, device="cpu")
        candidate = MLXUniPCScheduler.from_config(self.config)
        candidate.set_timesteps(15)
        np.testing.assert_array_equal(candidate.timesteps, reference.timesteps.numpy())
        np.testing.assert_allclose(candidate.sigmas, reference.sigmas.numpy(), rtol=1e-6, atol=1e-7)

        rng = np.random.default_rng(20261028)
        initial = rng.standard_normal((2, 4, 8, 8)).astype(np.float32)
        noise = [rng.standard_normal(initial.shape).astype(np.float32) for _ in range(15)]
        torch_latent = torch.from_numpy(initial.copy())
        mlx_latent = mx.array(initial)
        for index, (timestep, prediction) in enumerate(zip(reference.timesteps, noise)):
            torch_latent = reference.step(torch.from_numpy(prediction), timestep, torch_latent).prev_sample
            mlx_latent = candidate.step(mx.array(prediction), int(timestep), mlx_latent)
            mx.eval(mlx_latent)
            np.testing.assert_allclose(np.array(mlx_latent), torch_latent.numpy(),
                                       rtol=5e-4, atol=5e-4, err_msg=f"step {index}")
        self.assertEqual(candidate.step_index, 15)

    def test_reset_restarts_history_for_new_input(self):
        from hy3dpaint.mlx_unipc_scheduler import MLXUniPCScheduler

        scheduler = MLXUniPCScheduler.from_config(self.config)
        scheduler.set_timesteps(15)
        input_latent = mx.ones((1, 4, 2, 2), dtype=mx.float32)
        model_output = mx.full((1, 4, 2, 2), 0.2, dtype=mx.float32)
        first = np.array(scheduler.step(model_output, int(scheduler.timesteps[0]), input_latent))
        scheduler.set_timesteps(15)
        second = np.array(scheduler.step(model_output, int(scheduler.timesteps[0]), input_latent))
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()

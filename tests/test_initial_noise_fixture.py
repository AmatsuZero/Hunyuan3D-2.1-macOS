from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch
from diffusers import StableDiffusionPipeline

from tools.quality_demo.audit_mps import install_initial_noise


class InitialNoiseTests(unittest.TestCase):
    def make_pipe(self):
        class Pipe:
            vae_scale_factor = 8
            scheduler = SimpleNamespace(init_noise_sigma=2.0)
            prepare_latents = StableDiffusionPipeline.prepare_latents
        return Pipe()

    def call(self, pipe, seed):
        return pipe.prepare_latents(1, 4, 16, 16, torch.float16, torch.device('cpu'),
                                    torch.Generator().manual_seed(seed))

    def test_replays_exact_fixture_before_scheduler_scaling_and_captures_actual_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            noise = np.arange(16, dtype=np.float16).reshape(1, 4, 2, 2)
            np.save(root / 'noise.npy', noise)
            pipe = self.make_pipe()
            original = install_initial_noise(pipe, root / 'noise.npy', root / 'actual.npy')
            first, second = self.call(pipe, 0), self.call(pipe, 123)
            torch.testing.assert_close(first, torch.from_numpy(noise) * 2, rtol=0, atol=0)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            np.testing.assert_array_equal(np.load(root / 'actual.npy'), first.numpy())
            pipe.prepare_latents = original
            self.assertFalse(torch.equal(self.call(pipe, 0), self.call(pipe, 123)))

    def test_capture_without_fixture_preserves_original_rng_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipe = self.make_pipe()
            expected = self.call(pipe, 7)
            install_initial_noise(pipe, None, Path(tmp) / 'actual.npy')
            torch.testing.assert_close(self.call(pipe, 7), expected, rtol=0, atol=0)

    def test_rejects_wrong_shape_dtype_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for value, error in [(np.zeros((1, 4, 3, 3), dtype=np.float16), 'shape'),
                                 (np.zeros((1, 4, 2, 2), dtype=np.float32), 'dtype'),
                                 (np.full((1, 4, 2, 2), np.nan, dtype=np.float16), 'finite')]:
                with self.subTest(error=error):
                    np.save(root / 'noise.npy', value)
                    pipe = self.make_pipe()
                    install_initial_noise(pipe, root / 'noise.npy', root / 'actual.npy')
                    with self.assertRaisesRegex(ValueError, error):
                        self.call(pipe, 0)

"""The opt-in outer DINO projector adapter preserves the Torch module contract."""

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest import mock
import os

import mlx.core as mx
import numpy as np
import torch


class MLXDinoProjectorAdapterTests(unittest.TestCase):
    def setUp(self):
        mx.set_default_device(mx.cpu)

    def test_projector_adapter_returns_torch_output_with_matching_values(self):
        from hy3dpaint.mlx_dino_projector import MLXDinoProjector
        from hy3dpaint.mlx_dino_torch_adapter import MLXDinoProjectorAdapter

        projector = MLXDinoProjector(input_dim=4, output_dim=2, extra_tokens=2)
        weights = [
            ("proj.weight", mx.array(np.arange(16, dtype=np.float16).reshape(4, 4) / 16)),
            ("proj.bias", mx.zeros((4,), dtype=mx.float16)),
            ("norm.weight", mx.ones((2,), dtype=mx.float16)),
            ("norm.bias", mx.zeros((2,), dtype=mx.float16)),
        ]
        projector.load_weights(weights, strict=True)
        adapter = MLXDinoProjectorAdapter(projector)
        features = torch.arange(8, dtype=torch.float16, device="mps").reshape(1, 2, 4)

        actual = adapter(features)
        expected = np.array(projector(mx.array(features.cpu().numpy())))

        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(tuple(actual.shape), (1, 4, 2))
        self.assertEqual(actual.dtype, features.dtype)
        self.assertEqual(actual.device.type, "mps")
        np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=5e-3, atol=5e-3)

    def test_install_is_opt_in_and_replaces_outer_projector(self):
        from hy3dpaint.mlx_dino_torch_adapter import (
            MLXDinoProjectorAdapter,
            install_mlx_dino_projector,
        )

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "diffusion_pytorch_model.bin"
            config = checkpoint.with_name("config.json")
            bundle = checkpoint.with_name("conditions.safetensors")
            checkpoint.write_bytes(b"checkpoint")
            config.write_text("{}")
            outer = torch.nn.Module()
            original = torch.nn.Linear(4, 4)
            outer.image_proj_model_dino = original
            weights = {
                "image_proj_model_dino.proj.weight": mx.zeros((4096, 1536), dtype=mx.float16),
                "image_proj_model_dino.proj.bias": mx.zeros((4096,), dtype=mx.float16),
                "image_proj_model_dino.norm.weight": mx.ones((1024,), dtype=mx.float16),
                "image_proj_model_dino.norm.bias": mx.zeros((1024,), dtype=mx.float16),
            }
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertFalse(install_mlx_dino_projector(outer, checkpoint))
                self.assertIs(outer.image_proj_model_dino, original)
            with (mock.patch.dict(os.environ, {"HUNYUAN_MLX_DINO_PROJECTOR_BUNDLE": str(bundle)}),
                  mock.patch("hy3dpaint.mlx_condition_bundle.load_bundle", return_value=(weights, {})) as load):
                self.assertTrue(install_mlx_dino_projector(outer, checkpoint))
            self.assertEqual(load.call_args.args[0], bundle)
            self.assertEqual(len(load.call_args.args[1]), 64)
            self.assertEqual(len(load.call_args.args[2]), 64)
            self.assertIsInstance(outer.image_proj_model_dino, MLXDinoProjectorAdapter)
            self.assertEqual(outer.image_proj_model_dino.bundle_path, str(bundle))
            self.assertEqual(outer.image_proj_model_dino.source_sha256, load.call_args.args[1])
            self.assertEqual(outer.image_proj_model_dino.config_sha256, load.call_args.args[2])


if __name__ == "__main__":
    unittest.main()

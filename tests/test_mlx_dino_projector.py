"""The outer paint DINO projection must preserve token order and values."""

import unittest

import mlx.core as mx
import numpy as np
import torch


class MLXDinoProjectorTests(unittest.TestCase):
    def setUp(self):
        mx.set_default_device(mx.cpu)

    def test_two_and_three_dimensional_inputs_match_torch(self):
        from hy3dpaint.mlx_dino_projector import MLXDinoProjector

        torch.manual_seed(20261028)
        linear = torch.nn.Linear(6, 4 * 5).eval()
        norm = torch.nn.LayerNorm(5).eval()
        candidate = MLXDinoProjector(input_dim=6, output_dim=5, extra_tokens=4)
        candidate.load_weights([
            ("proj.weight", mx.array(linear.weight.detach().numpy())),
            ("proj.bias", mx.array(linear.bias.detach().numpy())),
            ("norm.weight", mx.array(norm.weight.detach().numpy())),
            ("norm.bias", mx.array(norm.bias.detach().numpy())),
        ], strict=True)

        for shape in ((2, 6), (2, 3, 6)):
            with self.subTest(shape=shape):
                input_value = torch.randn(shape)
                flat = input_value.reshape(-1, 6)
                projected = norm(linear(flat).reshape(-1, 4, 5))
                expected = projected.reshape(shape[0], -1, 5).detach().numpy()
                actual = np.array(candidate(mx.array(input_value.numpy())))
                self.assertEqual(actual.shape, expected.shape)
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

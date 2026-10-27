"""MLX classifier-free guidance must match the paint pipeline formula."""

import unittest

import mlx.core as mx
import numpy as np
import torch


class MLXCFGTests(unittest.TestCase):
    def setUp(self):
        mx.set_default_device(mx.cpu)

    def test_guidance_matches_torch_three_branch_formula(self):
        from hy3dpaint.mlx_cfg import combine_classifier_free

        torch.manual_seed(20261028)
        prediction = torch.randn(3 * 2 * 3, 4, 2, 2, dtype=torch.float32)
        azims = [0, 90, 180]
        view = torch.tensor([1.0, 2.0, 2.0, 1.0, 2.0, 2.0])[:, None, None, None]
        uncond, ref, full = prediction.chunk(3)
        expected = uncond + 3.0 * view * (ref - uncond)
        expected += 3.0 * view * (full - ref)
        actual = combine_classifier_free(mx.array(prediction.numpy()), 3.0, azims, pbr_views=2)
        mx.eval(actual)
        np.testing.assert_allclose(np.array(actual), expected.numpy(), rtol=1e-6, atol=1e-6)

    def test_rejects_wrong_branch_shape(self):
        from hy3dpaint.mlx_cfg import combine_classifier_free

        with self.assertRaisesRegex(ValueError, "three CFG branches"):
            combine_classifier_free(mx.zeros((5, 4, 2, 2), dtype=mx.float32), 3.0, [0, 90], pbr_views=1)

    def test_half_precision_keeps_dtype_for_nontrivial_camera_scales(self):
        from hy3dpaint.mlx_cfg import combine_classifier_free

        torch.manual_seed(7)
        prediction = torch.randn(12, 4, 2, 2, dtype=torch.float16)
        azims = [45, 350]
        scale = torch.tensor([1.5, 5.0 - 350.0 / 90.0] * 2,
                             dtype=torch.float16)[:, None, None, None]
        uncond, ref, full = prediction.chunk(3)
        expected = uncond + 3.0 * scale * (ref - uncond)
        expected += 3.0 * scale * (full - ref)
        actual = combine_classifier_free(mx.array(prediction.numpy()), 3.0, azims, pbr_views=2)
        self.assertEqual(actual.dtype, mx.float16)
        np.testing.assert_allclose(np.array(actual), expected.numpy(), rtol=3e-3, atol=3e-3)


if __name__ == "__main__":
    unittest.main()

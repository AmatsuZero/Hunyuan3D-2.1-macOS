"""MLX position voxel indices must match the Torch producer exactly."""

import unittest
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np
import torch


class MLXPositionIndicesTests(unittest.TestCase):
    def setUp(self):
        mx.set_default_device(mx.cpu)

    def test_four_resolution_cache_matches_torch_reference(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hy3dpaint"))
        from hy3dpaint.hunyuanpaintpbr.unet.modules import calc_multires_voxel_idxs
        from hy3dpaint.mlx_position_indices import calc_multires_voxel_idxs_mlx

        torch.manual_seed(20261028)
        position = torch.rand((2, 3, 3, 16, 16), dtype=torch.float32).half()
        position[:, :, :, :2, :3] = 1
        grid_resolutions = [8, 4, 2, 1]
        voxel_resolutions = [64, 32, 16, 8]
        expected = calc_multires_voxel_idxs(
            position.clone(), grid_resolutions, voxel_resolutions
        )
        actual = calc_multires_voxel_idxs_mlx(
            mx.array(position.numpy()), grid_resolutions, voxel_resolutions
        )

        self.assertEqual(set(actual), set(expected))
        for sequence, expected_entry in expected.items():
            value = actual[sequence]["voxel_indices"]
            mx.eval(value)
            np.testing.assert_array_equal(
                np.array(value), expected_entry["voxel_indices"].numpy()
            )
            self.assertEqual(actual[sequence]["voxel_resolution"],
                             expected_entry["voxel_resolution"])

    def test_dispatch_keeps_native_mlx_position_cache_and_resets(self):
        from hy3dpaint.mlx_utils import MLXDispatchUNet, make_mlx_position_cache

        first = torch.zeros((1, 3, 3, 16, 16), dtype=torch.float16)
        second = torch.full_like(first, 0.75)
        first_cache = make_mlx_position_cache(first, latent_resolution=8)
        second_cache = make_mlx_position_cache(second, latent_resolution=8)
        dispatch = MLXDispatchUNet(torch.nn.Identity(), mlx_model=None)
        encoder = torch.zeros((1, 77, 1024), dtype=torch.float16)

        dispatch._cache_constant_inputs(encoder, {
            "cross_attention_kwargs": {"position_voxel_indices": first_cache}
        })
        self.assertIs(dispatch._cached_mlx["position_voxel_indices"][192]["voxel_indices"],
                      first_cache[192]["voxel_indices"])
        dispatch.reset()
        dispatch._cache_constant_inputs(encoder, {
            "cross_attention_kwargs": {"position_voxel_indices": second_cache}
        })
        self.assertIs(dispatch._cached_mlx["position_voxel_indices"][192]["voxel_indices"],
                      second_cache[192]["voxel_indices"])
        self.assertFalse(np.array_equal(np.array(first_cache[192]["voxel_indices"]),
                                        np.array(second_cache[192]["voxel_indices"])))


if __name__ == "__main__":
    unittest.main()

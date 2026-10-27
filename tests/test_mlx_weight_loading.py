import unittest

import numpy as np
import torch

from hy3dpaint.mlx_utils import _load_upsample_weights


class _FakeConv:
    weight = None
    bias = None


class _FakeUpsample:
    def __init__(self):
        self.conv = _FakeConv()


class MLXWeightLoadingTests(unittest.TestCase):
    def test_loads_up_block_zero_upsample_weights_with_nhwc_layout(self):
        prefix = "up_blocks.0.upsamplers.0.conv"
        weight = torch.arange(24, dtype=torch.float32).reshape(2, 2, 2, 3)
        bias = torch.tensor([0.25, -0.5], dtype=torch.float32)

        upsample = _FakeUpsample()
        _load_upsample_weights(
            upsample, prefix,
            {f"{prefix}.weight": weight, f"{prefix}.bias": bias},
        )

        np.testing.assert_array_equal(np.array(upsample.conv.weight), weight.permute(0, 2, 3, 1))
        np.testing.assert_array_equal(np.array(upsample.conv.bias), bias)


if __name__ == "__main__":
    unittest.main()

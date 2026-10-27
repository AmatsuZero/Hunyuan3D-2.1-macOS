import unittest

import mlx.core as mx
import numpy as np

from hy3dpaint.mlx_unet import Basic2p5DTransformerBlock


class MLXReferenceAttentionTests(unittest.TestCase):
    def test_concatenated_material_values_preserve_checkpoint_head_layout(self):
        block = Basic2p5DTransformerBlock(dim=4, heads=2, cross_dim=4)
        query = np.array(
            [[[1.0, 0.0, 0.5, -0.5], [0.0, 1.0, -0.5, 0.5], [1.0, 1.0, 0.0, 0.0]]],
            dtype=np.float32,
        )
        key = np.array(
            [[[1.0, 0.0, 0.25, -0.25], [0.0, 1.0, -0.25, 0.25]]],
            dtype=np.float32,
        )
        value_albedo = np.arange(8, dtype=np.float32).reshape(1, 2, 4) / 10
        value_mr = np.arange(8, 16, dtype=np.float32).reshape(1, 2, 4) / 10

        actual = block._sdpa_reference_values(
            mx.array(query), mx.array(key), [mx.array(value_albedo), mx.array(value_mr)], B=1
        )
        mx.eval(*actual)

        q_heads = query.reshape(1, 3, 2, 2).transpose(0, 2, 1, 3)
        k_heads = key.reshape(1, 2, 2, 2).transpose(0, 2, 1, 3)
        joined_values = np.concatenate([value_albedo, value_mr], axis=-1)
        v_heads = joined_values.reshape(1, 2, 2, 4).transpose(0, 2, 1, 3)
        logits = np.matmul(q_heads, k_heads.transpose(0, 1, 3, 2)) / np.sqrt(2.0)
        weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        joined_output = np.matmul(weights, v_heads)
        expected = [
            part.transpose(0, 2, 1, 3).reshape(1, 3, 4)
            for part in np.split(joined_output, 2, axis=-1)
        ]

        for actual_material, expected_material in zip(actual, expected):
            np.testing.assert_allclose(np.array(actual_material), expected_material, rtol=2e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()

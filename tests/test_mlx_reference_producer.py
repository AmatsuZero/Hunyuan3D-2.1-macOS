import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import torch

from hy3dpaint.mlx_reference_cache import MLXReferenceCacheProducer


def _cache(value=1.0):
    return {
        f"layer_{index}": torch.full((1, 4 + index, 8), value + index)
        for index in range(16)
    }


class MLXReferenceCacheProducerTests(unittest.TestCase):
    def test_produces_eager_mlx_arrays_and_preserves_values(self):
        producer = MLXReferenceCacheProducer()
        result = producer.produce(_cache())
        self.assertEqual(len(result), 16)
        self.assertIs(producer.cache, result)
        mx.eval(list(result.values()))
        np.testing.assert_array_equal(np.array(result["layer_7"]), np.full((1, 11, 8), 8.0))

    def test_rejects_wrong_layer_count_shape_and_nonfinite_values(self):
        producer = MLXReferenceCacheProducer()
        with self.assertRaisesRegex(ValueError, "16 layers"):
            producer.produce({"only": torch.zeros(1, 1, 1)})
        bad_shape = _cache()
        bad_shape["layer_0"] = torch.zeros(2, 3)
        with self.assertRaisesRegex(ValueError, "rank-3"):
            producer.produce(bad_shape)
        bad_value = _cache()
        bad_value["layer_0"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            producer.produce(bad_value)

    def test_reset_clears_request_cache(self):
        producer = MLXReferenceCacheProducer()
        producer.produce(_cache())
        producer.reset()
        self.assertIsNone(producer.cache)

    def test_dispatch_opt_in_preserves_cache_and_resets_between_requests(self):
        from hy3dpaint.mlx_utils import MLXDispatchUNet

        dispatch = MLXDispatchUNet(torch.nn.Identity(), mlx_model=None)
        encoder = torch.zeros((1, 77, 1024), dtype=torch.float16)
        with patch.dict("os.environ", {"HUNYUAN_MLX_REFERENCE_CACHE": "1"}):
            dispatch._cache_constant_inputs(encoder, {
                "cross_attention_kwargs": {"condition_embed_dict": _cache(1.0)}
            })
            first = dispatch._cached_mlx["condition_embed_dict"]
            self.assertIs(first, dispatch._reference_cache_producer.cache)
            self.assertEqual(dispatch._cached_mlx["reference_cache_producer"],
                             "mlx-boundary")
            dispatch.reset()
            self.assertIsNone(dispatch._reference_cache_producer.cache)
            self.assertIsNone(dispatch._cached_mlx)
            dispatch._cache_constant_inputs(encoder, {
                "cross_attention_kwargs": {"condition_embed_dict": _cache(2.0)}
            })
            second = dispatch._cached_mlx["condition_embed_dict"]
            self.assertIsNot(first, second)
            self.assertEqual(float(np.array(second["layer_0"])[0, 0, 0]), 2.0)


if __name__ == "__main__":
    unittest.main()

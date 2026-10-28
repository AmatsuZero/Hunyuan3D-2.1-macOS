"""Opt-in MLX dual reference forward keeps request-local cache ownership."""

import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import torch

from hy3dpaint.mlx_utils import MLXDispatchUNet, _MLXReferenceUNetWrapper


class _ReferenceModel:
    def __call__(self, sample, timestep, encoder_hidden_states, **kwargs):
        cache = kwargs["reference_cache"]
        cache["layer"] = sample[:, :1].transpose(0, 2, 3, 1).reshape(1, 4, 1)
        return sample


class MLXReferenceForwardTests(unittest.TestCase):
    def setUp(self):
        mx.set_default_device(mx.cpu)

    def test_wrapper_writes_torch_contract_and_keeps_native_mlx_cache(self):
        dispatch = MLXDispatchUNet(torch.nn.Identity(), mlx_model=None)
        dispatch.set_reference_model(_ReferenceModel())
        wrapper = _MLXReferenceUNetWrapper(torch.nn.Identity(), dispatch)
        sample = torch.arange(4, dtype=torch.float16).reshape(1, 1, 2, 2)
        encoder = torch.zeros((1, 77, 1024), dtype=torch.float16)
        written = {}
        with patch.dict("os.environ", {"HUNYUAN_MLX_REFERENCE_FORWARD": "1"}):
            output = wrapper(
                sample, 0, encoder_hidden_states=encoder,
                cross_attention_kwargs={
                    "mode": "w", "num_in_batch": 1,
                    "condition_embed_dict": written,
                },
                return_dict=False,
            )
        torch.testing.assert_close(output[0], sample, rtol=0, atol=0)
        np.testing.assert_array_equal(written["layer"].numpy(),
                                      np.arange(4, dtype=np.float16).reshape(1, 4, 1))
        native = dispatch._pending_reference_cache
        self.assertIsInstance(native["layer"], mx.array)
        dispatch.reset()
        self.assertIsNone(dispatch._pending_reference_cache)

    def test_rejects_missing_reference_cache(self):
        dispatch = MLXDispatchUNet(torch.nn.Identity(), mlx_model=None)
        dispatch.set_reference_model(_ReferenceModel())
        wrapper = _MLXReferenceUNetWrapper(torch.nn.Identity(), dispatch)
        with patch.dict("os.environ", {"HUNYUAN_MLX_REFERENCE_FORWARD": "1"}):
            with self.assertRaisesRegex(ValueError, "condition_embed_dict"):
                wrapper(torch.zeros(1, 1, 2, 2), 0,
                        encoder_hidden_states=torch.zeros(1, 77, 1024),
                        cross_attention_kwargs={"mode": "w"})


if __name__ == "__main__":
    unittest.main()

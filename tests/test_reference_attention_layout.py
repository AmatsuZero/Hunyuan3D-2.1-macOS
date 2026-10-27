import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from diffusers.models.attention_processor import Attention


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hy3dpaint"))
from hunyuanpaintpbr.unet.attn_processor import RefAttnProcessor2_0  # noqa: E402


class ReferenceAttentionLayoutTests(unittest.TestCase):
    def test_local_processor_preserves_checkpoint_material_head_order(self):
        processor = RefAttnProcessor2_0(query_dim=4, heads=2, dim_head=2,
                                          pbr_setting=["albedo", "mr"])
        attention = Attention(query_dim=4, heads=2, dim_head=2, processor=processor)
        with torch.no_grad():
            for projection in (attention.to_q, attention.to_k, attention.to_v,
                               processor.to_v_mr, attention.to_out[0], processor.to_out_mr[0]):
                projection.weight.copy_(torch.eye(4))
                if projection.bias is not None:
                    projection.bias.zero_()

        query = torch.tensor([[[1.0, 0.0, 0.5, -0.5], [0.0, 1.0, -0.5, 0.5]]])
        context = torch.tensor([[[1.0, 0.0, 0.25, -0.25], [0.0, 1.0, -0.25, 0.25]]])
        with torch.no_grad():
            actual = attention(query, encoder_hidden_states=context)

        q = query.numpy().reshape(1, 2, 2, 2).transpose(0, 2, 1, 3)
        k = context.numpy().reshape(1, 2, 2, 2).transpose(0, 2, 1, 3)
        joined = np.concatenate([context.numpy(), context.numpy()], axis=-1)
        v = joined.reshape(1, 2, 2, 4).transpose(0, 2, 1, 3)
        logits = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(2.0)
        weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        expected_joined = np.matmul(weights, v)
        expected = np.stack([
            part.transpose(0, 2, 1, 3).reshape(1, 2, 4)
            for part in np.split(expected_joined, 2, axis=-1)
        ], axis=1)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

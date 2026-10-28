"""Reference Transformer2D normalization is isolated from generation."""
import unittest

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F

from hy3dpaint.mlx_unet import HunyuanInnerUNet, Transformer2D


def transformer_norms(model):
    return ([attn.norm for block in (model.down_block_0, model.down_block_1,
                                    model.down_block_2) for attn in block.attentions]
            + [model.mid_attn.norm]
            + [attn.norm for block in (model.up_block_1, model.up_block_2,
                                       model.up_block_3) for attn in block.attentions])


class ReferenceNormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.set_default_device(mx.cpu)

    def test_reference_epsilon_matches_torch_on_sensitive_input(self):
        layer = Transformer2D(4, 1, 4, [], norm_groups=1, norm_eps=1e-6)
        values = np.array([[[[0.0, 0.001, -0.001, 0.002],
                              [0.001, 0.002, -0.002, 0.0]]]], dtype=np.float32)
        layer.norm.weight = mx.ones((4,), dtype=mx.float32)
        layer.norm.bias = mx.zeros((4,), dtype=mx.float32)
        actual = np.array(layer.norm(mx.array(values)))
        expected = F.group_norm(torch.from_numpy(values).permute(0, 3, 1, 2),
                                1, eps=1e-6).permute(0, 2, 3, 1).numpy()
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
        default = Transformer2D(4, 1, 4, [], norm_groups=1)
        self.assertGreater(np.max(np.abs(np.array(default.norm(mx.array(values))) - expected)), 0.1)

    def test_reference_configuration_covers_all_layers_without_changing_generation(self):
        reference = HunyuanInnerUNet(input_channels=4, transformer_norm_eps=1e-6)
        reference_norms = transformer_norms(reference)
        self.assertEqual(len(reference_norms), 16)
        self.assertTrue(all(norm.eps == 1e-6 for norm in reference_norms))
        self.assertEqual(reference.mid_resnet_0.norm1.eps, 1e-5)
        self.assertEqual(reference.conv_norm_out.eps, 1e-5)
        del reference
        generation = HunyuanInnerUNet()
        self.assertTrue(all(norm.eps == 1e-5 for norm in transformer_norms(generation)))


if __name__ == '__main__':
    unittest.main()

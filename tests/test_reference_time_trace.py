"""Opt-in time probes preserve output and expose both sides of SiLU."""
import unittest

import mlx.core as mx
import numpy as np

from hy3dpaint.mlx_unet import ResnetBlock2D, TimestepEmbedding


class TimeTraceTests(unittest.TestCase):
    def test_time_and_resnet_probes_preserve_output(self):
        mx.set_default_device(mx.cpu)
        embedding = TimestepEmbedding(4, 8)
        resnet = ResnetBlock2D(4, 4, 8, groups=1)
        time = mx.array([[0.2, -0.4, 1.0, 0.0]])
        spatial = mx.ones((1, 2, 2, 4))
        temb = embedding(time)
        expected = np.array(resnet(spatial, temb))
        captured = {}

        def trace(name, value):
            self.assertNotIn(name, captured)
            captured[name] = np.array(value)

        embedding._reference_debug_trace = trace
        resnet._reference_debug_trace = trace
        actual = np.array(resnet(spatial, embedding(time)))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(captured["time_embedding.linear_1.input"], np.array(time))
        np.testing.assert_array_equal(captured["time_embedding.linear_2"], np.array(temb))
        np.testing.assert_array_equal(captured["down_0_resnet_0.temb"], np.array(temb))
        for before, after in (("time_embedding.linear_1", "time_embedding.linear_2.input"),
                              ("down_0_resnet_0.temb", "down_0_resnet_0.time_emb_proj.input")):
            x = captured[before]
            np.testing.assert_allclose(captured[after], x / (1 + np.exp(-x)), rtol=2e-6, atol=1e-7)
        self.assertEqual(captured["down_0_resnet_0.time_emb_proj"].shape, (1, 4))


if __name__ == "__main__":
    unittest.main()

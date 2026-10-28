"""Opt-in MLX convolution dispatch policy tests."""

import os
import unittest
from unittest import mock


class MLXConvPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from hy3dpaint import mlx_utils
        cls.mlx_utils = mlx_utils

    def test_disable_winograd_sets_minimal_working_set(self):
        with mock.patch.dict(os.environ, {"HUNYUAN_MLX_DISABLE_WINOGRAD": "1"}, clear=True):
            policy = self.mlx_utils._configure_mlx_conv_policy()

        self.assertTrue(policy["disable_winograd"])
        self.assertEqual(policy["working_set"], "1")

    def test_conflicting_working_set_is_rejected(self):
        with mock.patch.dict(os.environ, {
            "HUNYUAN_MLX_DISABLE_WINOGRAD": "1",
            "MLX_CONV_WINOGRAD_WORKING_SET": "4096",
        }, clear=True):
            with self.assertRaisesRegex(ValueError, "MLX_CONV_WINOGRAD_WORKING_SET"):
                self.mlx_utils._configure_mlx_conv_policy()

    def test_default_policy_does_not_change_environment(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            policy = self.mlx_utils._configure_mlx_conv_policy()

        self.assertFalse(policy["disable_winograd"])
        self.assertIsNone(policy["working_set"])
        self.assertNotIn("MLX_CONV_WINOGRAD_WORKING_SET", os.environ)


if __name__ == "__main__":
    unittest.main()

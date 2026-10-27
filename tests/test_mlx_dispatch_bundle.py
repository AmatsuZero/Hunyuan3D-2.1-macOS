"""R3 contract tests for selecting the Torch-free inner-UNet bundle."""

import os
from unittest import mock
import unittest


class MLXDispatchBundleTests(unittest.TestCase):
    def test_bundle_environment_selects_strict_runtime_loader(self):
        from hy3dpaint import mlx_utils

        model = object()
        with (
            mock.patch.dict(os.environ, {
                "HUNYUAN_MLX_INNER_UNET_BUNDLE": "/tmp/inner_unet.safetensors",
                "HUNYUAN_MLX_INNER_UNET_SOURCE_SHA256": "a" * 64,
                "HUNYUAN_MLX_INNER_UNET_CONFIG_SHA256": "b" * 64,
            }, clear=False),
            mock.patch.object(mlx_utils, "load_weights_from_pytorch") as from_torch,
            mock.patch("hy3dpaint.mlx_weight_bundle.load_model_weights", return_value=model) as from_bundle,
        ):
            source = mlx_utils._load_inner_unet_weights(model, "/tmp/fallback.bin")

        self.assertEqual(source, "bundle")
        from_bundle.assert_called_once_with(
            model, "/tmp/inner_unet.safetensors",
            expected_source_sha256="a" * 64,
            expected_config_sha256="b" * 64,
            expected_dtype="float16",
        )
        from_torch.assert_not_called()

    def test_default_selection_keeps_offline_torch_loader(self):
        from hy3dpaint import mlx_utils

        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
                mlx_utils, "load_weights_from_pytorch") as from_torch:
            self.assertEqual(mlx_utils._load_inner_unet_weights(object(), "/tmp/fallback.bin"), "checkpoint")
        from_torch.assert_called_once()


if __name__ == "__main__":
    unittest.main()

"""R2 contract tests for the Torch-free RealESRGAN/RRDBNet path."""

import importlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import mlx.core as mx
import numpy as np
import torch


class MLXRealESRGANTests(unittest.TestCase):
    def module(self):
        return importlib.import_module("hy3dpaint.mlx_realesrgan")

    def test_torch_checkpoint_mapping_preserves_conv_layout_and_output(self):
        """A mapped RRDBNet must match the reference on a hand-sized fixture."""
        module = self.module()
        from hy3dpaint.utils.torchvision_fix import apply_fix
        apply_fix()
        from basicsr.archs.rrdbnet_arch import RRDBNet

        torch.manual_seed(7)
        reference = RRDBNet(3, 3, scale=4, num_feat=4, num_block=1, num_grow_ch=2).eval()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "rrdb.pth"
            torch.save({"params_ema": reference.state_dict()}, checkpoint)
            converted = module.MLXRRDBNet(num_feat=4, num_block=1, num_grow_ch=2)
            module.load_weights_from_pytorch(converted, checkpoint, dtype=mx.float32)
            fixture = np.arange(3 * 5 * 7, dtype=np.float32).reshape(1, 5, 7, 3) / 100.0
            expected = reference(torch.from_numpy(fixture).permute(0, 3, 1, 2)).detach().numpy()
            actual = np.array(converted(mx.array(fixture))).transpose(0, 3, 1, 2)
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)

    def test_enhance_handles_odd_rgb_dimensions_and_returns_uint8_x4(self):
        module = self.module()
        model = module.MLXRRDBNet(num_feat=4, num_block=1, num_grow_ch=2)
        image = np.zeros((5, 7, 3), dtype=np.uint8)
        image[0, 0] = [255, 17, 3]
        output = module.enhance_rgb(model, image)
        self.assertEqual(output.shape, (20, 28, 3))
        self.assertEqual(output.dtype, np.uint8)
        self.assertTrue(np.isfinite(output).all())

    def test_runtime_module_import_does_not_load_torch(self):
        script = """
import sys
from hy3dpaint.mlx_realesrgan import MLXRRDBNet
assert 'torch' not in sys.modules
MLXRRDBNet(num_feat=4, num_block=1, num_grow_ch=2)
assert 'torch' not in sys.modules
print('mlx_realesrgan_torch_free')
"""
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("mlx_realesrgan_torch_free", completed.stdout)

    def test_bundle_round_trip_checks_source_and_dtype(self):
        module = self.module()
        model = module.MLXRRDBNet(num_feat=4, num_block=1, num_grow_ch=2)
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "rrdb.safetensors"
            module.save_bundle(model, bundle, source_sha256="a" * 64)
            restored = module.MLXRRDBNet(num_feat=4, num_block=1, num_grow_ch=2)
            module.load_bundle(restored, bundle, expected_source_sha256="a" * 64)
            with self.assertRaisesRegex(ValueError, "source_sha256"):
                module.load_bundle(restored, bundle, expected_source_sha256="b" * 64)

    def test_bundle_rejects_tensor_dtype_disagreeing_with_metadata(self):
        module = self.module()
        model = module.MLXRRDBNet(num_feat=4, num_block=1, num_grow_ch=2)
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "wrong.safetensors"
            parameters = dict(module._flatten_tree(model.parameters()))
            mx.save_safetensors(str(bundle), parameters, metadata={
                "schema": module.BUNDLE_SCHEMA,
                "source_sha256": "a" * 64,
                "mlx_version": "0.32.4",
                "dtype": "float16",
                "parameter_count": str(len(parameters)),
                "layout": module.LAYOUT,
            })
            with self.assertRaisesRegex(ValueError, "tensor dtype"):
                module.load_bundle(model, bundle)

    def test_offline_converter_records_source_and_bundle_hashes(self):
        from hy3dpaint.utils.torchvision_fix import apply_fix
        apply_fix()
        from tools.quality_demo.convert_mlx_realesrgan import convert_checkpoint
        from basicsr.archs.rrdbnet_arch import RRDBNet

        torch.manual_seed(3)
        reference = RRDBNet(3, 3, scale=4, num_feat=4, num_block=1, num_grow_ch=2).eval()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint = directory / "rrdb.pth"
            torch.save({"params_ema": reference.state_dict()}, checkpoint)
            bundle = directory / "rrdb.safetensors"
            manifest = directory / "manifest.json"
            report = convert_checkpoint(checkpoint, bundle, manifest, num_feat=4,
                                        num_block=1, num_grow_ch=2, dtype="float32")
            self.assertEqual(report["source_sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertEqual(report["bundle_sha256"],
                             hashlib.sha256(bundle.read_bytes()).hexdigest())
            self.assertEqual(json.loads(manifest.read_text()), report)


if __name__ == "__main__":
    unittest.main()

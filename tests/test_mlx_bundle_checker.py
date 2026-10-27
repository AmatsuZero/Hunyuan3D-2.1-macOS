"""The independent checker must reject a changed bundle or manifest."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import mlx.core as mx

from tools.quality_demo.check_mlx_weight_bundle import verify_manifest


class MLXBundleCheckerTests(unittest.TestCase):
    def test_manifest_accepts_exact_bundle_and_parameter_names(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "tiny.safetensors"
            mx.save_safetensors(str(bundle), {"weight": mx.ones((2, 2))})
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({
                "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                "parameter_names": ["weight"],
                "parameter_count": "1",
            }))
            self.assertEqual(verify_manifest(bundle, manifest)["bundle_sha256"],
                             hashlib.sha256(bundle.read_bytes()).hexdigest())

    def test_manifest_rejects_changed_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "tiny.safetensors"
            mx.save_safetensors(str(bundle), {"weight": mx.ones((2, 2))})
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({
                "bundle_sha256": "0" * 64,
                "parameter_names": ["weight"],
                "parameter_count": "1",
            }))
            with self.assertRaisesRegex(ValueError, "bundle_sha256"):
                verify_manifest(bundle, manifest)

    def test_manifest_rejects_changed_parameter_list(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "tiny.safetensors"
            mx.save_safetensors(str(bundle), {"weight": mx.ones((2, 2))})
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({
                "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                "parameter_names": ["bias"],
                "parameter_count": "1",
            }))
            with self.assertRaisesRegex(ValueError, "parameter_names"):
                verify_manifest(bundle, manifest)


if __name__ == "__main__":
    unittest.main()

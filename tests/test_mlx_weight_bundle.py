"""The MLX bundle must load independently of the Torch conversion environment."""

import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import mlx.core as mx
import mlx.nn as nn


SOURCE_SHA = "a" * 64


def metadata(**changes):
    value = {
        "schema": "hy3d-inner-unet-mlx-v1",
        "source_sha256": SOURCE_SHA,
        "config_sha256": "c" * 64,
        "mlx_version": "0.32.4",
        "dtype": "float16",
        "parameter_count": "2",
        "layout": "conv2d=OHWI;linear=OI;norm=channels",
    }
    value.update(changes)
    return value


class MLXWeightBundleTests(unittest.TestCase):
    def bundle_module(self):
        self.assertIsNotNone(importlib.util.find_spec("hy3dpaint.mlx_weight_bundle"))
        return importlib.import_module("hy3dpaint.mlx_weight_bundle")

    def make_bundle(self, directory, weights=None, bundle_metadata=None):
        path = Path(directory) / "weights.safetensors"
        mx.save_safetensors(
            str(path),
            weights or {
                "weight": mx.ones((2, 2), dtype=mx.float16),
                "bias": mx.zeros((2,), dtype=mx.float16),
            },
            metadata=bundle_metadata or metadata(),
        )
        return path

    def test_loads_exact_weights_without_importing_torch(self):
        self.bundle_module()
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_bundle(directory)
            script = """
import sys
import mlx.nn as nn
from hy3dpaint.mlx_weight_bundle import load_model_weights
model = nn.Linear(2, 2)
load_model_weights(model, sys.argv[1], expected_source_sha256='a' * 64, expected_dtype='float16')
assert 'torch' not in sys.modules
print('torch_free_load_passed')
"""
            completed = subprocess.run(
                [sys.executable, "-c", script, str(path)],
                check=False, capture_output=True, text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("torch_free_load_passed", completed.stdout)

    def test_rejects_wrong_source_and_dtype_metadata(self):
        bundle = self.bundle_module()
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_bundle(directory)
            with self.assertRaisesRegex(ValueError, "source_sha256"):
                bundle.load_bundle(path, expected_source_sha256="b" * 64)
            with self.assertRaisesRegex(ValueError, "dtype"):
                bundle.load_bundle(path, expected_dtype="float32")
            with self.assertRaisesRegex(ValueError, "config_sha256"):
                bundle.load_bundle(path, expected_config_sha256="d" * 64)

    def test_rejects_wrong_schema_and_array_dtype(self):
        bundle = self.bundle_module()
        with tempfile.TemporaryDirectory() as directory:
            wrong_schema = self.make_bundle(directory, bundle_metadata=metadata(schema="unrelated"))
            with self.assertRaisesRegex(ValueError, "schema"):
                bundle.load_bundle(wrong_schema)
            wrong_dtype = self.make_bundle(
                directory,
                weights={"weight": mx.ones((2, 2), dtype=mx.float32),
                         "bias": mx.zeros((2,), dtype=mx.float16)},
            )
            with self.assertRaisesRegex(ValueError, "weight.*dtype"):
                bundle.load_bundle(wrong_dtype)

    def test_rejects_missing_and_extra_parameters(self):
        bundle = self.bundle_module()
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_bundle(
                directory,
                weights={"weight": mx.ones((2, 2), dtype=mx.float16)},
                bundle_metadata=metadata(parameter_count="1"),
            )
            with self.assertRaisesRegex(ValueError, "Missing 1 parameters"):
                bundle.load_model_weights(nn.Linear(2, 2), path)
            path = self.make_bundle(
                directory,
                weights={"weight": mx.ones((2, 2), dtype=mx.float16),
                         "bias": mx.zeros((2,), dtype=mx.float16),
                         "extra": mx.zeros((1,), dtype=mx.float16)},
                bundle_metadata=metadata(parameter_count="3"),
            )
            with self.assertRaisesRegex(ValueError, "parameters not in model"):
                bundle.load_model_weights(nn.Linear(2, 2), path)


if __name__ == "__main__":
    unittest.main()

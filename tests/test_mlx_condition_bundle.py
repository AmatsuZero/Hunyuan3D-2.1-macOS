"""Outer paint condition weights load without Torch and fail closed."""

import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np


class MLXConditionBundleTests(unittest.TestCase):
    def setUp(self):
        mx.set_default_device(mx.cpu)

    def test_round_trip_and_metadata_validation(self):
        from hy3dpaint.mlx_condition_bundle import (
            CONDITION_SHAPES,
            CONDITION_BUNDLE_SCHEMA,
            load_bundle,
            save_bundle,
        )

        arrays = {name: mx.zeros(shape, dtype=mx.float16)
                  for name, shape in CONDITION_SHAPES.items()}
        arrays["learned_text_clip_ref"] = mx.ones((77, 1024), dtype=mx.float16)
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "conditions.safetensors"
            manifest = Path(directory) / "manifest.json"
            save_bundle(arrays, "a" * 64, "b" * 64, bundle, manifest)
            loaded, metadata = load_bundle(bundle, "a" * 64, "b" * 64)
            self.assertEqual(metadata["schema"], CONDITION_BUNDLE_SCHEMA)
            self.assertEqual(set(loaded), set(arrays))
            np.testing.assert_array_equal(np.array(loaded["learned_text_clip_ref"]),
                                          np.ones((77, 1024), dtype=np.float16))
            with self.assertRaisesRegex(ValueError, "source_sha256 mismatch"):
                load_bundle(bundle, "c" * 64, "b" * 64)

    def test_rejects_incomplete_weight_set(self):
        from hy3dpaint.mlx_condition_bundle import CONDITION_SHAPES, save_bundle

        arrays = {name: mx.zeros(shape, dtype=mx.float16)
                  for name, shape in CONDITION_SHAPES.items()
                  if name != "learned_text_clip_ref"}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "condition keys"):
                save_bundle(arrays, "a" * 64, "b" * 64,
                            Path(directory) / "conditions.safetensors",
                            Path(directory) / "manifest.json")

    def test_save_refuses_overwrite(self):
        from hy3dpaint.mlx_condition_bundle import CONDITION_SHAPES, save_bundle

        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "conditions.safetensors"
            manifest = Path(directory) / "manifest.json"
            bundle.write_bytes(b"protected")
            with self.assertRaises(FileExistsError):
                save_bundle({name: mx.zeros(shape, dtype=mx.float16)
                             for name, shape in CONDITION_SHAPES.items()},
                            "a" * 64, "b" * 64, bundle, manifest)


if __name__ == "__main__":
    unittest.main()

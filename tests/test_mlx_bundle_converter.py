"""Offline conversion must account for each inner-UNet source weight."""

import importlib
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
import mlx.nn as nn


class MLXBundleConverterTests(unittest.TestCase):
    def converter(self):
        self.assertIsNotNone(importlib.util.find_spec("tools.quality_demo.convert_mlx_weight_bundle"))
        return importlib.import_module("tools.quality_demo.convert_mlx_weight_bundle")

    def test_rejects_unmapped_inner_unet_weight(self):
        converter = self.converter()
        source = {"unet.a.weight", "unet.b.weight", "unet_dual.c.weight",
                  "unet.learned_text_clip_albedo"}
        with self.assertRaisesRegex(ValueError, "unmapped.*b.weight"):
            converter.audit_source_coverage(source, {"a.weight"})

    def test_allows_only_documented_non_model_source_keys(self):
        converter = self.converter()
        source = {"unet.a.weight", "unet_dual.c.weight",
                  "unet.learned_text_clip_albedo", "unet.learned_text_clip_mr",
                  "unet.learned_text_clip_ref",
                  "unet.image_proj_model_dino.norm.weight",
                  "unet.image_proj_model_dino.norm.bias",
                  "unet.image_proj_model_dino.proj.weight",
                  "unet.image_proj_model_dino.proj.bias"}
        report = converter.audit_source_coverage(source, {"a.weight"})
        self.assertEqual(report["mapped_inner_keys"], 1)
        self.assertEqual(report["excluded_inner_keys"], [
            "image_proj_model_dino.norm.bias", "image_proj_model_dino.norm.weight",
            "image_proj_model_dino.proj.bias", "image_proj_model_dino.proj.weight",
            "learned_text_clip_albedo", "learned_text_clip_mr", "learned_text_clip_ref"
        ])

    def test_bundle_writer_preserves_exact_small_mlx_model(self):
        converter = self.converter()
        with tempfile.TemporaryDirectory() as directory:
            model = nn.Linear(2, 2)
            model.weight = mx.array([[1, 2], [3, 4]], dtype=mx.float16)
            model.bias = mx.array([5, 6], dtype=mx.float16)
            source = Path(directory) / "source.bin"
            source.write_bytes(b"small trusted checkpoint")
            config = Path(directory) / "config.json"
            config.write_text("{}")
            bundle = Path(directory) / "weights.safetensors"
            manifest = Path(directory) / "manifest.json"
            converter.write_bundle(model, source, config, bundle, manifest,
                                   {"unet.weight", "unet.bias"}, {"weight", "bias"})
            self.assertTrue(bundle.is_file())
            self.assertTrue(manifest.is_file())
            self.assertEqual(json.loads(manifest.read_text())["bundle_sha256"],
                             hashlib.sha256(bundle.read_bytes()).hexdigest())
            loaded = nn.Linear(2, 2)
            from hy3dpaint.mlx_weight_bundle import load_model_weights
            load_model_weights(loaded, bundle, expected_dtype="float16")
            mx.eval(loaded.parameters())
            self.assertEqual(list(loaded.weight[0]), [1, 2])
            self.assertEqual(list(loaded.bias), [5, 6])

    def test_failed_bundle_write_leaves_no_manifest(self):
        converter = self.converter()
        with tempfile.TemporaryDirectory() as directory:
            model = nn.Linear(2, 2)
            bundle = Path(directory) / "weights.safetensors"
            manifest = Path(directory) / "manifest.json"
            with self.assertRaises(ValueError):
                converter.write_bundle(model, Path(directory) / "missing.bin",
                                       Path(directory) / "missing.json", bundle, manifest,
                                       set(), set())
            self.assertFalse(manifest.exists())
            self.assertFalse(bundle.exists())

    def test_existing_bundle_is_not_overwritten(self):
        converter = self.converter()
        with tempfile.TemporaryDirectory() as directory:
            model = nn.Linear(2, 2)
            source = Path(directory) / "source.bin"
            source.write_bytes(b"small trusted checkpoint")
            config = Path(directory) / "config.json"
            config.write_text("{}")
            bundle = Path(directory) / "weights.safetensors"
            bundle.write_bytes(b"protected bundle")
            manifest = Path(directory) / "manifest.json"
            with self.assertRaises(FileExistsError):
                converter.write_bundle(model, source, config, bundle, manifest,
                                       {"unet.weight", "unet.bias"}, {"weight", "bias"})
            self.assertEqual(bundle.read_bytes(), b"protected bundle")
            self.assertFalse(manifest.exists())


if __name__ == "__main__":
    unittest.main()

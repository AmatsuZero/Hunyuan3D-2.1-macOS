import numpy as np
import tempfile
import unittest
from pathlib import Path

from tools.quality_demo.probe_fixed_conditions import _load_reference_cache


def _write_cache(root, count=16):
    for index in range(count):
        np.save(root / f"condition_embed_layer_{index}.npy",
                np.zeros((3, 2, 4), dtype=np.float16))


class ReferenceCacheLoadTests(unittest.TestCase):
    def test_requires_exactly_sixteen_rank_three_layers(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            _write_cache(path)
            cache = _load_reference_cache(path)
            self.assertEqual(set(cache), {f"layer_{index}" for index in range(16)})
            self.assertEqual(cache["layer_0"].shape, (3, 2, 4))

    def test_rejects_incomplete_directory(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            _write_cache(path, count=15)
            with self.assertRaisesRegex(ValueError, "16"):
                _load_reference_cache(path)

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paint_entry", ROOT / "run_paint.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class PaintIsolationTests(unittest.TestCase):
    def test_dependency_root_follows_entry_checkout(self):
        self.assertEqual(Path(entry.PAINT_ROOT), ROOT / "hy3dpaint")

    def test_super_resolution_checkpoint_follows_entry_checkout(self):
        self.assertEqual(Path(entry.REALESRGAN_CKPT), ROOT / "ckpt/RealESRGAN_x4plus.pth")

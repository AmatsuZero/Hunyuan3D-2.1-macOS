import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/quality_demo/audit_mps.py"


class AuditEntryTests(unittest.TestCase):
    def test_rejects_negative_cache_limit_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "new-output"
            process = subprocess.run(
                [sys.executable, str(SCRIPT), "--mesh", "unused.glb", "--output-dir", str(out),
                 "--mlx-cache-limit-mib", "-1"],
                env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "0"}, capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("non-negative", process.stderr)
            self.assertFalse(out.exists())

    def test_rejects_enabled_fallback_before_import_or_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "new-output"
            process = subprocess.run(
                [sys.executable, str(SCRIPT), "--mesh", "unused.glb", "--output-dir", str(out)],
                env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}, capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("before import", process.stderr)
            self.assertFalse(out.exists())

    def test_refuses_existing_directory_without_modifying_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh = Path(tmp) / "input.glb"
            mesh.write_bytes(b"protected fixture")
            out = Path(tmp) / "output"
            out.mkdir()
            marker = out / "keep.txt"
            marker.write_text("keep")
            process = subprocess.run(
                [sys.executable, str(SCRIPT), "--mesh", str(mesh), "--output-dir", str(out)],
                env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "0"}, capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertEqual(marker.read_text(), "keep")
            self.assertEqual(mesh.read_bytes(), b"protected fixture")
            self.assertEqual(list(out.iterdir()), [marker])

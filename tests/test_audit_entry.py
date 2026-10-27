import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/quality_demo/audit_mps.py"
SPEC = importlib.util.spec_from_file_location("audit_entry", SCRIPT)
AUDIT_ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT_ENTRY)


class _FakeProcessor:
    pass


class _FakeReferenceContainer:
    def __init__(self):
        self.attn_refview = SimpleNamespace(processor=_FakeProcessor())


class _FakeTorchInner:
    def modules(self):
        return [_FakeReferenceContainer()]


class _FakeDispatch:
    def __init__(self):
        self._original = _FakeTorchInner()


class _FakeOuter:
    def __init__(self):
        self.unet = _FakeDispatch()
        self.unet_dual = None

    def modules(self):
        return []


class _FakePipeline:
    def __init__(self):
        self.unet = _FakeOuter()


class AuditEntryTests(unittest.TestCase):
    def test_runtime_sources_unwraps_dispatch_and_records_reference_processor(self):
        sources = AUDIT_ENTRY.runtime_sources(_FakePipeline())
        self.assertTrue(sources["inner_unet"]["class"].endswith("._FakeDispatch"))
        self.assertTrue(sources["inner_unet_torch"]["class"].endswith("._FakeTorchInner"))
        self.assertTrue(sources["reference_processor"]["class"].endswith("._FakeProcessor"))
        self.assertEqual(sources["reference_processor"]["module_file"], __file__)

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

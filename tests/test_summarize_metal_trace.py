import json
from contextlib import redirect_stdout
from datetime import datetime
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.quality_demo.summarize_metal_trace import _schema_rows, main


class SummarizeMetalTraceTests(unittest.TestCase):
    def _run_case(self, root, *, signpost=True, gpu=True, fallback=False, marker_end=2.0):
        start = datetime.fromisoformat("2026-10-27T00:00:00+00:00").timestamp()
        (root / "toc.xml").write_text("""<trace-toc><run><info>
          <target><process pid='42' return-exit-status='0'/></target>
          <summary><start-date>2026-10-27T00:00:00+00:00</start-date>
            <end-date>2026-10-27T00:00:03+00:00</end-date></summary>
        </info></run></trace-toc>""")
        (root / "markers.json").write_text(json.dumps({"fallback_env": "0", "trace_markers": [
            {"stage": "first_step", "event": "begin", "pid": 42, "wall_unix_s": start + 0.5},
            {"stage": "first_step", "event": "end", "pid": 42,
             "wall_unix_s": start + marker_end, "status": "completed"}]}))
        signpost_row = ("<node><row><time>1000000000</time>"
                        "<process fmt='python (42)'/><name fmt='PyTorchOperationEvents'/>"
                        "<subsystem fmt='PyTorchMPS'/><message fmt='" +
                        ("CPU fallback" if fallback else "aten::conv") + "'/></row></node>") if signpost else ""
        (root / "signposts.xml").write_text(
            "<root><node><schema name='os-signpost'>" + "".join(
                f"<col><mnemonic>{name}</mnemonic></col>"
                for name in ("time", "process", "name", "subsystem", "message")) +
            "</schema></node>" + signpost_row + "</root>")
        gpu_row = ("<node><row><start>1000000000</start><duration>1000000</duration>"
                   "<channel-name fmt='Compute'/><process fmt='python (42)'/></row></node>") if gpu else ""
        (root / "gpu.xml").write_text(
            "<root><node><schema name='metal-gpu-intervals'>" + "".join(
                f"<col><mnemonic>{name}</mnemonic></col>"
                for name in ("start", "duration", "channel-name", "process")) +
            "</schema></node>" + gpu_row + "</root>")
        args = ["summarize", "--toc", str(root / "toc.xml"), "--trace", str(root / "metal.trace"), "--markers", str(root / "markers.json"),
                "--signposts", str(root / "signposts.xml"), "--gpu-intervals", str(root / "gpu.xml"),
                "--output", str(root / "out.json")]
        with patch("sys.argv", args), redirect_stdout(io.StringIO()):
            try:
                main()
                exit_code = 0
            except SystemExit as exc:
                exit_code = exc.code
        return exit_code, json.loads((root / "out.json").read_text())

    def test_schema_rows_accepts_sibling_schema_and_rows(self):
        xml = """<trace-query-result>
          <node><schema name="os-signpost"><col><mnemonic>time</mnemonic></col>
            <col><mnemonic>process</mnemonic></col></schema></node>
          <node><row><event-time fmt="10">10</event-time>
            <process fmt="python (42)">python (42)</process></row></node>
        </trace-query-result>"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.xml"
            path.write_text(xml)
            rows = list(_schema_rows(path, "os-signpost"))
        self.assertEqual(rows[0]["process"]["formatted"], "python (42)")

    def test_schema_rows_fails_closed_on_duplicate_schema(self):
        xml = "<root><schema name='x'/><schema name='x'/></root>"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.xml"
            path.write_text(xml)
            with self.assertRaisesRegex(ValueError, "expected one x schema"):
                list(_schema_rows(path, "x"))

    def test_cli_rejects_marker_pid_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            toc = root / "toc.xml"
            toc.write_text("""<trace-toc><run><info>
              <target><process pid='42' return-exit-status='0'/></target>
              <summary><start-date>2026-10-27T00:00:00+00:00</start-date>
                <end-date>2026-10-27T00:01:00+00:00</end-date></summary>
            </info></run></trace-toc>""")
            markers = root / "markers.json"
            markers.write_text(json.dumps({"trace_markers": [{"pid": 99}]}))
            args = ["summarize", "--toc", str(toc), "--trace", str(root / "metal.trace"), "--markers", str(markers),
                    "--signposts", str(root / "signposts.xml"),
                    "--gpu-intervals", str(root / "gpu.xml"), "--output", str(root / "out.json")]
            with patch("sys.argv", args):
                with self.assertRaisesRegex(ValueError, "PID"):
                    main()

    def test_cli_requires_complete_stage_gpu_and_signposts_and_no_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, options, expected in (
                ("pass", {}, "passed"),
                ("missing_gpu", {"gpu": False}, "failed"),
                ("missing_signpost", {"signpost": False}, "failed"),
                ("fallback", {"fallback": True}, "failed"),
                ("truncated", {"marker_end": 4.0}, "failed"),
            ):
                with self.subTest(name=name):
                    code, report = self._run_case(root, **options)
                    self.assertEqual(report["status"], expected)
                    self.assertEqual(code, 0 if expected == "passed" else 1)
                    if name == "truncated":
                        self.assertFalse(report["stages"][0]["fully_recorded"])


if __name__ == "__main__":
    unittest.main()

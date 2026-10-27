"""Correlate one probe's markers with exported PyTorch and Metal trace rows."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def _schema_rows(path, schema_name):
    root = ET.parse(path).getroot()
    identifiers = {node.get("id"): node for node in root.iter() if node.get("id")}

    def resolve(node):
        return identifiers[node.get("ref")] if node.get("ref") else node

    def raw(node):
        node = resolve(node)
        return node.text

    def formatted(node):
        node = resolve(node)
        return node.get("fmt", node.text)

    schemas = [node for node in root.iter("schema") if node.get("name") == schema_name]
    if len(schemas) != 1:
        raise ValueError(f"expected one {schema_name} schema, found {len(schemas)}")
    names = [column.findtext("mnemonic") for column in schemas[0].findall("col")]
    # xctrace may place a schema and its rows in separate sibling <node>s.
    for row in root.iter("row"):
        if len(row) != len(names):
            raise ValueError(f"{schema_name} row has {len(row)} values, expected {len(names)}")
        yield {name: {"raw": raw(value), "formatted": formatted(value)}
               for name, value in zip(names, row)}


def _value(row, key, kind="formatted"):
    return row.get(key, {}).get(kind)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toc", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--markers", type=Path, required=True)
    parser.add_argument("--signposts", type=Path, required=True)
    parser.add_argument("--gpu-intervals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    toc = ET.parse(args.toc).getroot()
    run = toc.find("run")
    target = run.find("./info/target/process")
    summary = run.find("./info/summary")
    pid = int(target.get("pid"))
    trace_start = datetime.fromisoformat(summary.findtext("start-date")).timestamp()
    trace_end = datetime.fromisoformat(summary.findtext("end-date")).timestamp()
    marker_payload = json.loads(args.markers.read_text())
    target_fallback = next((item.get("value") for item in
                            run.findall("./info/target/environment/item")
                            if item.get("key") == "PYTORCH_ENABLE_MPS_FALLBACK"), None)
    marker_fallback = marker_payload.get("fallback_env")
    if target_fallback is not None and marker_fallback is not None and target_fallback != marker_fallback:
        raise ValueError("trace target fallback env and marker fallback env do not match")
    fallback_env = target_fallback if target_fallback is not None else marker_fallback
    markers = marker_payload["trace_markers"]
    if not markers or {entry["pid"] for entry in markers} != {pid}:
        raise ValueError("trace target PID and marker PID do not match")

    intervals = []
    stack = {}
    for marker in markers:
        key = marker["stage"]
        if marker["event"] == "begin":
            stack.setdefault(key, []).append(marker)
        elif marker["event"] == "end":
            begin = stack[key].pop(0)
            if marker.get("status") != "completed":
                raise ValueError(f"stage did not complete: {key}")
            intervals.append({"stage": key, "begin_wall_unix_s": begin["wall_unix_s"],
                              "end_wall_unix_s": marker["wall_unix_s"]})
    if any(stack.values()):
        raise ValueError("unclosed trace marker")

    target_name = f" ({pid})"
    signposts = []
    for row in _schema_rows(args.signposts, "os-signpost"):
        if (_value(row, "process") or "").endswith(target_name):
            signposts.append(row)
    gpu_rows = []
    for row in _schema_rows(args.gpu_intervals, "metal-gpu-intervals"):
        if (_value(row, "process") or "").endswith(target_name):
            gpu_rows.append(row)

    results = []
    for interval in intervals:
        fully_recorded = (trace_start <= interval["begin_wall_unix_s"]
                          <= interval["end_wall_unix_s"] <= trace_end)
        start_ns = round((interval["begin_wall_unix_s"] - trace_start) * 1e9)
        end_ns = round((interval["end_wall_unix_s"] - trace_start) * 1e9)
        selected_signposts = [row for row in signposts
                              if start_ns <= int(_value(row, "time", "raw")) <= end_ns]
        selected_gpu = [row for row in gpu_rows
                        if start_ns <= int(_value(row, "start", "raw")) <= end_ns]
        pytorch = [row for row in selected_signposts if _value(row, "subsystem") == "PyTorchMPS"]
        fallbacks = []
        for row in pytorch:
            text = " ".join(_value(row, key) or ""
                            for key in ("name", "category", "message", "format-string"))
            if "fallback" in text.lower():
                fallbacks.append(text)
        channels = Counter(_value(row, "channel-name") for row in selected_gpu)
        gpu_duration_ns = sum(int(_value(row, "duration", "raw")) for row in selected_gpu)
        results.append({**interval, "duration_s": (end_ns - start_ns) / 1e9,
                        "fully_recorded": fully_recorded,
                        "pytorch_signposts": len(pytorch),
                        "pytorch_names": dict(Counter(_value(row, "name") for row in pytorch)),
                        "cpu_fallback_signposts": len(fallbacks),
                        "gpu_intervals": len(selected_gpu),
                        "gpu_duration_ms": gpu_duration_ns / 1e6,
                        "gpu_channels": dict(channels),
                        "status": "passed" if fully_recorded and pytorch and selected_gpu
                                  and not fallbacks else "failed"})

    report = {"trace": str(args.trace.resolve()), "target_pid": pid,
              "target_exit_status": target.get("return-exit-status"),
              "termination_reason": target.get("termination-reason"),
              "fallback_env": fallback_env,
              "trace_start": trace_start, "trace_end": trace_end, "stages": results,
              "status": "passed" if results and all(item["status"] == "passed" for item in results)
                        and target.get("return-exit-status") == "0" and fallback_env == "0" else "failed",
              "limitations": [
                  "Stage timing uses profiler synchronization and is not a performance measurement.",
                  "The summary covers only the named stage windows and target PID.",
              ]}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

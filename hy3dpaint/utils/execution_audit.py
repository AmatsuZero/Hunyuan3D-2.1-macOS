"""Opt-in placement/call audit. Metal traces remain a separate acceptance gate."""
import inspect
import json
import time
from collections.abc import Mapping
from pathlib import Path

import torch


class AuditError(RuntimeError):
    pass


def tensor_metadata(value, path=""):
    if isinstance(value, torch.Tensor):
        yield {"path": path, "device": str(value.device),
               "dtype": str(value.dtype), "shape": list(value.shape)}
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from tensor_metadata(item, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensor_metadata(item, f"{path}.{index}")
    elif type(value).__name__ == "DiagonalGaussianDistribution":
        # Diffusers VAE.encode returns a distribution, rather than a tensor.
        yield from tensor_metadata(value.parameters, f"{path}.parameters")


class ExecutionAudit:
    def __init__(self, required, event_path=None):
        self.required = set(required)
        self.events = []
        self._restore = []
        self.event_path = Path(event_path) if event_path is not None else None

    @staticmethod
    def require_mps():
        if not torch.backends.mps.is_available():
            raise AuditError("MPS is required")

    @staticmethod
    def _check(items, kind, cpu_scalar_paths=()):
        for item in items:
            if item["device"].split(":")[0] == "mps":
                continue
            # Only explicitly named scalar control inputs may remain on CPU.
            if (kind == "input" and item["device"] == "cpu"
                    and item["path"] in cpu_scalar_paths and item["shape"] in ([], [1])):
                continue
            raise AuditError(f"{kind} {item['path']} is on {item['device']}, expected mps")

    def wrap(self, label, module, method="forward", cpu_scalar_paths=()):
        original = getattr(module, method)
        signature = inspect.signature(original)

        def audited(*args, **kwargs):
            event = {"stage": label, "backend": "torch/mps", "status": "failed"}
            event["start_monotonic_s"] = time.perf_counter()
            self.events.append(event)
            started = time.perf_counter()
            try:
                event["parameters"] = [
                    {"path": name, "device": str(t.device), "dtype": str(t.dtype),
                     "shape": list(t.shape)} for name, t in module.named_parameters()]
                event["buffers"] = [
                    {"path": name, "device": str(t.device), "dtype": str(t.dtype),
                     "shape": list(t.shape)} for name, t in module.named_buffers()]
                self._check(event["parameters"], "parameter")
                self._check(event["buffers"], "buffer")
                bound = signature.bind_partial(*args, **kwargs)
                event["inputs"] = list(tensor_metadata(bound.arguments))
                self._check(event["inputs"], "input", cpu_scalar_paths)
                if not event["inputs"]:
                    raise AuditError(f"{label}: no tensor inputs observed")
                self.require_mps()
                torch.mps.synchronize()
                started = time.perf_counter()
                event["compute_start_monotonic_s"] = started
                output = original(*args, **kwargs)
                torch.mps.synchronize()
                event["outputs"] = list(tensor_metadata(output))
                self._check(event["outputs"], "output")
                if not event["outputs"]:
                    raise AuditError(f"{label}: no tensor outputs observed")
                event["status"] = "passed"
                return output
            except Exception as exc:
                event["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                event["end_monotonic_s"] = time.perf_counter()
                event["elapsed_s"] = time.perf_counter() - started
                if self.event_path is not None:
                    with self.event_path.open("a") as stream:
                        stream.write(json.dumps(event) + "\n")

        self._restore.append((module, method, original))
        setattr(module, method, audited)

    def validate(self):
        failed = [e["stage"] for e in self.events if e["status"] != "passed"]
        if failed:
            raise AuditError(f"failed stages: {failed}")
        missing = self.required - {e["stage"] for e in self.events}
        if missing:
            raise AuditError(f"required stages not called: {sorted(missing)}")

    def save(self, path, **metadata):
        Path(path).write_text(json.dumps({
            **metadata, "required": sorted(self.required), "events": self.events,
            "limitation": "Placement and calls do not prove exclusive GPU execution; correlate Metal traces."
        }, indent=2))

    def close(self):
        for module, method, original in reversed(self._restore):
            setattr(module, method, original)
        self._restore.clear()

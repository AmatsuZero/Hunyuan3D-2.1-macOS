"""Validate numeric acceptance gates for an MLX dual-reference replay report."""

import argparse
import json
import math
from pathlib import Path


DEFAULT_MAX_CACHE_NRMSE = 0.01
DEFAULT_MAX_REFERENCE_WRITE_NRMSE = 0.03
EXPECTED_CHECKPOINT_SHA256 = "675a1b5cd0098b2002637c443946529c03c5cd54427f40245263350feb3dd5b8"
EXPECTED_LAYERS = frozenset((
    "down_0_0_0", "down_0_1_0", "down_1_0_0", "down_1_1_0",
    "down_2_0_0", "down_2_1_0", "mid_0_0", "up_1_0_0",
    "up_1_1_0", "up_1_2_0", "up_2_0_0", "up_2_1_0",
    "up_2_2_0", "up_3_0_0", "up_3_1_0", "up_3_2_0",
))


def _finite_metric(metric, name):
    value = metric.get("nrmse")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name}.nrmse must be finite")
    return float(value)


def evaluate_report(
    report,
    *,
    max_cache_nrmse=DEFAULT_MAX_CACHE_NRMSE,
    max_reference_write_nrmse=DEFAULT_MAX_REFERENCE_WRITE_NRMSE,
):
    """Return observed values and pass/fail status for a replay report."""
    if report.get("inputs_exact") is not True:
        raise ValueError("replay inputs are not exact")
    if report.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA256 differs from frozen reference")
    if report.get("mapped_weights") != 686:
        raise ValueError("expected exactly 686 mapped dual-reference weights")
    layers = report.get("layers")
    if not isinstance(layers, dict) or set(layers) != EXPECTED_LAYERS:
        raise ValueError("replay report must have the exact 16 reference layers")
    observed_cache_nrmse = max(
        _finite_metric(metric, f"layers.{name}") for name, metric in layers.items()
    )
    observed_reference_write_nrmse = _finite_metric(
        report.get("reference_write", {}), "reference_write"
    )
    if not all(
        math.isfinite(value) and value >= 0
        for value in (max_cache_nrmse, max_reference_write_nrmse)
    ):
        raise ValueError("tolerances must be finite and non-negative")
    return {
        "max_cache_nrmse": float(max_cache_nrmse),
        "max_reference_write_nrmse": float(max_reference_write_nrmse),
        "observed_cache_nrmse": observed_cache_nrmse,
        "observed_reference_write_nrmse": observed_reference_write_nrmse,
        "passed": (
            observed_cache_nrmse <= max_cache_nrmse
            and observed_reference_write_nrmse <= max_reference_write_nrmse
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-cache-nrmse", type=float, default=DEFAULT_MAX_CACHE_NRMSE)
    parser.add_argument(
        "--max-reference-write-nrmse",
        type=float,
        default=DEFAULT_MAX_REFERENCE_WRITE_NRMSE,
    )
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text())
        result = evaluate_report(
            report,
            max_cache_nrmse=args.max_cache_nrmse,
            max_reference_write_nrmse=args.max_reference_write_nrmse,
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

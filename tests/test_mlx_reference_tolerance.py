"""Numeric acceptance gates for captured MLX reference reports."""

import unittest

from tools.quality_demo.validate_mlx_reference_report import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_LAYERS,
    evaluate_report,
)


def _report(cache_nrmse=0.006, write_nrmse=0.02):
    return {
        "inputs_exact": True,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "mapped_weights": 686,
        "layers": {name: {"nrmse": cache_nrmse} for name in EXPECTED_LAYERS},
        "reference_write": {"nrmse": write_nrmse},
    }


class MLXReferenceToleranceTests(unittest.TestCase):
    def test_observed_replay_is_inside_default_gates(self):
        result = evaluate_report(_report())
        self.assertTrue(result["passed"])
        self.assertEqual(result["max_cache_nrmse"], 0.01)
        self.assertEqual(result["max_reference_write_nrmse"], 0.03)

    def test_cache_gate_rejects_layer_outlier(self):
        result = evaluate_report(_report(cache_nrmse=0.0101))
        self.assertFalse(result["passed"])

    def test_reference_write_gate_rejects_outlier(self):
        result = evaluate_report(_report(write_nrmse=0.0301))
        self.assertFalse(result["passed"])

    def test_requires_exact_inputs(self):
        with self.assertRaisesRegex(ValueError, "inputs are not exact"):
            evaluate_report({**_report(), "inputs_exact": False})

    def test_rejects_incomplete_layer_report(self):
        with self.assertRaisesRegex(ValueError, "exact 16"):
            evaluate_report({**_report(), "layers": {}})

    def test_rejects_checkpoint_drift(self):
        with self.assertRaisesRegex(ValueError, "checkpoint SHA256"):
            evaluate_report({**_report(), "checkpoint_sha256": "0" * 64})


if __name__ == "__main__":
    unittest.main()

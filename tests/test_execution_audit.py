import sys
from pathlib import Path
import unittest
import tempfile
import json
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hy3dpaint"))
from utils.execution_audit import ExecutionAudit, AuditError, tensor_metadata


class ExecutionAuditTests(unittest.TestCase):
    def test_failed_event_is_persisted_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            audit = ExecutionAudit(required=["model"], event_path=path)
            model = torch.nn.Linear(2, 2)
            audit.wrap("model", model)
            with self.assertRaises(AuditError):
                model(torch.ones(1, 2))
            event = json.loads(path.read_text())
            self.assertEqual(event["status"], "failed")
            self.assertIn("cpu", event["error"])
            audit.close()

    def test_mps_unavailable_is_an_error(self):
        with patch.object(torch.backends.mps, "is_available", return_value=False):
            with self.assertRaisesRegex(AuditError, "MPS is required"):
                ExecutionAudit.require_mps()

    def test_captures_vae_encode_distribution_output(self):
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        from diffusers.models.modeling_outputs import AutoencoderKLOutput
        output = AutoencoderKLOutput(latent_dist=DiagonalGaussianDistribution(torch.ones(1, 8, 2, 2)))
        tensors = list(tensor_metadata(output))
        self.assertTrue(tensors)
        self.assertEqual(tensors[0]["shape"], [1, 8, 2, 2])

    def test_rejects_cpu_model_before_computation(self):
        model = torch.nn.Linear(2, 2)
        audit = ExecutionAudit(required=["model"])
        audit.wrap("model", model)
        with self.assertRaisesRegex(AuditError, "parameter.*cpu"):
            model(torch.ones(1, 2))
        self.assertEqual(audit.events[-1]["status"], "failed")
        audit.close()

    def test_rejects_cpu_input_and_restores_method(self):
        model = torch.nn.Identity()
        original = model.forward
        audit = ExecutionAudit(required=["identity"])
        audit.wrap("identity", model)
        with self.assertRaisesRegex(AuditError, "input.*cpu"):
            model(torch.ones(2))
        audit.close()
        self.assertEqual(model.forward, original)

    def test_empty_or_failed_audit_cannot_pass(self):
        audit = ExecutionAudit(required=["vae.encode"])
        with self.assertRaisesRegex(AuditError, "not called"):
            audit.validate()

    @unittest.skipUnless(torch.backends.mps.is_available(), "requires Apple GPU")
    def test_validates_actual_mps_inputs_outputs_and_nonforward_method(self):
        class Model(torch.nn.Module):
            def encode(self, x):
                return {"latent": x.square()}
        model = Model()
        audit = ExecutionAudit(required=["vae.encode"])
        audit.wrap("vae.encode", model, method="encode")
        x = torch.tensor([2., 3.], device="mps")
        self.assertTrue(torch.equal(model.encode(x)["latent"].cpu(), torch.tensor([4., 9.])))
        audit.validate()
        self.assertEqual(audit.events[0]["outputs"][0]["device"], "mps:0")
        audit.close()

    @unittest.skipUnless(torch.backends.mps.is_available(), "requires Apple GPU")
    def test_rejects_cpu_output_even_when_input_is_mps(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.cpu()
        audit = ExecutionAudit(required=["model"])
        model = Model()
        audit.wrap("model", model)
        with self.assertRaisesRegex(AuditError, "output.*cpu"):
            model(torch.ones(2, device="mps"))
        with self.assertRaisesRegex(AuditError, "failed"):
            audit.validate()
        audit.close()

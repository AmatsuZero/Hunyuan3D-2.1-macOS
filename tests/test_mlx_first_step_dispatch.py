"""Opt-in MLX first generation must reuse the already built outer cache."""

import os
from unittest import mock
import unittest

import torch


class _Original(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, *_args, **_kwargs):
        self.calls += 1
        return (torch.tensor(1),)


class MLXFirstStepDispatchTests(unittest.TestCase):
    def test_opt_in_runs_first_generation_in_mlx_and_resets(self):
        from hy3dpaint.mlx_utils import MLXDispatchUNet

        original = _Original()
        dispatch = MLXDispatchUNet(original, mlx_model=None)
        sample = torch.zeros((1, 4, 2, 2))
        def populate(*_args, **_kwargs):
            dispatch._cached_mlx = {"condition_embed_dict": {"layer": sample}}
        with (mock.patch.dict(os.environ, {"HUNYUAN_MLX_FIRST_STEP": "1"}),
              mock.patch.object(dispatch, "_cache_constant_inputs", side_effect=populate) as cache,
              mock.patch.object(dispatch, "_mlx_forward", return_value=(torch.tensor(2),)) as mlx):
            self.assertEqual(dispatch(sample, torch.tensor(999), sample,
                                      cross_attention_kwargs={"mode": "r"})[0].item(), 2)
            self.assertEqual(original.calls, 0)
            cache.assert_called_once()
            mlx.assert_called_once()
            dispatch.reset()
            self.assertEqual(dispatch(sample, torch.tensor(999), sample,
                                      cross_attention_kwargs={"mode": "r"})[0].item(), 2)
            self.assertEqual(original.calls, 0)

    def test_default_first_generation_stays_on_torch(self):
        from hy3dpaint.mlx_utils import MLXDispatchUNet

        original = _Original()
        dispatch = MLXDispatchUNet(original, mlx_model=None)
        sample = torch.zeros((1, 4, 2, 2))
        with (mock.patch.dict(os.environ, {}, clear=True),
              mock.patch.object(dispatch, "_cache_constant_inputs"),
              mock.patch.object(dispatch, "_mlx_forward") as mlx):
            self.assertEqual(dispatch(sample, torch.tensor(999), sample,
                                      cross_attention_kwargs={"mode": "r"})[0].item(), 1)
            self.assertEqual(original.calls, 1)
            mlx.assert_not_called()


if __name__ == "__main__":
    unittest.main()

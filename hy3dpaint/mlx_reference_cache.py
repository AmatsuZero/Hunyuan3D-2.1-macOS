"""Request-local MLX reference cache producer.

The dual reference UNet produces one normalized token tensor per transformer
layer during its ``mode='w'`` pass.  This module owns the one-time boundary
conversion for the opt-in MLX path.  It deliberately does not retain tensors
between requests and does not alter the Torch producer, which remains the
fallback until a Torch-free dual UNet is validated.
"""

from __future__ import annotations

from typing import Mapping

import mlx.core as mx
import torch


class MLXReferenceCacheProducer:
    """Convert and validate one dynamic reference-write cache.

    ``produce`` preserves insertion-independent layer lookup and tensor values;
    MLX arrays are evaluated before returning so the request owns a complete,
    stable cache.  ``reset`` drops all request state and is required between
    pipeline requests.
    """

    def __init__(self, expected_layers: int = 16):
        if expected_layers <= 0:
            raise ValueError("expected_layers must be positive")
        self.expected_layers = int(expected_layers)
        self._cache: dict[str, mx.array] | None = None

    @property
    def cache(self) -> dict[str, mx.array] | None:
        return self._cache

    def produce(self, condition_embed_dict: Mapping[str, torch.Tensor],
                *, dtype=None) -> dict[str, mx.array]:
        self._cache = None
        if not isinstance(condition_embed_dict, Mapping):
            raise TypeError("condition_embed_dict must be a mapping")
        if len(condition_embed_dict) != self.expected_layers:
            raise ValueError(
                f"reference cache must contain {self.expected_layers} layers, "
                f"got {len(condition_embed_dict)}"
            )
        produced: dict[str, mx.array] = {}
        for name, tensor in condition_embed_dict.items():
            if not isinstance(name, str) or not name:
                raise ValueError("reference cache layer names must be non-empty strings")
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
                raise ValueError(f"reference cache layer {name!r} must be a rank-3 Torch tensor")
            if not tensor.is_floating_point():
                raise ValueError(f"reference cache layer {name!r} must be floating point")
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"reference cache layer {name!r} contains non-finite values")
            value = mx.array(tensor.detach().cpu().numpy())
            produced[name] = value.astype(dtype) if dtype is not None else value
        mx.eval(list(produced.values()))
        self._cache = produced
        return produced

    def reset(self) -> None:
        self._cache = None

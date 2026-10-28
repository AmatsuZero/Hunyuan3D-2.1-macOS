"""Opt-in Torch boundary for the MLX outer DINO projector."""

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import mlx.core as mx

try:
    from .mlx_dino_projector import MLXDinoProjector
except ImportError:
    from mlx_dino_projector import MLXDinoProjector


_WEIGHT_NAMES = ("proj.weight", "proj.bias", "norm.weight", "norm.bias")


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MLXDinoProjectorAdapter(torch.nn.Module):
    """Run a validated MLX projector behind the outer UNet's Torch contract.

    The dynamic outer UNet expects a Torch module that receives a Torch tensor
    and returns a Torch tensor.  This adapter keeps that contract while making
    the projector itself an MLX operation.  It is intentionally explicit about
    the two NumPy synchronization boundaries; the surrounding outer UNet and
    reference write remain Torch in this migration slice.
    """

    def __init__(self, projector, *, bundle_path=None, source_sha256=None, config_sha256=None):
        super().__init__()
        self.projector = projector
        self.bundle_path = str(bundle_path) if bundle_path is not None else None
        self.source_sha256 = source_sha256
        self.config_sha256 = config_sha256

    def forward(self, image_embeds):
        if not isinstance(image_embeds, torch.Tensor):
            raise TypeError("MLX DINO projector adapter expects a Torch tensor")
        if image_embeds.device.type != "mps":
            raise ValueError("MLX DINO projector adapter requires an MPS input")
        input_array = mx.array(image_embeds.detach().cpu().numpy())
        output = self.projector(input_array)
        mx.eval(output)
        return torch.from_numpy(np.array(output)).to(
            device=image_embeds.device, dtype=image_embeds.dtype
        )


def install_mlx_dino_projector(outer_unet, checkpoint_path):
    """Install the bundle-backed MLX projector only for an explicit opt-in."""
    bundle_path = os.environ.get("HUNYUAN_MLX_DINO_PROJECTOR_BUNDLE")
    if not bundle_path:
        return False
    if checkpoint_path is None:
        raise FileNotFoundError("MLX DINO projector requires the source checkpoint")
    checkpoint_path = Path(checkpoint_path)
    config_path = checkpoint_path.with_name("config.json")
    try:
        from .mlx_condition_bundle import load_bundle
    except ImportError:
        from mlx_condition_bundle import load_bundle
    source_sha = _sha256_file(checkpoint_path)
    config_sha = _sha256_file(config_path)
    arrays, _ = load_bundle(
        Path(bundle_path), source_sha, config_sha
    )
    prefix = "image_proj_model_dino."
    projector = MLXDinoProjector()
    projector.load_weights(
        [(name, arrays[prefix + name]) for name in _WEIGHT_NAMES], strict=True
    )
    mx.eval(projector.parameters())
    outer_unet.image_proj_model_dino = MLXDinoProjectorAdapter(
        projector,
        bundle_path=bundle_path,
        source_sha256=source_sha,
        config_sha256=config_sha,
    )
    return True

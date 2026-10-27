"""Torch-free MLX implementation of the RealESRGAN x4 RRDB network.

The checkpoint conversion helper intentionally imports Torch only inside the
offline conversion function.  Runtime callers can load the resulting MLX
``.safetensors`` bundle with :func:`load_bundle` and never import Torch.
"""

from importlib.metadata import version
import os
from pathlib import Path
import tempfile

import mlx.core as mx
import mlx.nn as nn
import numpy as np



BUNDLE_SCHEMA = "hy3d-realesrgan-mlx-v1"
LAYOUT = "conv2d=OHWI;linear=OI;norm=channels"


def _leaky_relu(value, negative_slope=0.2):
    return mx.maximum(value, value * negative_slope)


def _nearest_x2(value):
    value = mx.repeat(value, 2, axis=1)
    return mx.repeat(value, 2, axis=2)


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, padding=1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, padding=1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, padding=1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, padding=1)

    def __call__(self, value):
        x1 = _leaky_relu(self.conv1(value))
        x2 = _leaky_relu(self.conv2(mx.concatenate([value, x1], axis=-1)))
        x3 = _leaky_relu(self.conv3(mx.concatenate([value, x1, x2], axis=-1)))
        x4 = _leaky_relu(self.conv4(mx.concatenate([value, x1, x2, x3], axis=-1)))
        x5 = self.conv5(mx.concatenate([value, x1, x2, x3, x4], axis=-1))
        return x5 * 0.2 + value


class RRDB(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def __call__(self, value):
        residual = value
        value = self.rdb1(value)
        value = self.rdb2(value)
        value = self.rdb3(value)
        return value * 0.2 + residual


class MLXRRDBNet(nn.Module):
    """RRDBNet with the same NHWC contract as the rest of the MLX path."""

    def __init__(self, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, num_feat, 3, padding=1)
        self.body = [RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, padding=1)

    def __call__(self, value):
        feature = self.conv_first(value)
        body = feature
        for block in self.body:
            body = block(body)
        feature = feature + self.conv_body(body)
        feature = _leaky_relu(self.conv_up1(_nearest_x2(feature)))
        feature = _leaky_relu(self.conv_up2(_nearest_x2(feature)))
        return self.conv_last(_leaky_relu(self.conv_hr(feature)))


def _flatten_tree(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _flatten_tree(child, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _flatten_tree(child, f"{prefix}.{index}")
    else:
        yield prefix, value


def _torch_state_dict(checkpoint: str | Path):
    import torch

    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "params_ema" in loaded:
        return loaded["params_ema"]
    if "params" in loaded:
        return loaded["params"]
    return loaded


def load_weights_from_pytorch(model: MLXRRDBNet, checkpoint: str | Path,
                              dtype=mx.float16):
    """Offline-only conversion from BasicSR OIHW tensors to MLX OHWI."""
    state = _torch_state_dict(checkpoint)
    expected = {name for name, _ in _flatten_tree(model.parameters())}
    converted = {}
    for name, tensor in state.items():
        if name not in expected:
            raise ValueError(f"unexpected RealESRGAN checkpoint key: {name}")
        array = tensor.detach().cpu().numpy()
        if array.ndim == 4:
            array = array.transpose(0, 2, 3, 1)
        converted[name] = mx.array(array, dtype=dtype)
    missing = sorted(expected - set(converted))
    if missing:
        raise ValueError(f"missing RealESRGAN checkpoint keys: {', '.join(missing)}")
    model.load_weights(list(converted.items()), strict=True)
    mx.eval(model.parameters())
    return model


def _bundle_metadata(source_sha256, dtype, parameter_count):
    return {
        "schema": BUNDLE_SCHEMA,
        "source_sha256": source_sha256,
        "mlx_version": version("mlx"),
        "dtype": dtype,
        "parameter_count": str(parameter_count),
        "layout": LAYOUT,
    }


def _validate_bundle_metadata(metadata, expected_source_sha256=None, expected_dtype=None):
    if metadata.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("bundle schema mismatch")
    if metadata.get("layout") != LAYOUT:
        raise ValueError("bundle layout mismatch")
    if metadata.get("mlx_version") != version("mlx"):
        raise ValueError("bundle mlx_version mismatch")
    source_sha = metadata.get("source_sha256", "")
    if len(source_sha) != 64 or any(c not in "0123456789abcdef" for c in source_sha):
        raise ValueError("invalid source_sha256")
    if expected_source_sha256 is not None and source_sha != expected_source_sha256:
        raise ValueError("source_sha256 mismatch")
    dtype = metadata.get("dtype")
    if dtype not in {"float16", "float32"} or (expected_dtype is not None and dtype != expected_dtype):
        raise ValueError("bundle dtype mismatch")
    count = metadata.get("parameter_count", "")
    if not count.isdecimal() or int(count) <= 0:
        raise ValueError("invalid parameter_count")


def save_bundle(model: MLXRRDBNet, bundle: str | Path, source_sha256: str,
                dtype=None):
    """Write an MLX-native bundle atomically, preserving exact parameter names."""
    bundle = Path(bundle)
    if bundle.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {bundle}")
    parameters = dict(_flatten_tree(model.parameters()))
    if not parameters:
        raise ValueError("model has no parameters")
    dtypes = {str(value.dtype).removeprefix("mlx.core.") for value in parameters.values()}
    if len(dtypes) != 1:
        raise ValueError("bundle parameters have mixed dtypes")
    actual_dtype = dtypes.pop()
    if dtype is not None and actual_dtype != dtype:
        raise ValueError("requested dtype does not match model parameters")
    metadata = _bundle_metadata(source_sha256, actual_dtype, len(parameters))
    bundle.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{bundle.name}.", suffix=".safetensors",
                                         dir=bundle.parent, delete=False) as stream:
            temporary = Path(stream.name)
        mx.save_safetensors(str(temporary), parameters, metadata=metadata)
        os.replace(temporary, bundle)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return metadata


def load_bundle(model: MLXRRDBNet, bundle: str | Path, expected_source_sha256=None,
                expected_dtype=None):
    """Load and strictly validate a previously converted bundle without Torch."""
    arrays, metadata = mx.load(str(bundle), return_metadata=True)
    _validate_bundle_metadata(metadata, expected_source_sha256, expected_dtype)
    if len(arrays) != int(metadata["parameter_count"]):
        raise ValueError("bundle parameter_count mismatch")
    for name, value in arrays.items():
        if str(value.dtype).removeprefix("mlx.core.") != metadata["dtype"]:
            raise ValueError(f"{name} tensor dtype mismatch")
    model.load_weights(list(arrays.items()), strict=True)
    mx.eval(model.parameters())
    return model


def enhance_rgb(model: MLXRRDBNet, image: np.ndarray, dtype=mx.float16) -> np.ndarray:
    """Enhance one RGB uint8 image with RealESRGAN's x4 channel semantics."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("RGB input must have shape [height, width, 3]")
    if image.dtype != np.uint8:
        raise ValueError("RGB input must be uint8")
    # RealESRGANer receives PIL RGB data and swaps BGR/RGB at both boundaries.
    value = mx.array(image[..., ::-1].astype(np.float32) / 255.0, dtype=dtype)[None]
    output = model(value)
    mx.eval(output)
    output = np.array(output[0], dtype=np.float32)[..., ::-1]
    return np.rint(np.clip(output, 0.0, 1.0) * 255.0).astype(np.uint8)

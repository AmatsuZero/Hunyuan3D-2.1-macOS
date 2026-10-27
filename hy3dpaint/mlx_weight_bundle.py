"""Load a converted inner-UNet bundle without importing PyTorch."""

from importlib.metadata import version
from pathlib import Path

import mlx.core as mx


BUNDLE_SCHEMA = "hy3d-inner-unet-mlx-v1"
LAYOUT = "conv2d=OHWI;linear=OI;norm=channels"
_DTYPES = {"float16", "float32"}


def build_metadata(source_sha256: str, dtype: str, parameter_count: int) -> dict[str, str]:
    """Describe the exact source and tensor contract of a converted bundle."""
    return {
        "schema": BUNDLE_SCHEMA,
        "source_sha256": source_sha256,
        "mlx_version": version("mlx"),
        "dtype": dtype,
        "parameter_count": str(parameter_count),
        "layout": LAYOUT,
    }


def validate_metadata(metadata: dict[str, str], expected_source_sha256: str | None = None,
                      expected_dtype: str | None = None,
                      expected_config_sha256: str | None = None) -> None:
    """Fail closed on a bundle from another architecture, source, or precision."""
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
    config_sha = metadata.get("config_sha256", "")
    if len(config_sha) != 64 or any(c not in "0123456789abcdef" for c in config_sha):
        raise ValueError("invalid config_sha256")
    if expected_config_sha256 is not None and config_sha != expected_config_sha256:
        raise ValueError("config_sha256 mismatch")
    dtype = metadata.get("dtype")
    if dtype not in _DTYPES or (expected_dtype is not None and dtype != expected_dtype):
        raise ValueError("bundle dtype mismatch")
    count = metadata.get("parameter_count", "")
    if not count.isdecimal() or int(count) <= 0:
        raise ValueError("invalid parameter_count")


def load_bundle(path: str | Path, expected_source_sha256: str | None = None,
                expected_dtype: str | None = None,
                expected_config_sha256: str | None = None) -> tuple[dict[str, mx.array], dict[str, str]]:
    """Load and validate the tensor set; keep MLX as the only tensor runtime."""
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError("bundle must be .safetensors")
    arrays, metadata = mx.load(str(path), return_metadata=True)
    validate_metadata(metadata, expected_source_sha256, expected_dtype, expected_config_sha256)
    if len(arrays) != int(metadata["parameter_count"]):
        raise ValueError("bundle parameter_count mismatch")
    for name, array in arrays.items():
        if str(array.dtype).removeprefix("mlx.core.") != metadata["dtype"]:
            raise ValueError(f"{name} dtype mismatch")
    return arrays, metadata


def load_model_weights(model, path: str | Path, expected_source_sha256: str | None = None,
                       expected_dtype: str | None = None,
                       expected_config_sha256: str | None = None):
    """Strictly install a validated bundle into a matching MLX model."""
    arrays, _ = load_bundle(path, expected_source_sha256, expected_dtype, expected_config_sha256)
    model.load_weights(list(arrays.items()), strict=True)
    mx.eval(model.parameters())
    return model

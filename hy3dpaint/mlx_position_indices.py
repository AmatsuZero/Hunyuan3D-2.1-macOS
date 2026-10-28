"""MLX producer for the multiresolution paint position voxel cache."""

import mlx.core as mx


def _compute_discrete_voxel_indices(position, valid, grid_resolution, voxel_resolution):
    if position.ndim != 5 or position.shape[2] != 3:
        raise ValueError("position maps must have shape [B, N, 3, H, W]")
    batch, views, _, height, width = position.shape
    if height % grid_resolution or width % grid_resolution:
        raise ValueError("position map dimensions must divide grid resolution")

    cell_height = height // grid_resolution
    cell_width = width // grid_resolution
    position = position.reshape(
        batch, views, 3, grid_resolution, cell_height, grid_resolution, cell_width
    ).transpose(0, 1, 3, 5, 2, 4, 6)
    valid = valid.reshape(
        batch, views, 3, grid_resolution, cell_height, grid_resolution, cell_width
    ).transpose(0, 1, 3, 5, 2, 4, 6)

    grid_position = position.astype(mx.float32).sum(axis=(-2, -1)).astype(mx.float16)
    count_masked = valid.sum(axis=(-2, -1))
    denominator = mx.maximum(count_masked, mx.array(1, dtype=count_masked.dtype))
    grid_position = grid_position / denominator
    threshold = cell_height * cell_width // (4 * 4)
    grid_position = mx.where(
        count_masked < threshold,
        mx.zeros_like(grid_position),
        grid_position,
    )

    grid_position = grid_position.transpose(0, 1, 4, 2, 3)
    grid_position = mx.clip(grid_position, 0, 1)
    voxel_indices = mx.round(grid_position * (voxel_resolution - 1)).astype(mx.int32)
    voxel_indices = voxel_indices.transpose(0, 1, 3, 4, 2)
    return voxel_indices.reshape(batch, views * grid_resolution * grid_resolution, 3)


def calc_multires_voxel_idxs_mlx(
    position_maps,
    grid_resolutions=(64, 32, 16, 8),
    voxel_resolutions=(512, 256, 128, 64),
):
    """Return the four cache entries consumed by the MLX inner UNet."""
    if not isinstance(position_maps, mx.array):
        raise TypeError("position_maps must be an MLX array")
    if len(grid_resolutions) != len(voxel_resolutions):
        raise ValueError("grid and voxel resolution counts differ")
    position = position_maps.astype(mx.float16)
    valid = mx.all(position != mx.array(1, dtype=position.dtype), axis=2, keepdims=True)
    valid = mx.broadcast_to(valid, position.shape)
    position = mx.where(valid, position, mx.zeros_like(position))
    result = {}
    for index, (grid_resolution, voxel_resolution) in enumerate(zip(grid_resolutions, voxel_resolutions)):
        indices = _compute_discrete_voxel_indices(
            position,
            # Torch's first pass zeros invalid pixels in-place. Later passes
            # therefore count those zero pixels as valid.
            valid if index == 0 else mx.ones(position.shape, dtype=mx.bool_),
            int(grid_resolution),
            int(voxel_resolution),
        )
        result[indices.shape[1]] = {
            "voxel_indices": indices,
            "voxel_resolution": int(voxel_resolution),
        }
    return result

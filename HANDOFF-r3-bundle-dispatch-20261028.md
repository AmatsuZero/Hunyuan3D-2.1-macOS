# R3 交接：inner UNet bundle dispatch slice（2026-10-28）

## 已完成

- `hy3dpaint/mlx_utils.py` 增加显式 opt-in 环境变量：
  `HUNYUAN_MLX_INNER_UNET_BUNDLE`、`HUNYUAN_MLX_INNER_UNET_SOURCE_SHA256`、
  `HUNYUAN_MLX_INNER_UNET_CONFIG_SHA256`。设置后使用 Torch-free
  `load_model_weights(..., strict=True)`；未设置时默认行为仍是 Torch checkpoint
  离线转换。
- `tests/test_mlx_dispatch_bundle.py` 覆盖 bundle/default 两个选择分支。

## 真实两步证据

外部目录：

- bundle：`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-bundle-dispatch-2step-01/`
- fresh checkpoint 对照：`.../r3-checkpoint-dispatch-2step-01/`

两次使用同一 fixed fixtures、初始 noise、guidance/reference/MVA scales、15-step
schedule、1 GiB MLX idle-cache threshold。两次 `reference_write`、`noise_pred_0/1`
和 `latents_0/1` 逐元素完全相同；bundle run 的 loader 日志明确为 `weights loaded
from bundle`，fallback=0，MLX dispatch enabled。

旧的 `full-hybrid-up0-loaded-01` capture 与本次 fresh run 的 `noise_pred` 不同，
虽 `reference_write` 相同；它不是当前 bundle parity 的对照，应以本次新跑的
checkpoint 目录为 paired baseline。

## 尚未完成

R3 仍只是 inner UNet 权重来源替换：reference write、首步和 scheduler 仍走
Torch，后续 MLX 每步仍有 `.cpu().numpy()` bridge 回 Torch。尚未满足“15 步不再
往返 Torch”的门槛，也未切换默认 dispatch。下一步应逐项迁移 outer reference/cache、
首步输入和 scheduler latent 更新，并为每步保留数值/设备 trace。

# R3 交接：MLX first-step opt-in 集成（2026-10-28）

总交接仍是 `HANDOFF-torch-mps-upgrade.md`。这是 R3 的首步集成切片，**完整 R3 尚未完成，默认路径未切换**。

## 已完成

- `hy3dpaint/mlx_utils.py` 增加显式 `HUNYUAN_MLX_FIRST_STEP=1`。在 outer Torch
  已完成 reference-write、condition cache 和 position cache 后，首个 generation
  inner UNet 调用消费 `_cache_constant_inputs` 的 MLX cache，直接走 MLX；默认仍是
  首步 Torch、后续 14 步 MLX。缺少 outer reference cache 时显式失败。
- `tests/test_mlx_first_step_dispatch.py` 验证 opt-in、默认分支和 `reset()` 后再次
  从首步开始；`probe_fixed_conditions.py` 报告 `first_step_mlx`。
- `tools/quality_demo/replay_mlx_inner_first_step.py` 用真实 cache、输入和 inner
  bundle 做单步 MLX replay；CPU backend 因 36.2 GB 临时 attention buffer 超过
  单 buffer 上限失败，GPU replay 通过。

## 真实证据

- 单步输入/cache：`r3-inner-first-step-01/`；固定 512px、6 视角、3 CFG 分支，
  inner sample `(36,12,64,64)`，DINO `(3,1028,1024)`，输出 `(36,4,64,64)`。
- GPU replay：`r3-inner-first-step-gpu-01.json`，finite=true，mean abs
  `0.00035177`，p99 `0.00219727`，max abs `0.02148438`，NRMSE `0.00082382`。
- 两步集成：`r3-mlx-first-step-2step-01/` 与
  `r3-mlx-first-step-compare-01.json`。初始 latent/reference-write 完全一致；
  首步 noise NRMSE `0.00082382`，第二步 `0.00085213`，第二步 latent `0.00047128`。
- 完整 15 步：`r3-mlx-first-step-full-01/`。Torch 2.14.1、fallback 环境变量未设
  （默认禁用）、inner
  bundle、`first_step_mlx=true`、12 张 PNG、elapsed `61.43s`（capture 诊断时间，
  不是正式性能结论）。与现有 Torch-first hybrid capture 的逐步比较在
  `r3-mlx-first-step-full-compare-01.json`：最终 latent NRMSE `0.00513835`，最终
  noise NRMSE `0.00947690`，cosine `0.99995509`。
- 12 张图像比较：`r3-mlx-first-step-image-compare-01.json`，平均像素绝对差
  `0.05082/255`、最大 `38/255`；首张正面保持黑 H/Y/D、红 3，未出现旧版全黑或
  候选染红设计变化。

## 边界与后续

首步仍依赖 outer Torch reference-write、DINO/condition preparation、CFG 和 Torch
scheduler；这只是 opt-in inner dispatch，不能宣称完整 MLX。需要继续把 outer cache
生产和 scheduler/CFG latent 更新迁移到 MLX，补齐 Metal trace、非诊断性能、内存和
新请求 reset 验收；默认仍不开启 `HUNYUAN_MLX_FIRST_STEP`。

遵守总交接 §3：一次只跑一个 GPU 推理；不覆盖已交付 GLB；不提交或清理根目录
`torchvision_fix.py`；不在原始 checkout 重编扩展。

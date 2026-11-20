# R3 交接：首步 MLX output 直通 resident loop（2026-10-28）

总交接仍是 `HANDOFF-torch-mps-upgrade.md`。本阶段只移除 resident loop
首步 inner UNet output 的 MLX→NumPy/Torch→MLX 往返；默认路径和 outer
Torch 条件准备没有切换。

## 已完成

- `MLXDispatchUNet._mlx_forward_raw()` 抽出原生 MLX output 路径。
- `HUNYUAN_MLX_RESIDENT_LOOP=1` 且 `HUNYUAN_MLX_FIRST_STEP=1` 时，首个
  inner UNet generation 直接返回 MLX array；普通 first-step 仍返回
  Diffusers 兼容的 `(torch_output,)`。
- `HunyuanPaintPipeline` 在 resident 模式接收该 MLX output，交给
  `run_mlx_denoise()`；只有 resident loop 完成后才把最终 latent 转回 Torch
  给现有 VAE/image postprocess。
- `_as_mlx()` 接受已有 MLX array，避免 resident 首步重复转换；形状、dtype
  和 CFG 分支合同继续 fail closed。
- 新增首步直通与 resident first-prediction 单元回归；默认 first generation
  的 Torch 行为保持覆盖。

## 真实固定条件证据

运行约束：单 GPU、`PYTORCH_ENABLE_MPS_FALLBACK=0`、512px/6 views/15 steps、
1 GiB MLX idle-cache threshold、固定条件和初始 latent。输出目录：

`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-mlx-direct-first-01/`

- exit 0，耗时 `63.044 s`，生成 12 张图；runtime source audit 通过。
- 45 个 resident trace 数组全部 finite，15 步 shape/dtype 合同通过。
- 与上一阶段 `r3-mlx-resident-loop-02` 的首步 `inner_output`、`noise_pred_0`
  以及 15 步 `resident_noise_pred`、CFG scheduler output、latent 均逐元素
  相同；两组 12 张 PNG 也逐元素相同。
- 与 Torch-first/MLX-inner 基线的既有完整对照保持：最终 latent NRMSE
  `0.00425918`，12 张图平均绝对差 `0.04558/255`，最大差 `30/255`。

复现时先确认 ComfyUI queue 为空、无其他 GPU 推理，并使用新的 output 目录；
下面的 `r3-mlx-direct-first-replay-01` 必须尚不存在：

```sh
rtk proxy curl -s --max-time 3 http://127.0.0.1:8188/queue
rtk proxy env PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  HUNYUAN_MLX_FIRST_STEP=1 HUNYUAN_MLX_RESIDENT_LOOP=1 \
  HUNYUAN_MLX_RESIDENT_TRACE_DIR=/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-mlx-direct-first-replay-01/resident-trace \
  /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python \
  tools/quality_demo/probe_fixed_conditions.py --stage capture \
  --fixtures /Users/daubert/Github/hy3d-upgrade-evidence/20261027/color-diagnosis/fixed-conditions-fixtures \
  --output-dir /Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-mlx-direct-first-replay-01 \
  --steps 15 --stop-after 0 \
  --initial-noise /Users/daubert/Github/hy3d-upgrade-evidence/20261027/color-diagnosis/initial-noise-torch251-seed0.npy \
  --guidance-scale 3 --reference-scale 1 --mva-scale 1 \
  --mlx --capture-cache --mlx-cache-limit-mib 1024
```

## 代码验证

- Metal 可用的候选 venv 全量 unittest：`65/65` 通过。
- `py_compile`：`mlx_utils.py`、`mlx_resident_denoise.py`、pipeline 和
  capture 工具通过。
- `git diff --check` 通过。
- 受限 shell 直接运行 MLX 会报 `No Metal device available`；这只是当前
  shell 的 Metal 访问限制，不能替代 Metal 可用环境的 65/65 结果。

## 边界与下一步

- 这仍不是全 MLX：outer reference-write、DINO、VAE、条件生成和首次输入
  的 Torch→MLX sample 转换仍在 Torch/MLX 混合边界；MLX checkpoint loader
  仍是 opt-in，默认路径未切换。
- 不应由本阶段宣称端到端性能提升；需要统一默认路径后的冷/热启动、内存和
  质量比较。
- 下一阶段优先移植并验证 outer DINO projector、normal/position condition
  准备和 reference cache 的 MLX consumer；每阶段使用 fresh output、单 GPU、
  固定条件和独立交接。

## 保护项

- 不覆盖已交付 GLB；不提交、还原或清理根目录 dirty `torchvision_fix.py`。
- 不在原始 checkout 重编原生扩展；远端只推送 `fork/codex/torch-mps-upgrade`。

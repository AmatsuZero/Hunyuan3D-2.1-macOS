# R3 交接：CFG + UniPC MLX replay（2026-10-28）

总交接仍是 `HANDOFF-torch-mps-upgrade.md`。这是 R3 的 CFG/scheduler 边界切片，**尚未接入正式 denoise loop**。

## 已完成

- `hy3dpaint/mlx_cfg.py` 实现当前 paint pipeline 的三分支 CFG：
  `uncond/ref/full`、camera azimuth scale、`pbr_views=2` 展开，保留输入 dtype。
- `tests/test_mlx_cfg.py` 对照 Torch 公式并覆盖错误 branch shape。
- `tools/quality_demo/replay_mlx_cfg_scheduler.py` 使用真实 15 步 capture 的
  pre-CFG noise prediction、scheduler sample 和 timesteps，在 MLX CPU/GPU 上重算
  CFG 与 UniPC latent 更新，拒绝覆盖报告。

## 真实证据

输入 capture：`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-unipc-capture-01/`，
`capture.json` SHA256 `e4427786960095f6797b38d286c5f039a5ea380e51cf1e3834d36968d16210f2`。

- GPU 报告：`r3-mlx-cfg-scheduler-gpu-01.json`；15 步，guidance scale `3.0`，
  `max_cfg_nrmse=0`，最大 scheduler NRMSE `0.00078931`，最终 `0.00041650`，全输出有限。
- CPU replay 报告在 `/private/tmp/r3-mlx-cfg-scheduler-cpu.json`，指标与 GPU 完全相同。
- 2026-10-28 磁盘清理后复验：工作卷可用 `117 GiB`；ComfyUI queue 的
  `queue_running`/`queue_pending` 都为空。MLX 相关回归 11/11 通过。新 GPU/CPU
  replay 分别写入 `/private/tmp/r3-mlx-cfg-scheduler-gpu-after-cleanup-20261028.json`
  和 `/private/tmp/r3-mlx-cfg-scheduler-cpu-after-cleanup-20261028.json`；两份报告的
  15 步指标完全相同，`max_cfg_nrmse=0`、最大 scheduler NRMSE `0.00078931`、
  最终 `0.00041650`。sandbox 内 MLX 即使选择 CPU 也无法初始化 Metal device；
  上述验证在已授权的 sandbox 外执行。
- 首次 replay 暴露并修复了相机 scale 将 float16 结果提升到 float32 的 dtype 问题；
  现在 MLX CFG 结果保持输入 dtype，scheduler 接口严格检查通过。

这证明 CFG 组合和 scheduler latent 更新的独立数值合同；它仍使用 Torch capture 产生的
pre-CFG noise prediction，不能证明正式循环已经不再经过 Torch 或 bridge。

## 后续

1. 在 opt-in latent loop 中让 MLX inner output 直接进入 MLX CFG 和 MLX UniPC，避免每步
   `Torch → NumPy → MLX → Torch`。
2. 逐步比较正式 pipeline 的 noise/latent、cache reset、Metal trace 和内存生命周期。
3. 完整 15 步质量与性能通过后才考虑默认切换。

遵守总交接 §3：一次只跑一个 GPU 推理；不覆盖已交付 GLB；不提交或清理根目录
`torchvision_fix.py`；不在原始 checkout 重编扩展。

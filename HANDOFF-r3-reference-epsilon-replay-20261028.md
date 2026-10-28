# R3 交接：reference epsilon 新代码 replay（2026-10-28）

总交接为 `HANDOFF-torch-mps-upgrade.md`；本阶段基于 `26f72a0`。本阶段完成了
opt-in reference epsilon 配置的真实冻结输入 replay，但未进行完整 15 步验收。

## 运行条件

- worktree：`/Users/daubert/Github/hy3d-torch-upgrade`，分支
  `codex/torch-mps-upgrade`。
- replay 前成功确认 ComfyUI `/queue` 返回
  `{"queue_running": [], "queue_pending": []}`；进程检查只发现既有 ComfyUI
  服务，没有 Hunyuan/MLX 推理进程。
- replay 使用新的 `replay_mlx_reference.py`，`--fp32`，冻结
  `occluded512/torch-reference` 输入，reference Transformer2D epsilon `1e-6`。

## 结果

报告：`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-reference-up2-stage-trace-01/hy3d-reference-eps-newcode-20261028.json`

- `inputs_exact=true`。
- checkpoint SHA256：`675a1b5cd0098b2002637c443946529c03c5cd54427f40245263350feb3dd5b8`。
- `mapped_weights=686`、16 层完整。
- `transformer_norm_eps=0.000001`。
- 最大 cache NRMSE：`0.0130166439`，失败（门槛 `0.01`）。
- reference write NRMSE：`0.0257892131`，通过（门槛 `0.03`）。
- 最坏 cache 层：`up_2_2_0`；其 NRMSE `0.0130166439`。
- 报告 SHA256：`3639232ed149c596a09d268a7e604ae55b41f1eae2996a5943dcbd183fcd9f73`。

验证器退出码为 1，且只因 cache 门槛失败；这不是 crash。与旧 epsilon replay
`0.0143295` 相比，`1e-6` 有小幅改善，但没有达到切换条件。未运行完整 15 步图像、
reset 或 GLB；默认 dispatch 仍未切换。

## 下一阶段

1. 保留 `1e-6` reference 配置，但继续追踪 `up_2_2_0` 之前更早的 down/mid/up0
   主干误差；重点检查高均值低方差 GroupNorm、linear、SDPA 的同输入对照。
2. 冻结正式 fp32 dispatch 与离线参数 fp32 两种输入/cache 合同，避免把单一 replay
   结果当成完整 resident 质量证据。
3. 只有 cache 通过 `0.01` 后，才运行完整 15 步、reset 和视觉对照；继续保持单 GPU、
   不碰 `torchvision_fix.py` dirty、原始 checkout、旧 venv 和已交付 GLB。

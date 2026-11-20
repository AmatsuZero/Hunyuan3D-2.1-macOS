# R3 交接：reference FF/GEGLU 误差归因（2026-10-28）

总交接为 `HANDOFF-torch-mps-upgrade.md`；本阶段基于 `7bd6f54`。本阶段只对上一
阶段冻结的 `up_1_1_0` trace 做 CPU 数值归因，没有 GPU 推理、没有改模型代码、门槛或
默认 dispatch。

## 证据

目录：`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-reference-up2-stage-trace-01/`

- `ff-exact-input.json`：同一 frozen FF 输入的 CPU Torch/MLX 公式对照。
- `hy3d_ff_geglu_attribution-20261028.json`：value/gate 替换归因。
- `hy3d_ff_projection_attribution-20261028.json`：同一 checkpoint `ff.net.0.proj`
  的 CPU 重放。
- 对应脚本为同目录的 `hy3d_ff_geglu_attribution.py` 和
  `hy3d_ff_projection_attribution.py`。

## 结果

- 当前失败 `occluded512` 的 `norm3` 输入 Torch fp32/MLX fp32 NRMSE 为
  `0.0042290795`。
- 用同一 checkpoint 的 `ff.net.0.proj` 在 CPU 重放这两个输入后，完整 projection
  NRMSE 为 `0.0026705757`；gate 半部 NRMSE 为 `0.0022470714`。
- Torch 捕获的 `ff_proj` 与 CPU 同输入重放 NRMSE `3.2e-7`，最大绝对差
  `1.14e-5`；MLX 捕获与 CPU 同输入重放 NRMSE `0.00035749`，最大绝对差
  `0.00838`。两边使用相同的 checkpoint 权重，未观察到明显权重缺失或布局错位。
- GEGLU 输出差异 NRMSE `0.0074724`。固定 value、只替换 gate 时 NRMSE
  `0.0071673`；固定 gate、只替换 value 时 NRMSE `0.0033802`。最大差异位于
  token/channel `[0,104,4380]`，gate 从 `2.3007` 变为 `2.4267`，GEGLU 从
  `24.9669` 变为 `26.2628`。
- CPU 公式与 Torch/MLX 各自记录的 GEGLU 输出误差分别为 `6.83e-8` 和
  `5.68e-7`。因此 GELU 公式本身不是当前主要触发点；差异主要由 `norm3` 输入
  经过 projection 进入 gate 后被放大。

## 结论边界

这项结果把问题从“FF 权重/GEGLU 实现错误”收窄到更早的 `up_1_1_0` hidden state
差异。它没有定位最初差异来自 `up_1.1` 的哪一个 attention 分支、skip residual、
GroupNorm 或 MLX SDPA。不得据此放宽 cache `0.01` / write `0.03`，也不得切默认
MLX reference forward。

## 下一阶段

1. 对 `up_1.1` 的 concat/resnet/Transformer2D 输入及 `up_1_1_0` 的 self-attention
   和 residual 逐段对照，确认 `norm3` 差异首次进入的具体分支。
2. 在单独的 frozen tensor probe 中对该分支的 GroupNorm、linear 和 SDPA 做同输入
   CPU/MPS/MLX 对照；每次只改一个算子或 dtype。
3. 保持单 GPU、默认 Torch reference、根目录 `torchvision_fix.py` dirty、原始
   checkout、旧 venv 和已交付 GLB 保护。阶段完成后写交接、验证、提交并推送。

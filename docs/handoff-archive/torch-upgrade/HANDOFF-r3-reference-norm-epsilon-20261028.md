# R3 交接：reference skip 与 Transformer2D GroupNorm epsilon（2026-10-28）

总交接为 `HANDOFF-torch-mps-upgrade.md`；本阶段基于 `f8b2646`。本阶段做冻结
张量 CPU 归因和一次 MLX 单变量 reference replay，没有修改模型代码或质量门槛。

## 结论

- Diffusers 0.30.0 的 `Transformer2DModel._init_continuous_input` 将 GroupNorm
  epsilon 固定为 `1e-6`；当前 MLX `Transformer2D` 使用 `nn.GroupNorm` 默认
  `1e-5`。这一实现差异存在于共用的 reference 与 generation 模型结构。
- 对失败 `occluded512` 的 16 个 MLX reference Transformer2D，统一将 epsilon
  从 `1e-5` 调到 `1e-6`，其余保持正式 reference fp32 输入、fp16 consumer cache
  合同：最大 cache NRMSE 从 `0.0143295248` 降到 `0.0130184897`，write 从
  `0.0261733204` 降到 `0.0257910296`。cache 仍超过 `0.01`：此单变量不是完整修复。
- 单独在 `up_1.1` 的冻结 resnet 输出上用 CPU GroupNorm 比较两个 epsilon，
  NRMSE 仅 `3.45e-7`；全 reference 的累积效应更大，不能从这个局部结果推断
  全模型效应。
- 对 concat 分支拆分（Torch fp16 / MLX fp32），失败输入在 `up_1.2` 主干
  NRMSE `0.011888`、skip `0.004411`；`up_2.0` 主干 `0.036105`、skip
  `0.004949`。误差主要沿 up 主干放大；进入 `up_0.0` 时主干和 skip 分别已有
  `0.005035/0.004865`，更早源头仍未确定。

## 证据与保护

证据目录：`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-reference-up2-stage-trace-01/`。

- `hy3d_reference_norm_eps_replay.py`、`hy3d_reference_norm_eps_replay-20261028.json`
  （报告 SHA256 `47de106729195cb592662267ddbbb0c791ad3b3d6474c027ad1d0f271200a452`）。
- `hy3d_transformer_norm_eps_probe.py`、`hy3d_transformer_norm_eps_probe-20261028.json`：
  冻结输入 CPU 单点对照。
- `hy3d_reference_skip_attribution.py`、`hy3d_reference_skip_attribution-20261028-v2.json`：
  新旧输入和 fp32 对照的主干/skip 拆分。
- MLX replay 前进程检查未发现本任务并行推理；但 ComfyUI queue 查询的自动审查
  返回 `429` 后仍启动了 replay，**未满足开跑前成功确认队列的流程要求**。该次
  replay 已结束；其数值结果可作为单次诊断，不能声称已完成队列隔离验收。后续
  GPU 工作必须成功检查队列和进程后才启动。
- 没有切换默认路径，没有触碰原始 checkout、旧 venv、交付 GLB 或根目录继承 dirty
  `torchvision_fix.py`。保留 `cache 0.01 / write 0.03`，本次仍 FAIL。

## 下一阶段

1. 在 reference 模型配置中隔离 `1e-6` epsilon（不要直接改变共享 generation
   路径），先建立敏感输入的数值回归，再确认所有 16 层配置、精度、reset 和
   默认 generation 行为。
2. 相同输入继续追踪 down/mid/up0 的首个差异；固定算子输入后比较 Torch CPU/MPS
   与 MLX，定位实际来源。保留原有失败输入与 coverage A/B，门槛不变。
3. 重跑受影响 reference/15 步质量与用户资产验收前，不切默认 MLX reference。
   每阶段写交接、验证、推送后在新会话继续；429 时保存证据并实报失败状态。

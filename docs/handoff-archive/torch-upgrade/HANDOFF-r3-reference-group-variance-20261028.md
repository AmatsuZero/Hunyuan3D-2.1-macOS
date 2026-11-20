# R3 交接：reference `up_1.1` GroupNorm 逐组方差（2026-10-28）

总交接为 `HANDOFF-torch-mps-upgrade.md`。本阶段基于 `0424a1a`，只读分析已有冻结 trace，未启动 GPU，也未改变模型代码或数值门槛。

## 输入与方法

证据目录：`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-reference-up2-stage-trace-01/`。比较 `torch-fp32-02/up_1.1.resnet.npy` 与 `mlx-04/up_1.1.resnet.npy`；两份 `trace.json` 的 reference sample、timestep、encoder hidden states 和 checkpoint SHA256 相同。两个数组均为 NCHW `(3,1280,16,16)`、float32。该位置是 `up_1.1` ResNet 输出，也是随后 Transformer2D GroupNorm 的输入。按 32 组、每组 40 通道，使用 NumPy float64 计算每个 batch/group 的均值、方差与标准差，避免 float16 reduction overflow。

## 结果

- Torch 最小组标准差 `2.9591695314`，位于 batch 0/group 5；三个 batch 在该 trace 中相同。最大 `abs(mean)/std` 为 `0.2542822875`，位于 group 24。
- 同位置 Torch/MLX 的最大逐组输入 NRMSE `0.0047140548`，位于 group 0；group 5 为 `0.0045664076`。这些是输入差异，不能当作 GroupNorm 算子误差。
- 对各自输入分别用同一个 float64 GroupNorm 公式计算，epsilon `1e-5` 的归一化输出 NRMSE 为 `0.00422435572`，epsilon `1e-6` 为 `0.00422435574`。这里只对照输入敏感性，未重放真实 GroupNorm 权重或 GPU kernel。

因此，早前“接近常数的合成输入使 MLX CPU GroupNorm 与 Torch CPU 相差约 `0.0175`”的现象，在这个真实 `up_1.1` GroupNorm 输入上没有低方差前提。当前不能据此归因或修改 GroupNorm 实现。`1e-6` 新代码 replay 仍为 cache `0.0130166 > 0.01`、write `0.0257892 < 0.03`。

## 下一阶段

1. 从相同冻结输入继续向更早的 `up_1.1.resnet` 前追溯：对 `up_1.1.concat`、ResNet 内部 norm/conv/shortcut 的同输入输出做分界 trace，找出首次明显放大的算子；同时核对 `up_1.0` 输出和 skip 两半。
2. 在已有 trace 内先做 CPU 离线计算。任何新的 GPU replay 前，必须成功确认 ComfyUI queue 空和无其他推理进程；检查受沙箱拒绝时不得启动。
3. 保持 cache NRMSE `0.01`、reference write `0.03`，默认 Torch reference，不扩大 outer condition/VAE 边界；保护根目录 `torchvision_fix.py`、原始 checkout、旧 venv 和已交付 GLB。

新会话 `01a18221-557e-73a2-90ae-634669be6bdc` 已创建但服务端随即返回 `429 Too Many Requests`，状态 `systemError`，没有执行诊断；本阶段因此留在当前会话完成。

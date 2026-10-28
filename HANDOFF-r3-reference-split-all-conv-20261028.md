# R3 交接：高通道卷积分块 reference replay（2026-10-28）

总交接为 `HANDOFF-torch-mps-upgrade.md`。本阶段基于 `20c956e`，仅用临时脚本
验证数值假设；没有修改默认模型或门槛，也没有运行完整 15 步图像/GLB 验收。

## 实验与隔离

在每次实际 GPU replay 前成功确认 ComfyUI `/queue` 为空，进程列表无其他推理；
一次只运行一个 MLX GPU 任务，1 GiB MLX idle-cache 上限。第一版临时脚本因
`dilation` 字段类型假设错误，在构建模块时退出，未做模型 forward；修正后重新
检查队列再运行。输入为冻结 `occluded512` resident reference，checkpoint SHA256
`675a1b5cd0098b2002637c443946529c03c5cd54427f40245263350feb3dd5b8`，
686 权重、16 层 reference、参数 fp32、输入保持原 capture dtype、Transformer
epsilon `1e-6`。

临时脚本把 65 个输入通道数至少 320 且能被 4 整除的 ResNet、上下采样和输出
Conv2d 各拆成 4 个输入通道卷积，分别无 bias 后求和，再加原 bias。它只用于
诊断，没有接入 `hy3dpaint` 运行时。脚本及完整报告在外部证据目录
`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-reference-down0-resnet-01/`：
`hy3d_reference_split_all_conv_replay.py`、`split-all-conv-replay.json`。报告
SHA256 为 `9c6f2bd0a4801d5d77d2ff017bae1e19fe4a42e673c09100183dfa133889e5df`。

## 结果

`validate_mlx_reference_report.py` 使用原 cache `0.01`、write `0.03` 门槛
返回 exit 0、`passed=true`：

- 最大 cache NRMSE `0.0092018083`，位于 `up_2_2_0`；16 层全在门槛内。
- reference write NRMSE `0.0158142095`，在门槛内。
- 上一单点 conv1 4 块 replay 为 cache `0.0147171041`（失败）、write
  `0.0238398595`；未分块的新 epsilon replay 为 cache `0.0130166439`（失败）、
  write `0.0257892131`。

这个结果证明多处 MLX Metal Conv2d 数值误差共同影响 reference 路径；
只修首个 conv1 不够。它只覆盖一个冻结输入和临时 65 层替换，不证明其他
resolution/遮挡、完整 15 步颜色、reset 或性能，不能据此切默认 MLX reference。

## 下一阶段

1. 先收窄 65 处替换的必要集合，记录各 Conv2d 的输入通道/核/stride 与局部误差；
   优先尝试仅高误差形状的 opt-in 改法，避免无条件付出 65 次分块性能成本。
2. 对至少原先失败的 512 遮挡和 384 reference 做完整 16 层 cache/write replay，
   保持 `0.01/0.03` 原门槛；再做 15 步固定条件、reset、颜色与端到端性能/内存。
3. 每次 GPU 前成功检查 queue 和进程，保持单 GPU、1 GiB cache、fresh output；
   保护 `torchvision_fix.py`、旧 venv、原始 checkout、已交付 GLB。

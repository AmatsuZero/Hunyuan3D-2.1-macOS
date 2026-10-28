# R3 交接：MLX Metal `conv1` 分块探针与单点 replay（2026-10-28）

总交接为 `HANDOFF-torch-mps-upgrade.md`。本阶段基于 `4e41650`，没有把临时分块
实现接入默认模型，也没有改变 cache/write 门槛。

## 最小卷积探针

输入为已冻结的 `down_0_resnet_0.norm1`，权重为同一 checkpoint 的
`down_blocks.0.resnets.0.conv1`，MLX 参数使用 fp32、OHWI 布局，cache limit 为
1 GiB。直接 `mlx.nn.Conv2d(320,320,3)` 的 GPU 输出与完整 MLX trace 的
`conv1` 逐元素相同（NRMSE `0`）；相对 Torch MPS reference 的 NRMSE
`0.0020604500`。

把输入通道拆成多个独立卷积、去掉每块 bias 后求和并只加一次总 bias，得到：

| 输入通道块数 | NRMSE vs Torch MPS |
| ---: | ---: |
| 1 | `0.0020589877` |
| 2 | `0.0020589889` |
| 4 | `4.90e-7` |
| 5 | `0.0020589895` |
| 8 | `4.24e-7` |
| 10 | `0.0020589928` |
| 16 | `3.91e-7` |
| 20 | `3.86e-7` |
| 32 | `3.76e-7` |
| 40 | `3.74e-7` |
| 64 | `3.70e-7` |

这说明误差与完整卷积的 Metal 执行形状有关；4、8 或更高块数能把该局部误差降到
Torch/MPS 数值量级。探针源码和摘要在外部证据目录
`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-reference-down0-resnet-01/`，
摘要 SHA256 为 `a48d80ef5973e687f195415fc60a9712cff966a4fc14d76e3074f6c801b58812`。
第一次分块探针未重新做 queue 检查；随后在成功 queue/进程 preflight 后用相同脚本
复跑，表中数值来自后一次运行。

## 单点完整 reference replay

临时只替换第一个 `down_0.resnets[0].conv1` 为 4 块输入通道累加，其他路径保持
MLX reference fp32 参数、原始输入合同、Transformer epsilon `1e-6`。在成功确认
ComfyUI queue 空和无其他推理后完成 16 层 replay：

- 最大 cache NRMSE `0.0147171041`（`up_2_2_0`），仍失败 `0.01`。
- reference write NRMSE `0.0238398595`，通过 `0.03`。
- 未替换的同 epsilon replay 为 cache `0.0130166439`、write `0.0257892131`。

局部 conv1 误差确实被修正，但最终 cache 误差上升，不能把这个单点实现作为修复
或接入默认路径；它只证明卷积形状是一个真实的局部数值因素，仍有其他层/路径误差。

## 下一阶段

1. 不扩大修改范围，先对其他高通道 MLX Conv2d（`down_0` 后续 ResNet、upsampler）
   做同样的最小形状探针，建立形状与误差的关系。
2. 任何候选 split 实现必须同时通过局部 CPU/MPS 对照和完整 cache `0.01`、write
   `0.03`；单点 NRMSE 改善不足以切默认。
3. GPU 前成功确认 queue 空、无其他推理，保持单 GPU、1 GiB cache、fresh output；
   保护 `torchvision_fix.py`、原始 checkout、旧 venv 和已交付 GLB。

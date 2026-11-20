# R3 occluded512 conv1 policy/source 核查（2026-10-29）

本阶段没有启动 GPU；只审查已归档的真正 step 1 证据、MLX convolution
dispatch 源码和当前 Python wheel 版本。上一阶段的首步边界保持：`conv_in` 与
`down_0_resnet_0.norm1` exact，最早差异是 `down_0_resnet_0.conv1`，
disabled policy 更接近 CPU oracle。

## MLX dispatch 机制

本机可读的 MLX Swift 源码 checkout：
`/Users/daubert/Github/ondevice-ai-kit/.worktrees/foundation/apple-mlx/.build/checkouts/mlx-swift/`

- `Source/Cmlx/mlx/mlx/backend/metal/conv.cpp` 的 `dispatch_conv_2D_gpu`
  对非分组卷积在满足以下条件时尝试 Winograd：stride/dilation=1、3×3、
  输入/输出通道均为 32 的倍数、输入空间 batch 足够大且通道总量至少 256。
- `winograd_batch_step()` 用 `MLX_CONV_WINOGRAD_WORKING_SET` 覆盖 Metal
  recommended working set；budget 为 working set 的 3/4。设置为 `1` 时
  budget 不足，函数返回 0，dispatch 随后继续 implicit GEMM 路径。
- `mlx.nn.Conv2d` 本身统一调用 `mx.conv2d`；仓库的 `conv_in` 与
  `down_0_resnet_0.conv1` 都没有第二套实现。权重仍由同一 OIHW→OHWI
  转换加载。

## 版本边界

- 当前运行环境：Python wheel `mlx==0.32.4`、`mlx-metal==0.32.4`。
- 可读源码 checkout 为 `0.32.3-dirty`（commit
  `19601207e9a0de51e03ee6ec0c3c5f3784275075`），因此源码条件与环境变量
  语义是机制参考，不是当前 0.32.4 binary 的逐字节来源证明。
- 直接从 `core.cpython-311-darwin.so` 提取字符串未得到该环境变量，不能用
  binary strings 反推实现版本；实测 A/D oracle 与完整 capture 才是当前
  0.32.4 的有效证据。

## 结论边界

结合真正 step 1 的 CPU oracle（`conv1-cpu-oracle.json` SHA256
`2951471958fd856484da51f384d7ed6ed9488ca5cf553be0f24965408b16df3b`）和
源码机制，当前最强结论是：A/D 的首个数值分叉发生在符合 Winograd 条件的
320→320、3×3 `down_0_resnet_0.conv1` 卷积路径；`working_set=1` 的 implicit
GEMM 结果更接近 CPU。尚未证明 0.32.4 的具体 Metal kernel 实现有 bug，也
没有建立该局部差异与最终 `43/255` 像素的直接因果。

默认 Torch reference、默认 Winograd 和默认 MLX dispatch 继续不切换；
`43/255` occluded512 探索性图像门槛继续失败。下一步若继续实验，应优先
在同一 frozen `norm1` 输入上做完整 spatial/channel conv1 oracle 或获取
0.32.4 对应的官方 source tag，再决定是否值得窄范围 kernel 复验。

## 变更与验证

本阶段仅新增本交接文档；工作树代码无改动。上一阶段 fresh 全量 unittest
为 `88/88 OK`，本阶段未重复 GPU 或全量回归。受保护的
`torchvision_fix.py` 未提交，哈希保持
`d6139690b099ca3ecef80ddf3f1a25edfdf57f027c79bf8655a8f8aac1c8b493`。

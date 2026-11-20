# R3 reference producer 精度/后端差分（2026-10-29）

本阶段补齐了上一交接要求的三路 reference producer 对照：当前 C（MLX fp32）、
当前 D（Torch fp16）以及 Torch fp32 控制。三路均使用同一 `occluded512` frozen
reference 输入、同一 checkpoint 和全新的临时输出；本阶段没有生成图像，没有改默认
dispatch，也没有修改模型代码。

## 固定合同

- `reference_sample`、`reference_timestep`、`reference_encoder_hidden_states` 的
  SHA256 在三路均一致；完整哈希保存在各 `trace.json` 与摘要中。
- checkpoint SHA256：`675a1b5cd0098b2002637c443946529c03c5cd54427f40245263350feb3dd5b8`；
  `unet_dual` 共有 686 个权重，checkpoint 中均为 Torch `float16`。
- MLX：reference fp32、`Transformer2D` epsilon `1e-6`；Torch 控制分别为 fp16/fp32。
- MLX 复跑显式设置 `MLX_CONV_WINOGRAD=0`、`MLX_CONV_WINOGRAD_WORKING_SET=1`，
  并与前一份 MLX fp32 trace 的 52 个 `.npy` 逐元素 exact。
- 本阶段 GPU 前确认 ComfyUI `queue_running=[]`、`queue_pending=[]`，无本任务残留
  推理进程；三路没有并行运行。

证据归档：
`/Users/daubert/Github/hy3d-upgrade-evidence/20261029/r3-reference-producer-precision-20261029/`

- 215 个文件逐文件 SHA256 exact。
- manifest SHA256：`d8d7d053f366ba87a06fbe6f7df2316861746b8ef4c4f13e231761bd34fdf552`。
- precision summary SHA256：`3b2d0c490fc0fd40e26b06638ddc3921ef5b5ba51c934896a62dfaa16a2723ee`。
- trace 工具副本：`trace_mlx_reference_up2.py`。

## 结果

| 探针 | Torch fp16 → fp32 | Torch fp32 → MLX fp32 | 解释 |
| --- | ---: | ---: | --- |
| `conv_in` | `1.95e-4` | `0` | 空间输入卷积在 fp32 中 exact |
| `down_0_resnet_0.norm1` | `2.94e-4` | `5.93e-8` | 仅舍入级差异 |
| `down_0_resnet_0.conv1` | `2.72e-4` | `9.37e-7` | 当前 C/D 差主要不是该卷积 backend 差异 |
| `down_0_resnet_0.time_emb_proj` | `4.75e-4` | `6.34e-4` | 独立时间嵌入支路出现可见差异 |
| `down_0_resnet_0.conv2` | `5.36e-4` | `5.23e-4` | 首个后续空间卷积已有差异 |
| `down_0_resnet_1` | `1.15e-3` | `2.09e-3` | block 内差异继续放大 |
| `down_1` | `1.80e-3` | `4.62e-3` | 深层累计差异 |

`down_0` 聚合输出的 Torch fp32 → MLX fp32 NRMSE 为 `4.14e-3`；这不是首个算子
边界，因为它包含多个子模块。精细探针显示 `norm1/conv1` 在 fp32 中接近一致，
但尚未捕获 `time_embedding` 输入、`time_emb_proj` 输入和逐算子线性输出，因此
不能把 `time_emb_proj` 或 `conv2` 直接认定为根因。

## 结论与限制

当前 C/D 的 cache 差异同时包含 checkpoint 原生 fp16 与 MLX fp32 的精度因素；
把 Torch 提升到 fp32 后，首个空间 conv1 的差异降到 `9.37e-7`，说明旧的“两端
fp32、默认 Winograd” trace 不能直接解释当前 C/D。另一方面，Torch fp32 与 MLX
fp32 在 `down_0` 后仍会分叉，说明仅归因于 fp16 也不充分。此次证据把下一定位
边界缩到时间嵌入支路和首个 ResNet 后半段，尚未证明它们足以解释最终 image 5/11。
本 trace 工具只报告 cache 层数与 reference write 形状，不保存两者完整数值；因此
本阶段不能声称 replay 的最终 16 层 cache 已逐元素匹配正式 C/D capture。这一消费
边界验真仍需在后续增加输出捕获时补齐。

## 下一阶段

在保持相同 frozen input、fp32 两端和 disabled Winograd 的条件下，为
`time_embedding.linear_1/linear_2`、`temb`、`time_emb_proj` 增加输入/输出探针，
并用同权重 CPU float32 oracle 区分线性层布局、激活和 MPS/Metal kernel 差异。先
做首个 `down_0_resnet_0` 的局部合同，再决定是否需要完整 16 层 replay；不切换
默认路径。每次仍只运行一个 GPU 任务、使用 fresh output，并在阶段结束写交接、推送
和创建下一会话。

## 验证

- MLX 显式 disabled replay 与既有 MLX fp32 trace：52/52 `.npy` exact。
- 三路 trace 维度、有限值、输入/权重 SHA256 均通过；摘要由独立 venv Python 生成。
- 外部归档 manifest：215/215 SHA256 exact。
- 受保护 `torchvision_fix.py` 未改动；本阶段无 GLB、原始 checkout 或默认代码变更。
- 新接力会话两次因平台 429 终止；本阶段由当前会话在单 GPU 约束下完成。

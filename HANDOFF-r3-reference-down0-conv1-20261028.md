# R3 交接：reference 首次分叉定位到 MLX Metal `down_0` conv1（2026-10-28）

总交接为 `HANDOFF-torch-mps-upgrade.md`。本阶段基于 `7f6c6da`，只增加可选的
`down_0` 和首个 ResNet 内部 trace 点，默认 dispatch、reference cache 门槛和
已交付资产均未改变。

## 隔离与证据

每次 GPU trace 前都成功确认 ComfyUI `/queue` 为
`{"queue_running": [], "queue_pending": []}`，进程列表只有既有 ComfyUI 服务；
Torch MPS 与 MLX Metal 串行运行，MLX idle-cache 上限为 1 GiB。两端使用相同的
冻结 `occluded512` resident 输入与 checkpoint，fp32 参数/输入，MLX reference
Transformer2D epsilon `1e-6`；两端均完成 16 层 cache。

关键数组、`trace.json`、compare 报告和 CPU 归因报告已保存在外部新目录：
`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-reference-down0-resnet-01/`。
对照报告 `compare.json` 中的 NRMSE 相对 Torch RMS。

| 边界 | NRMSE |
| --- | ---: |
| `conv_in` | `0` |
| `down_0_resnet_0.norm1` | `0.000000059` |
| `down_0_resnet_0.conv1` | `0.002058988` |
| `down_0_resnet_0.time_emb_proj` | `0.000633688` |
| `down_0_resnet_0.norm2` | `0.002104856` |
| `down_0_resnet_0.conv2` | `0.002962613` |
| `down_0_resnet_0` | `0.002159596` |
| `down_0_attn_0` | `0.003416365` |
| `down_0_resnet_1` | `0.004514267` |
| `down_0_attn_1` | `0.005624992` |
| `down_0` | `0.004971313` |

首个显著跳变在 `down_0.resnets[0].conv1`，而其前面的 `norm1` 仅有舍入级差异。
对已捕获的两份 `norm1` 输入分别执行同 checkpoint 权重的 CPU Torch
`conv2d(silu(norm1))`：两份 CPU 输出的 NRMSE `1.416e-7`；Torch MPS 捕获与其
CPU 重放 NRMSE `9.425e-7`；MLX Metal 捕获与对应 CPU 重放 NRMSE
`0.002060445`，最大绝对差 `0.079543`。另一次 MLX CPU 同输入重放与 Torch CPU
仅差约 `1.052e-6`。实际加载到 MLX 模型的 `conv1` weight/bias 与 checkpoint
按 OHWI 布局转置后逐元素相同，weight/bias 最大绝对差均为 `0`，共映射 686 权重。

因此本阶段已将原始数值分叉定位到 **MLX Metal 的首个高通道 Conv2d 执行**，
而不是 checkpoint 错配、`conv_in` 布局、`norm1` 或该卷积的输入变化。这个局部
定位尚不能证明单个卷积误差足以解释最终 cache `0.0130166 > 0.01`；不得因此
切默认 reference 或放宽门槛。

## 下一阶段

1. 用同一冻结输入和权重做最小 MLX GPU Conv2d probe，核对直接调用的输出是否
   复现 `0.00206`，调查 MLX Metal 卷积的数值模式或可替代实现；一次只改一个变量。
2. 若找到局部改法，先以首个 conv1 同输入 CPU/Torch 对照验证，再完整 replay
   16 层 cache/write；原门槛仍为 `0.01/0.03`，之后才考虑 15 步质量验收。
3. GPU 前必须成功确认 queue 空和无其他推理；保持 1 GiB MLX idle-cache 上限、
   单 GPU、fresh output。保护根目录 `torchvision_fix.py`、旧 venv、原始 checkout
   和已交付 GLB。

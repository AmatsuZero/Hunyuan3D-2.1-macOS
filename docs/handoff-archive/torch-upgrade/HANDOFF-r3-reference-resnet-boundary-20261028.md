# R3 交接：reference `up_1.1` ResNet 边界追踪（2026-10-28）

基于 `45117c5` 与同一冻结 reference 输入；只读使用已有 Torch/MLX trace，未启动 GPU、未修改默认路径或门槛。

## 证据

目录：`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-reference-up2-stage-trace-01/`。比较 `torch-fp32-02` 和 `mlx-04`，二者输入、checkpoint 和 16 层映射合同一致。

对相同位置的数组计算 float64 NRMSE（相对 Torch RMS）：

| 边界 | NRMSE |
| --- | ---: |
| `up_1.0.resnet` | 0.004372 |
| `up_1.1.concat` 整体 | 0.004855 |
| `up_1.1.concat` 主干半部 | 0.004952 |
| `up_1.1.concat` skip 半部 | 0.004619 |
| `up_1.1.resnet` | 0.004198 |
| `up_1.1.attn` | 0.007576 |
| `up_1_1_0.norm1` | 0.004264 |
| `up_1_1_0.attn1` | 0.004089 |
| `up_1_1_0.norm2` | 0.004065 |
| `up_1_1_0.attn2` | 0.000000 |
| `up_1_1_0.norm3` | 0.004229 |
| `up_1_1_0.ff` | 0.007635 |
| `up_1_1_0.output` | 0.007837 |

`up_1.1.concat` 在进入 ResNet 前已带有约 `0.0046–0.0050` 的差异；ResNet 输出没有放大，反而为 `0.00420`。更明显的放大出现在 `up_1.1` attention 输出及其 Transformer FF。`attn2` 的零差异与 reference attention/cache 合同一致，不能据此推断整个 block 无误差。

## 结论与边界

真实 `up_1.1` 输入没有 GroupNorm 低方差异常，且 ResNet 不是当前首次放大点。下一优先级应追踪 `up_1.0` attention 输出到 concat 主干、skip 的来源，以及 `up_1.1` attention1/FF 的同输入算子和布局；不要修改 GroupNorm epsilon 或默认 dispatch 以外推断。

仍保持 cache NRMSE `0.01`、reference write `0.03`；最近正式 epsilon `1e-6` replay 为 cache `0.0130166`（fail）、write `0.0257892`（pass）。

新会话 `01a1822c-3056-76a2-a4b1-6e5d0dd6ccc2` 已按用户授权请求，但服务端返回 `429 Too Many Requests`，未执行任务；当前会话继续记录本阶段。

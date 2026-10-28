# R3 交接：reference coarse boundary 首次分叉（2026-10-28）

基于 `cdaf701`。本阶段给 `trace_mlx_reference_up2.py` 和 MLX model 增加了可选的
`conv_in`、`down_0`、`down_1`、`down_2`、`down_3`、`mid` trace 点；默认未设置回调
时计算路径不变。没有改变 cache 门槛、默认 Torch reference 或已交付资产。

## 运行与证据

在成功确认 `http://127.0.0.1:8188/queue` 返回
`{"queue_running": [], "queue_pending": []}`，且进程列表只有既有 ComfyUI 服务后，
串行完成同一冻结输入的 Torch fp32 和 MLX fp32 reference capture。两次输入哈希、
checkpoint SHA256、16 层 cache 和 reference write shape 均一致。新 trace 位于：

- Torch：`/private/tmp/hy3d-r3-coarse-torch-20261028`
- MLX：`/private/tmp/hy3d-r3-coarse-mlx-cache1g-20261028`
- compare：`/private/tmp/hy3d-r3-coarse-compare-cache1g-20261028`

可审阅的 compare 摘要已复制到
`docs/r3-reference-coarse-boundary-20261028.json`（SHA256
`8b95cc497afa223262bafa247a1ed8a1ca407cfafd96aec7744199c71e25b084`）；原始数组仍只在
临时证据目录。
MLX trace 入口显式设置 1 GiB idle-cache 上限；更早一次未显式设置上限的
capture 与本次 compare 数值相同，但不作为本阶段隔离证据。

## 结果

相对 Torch RMS 的 NRMSE：

| 边界 | NRMSE |
| --- | ---: |
| `conv_in` | `0.000000` |
| `down_0` | `0.004971` |
| `down_1` | `0.004918` |
| `down_2` | `0.005368` |
| `down_3` | `0.005388` |
| `mid` | `0.005589` |
| `up_0.output` | `0.006143` |
| `up_1.1.attn` | `0.006511` |
| `up_1.2.resnet` | `0.009876` |
| `up_2.0.resnet` | `0.022731` |

因此在这份真实 resident 输入上，`conv_in` 权重和布局对照逐元素一致；首次可观测
的后端差异在 `down_0` 内部，之后主要是传播与逐步放大。此前追到 `up_1.1` 的
FF/GEGLU 仍是放大点，但不是原始分叉。

## 验证与限制

- `py_compile` 对修改的 trace/model 文件通过。
- 普通沙箱完整 unittest 启动时因 MLX nanobind 重复注册终止：`Critical nanobind
  error: refusing to add duplicate key "cpu" to enumeration "mlx.core.DeviceType"`，
  进程 signal 6。随后在正常 Metal 权限下重跑同一套 unittest，`85/85` 通过。
- 本阶段未运行第二个并发 GPU 任务；未覆盖任何 GLB，也未触碰根目录
  `torchvision_fix.py`。

## 下一阶段

1. 继续只读拆分 `down_0` 的两个 ResNet 与 attention：记录 `down_0` 每个 resnet、
   attention 输入/输出及 transformer 内部边界，找出 `conv_in → down_0` 首次算子。
2. 新 GPU capture 前仍须成功确认 queue 空和无其他推理；若沙箱拒绝检查，不启动。
3. 保持 cache `0.01`、write `0.03`、默认 Torch reference，保护旧 venv、原始 checkout、
   已交付 GLB 和 `torchvision_fix.py`。

本阶段新会话入口曾有多次 429；交接已写入当前 worktree，提交和推送后再重试创建下一会话。

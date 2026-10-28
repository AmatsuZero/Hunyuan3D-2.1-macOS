# R3 交接：reference up block 逐层 stage trace（2026-10-28）

总交接仍为 `HANDOFF-torch-mps-upgrade.md`。本阶段在 `f8a909f` 基础上做了
reference `mode="w"` 的逐层诊断，目标是解释 resident `occluded512` 的 cache
门槛失败。没有改默认 dispatch、门槛、模型权重或交付 GLB。

## 运行约束与代码

- worktree：`/Users/daubert/Github/hy3d-torch-upgrade`，分支
  `codex/torch-mps-upgrade`。
- 多次只读检查的 ComfyUI queue 均为
  `{"queue_running": [], "queue_pending": []}`；Torch 与 MLX capture 串行执行。
  单元测试前的队列检查因自动审查超时没有执行，测试本身已完成。
- 根目录 `torchvision_fix.py` 仍是继承 dirty 文件，SHA256
  `d6139690b099ca3ecef80ddf3f1a25edfdf57f027c79bf8655a8f8aac1c8b493`，没有提交或清理。
- 新诊断入口：`tools/quality_demo/trace_mlx_reference_up2.py`。
  `hy3dpaint/mlx_unet.py` 只增加可选 `_reference_debug_trace` 回调；回调未设置时
  计算路径不变。捕获点覆盖 `up_block_0`、`up_block_1`、`up_block_2`，以及
  `up_1_1_0` 的 `norm1/attn1/norm2/attn2/norm3/ff_proj/ff_geglu/ff/output`。

## 证据

外部证据目录：`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-reference-up2-stage-trace-01/`
（原始临时目录为 `/private/tmp/hy3d-r3-up2-trace-20261028/`，两者已用 `ditto`
复制，目录大小约 1.7 GiB）。

- 当前失败输入 `occluded512`：Torch fp16 `torch-04`、Torch fp32 `torch-fp32-02`、
  MLX fp32 `mlx-04`；对照报告为 `compare-fp32-02/trace.json`，SHA256
  `5c1e2f622dfe27a916bb3138664c996274e51e842c960f4d57c43bd0011b584e`。
- 旧 coverage 精确输入：Torch `torch-old-03`、MLX `mlx-old-03`，对照报告
  `compare-old-03/trace.json`，SHA256
  `f7fd040bafeb4f7a2137e030ff68936744c8f801020e642f7c398c878df4b95a`。
- 当前失败输入的 Torch fp16/MLX fp32 对照报告 `compare-03/trace.json`，SHA256
  `c529e5767527b296017a565e9c89e714f82331b0e44a052e7fda8185dfc3781e`。
- 同一输入、同一 checkpoint 的 FF 精确输入 CPU Torch/MLX 对照：
  `ff-exact-input.json`，SHA256
  `e2dc6320cd01cc0197974e8100d42310224b816a11cf47e12c9ff8580e3e0db4`。
  （该报告只做离线算子对照，不代表完整 GPU forward。）

每组 Torch/MLX trace 的 reference sample、timestep、encoder hidden states SHA256 均一致；
checkpoint SHA256 均为
`675a1b5cd0098b2002637c443946529c03c5cd54427f40245263350feb3dd5b8`。
新旧 sample 的 `up_2_0_0.norm1` trace 与各自历史 capture 在 cast 后逐元素一致，
证明观测点和布局没有错位。

## 结果与解释边界

当前失败 sample 的 Torch fp32 → MLX fp32 NRMSE：

| boundary | NRMSE |
| --- | ---: |
| `up_0.output` | 0.006795 |
| `up_1.1.resnet` | 0.004198 |
| `up_1.1.attn` | 0.007576 |
| `up_1.2.resnet` | 0.012312 |
| `up_1.upsample` | 0.019100 |
| `up_2.0.resnet` | 0.029022 |
| `up_2_0_0.norm1` | 0.006812 |

在 `up_1_1_0` 内部，`attn2` 是逐元素相同；`norm3` NRMSE `0.004229`，
`ff_proj` `0.002674`，经过 GEGLU 后 `ff_geglu` `0.007472`，FF 输出 `0.007635`。
这说明该 FF/GEGLU 路径会放大上游微小差异，但尚未证明 GELU 或 MLX kernel 是原始
触发点。旧 coverage 同一边界的 `up_1_1_0.ff` 为 `0.004472`、`output` 为
`0.005139`，因此放大明显依赖输入内容。

Torch fp16 → MLX fp32 的当前失败 sample 中，`up_1.2.resnet` 为 `0.021929`、
`up_1.upsample` 为 `0.036105`、`up_2.0.resnet` 为 `0.055230`；Torch fp32
并没有消除差异（对应 `0.012312/0.019100/0.029022`）。因此不能把问题归结为
单纯输入 dtype 或 MPS fp16。

FF 精确输入离线对照中，同一 fp32 输入和 fp16 checkpoint 参数提升到 fp32 后，
CPU Torch 与 MLX FF NRMSE 约 `0.00039–0.00094`；完整 forward 的 FF 差异更大，
支持“上游差异被放大”的判断。该结果不足以授权任何容差放宽或默认切换。

## 验证

- `py_compile`：`hy3dpaint/mlx_unet.py`、`tools/quality_demo/trace_mlx_reference_up2.py`
  和临时 FF probe 通过。
- `git diff --check` 通过。
- 完整 unittest：`83/83`，`Ran 83 tests in 0.849s`，`OK`。

## 下一阶段

1. 保持 cache `0.01` / write `0.03` 不变，继续向 `up_1.1` 的 attention 输入及
   residual/skip 来源追踪；优先比较 `up_1.1.attn` 输出与 `up_1.2.concat` 输入的空间热点，
   再决定是否需要逐算子复现 GroupNorm/linear/GELU。
2. 若要测试 GELU，必须使用同一 frozen `ff_proj` 输入分别对照 Torch CPU/MPS、
   MLX，并记录输入与权重 dtype；不能用单一完整 forward 结果归因。
3. 修复前不切默认 MLX reference forward，不扩大 outer condition/VAE 边界；保留
   原始 checkout、旧 venv、已交付 GLB 和 `torchvision_fix.py` 保护。

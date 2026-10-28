# R3 交接：Winograd fallback opt-in dispatch（2026-10-28）

总交接为 `HANDOFF-torch-mps-upgrade.md`。2026-10-29 补充了 reference384
独立输入证据。本阶段在 `98a75f5` 之后实现了
`HUNYUAN_MLX_DISABLE_WINOGRAD=1` 的 opt-in 配置，并在真实 resident loop 上
做了一个遮挡 512 输入的完整验证。默认 dispatch 不改变。

## 代码与合同

- `hy3dpaint/mlx_utils.py` 新增 `_configure_mlx_conv_policy()`。
  `HUNYUAN_MLX_DISABLE_WINOGRAD=1` 且未设置冲突的底层变量时，在 MLX 模型
  构建和权重 materialize 前设置 `MLX_CONV_WINOGRAD_WORKING_SET=1`；该值
  使 C++ `winograd_batch_step()` 走 implicit GEMM fallback。若同时设置其他
  working set，函数 fail closed；未启用 opt-in 时不修改环境。
- dispatch 保存实际 policy，后续 capture report 会记录 `conv_policy`；本阶段
  完整 15 步报告在该记录字段接入前生成，日志已显示 opt-in 启用。默认路径
  仍不设置该变量。
- `tests/test_mlx_conv_policy.py` 以 TDD 验证三个合同：opt-in 设置值、冲突
  值拒绝、默认不改环境。

## 真实完整 loop 证据

每次 GPU 运行前均确认 ComfyUI `/queue` 的 running/pending 为空，并确认没有
Hunyuan/MLX/Torch 推理进程；两个 replay 串行运行。固定输入为
`r3-reference-coverage-01/occluded512/fixtures`，generation fp16，reference
fp32，1 GiB MLX idle-cache，DINO projector/position indices/reference cache/
first-step/resident loop 全部 opt-in，15 steps，12 views。关键输入张量与既有
Torch reference 的 8 类文件逐元素相同。

外部证据目录：
`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-reference-winograd-optin-01/`

- `capture.json` + `resident-trace/`：单次完整 loop；capture SHA256
  `349c097db06d1c4b5f241d11ec6fdd8c8b4db5796558f7327d02b371eead7c34`。
- `repeat/capture.json`：同一进程第二请求与 reset；SHA256
  `bce8a9399838e0db7ca7d3862930d85f317da20e2754f710e7331820837245b2`。
- `optin-summary.json`：逐层 cache、逐步 latent/noise、12 图、内存和性能摘要。
- `reference384/summary.json`：reference384 的独立输入摘要；capture SHA256
  `4471076d2c1bb8068e08c0ee753dedab4590f62093b9b6186b49cc57615ef270`。

结果：

| 指标 | 结果 | 判定 |
| --- | ---: | --- |
| reference cache 最大 NRMSE | `0.0074060` (`up_2_2_0`) | PASS ≤ `0.01` |
| reference write NRMSE | `0.0132010` | PASS ≤ `0.03` |
| 15 步 latent 最大 NRMSE | `0.0064423` | PASS 探索性 ≤ `0.01` |
| 12 图平均像素差 | `0.0595/255` | PASS 探索性 ≤ `0.25/255` |
| 12 图最大像素差 | `43/255` | FAIL 探索性 ≤ `32/255` |
| 同实例重复 | 12 图 `max_image_abs=0`，cache exact | PASS |
| reset | 两次均清空 cached/pending/producer cache | PASS |

因此 reference 数值和循环状态合同通过，探索性图像最大差仍未通过，不能切
默认。完整 loop 的 fallback 路径耗时 `107.8475s`；同一遮挡输入此前 MLX
fp32 路径为 `83.2974s`，单样本比值 `1.295`，显示 fallback 有明显性能成本。
该性能比较不是完整 benchmark 认证，不能推广到所有输入或组件。

独立 `reference384` 已通过本轮全部探索性门槛：cache 最大 NRMSE
`0.0045591`（`up_2_0_0`）、write `0.0105944`、12 图平均差
`0.05165/255`、最大差 `20/255`；同实例第二请求图像 `max_image_abs=0`、
reference cache exact。耗时 `84.4022s`，同输入此前 MLX fp32 为 `73.6512s`，
比值 `1.146`。这说明 fallback 对 384 输入的质量筛查通过，但性能仍有成本。

全量回归现为 `88/88`，`py_compile` 与 `git diff --check` 通过。受保护的
`torchvision_fix.py` 未加入提交。

## 下一阶段

1. 用相同 opt-in 配置验证 fixed512，尤其复核图像最大差，并把该输入的
   quality/performance 摘要写入外部证据。
2. 若遮挡 512 图像门槛仍失败，按逐步 trace 先定位 `noise_pred` 后段误差，比较
   fallback 对 generation 全部卷积的代价；不要无条件分块 65 个卷积。
3. 在多输入质量与性能结果齐备前，保持默认 Torch reference、默认 Winograd
   和默认 MLX dispatch 不变。继续单 GPU、空 queue、1 GiB cache、fresh output，
   保护旧 venv、原始 checkout、已交付 GLB 与 dirty `torchvision_fix.py`。

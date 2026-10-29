# R3 occluded512 reference cache swap 因果闭环（2026-10-29）

本阶段在已有 C/D disabled-Winograd 合同上，加入了仅诊断用的完整 16 层
`condition_embed_*` cache 注入。目标是把 MLX reference write 与 generation
consumer 解耦，确认 cache 是否足以解释首步分支和最终图像差。没有修改默认
reference、Winograd、MLX dispatch 或任何已交付 GLB。

## 代码变更

- `tools/quality_demo/probe_fixed_conditions.py`
  - 新增 `_load_reference_cache()`：CPU-only 校验目录必须恰好 16 个 rank-3
    浮点、finite 的 `condition_embed_*.npy`。
  - 新增 opt-in `--inject-cache DIR`，仅允许 `capture --mlx`，并要求固定
    `--capture-cache`、`--initial-noise`、`HUNYUAN_MLX_FIRST_STEP=1`、
    `HUNYUAN_MLX_RESIDENT_LOOP=1`。
  - 在真实 reference write 返回后替换 live Torch cache，同时设置
    `MLXDispatchUNet._pending_reference_cache`；首个 MLX generation step
    校验 consumer cache 与注入源逐元素 exact。
  - 报告写入源 capture/layer SHA256、`applied` 和 `consumer_exact`。
- `tests/test_probe_fixed_conditions_cache.py`
  - 2 个 unittest：完整 16 层加载和不完整目录拒绝。

## 固定合同

- C：`occluded512-replay`，MLX fp32 reference + fp16 generation，Winograd
  disabled。
- D：`occluded512-torchref-winograd-off`，Torch reference + fp16 generation，
  Winograd disabled。
- 相同 fixtures、initial noise、DINO/text/position 输入、guidance 3、reference
  scale 1、MVA scale 1；只运行一个 GPU 任务，ComfyUI queue 在每次开始前为空。
- 外部来源：
  `/Users/daubert/Github/hy3d-upgrade-evidence/20261029/r3-reference-winograd-optin-20261029/`

## 结果

机器可读汇总：`/private/tmp/hy3d-cache-swap-summary-20261029.json`，SHA256
`7406ee9b89a4b05f16128317e686ecfdf5c34a5b27287a9abb7f4559ddcca822`。
成功 capture 与摘要已归档到
`/Users/daubert/Github/hy3d-upgrade-evidence/20261029/r3-cache-swap-causal-20261029/`；
逐文件 manifest 共 120 项，manifest SHA256
`0d277f6dc1fa3ec40878a7b8ac5fba8437bf23e30e2a1c9240c0c5d684d12237`，归档与
`/private/tmp` 逐文件 exact。

### 首步交换

1. Torch reference + **C cache**：
   `/private/tmp/hy3d-cache-swap-cd-step1-20261029-elevated-2/`
   - capture SHA256 `e7d1d46cf5cec787cfe247d00463d0939378acf9ca1fe4ae451fe77257a42ba6`
   - `consumer_exact=true`；冻结 input、inner sample、DINO/text exact。
   - 三条 `inner_output`/`noise_pred_0` 分支与 C 逐元素 exact；与 D 的两条
     条件分支 NRMSE `0.001444446`、`0.000810333`。
   - `reference_write` 与 D 逐元素 exact、与 C 不同。

2. MLX fp32 reference + **D cache**：
   `/private/tmp/hy3d-cache-swap-dc-step1-20261029/`
   - capture SHA256 `aa8f882e11a49328ee0eeab351d9aaf21a55953fa2b6abb4fe72503160d49003`
   - `consumer_exact=true`；三条 `inner_output`/`noise_pred_0` 分支与 D
     逐元素 exact。
   - reference write 仍与 D 不同，说明注入只替换 cache，不伪造 reference
     output。

### 完整 15 步

MLX fp32 reference + D cache：
`/private/tmp/hy3d-cache-swap-cd-full-20261029/`

- capture SHA256 `b322e2a81797e3f16480040ae5f85bfeb77cc3f5bd7fc836b5eb6194fad1e251`
- `consumer_exact=true`，`generated_images=12`。
- `noise_pred_0` 与 D 逐元素 exact；12 张生成图与 D 逐元素 exact，image 11
  相对 C 的最大差 `42/255`，复现了此前 C/D 边界。
- `reference_write` 相对 D 仍 NRMSE `0.013204804`、max abs `0.0512085`；
  因此 reference write 不是必须和 generation cache 相同的可视输出边界，
  但 16 层 cache 已足以决定这条 generation 轨迹。

## 失败尝试与环境证据

- 普通沙箱首次运行在 `pipe.to("mps")` 失败：PyTorch 报当前 MPS 不可用；
  宿主权限下 `sw_vers=27.2`、`torch==2.14.1`、`torch.backends.mps.is_available()`
  为 True，重跑成功。首次失败 fresh 目录保留为诊断失败证据。
- 普通沙箱中的完整 unittest 进程在导入 MLX 时因无 Metal 设备触发 nanobind
  重复注册并 abort（signal 6）；宿主 Metal 权限下同一 suite `90/90 OK`。
- 第二次命令曾把 bundle 路径写成 `hy3-upgrade-evidence`，在 MLX 初始化时
  退出；修正为 `hy3d-upgrade-evidence` 后成功。
- 每次 GPU 任务前 `GET http://127.0.0.1:8188/queue` 均为
  `queue_running=[]、queue_pending=[]`；没有并行旧/新版推理。

## 验证

- 候选环境 `python -m unittest -v tests.test_probe_fixed_conditions_cache`：
  `2/2 OK`；宿主 Metal 权限下完整 `unittest discover -s tests`：`90/90 OK`。
- `python -m py_compile tools/quality_demo/probe_fixed_conditions.py`：exit 0。
- `git diff --check`：通过。
- 保护文件 `torchvision_fix.py` SHA256 仍为
  `d6139690b099ca3ecef80ddf3f1a25edfdf57f027c79bf8655a8f8aac1c8b493`。

## 结论与下一步

本阶段首次把 C/D 残余差异闭环到完整 reference cache consumer：同一 MLX
generation、同一 frozen input 下，交换 16 层 cache 即可在首步和完整 15 步
逐元素复现另一侧的 noise 与图像。这个结果支持“reference cache 是 C/D
`42/255` 差异的充分传递边界”，但不证明单个 cache 层或 reference write 的
某个单算子是更早根因。

下一优先级是把这项证据写入总交接并推送；如需继续定位，应做单层 cache 替换
或 reference producer 的层级差分，仍保持一次一个 GPU、fresh output、默认路径
不切换。无需再怀疑 Torch 2.14.1 的 MPS bicubic 修复。

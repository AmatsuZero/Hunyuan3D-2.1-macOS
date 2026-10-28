# R3 交接：MLX dual reference forward（2026-10-28）

总交接仍是 `HANDOFF-torch-mps-upgrade.md`。本阶段实现并验证了 opt-in 的
`unet_dual` reference forward：它从同一 checkpoint 的 `unet_dual.*` 子树加载
686 个权重，在 MLX/Metal 上写出 16 层 reference cache；默认路径仍是 Torch/MPS。

## 实现

- `hy3dpaint/mlx_unet.py` 增加 reference mode：标准 self-attention、text
  cross-attention、GEGLU 和 16 层 cache 写入；`input_channels=4` 用于 dual stream。
- `hy3dpaint/mlx_utils.py` 增加严格的 `unet_dual` 权重映射、source key closure、
  `_MLXReferenceUNetWrapper` 和 dispatch 的 pending native MLX cache。仅设置
  `HUNYUAN_MLX_REFERENCE_FORWARD=1` 才替换 `unet_dual`；未设置时默认 Torch 路径不变。
- `tools/quality_demo/replay_mlx_reference.py` 提供离线 replay，先逐元素核对
  reference sample/timestep/encoder 输入，再输出逐层 NRMSE/mean/max 及 checkpoint SHA。
- 新增 `tests/test_mlx_reference_forward.py`；原有 cache producer/reset 测试继续保留。

## 真实证据

### 输入合同与逐层 replay

- MLX capture：`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-mlx-reference-forward-02/`
- Torch input capture：`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-torch-reference-input-01/`
- 正式报告：`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-mlx-reference-replay-01/exact-input-hashed-metrics.json`
- 两个 capture 的 `reference_sample`、timestep、encoder states 逐元素相同；checkpoint
  SHA256 `675a1b5cd0098b2002637c443946529c03c5cd54427f40245263350feb3dd5b8`；686
  个 `unet_dual` 权重键全部映射。
- 16 层 cache 的最大 NRMSE `0.006852`（`up_3_2_0`），最大 mean abs `0.002531`；
  `reference_write` 输出 NRMSE `0.026349`。这是 fp16 MLX 与 Torch/MPS 的数值近似，
  **不是逐元素 parity**；当前报告已通过工程门槛，全部层指标仍保留供后续 replay。
- 已加入离线门槛工具 `tools/quality_demo/validate_mlx_reference_report.py`：默认
  要求所有 cache 层最大 NRMSE ≤ `0.01`、`reference_write` NRMSE ≤ `0.03`，并拒绝
  输入不等、checkpoint SHA/686 权重键或 16 层结构漂移。这两个值是当前单一固定条件
  报告上方的工程验收门槛，不代表
  已完成遮挡或多分辨率质量验收；后续 replay 若超门槛必须停在 opt-in 路径。

### 15 步集成与 reset

- capture 使用 fallback=0、固定 fixture/初始 noise、512px、6 views、15 steps、
  resident loop、1 GiB MLX idle-cache threshold；exit 0，首轮 `63.444 s`，
  MLX reference-forward 单轮 `62.566 s`，不能用单样本宣称性能收益。
- 12 张图相对 boundary 基线平均绝对差 `0.0476/255`、最大 `18/255`；视觉保持
  黑 H/Y/D、红 3。未写入或覆盖任何 GLB。
- 同实例第二次请求：`max_image_abs=0`，`reference_cache_exact=true`，第二次
  `first_gen_count=1`；证明 MLX reference cache reset 不串请求。

## 验证与边界

- Metal 可用候选 venv 全量 unittest：`75/75` 通过；reference wrapper 定向
  测试 `2/2` 通过；`py_compile`、`git diff --check` 通过。
- 这是 opt-in 实验路径，仍把 MLX cache 转回 Torch 供 outer wrapper 合同使用；
  outer generation、DINO/VAE 和最终解码仍未 Torch-free。未切换默认，也未声称
  全 MLX 或 Torch-free 正式运行时。
- 当前已冻结首版工程门槛（cache `0.01`、reference write `0.03`），仍需确认更多
  遮挡/分辨率与用户设备质量，再评估是否保留该 opt-in；默认路径继续使用已验证
  Torch reference producer。

本阶段验证命令（不启动 GPU）：

```sh
rtk proxy /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python \
  tools/quality_demo/validate_mlx_reference_report.py \
  --report /Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-mlx-reference-replay-01/exact-input-hashed-metrics.json
```

预期 `passed=true`；当前报告的观测值为 `0.006851671...` 与 `0.026348907...`。

## 复现开关

在 `HANDOFF-r3-reference-cache-boundary-20261028.md` 的 capture 命令基础上增加：

```sh
HUNYUAN_MLX_REFERENCE_FORWARD=1
```

同时保留 `HUNYUAN_MLX_FIRST_STEP=1`、`HUNYUAN_MLX_RESIDENT_LOOP=1`、
`HUNYUAN_MLX_REFERENCE_CACHE=1` 和 `HUNYUAN_MLX_POSITION_INDICES=1`。运行前检查
GPU 进程和 ComfyUI queue；保护 `torchvision_fix.py`、已交付 GLB 及原始 checkout。

## 下一会话入口

- 当前分支 `codex/torch-mps-upgrade`，本阶段基于 `1c3c48b`；提交后以
  `git log -1` 和 `git ls-remote fork refs/heads/codex/torch-mps-upgrade` 核对推送。
- 优先补有遮挡输入及其他支持分辨率的相同输入 replay，并用上述门槛逐份判定；
  通过后继续 outer condition/VAE/最终 Torch 边界迁移。
- 运行前核对进程和 ComfyUI 空队列，单次仅一项 GPU 推理；新证据写 fresh 目录。
- 本次交接前重新执行宿主 Metal 全量测试 `75/75`、语法编译和 diff 检查通过。
  受限 shell 测试在 MLX 导入时 exit 134，宿主复验通过；独立 reviewer 因服务
  `429 Too Many Requests` 未运行，不能声称独立审查通过。
- 用户已明确授权每阶段写交接、提交推送后自动创建新 Codex 任务继续同一目标；
  创建失败时保留原始错误，避免重复并行创建或并行推理。

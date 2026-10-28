# R3 交接：outer DINO projector MLX consumer（2026-10-28）

总交接仍是 `HANDOFF-torch-mps-upgrade.md`。本阶段把 outer UNet 的 DINO
projection 作为显式 opt-in MLX consumer 接入；reference-write、DINO backbone、
position voxel index 和 VAE 仍保持 Torch。

## 已完成

- 新增 `hy3dpaint/mlx_dino_torch_adapter.py`：
  - 严格从已验证 condition bundle 加载 `image_proj_model_dino` 的 4 组权重；
  - 只在设置 `HUNYUAN_MLX_DINO_PROJECTOR_BUNDLE` 时安装；未设置时返回 false，
    默认路径不变；
  - 在 dynamic outer UNet 不变的前提下，用 Torch module contract 接收 MPS
    tensor，MLX 执行 projector，再将结果返回 MPS Torch tensor；两处 NumPy
    同步边界显式保留，避免误称为全 MLX；
  - 记录 bundle、source checkpoint 和 config SHA256，便于版本漂移审计。
- `hy3dpaint/mlx_utils.py` 的 opt-in setup 在替换 inner dispatch 时安装该
  projector；`probe_fixed_conditions.py` 把实际 adapter 类和 bundle provenance
  写入 capture report。
- 新增 `tests/test_mlx_dino_projector_adapter.py`，覆盖数值、MPS 输出合同、
  opt-in 选择和 bundle provenance。

## 真实证据

条件 bundle：`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-condition-bundle-01/conditions.safetensors`

- bundle SHA256：`94084b1ea331ec83f1d52596ecb799b52fea5ce8247777b974da4da9143fa981`。
- source checkpoint SHA256：`675a1b5cd0098b2002637c443946529c03c5cd54427f40245263350feb3dd5b8`。
- config SHA256：`ce0c6d379e3b1d3e1f79338de70c80a1e36f17a6372439a8249b6fb0dfa1b608`。

### 单步边界

`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-mlx-dino-projector-consumer-01/`

- fallback=0、固定 fixture/初始 noise、512px/6 views、`stop-after=1`，exit 0，
  runtime source SHA 与模型源一致；capture report 标记实际类为
  `mlx_dino_torch_adapter.MLXDinoProjectorAdapter`。
- 相对既有 Torch projector capture `r3-mlx-direct-first-01`：
  - `inner_dino_hidden_states` NRMSE `0.00034624`，mean abs `0.00006986`，
    max abs `0.00390625`；
  - `noise_pred_0` NRMSE `0.00082671`，mean abs `0.00035158`，max abs
    `0.0234375`；
  - `reference_write`、`inner_sample`、generation encoder states、position
    voxel indices 和 condition embed 逐元素相同。

### 15 步 resident 质量

`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-mlx-dino-projector-resident-01/`

- fallback=0、1 GiB MLX idle-cache threshold、`HUNYUAN_MLX_FIRST_STEP=1`、
  `HUNYUAN_MLX_RESIDENT_LOOP=1`，exit 0，耗时 `67.732 s`，生成 12 张图。
- 相对 `r3-mlx-direct-first-01` 的 12 张 PNG：平均绝对差 `0.04741/255`，
  最大差 `17/255`；正面图保持黑 H/Y/D、红 3。该 capture 未启用 resident
  trace，因此不把它宣称为逐步 trace 证据；首步和最终图像是本阶段的验收边界。
- 既有 resident loop 的逐步 trace/同实例 reset 证据仍见
  `HANDOFF-r3-resident-loop-20261028.md`。

复现命令使用全新 output 目录；运行前先检查 ComfyUI queue 和其他推理进程：

```sh
rtk proxy curl -s --max-time 3 http://127.0.0.1:8188/queue
rtk proxy pgrep -alf 'python|ComfyUI|mlx|torch|hy3d'
rtk proxy env PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  HUNYUAN_MLX_DINO_PROJECTOR_BUNDLE=/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-condition-bundle-01/conditions.safetensors \
  HUNYUAN_MLX_FIRST_STEP=1 HUNYUAN_MLX_RESIDENT_LOOP=1 \
  /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python \
  tools/quality_demo/probe_fixed_conditions.py --stage capture \
  --fixtures /Users/daubert/Github/hy3d-upgrade-evidence/20261027/color-diagnosis/fixed-conditions-fixtures \
  --output-dir /Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-mlx-dino-projector-replay-01 \
  --steps 15 --stop-after 0 --capture-cache \
  --initial-noise /Users/daubert/Github/hy3d-upgrade-evidence/20261027/color-diagnosis/initial-noise-torch251-seed0.npy \
  --guidance-scale 3 --reference-scale 1 --mva-scale 1 \
  --mlx --mlx-cache-limit-mib 1024
```

## 代码验证

- Metal 可用候选 venv 全量 unittest：`67/67` 通过。
- `py_compile`（adapter、dispatch、capture tool）通过。
- `git diff --check` 通过。
- 受限 shell 的 MLX “No Metal device available” 只适用于无 Metal 的命令环境；
  本阶段真实 capture 和全量测试均在 Metal 可用环境执行。

## 边界与下一步

- 这是 opt-in consumer，不是默认切换，也没有端到端性能收益结论；每次
  projector 调用仍有 Torch↔NumPy↔MLX 同步。
- 下一阶段迁移 position voxel indices consumer/producer 边界，要求四个分辨率
  integer cache 逐元素相同、跨请求 reset 通过，再继续处理 dynamic reference
  cache。不要在本阶段修改 dynamic model source 或 reference attention layout。

## 保护项

- 一次只跑一个 GPU 推理；不覆盖已交付 GLB。
- 不提交、还原或清理根目录 dirty `torchvision_fix.py`。
- 不在原始 checkout 重编原生扩展；远端只推送 `fork/codex/torch-mps-upgrade`。

# R3 交接：position voxel indices MLX producer/consumer（2026-10-28）

总交接仍是 `HANDOFF-torch-mps-upgrade.md`。本阶段把 resident loop 请求内的四组
position voxel indices 从 Torch producer 切到 opt-in MLX producer；默认路径保持不变。

## 已完成

- 新增 `hy3dpaint/mlx_position_indices.py`：复刻 Torch
  `calc_multires_voxel_idxs` 的四个分辨率计算、invalid pixel 置零、阈值、量化和
  `[B, N*H*W, 3]` 布局；输出 MLX `int32` indices。
- `hy3dpaint/mlx_utils.py` 新增
  `make_mlx_position_cache()`，在 outer Torch→MLX 边界只转换一次 position map；
  `MLXDispatchUNet` 识别已有 MLX arrays，避免 inner dispatch 再次 Torch→MLX 转换。
- `hunyuanpaintpbr/pipeline.py` 仅在
  `HUNYUAN_MLX_POSITION_INDICES=1` 且 resident loop 已启用时预计算请求内 cache，
  通过既有 `cache` 传入 inner UNet；缺少 position maps 或错误配置会 fail closed。
- `tools/quality_demo/probe_fixed_conditions.py` 可保存 MLX arrays，并在 capture
  report 标记 `position_indices_mlx`。
- 新增 `tests/test_mlx_position_indices.py`：四分辨率 Torch/MLX 数值对照，以及
  native MLX cache 直通与跨请求 reset。

## 真实证据

运行目录：`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-mlx-position-indices-01/`

- 单 GPU、`PYTORCH_ENABLE_MPS_FALLBACK=0`、固定条件、512px、6 views、15 steps、
  resident loop、1 GiB MLX idle-cache threshold，exit 0；耗时 `62.142 s`，生成
  12 张图；capture report 的 `mlx_resident_loop.position_indices_mlx=true`。
- 与 DINO-only resident 基线
  `/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-mlx-dino-projector-resident-01/`
  逐元素比较：四组 indices（`24576/6144/1536/384`）数值完全相同；基线 dtype
  `int64`、新 producer dtype `int32`，shape 与数值一致。
- `reference_write`、`inner_output`、`noise_pred_0`、`inner_sample`、inner DINO
  states、generation encoder states 均逐元素相同；12 张 `generated_*.png` 均逐字节
  相同（最大像素差 0）。
- 两个 capture 均 fallback=0、无 warnings、source/provenance 校验通过；没有覆盖
  GLB 或已交付资产。

## 代码验证

- Metal 可用候选 venv 全量 unittest：`69/69` 通过。
- 定向 `tests.test_mlx_position_indices`：`2/2` 通过。
- 相关文件 `py_compile` 通过；`git diff --check` 通过。
- 受限 shell 的 MLX 无 Metal 设备错误不代表代码失败；定向和全量测试均在宿主
  Metal 环境执行。

## 边界与下一步

- 这是 opt-in producer/consumer 切片，不是默认切换，也不是 Torch-free outer
  condition production；outer reference-write、条件 embedding、VAE 和 dynamic
  model source 仍保持现状。
- MLX cache 使用 `int32` 是索引表示优化；数值 parity 已在真实四分辨率 fixture 和
  完整 15 步输出上验证。仍需在后续阶段继续检查更广泛遮挡/分辨率配置。
- 下一阶段处理 dynamic reference cache producer，继续要求请求间 reset、真实尺寸
  cache 数值和 15 步图像逐步对照；完成后再评估移除更多 Torch 条件生产边界。

## 复现命令

```sh
rtk proxy curl -s --max-time 3 http://127.0.0.1:8188/queue
rtk proxy pgrep -alf 'python|ComfyUI|mlx|torch|hy3d'
rtk proxy env PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  HUNYUAN_MLX_DINO_PROJECTOR_BUNDLE=/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-condition-bundle-01/conditions.safetensors \
  HUNYUAN_MLX_FIRST_STEP=1 HUNYUAN_MLX_RESIDENT_LOOP=1 \
  HUNYUAN_MLX_POSITION_INDICES=1 \
  /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python \
  tools/quality_demo/probe_fixed_conditions.py --stage capture \
  --fixtures /Users/daubert/Github/hy3d-upgrade-evidence/20261027/color-diagnosis/fixed-conditions-fixtures \
  --output-dir /Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-mlx-position-indices-replay-01 \
  --steps 15 --stop-after 0 --capture-cache \
  --initial-noise /Users/daubert/Github/hy3d-upgrade-evidence/20261027/color-diagnosis/initial-noise-torch251-seed0.npy \
  --guidance-scale 3 --reference-scale 1 --mva-scale 1 \
  --mlx --mlx-cache-limit-mib 1024
```

## 保护项

- 一次只跑一个 GPU 推理；不覆盖已交付 GLB。
- 不提交、还原或清理根目录 dirty `torchvision_fix.py`。
- 不在原始 checkout 重编原生扩展；远端只推送 `fork/codex/torch-mps-upgrade`。

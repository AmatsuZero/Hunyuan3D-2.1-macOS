# R3 occluded512 full `down_0` / `conv1` oracle（2026-10-29）

本阶段在空 ComfyUI queue 下串行完成两次 Torch-reference capture：默认 Winograd
与显式 disabled Winograd。目标是把上一阶段局部 `down_0_resnet_0.conv1` 证据扩展
到完整空间/通道，同时保持同一 frozen 条件与单 GPU 约束。

## 合同与约束

- worktree `/Users/daubert/Github/hy3d-torch-upgrade`，分支
  `codex/torch-mps-upgrade`；Torch 2.14.1，`PYTORCH_ENABLE_MPS_FALLBACK=0`。
- 两轮都使用同一 occluded512 fixtures、Torch 2.5.1 初始 noise fixture、DINO
  condition bundle、15 steps、guidance/reference/mva = `3/1/1`、1 GiB MLX
  cache；只改变 `HUNYUAN_MLX_DISABLE_WINOGRAD`。
- A：默认 Winograd；D：`HUNYUAN_MLX_DISABLE_WINOGRAD=1`。两轮均为
  Torch reference（`HUNYUAN_MLX_REFERENCE_FORWARD=0`），MLX generation 仅用于
  resident inner loop。没有覆盖 GLB、没有修改原始 checkout。
- 运行前后提升权限只读检查 ComfyUI：`queue_running=[]`、`queue_pending=[]`；
  相关进程只有既有 ComfyUI 服务，没有第二个 Hunyuan 推理进程。

## 证据归档

外部目录：
`/Users/daubert/Github/hy3d-upgrade-evidence/20261029/r3-reference-winograd-optin-20261029/occluded512-full-down0-20261029/`

- `torch-default/` 与 `torch-disabled/` 各 89 个文件；两侧 12 张 generated 图分别
  与此前 A/D 归档逐字节 exact。
- 两轮 `initial-latents`、`inner_sample`、DINO/text/reference 输入和 16 个
  condition cache 全部逐元素 exact；完整 `norm1` 也 exact。
- `step1_down_0_resnet_0.norm1_full.npy` 与 `conv1_full.npy` 均为
  `(36, 64, 64, 320)`，保存 NHWC、三 CFG 分支、全部空间/通道。
- 阶段摘要 `summary.json` SHA256：
  `5f342ba07cbf48944334c281dd9400602ffa65ddc956f5729067d45a619014ff`。
- CPU float32 oracle `conv1-full-cpu-oracle.json` SHA256：
  `111181f2e139e88b59997e0d0108cf0634c7f78bdb1dcfec9b40353330584fe2`。
- `sha256-manifest.json` 覆盖 182 个归档文件并逐文件回读核验；自身 SHA256
  `245b421cdf1eb9faff0318dd98e8754398fc32c324fb535d3d69e50a8fcb2568`。

## 结果

- 完整 `conv1` 默认 vs disabled：NRMSE `0.00388407`，MAE `0.00206658`，最大
  abs `0.07484436`；`norm1` 完全相同。
- generation `unet.down_blocks.0.resnets.0.conv1` 的 CPU float32 oracle（先 SiLU，
  再 OIHW 3×3 conv，padding=1）结果：
  - 默认 Winograd：全量 NRMSE `0.00388886`，MAE `0.00208085`，max
    `0.07493279`；三 CFG 分支 NRMSE 均约 `0.003889`，最大通道 NRMSE `0.0103304`。
  - disabled implicit GEMM：全量 NRMSE `0.000331384`，MAE `0.000183129`，max
    `0.00583744`；三 CFG 分支 NRMSE 均约 `0.000331`，最大通道 NRMSE `0.00053771`。
- 这加强了“首步差异来自 Winograd 相关 conv1 数值路径”的证据，并显示 disabled
  更接近 CPU float32；它仍未证明具体 Metal kernel 源码错误，也未证明最终图像
  超阈值像素的直接因果。`43/255` 探索性图像门槛仍未通过，默认路径不切换。

## 代码变化与验证

- [capture_occluded512_local_layers.py](tools/quality_demo/capture_occluded512_local_layers.py)
  新增 `--full-down0`；它必须与 `--detailed-down0` 同时使用，只保存首个分叉所需
  的完整 `norm1`/`conv1`，旧默认行为不变。
- `python3 -m py_compile tools/quality_demo/capture_occluded512_local_layers.py`：通过。
- 参数负例 `--full-down0` 无 `--detailed-down0`：按预期拒绝。
- 提升权限宿主 Metal：`python -m unittest discover -s tests -v` → `Ran 88 tests ... OK`。
- `git diff --check`：通过。受保护根目录 `torchvision_fix.py` SHA256 仍为
  `d6139690b099ca3ecef80ddf3f1a25edfdf57f027c79bf8655a8f8aac1c8b493`，未暂存。

## 下一步

1. 只暂存诊断脚本与两份交接，追加总交接 §24，提交并推送 `fork/codex/torch-mps-upgrade`。
2. 以本摘要为输入继续 CPU-only 的 Winograd/MLX policy 解释；若需要源码级结论，
   必须先确认运行 wheel 0.32.4 的精确 dispatch source，不能把 0.32.3 dirty
   checkout 当作 binary proof。
3. 默认 Torch reference、默认 Winograd、默认 MLX dispatch 继续保持；不覆盖已交付
   GLB，不在原始 checkout 重编扩展。

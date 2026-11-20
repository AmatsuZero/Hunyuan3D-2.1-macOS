# R3 交接：Winograd fallback 双输入 reference replay（2026-10-28）

唯一权威总交接为 `HANDOFF-torch-mps-upgrade.md`。本阶段基于 `1511305`，
在隔离 worktree `codex/torch-mps-upgrade` 诊断 MLX reference 的卷积数值分叉。
默认运行路径及 `0.01/0.03` 门槛均未切换。

## 发现与边界

- 最小真实 `down_0.resnets[0].conv1`，320→320、3×3，MLX Metal 直接卷积
  与 Torch MPS NRMSE 约 `0.00206`；用
  `MLX_CONV_WINOGRAD_WORKING_SET=1` 后 NRMSE 为 `9.3704267e-7`、
  max abs `3.0517578e-5`。这比临时 65 处高通道分块更准确且不需要改模型结构。
- 本地参考的 mlx-swift C++ 源码（头文件版本 0.32.2）在
  `Source/Cmlx/mlx/mlx/backend/metal/conv.cpp` 约 1424–1450 行，将满足
  3×3/stride 1、C/O 均为 32 倍数等条件的卷积派到 Winograd；
  `winograd_batch_step()` 读取上述环境变量，并在极小 working set 下返回 0。
  当前 Python 环境头文件版本是 MLX 0.32.4，源码版本并非逐字节对应；
  因果判断以当前环境最小 probe 的实测结果为主。
- 环境变量须在进程启动前设置。它改变该 MLX 进程的卷积调度，不能视为仅
  作用于 `unet_dual`；RealESRGAN、generation 等 MLX 卷积的性能与数值需要
  在完整 pipeline 内复测。

## 正式双输入 replay

每次 GPU 运行前均成功查询 ComfyUI `/queue`：running/pending 为空；
进程列表未见其他 Hunyuan/MLX/Torch 推理。两次 replay 串行运行，
`MLX_CONV_WINOGRAD_WORKING_SET=1`、reference 参数 fp32、
Transformer GroupNorm epsilon `1e-6`、checkpoint SHA256
`675a1b5cd0098b2002637c443946529c03c5cd54427f40245263350feb3dd5b8`、
686 映射权重。`replay_mlx_reference.py` 现在在模型/输入 MLX 分配前
设置默认 1024 MiB idle-cache limit，并把实际设置与前值写入报告。
两份报告均记录 `limit_bytes=1073741824`；这是 MLX 空闲缓存回收阈值，
不是进程内存硬上限。

报告在外部证据目录
`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-reference-winograd-fallback-01/`：

| 冻结输入 | 最大 cache NRMSE | reference write NRMSE | 原门槛验证 |
| --- | ---: | ---: | --- |
| occluded512 | `0.0074033638` (`up_2_2_0`) | `0.0131999117` | PASS |
| reference384 | `0.0045537077` (`up_2_0_0`) | `0.0105911968` | PASS |

`occluded512-replay.json` SHA256：
`703f067896db251747872f8471d4ae7b4a5531ba312c7e8773dd25de07feb16f`。
`reference384-replay.json` SHA256：
`09873dca1207ff434666a31a14f242215ffeb6b494b9163079fdc1500f6648e8`。
两个报告均 `inputs_exact=true`、16 层完整，
`validate_mlx_reference_report.py` 各自 exit 0。全量 unittest `85/85`，
`py_compile` 和 `git diff --check` 通过。受保护的根目录
`torchvision_fix.py` SHA256 仍为
`d6139690b099ca3ecef80ddf3f1a25edfdf57f027c79bf8655a8f8aac1c8b493`。

## 下一阶段优先级

1. 在 opt-in dispatch 中实验 Winograd fallback，并对**真实** fp32 reference
   输入转 fp32、cache 转 generation fp16 的完整合同做 15 步对照。离线
   `replay_mlx_reference.py` 目前只将参数转 fp32、保留 capture 输入 dtype；
   不能把此 replay 当成完整 resident 质量通过。
2. 对遮挡 512 和 384 两组真实固定 posterior 条件分别比较 Torch reference、
   opt-in MLX reference 的 16 层 cache、write、逐步 latent 和 12 图；执行同实例
   reset/重复请求。保持既有 cache `0.01`、write `0.03`，完整 loop 的探索性
   `0.01` latent 和图像筛查见 `HANDOFF-r3-reference-fp32-resident-20261028.md`。
3. 测 fallback 对 generation、RealESRGAN 和端到端延迟/内存的影响。只有
   多输入质量、颜色、reset、性能证据齐备后才考虑默认切换。继续一次一个 GPU
   任务，启动前成功查询队列和进程；保持 1 GiB idle-cache、fresh output。

复现单输入：

```sh
rtk proxy env MLX_CONV_WINOGRAD_WORKING_SET=1 \
  /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python \
  tools/quality_demo/replay_mlx_reference.py \
  --capture /Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-reference-fp32-resident-01/occluded512/torch-reference \
  --torch-reference /Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-reference-fp32-resident-01/occluded512/torch-reference \
  --checkpoint /Users/daubert/Github/hunyuan3d-artifacts/Hunyuan3D-2.1-modelscope/hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin \
  --output /private/tmp/hy3d-r3-winograd-replay-new.json \
  --fp32 --transformer-norm-eps 1e-6 --mlx-cache-limit-mib 1024
rtk proxy /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python \
  tools/quality_demo/validate_mlx_reference_report.py \
  --report /private/tmp/hy3d-r3-winograd-replay-new.json
```

先确认 output 不存在、ComfyUI queue 为空、无其他推理进程。不要覆盖已交付
GLB，不改原始 checkout/旧 venv，不提交或清理受保护的 `torchvision_fix.py`。

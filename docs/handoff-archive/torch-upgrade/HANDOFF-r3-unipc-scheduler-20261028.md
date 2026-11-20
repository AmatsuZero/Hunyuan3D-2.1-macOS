# R3 交接：MLX UniPC scheduler 独立 replay（2026-10-28）

总交接仍是 `HANDOFF-torch-mps-upgrade.md`。这是 R3 的 scheduler 切片，**完整 R3 尚未完成**。

## 已完成

- `hy3dpaint/mlx_unipc_scheduler.py` 实现当前 paint checkpoint 使用的 UniPC 配置：
  `scaled_linear`、zero-terminal-SNR、`trailing`、`v_prediction`、`bh2`、2 阶、
  `predict_x0=true`。未支持的 scheduler 配置会显式失败。15 步 latent 和模型输出
  历史一直是 MLX 数组，模块不导入 Torch。
- `tools/quality_demo/probe_fixed_conditions.py` 现在在诊断 capture 中保存每步实际
  scheduler `model_output`、`sample`、`timestep`、timesteps/sigmas/config；正常推理
  入口没有改变。该捕获每步同步 MPS 张量，不能用其 wall time 做性能比较。
- `tools/quality_demo/replay_mlx_unipc.py` 用保存的真实张量做 teacher-forced 和连续
  15 步 replay，支持 MLX CPU/GPU，拒绝覆盖已有报告。
- `tests/test_mlx_unipc_scheduler.py` 对照 Diffusers CPU 的 15 步输出和 cache reset。
  全量 unittest 52/52 通过；`py_compile` 与 `git diff --check` 通过。

## 真实数值证据

外部目录：`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-unipc-capture-01/`。

- 固定条件 hybrid capture：Torch 2.14.1，fallback=0，15 步完整退出，12 张图、
  15 组 scheduler 输入/输出保存；`capture.json` SHA256
  `e4427786960095f6797b38d286c5f039a5ea380e51cf1e3834d36968d16210f2`。
- `mlx_cpu_replay.json`、`mlx_gpu_replay.json`：两设备的逐步误差统计相同，所有
  输出有限。teacher-forced 最大 NRMSE `0.00078931`；连续 15 步最终 NRMSE
  `0.00136145`、平均绝对误差 `0.00087780`、最大绝对误差 `0.01953125`。
- 这是保存的同一批 `scheduler_model_output` 的 replay，证明 scheduler 数值接近；
  它尚未证明 MLX scheduler 已在正式去噪循环里运行，也不是 Metal trace 验收。

## 后续

1. 将 reference-write、DINO projection、position voxel indices 和首步/cache 搬入
   MLX outer loop；保持现有固定条件数据流和每步对照。
2. 在 MLX 去噪循环中接入本 scheduler 与 CFG 条件融合，让 latent 在 15 步始终留
   在 MLX；验证新请求的 cache reset 与原始 albedo/MR 颜色。
3. 采集有界 Metal trace 和无 profiling 性能/内存样本。未达到 R3 完整验收前不切
   默认 dispatch，也不宣称全 MLX。

本阶段命令：

```sh
rtk proxy /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python \
  -m unittest discover -s tests -v
rtk proxy /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python \
  tools/quality_demo/replay_mlx_unipc.py \
  --capture /Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-unipc-capture-01 \
  --output /private/tmp/r3-unipc-replay-fresh.json --device gpu
```

遵守总交接 §3：单 GPU 任务；不要覆盖已有目录、GLB 或根目录 dirty 的
`torchvision_fix.py`；不要在原 checkout 重编扩展。

# R3 交接：独立 reference fp32 / generation fp16 consumer（2026-10-28）

总交接为 `HANDOFF-torch-mps-upgrade.md`；本阶段基于 `45b575b`。

## 实现

- `MLXDispatchUNet.set_reference_model(model, use_fp32=...)` 增加 reference 独立精度，
  由 `HUNYUAN_MLX_REFERENCE_FP32=1` opt-in；`HUNYUAN_MLX_FP32=1` 仍可同时提升
  generation 模型，默认两者仍 fp16。
- reference forward 的输入、timestep、encoder 和模型计算使用 `_reference_dtype`；
  写入 `_pending_reference_cache` 前统一 cast 到 generation `_mlx_dtype`。因此
  reference fp32 不会把后续 generation cache 或 inner UNet 变成 fp32。
- capture report 记录 `reference_dtype`、`generation_dtype` 和
  `reference_fp32_requested`；reset 路径保持原有清空合同。
- 新测试先复现了 API 缺口和 fp32 cache 泄漏，再覆盖独立精度、consumer dtype、
  output dtype 与 reset；当前定向测试通过。

## 真实证据

证据目录：
`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-reference-coverage-01/`

使用同一 checkpoint、同一 Torch capture 输入、fp32 MLX reference model 和 fp16
consumer 的串行集成 replay：

| case | cache max NRMSE | reference write NRMSE | consumer dtype | reset |
| --- | ---: | ---: | --- | --- |
| `occluded512` | `0.0063985774` | `0.0230500719` | fp16 | passed |
| `reference384` | `0.0065309129` | `0.0219643266` | fp16 | passed |

两份报告在各自 `mlx-fp32-consumer-fp16/exact-input-hashed-metrics.json`；验证器均
exit 0，门槛仍为 cache `0.01`、write `0.03`。同一进程对每组调用 `dispatch.reset()`，
随后确认 `_pending_reference_cache is None`。无完整 15 步图像、用户设备或性能结论，
没有写入 GLB。

## 验证

- `tests.test_mlx_reference_forward`：4/4 通过。
- 宿主 Metal 全量 unittest：本阶段重新运行 `83/83` 通过（exit 0），包括上述 4 项。
- 三个修改的 Python 文件 `py_compile`、`git diff --check` 均通过；两份真实报告
  重新运行 `validate_mlx_reference_report.py` 均 exit 0。
- 沙箱内 MLX 导入因 `No Metal device available` 退出，宿主重跑通过；这是运行权限
  差异。独立 native reviewer 因 `429 Too Many Requests` 失败，未形成独立审查结论。
- 测试前宿主仅见既有 ComfyUI 服务，`queue_running` / `queue_pending` 均为空；
  本阶段没有并行启动推理。ComfyUI 队列只用于排除 GPU 争用，本项目仍由 Python
  入口直接生成模型。

可复制检查（工作目录为本 worktree，MLX 测试需要宿主 Metal 权限）：

```sh
rtk proxy /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python -m unittest discover -s tests -v
rtk proxy /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python -m py_compile hy3dpaint/mlx_utils.py tests/test_mlx_reference_forward.py tools/quality_demo/probe_fixed_conditions.py
rtk proxy git diff --check
```

提交只包含本阶段三个 Python 文件、本文、总交接、上一诊断交接和 roadmap。
推送目标仍为 `fork/codex/torch-mps-upgrade`。继承 dirty `torchvision_fix.py` 保留，
交接时 SHA256 为 `d6139690b099ca3ecef80ddf3f1a25edfdf57f027c79bf8655a8f8aac1c8b493`。

## 下一会话

1. 在真实 resident loop 中启用 `HUNYUAN_MLX_REFERENCE_FORWARD=1` 与
   `HUNYUAN_MLX_REFERENCE_FP32=1`，对固定 512px 基线、遮挡和 384px 输入做完整
   15 步逐张图、逐步 latent、同实例 reset 对照；保持 generation fp16。
2. 记录 fp32 reference 的加载/推理耗时和 MLX peak memory，与当前 fp16 reference
   opt-in 单独比较；若性能或内存回退超过门槛，保留 fp16/默认 Torch路径。
3. 只有完整图像和 reset 通过，才评估 outer condition/VAE/默认 dispatch；每阶段写
   交接、提交推送并创建新会话。

复现时先查询 ComfyUI queue，单次仅一项 GPU 推理，使用 fresh output/evidence 目录。
保护根目录 dirty `torchvision_fix.py`、原始 checkout 和已交付 GLB。

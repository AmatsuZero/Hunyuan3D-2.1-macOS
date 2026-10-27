# P2 交接：Torch/MPS bounded Metal trace（2026-10-27）

这是 P2 阶段交接，供新会话继续使用；唯一总交接仍是 [`HANDOFF-torch-mps-upgrade.md`](HANDOFF-torch-mps-upgrade.md)。

## 结果

P2 的 named Torch-only trace gate 已通过。五个真实尺寸窗口均在候选环境 Torch 2.14.1、`PYTORCH_ENABLE_MPS_FALLBACK=0` 下运行；xctrace TOC 的目标 PID、Python marker、PyTorchMPS signpost 和该 PID 的 Metal GPU interval 已关联。每个阶段 target exit 为 0，marker begin/end 完整落在 trace 时间范围内，fallback signpost 为 0。

| 窗口 | 外部证据 | PyTorch signpost | Metal GPU interval | GPU channels | 结论 |
| --- | --- | ---: | ---: | --- | --- |
| DINO + VAE encode ×3 | `.../p2-conditions-trace-01/summary.json` | 2,571 | 1,314 | Compute/Fragment/Vertex | pass |
| VAE decode `(12,4,64,64) → (12,3,512,512)` | `.../p2-vae-decode-trace-01/` | 441 | 296 | Compute/Fragment | pass |
| RealESRGAN `(512,512,3) → (2048,2048,3)` | `.../p2-superres-trace-01/` | 2,445 | 1,055 | Compute/Fragment/Vertex | pass |
| reference UNet | `.../p2-reference-trace-02/` | 2,138 | 784 | Compute/Fragment/Vertex | pass |
| first-step outer/inner UNet | `.../p2-first-step-trace-03/` | 5,961 | 9,251 | Compute/Fragment/Vertex | pass |

外部根目录：`/Users/daubert/Github/hy3d-upgrade-evidence/20261027/`。raw trace 和模型输出均在各自外部目录；TOC/XML 导出与汇总保留在 `/private/tmp/hy3d-p2-*`。`p2-conditions-trace-01/summary.json` 与 `p2-vae-decode-trace-01/summary.json` 已写回外部 evidence。其余三份摘要写回时自动审批服务返回 429，操作未执行；可从原始 trace 重新导出并汇总。raw trace、TOC、XML 和模型输出均不进入 Git。

## 关键实现

- `tools/quality_demo/probe_fixed_conditions.py` 增加 `--trace-stage dino|vae.encode|reference|first_step|all`，只在选定 forward 内启用 `torch.mps.profiler.profile(mode="interval,event", wait_until_completed=True)`。
- `--wait-for-file` 允许模型加载完成后再用 `xctrace --attach` 录首步；不会把加载期高容量 Metal 表混入阶段窗口。该选项只允许 `first_step`。
- `tools/quality_demo/probe_torch_aux_trace.py` 覆盖真实尺寸 VAE decode 和 RealESRGAN forward，并在无 fallback 下执行 placement audit。
- `tools/quality_demo/summarize_metal_trace.py` 处理 xctrace 将 schema 与 rows 分在不同 XML node 的格式；检查目标 PID、marker PID、fallback 环境、stage begin/end 是否完整记录、PyTorchMPS signpost、Metal GPU interval 和 target exit。任何一项缺失都返回失败。
- `tests/test_summarize_metal_trace.py` 覆盖 sibling schema/rows、重复 schema、PID 不匹配、缺 signpost/GPU、fallback、截断窗口等 fail-closed 路径。

## 负结果与边界

- `p2-first-step-trace-02` 的目标 forward 已完成，但 recorder 在 marker end 前约 5 秒结束；新汇总器将 `fully_recorded=false` 判为失败，不把它算作 GPU 证据。
- 之前约 17 GiB 的整段 trace 及本次约 10 GiB 的 reference 启动期 trace 已保留/按交接约束处理；不要再录整流水线。
- 这些结果证明命名窗口内目标进程确实执行 Metal GPU 工作；不外推到显式 CPU preprocessing、mesh/UV/native bake、`.cpu().numpy()` 桥接或完整 pipeline 的“全 GPU”。P3 的真实 bake CPU 例外仍开放。
- 独立 reviewer dispatch 仍因 429 未返回；作者复查和正常 Metal 权限下的测试结果不能冒充独立 review。

## 验证

- 正常 Metal 权限：`python -m unittest discover -s tests -v` → **30/30 passed**。
- `py_compile`：trace 入口、汇总器和新增测试通过。
- `git diff --check`：通过。
- 受限无 Metal 沙箱下，MLX 导入会报 `No Metal device available`，不能用作本阶段失败证据。

## 下一阶段

P2 trace gate 关闭后进入 P3/P5：先处理真实尺寸 bake CPU 例外的数值/设备边界，再做相同 cache policy、固定实际随机张量的旧/新串行性能与内存配对。不要切换默认环境、不要覆盖交付 GLB、不要向 `origin` 推送；只推 `fork/codex/torch-mps-upgrade`。新会话应先读总交接和本文档，重新检查 branch/status/进程/磁盘后继续。

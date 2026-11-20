# P5 交接：旧/新版配对性能与内存样本（2026-10-28）

这是 P5 阶段交接；总交接仍是 [`HANDOFF-torch-mps-upgrade.md`](HANDOFF-torch-mps-upgrade.md)。

## 结果

在 ComfyUI queue 为空、一次只运行一个 GPU 任务的条件下，使用同一 `shape_quality.glb`、同一固定实际初始 diffusion noise、6 视角、512 分辨率、15 步和 1 GiB MLX 空闲缓存回收阈值，旧/候选各完成 3 次完整 paint benchmark。所有输出目录均 fresh；交付 GLB 未覆盖。

外部摘要：`/Users/daubert/Github/hy3d-upgrade-evidence/20261027/p5-performance-01/summary.json`。

| 栈 | Torch | fallback env | 总耗时中位数 | inference 中位数 | 范围 |
| --- | --- | ---: | ---: | ---: | ---: |
| 候选 worktree | 2.14.1 | 0 | 117.711 s | 107.722 s | 117.180–123.536 s |
| 保护旧 checkout | 2.5.1 | 1（旧版 bicubic 需要） | 196.740 s | 185.052 s | 194.004–199.497 s |

当前两套栈的完整 benchmark 中位数差约 `40.2%`，inference 差约 `41.8%`。这是可复现的栈级样本，**不能归因成 Torch 2.14.1 单独带来的加速**：旧 checkout 的 MLX reference layout/upsampler 实现不同，旧版需要 fallback=1；候选包含已修复的 MLX layout、upsampler 权重加载和 P3 MPS bake 路径。固定的是输入 mesh、初始 diffusion latents、cache threshold 和运行顺序，不是全部 VAE posterior/条件随机张量。

## 内存采样

计数来自每秒采样和阶段结束 counters；RSS、MPS driver、MLX allocator 互有重叠，不能相加；swap 是全机共享量，也不能归因给单进程。

| 栈 | RSS 峰值中位数 | MPS driver sampled 峰值中位数 | MLX peak 中位数 | swap delta 范围 |
| --- | ---: | ---: | ---: | ---: |
| 候选 | 8.20 GiB | 26.46 GiB | 5.64 GiB | −0.65–+3.32 GiB |
| 旧版 | 11.02 GiB | 36.67 GiB | 8.98 GiB | +6.97–+9.06 GiB |

这组结果显示候选栈的本次运行内存计数更低，但仍不是 48 GiB 安全内存验收：系统状态、采样间隔和计数重叠都限制了结论；没有 OOM 不是完整安全证明。

## 一致性与边界

- 6 次运行均正常生成结构有效的 GLB（24,087 vertices、40,000 faces）；输入 mesh SHA256、`initial-latents.npy` 和 1 GiB cache threshold 均逐次验证一致，runtime model source hash 也匹配。
- 同一栈三次输出并非逐字节相同，因为只固定了初始 diffusion noise，VAE posterior sampling 和 image conditions 没有 replay；不能从此声明端到端输出确定性。
- 新会话创建仍因本地 `codex_tui` transport `http://127.0.0.1:54618/mcp` 返回 HTTP send error 失败；没有绕过该错误，继续在当前会话完成 P5 样本。
- 这关闭了“完全没有性能证据”的门槛，但没有关闭 Torch-only attribution、用户设备/独立 3D 查看或默认环境切换门槛。后续需要把候选默认路径和完整质量/设备/内存决策分开。

## 可复核命令

候选和旧版均通过当前 worktree 的 `tools/quality_demo/audit_mps.py --mode benchmark` 运行；旧版使用 `--checkout /Users/daubert/Github/Hunyuan3D-2.1-macOS`，没有修改该 checkout。所有 GPU 实验串行执行，ComfyUI queue 每轮前后均为 `queue_running=[]`, `queue_pending=[]`。

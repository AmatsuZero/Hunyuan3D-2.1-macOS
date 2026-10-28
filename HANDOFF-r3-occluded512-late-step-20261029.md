# R3 occluded512 逐步差异离线诊断（2026-10-29）

本阶段只读取已归档的 resident trace，未启动 GPU、未修改模型代码或覆盖输出。工作区是
`/Users/daubert/Github/hy3d-torch-upgrade`，分支 `codex/torch-mps-upgrade`。
完整约束和迁移状态以 [`HANDOFF-torch-mps-upgrade.md`](HANDOFF-torch-mps-upgrade.md)
为唯一权威交接；本文件是该阶段的证据补充。

## 输入和算法

- 候选：`/Users/daubert/Github/hy3d-upgrade-evidence/20261029/r3-reference-winograd-optin-20261029/occluded512-replay/resident-trace/`，
  MLX reference fp32、generation fp16、Winograd-disabled opt-in。
- 基线：`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-reference-fp32-resident-01/occluded512/torch-reference/resident-trace/`，
  Torch reference。
- 对 15 步每一步的 `resident_noise_pred_{step}.npy` 和
  `resident_latents_{step}.npy`，按 float64 计算
  `NRMSE = sqrt(sum((candidate-reference)^2)/sum(reference^2))`；另外记录
  mean/max absolute error、最大误差索引以及 60 个源文件 SHA256。
- 明细：`/Users/daubert/Github/hy3d-upgrade-evidence/20261029/r3-reference-winograd-optin-20261029/occluded512-late-step-diagnostic.json`，
  SHA256 `26bd4518e8bb208d109ceea8f947090ef5a8a2cca00b7de39f493d05920fe38c`。
  JSON 写入前的 numpy `int64` 序列化错误已修正为原生 `int`；临时结果与归档
  `cmp` 一致。两组输入和 15×2 对数组均通过形状与 finite 检查。

| step | noise_pred NRMSE | latent NRMSE | noise 最大误差索引 `[batch,channel,y,x]` |
| ---: | ---: | ---: | --- |
| 0 | 0.00121164 | 0.00025900 | `[15,2,53,37]` |
| 5 | 0.00176847 | 0.00139006 | `[29,2,38,18]` |
| 10 | 0.00337343 | 0.00331994 | `[22,0,43,35]` |
| 11 | 0.00422777 | 0.00369466 | `[10,0,43,35]` |
| 12 | 0.00527246 | 0.00401737 | `[34,0,43,35]` |
| 13 | 0.00691702 | 0.00475034 | `[34,0,43,35]` |
| 14 | 0.01022438 | 0.00644233 | `[34,0,43,36]` |

误差从 step 0 已存在，后段累积；step 10–14 的 noise 最大误差空间坐标
集中在 `(y≈43,x≈35–36)`，但 batch 索引并不固定。这与上一阶段发现的
`generated_5/11.png` 少量超阈值像素只能并列记录，尚未建立张量位置到
解码图像像素的因果映射。occluded512 的探索性图像最大差仍是 `43/255`，
超过 `32/255` 门槛；不能切换默认 Torch reference、Winograd 或 MLX dispatch。

## 下一阶段

1. 先离线读取 `generated_5/11.png` 的局部差分与相应视角的 latent/noise
   索引，确认 view/batch 展开顺序及 VAE 解码映射。避免仅凭空间坐标相近
   就归因某个卷积。
2. 若需要 GPU 归因，使用已有 frozen inputs 做单 GPU、空 ComfyUI queue、
   1 GiB idle-cache、fresh output 的针对性 step 10–14 capture；优先核查
   reference cache 与 generation 的明确边界，再决定是否缩小 Winograd fallback。
3. 性能仍须同一宿主状态下交错配对重复。`107.848s` 的单次回退未在
   `83.437s` 复跑中出现，不能归因于 fallback。

保留原始 checkout、旧 venv、已交付 GLB 和根目录 dirty `torchvision_fix.py`；
不重跑形状，不并行 GPU。

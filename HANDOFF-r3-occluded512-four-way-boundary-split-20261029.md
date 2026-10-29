# R3 occluded512 四路边界与交互拆分（2026-10-29）

本阶段没有启动 GPU；只读取四路已归档 resident trace，做 CPU-only 的首步、CFG、
scheduler、latent 和图像拆分。唯一权威总交接为
[`HANDOFF-torch-mps-upgrade.md`](HANDOFF-torch-mps-upgrade.md)。

## 对照定义

- A：Torch reference / 默认 Winograd。
- B：MLX fp32 reference / 默认 Winograd。
- C：MLX fp32 reference / `HUNYUAN_MLX_DISABLE_WINOGRAD=1`。
- D：Torch reference / `HUNYUAN_MLX_DISABLE_WINOGRAD=1`。

四路 15 类输入均 exact，输入合同 SHA256
`d095fc0d8c128c0d6e7356d175df3aeb5d79f482b3e14801f3410f19266dc2c9`。
边界拆分 JSON：
`occluded512-four-way-boundary-split.json`，SHA256
`541b8436f1bc75a1512a543407a887e1ffff014a8086de56acddf9545808c2b1`。
交互组合 JSON：
`occluded512-four-way-interaction.json`，SHA256
`c5587f13f30379877e392549381cb6742b0496f3795129db51d988f35fbce44e`。
两份文件均从 `/private/tmp` 生成后复制到
`/Users/daubert/Github/hy3d-upgrade-evidence/20261029/r3-reference-winograd-optin-20261029/`，并通过 `cmp`。

## 结果

相对 A 的首步/后段 NRMSE：

| pair | reference_write | noise_pred_0 | step0 guided | step3 latent |
| --- | ---: | ---: | ---: | ---: |
| B vs A | 0.026173 | 0.001260 | 0.002348 | 0.001195 |
| C vs A | 0.013201 | 0.001212 | 0.001991 | 0.001018 |
| D vs A | 0 | 0.000837 | 0.001601 | 0.000917 |
| C vs D | 0.013201 | 0.000995 | 0.001443 | 0.000661 |

D 与 A 的 `reference_write`/cache exact，但 `noise_pred_0` 已不同，证明关闭
Winograd 会在 generation inner UNet 产生首步差异。C 与 D 保持相同 disabled
policy，仍有 reference_write 和 noise 差异，证明 reference forward 也影响轨迹。

四路交互组合 `C - B - D + A` 在 `reference_write` 的 NRMSE 为 `0.021470`，
在首步 noise 为 `0.001052`，step 14 latent 为 `0.012600`；image 5/11
的组合最大像素差分别为 `85/83`。这说明两条路径不是简单可加，不能从 C
相对 A 的较小 `43/255` 差异推出某一个开关已经修复问题。

## 结论边界

`HUNYUAN_MLX_DISABLE_WINOGRAD` 作用于整个 MLX 进程，因此 B↔C 同时改变
MLX reference 和 generation 卷积策略；只有 A↔D 是 generation policy 的
单变量隔离。当前证据已确认差异在 reference write 或首个 inner UNet 之前/之中
出现，并沿 CFG、scheduler、resident latent 继续传播；仍未定位具体 attention、
convolution 或 dtype kernel。默认 Torch reference、默认 Winograd 和默认 MLX
dispatch 继续不切换；`43/255` 探索性门槛继续失败。

## 验证与下一步

- 本阶段使用已归档的完整 resident trace，无 GPU 新运行。
- 宿主 Metal 全量 unittest `88/88 OK`，前一阶段已验证；本阶段只修改交接文档。
- 受保护的 `torchvision_fix.py` 未提交，原始 checkout、旧 venv 和 GLB 未触碰。
- 下一步优先对 A/D 的首步 inner UNet 局部层做单变量生成策略复核，并把 B/C
  仅作为 MLX 进程级交互对照；GPU 前重新检查 queue/process，仍一次一任务。

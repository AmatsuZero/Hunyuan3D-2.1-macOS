# R3 occluded512 reference / Winograd 四路对照（2026-10-29）

本阶段补齐了此前缺失的组合：**Torch reference + Winograd-disabled MLX
generation**。四路实验均使用同一 frozen input、初始 diffusion noise、15 步、6
视角、1 GiB MLX cache 和 `PYTORCH_ENABLE_MPS_FALLBACK=0`；每次只有一个 GPU
推理任务，ComfyUI `/queue` 为 running/pending 空。没有修改模型代码、默认路径或
任何已交付 GLB。

## 四路合同

| label | reference forward | generation Winograd | 来源 |
| --- | --- | --- | --- |
| A | Torch reference | 默认 | `20261028/r3-reference-fp32-resident-01/occluded512/torch-reference` |
| B | MLX fp32 reference | 默认 | `20261028/r3-reference-fp32-resident-01/occluded512/mlx-fp32` |
| C | MLX fp32 reference | disabled (`working_set=1`) | `20261029/r3-reference-winograd-optin-20261029/occluded512-replay` |
| D | Torch reference | disabled (`working_set=1`) | 本阶段 fresh capture，已归档为 `occluded512-torchref-winograd-off/` |

四路的 15 类初始/inner/reference/position 输入逐元素 exact；输入合同报告
`occluded512-four-way-input-contract.json` SHA256
`d095fc0d8c128c0d6e7356d175df3aeb5d79f482b3e14801f3410f19266dc2c9`。
D 的完整 capture 共 92 个文件，
临时目录与外部归档逐文件 SHA256 exact；capture SHA256
`0e22b3ee4b1656b7199681404f7dc46392f4ca09ecace10e3ca04ec743f78e5a`。
四路比较摘要：

- `occluded512-four-way-winograd-reference.json`：SHA256
  `e0808bce051da2fbecea210aa2a15957bf72e229a1d789b038388f0335b41ce1`。
- `occluded512-torchref-winograd-off-summary.json`：SHA256
  `12ed9485f55cd3bd4e6a4afee965b9aa46ddd19937c9a3922eb80f7b0fb633df`。

## 结果

下面的图像差是相对 A（Torch reference + 默认 Winograd）；A 自身为零。

| label | reference write NRMSE vs A | `noise_pred_0` NRMSE vs A | final latent NRMSE vs A | 12 图平均差 | 12 图最大差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B | 0.026173 | 0.001260 | 0.012732 | 0.10135/255 | 79/255 |
| C | 0.013201 | 0.001212 | 0.006442 | 0.05954/255 | 43/255 |
| D | 0 | 0.000837 | 0.005545 | 0.04693/255 | 66/255 |

D 的 reference_write 与 A exact，但仅关闭 Winograd 就产生新的 `noise_pred_0`
差异和 `66/255` 图像最大差；因此 generation Winograd policy 确实影响结果。
在相同 disabled policy 下比较 C/D，reference forward 改变使图像最大差为
`42/255`；在相同 MLX reference 下比较 B/C，Winograd 改变也明显改变轨迹。
四路的相互作用不是可线性相加：C 相对 A 的最大差为 `43/255`，D 相对 A
反而为 `66/255`，不能把 C 的较小差值归因于某一个开关单独修复。

超阈值位置也不稳定：A 对照下，B 的 image 5/11 分别有 42/148 个像素超过
32，C 为 17/2，D 为 41/14；D 的 image 5 bbox 为 `(205,305)-(305,335)`，
而 C 的 image 5 只集中在此前的局部窄带。该变化进一步说明必须保留完整四路
合同，不能只看一张视图或一个全局 NRMSE。

## 结论边界与下一步

这轮把问题从“reference 或 Winograd 单独导致失败”收窄为两条数值路径的交互：
reference write/cache 差异和 generation convolution policy 都能改变 resident
轨迹，当前没有证据授权默认切换或接受 `43/255` 门槛。下一步优先做 CPU-only
分析：对 A/B/C/D 的 `reference_write`、`noise_pred_0` 和 step 0→3 的 guided
输出做空间热点与分支对齐，确认交互是在 CFG 之前还是 generation convolution
之后放大。需要新 GPU 时仍必须 fresh output、空 queue、单任务、1 GiB cache。

本阶段不改默认 Torch reference、默认 Winograd 或默认 MLX dispatch；保护根目录
dirty `torchvision_fix.py`、原始 checkout、旧 venv 和已交付 GLB。

## 验证

- D capture exit 0，归档 92/92 文件 SHA256 与临时输出相同。
- 四路 15 类输入 exact；四路摘要和输入合同 JSON 已逐字节核对归档。
- 宿主 Metal 环境全量 unittest `88/88 OK`；`git diff --check` 通过。
- `torchvision_fix.py` SHA256 仍为
  `d6139690b099ca3ecef80ddf3f1a25edfdf57f027c79bf8655a8f8aac1c8b493`。

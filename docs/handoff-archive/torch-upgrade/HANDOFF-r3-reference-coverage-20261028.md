# R3 交接：reference 输入覆盖 replay（2026-10-28）

总交接仍为 `HANDOFF-torch-mps-upgrade.md`；本阶段基于提交 `2b742d8` 的 reference
数值门槛。实验只运行 `unet_dual(mode="w")` 到 reference-write，未执行完整生成，
未写入或覆盖任何 GLB。

## 实验范围

外部证据目录：
`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-reference-coverage-01/`

- `occluded512`：512px reference，在中心区域写入白色遮挡块；reference sample
  shape `(3,4,64,64)`。
- `reference256`：reference 图缩放到 256px；sample shape `(3,4,32,32)`。
- `reference384`：reference 图缩放到 384px；sample shape `(3,4,48,48)`。
- normal/position/DINO 条件从已核对的固定条件复制并注入，避免把其他条件生产
  重新混入本阶段；reference latents 由当前 case 的 reference 图在 Torch/MPS 上生成。
- 三次 Torch capture 都使用 `PYTORCH_ENABLE_MPS_FALLBACK=0`、真实动态模块来源
  SHA 校验、同一 checkpoint；随后每次单独运行 MLX/Metal 离线 replay。GPU 任务
  严格串行，ComfyUI 队列在首个 capture 前为空。

## 结果

门槛：cache 最大 NRMSE ≤ `0.01`，reference write NRMSE ≤ `0.03`；验证器同时检查
输入 exact、checkpoint SHA、686 个映射权重和 16 个 layer key。

| case | max cache NRMSE | worst layer | reference write NRMSE | verdict |
| --- | ---: | --- | ---: | --- |
| `occluded512` | `0.0102111259` | `up_3_0_0` | `0.0277900531` | FAIL，cache 超限 |
| `reference256` | `0.0026610550` | `up_3_2_0` | `0.0130105792` | PASS |
| `reference384` | `0.0142296427` | `up_2_0_0` | `0.0255110706` | FAIL，cache 超限 |

全部报告都满足：`inputs_exact=true`、checkpoint SHA256
`675a1b5cd0098b2002637c443946529c03c5cd54427f40245263350feb3dd5b8`、`mapped_weights=686`、
16 层完整；Torch capture 的 outer UNet/reference processor 均匹配模型源码，fallback=0。
每组的报告、输入和 sample 哈希记录在 `summary.json`。

## 结论与边界

- 当前 `0.01/0.03` 门槛对固定 512px 无遮挡基线和 256px case 通过，对 512px 遮挡及
  384px case 失败。**不放宽门槛，也不把失败 case 当作 MLX reference 可接受。**
- 超限集中在后段 up block cache，reference-write 本身三组都在 `0.03` 内；这把下一
  个诊断重点收窄到 fp16 reference forward 的空间尺寸/上采样误差，仍不能直接归因
  为 Torch 或 MLX kernel bug。
- 结果只覆盖 reference-write 数值，未证明完整 15 步图像、颜色、用户设备或默认
  dispatch；默认路径继续使用 Torch reference producer，opt-in 仍不可切换默认。

## 下一会话入口

1. 先读总交接、本交接和 `summary.json`；不要重跑已通过的 DINO/颜色根因实验。
2. 用同一三组输入先做 MLX reference fp32/逐层 stage 对照（仍一项 GPU 任务），
   确认超限是否来自 dtype 累积；若 fp32 不能消除，比较 up block 的具体算子和
   checkpoint layout。保持 `0.01/0.03` 门槛不变。
3. 只有遮挡/多分辨率 reference replay、完整固定条件图像和 reset 都通过后，才评估
   outer condition/VAE/最终 Torch 边界；完成阶段后写交接、提交推送并创建新会话。

## 复现入口

每个 case 的 Torch capture 目录含 `capture.json` 与 16 个
`condition_embed_*.npy`；MLX 结果为 `mlx/exact-input-hashed-metrics.json`。输入制作
脚本和汇总脚本已复制在证据目录的 `setup_inputs.py`、`summarize.py`。门槛命令：

```sh
rtk proxy /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python \
  tools/quality_demo/validate_mlx_reference_report.py \
  --report /Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-reference-coverage-01/<case>/mlx/exact-input-hashed-metrics.json
```

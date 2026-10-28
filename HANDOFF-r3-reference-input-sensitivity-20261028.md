# R3 交接：reference 输入敏感性与精度契约诊断（2026-10-28）

总交接为 `HANDOFF-torch-mps-upgrade.md`；本阶段基于 `96115f3`。本阶段只做固定
输入的 reference forward 诊断，没有修改模型代码、容差、默认 dispatch 或交付资产。

## 已确认

- 对本轮 `occluded512` resident 失败输入，逐层 MLX fp32 cache 与 Torch reference
  的误差在 `up_2_0_0` 首次明显越过旧门槛：`up_1_2_0=0.005139`，
  `up_2_0_0=0.011415`，随后 `up_2_1_0=0.011192`、`up_2_2_0=0.014330`、
  `up_3_0_0=0.013191`。更早的 down/mid/up1 层均低于 `0.006`。
- 对同一失败 sample，用相同 fp32 MLX 参数分别运行两种契约：正式 dispatch（输入
  fp32、cache cast fp16）得到 cache `0.0143295248`、write `0.0261733204`；
  离线 replay（输入保持 fp16、cache 不 cast）得到 cache `0.0143283580`、write
  `0.0261686476`。差异约 `1e-6`，输入 dtype 不是主因。
- 旧 coverage 的精确 `occluded512` 输入在同一正式 dispatch 诊断探针下通过：
  cache `0.0063985774`、write `0.0230500719`；离线参数 fp32 为
  `0.0063950941 / 0.0230458305`。旧 PASS 与本轮 FAIL 由精确输入合同区分。
- 新旧遮挡输入的 reference timestep 与 encoder hidden states 逐元素相同；reference
  sample NRMSE `0.0000270226`（49152 元素中 4503 个变化，最大绝对差
  `0.0009765625`）。旧/新 Torch cache 最大 NRMSE `0.0047341793`，最大层为
  `up_2_2_0`。微小 sample 改变会在上采样路径放大，具体算子仍未定位。

## 证据

- resident 完整归档：`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-reference-fp32-resident-01/`
- 旧 coverage：`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-reference-coverage-01/`
- 本轮逐层结果：`r3-reference-fp32-resident-01/occluded512/mlx-fp32/capture.json`。
- 临时精度契约探针：`/private/tmp/hy3d_reference_precision_contract_probe.py`；报告：
  `/private/tmp/hy3d-reference-precision-contract-probe-20261028.json` 和
  `/private/tmp/hy3d-reference-precision-contract-probe-old.json`。探针不属于版本化代码。

## 下一阶段

1. 为 reference forward 增加临时逐层 trace，至少覆盖 up block 2 的输入、ResNet、
   attention、上采样和 cache 写入；同一 sample 分别跑 Torch 与 MLX fp32，定位首个
   数值分叉，不改变 gate。
2. 以旧 coverage sample 和本轮 sample 做 A/B，确认差异来自 MPS/Torch 数值敏感性、
   MLX 上采样/attention 算子，还是 checkpoint layout，再设计单变量修复。
3. 修复前不切默认 reference forward，不扩大 outer condition/VAE 迁移范围。保持一项
   GPU 推理、fresh output、`torchvision_fix.py` dirty 保护、原始 checkout 与 GLB
   保护。阶段完成后重新写交接、验证、推送并创建新任务。

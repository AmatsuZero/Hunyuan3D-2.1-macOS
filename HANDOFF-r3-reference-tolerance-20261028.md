# R3 交接：MLX reference 数值门槛（2026-10-28）

总交接仍为 `HANDOFF-torch-mps-upgrade.md`；前一阶段实现见
`HANDOFF-r3-mlx-dual-reference-forward-20261028.md`。

## 本阶段结果

- `tools/quality_demo/validate_mlx_reference_report.py` 为真实同输入 replay 报告
  固定首版工程门槛：16 层 cache 的最大 NRMSE ≤ `0.01`，reference write NRMSE
  ≤ `0.03`。任一超限返回 exit 1。
- 验证器先检查输入逐元素相同标志、checkpoint SHA256、686 个已映射权重和完整
  16 层名称；缺失、漂移或非有限指标会失败。只读取报告 JSON，不运行 GPU。
- 现有报告
  `/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-mlx-reference-replay-01/exact-input-hashed-metrics.json`
  通过：最大 cache NRMSE `0.006851671088888069`、reference write NRMSE
  `0.02634890708159594`。这是固定条件 512px、6 views 的 fp16 近似门槛，尚未
  对有遮挡输入或其他支持分辨率验收。
- 新增 `tests/test_mlx_reference_tolerance.py`，覆盖门槛通过、两类超限、输入不等、
  缺层和 checkpoint 漂移。

## 复现

```sh
cd /Users/daubert/Github/hy3d-torch-upgrade
rtk proxy /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python \
  -m unittest tests.test_mlx_reference_tolerance -v
rtk proxy /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python \
  tools/quality_demo/validate_mlx_reference_report.py \
  --report /Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-mlx-reference-replay-01/exact-input-hashed-metrics.json
rtk proxy git diff --check
```

## 下一会话

1. 核对当前分支、HEAD、远端 SHA 与受保护的根目录 `torchvision_fix.py`；读总交接
   §3 及本交接。默认 hybrid 仍使用 Torch reference，opt-in 不可据此切默认。
2. 在 fresh 证据目录中，串行捕获有遮挡输入及其他支持分辨率的 Torch/MLX 同输入
   reference replay；每份报告运行上述门槛验证器。运行前查进程及 ComfyUI queue，
   一次仅一个 GPU 推理。不同尺寸可能改变层序列长度，16 层名称合同仍固定。
3. 门槛、视觉与重复性通过后，继续 outer condition/VAE/最终 Torch 边界迁移；
   每阶段写新交接、提交推送，并按用户授权创建新 Codex 任务。

当前会话接力服务数次因 `429 Too Many Requests` 终止，本阶段改在原会话完成
离线门槛。独立代码审查仍未完成；不把该限制写成审查通过。

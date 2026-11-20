# R3 交接：隔离 reference Transformer2D epsilon（2026-10-28）

总交接为 `HANDOFF-torch-mps-upgrade.md`；本阶段基于 `55fee6e`。这是配置和
单元合同的阶段交付，不是 reference 质量闭环。

## 改动

- `Transformer2D` 和 `HunyuanInnerUNet` 支持显式 Transformer2D GroupNorm epsilon；
  默认仍为原 `1e-5`，generation 路径不改。
- 显式 `HUNYUAN_MLX_REFERENCE_FORWARD=1` 的 reference 模型构造传入 `1e-6`，
  与 Diffusers Transformer2D GroupNorm 一致；默认 reference 仍走 Torch。
- `replay_mlx_reference.py` 和 `trace_mlx_reference_up2.py` 新增
  `--transformer-norm-eps`，默认 `1e-6`，重现旧诊断可显式传 `1e-5`。
  两者报告写出 epsilon；`probe_fixed_conditions.py` 的 dispatch 报告增加实际
  reference epsilon。
- `tests/test_mlx_reference_norm.py` 验证 16 个 reference Transformer2D 设置、
  其他 norm 不变、generation 默认 `1e-5`，以及零中心低方差输入上的 Torch 对照。

## 验证及限制

- 全量 unittest `85/85`，`Ran 85 tests in 1.005s`，`OK`；`py_compile` 和
  `git diff --check` 通过。
- 新会话留下的测试在接口不存在时因缺失构造参数产生两条 `TypeError`，后来新增
  接口后配置测试通过。低方差测试最初用均值约 1 的输入，MLX CPU/Torch GroupNorm
  最大差约 `0.0175`，随后使用零中心输入确认 epsilon 合同。该前者差异需独立调查，
  不应当作 epsilon 修复已证明 backend 完全一致。
- 上一阶段的临时单变量试验得到失败样本 cache `0.0130184897` > `0.01`；该次
  replay 的 ComfyUI queue 检查在自动审查 429 后未成功，见
  `HANDOFF-r3-reference-norm-epsilon-20261028.md`。因此它只是数值线索。
- 本阶段尝试用新版本 `replay_mlx_reference.py` 重跑冻结失败输入：队列和进程
  检查均通过，但启动命令的自动审查返回 `429 Too Many Requests`，命令未执行，
  `/private/tmp/hy3d-reference-eps-newcode-20261028.json` 不应视为已有结果。
  新实现的 checkpoint replay、正式 fp32 dispatch、15 步图像和 reset 尚未验收。
- 未放宽 cache `0.01` / write `0.03`；不得据单元测试宣称迁移完成或切默认。
  一次一个 GPU 任务；不碰原始 checkout、旧 venv、已交付 GLB 和根目录 dirty
  `torchvision_fix.py`。

## 下一阶段

1. 成功核对 ComfyUI queue 为空且无其他 GPU 推理进程，再运行以下冻结输入 replay；
   自动审查未通过时不启动 GPU：

```sh
rtk proxy env PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONUNBUFFERED=1 \
  /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python \
  tools/quality_demo/replay_mlx_reference.py --fp32 \
  --capture /Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-reference-fp32-resident-01/occluded512/torch-reference \
  --torch-reference /Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-reference-fp32-resident-01/occluded512/torch-reference \
  --checkpoint /Users/daubert/Github/hunyuan3d-artifacts/Hunyuan3D-2.1-modelscope/hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin \
  --output /private/tmp/hy3d-reference-eps-newcode-20261028.json
```

2. 注意该 replay 只提升参数，输入依旧 fp16；正式 dispatch 会提升输入并把 cache
   cast 回 fp16，需单独对照。若失败样本仍超过门槛，追踪更早 down/mid/up0，
   包括高均值低方差 GroupNorm 的同输入 CPU/MPS/MLX 对照。
3. 只有 reference gate 通过后才执行完整 15 步、reset 与图像验收；完成阶段后
   继续写交接、验证、提交推送并创建新会话。

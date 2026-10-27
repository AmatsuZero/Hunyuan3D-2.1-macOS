# R3 交接：外层条件权重 bundle（2026-10-28）

总交接仍是 `HANDOFF-torch-mps-upgrade.md`。这是 R3 的条件权重切片，**完整 R3 尚未完成**。

## 已完成

- `hy3dpaint/mlx_condition_bundle.py` 定义 `hy3d-paint-condition-mlx-v1`，严格校验
  实际 outer UNet 的 7 组权重、形状、float16、source/config SHA256、MLX 版本和
  参数数量；只使用 MLX 加载，不导入 Torch。
- `tools/quality_demo/convert_mlx_condition_bundle.py` 是离线 Torch converter，
  从可信 checkpoint 提取 `image_proj_model_dino.{proj,norm}` 与
  `learned_text_clip_{albedo,mr,ref}`，拒绝缺失或额外 key，并拒绝覆盖输出。
- `tools/quality_demo/check_mlx_condition_bundle.py` 在 Torch import guard 子进程中
  完成 bundle load；`tests/test_mlx_condition_bundle.py` 覆盖 round-trip、metadata、
  完整 key 集和防覆盖。
- `tools/quality_demo/probe_mlx_dino_projector.py` 支持从该 bundle 加载 projector，
  对固定 DINO fixture 做 Torch/MPS 与 MLX/GPU 对照。

## 真实证据

持久目录：`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-condition-bundle-01/`。

- bundle SHA256：`94084b1ea331ec83f1d52596ecb799b52fea5ce8247777b974da4da9143fa981`。
- source checkpoint SHA256：`675a1b5cd0098b2002637c443946529c03c5cd54427f40245263350feb3dd5b8`。
- config SHA256：`ce0c6d379e3b1d3e1f79338de70c80a1e36f17a6372439a8249b6fb0dfa1b608`。
- Torch-forbidden load：`status=passed`、`torch_imported=false`、7 keys、float16。
- bundle-backed DINO projector output：`r3-dino-projector-bundle-parity-01.json`；
  输入 `1×257×1536`，输出 `1×1028×1024`，NRMSE `0.0003501493`、mean abs
  `0.0001003800`、max abs `0.00390625`、finite=true；与直接 checkpoint 对照一致。

## 边界与后续

这是权重来源和 projector 的独立验收，不是 outer UNet 完整迁移。`condition_embed_dict`
仍由每个请求的 Torch `unet_dual(mode="w")` 产生，依赖参考图、reference token、层名和
序列长度；不能把它当成静态 bundle。下一步应做单请求 cache contract capture，验证
layer-keyed reference tensors、position voxel indices、reset，再比较 MLX reference
attention 输入和首步输出，最后才能接入正式 15 步循环。

遵守总交接 §3：一次只跑一个 GPU 推理；不覆盖已交付 GLB；不提交或清理根目录
`torchvision_fix.py`；不在原始 checkout 重编扩展。

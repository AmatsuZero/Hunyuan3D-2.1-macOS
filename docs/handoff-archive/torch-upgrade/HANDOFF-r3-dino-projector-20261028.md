# R3 交接：DINO projector 独立数值对照（2026-10-28）

总交接仍是 `HANDOFF-torch-mps-upgrade.md`。这是 R3 的外层条件切片，**完整 R3 尚未完成**。

## 已完成

- `hy3dpaint/mlx_dino_projector.py` 实现实际模型 `unet/modules.py` 中
  `ImageProjModel` 的 linear、4 context tokens、LayerNorm 和 token 重排；模块
  不导入 Torch。只接受 `[B,C]` / `[B,N,C]` 输入。
- `tests/test_mlx_dino_projector.py` 先因模块缺失失败，随后小尺寸二维/三维输入
  对照 Torch `Linear` + `LayerNorm` 通过，验证 token 顺序。
- `tools/quality_demo/probe_mlx_dino_projector.py` 从可信 checkpoint 读取且只复制
  4 组 projector 权重，以固定条件 DINO 张量串行对比 Torch/MPS 和 MLX/GPU；
  报告拒绝覆盖已有文件。它是诊断入口，不参与正式推理。

## 真实数值证据

`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-dino-projector-parity-01.json`：
输入 `1×257×1536` float16，输出 `1×1028×1024` float16；mean absolute error
`0.00010038`，max absolute error `0.00390625`，NRMSE `0.00035015`，输出全有限。
固定条件 DINO 文件 SHA256：
`5abf066b211e3d83263a479abb54468547b5cade35cb02a5b93612a851177060`。
对照前 ComfyUI queue 为空，只有 ComfyUI 服务进程；Torch/MPS 与 MLX/GPU 在
同一进程内顺序执行，没有并行推理。

这证明单独 projector 的真实权重数值接近；尚未证明 MLX outer loop、reference
write、首步/cache、原始颜色或 Metal trace。R1 inner UNet bundle 明确排除这 4
组权重；后续必须为外层条件提供有来源校验的独立权重 bundle，不能用 inner bundle
假装它们已经加载。

## 后续

1. 为 DINO projector 与 learned text embeddings 建立独立、来源校验的条件权重
   bundle，固定真正运行的模型目录 `unet/modules.py` 合同。
2. 迁移 position voxel indices 与 reference-write/cache，验证跨请求 reset；再将
   首步、CFG 与已完成的 MLX UniPC scheduler 接入正式 15 步循环。
3. 逐步数值、原始 albedo/MR 颜色、设备 trace、性能和内存都过门槛后才考虑
   默认 dispatch；目前仍是 hybrid。

遵守总交接 §3：一次只跑一个 GPU 推理；不覆盖已交付 GLB；不提交或清理根目录
`torchvision_fix.py`；不在原始 checkout 重编扩展。

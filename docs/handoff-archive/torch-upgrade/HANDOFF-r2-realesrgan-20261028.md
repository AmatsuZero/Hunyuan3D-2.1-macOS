# R2 交接：MLX RealESRGAN 试点（2026-10-28）

## 结果

- 新增 `hy3dpaint/mlx_realesrgan.py`：Torch-free MLX RRDBNet，保留 RealESRGAN
  x4 的 23 个 RRDB、nearest×2×2、RGB/BGR 边界和 uint8 clamp/round 语义。
- 新增离线 converter `tools/quality_demo/convert_mlx_realesrgan.py`。Torch 只在
  转换入口导入；运行时通过 `.safetensors` metadata + `strict=True` 加载。
- R2 focused tests 6/6、全量 unittest 48/48 通过（候选环境、正常 Metal 权限）。

## 真实证据

外部目录：`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r2-realesrgan-01/`。

- bundle `rrdb.safetensors` 32 MiB，schema `hy3d-realesrgan-mlx-v1`，MLX 0.32.4，
  float16，702 个参数；source checkpoint SHA256
  `4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1`，bundle SHA256
  `b14bcfbd14ab03e1f1bad60ca0fa463d0ff120f7c7ceaac771f18df73869e20e`。
- 同一 512×512 albedo fixture 的 MLX 真实输出为 2048×2048；Torch/MPS 对照逐像素
  exact `88.66%`，MAE `0.114/255`，p99 `1`，max `5`，NRMSE `0.002540`。
- MLX 三次串行稳态 `2.131/2.135/2.159 s`，peak `3.035 GiB`；Torch/MPS 三次
  `0.859/0.865/0.940 s`。MLX 当前约慢 `2.3–2.5×`，因此没有切换默认超分路径。
- MLX smoke `torch_imported=false`；奇数尺寸 `[5,7,3]` 回归覆盖输出 `[20,28,3]`。

## 边界与下一步

R2 模块正确性和真实尺寸 parity 已通过，但默认 `imageSuperNet` 仍是 Torch，混合
pipeline 尚未切换超分模块。性能负结果保留为决策证据；下一步优先 R3，消除 diffusion
循环中每步 Torch↔MLX 往返，再测端到端收益。不要把 R2 局部 MLX 速度外推为整条流水线
加速，也不要把外部 bundle 加入 Git。

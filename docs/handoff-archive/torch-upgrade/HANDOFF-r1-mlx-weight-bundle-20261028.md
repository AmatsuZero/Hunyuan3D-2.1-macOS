# R1 交接：MLX inner UNet 权重 bundle（2026-10-28）

这是 MLX roadmap R1 阶段交接；总交接仍是 [`HANDOFF-torch-mps-upgrade.md`](HANDOFF-torch-mps-upgrade.md)。

## 结果

R1 的离线转换和 Torch-free 加载门槛已通过。现有 Torch-backed loader 继续作为离线转换的唯一映射来源；新的运行时 bundle loader 不导入 Torch，并使用 MLX 0.32.4 的 `.safetensors` + metadata + `strict=True` 参数加载。

代码：

- `hy3dpaint/mlx_weight_bundle.py`：纯 MLX loader，校验 schema、source/config SHA256、MLX 版本、dtype、parameter count、layout，并调用 `model.load_weights(..., strict=True)`。
- `tools/quality_demo/convert_mlx_weight_bundle.py`：离线 Torch converter；记录实际 consumed keys，拒绝未映射 inner keys，原子写入 bundle 和 manifest。
- `tools/quality_demo/check_mlx_weight_bundle.py`：插入 import guard，禁止 `torch` 出现在 `sys.modules`，执行真实 bundle strict load。
- `hy3dpaint/mlx_utils.py`：增加 `return_used_keys=True` 的审计返回值；默认混合 dispatch 行为不变。

提交前的 focused tests：8/8 passed。真实 bundle 的转换第一次因 4 个外层 DINO projection 参数未分类而 fail closed；把这 4 个参数和 3 个 learned text embedding 明确列入排除清单后重跑通过。

## 真实 bundle 证据

外部目录：`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r1-mlx-weight-bundle-01/`。

- `inner_unet.safetensors`：约 2.0 GiB，不进入 Git。
- `manifest.json`：schema `hy3d-inner-unet-mlx-v1`，source checkpoint SHA256 `675a1b5cd0098b2002637c443946529c03c5cd54427f40245263350feb3dd5b8`，config SHA256 `ce0c6d379e3b1d3e1f79338de70c80a1e36f17a6372439a8249b6fb0dfa1b608`，MLX `0.32.4`，dtype `float16`。
- source keys `1747`；inner keys `1061`；MLX parameters `1054`，全部映射；`unet_dual` `686` 个键不进入 inner bundle。
- 明确排除的 7 个 inner keys：`learned_text_clip_{albedo,mr,ref}` 和 `image_proj_model_dino.{norm,proj}.{weight,bias}`；它们属于外层/条件路径，不是当前 inner UNet 参数。
- `check.json`：全新 `-I` 子进程、Torch import guard、严格 source/config/dtype 校验通过；parameter count `1054`，`conv_in` `[320,3,3,12]`，`up_block_0` upsample `[1280,3,3,1280]`，加载约 `0.078 s`。

代表性参数的 safetensors 转换后 SHA256 与 PyTorch OIHW→MLX OHWI 参考逐项一致；这验证了卷积布局和 upsampler 权重没有丢失。

## 边界与下一步

- 当前默认 `setup_mlx_dispatch` 仍从 PyTorch checkpoint 建立 hybrid wrapper；R1 bundle 尚未切换默认 dispatch，也不宣称已完成全 MLX 推理。
- R1 只证明权重合同和 Torch-free 加载；没有运行 GPU forward，也没有改变当前已验收的颜色路径。
- 下一步进入 R2/R3：先在真实尺寸 RealESRGAN 或 diffusion loop 中接入 bundle，逐模块对比输出、cache/reset、性能和设备证据；每次只改变一个后端模块。

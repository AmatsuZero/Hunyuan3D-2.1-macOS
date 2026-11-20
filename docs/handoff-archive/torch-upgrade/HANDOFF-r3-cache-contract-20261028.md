# R3 交接：outer reference/cache contract（2026-10-28）

总交接仍是 `HANDOFF-torch-mps-upgrade.md`。这是 R3 的真实请求 cache 切片，**完整 R3 尚未完成**。

## 已完成

- `tools/quality_demo/probe_fixed_conditions.py --capture-cache` 保存实际
  `unet_dual(mode="w")` 写出的 layer-keyed `condition_embed_dict`、首个生成调用的
  position voxel indices 和 generation encoder hidden states；只用于诊断 capture。
- `tools/quality_demo/check_paint_cache_capture.py` 验证固定 512px/6 视角/3 CFG 分支
  的 16 个 reference layers、4 个 position sequence keys、dtype/shape/resolution，
  并对 reference-only 与 first-step 两次独立请求比较 cache 是否逐元素一致。

## 真实证据

外部目录：

- `r3-cache-contract-01/`：单次 Torch/MPS reference-write；16 个 layer cache，首层
  shape `(3,4096,320)`，没有进入首个生成 step。
- `r3-cache-contract-02/`：单次 Torch/MPS reference-write + 首步；16 个 layer cache、
  position keys `384/1536/6144/24576`，对应 voxel resolutions `64/128/256/512`；
  position shapes `(3,seq,3)`、dtype int64；generation encoder shape `(36,77,1024)`。
- `r3-cache-contract-compare-01.json`：两个独立请求的 16 个 reference cache tensors
  逐元素完全一致；首步 capture 的 4 个 position entries 通过 verifier。

这些证据固定了可迁移边界：reference cache 依赖 reference image、reference token、
layer key 和 sequence length，不能由静态权重单独重建。position cache 是请求级、分辨率
和 view count 敏感，必须在新请求清空。generation encoder 已被确认是 3 CFG 分支 × 2 PBR
× 77 tokens × 1024 channels 的实际输入。
两次 capture 来自不同进程，尚未证明同一个 pipeline 实例连续请求时的 reset。

## 后续

1. 在 MLX inner UNet 中直接消费这些冻结 cache，先做 reference-attention 输入和首步
   noise prediction 的逐层/逐张量对照；保留 Torch outer write 作为过渡。
2. 接入 bundle-backed learned text/DINO projection、position indices 和 MLX scheduler，
   再消除首步及逐步 Torch bridge。
3. 通过每步数值、cache reset、原始颜色、设备 trace 和性能门槛后才考虑默认 dispatch。

遵守总交接 §3：一次只跑一个 GPU 推理；不覆盖已交付 GLB；不提交或清理根目录
`torchvision_fix.py`；不在原始 checkout 重编扩展。

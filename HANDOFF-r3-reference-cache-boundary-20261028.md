# R3 交接：动态 reference cache 的 MLX 边界（2026-10-28）

总交接仍是 `HANDOFF-torch-mps-upgrade.md`。本阶段为 16 层动态 reference cache
建立请求内 MLX 转换与 reset 合同；**reference `unet_dual` forward 仍由 Torch/MPS
执行，尚未迁移为 MLX producer**。

## 改动

- `hy3dpaint/mlx_reference_cache.py` 新增 `MLXReferenceCacheProducer`，接收当前
  Torch reference-write 产生的 16 层 rank-3 浮点张量，验证层数、名称、有限值，
  一次性转换并 eval 为 MLX arrays；失败时清除旧 cache，请求重置时释放引用。
- `hy3dpaint/mlx_utils.py` 的 inner dispatch 仅在
  `HUNYUAN_MLX_REFERENCE_CACHE=1` 时使用该边界；默认转换路径不变，dispatch
  `reset()` 同步重置 reference cache。
- 固定条件 capture 在 `mlx_resident_loop.reference_cache_mlx` 记录 opt-in 状态；
  `tests/test_mlx_reference_producer.py` 覆盖转换、拒绝坏输入及跨请求 reset。

## 真实对照

新目录：`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-mlx-reference-cache-01/`
基线：`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-mlx-position-indices-01/`

- 单 GPU、fallback=0、相同 fixture/初始 noise、512px、6 views、15 steps、
  resident loop、1 GiB MLX idle-cache threshold；exit 0，新 capture 耗时
  `63.445 s`，生成 12 张 PNG。基线单次 `62.142 s`；样本不足以判断性能变化。
- 16 个 `condition_embed_*` 数组、`reference_write`、`inner_output`、
  `noise_pred_0`、首步输入/DINO/encoder states 与四组 position indices
  逐元素一致；12 张 PNG 逐字节相同。
- 动态 outer UNet 与 reference processor 来源 SHA 均与模型源码相符；
  `reference_cache_mlx=true` 只表示 cache 转换/持有在 MLX，**不表示 reference
  forward 在 MLX**。

## 验证与边界

- Metal 可用候选 venv 全量 unittest：`73/73` 通过；定向 reference/position
  测试 `6/6` 通过；相关文件 `py_compile` 与 `git diff --check` 通过。
- 默认 hybrid 路径不启用新开关；没有切默认、覆盖 GLB 或改模型动态源码。
- 当前边界仍有 Torch→CPU/NumPy→MLX 同步和 Torch reference `unet_dual`
  权重驻留。此切片只为下一步真正 MLX dual reference forward 提供明确的
  cache 消费合同，不能宣称 R3 reference producer 完成或全 MLX。

## 下一优先级

1. 从已核对的 source checkpoint/配置提取 `unet_dual` 权重，严格检查键表、
   dtype、layout 与 source SHA；构建独立、可回退的 MLX reference forward。
2. 用固定的 `ref_latents`、learned text embedding 和 timestep 0，在真实尺寸
   离线 replay 16 层 `condition_embed_*`；逐层报告误差并检查颜色敏感层。
3. 数值与请求 reset 通过后才以 opt-in 替换 Torch reference-write；再做
   15 步逐步 trace、12 张原图和内存/性能对照。现有 boundary capture 不可替代
   这些门槛。

复现命令参照 `HANDOFF-r3-position-indices-20261028.md` 的唯一新 output
`--stage capture` 命令，再增加 `HUNYUAN_MLX_REFERENCE_CACHE=1`；运行前先确认
无其他 GPU 推理且 ComfyUI queue 为空。保护根目录 dirty `torchvision_fix.py`、
已交付 GLB 和原始 checkout 的原生扩展。

# R3 交接：MLX resident denoise loop（2026-10-28）

总交接仍是 `HANDOFF-torch-mps-upgrade.md`。这是 R3 的 resident latent loop
阶段，默认路径没有切换。

## 已完成

- `hy3dpaint/mlx_utils.py` 增加 `MLXDispatchUNet.forward_mlx()`：消费已经由
  outer Torch 首次请求建立的 reference/position cache，直接接收和返回 MLX
  arrays，不调用 `.cpu().numpy()`。
- `hy3dpaint/mlx_resident_denoise.py` 实现 opt-in resident loop：
  - 第一个 outer generation 调用仍负责 reference-write、DINO/position/cache
    准备；
  - 首次 CFG + UniPC 更新以及后续更新在 MLX 中完成；
  - 第 2–15 步的 latent、CFG、UniPC history 和 inner UNet 输入/输出保持在
    MLX；最后只转换一次 Torch latent 给现有 VAE/image postprocess；
  - 支持外部逐步 trace，trace 不进入 Git。
- `hunyuanpaintpbr/pipeline.py` 增加显式环境变量
  `HUNYUAN_MLX_RESIDENT_LOOP=1`。该路径只接受 MPS、`HUNYUAN_MLX_FIRST_STEP=1`、
  UniPC、2 个 PBR、3 分支 CFG、无 callback/added conditioning 的当前 paint
  配置；不满足时直接失败，不静默改变结果。默认环境完全不进入该分支。
  FP32 MLX 和非 identity scheduler input scaling 也会失败，防止版本漂移时
  继续使用未经验证的数值路径。
- `probe_fixed_conditions.py` 为 resident capture 写入环境/trace 元数据。

## 代码验证

- 全量 unittest：`63/63` 通过。
- `py_compile`：resident loop、pipeline、dispatch、capture 工具通过。
- resident 单元测试验证：
  - 两步 MLX loop 与 Diffusers UniPC 对照；
  - inner dispatch 收到 MLX arrays，测试中禁止 `_pt_to_mx` bridge；
  - shape/dtype、CFG branch、request reset 合同。

## 真实 15 步证据

运行约束：单 GPU、ComfyUI queue 为空、`PYTORCH_ENABLE_MPS_FALLBACK=0`、
1 GiB MLX idle-cache threshold、fixed 512px/6 views/15 steps/同一初始 noise。

- capture：`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-mlx-resident-loop-02/`
- resident trace：该目录下 `resident-trace/`
- comparison：`r3-mlx-resident-loop-compare-01.json`
- capture SHA256：`f43b1c7f975592b9aaf10cbe537fe393cfba1d7e8122484a5f6c4326625a3a55`
- elapsed：`60.954 s`，生成 12 张图，fallback=0，runtime source audit 通过。
- 相对既有 Torch-first/MLX-inner 15 步 capture `r3-mlx-first-step-full-01`：
  - 第 0 步 raw noise prediction 逐元素相同；
  - resident raw noise 最大 NRMSE `0.00903750`；
  - CFG scheduler output 最大 NRMSE `0.01585895`；
  - final latent NRMSE `0.00425918`；
  - 12 张图平均像素差 `0.04558/255`，最大 `30/255`；
  - 所有 resident trace 数组 finite，shape 与基线一致。
- 同实例 reset capture：`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-mlx-resident-repeat-01/`。
  两个完整请求在同一 pipeline 进程串行完成；每次请求后 `_gen_call_count=1`，
  16 层 reference cache `reference_cache_exact=true`，12 张图 `max_image_abs=0`。
  第一/第二请求耗时分别 `67.846s`/`60.592s`。

## 边界与未完成项

- 这不是全 MLX：outer reference-write、DINO、VAE、条件准备仍是 Torch；首步
  outer 调用返回的 prediction 仍有一次 Torch→MLX 输入桥接。第 2–15 步不再
  经过 Torch scheduler/CFG 或 Torch latent 往返。
- resident loop 是显式 opt-in，尚未默认启用；没有完整 GLB 产物，也没有用户
  设备/独立 3D 查看验收。
- 本阶段没有修改模型目录或 Diffusers dynamic module source，因此既有 runtime
  source SHA audit 仍通过。

## 下一步

1. 把首个 MLX inner output 直接作为 resident loop 的 MLX 输入，去掉当前唯一的
   首步 output bridge，并保存首步逐元素对照。
2. 在不改 dynamic model source 的前提下，将 outer DINO projector、normal/
   position embedding 和 reference cache 的消费边界逐步移到 MLX；每个阶段单独
   capture/reset/质量验证。
3. resident loop 的完整性能、Metal trace、内存生命周期和 GLB 质量通过后，才
   讨论默认切换。

遵守总交接约束：一次只跑一个 GPU 推理；不覆盖已交付 GLB；不提交或清理根目录
`torchvision_fix.py`；不在原始 checkout 重编原生扩展。

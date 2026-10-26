# Hunyuan3D 2.1 macOS / MLX 改造计划

> 当前分支：`macos-baseline`
>
> 本文件是实施边界，不代表 Shape 或 Paint 已经在 Apple Silicon 上端到端验证成功。

## 固定参考

- 官方代码仓库：`Tencent-Hunyuan/Hunyuan3D-2.1`
- 官方代码 commit：`82920d643c0dc2f7bfd7255f45f62d386edfe60c`
- 模型仓库：`tencent/Hunyuan3D-2.1`
- 模型 revision：`0b94677654c57bb9a6b6845cd7b704ccf551d327`
- 当前本地硬件：Apple Silicon macOS，48 GB unified memory

所有质量和性能对比都必须在上述代码/模型 revision 下完成。模型权重不能以“latest”代替固定 revision。

## 后端边界

```text
Shape image preprocessing / conditioner  PyTorch
Shape DiT denoiser                     MLX（第一目标）
Shape scheduler                        PyTorch
Shape VAE geo_decoder                  MLX（第二目标）
Shape volume decoder                   PyTorch / CPU
Shape MC surface extraction             CPU
Mesh cleanup / GLB                      CPU

Paint reference write                   PyTorch
Paint inner UNet                       MLX（后续目标）
Paint scheduler / outer pipeline        PyTorch
Rasterizer                              CPU reference / Metal
Bake / mesh painter / export            CPU / C++ / Metal
Training                                PyTorch
```

禁止第一阶段做全局 `torch -> mlx` 替换，也禁止把 Rasterizer 当作普通神经网络模块迁移到 MLX。

## 阶段 0：建立 PyTorch 基线

退出标准：

- 官方 Shape 权重可以加载；
- 固定 demo 输入能够完成 Shape 推理；
- 输出 GLB 非空、无 NaN/Inf；
- 顶点数和面数可记录；
- Blender 可以导入；
- 记录模型加载、sampling、VAE/surface extraction、导出耗时和峰值统一内存；
- 固定随机种子并保存最小输入和完整错误日志。

推荐首版配置：

```text
batch_size = 1
float32
surface extractor = mc
FlashVDM = off
MPS fallback = off（定位问题时可单独启用并记录）
```

## 阶段 1：Shape DiT MLX parity

按照 TDD 顺序实施：

1. 为权重转换写失败测试：Linear、LayerNorm、Q/K/V、MoE gate、Conv/布局（如适用）。
2. 为单个 Attention block 写 PyTorch vs MLX 输出比较。
3. 为单个 DiT block 写输出比较。
4. 为含 MoE 的末端 block 写 top-k expert 与输出比较。
5. 实现 MLX dispatch，先固定 float32。
6. 验证完整单步 denoiser。
7. 接入完整 flow-matching sampling。
8. 和 PyTorch reference 比较 latent、轮廓、几何完整性。

误差报告至少包含：

```text
max_abs_error
mean_abs_error
relative_error
finite check
seed / dtype / shape
```

MLX 端出现 `mx.eval()` 时，必须记录原因。不得在循环中无理由频繁同步。

## 阶段 2：Shape VAE geo_decoder MLX

只迁移推理所需的 `CrossAttentionDecoder/geo_decoder`：

- Fourier embedding；
- query projection；
- cross attention；
- MLP；
- output projection；
- latent K/V cache。

暂不迁移：VAE encoder、完整 volume decoder、FlashVDM、DMC、Marching Cubes。

退出标准：

- 单 query chunk parity；
- 多 chunk parity；
- cache reset 后不同输入不串数据；
- 完整 mesh 与 PyTorch baseline 轮廓和拓扑可解释一致。

## 阶段 3：Paint inner UNet MLX

只做最小路径：

```text
2–4 views
256×256
Diffuse-only
```

执行边界：

```text
reference write：PyTorch
第 1 个 step：PyTorch
后续 denoising step：MLX
scheduler：PyTorch
super-resolution：PyTorch MPS
```

必须逐步验证：

1. NCHW/NHWC activation；
2. Conv2d OIHW/OHWI 权重转换；
3. GroupNorm；
4. GEGLU；
5. Material attention；
6. Reference attention；
7. Multiview attention + 3D RoPE；
8. Text/DINO cross-attention；
9. 一个完整 Transformer block；
10. inner UNet 单步；
11. 15-step diffusion；
12. albedo/roughness/metallic 的误差和最终渲染图。

## Rasterizer 计划

Rasterizer 不使用 MLX 重写。顺序为：

1. CPU Python reference；
2. 官方 C++ CPU rasterizer；
3. NVIDIA CUDA rasterizer（如有条件）；
4. Metal compute rasterizer。

比较：

- coverage；
- face index；
- barycentric coordinates；
- depth/occlusion；
- final rendered image；
- UV seam 和 bake 结果。

特别记录官方 64-bit packed z-buffer 与 Metal 多 pass 32-bit z-buffer 在相等/接近深度下的差异。

## 性能记录格式

每次实验保存：

```text
code_commit
model_revision
host / macOS / Python / PyTorch / MLX
input_sha256
seed / dtype / steps / resolution / views
model download size
model load seconds
Shape sampling seconds
VAE / surface extraction seconds
Paint stage seconds
PyTorch↔MLX bridge seconds
mx.eval wait seconds
peak unified memory
GLB size / vertices / faces / components
non-manifold edges / degenerate faces
```

性能报告不得只报告某个 attention kernel 的耗时。

## 三类结论标签

每个阶段只能使用以下标签之一：

- `PASS`：满足退出标准，有可复现实验和输出证据；
- `PARTIAL`：部分模块通过，但整条链路未通过；
- `BLOCKED`：存在明确环境/依赖阻塞，保留完整错误；
- `NOT_RUN`：尚未运行，不得写成成功。

## 当前明确不做

- 不迁移训练到 MLX；
- 不做完整 PBR 首轮迁移；
- 不把 Metal Rasterizer 伪称为 MLX Rasterizer；
- 不通过 `PYTORCH_ENABLE_MPS_FALLBACK=1` 掩盖算子失败；
- 不把 README 声明或截图作为端到端完成证据；
- 不把模型加载成功、空 GLB 或损坏 GLB 作为 Shape 成功。

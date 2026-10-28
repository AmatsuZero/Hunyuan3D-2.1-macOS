# macOS MLX 推理迁移 Roadmap

日期：2026-10-27。状态：待实施；本轮仅制定路线与验证门槛。
当前版本：`quality-shape-demo` / `5b0070e`，已知成功运行还依赖本地补丁和外部资产。

## 1. 目标及与 Torch 升级的关系

目标是逐步实现 macOS 上**神经网络推理与主要张量管线以 MLX 执行，正式推理
进程不依赖 PyTorch**。保留适合原生 Metal 的光栅化、CPU 网格/UV/图片处理。
“全 MLX 推理”不表示把 I/O、PyMeshLab、xatlas、OpenCV 也重写成 MLX。

先按 [Torch 升级/MPS 计划](torch-upgrade-mps-plan.md) 建立可复现、设备可审计
的新混合基线，再逐模块迁移。每次只改变一个后端/模块；候选 Torch 升级、
MLX 移植、精度、量化和模型质量修复不能同时改动后一起比较。

若 Torch 升级受阻，可继续使用冻结的旧混合基线做独立模块实验，但不得把
“升级”和“MLX 移植”混成同一组性能结论。
依据：[已完成评估](pytorch-mlx-assessment.md)，`hy3dpaint/mlx_utils.py:232`。

## 2. 性能假设及其验证

统一 MLX 可能减少往返转换、同步、双份模型驻留，并使连续计算保留在一个
数组执行系统内。这是待验证的收益假设，不是已有端到端加速证据。
当前可见的机会：

- Paint 每次 MLX generation 都经 CPU/NumPy 输入并转回 Torch；
  `hy3dpaint/mlx_utils.py:34`、`:349`、`:373`。
- Shape DiT 每步和 decoder 每个查询块同样有转换；
  `hy3dshape/hy3dshape/mlx_dit_utils.py:210`、`mlx_vae_decoder_utils.py:175`。
- Paint wrapper 同时保留原 Torch UNet 和 MLX 模型；释放原模型的收益需实际
  测量，不能根据参数数量直接宣称进程内存减少多少；`mlx_utils.py:239`。
- 原生 Metal rasterizer 自身已有 GPU kernel，其 Torch wrapper 经 NumPy
  读写，替换接口可能有益，但重写已有 Metal shader 不必然更快；
  `custom_rasterizer/custom_rasterizer/metal_rasterizer.py:237`。

Torch/MPS 与 MLX 都面向 Apple GPU；具体 kernel、布局、精度、调度和实际
瓶颈决定速度。不能把局部 SDPA 加速倍数外推为整个上色管线倍数。例如某阶段
仅占总耗时 20%，将它加速两倍，在其他耗时不变时整体只有约 1.11 倍。

不预设全 MLX 必须胜出。正确但更慢的实现可保留为明确标记的实验选项；正式
默认路径只在质量与稳定性通过且端到端性能达标后切换。

## 3. 通用实验与验收合同

所有阶段都交付：代码 commit、依赖锁/模型哈希、命令、输入 fixture、原始
日志、数值报告、设备 trace、分项/端到端耗时、内存与结果状态。

1. **同输入比较**：跨框架 seed 相同不保证随机序列相同。数值实验使用预先
   导出的同一份初始噪声、timesteps、条件和权重；涉及 VAE 采样时也固定噪声。
   逐步对比 hidden states、cache、noise prediction、scheduler latent。
   依据：现有 CPU generator，`hy3dpaint/utils/multiview_utils.py:89`。
2. **先模块再集成**：小算子/小输入 → 真实尺寸离线 replay → 单模块集成 →
   累积端到端。独立模块测量可重用冻结条件，但正式全 MLX 验收不得依赖 Torch
   预生成的条件或缓存。
3. **先固定容差**：沿用 MPS 计划的小算子初始容差；模型级阈值先根据参考
   重复性、dtype 和层的敏感性校准，再冻结后测试。不得为候选结果事后放宽。
   逐层误差合格仍需原始视觉/颜色验收。
4. **设备真实执行**：混合阶段所有剩余 Torch 模型前向遵守 MPS 严格审计；
   MLX 显式记录 GPU device/stream 并用 Metal trace 验证。不能只看数组类型。
   关闭自动 fallback 不限制应用显式 CPU 转换，仍需审计。
5. **公平计时**：强制完成 lazy/异步计算；分别测无 profiling 的冷启动、
   热运行、转换/同步、模型加载、输出写入。Torch 与 MLX 参考串行执行，不在
   48GB 内同时持有两套大模型做在线双算。
6. **性能切换**：候选端到端回退超过 10% 时扩大匹配重复到至少 3 次，确认后
   暂停默认切换并定位。要声称“更快”，改善需超过实测波动；不通过则报告
   无明显收益/回退，不承诺固定倍数。采样内存各指标不得简单相加。
7. **质量独立报告**：升级兼容、后端数值、原始颜色、外观、性能分别有结论。
   红 3 若仍黑，颜色检查维持已知 FAIL，不能用手工修色证明模型已正确。
   依据：[颜色验证待办](quality-texture-validation.md)。

始终单 GPU 串行；禁止 MLX DiT + MPS 形状解码；不覆盖已交付资产，不改旧
venv/原 checkout 的扩展。以下每个阶段都可独立回退到上一通过版本。

## 4. 阶段路线

| 阶段 | 主要交付 | 完成门槛 |
| --- | --- | --- |
| R0 混合基线与性能账本 | Torch/MPS 升级验证、阶段 trace、转换/驻留统计 | 能复现与归因，剩余 Torch 网络前向 MPS 已验收 |
| R1 权重与张量合同 | 离线模型转换、版本化 fixture/replay | 独立 MLX 模块不需 Torch 加载权重，映射完整可验证 |
| R2 小模块迁移试点 | RealESRGAN/RRDB MLX 实现 | 真尺寸输出、边缘处理、性能/内存合格，验证移植方法 |
| R3 完整贴图去噪循环 | reference write、首步/cache、scheduler 的 MLX 实现 | 每步/cache 可对比，15 步中不再往返 Torch |
| R4 贴图配套网络 | VAE、CLIP、DINO 分别迁移 | 每个模块逐项通过，神经网络全部 MLX |
| R5 Shape 独立 MLX 管线 | conditioner、调度、采样、体查询/提取接口 | 自包含 shape 推理不加载 Torch，形体质量通过 |
| R6 渲染/烘焙解耦 | Torch-free Metal/数组接口、MLX bake | 不再导入 libtorch 扩展，几何/UV/可见性语义不变 |
| R7 独立运行时与发布 | 可独立安装的 macOS MLX 依赖与入口 | 无 Torch 环境中完整运行；性能、质量、设备覆盖可核验 |

### R0 — 先知道时间和内存花在哪里

- 执行 `torch-upgrade-mps-plan.md` 的隔离、设备与质量门槛；保存原始 stage
  输出，确保现有 dirty 补丁和 eval 导入来源可复现。
- 记录 reference write、首步、2–15 步、DINO/CLIP、VAE、超分、bake/inpaint、
  bridge 次数/字节/耗时、加载/双模型驻留。不可把历史 444.4s 总时间当成某个
  单项的时间，现有运行没有完整的分阶段证据。
- **交付**：baseline manifest、stage-ledger JSON、设备/trace、未经修色的输出。
- **顺序调整点**：R2 是低耦合试点的默认顺序；若 R0 明确证明首步/bridge
  主导耗时，则在 R1 后直接优先 R3。调整以实测占比和工作量为依据。
- 代码位置：`run_paint.py:58`，`textureGenPipeline.py:156`，`mlx_utils.py:254`。

### R1 — 一次性转换权重，独立模块测试不加载 Torch

- 独立离线 CPU 转换工具读取可信 checkpoint，导出 MLX 可直接读取的权重
  与配置，记录 source SHA、参数映射、维度/layout、dtype、buffer、缺失/多余键。
  转换环境允许 Torch；正式推理环境不因此获得 Torch 依赖。
- 区分 learned weights 与可按公式重建的 buffer；任何未加载项目都需有依据，
  不能忽略现有 `unloaded params` 警告后默认当作正确。
- 为 replay 输出带 schema 的中间张量：数据布局、shape、mask、view ordering、
  diffusion timestep、缓存生命周期与重置规则。固定配置而非发明通用框架。
- **验收**：完整键表/形状检查、每模型最小推理；新子进程禁用 Torch import
  仍能直接加载这些权重。转换时间、磁盘成本与推理时间分开报告。
- 代码位置：`mlx_utils.py:153`，`mlx_unet.py:1`，
  `hy3dshape/hy3dshape/mlx_dit_utils.py:66`、`:166`。

### R2 — RealESRGAN 作为默认小范围试点

- 按实际 RRDBNet 参数、padding、upsample、RGB/BGR、归一化、精度、clamp 和
  输出尺寸移植；先保持与旧算法一致，不同时引入量化或其他超分模型。
- 比较每块与全图输出；覆盖边缘、奇数尺寸和实际 6 视角的 albedo/MR 输入。
  加速很小也如实记录，不能只报告模型 kernel 时间而遗漏 layout/图像转换。
- **验收**：Torch-free 单模块测试通过，混合管线仅替换超分模块的对比通过；
  数值/视觉/设备/内存与耗时均有报告。
- 代码位置：`hy3dpaint/utils/image_super_utils.py:21`。

### R3 — 优先消除贴图去噪循环中的双框架往返

- 移植 outer UNet 需要的 DINO projection、位置 voxel indices、参考条件
  embedding、reference-write 和首步行为；已有 inner UNet 可复用，但不是
  简单让首步调用原 `mlx_forward` 就完成移植。
- 固定 fixture 逐项比较 reference/cache，再逐步比较 noise prediction；
  验证缓存在不同输入之间确实重置，避免单次正确但后续请求串缓存。
- 移植当前启用的 scheduler、CFG/条件组合与 latent 更新，保持 timestep
  spacing、历史状态、初始化及 dtype；每个 diffusion step 的 latent 都留在
  MLX。可以暂时一次性接收 Torch 生成的外部特征，但明确该阶段仍是混合。
- 通过后仅保留预期模块的权重，移除运行时原 UNet 的不必要驻留；先用内存
  证据确认生命周期，再衡量释放收益。
- **验收**：reference、首步、后续各步及 scheduler 分别过数值门槛；正式
  15 步执行 trace 不再出现每步 Torch 转换。输出 albedo/MR 原始质量通过。
- 代码位置：`hy3dpaint/mlx_utils.py:239`、`:254`、`:349`；
  `hunyuanpaintpbr/unet/modules.py:985`、`:1000`、`:1011`；
  `hunyuanpaintpbr/pipeline.py:695`。

### R4 — 分别移植 VAE、CLIP、DINO

- 三个模块分别实现、验证和提交，默认先贴图 VAE，再 CLIP，再 DINO；R0 的
  占比和 R3 接口依赖允许调整顺序，每次只替换一个模块。
- VAE：encode/decode、scaling factor、posterior sampling 与 layout 均需匹配；
  不能把已有 shape decoder 当作贴图 AutoencoderKL 的替代。
- CLIP：tokenizer 行为/EOS、hidden-state 选层、projection 与 dtype 固定。
- DINO：processor resize/crop/normalize、位置插值、patch token 与输出重排
  都属于合同，不仅对齐网络层。先分开对比预处理 tensor 和 backbone 输出。
- **验收**：各模块在无 Torch 子进程的真实 fixture 通过；累积集成不用 Torch
  预生成条件，也不通过偷偷 import Diffusers/Transformers Torch 模型运行。
- 代码位置：`utils/multiview_utils.py:42`、`:53`；
  `hunyuanpaintpbr/pipeline.py:150`、`:285`、`:721`；`unet/modules.py:59`。

### R5 — Shape 管线也移除外层 Torch

- 复用已验证的 MLX DiT/MoE 和 shape VAE decoder，补齐图像 conditioner、
  scheduler、noise/CFG、latent/volume queries 与提取接口；从外部 eval 补丁
  中整理需要的实现并单独验证，不改原沙盒。
- DiT step 与体查询块之间都保留 MLX 数据，避免仍通过 Torch wrapper 往返。
  检查 KV cache 生命周期、octree/chunking、边界值和浮点精度。
- **验收**：先已有 fixture/小分辨率，再单次独立质量输出；保存新的文件名，
  不覆盖已批准白模。几何、可见细节与占用合格，无 Torch import。
- 此阶段才需要新的 shape 生成验证；不能因前面的 paint 迁移重复跑形体。
- 代码位置：`hy3dshape/hy3dshape/pipelines.py:174`、`:447`、`:507`；
  `mlx_dit_utils.py:210`，`mlx_vae_decoder_utils.py:175`。

### R6 — 保留原生 Metal，移除渲染/烘焙的 Torch 接口

- 为现有 rasterizer 设计最小 MLX/NumPy 接口，保留已有 Metal shader；先用
  明确的同步/拷贝路径验证正确性，只有 profile 支持时再做共享 buffer/DLPack
  优化。没有 Torch 的目标运行时不需要 Torch↔MLX DLPack 来证明完整性。
- 将投影、mask、indexing、插值/融合等 Torch 张量计算逐段移植为 MLX；大索引
  场景分块验证，不把原 CPU workaround 原样隐藏到“全 MLX”名下。
- 原有网格/UV、OpenCV/mesh inpaint 可继续 CPU/native，报告耗时；检查 native
  扩展是否传递依赖 libtorch/ATen，按实际接口拆分。
- **验收**：已知三角形的像素/深度/面索引、遮挡、有效 UV、纹理与 seams
  通过；无需导入 `custom_rasterizer_kernel` 的 Torch 扩展即可走正式路径。
- 代码位置：`custom_rasterizer/custom_rasterizer/render.py:15`；
  `metal_rasterizer.py:237`；`DifferentiableRenderer/MeshRender.py:1272`、`:1401`。

### R7 — 无 Torch 推理环境与默认后端决策

- 单独声明 macOS MLX runtime 依赖、入口与部署步骤；离线转换工具与测试
  oracle 的 Torch 依赖独立。CUDA/历史混合路线保留自己的入口与依赖。
- 新环境不安装 Torch/Torchvision/依赖 libtorch 的扩展；记录完整依赖树、
  `sys.modules`/import 审计和 native link 审计。模型原始 checkpoint 可以提前
  转换，但不能在正式推理进程旁偷偷启动 Torch 子进程补条件。
- **最终验收**：从原图到 shape，再对固定/新 shape 上色都能独立完成；GPU
  trace、数值、未修色外观/颜色、内存和公平性能报告完整。记录支持的模型、
  分辨率、视角、步数；未覆盖的配置不声称支持。
- 默认切换依据相同输入/配置下的全流程结果。全 MLX 更慢时保留实验后端，
  继续使用更优的已验证路径，明确记录性能假设未成立。
- 发布前为每个阶段保留可回退 commit、依赖锁、原始资产及证据索引。
- 依据：`run_paint.py:28`；`hy3dshape/hy3dshape/pipelines.py:141`；上述各接口。

## 5. 每步报告格式与停止条件

| 项目 | 结果要求 |
| --- | --- |
| 改动范围 | 本阶段替换了哪个模块，哪些仍为混合；对应 commit |
| 正确性 | fixture、容差、层/步误差、cache reset、finite 结果 |
| 设备覆盖 | MLX GPU、剩余 Torch/MPS、native Metal、CPU 例外 |
| 性能 | 同条件 cold/warm、模块及全流程 wall time、变化百分比与波动 |
| 内存 | 采样范围、各 allocator/RSS、压力/swap，不混加重叠计数 |
| 质量 | 未修色的结构、原图颜色、文字、接缝、正面/斜侧渲染 |
| 决策 | 通过并推进 / 继续定位 / 回退；已知 FAIL 与未验收项独立列出 |

数值/设备不明、OOM、意外后端回退、原始质量新回归均先停止本阶段推进。
排查到小用例后再修复，不同时引入第二个模型后端变化。不能用“已全 MLX”
替代质量或性能结论。时间估计需 R0/R1 完成后据实际瓶颈与接口复杂度给出。

## 6. 证据与计划状态

R0 混合基线、P2/P3 设备与 bake 门槛以及 P5 栈级性能账本已完成；R1 的 inner
UNet 权重 bundle 已通过 Torch-free strict load。R2 RealESRGAN/RRDB 独立 MLX
试点已通过真实尺寸数值对照，但稳态约慢 2.3–2.5 倍，未切换默认超分路径；证据见
`HANDOFF-r2-realesrgan-20261028.md`。R1/R2 的 `.safetensors` bundle、manifest
和原始输出保留在外部 evidence，不进入 Git。当前默认 hybrid dispatch 仍从
PyTorch checkpoint 加载。R3 已有 opt-in inner UNet bundle dispatch，两步 fresh
checkpoint 对照逐元素一致（见 `HANDOFF-r3-bundle-dispatch-20261028.md`）；
UniPC scheduler 已对真实 15 步张量做独立 MLX replay（见
`HANDOFF-r3-unipc-scheduler-20261028.md`）；外层 DINO projector 已通过真实权重
Torch/MPS 与 MLX/GPU 数值对照（见 `HANDOFF-r3-dino-projector-20261028.md`）；
7 组 outer condition 权重已独立打包并通过 Torch-free load（见
`HANDOFF-r3-condition-bundle-20261028.md`）。真实 reference/cache 合同已捕获（见
`HANDOFF-r3-cache-contract-20261028.md`），但尚未由 MLX outer loop 消费。
opt-in 首步 inner UNet 已用 MLX 运行完整 15 步并做固定条件对照（见
`HANDOFF-r3-first-step-20261028.md`）。CFG + UniPC 已对真实 15 步张量做组合 replay
（见 `HANDOFF-r3-cfg-scheduler-20261028.md`）。opt-in resident loop 已让第 2–15 步
MLX latent、CFG、UniPC 和 inner UNet 直接连接，并完成真实 15 步固定条件对照
（见 `HANDOFF-r3-resident-loop-20261028.md`）。首步 inner UNet 的 MLX output
现已直接进入 resident loop，真实 15 步固定条件输出与桥接版逐元素相同（见
`HANDOFF-r3-direct-first-output-20261028.md`）。outer reference/条件生产、
首步 sample 输入及最终 VAE 仍依赖 Torch，因此“全 MLX 推理”仍未完成。上面的性能
收益均为待证假设，不能据现有代码注释的 SDPA 倍数推算最终速度。
outer DINO projector 已通过 opt-in bundle-backed MLX adapter 接入并完成单步和
15 步原图对照（见 `HANDOFF-r3-dino-projector-consumer-20261028.md`）；outer Torch
接口仍需两次 NumPy 同步，不能把该切片写成 Torch-free outer 条件生产。
position voxel indices 的 opt-in MLX producer/consumer 已通过四分辨率数值、首步
输出及 12 张原图逐元素对照（见 `HANDOFF-r3-position-indices-20261028.md`）；
dynamic reference-write 和外层条件 embedding 仍使用 Torch。
动态 reference cache 已有 opt-in MLX 转换与 reset 边界，16 层 cache 和 15 步
固定条件原图保持逐元素/逐字节一致（见
`HANDOFF-r3-reference-cache-boundary-20261028.md`）。该边界阶段仍由 Torch
`unet_dual` 生成 reference states；后续 MLX forward 见下段。
opt-in MLX `unet_dual` reference forward 现已完成权重映射、逐层 replay、15 步
集成和同实例 reset；误差与边界见
`HANDOFF-r3-mlx-dual-reference-forward-20261028.md`。该路径仍是 fp16 近似、
未切默认，且 outer Torch 条件合同尚未移除。
当前固定条件的首版工程门槛为 cache 最大 NRMSE `0.01`、reference write NRMSE
`0.03`，由 `tools/quality_demo/validate_mlx_reference_report.py` 执行；它不替代
遮挡、多分辨率和视觉验收。
覆盖 replay 已验证 256px reference 通过；512px 中央遮挡和 384px reference 的
后段 cache 超过 `0.01`，因此暂不扩大 opt-in 范围或切换默认路径。下一步先做 fp32
和逐层 stage 诊断。
两组失败输入随后在 MLX reference fp32 参数 replay 中均通过原门槛；下一步验证
独立 reference fp32 opt-in 的 cache dtype、15 步图像、reset 与内存性能合同，见
`HANDOFF-r3-reference-fp32-diagnostic-20261028.md`。
独立 reference fp32 已完成两组超限输入的 fp16 consumer replay 与 reset 对照，
实现见 `HANDOFF-r3-reference-fp32-consumer-20261028.md`；完整 15 步和性能仍未验收。
真实 resident 后续验证已发现固定 posterior 的遮挡 512px 输入在 reference fp32 下仍
超过原 cache 门槛（`0.014330 > 0.01`），完整 15 步 latent/图像筛查也失败。旧 coverage
使用注入非 reference 条件和 ambient reference sampling，本轮无注入且固定 posterior；
先前 replay PASS 不覆盖新输入。暂停扩大 outer condition/VAE 边界，先冻结新旧输入并
逐层定位，见 `HANDOFF-r3-reference-fp32-resident-20261028.md`。默认 Torch reference 不变。
框架转换与同步边界参考
[MLX 官方文档](https://ml-explore.github.io/mlx/build/html/usage/numpy.html)。
当前已验收代码、Torch 升级风险和颜色缺陷分别见同目录的
`pytorch-mlx-assessment.md`、`torch-upgrade-mps-plan.md`、
`quality-texture-validation.md`。

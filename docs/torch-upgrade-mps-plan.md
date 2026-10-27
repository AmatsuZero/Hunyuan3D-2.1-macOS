# PyTorch 升级与 MPS 执行验证计划

日期：2026-10-27。状态：计划，尚未安装候选版本或执行新推理。
实施进度与已发现的失败门槛见 [验证报告](torch-upgrade-validation-20261027.md)；下文保留原始计划。
代码基线：`quality-shape-demo` / `5b0070e`，另有待保护的本地修改。

## 1. 目标与边界

将 PyTorch 从本机已验证的 2.5.1 升级到经验证的新版本，并证明仍使用
Torch 的神经网络前向计算实际执行于 MPS。保留已有 MLX 加速子图。
候选先使用 `torch==2.14.1`、`torchvision==0.29.1`、Python 3.11；安装前
重新核对官方包元数据与 macOS arm64 wheel。旧环境保留为回退基线。
依据：[现有评估](pytorch-mlx-assessment.md)。

MPS 保证针对明确列出的模型/张量计算阶段，不把所有 Python、I/O、PIL、
PyMeshLab、xatlas、OpenCV 操作都称作 MPS。CPU 例外必须说明原因、范围、
调用次数和耗时；不能通过打开全局 fallback 掩盖模型前向不支持的算子。

保持 48GB 单 GPU 任务串行；不运行 MLX DiT + MPS 形状解码组合。
使用现有白模进行贴图升级验收；不覆盖任一已交付 GLB。完整移除 Torch、
DLPack/零拷贝改造、提高视角数/分辨率均不纳入本轮。

## 2. 已发现的问题与设备合同

| 阶段 | 本轮目标 | 现有代码依据 |
| --- | --- | --- |
| Paint reference-write、首个 generation pass | Torch 参数/计算输入输出为 MPS，实际有 Metal GPU 执行；不自动退 CPU | `hy3dpaint/mlx_utils.py:254` |
| Paint inner UNet 后续 generation passes | MLX GPU；记录 dispatch 次数和边界转换 | `hy3dpaint/mlx_utils.py:349` |
| CLIP/text encoder | 启用的前向为 MPS；tokenization 可为 CPU | `hy3dpaint/utils/multiview_utils.py:42`，`hunyuanpaintpbr/pipeline.py:285` |
| DINOv2 | Transformer 前向为 MPS；processor 与 bicubic 单独审计 | `hy3dpaint/hunyuanpaintpbr/unet/modules.py:59`，`utils/multiview_utils.py:53` |
| Paint VAE encode/decode | 两条实际调用路径都在 MPS；不能只检查 `forward` | `hy3dpaint/hunyuanpaintpbr/pipeline.py:150`、`:721` |
| RealESRGAN / RRDBNet | 网络前向 MPS；图片转换、输出转 NumPy 可为 CPU | `hy3dpaint/utils/image_super_utils.py:21` |
| Rasterizer | 原生 Metal GPU，Torch 输入/输出按 MPS 合同检查；CPU kernel 不冒充 MPS | `hy3dpaint/custom_rasterizer/custom_rasterizer/render.py:15` |
| Bake tensor operations | 验证可支持的 Torch 张量计算在 MPS；迁移旧 CPU 大索引 workaround | `hy3dpaint/DifferentiableRenderer/MeshRender.py:1272` |
| Mesh / UV / image inpaint / 文件读写 | 明确标注 CPU/native；不属于神经网络前向保证 | `hy3dpaint/utils/simplify_mesh_utils.py:23`，`DifferentiableRenderer/MeshRender.py:1401` |
| Shape DiT / VAE decoder | 保留 MLX GPU 路线；Torch wrapper 的加载、调度和转换单列 | `hy3dshape/hy3dshape/pipelines.py:247`；已验证形状使用外部 eval 补丁 |

当前 `get_device()` 会在 MPS 不可用时返回 CPU，而 MLX 初始化失败也会回退。
升级验收需要严格入口，不能沿用这种静默降级作为通过依据。
依据：`hy3dpaint/utils/device_utils.py:11`、`textureGenPipeline.py:91`。

当前 bridge 经 `.cpu().numpy()` 转换；关闭自动 fallback 不会禁止这些显式
转换。它们应计时并与模型前向区间分开，不要求本轮重写。
依据：`hy3dpaint/mlx_utils.py:34`、`:373`。

## 3. 分阶段实施与交付

### P0 — 冻结可复现基线

- 保存原 venv 的包版本、Python/macOS/Xcode、相关源码 SHA/diff、模型与输入
  文件 SHA；记录模型实际导入路径。相关本地 dirty 修改以独立补丁/快照保存，
  不将无关资源改动或缺失文件混入升级分支。
- 盘点 `run_paint.py` 的绝对路径与外部 shape PYTHONPATH；隔离运行必须显式
  使用候选 checkout，拒绝混入原 checkout/旧 venv 模块。形状扩展验收需要独立
  副本的 eval 补丁，不能修改原 eval 路径。
- 固定已有原始白模、参考图和未修色贴图作为基线输入；手工红 3 修正版仅作
  对照，不能代表原始管线正确性。
- **交付/门槛**：baseline manifest、任务相关 overlay、输入/产物哈希清单；
  旧环境可定位恢复，所有 dirty 状态被保留。
- 依据：`run_paint.py:18`、`:58`；`docs/pytorch-mlx-assessment.md`。

### P1 — 独立 checkout、venv 与依赖候选

- 新建升级分支的独立 checkout 和 Python 3.11 venv，安装候选 Torch/Torchvision
  配套版本。先保留已用 Diffusers/Transformers/MLX 版本；只有发现具体不兼容时
  才局部升级相关依赖并记录原因。
- 整理 macOS 约束/锁定文件；通过解析检查，不照搬包含 CUDA/Blender 项的根
  requirements，也不靠长期 `--no-deps` 掩盖依赖冲突。
- 在隔离源码中构建 rasterizer，检查链接库、rpath、import 路径都属于新环境；
  测试输入输出尺寸、数值与 Metal 后端。不能在原源码路径重编覆盖旧 `.so`。
- 审计 checkpoint loader 的 `weights_only` 行为和 BasicSR 的 Torchvision shim。
- **交付/门槛**：候选 lock、安装/build 日志、包依赖一致性检查、native smoke
  全通过；候选进程 `torch.__file__` 和 native module 路径不指向旧环境。
- 依据：`requirements.txt:8`；`custom_rasterizer/setup.py:18`；
  `hy3dpaint/mlx_utils.py:153`；`hy3dpaint/utils/torchvision_fix.py:8`。

### P2 — MPS 强制执行与可观测性

拟新增 `hy3dpaint/utils/execution_audit.py`、`tools/quality_demo/audit_mps.py`
和 `tests/test_execution_audit.py`；在 `run_paint.py` 暴露显式审计/严格运行选项。

- 只在严格验收入口要求 MPS 与预期 MLX 可用；不可用即失败。保留普通 CPU/CUDA
  入口的原行为，避免破坏跨平台支持。混合基准若 MLX 初始化失败也必须失败。
- 建立阶段 registry，为模型前向、VAE encode/decode、reference write、首步、
  RealESRGAN 等记录：模型/参数/buffer 的 device/dtype，实际计算 tensor 的
  device/dtype/shape、阶段名、调用次数、起止与失败原因。检查实际参与计算的
  子模型，区分必要的 CPU scalar/控制信息，不能以整个 pipeline 的单一
  `.device` 属性代替检查；挂在顶层的 hook 不能覆盖所有方法调用。
- 分别记录 Torch/MPS、MLX/Metal、原生 Metal、显式 CPU 区间与跨框架转换；
  CPU preprocessing 必须在进入 GPU 模型前结束，并检查返回 MPS 的位置。
- 全新子进程在 import Torch 前移除 `PYTORCH_ENABLE_MPS_FALLBACK`，逐阶段执行
  及跑集成用例；默认自动 fallback 关闭。若发现不支持的操作，报告具体算子和
  阶段，不把模型整个移到 CPU，不用开启全局 fallback 使验收变绿。
- 为 Torch-only 阶段采集 `torch.mps.profiler` signpost/Metal trace；校验阶段
  中存在相关 GPU command/kernel，而非只有设备标签或其他 MLX 任务的 GPU 活动。
  CPU profiler 用于关联 host 调度；不假设存在 `ProfilerActivity.MPS`。
- 候选 API 核验后，归因采样可使用
  `torch.mps.profiler.profile(mode="interval,event", wait_until_completed=True)`，
  关联 Instruments 的 OS Signpost 与 Metal GPU 时间线；若当前系统工具不可用，
  该项记为未验收，不以设备日志替代。等待每次 GPU 完成会影响性能，只用于审计。
- fallback 关闭只约束通用自动回退；部分算子可能独立注册 CPU fallback，显式
  `.cpu()` 也不受它控制。警告可能只打印一次，不能将日志行数当调用次数。
  如 trace 发现模型前向存在 CPU 算子，继续定位或把 MPS 覆盖标记为未通过。
- 负向回归覆盖：MPS 不可用、模型/输入被错误放在 CPU、混合 dispatch 意外
  缺失、审计阶段未被调用。都必须失败，不能生成空审计报告后返回成功。
- **交付/门槛**：device audit JSON、逐阶段 trace、fallback 原始日志；神经网络
  前向未出现未登记 CPU 执行，必需阶段调用次数大于零且与路径一致。
- 依据：`device_utils.py:11`；`multiview_utils.py:51`；
  `hunyuanpaintpbr/pipeline.py:166`、`:722`；`mlx_utils.py:254`。

### P3 — 清理/验证 CPU 绕行与数值兼容

- 对 DINO/条件处理的 bicubic 路径先用新版本做关闭 fallback 的单项探针。
  若 MPS 支持且质量一致，则保留 MPS；否则将不可支持的图片预处理明确置于
  CPU preprocessing 边界并计时，DINO Transformer 仍必须执行 MPS。
- 对 `back_sample` 的大索引 CPU workaround，先构建小输入正确性用例，再用
  有界的实际尺寸探针评估分块 MPS indexing/gather。先锁定有效像素集合、深度
  判定与插值输出，再迁移；若新版仍无法安全支持，标明 bake 计算尚有 CPU
  例外，不能宣称该阶段已全 MPS。
- 验证 SDPA 的连续/非连续张量、mask、分块与非分块、fp16/fp32；检查 finite
  输出。shape wrapper 的 `.eval()`/MoE/index_add_ 用小输入单项回归检查，避免
  用不安全的整套形状配置做探针。
- CPU 参考只在串行的小用例中执行，明确标记为测试 oracle，不能混入正式推理
  的设备覆盖统计。参考与候选保留相同数学输入、dtype 和权重。
- 开始候选比较前预登记容差：小算子初始目标 fp32 `rtol=1e-4, atol=1e-5`，
  fp16 `rtol=1e-2, atol=2e-3`。模型级使用归一化误差、相关性及视觉指标，
  阈值由旧版可重复性校准后固定；不能在看到候选失败后随意放宽。
- **交付/门槛**：operator parity 报告、显式 CPU 例外清单（含未迁移项）、
  fallback 禁用的模型阶段测试；不把 API 存在或输出 tensor 在 MPS 当数值通过。
- 依据：`unet/modules.py:59`；`MeshRender.py:1272`；
  `hunyuanpaintpbr/unet/attn_processor.py:63`；`docs/quality-texture-validation.md`。

### P4 — 固定白模端到端与颜色验证

- 串行运行旧版基线与新版候选；使用相同输入、seed、6 视角、512 分辨率、
  15 步、remesh 与输出设置。独立进程输出到新目录；不覆盖白模及历史文件。
- 保存 multiview albedo、烘焙前后/补洞前后纹理、相机与材质，定位红色最早
  消失的阶段。先实现语义区域对齐/可见性，再比较红 3、黑 H/Y/D、浅色牌底。
  不对不同 UV 布局机械逐像素比较，不用整图 PSNR 代替局部颜色检查。
- 结构验收：40,000 面目标、有效有限几何/UV、内嵌图片可解码、PBR 引用有效；
  视觉验收：正面/斜侧视图，文字、颜色、接缝和材质表现。
- **交付/门槛**：候选正常退出、原文件哈希未改变、设备合同通过、无新增质量
  回归。结果分别标记“升级兼容”“设备覆盖”“颜色还原”；旧版已有红 3 缺陷
  不冒充升级引入的故障，但未修复时颜色检查必须保持 FAIL/已知问题，不能
  用局部重染后产物将原始管线质量检查标绿。
- 依据：`run_paint.py:58`；`textureGenPipeline.py:156`；
  `utils/multiview_utils.py:75`；`docs/quality-texture-validation.md`。

### P5 — 性能、48GB 内存与切换

- 分离 profiling 与性能运行。Torch GPU 阶段计时前后调用 MPS synchronize；
  MLX 阶段完成 lazy evaluation/synchronization 后结束计时，外层 wall time
  覆盖桥接和所有输出写入，避免异步提交时间伪装加速。
- 一次冷启动与匹配条件的热运行；差异超过 10% 时才扩大到至少 3 次匹配重复
  比较中位数。记录加载、reference/首步、MLX 后续、VAE、超分、bake 与 CPU
  例外分项耗时。候选稳定回退超过 10% 时分析原因并阻止自动切换。
- 采集 MPS allocated/driver memory、MLX 可用内存计数、进程 RSS、系统压力和
  swap 增量。不同计数有重叠，不能直接相加作为统一内存峰值。记录采样间隔，
  不将采样最大值声称为精确峰值；禁用 watermark 保护不能作为成功条件。
- **切换门槛**：build/import、关闭自动 fallback 的模型前向、设备/trace、
  数值/原始质量、正常退出、无 OOM 和稳定性能全部满足。CPU 例外逐项报告；
  如 bake 仍使用 CPU，只能声明“模型前向 MPS”，不能声明全 Torch 算子 MPS。
- 通过后提交候选依赖约束、针对性代码、测试、README 与证据索引，推送独立
  分支。旧 venv/checkouts/产物保留；回退使用旧解释器与旧源码组合，不原地
  降级或复用被新版覆盖的扩展。
- 依据：`mlx_utils.py:370`；`attn_processor.py:43`；
  `custom_rasterizer/setup.py:21`；`docs/pytorch-mlx-assessment.md`。

## 4. 总体验收与停止条件

- 每个阶段有代码版本、环境/输入、命令、退出码、日志、设备、耗时及结果。
- 设备错误、自动 fallback、MLX 路由缺失、native 路径污染、OOM 或未定位的
  数值错误均停止进入下一阶段；保留证据，修复当前最小失败用例后重验。
- 小步兼容修复通过后才跑完整模型；只在发生明确版本相关失败时做版本二分。
- 不以一次成功运行、`mps.is_available()`、参数 `.device` 或 CPU 占用低单独
  证明 MPS 执行。模块放置 + 禁止静默降级 + 实际 trace 共同支持结论。
- 当前文档未改变任何模型运行代码或当前 venv；以上各项均为待实施。

## 5. 官方参考

- [MPS environment variables, 2.14](https://docs.pytorch.org/docs/2.14/mps_environment_variables.html)
- [MPS profiler.profile, 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.mps.profiler.profile.html)
- [MPS synchronize, 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.mps.synchronize.html)
- [MPS fallback implementation, v2.14.1](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/mps/MPSFallback.mm)
- [Apple Metal developer tools](https://developer.apple.com/metal/tools/)
- [MLX synchronization / framework conversion](https://ml-explore.github.io/mlx/build/html/usage/numpy.html)
- 候选版本来源与升级兼容风险见 [评估文档](pytorch-mlx-assessment.md)。

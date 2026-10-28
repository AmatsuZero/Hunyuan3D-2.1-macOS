# Hunyuan3D macOS：Torch/MPS 升级与颜色诊断交接

更新日期：2026-10-29。本文是该分支继续工作的唯一权威交接；历史失败与当前修复状态在下文明确区分。

## 1. 接手结论

- 当前阶段在隔离验证 Torch 2.5.1 → 2.14.1 的基础上逐步迁移 MLX。P2 命名 Torch-only Metal trace 已完成（见 §11）；P3 真实 bake CPU 例外已完成数值对照并移除旧 workaround（见 §12）；P5 已完成三轮串行旧/新版 benchmark 样本（见 §13）。R3 的 DINO projector、position indices、reference cache/forward、first-step 和 resident loop 已有 opt-in 实现与对照（见 §14–15），但默认 Torch reference、默认 Winograd、默认 MLX dispatch 和默认环境未切换；occluded512 图像最大差仍超探索性门槛。
- 新版已完成关闭自动 fallback 的完整上色；所审计神经网络的参数、计算输入/输出均在 MPS，MLX 部分在 Metal。P2 命名 Torch-only 窗口已有目标 PID 的 Metal trace；不能据此宣称全流水线都在 GPU：显式 CPU 路径仍存在，完整 pipeline 的 trace 验收仍未完成。
- 颜色诊断已闭环到固定条件及本地完整 GLB 复验：旧版（Torch 2.5.1）全黑 HY3D 来自 MPS bicubic/DINO 错误；候选版早期 hybrid 的 H/Y/D 染红来自 MLX reference attention 材质 V head layout；后续重复漂移来自 MLX loader 漏载 `up_block_0` upsampler 权重。根因均已修复，修复后固定条件 15 步 hybrid 及完整 GLB 的 Blender 四视图恢复黑 H/Y/D、红 3。用户设备/独立 3D 查看仍未验收。
- 相同 seed 在两个 Torch 版本下产生不同实际噪声，旧的同 seed 对比不充分。把旧版实际初始噪声注入修复前候选 hybrid 后 H/Y/D 仍红——**初始噪声不是染红根因（负结果保留）**；后续纯 Torch/hybrid 固定条件对照才把分叉定位到 MLX reference attention。
- 当前没有本任务遗留的运行中推理/trace 进程。ComfyUI 是用户的既有进程，下一次 GPU 实验前重新检查队列。
- **重要结论反转**：Torch 升级本身不是候选版颜色失败的原因，反而**修复**了一个真实的 2.5.1 MPS bug（见 §6.2）。`demo.png` 的设计意图已确认是黑 H/Y/D、红 3；候选版染红来自 MLX inner UNet 的 reference attention 材质 V head layout 差异，已在固定条件下修复并复验（见 §6.4）。

## 2. 工作目录、分支、提交

| 用途 | 路径 / 状态 |
| --- | --- |
| 实际继续工作的隔离 worktree | `/Users/daubert/Github/hy3d-torch-upgrade` |
| 当前分支 | `codex/torch-mps-upgrade` |
| 原始 checkout（保护） | `/Users/daubert/Github/Hunyuan3D-2.1-macOS`，`quality-shape-demo`，HEAD `5b0070e` |
| 旧环境（保护） | `/Users/daubert/Github/.venvs/hunyuan3d-shape/bin/python` |
| 候选环境 | `/Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python` |
| 外部实验数据 | `/Users/daubert/Github/hy3d-upgrade-evidence/20261027/` |
| 本地进度账本 | `.superpowers/sdd/torch-upgrade-mps-plan/progress.md`（不作为版本化交接唯一来源） |

代码提交：

1. `d8a7abd`：记录已验证的现有运行前提（原生扩展 Torch rpath、SDPA workspace 上限、MLX 本地权重查找）。
2. `3e6bb2f`：隔离升级方案、MLX roadmap、实验依赖锁、MPS 执行审计、算子探针、验证报告。
3. `026c1be`：初始噪声 replay/capture、3 个测试、VAE 与 run 07 诊断结果。
4. `d2f89a0`：交接与固定条件颜色诊断工具。
5. `92bc4e0`：修复 MLX reference attention 材质 V head layout。
6. `0b0e285`、`83e84de`、`e0aba4a`：记录 reference layout 修复后的固定条件与完整 GLB 本地验证，并更新交接。
7. `d2dab11`、`68d320d`：记录修复前的受控性能、重复漂移与 FP32 负结果。
8. `889a796`：补齐 MLX `up_block_0` upsampler 权重加载与回归测试。**不要**触碰根目录 `torchvision_fix.py` 这个继承来的 dirty 文件（受保护）。

推送目标是 `fork/codex/torch-mps-upgrade`，remote `fork` 为 `https://github.com/AmatsuZero/Hunyuan3D-2.1-macOS.git`。`origin` 是上游，不要向它推送。上述提交已推送到 fork；后续以实际 `git ls-remote` 为准。没有创建 PR，也没有合并。

隔离 worktree 是手动 sparse linked worktree，排除了 `hy3dshape/tools/mini_trainset/`。原生 managed-worktree 创建曾因 partial-clone 缺少训练 blobs 失败，不必重建整个工作区。

**有意保留未提交的 `torchvision_fix.py`（仓库根目录）**：这是继承来的无关 dirty 文件；验证入口使用的是 `hy3dpaint/utils/torchvision_fix.py`。不要提交、还原或清理它。原始 checkout 的全部 porcelain 状态与初始快照逐字节相同。

## 3. 不可破坏的约束

- 48 GiB 统一内存；一次只运行一个 GPU 推理任务。不要并行跑旧/新版，不要启动额外 MLX 或 MPS 推理占用。
- 禁止 MLX DiT + MPS shape decoder；不要重跑已认可的形体。
- 不覆盖任何已交付 GLB，尤其 `shape_textured.glb`。新实验使用全新 output directory，审计入口会拒绝已有目录，并复制输入 mesh 以隔离 remesh 副产物。
- 不升级旧 venv，不在原始源码目录重编原生扩展，不提交无关 dirty 文件/模型权重/GLB/raw trace/机器专属外部证据。
- 不用手工局部修色结果充当升级颜色验收；旧版红 3 修复是用户交付物，不是 pipeline 修复。
- 若未来涉及形状：仍使用 `PYTHONPATH=/Users/daubert/Github/hy3d-mlx-eval/hy3dshape` 和既有 `HY3DGEN_MODELS`。详细命令见 `/Users/daubert/Github/hy3d-mlx-eval/HANDOFF-quality-texture.md` §3。本次上色实验无需重新生成形状。
- Shell 命令按 `/Users/daubert/.codex/RTK.md` 使用 `rtk` 或 `rtk proxy`。

## 4. 环境与当前代码分工

- Python 3.11.16；旧 Torch/Torchvision = 2.5.1/0.20.1，候选 = 2.14.1/0.29.1。MLX 0.32.4、Diffusers 0.30.0、Transformers 4.46.0 保持相同。
- 候选依赖 `uv pip check` 上次检查 116 packages compatible。候选 native rasterizer 已单独重编，真实 Metal 三角形 smoke 覆盖 288 pixels；inpaint 扩展也在候选环境完成构建，xatlas 0.0.11。
- Paint：参考网络 `unet_dual`、第一次生成 UNet、DINO、VAE、RealESRGAN 用 Torch/MPS；其余 14 次生成 UNet 调用用 MLX/Metal（总 15 步）。CLIP text 路径未调用，checkpoint 用 learned text embeddings；图像条件由 DINO 提供。
- 实际加载的外层 Torch UNet 与 reference processor 来自 Diffusers 动态模块缓存 `diffusers_modules.local`，源文件由模型目录 `hunyuan3d-paintpbr-v2-1/unet/` 复制。缓存 `attn_processor.py` 与模型文件 SHA256 同为 `a3f7c70e...a91088`，但与工作区 `hy3dpaint/hunyuanpaintpbr/unet/attn_processor.py` 不同；后者的 CPU layout 回归测试只证明工作区实现，不能冒充正式运行路径。诊断与正式审计现在都会把实际类、模块名和模块文件写入报告。
- 形状路线仍按旧方案运行，本轮未变更/重跑。
- P3 已移除 `hy3dpaint/DifferentiableRenderer/MeshRender.py` 中强制 MPS bake 大索引转 CPU 的 workaround。显式 CPU/native 边界仍包括 Torch/MLX `.cpu().numpy()` 桥接、mesh/UV/inpaint 等流程；不能称全 pipeline GPU。

关键代码：

- `hy3dpaint/utils/execution_audit.py`：审计真实 forward/encode/decode，检查参数/buffer/计算输入输出设备；CPU scalar timestep 是明确例外。缺阶段/缺 MPS/CPU 计算张量会失败。每个事件立即写 JSONL，避免原生 crash 丢失全部已完成证据。
- `tools/quality_demo/audit_mps.py`：fresh-output、输入复制、工作区与 Diffusers 动态模块来源记录、阶段截图/数组、耗时/内存、可选 MLX cache budget、初始噪声 replay/capture。严格模式拒绝启用 fallback。默认不改变 MLX cache policy。
- `tools/quality_demo/probe_mps_ops.py`：16 个有界 MPS 算子探针；CPU float32 oracle 使用与 half 输入相同的已舍入数据。
- `tools/quality_demo/probe_fixed_conditions.py`：**固定条件张量诊断主工具**。四阶段 `fixtures`（渲染固定法线/位置图+参考图）、`conditions`（捕获 DINO/VAE ref/normal/position 条件张量，固定 posterior 噪声以隔离采样器）、`capture`（钩住 unet_dual/unet/scheduler 捕获 noise_pred、latents、reference_write，记录实际 UNet/processor 来源，支持 `--inject`、`--stop-after`、`--initial-noise`、真实 fixtures 相机、条件权重扫描、`--mlx` 混合路径和 posterior 敏感性对照）、`compare`（逐 npy 相对误差）。诊断专用，不写已交付资产、不覆盖 GLB。
- `tools/quality_demo/probe_dino_parity.py`：逐层（含 `embeddings`）钩住真实 DINOv2，输出每 transformer 层与 final 到 npy，用于两版本/两设备边界定位。
- `tools/quality_demo/probe_dino_ops.py`：DINO 单层的 layernorm/linear/gelu/sdpa/softmax/add/mul 等算子在 MPS vs CPU 的逐项误差。
- `tools/quality_demo/probe_dino_embeddings.py`：把真实 `Dinov2Embeddings` 拆成 patch_conv / pos_interp_bicubic / embeddings_sum 三组件，各与 CPU 同权重副本对比。
- `tools/quality_demo/probe_bicubic_pos.py`：独立验证 MPS `F.interpolate(mode="bicubic")` 在位置编码插值尺寸下的误差。
- `tests/test_initial_noise_fixture.py`：通过真实 Diffusers `prepare_latents` 测试 scheduler scaling、无 fixture 时 RNG 不变、shape/dtype/finite 拒绝和方法恢复。

## 5. 已完成证据与未过的门槛

详细报告：`docs/torch-upgrade-validation-20261027.md`。

| 项目 | 结果与边界 |
| --- | --- |
| 完整严格 run 06 | exit 0，load/inference/total = 19.738/200.323/220.077 s，fallback=0，cache threshold=1 GiB |
| 完整严格 run 07 | exit 0，21.498/205.282/226.793 s，fallback=0，相同 cache policy，旧版初始 noise fixture |
| run 07 设备事件 | reference Torch 1、first-generation Torch 1、outer UNet 15、DINO 1、VAE encode/decode 3/1、super-resolution 12 全通过；MLX 14 次 |
| run 07 实际噪声 | `initial-latents.npy` 与旧 fixture 逐元素完全一致；UniPC initial sigma=1.0 |
| MPS bounded ops | 新版 16/16；旧版 12/16，4 个 bicubic fp32/fp16 ±antialias 不支持 MPS（fallback 关闭） |
| 测试 | P2 时 30/30 unittest（正常 Metal 权限）；P3 最新验证见 §12 |
| GLB 结构与本地查看 | run 06/07/08/09 均为 24,087 个 UV-split 顶点、40,000 面、1 个 PBR 材质和 2 张内嵌贴图，几何有限。最终修复后的 run 09 Blender CPU 四视图核对了正面颜色及侧背面贴图连续性；用户设备/独立查看待验收。 |
| 颜色（旧版 2.5.1） | FAIL 但根因已定位：2.5.1 MPS 缺 `upsample_bicubic2d` 内核，DINOv2 位置编码 37×37→16×16 bicubic 插值结果错误（cos vs CPU = −0.0014、nrmse 1.32），使 DINO 隐状态语义崩坏（final cos 0.29）。旧版全黑 HY3D 正是此错误的后果。 |
| 颜色（候选版 2.14.1） | 固定条件及完整 GLB 本地查看通过：`demo.png` 是黑 H/Y/D、红 3；修复后的 15 步 hybrid 输出与 Blender 四视图保持该设计。该结论不覆盖默认环境或用户设备/独立 3D 查看。 |
| 修复前因果链 | 旧版 corrupted DINO 注入证明 DINO 差异进入 UNet；修复前固定条件对照中纯 Torch 15 步正确、MLX hybrid 染红：`noise_pred_0` 完全相同，`noise_pred_1` nrmse 0.1088，最终 `latents_14` nrmse 0.4171。 |
| 修复前 MLX parity | `capture-hybrid-compare-step2`：第一个 MLX step 的 Torch/MLX mean abs 0.047266、p99 0.398438、max 4.521484；`HUNYUAN_MLX_FP32=1` 仅降至 mean abs 0.046533，排除简单 fp16 累积。 |
| DINO 跨版本 CPU 对照 | CPU 上 2.5.1 vs 2.14.1 的 DINO final 几乎一致（nrmse 0.0015），排除权重/输入差异；差异纯由 2.5.1 的 MPS 路径引起。 |
| P2 Torch-only Metal trace | DINO、VAE encode/decode、reference UNet、first-step UNet、RealESRGAN 命名窗口均有目标 PID 的 PyTorchMPS signpost 与 Metal GPU interval，fallback 0、stage 全时段落在 trace 内；这不是全流水线“全 GPU”证明。详情 §11。 |
| 受控性能重复 | 修复 upsampler 权重漏载后，同一固定条件 Hybrid 两次为 67.812/68.498 s，逐步张量和 12 张图逐元素一致；相对此前 Torch-only 181.897/211.721 s 的诊断轮分别快 62.7%/67.6%。这仍是固定条件扩散阶段结果，不是完整旧/新版性能验收。 |
| P3 bake | 真实尺寸 CPU/MPS 有效像素与输出逐元素对照通过；旧 MPS CPU workaround 已移除，完整严格 audit 通过。详情 §12。 |
| 内存/性能门槛 | P5 已完成旧/新版各 3 次串行完整栈样本（§13），候选栈中位数较快且内存计数较低；Torch-only 归因和 48 GiB 安全内存验收仍开放，不能据此推广候选默认环境。 |
| 独立 review | native reviewer 多次 429，无独立审查通过；仅作者复查 |
| MLX reference layout 修复 | checkpoint 的 reference processor 先把 albedo/MR value 投影拼接后按 attention head 切分；MLX 原先分别切 head，导致第 2 步 reference 输出 NRMSE 1.329121。保持训练通道排列后降至 0.002778；完整 15 步固定条件 hybrid 在 1 GiB MLX cache limit 下完成，H/Y/D 红色占比均 0，3 红色占比 0.3525（纯 Torch 0.3545）。 |
| MLX upsampler 权重修复 | 原 loader 漏掉 `up_blocks.0.upsamplers.0.conv.weight/bias`，导致每次 MLX 模型新建时 `up_block_0` 卷积随机初始化；补齐 OIHW→OHWI 加载后，重复 2 步和完整 15 步输出均逐元素稳定。 |

run 06 的 MLX steps 7.388–9.482 s，run 07 为 7.586–10.936 s；单次无明显后程断崖不足以证明稳定性能。MLX 默认 free-cache limit 曾是 48,962,627,174 bytes，约 45.6 GiB，与 Torch 争用 48 GiB 池。`--mlx-cache-limit-mib 1024` 是空闲缓存回收阈值（后续 allocation 触发回收），不是瞬时硬内存上限。不能把缓存策略收益归因于 Torch 升级。

run 06 MLX peak 8.985 GiB、sampled MPS driver 25.698 GiB、RSS 11.083 GiB；这些计数重叠，不能相加。系统 swap 24.714→30.710 GiB，内存门槛仍开放。

## 6. 颜色诊断：根因已定位（DINOv2 位置编码 bicubic）

外部目录 `color-diagnosis/`：

### 6.1 已排除的假设（保留为负结果）

- `probe_vae.py`，旧版 `torch251/`、新版 `torch214/`、`vae-comparison.json`、`fixed-noise.npy`。
- 相同 normalized 512 输入 tensor 完全一致，真实 VAE 权重、fp16 MPS、fallback 关闭。固定 posterior noise 后，两版 reconstruction 都保留黑 H/Y/D 和红 3。
- posterior mean NRMSE=0.000632；controlled decode NRMSE=0.000350，max abs=0.0078125。只覆盖该 VAE 小实验，不能排除完整流水线中的 VAE 交互。
- 同 seed 的 MPS noise correlation=0.007770，CPU half noise=-0.001714；因此历史同 seed 对比实际混入不同噪声。
- 用旧 Torch CPU generator seed 0 生成 raw initial noise：`initial-noise-torch251-seed0.npy`，float16，shape `(12,4,64,64)`，SHA256 `925e9a05002409d327ec0bee2114bd9cad1e729971990b15797d8daa1dda10f2`；旁边 JSON 有来源。
- run 07 只 replay 初始 diffusion noise，VAE sampling、DINO/reference 条件仍由候选环境计算。前视角与 run 06 有变化（0–255 通道 max delta=202，mean abs delta=5.050），但 H/Y/D 仍红。这个负结果应保留，不能说“随机性完全无关”，也不能说“新版 Torch 算子已被证明有错”。

run 07 关键输出（均在 `/Users/daubert/Github/hy3d-upgrade-evidence/20261027/` 下）：

- `candidate-audit-07-old-noise/audit.json`、`events.jsonl`、`timings.jsonl`
- `candidate-audit-07-old-noise/initial-latents.npy`
- `candidate-audit-07-old-noise/multiview_0_albedo_0.png`（肉眼已确认 H/Y/D 红）
- `candidate-audit-07-old-noise/textured.glb`，3400212 bytes，SHA256 `ae28f0603b2423f5def597da7b9db3b77cdc2a6c0816fe7980eee14a5910675e`
- `candidate-07-glb-verification.json`：结构、噪声、调用次数、protected hashes、原始 git status 核对结果。

### 6.2 已定位根因：Torch 2.5.1 MPS 缺 `upsample_bicubic2d` 内核

DINOv2 用 `F.interpolate(mode="bicubic", align_corners=False)` 把 37×37 的位置编码网格缩到 16×16 patch 数。Torch 2.5.1 的 MPS 后端**没有该内核**，按 `PYTORCH_ENABLE_MPS_FALLBACK=1` 静默回退到 CPU，但回退路径产生的数值与直接 CPU 计算**完全不同**（见下）。Torch 2.14.1 的 MPS 已有原生 bicubic 内核，结果正确。

证据链（均在 `color-diagnosis/`，新脚本见 §4）：

1. **条件张量对比**（`conditions-comparison.json`）：两版 DINO 输入像素完全一致（nrmse 0.0），VAE 条件 `ref_latents`/`embeds_normal`/`embeds_position` 几乎一致（nrmse ~5e-4）。唯独 `dino_hidden_states` 差异巨大：**nrmse 1.179、cosine 0.290**。差异不可能来自输入或 VAE，只能来自 DINO 前向本身。
2. **CPU 跨版本对照**（目录 `dino251-cpu2`/`dino214-cpu2`）：CPU 上 2.5.1 与 2.14.1 的 DINO final 几乎一致（nrmse 0.0015）。排除权重/输入差异——差异纯由 MPS 路径引起。
3. **逐层边界**（目录 `dino251-mps2`/`dino214-mps2`）：差异自 `embeddings` 阶段即出现（2.5.1 CPU-vs-MPS nrmse 0.092，逐层放大到 final nrmse 1.179）；2.14.1 同对比仅 nrmse 1e-5。
4. **embedding 组件隔离**（`probe_dino_embeddings.py`，`dino-embed-torch251.json` vs `dino-embed-torch214.json`）——决定性证据：
   - `patch_conv`：两版 MPS-vs-CPU 都 nrmse 1.3e-5、cos 0.999999（正确）。
   - `pos_interp_bicubic`：**2.5.1 nrmse 1.319、cos −0.0014（≈随机）**；**2.14.1 nrmse 2.6e-6、cos 0.9999999999（正确）**。
   - `embeddings_sum`：2.5.1 nrmse 0.092；2.14.1 nrmse 1.3e-5。
5. **独立算子验证**（`probe_bicubic_pos.py`、`bicubic-pos-torch251.json`）确认该尺寸 bicubic 在 2.5.1 MPS 上结果错误；`probe_dino_ops.py` 确认 layernorm/gelu/sdpa/softmax 等基本算子两版都正确（nrmse ≤ 4.5e-4），锅不在它们。

**结论**：旧版（2.5.1）全黑 HY3D 是 DINO 位置编码被错误 bicubic 插值破坏、参考条件语义丢失的直接后果。候选版（2.14.1）修复了这个真实 torch bug；其混合路径染红则来自 MLX reference attention 的材质 V head layout，见 §6.4。

**因果链已部分验证**：`capture-214-inject-old-dino`（把旧版 corrupted DINO 隐状态注入候选 pipeline）vs `capture-214-own-dino`（候选自身 DINO）。首步 `noise_pred_0`（shape 36,4,64,64）nrmse 0.28、cos 0.967 相对改变，证明 DINO 差异确实进入 UNet。完整解码到贴图（预期还原为黑）尚未跑——`probe_fixed_conditions.py` 的 `capture` 阶段只截断到首步、不解码；若要肉眼确认需补一个 `--stop-after 15` 的完整 capture + 解码，或用 `audit_mps.py` 跑一次 `--initial-noise` 固定、DINO 注入的完整上色。

### 6.3 修复前候选版（2.14.1）颜色问题：H/Y/D 染红

本节记录已结束的诊断分支；当前结论以 §6.4 为准：

1. `demo.png` 设计意图已确认，不再作为主嫌疑。
2. guidance/reference/mva 首步扫描只改变噪声幅度，不能解释纯 Torch 正确与 MLX 混合染红的分叉；不要先用调权重掩盖 MLX parity。
3. DINO、VAE、基础 Torch 算子在新版均正确；MLX inner UNet 的首个关键布局差异已在 §6.4 定位并修复。

### 6.4 新增决定性证据：颜色分叉发生在 MLX inner UNet

- 参考图 `/Users/daubert/Github/hy3d-mlx-eval/demo.png` 肉眼确认是黑色 H/Y/D 与红色 3。
- 在 `/Users/daubert/Github/hy3d-upgrade-evidence/20261027/color-diagnosis/full-g3-r1-m1-fixed-cameras-2/`，候选 Torch 2.14.1 使用 `conditions-torch214`、旧版初始噪声 fixture、fixtures 中真实相机 `[0,90,180,270,0,180]`，纯 Torch 15 步前视图保持黑 H/Y/D、红 3。
- 在 `/Users/daubert/Github/hy3d-upgrade-evidence/20261027/color-diagnosis/full-hybrid-g3-r1-m1-fixed/`，完全相同的输入/条件/相机，仅启用仓库 `--mlx` dispatch（第 1 步 Torch，第 2–15 步 MLX），前视图变为 H/Y/D 深红；与 run 07 视觉一致。
- 逐步比较：`noise_pred_0` 完全一致；`noise_pred_1` nrmse 0.1088、cos 0.9941；`latents_14` nrmse 0.4171、cos 0.9167。`capture-hybrid-compare-step2` 的 MLX/Torch 边界输出 mean abs 0.047266、p99 0.398438、max 4.521484。
- `HUNYUAN_MLX_FP32=1` 的边界 mean abs 0.046533，改善约 1.6%，不能视为精度模式修复。`mva_scale=0.5` 的完整混合实验因约 10 分钟无阶段输出、进程处于 macOS `U` 状态而中止，目录 `full-hybrid-g3-r1-m05-fixed/aborted.json` 保留失败记录。
- 实际 Torch reference processor 的 checkpoint 实现先拼接 albedo/MR 的 V 投影，再按 5 个 head 切分；旧 MLX 实现先分别按 head 切两个 V，改变了每个 head 看到的训练通道。修复后第 2 步 reference 输出 Torch/MLX NRMSE 从 `1.329121` 降至 `0.002778`，首个 down block 从约 `0.110` 降至约 `0.004`。
- 完整修复复验目录：`/Users/daubert/Github/hy3d-upgrade-evidence/20261027/color-diagnosis/full-hybrid-refsplit-cache1g-g3-r1-m1-fixed/`。15 步 exit 0，12 张视图生成，耗时 `115.98 s`；MLX 第 2–15 步约 `6.5–7.7 s`，MLX peak `8.942 GiB`。`generated_0.png` 为黑 H/Y/D、红 3；区域红色占比 H/Y/D 均为 `0`，3 为 `0.3525`（纯 Torch `0.3545`）。最终 `latents_14` 相对纯 Torch NRMSE 从修复前 `0.4171` 降至 `0.0967`。
- 权重漏载修复后的固定条件复验目录：`/Users/daubert/Github/hy3d-upgrade-evidence/20261027/color-diagnosis/full-hybrid-up0-loaded-01/` 和 `full-hybrid-up0-loaded-02/`。两次 15 步 exit 0，耗时 `67.812/68.498 s`；15 个 `noise_pred`、15 个 `latents` 及 12 张 PNG 逐元素一致。相对纯 Torch `latents_14` NRMSE `0.003734`；正面仍为黑 H/Y/D、红 3。
- 完整运行能稳定完成的关键实验条件是 `--mlx-cache-limit-mib 1024`。这是 MLX 空闲缓存回收阈值，不是瞬时硬内存上限；不应把它误写成 1 GiB 总内存限制。
- Diffusers 真正执行的 outer UNet 是 `diffusers_modules.local.modules.UNet2p5DConditionModel`，reference processor 是 `diffusers_modules.local.attn_processor.RefAttnProcessor2_0`；基础 inner UNet 是环境里的 Diffusers `UNet2DConditionModel`。模型目录与缓存的 `attn_processor.py` SHA256 一致；工作区同名实现不是这次固定条件 Torch 路径。工作区实现已另有 CPU 回归测试覆盖相同 layout，但二者证据边界必须分开。CPU-only 真实 checkpoint 加载烟测已确认这些类和路径，未执行推理。
- 修复后完整严格 GLB audit 目录：`/Users/daubert/Github/hy3d-upgrade-evidence/20261027/candidate-audit-08-ref-layout-fixed/`。`audit.json` exit 0，fallback=0，15 次生成（1 次 Torch + 14 次 MLX），inference `189.718 s`，总耗时 `214.979 s`，MLX peak `8.942 GiB`，source SHA 校验均为 `matches_model_source=true`。`textured.glb` 为 3,500,288 bytes，SHA256 `a4fe11daf92e584d434e524bd8fb406cd17dcee83fbf8fc5a87b50edfc5aa1cf`；结构为 24,087 顶点、40,000 面、1 个 PBR 材质、2048×2048 base-color，几何/UV 有限。
- 该 GLB 的 Blender 5.2.3 CPU 四视图（`glb-view-0/90/180/270.png`）显示正面黑 H/Y/D、红 3，侧面与背面贴图连续，没有翻转或丢失。这个本地查看结果不替代用户设备或独立 3D 查看验收。
- upsampler 权重修复后的完整严格 audit 目录：`/Users/daubert/Github/hy3d-upgrade-evidence/20261027/candidate-audit-09-up0-loaded/`。exit 0，fallback=0，15 次生成（1 次 Torch + 14 次 MLX），inference `112.975 s`，总耗时 `125.763 s`，MLX peak `5.64 GiB`；`textured.glb` SHA256 `3daab3b43fae5f72d7df41c4dbdafd3cafe9a504b7bf3bd36704ae97cc1eb7b1`，结构为 24,087 顶点、40,000 面、1 个 PBR 材质。Blender 5.2.3 CPU 四视图显示正面黑 H/Y/D、红 3，侧背面贴图连续；用户设备/独立查看仍未验收。
- 修复前固定条件性能/漂移重复证据在 `color-diagnosis/perf-torch-repeat01/`、`perf-hybrid-repeat01/` 及其 `*-compare.json`。Torch-only 两次逐元素一致；Hybrid 漂移来自后续定位的 upsampler 权重漏载。这些修复前 Hybrid 耗时和质量不能作为现行基线。
- 修复前两次 2 步 `HUNYUAN_MLX_COMPARE=1` 内部 trace（`color-diagnosis/nondet-trace-a/`、`nondet-trace-b/`）及 FP32 对照（`nondet-fp32-a/b/`）记录了漏载 upsampler 权重导致的分叉；这些目录保留为负证据，不是当前实现的漂移结论，也不是 P2 所需的 Metal GPU trace。
- 外部 stage wrapper 的 fp16 trace（`color-diagnosis/stage-trace-c/d/e/f`）证明 `down_0..3`、`mid`、所有 down 残差、`up_0` 三个 ResNet 与 nearest 输入逐元素一致，只有 `up_0_upsample` 输出 NRMSE `1.419`；FP32 `stage-trace-g/h` 同样 NRMSE `1.414`。检查 loader 后确认 `up_blocks.0.upsamplers.0.conv.weight/bias` 漏载，导致每次随机初始化；并非 Metal kernel 非确定性。修复后 `stage-trace-fixed-a/b` 的 `up_0_upsample`、`up_0` 和最终输出均逐元素一致。

**当前结论**：候选版染红和重复运行漂移的两个代码根因均已修复：reference attention 材质 V head layout，以及 MLX loader 漏载 `up_block_0` upsampler 权重。修复后固定条件 Hybrid 两次逐元素稳定，最终 `latents_14` 相对 Torch-only NRMSE `0.003734`；完整严格 GLB audit 与 Blender 本地查看通过。P2 命名 Torch-only Metal trace 已通过；P3 bake CPU workaround 已通过真实尺寸数值对照并移除。P5 已有三轮串行旧/新版栈级性能/内存样本，但不能归因成 Torch 版本单独加速；用户设备/独立查看和默认环境切换仍开放，尚未切换默认 MLX 路径。

## 7. 不要重复的失败尝试

- 旧版完整配对 benchmark 01：超过 10 分钟尚无 multiview，进程 footprint 曾 48.1G；主动 SIGTERM，exit 143。不是完整速度基线，不能用历史 476.476 s 和新版直接计算加速比。
- 整流水线 xctrace 02/03：AGXMetalG17X `setComputePipelineState` 原生 SIGSEGV，调用涉及 Torch MPS reduction/max；最小 max 四 dtype 在普通/profiler/xctrace/两者结合都通过，真实根因未定位。
- attach trace 04：raw trace 约 15 GiB，转换 RSS >12 GiB，工作负载停滞，主动终止。trace 不完整，不算 GPU 验收。
- 普通无 trace run 05：MLX step 由 8–12 s 逐渐到 26/59/79/94 s，主动终止。这证明不能仅归咎于 trace；JSONL 中已完成 reference placement 证据保留。
- 后续 trace 必须限制到有界 Torch-only 阶段，不要再直接录整个 MLX+MPS 全流水线。

## 8. 下一位 agent 的建议顺序

1. 先读本文件、`docs/torch-upgrade-mps-plan.md`、`docs/mlx-migration-roadmap.md`、验证报告；检查当前 branch/status、进程和 ComfyUI queue。保护 dirty 文件。重点先读 §6，根因已定位，不要重复第 7 节列出的失败实验。
2. **颜色根因、固定条件复验及完整 GLB 本地查看已完成（§6.4）**：不要重复 Torch DINO、guidance 扫描、修复前 MLX 布局或 upsampler 权重实验。后续颜色工作用于用户设备/独立 3D 资产验收。
3. 固定条件 Torch-only 与 hybrid 对照及修复后完整复验均已完成；复现命令在 §9.2。任何新增 GPU 实验必须使用 fresh output、1 GiB MLX idle-cache threshold、超时/内存观察并保持串行。
4. GPU 证据：P2 命名 Torch-only trace 已完成（§11）；P3 真实尺寸 bake CPU 例外已通过数值对照并完成代码修改（§12）。这仍不证明完整 pipeline 全 GPU。
5. 内存/速度：P5 三轮串行样本已完成，见 §13；候选中位数总耗时 117.711s、旧版 196.740s，内存计数也较低，但差异混合了 Torch、MLX 修复和 fallback 配置，不能宣称 Torch-only 加速。未完成质量/设备/安全内存决策前不要推广候选默认环境。
6. R1 权重 bundle 已完成（见 `HANDOFF-r1-mlx-weight-bundle-20261028.md`）。R2 RealESRGAN 的独立 MLX 试点数值接近，但真实尺寸稳态约慢 2.3–2.5 倍，故未切默认路径（见 `HANDOFF-r2-realesrgan-20261028.md`）。R3 已加入 opt-in inner UNet bundle dispatch，两步 fresh checkpoint 对照逐元素一致（见 `HANDOFF-r3-bundle-dispatch-20261028.md`）；MLX UniPC scheduler 的真实 15 步离线 replay 已完成（见 `HANDOFF-r3-unipc-scheduler-20261028.md`）；外层 DINO projector 的真实权重对照已通过（见 `HANDOFF-r3-dino-projector-20261028.md`），7 组外层条件权重 bundle 通过 Torch-free 加载及 projector 输出对照（见 `HANDOFF-r3-condition-bundle-20261028.md`）；16 层 reference cache 与 4 组 position indices 的真实请求合同已捕获（见 `HANDOFF-r3-cache-contract-20261028.md`）；opt-in MLX first-step 完整 15 步固定条件运行通过（见 `HANDOFF-r3-first-step-20261028.md`）；真实 15 步 CFG+UniPC MLX replay 数值通过（见 `HANDOFF-r3-cfg-scheduler-20261028.md`）；opt-in resident loop 的第 2–15 步不再逐步 Torch/MLX 往返，真实固定条件 15 步逐步与图像对照及同实例 reset 通过（见 `HANDOFF-r3-resident-loop-20261028.md`）。默认 hybrid dispatch 仍从 Torch checkpoint 加载，outer reference/条件生产和首步输入桥接尚未移除，不能宣称全 MLX。全 MLX 并不自动更快，需要端到端测量和质量对照。

R3 首步 MLX output 已直接进入 opt-in resident loop：真实 15 步固定条件运行与前一桥接版的 45 个逐步 trace 数组、首步输出和 12 张图逐元素一致；本阶段交接见 `HANDOFF-r3-direct-first-output-20261028.md`。默认路径仍未切换。
R3 outer DINO projector 已增加 bundle-backed MLX adapter 作为 opt-in consumer；单步 DINO/首步 noise 和 15 步原图对照通过，详见 `HANDOFF-r3-dino-projector-consumer-20261028.md`。position indices 的后续迁移见下段，dynamic reference-write 仍待迁移。

R3 position voxel indices 已加入 opt-in MLX producer/consumer；真实四分辨率 cache 数值、首步输出及 12 张图与 DINO-only resident 基线逐元素相同，详见 `HANDOFF-r3-position-indices-20261028.md`。dynamic reference-write 及外层条件 embedding 仍待迁移，默认路径未切换。

R3 动态 reference cache 边界阶段已建立 opt-in MLX 转换与请求 reset 合同；真实 16 层 cache、首步和 12 张图与 position-indices 基线逐元素/逐字节相同，详见 `HANDOFF-r3-reference-cache-boundary-20261028.md`。随后已完成真正 MLX `unet_dual` opt-in forward，见下段。

R3 已实现 opt-in MLX `unet_dual` reference forward，并完成 686 键映射、逐层离线 replay、15 步集成与同实例 reset；最大 cache NRMSE `0.006852`。首版工程门槛已冻结为 cache `0.01`、reference write `0.03`，当前报告通过；详见 `HANDOFF-r3-mlx-dual-reference-forward-20261028.md`。这是 fp16 近似路径，默认仍不切换，outer Torch 合同和更广输入验收仍开放。

R3 reference 覆盖 replay 已完成三组 reference-write：512px 中央遮挡和 384px reference
分别在后段 up block 超过 cache 门槛（最大 NRMSE `0.010211`、`0.014230`），256px
reference 通过（`0.002661`）；三组 reference-write 均在 `0.03` 内。详见
`HANDOFF-r3-reference-coverage-20261028.md`。门槛保持不变，下一步先做 fp32/逐层
误差诊断，再决定是否继续扩大 MLX 边界。

上述两个失败输入的 MLX reference fp32 参数 replay 均已恢复到原门槛内：最大 cache
NRMSE 分别为 `0.006395`、`0.006528`，write 为 `0.023046`、`0.021966`。详见
`HANDOFF-r3-reference-fp32-diagnostic-20261028.md`。下一优先级是独立 reference fp32
opt-in、cache consumer dtype 合同及完整 15 步/reset 验收；正式 dispatch 尚未改变。

独立 reference fp32 opt-in 已实现：`HUNYUAN_MLX_REFERENCE_FP32=1` 只提升
`unet_dual` reference，native cache 在 consumer 边界回到 generation fp16。两组超限
输入的集成 replay 均通过（最大 cache `0.006399/0.006531`），reset 合同通过；详见
`HANDOFF-r3-reference-fp32-consumer-20261028.md`。后续真实 resident 输入发现遮挡
512px 在 reference fp32 下仍失败：cache 最大 NRMSE `0.014330` > `0.01`，write
`0.026173`，逐步 latent 最大 NRMSE `0.012732`，最大像素差 `79/255`。本轮不注入
预生成非 reference 条件，使用固定 posterior noise，与此前 coverage replay 的输入
合同不同；旧 PASS 不覆盖该新输入。详见 `HANDOFF-r3-reference-fp32-resident-20261028.md`。
下一优先级是冻结本轮失败张量/哈希、比较新旧 sample 并逐层定位，保持原门槛及默认 Torch
reference 路径，暂停扩大 outer condition/VAE 边界。

当前阶段判定：P0/P1 完成；P2 placement/calls 与命名 Torch-only trace 通过（不是全流水线全 GPU）；P3 bounded ops 与真实 bake CPU 例外验证/移除完成；**P4 颜色——旧版全黑、候选 hybrid 染红和随机漂移根因均已定位修复，最终固定条件颜色及重复性复验通过，run 09 完整 GLB 结构与本地 Blender 四视图已核对，用户设备/独立 3D 资产验收未完成**；P5 三轮旧/新版栈级性能/内存样本完成，但 Torch-only 归因、安全内存与默认切换仍未通过；R1 inner UNet 权重 bundle 的 Torch-free strict load 完成；R2 RealESRGAN 独立试点因性能回退未切默认；R3 inner UNet bundle、独立 UniPC scheduler replay、DINO projector 对照、外层条件权重 bundle、真实 reference/cache 合同、opt-in first-step 15 步运行、CFG+UniPC replay 和 opt-in resident loop 切片通过，完整 R3 迁移未完成；DINO projector consumer、position indices producer/consumer、opt-in MLX reference forward 及独立 reference fp32/cache dtype 切片已通过先前精确输入 replay，但本轮真实 resident 遮挡输入的 reference/15 步质量门槛失败，暂停扩大边界；性能、安全内存和默认切换仍开放。

本轮追加 reference 输入敏感性诊断见
`HANDOFF-r3-reference-input-sensitivity-20261028.md`：遮挡失败的误差首次在
`up_2_0_0` 明显越门槛；正式 fp32 输入/cache fp16 与离线参数 fp32 合同在同一输入下
结果几乎相同，输入 dtype 不是主因。旧 coverage 精确输入仍通过。

逐层 trace 已将明显放大追到 `up_1_1_0` 的 FF/GEGLU；同输入、同权重的孤立 FF
CPU Torch/MLX 对照差异较小，支持上游误差被放大，首个来源仍待定位。Torch fp32
对照也未消除误差。证据和下一步见 `HANDOFF-r3-reference-up2-stage-trace-20261028.md`；
保持默认 Torch reference 与 cache `0.01` / write `0.03` 门槛。

随后对冻结的 `norm3 → ff.net.0.proj → GEGLU` 做了离线归因：同 checkpoint 的 CPU
projection 重放与 Torch 捕获几乎逐元素一致，MLX 捕获差异仅 `0.000357`；gate
输入差异被 GEGLU 放大到 `0.007472`。这支持误差来自更早的 `up_1.1` hidden state
分叉，尚未定位到具体 attention/skip/normalization 分支。详见
`HANDOFF-r3-reference-ff-attribution-20261028.md`；默认 dispatch 和门槛继续冻结。

随后确认 Diffusers Transformer2D GroupNorm epsilon 为 `1e-6`，当前 MLX 默认是
`1e-5`。单变量 reference replay 将失败输入 max cache 从 `0.014330` 降到
`0.013018`，仍未过 `0.01`；up concat 拆分显示误差主要沿主干传播。详见
`HANDOFF-r3-reference-norm-epsilon-20261028.md`。该 replay 启动前 queue 查询被
自动审查 429 拒绝，流程证据标为未验证；默认路径和门槛继续冻结。

现已将 `1e-6` 隔离接入 opt-in MLX reference，generation 默认仍为 `1e-5`；
85/85 unittest 通过。新会话在成功确认队列及进程后完成冻结失败 sample replay：
最大 cache `0.0130166` 仍超过 `0.01`，write `0.0257892` 在 `0.03` 内。
详见 `HANDOFF-r3-reference-epsilon-replay-20261028.md`；默认 reference 不切换。

随后对冻结 `up_1.1` Transformer2D GroupNorm 输入做 32 组逐组 float64 统计：
最小组标准差 `2.95917`，没有接近常数的组；此处合成低方差 GroupNorm
异常尚不能解释真实误差。详见 `HANDOFF-r3-reference-group-variance-20261028.md`。
下一步从 `up_1.1.resnet` 前继续追踪 concat、ResNet 内部与主干/skip。

随后已有 trace 的边界对照显示：`up_1.1.concat` 在进入 ResNet 前已有
NRMSE `0.004855`（主干半部 `0.004952`、skip 半部 `0.004619`），ResNet
输出为 `0.004198`，没有放大；更明显放大出现在 `up_1.1.attn` (`0.007576`)
及其 Transformer FF/output (`0.007635/0.007837`)。详见
`HANDOFF-r3-reference-resnet-boundary-20261028.md`，下一步追踪
`up_1.0` attention 到 concat 以及 `up_1.1` attention1/FF 的同输入算子。

新增 coarse boundary trace 已把当前 resident 失败输入的首次可观测分叉前移到
`conv_in → down_0`：`conv_in` 逐元素一致，`down_0` NRMSE `0.004971`，之后
`mid` `0.005589`、`up_0.output` `0.006143`。详见
`HANDOFF-r3-reference-coarse-boundary-20261028.md` 和
`docs/r3-reference-coarse-boundary-20261028.json`；下一步拆 `down_0` 的两个
ResNet 与 attention。

随后把首次分叉定位到 `down_0.resnets[0].conv1`：前置 `norm1` NRMSE 仅
`5.9e-8`，Torch MPS 与 CPU conv1 重放 NRMSE `9.4e-7`，MLX Metal 与 CPU
重放 NRMSE `0.002060445`；实际 MLX OHWI 权重/bias 与 checkpoint 逐元素一致。
详见 `HANDOFF-r3-reference-down0-conv1-20261028.md`，证据目录为
`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-reference-down0-resnet-01/`。
下一步做最小 MLX GPU Conv2d probe，确认并分析该局部执行差异。

最小 MLX Metal `Conv2d` probe 与完整 trace 输出逐元素相同；按 4 块输入通道
累加可把局部 NRMSE 从约 `0.002059` 降至 `4.90e-7`。但只替换这一层的
16 层 reference replay 使最大 cache 从 `0.0130166` 升至 `0.0147171`，
仍超 `0.01`；write 为 `0.0238399`，在 `0.03` 内。详见
`HANDOFF-r3-reference-conv1-split-probe-20261028.md`。单点改法未接入默认，
下一步先按卷积形状/层位隔离其他数值因素。

随后将 65 个高通道 Conv2d 临时按 4 块输入通道拆分并完成冻结 reference replay：
cache 最大 NRMSE `0.0092018`、reference write `0.0158142`，均通过原
`0.01/0.03` 门槛；验证器 exit 0。该实现仍是临时诊断，未接入默认路径，尚未
覆盖其他输入、完整 15 步或性能。详见 `HANDOFF-r3-reference-split-all-conv-20261028.md`，
后续通过 Winograd 调度探针收窄到卷积算法路径。

Winograd fallback 双输入阶段已完成，详见
`HANDOFF-r3-reference-winograd-fallback-20261028.md`。当前 MLX 0.32.4 上，
`MLX_CONV_WINOGRAD_WORKING_SET=1` 使真实 `down_0` Conv2d 与 Torch MPS 的
NRMSE 降至 `9.37e-7`；在 1 GiB MLX idle-cache 下，occluded512 的完整
reference cache/write NRMSE 为 `0.007403/0.013200`，reference384 为
`0.004554/0.010591`，均通过原 `0.01/0.03` 门槛，85/85 unittest 通过。
该变量影响整个 MLX 进程；完整 15 步、同实例 reset、颜色、性能/内存仍待验证，
默认 Torch reference 与 dispatch 均不切换。离线 replay 目前保留半精度 capture
输入，真实 opt-in fp32 reference 会将输入也转 fp32；下一阶段须验证这个差异。

随后完成 Winograd fallback 的 opt-in dispatch，详见
`HANDOFF-r3-reference-winograd-optin-20261028.md`。新增
`HUNYUAN_MLX_DISABLE_WINOGRAD=1`，默认环境不变；真实遮挡 512 resident loop
的 reference cache/write 为 `0.007406/0.013201`，15 步 latent 最大 NRMSE
`0.006442`，同实例重复图像像素逐元素一致、两次 reset 清空。12 图平均差
`0.0595/255`，最大差 `43/255`，超过本轮探索性 `32/255` 上限；fallback 单样本
耗时 `107.85s`，此前同输入 MLX fp32 为 `83.30s`。因此 opt-in 质量和状态合同
部分通过，图像最大差与性能仍开放，默认 dispatch 不切换；全量回归 `88/88`。
独立 `reference384` 随后通过相同探索性门槛：cache/write `0.004559/0.010594`、
图像最大差 `20/255`、同实例重复 exact；耗时 `84.40s`，同输入旧 MLX fp32
`73.65s`，比值 `1.146`。下一步只剩 fixed512 筛查及遮挡 512 后段误差/性能决策。

## 9. 可复制验证命令

工作目录均为 `/Users/daubert/Github/hy3d-torch-upgrade`。

```sh
rtk proxy git status --short
rtk proxy git log -4 --oneline
rtk proxy curl -s --max-time 3 http://127.0.0.1:8188/queue
rtk proxy /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python -m unittest discover -s tests -v
rtk proxy git diff --check
```

以下是 run 07 的等价复现命令。`candidate-audit-08-replay` 必须不存在；先确认没有其他 GPU 推理任务，只在需要完整复现时运行：

```sh
rtk proxy env PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python \
  tools/quality_demo/audit_mps.py \
  --mesh /Users/daubert/Github/hy3d-mlx-eval/shape_quality.glb \
  --output-dir /Users/daubert/Github/hy3d-upgrade-evidence/20261027/candidate-audit-08-replay \
  --mlx-cache-limit-mib 1024 \
  --initial-noise /Users/daubert/Github/hy3d-upgrade-evidence/20261027/color-diagnosis/initial-noise-torch251-seed0.npy
```

### 9.2 颜色诊断复现命令（§6 证据链）

遵守 §3 的“一次只跑一个 GPU 推理”约束：旧环境命令在受保护 checkout + `hunyuan3d-shape` venv；候选命令在 worktree + `hunyuan3d-torch-upgrade` venv。不要在两环境并行。

```sh
FIX=/Users/daubert/Github/hy3d-upgrade-evidence/20261027/color-diagnosis/fixed-conditions-fixtures
EVID=/Users/daubert/Github/hy3d-upgrade-evidence/20261027/color-diagnosis

# 0) 固定 fixtures（一次；候选环境即可，无 GPU 推理竞争）
rtk proxy env PYTHONUNBUFFERED=1 \
  /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python \
  tools/quality_demo/probe_fixed_conditions.py --stage fixtures --output-dir $FIX

# 1) 两版本条件张量（DINO/VAE ref/normal/position），固定 posterior 噪声隔离采样器
rtk proxy env PYTHONUNBUFFERED=1 \
  /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python \
  tools/quality_demo/probe_fixed_conditions.py --stage conditions --fixtures $FIX --output-dir $EVID/conditions-torch214
rtk proxy env PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
  /Users/daubert/Github/.venvs/hunyuan3d-shape/bin/python \
  tools/quality_demo/probe_fixed_conditions.py --stage conditions \
  --checkout /Users/daubert/Github/Hunyuan3D-2.1-macOS --fixtures $FIX \
  --output-dir $EVID/conditions-torch251 --fallback 1

# 2) 条件张量逐文件对比（应见 dino_hidden_states 差异巨大、其余 ~5e-4）
rtk proxy /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python \
  tools/quality_demo/probe_fixed_conditions.py --stage compare \
  --left $EVID/conditions-torch214 --right $EVID/conditions-torch251 --output $EVID/conditions-comparison.json

# 3) DINO embedding 组件隔离（决定性证据：pos_interp_bicubic 在 2.5.1 MPS 上 cos≈-0.0014）
rtk proxy env PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
  /Users/daubert/Github/.venvs/hunyuan3d-shape/bin/python \
  tools/quality_demo/probe_dino_embeddings.py --output $EVID/dino-embed-torch251.json \
  --checkout /Users/daubert/Github/Hunyuan3D-2.1-macOS --fixtures $FIX --device mps --fallback 1
rtk proxy env PYTHONUNBUFFERED=1 \
  /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python \
  tools/quality_demo/probe_dino_embeddings.py --output $EVID/dino-embed-torch214.json \
  --fixtures $FIX --device mps --fallback 0

# 4) 因果链验证：把旧版 corrupted DINO 注入候选 pipeline，对比首步 noise_pred
rtk proxy env PYTHONUNBUFFERED=1 \
  /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python \
  tools/quality_demo/probe_fixed_conditions.py --stage capture --fixtures $FIX \
  --output-dir $EVID/capture-214-inject-old-dino --inject $EVID/conditions-torch251 \
  --steps 15 --stop-after 1 --initial-noise $EVID/initial-noise-torch251-seed0.npy
rtk proxy env PYTHONUNBUFFERED=1 \
  /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python \
  tools/quality_demo/probe_fixed_conditions.py --stage capture --fixtures $FIX \
  --output-dir $EVID/capture-214-own-dino --steps 15 --stop-after 1 \
  --initial-noise $EVID/initial-noise-torch251-seed0.npy
```

把旧版 corrupted DINO 注入后若想肉眼确认“变黑”，需跑完整 15 步并解码：把上面第 4 步的 `--stop-after 1` 去掉，再用既有 `audit_mps.py` 的渲染路径解码 `latents_*`；或直接用 `audit_mps.py --initial-noise ...` 配合一个能注入 DINO 的入口（当前未内置，需小改）。

修复后的固定条件 hybrid 复现（只在确认没有其他 GPU 推理后运行；目标目录必须不存在）：

```sh
rtk proxy curl -s --max-time 3 http://127.0.0.1:8188/queue
rtk proxy env PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python \
  tools/quality_demo/probe_fixed_conditions.py --stage capture \
  --fixtures $FIX \
  --output-dir $EVID/full-hybrid-refsplit-cache1g-g3-r1-m1-replay \
  --steps 15 --stop-after 0 --initial-noise $EVID/initial-noise-torch251-seed0.npy \
  --guidance-scale 3 --reference-scale 1 --mva-scale 1 --mlx --mlx-cache-limit-mib 1024
```

该命令会写 12 张 `generated_*.png` 和逐步 `noise_pred_*.npy`/`latents_*.npy`，不会写交付 GLB。它只能复验固定条件颜色；不要把它当成完整 `audit_mps.py` GLB、Metal trace 或默认环境验收。

## 10. 已交付资产与保护证据

`/Users/daubert/Github/hy3d-mlx-eval/` 下：

- `shape_quality.glb`：151613 vertices /303234 faces，用户认可白模；SHA256 `af195a0d8dabeb396fcb368dd2740b65be588647d0520184142646c3f9c43a80`。
- `shape_quality_textured.glb`：原始上色，3 变黑；SHA256 `d5aa1ac89fd99db20b3fa2bed3747f58af1dbb67c6acd340b1c12ed62223e8c8`。
- `shape_quality_textured_red3.glb`：用户授权 Python 精确局部修红 3，几何/UV/区域外像素保持；SHA256 `de694c37a22f7b6ffc641769d9a8164538acab3fbe937e8942439b14de0836c9`。
- `shape_textured.glb`：更早历史交付，禁止覆盖；SHA256 `0ac916c04e3a3033897ab0dfac7b016187749494cbd231bfccff3a80168995b0`。
- `demo.png` 是原始参考图。五个文件最近均校验 unchanged。

初始完整快照、权重 hash、旧依赖和原始 dirty 状态在外部 `baseline/manifest.json`、`baseline/status.txt`、`baseline/runtime-overlay.patch`。不要把保护检查误认为新版质量验收。

## 11. P2 named Torch-only Metal trace closure (2026-10-27)

P2 trace gate 已完成。五个真实尺寸 Torch/MPS 窗口在候选 Torch 2.14.1、`PYTORCH_ENABLE_MPS_FALLBACK=0` 下录制并关联目标 PID。每个窗口 target exit=0，marker begin/end 完整落在 trace 时间范围内，PyTorchMPS signpost 与同 PID Metal GPU interval 均存在，fallback signpost=0。原始 trace 和导出表保留在外部 evidence 根目录，不进入 Git；阶段交接见 [`HANDOFF-p2-metal-trace-20261027.md`](HANDOFF-p2-metal-trace-20261027.md)。

| 窗口 | 外部目录 | PyTorch signpost | Metal GPU interval | 结果 |
| --- | --- | ---: | ---: | --- |
| DINO + VAE encode ×3 | `p2-conditions-trace-01` | 2,571 | 1,314 | pass |
| VAE decode | `p2-vae-decode-trace-01` | 441 | 296 | pass |
| RealESRGAN | `p2-superres-trace-01` | 2,445 | 1,055 | pass |
| reference UNet | `p2-reference-trace-02` | 2,138 | 784 | pass |
| first-step outer/inner UNet | `p2-first-step-trace-03` | 5,961 | 9,251 | pass |

`p2-first-step-trace-02` 是负结果：目标 forward 完成，但 recorder 在 marker end 前约 5 秒结束；新增汇总器的 `fully_recorded` 检查会 fail closed。此次 trace 只证明上述命名窗口内目标进程执行 Metal GPU 工作，不证明显式 CPU preprocessing、mesh/UV/native bake、`.cpu().numpy()` 桥接或完整 pipeline 全程 GPU。P3 后续关闭了真实 bake CPU workaround（§12）。P2 时正常 Metal 权限下全量 unittest 为 30/30；独立 reviewer dispatch 仍因 429 未返回。

## 12. P3 真实尺寸 bake MPS 路径（2026-10-27）

阶段交接见 [`HANDOFF-p3-bake-mps-20261027.md`](HANDOFF-p3-bake-mps-20261027.md)。Torch 2.14.1 的真实尺寸 MPS gather 可用；`torch.isin` 在约 11M 索引上会尝试约 348 GB MPSGraph 临时分配，因此改用保持顺序的 `inner_mask[valid_idx]`。小输入 CPU oracle 与真实尺寸 CPU/MPS 边界对照均通过。完整严格 audit 在新目录 `/private/tmp/hy3d-p3-bake-audit-01/` 以 fallback=0、15 步、2 次 bake 正常完成；小型审计证据已复制至外部 `p3-bake-mps-01/`。run09 对照的 multiview、enhance、inpaint 和最终贴图逐字节相同。bake 中间浮点数组最大绝对差小于 `0.001`；GLB 容器 hash 不同但几何、UV、内嵌贴图一致。剩余主要门槛是 P5 旧/新版配对性能/内存与用户设备/独立 3D 查看；不要仅由本次 bake 烟测宣称全流水线 GPU 或 Torch 升级加速。

## 13. P5 串行性能/内存样本（2026-10-28）

阶段交接见 [`HANDOFF-p5-performance-memory-20261028.md`](HANDOFF-p5-performance-memory-20261028.md)。在空 ComfyUI queue、相同 mesh/固定初始 diffusion noise/6 视角/512/15 步/1 GiB MLX cache threshold 下，候选 2.14.1 与旧版 2.5.1 各完成 3 次完整 benchmark。候选总耗时中位数 `117.711s`（范围 `117.180–123.536s`），旧版 `196.740s`（`194.004–199.497s`）；候选 RSS/MPS driver/MLX peak 中位数约 `8.20/26.46/5.64 GiB`，旧版约 `11.02/36.67/8.98 GiB`。旧版 swap delta 为 `+6.97–+9.06 GiB`，候选为 `−0.65–+3.32 GiB`。这些是当前两套完整栈的可复现样本，不是 Torch 2.14.1 的独立加速证明：旧版 fallback=1、旧 MLX layout/upsampler 和 bake 路径不同；只有输入 mesh、初始 diffusion latents、cache policy 与串行条件被匹配。外部摘要位于 `p5-performance-01/summary.json`。新会话创建仍因本地 `codex_tui` transport `http://127.0.0.1:54618/mcp` send error 失败，未绕过。

## 14. R3 Winograd opt-in fixed512 / occluded512 复验（2026-10-29）

阶段交接见 [`HANDOFF-r3-reference-winograd-optin-20261028.md`](HANDOFF-r3-reference-winograd-optin-20261028.md) 的 2026-10-29 补充。
本轮完成了 fixed512 的合同验收，并复跑 occluded512 诊断性能异常。证据归档于
`/Users/daubert/Github/hy3d-upgrade-evidence/20261029/r3-reference-winograd-optin-20261029/`，
汇总 SHA256 `f2d0d4dda929cdfd71aa296a73268aa4185d3b06a86652b888ca4e481493c52f`。

- fixed512：cache `0.00391466`、write `0.01101064`、latent `0.00364171`、图像
  `0.03951/255` mean、`26/255` max，耗时 `81.282s`，相对既有同输入 fp32
  `81.508s` 为 `0.9972x`；探索性门槛通过。
- occluded512：cache `0.00740603`、write `0.01320104`、latent `0.00644233`、
  图像 `0.05954/255` mean、`43/255` max，耗时 `83.437s`；43/255 位于
  `generated_5.png` 的 17 个像素（`x=206..211,y=309..333`），
  `generated_11.png` 另有 2 个超 32 像素。该质量
  max 门槛仍失败，默认路径保持不变。
- 旧记录的 `107.848s/83.297s=1.295x` 性能回退未复现；新复验为
  `83.437s/83.297s=1.002x`，且 12 图与旧 opt-in 逐像素 exact。29.5% 只能作为
  单次异常记录，尚不足以归因或宣称性能结论。

运行前 queue 均为空，只有既有 ComfyUI 服务；输出未写入仓库或交付 GLB。
默认 Torch reference、默认 Winograd 和默认 MLX dispatch 未切换。

## 15. R3 occluded512 逐步离线诊断（2026-10-29）

阶段证据见 [`HANDOFF-r3-occluded512-late-step-20261029.md`](HANDOFF-r3-occluded512-late-step-20261029.md)。
仅对已归档 Torch reference 与 Winograd-disabled MLX opt-in resident trace 做 CPU
离线对照，无新 GPU 推理或模型代码改动。15 步 `noise_pred` NRMSE 从
`0.00121164`（step 0）增至 `0.01022438`（step 14），latent 从 `0.00025900`
增至 `0.00644233`；step 10–14 的 noise 最大差多在空间 `(y≈43,x≈35–36)`。
明细 JSON（含源文件 SHA256）归档于
`/Users/daubert/Github/hy3d-upgrade-evidence/20261029/r3-reference-winograd-optin-20261029/occluded512-late-step-diagnostic.json`，
SHA256 `26bd4518e8bb208d109ceea8f947090ef5a8a2cca00b7de39f493d05920fe38c`。
该证据只说明首步已有差异且后段累积，尚未证明具体算子根因，也未建立
latent 坐标到 `generated_5/11.png` 局部超阈值像素的因果映射。occluded512
的 `43/255` 探索性最大差仍未过 `32/255` 门槛，默认路径保持不变。

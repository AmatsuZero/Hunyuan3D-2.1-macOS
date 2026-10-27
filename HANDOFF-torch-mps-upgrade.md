# Hunyuan3D macOS：Torch/MPS 升级与颜色诊断交接

更新日期：2026-10-27。用户要求交接给其他 agent；本轮在完成 run 07、验证和提交后停止发起新实验。

## 1. 接手结论

- 当前阶段是 **隔离验证 Torch 2.5.1 → 2.14.1**，随后按 roadmap 逐步迁移 MLX。尚未切换默认环境，也没有开始全面 MLX 迁移。
- 新版已完成关闭自动 fallback 的完整上色；所审计神经网络的参数、计算输入/输出均在 MPS，MLX 部分在 Metal。不能据此宣称全流水线都在 GPU：显式 CPU 路径和完整 GPU trace 验收仍未完成。
- 颜色质量仍不通过：输入是黑 H/Y/D、红 3；候选输出在**原始前视角多视图生成阶段**已经把 H/Y/D 染红，早于超分、bake、inpaint。
- 相同 seed 在两个 Torch 版本下产生不同实际噪声，旧的同 seed 对比不充分。但本轮把旧版实际初始噪声注入新版后，H/Y/D 仍红；**仅替换初始噪声没有修复问题，根因尚未确定**。
- 当前没有本任务遗留的运行中推理/trace 进程。run 07 已完成，shell exit 0。ComfyUI 是用户的既有进程，最后检查队列为空；下一次 GPU 实验前重新检查。

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
4. 本交接文件有独立文档提交；使用 `git log -4 --oneline` 查看其 SHA。

推送目标是 `fork/codex/torch-mps-upgrade`，remote `fork` 为 `https://github.com/AmatsuZero/Hunyuan3D-2.1-macOS.git`。`origin` 是上游，不要向它推送。交接收尾会推送以上提交并核对远端 SHA；以实际 `git ls-remote` 为准。没有创建 PR，也没有合并。

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
- 形状路线仍按旧方案运行，本轮未变更/重跑。
- 显式 CPU 仍包括 `hy3dpaint/DifferentiableRenderer/MeshRender.py` 大索引 bake workaround（`_on_cpu = self.device != "cuda"`）、Torch/MLX `.cpu().numpy()` 桥接、mesh/UV/inpaint 等 native/预处理流程。

关键代码：

- `hy3dpaint/utils/execution_audit.py`：审计真实 forward/encode/decode，检查参数/buffer/计算输入输出设备；CPU scalar timestep 是明确例外。缺阶段/缺 MPS/CPU 计算张量会失败。每个事件立即写 JSONL，避免原生 crash 丢失全部已完成证据。
- `tools/quality_demo/audit_mps.py`：fresh-output、输入复制、导入来源验证、阶段截图/数组、耗时/内存、可选 MLX cache budget、初始噪声 replay/capture。严格模式拒绝启用 fallback。默认不改变 MLX cache policy。
- `tools/quality_demo/probe_mps_ops.py`：16 个有界 MPS 算子探针；CPU float32 oracle 使用与 half 输入相同的已舍入数据。
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
| 测试 | 最新 22/22 unittest；py_compile、git diff --check 通过 |
| GLB 结构 | run 06/07 都是 24087 UV-split 顶点、40000 面、1 PBR material、2 张内嵌 2048×2048 可解码贴图，几何/UV 有限 |
| 颜色 | 仍 FAIL，raw front multiview 已红 H/Y/D；没有用户接受新版 3D 查看效果的证据 |
| 全 GPU trace | 未通过；设备 placement/call audit 不等于完整 GPU trace |
| 内存/性能 | 仍未通过配对重复实验，不能声称 Torch 升级加速或全流程安全内存 |
| 独立 review | native reviewer 多次 429，无独立审查通过；仅作者复查 |

run 06 的 MLX steps 7.388–9.482 s，run 07 为 7.586–10.936 s；单次无明显后程断崖不足以证明稳定性能。MLX 默认 free-cache limit 曾是 48,962,627,174 bytes，约 45.6 GiB，与 Torch 争用 48 GiB 池。`--mlx-cache-limit-mib 1024` 是空闲缓存回收阈值（后续 allocation 触发回收），不是瞬时硬内存上限。不能把缓存策略收益归因于 Torch 升级。

run 06 MLX peak 8.985 GiB、sampled MPS driver 25.698 GiB、RSS 11.083 GiB；这些计数重叠，不能相加。系统 swap 24.714→30.710 GiB，内存门槛仍开放。

## 6. 颜色诊断：已排查到哪里

外部目录 `color-diagnosis/`：

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

## 7. 不要重复的失败尝试

- 旧版完整配对 benchmark 01：超过 10 分钟尚无 multiview，进程 footprint 曾 48.1G；主动 SIGTERM，exit 143。不是完整速度基线，不能用历史 476.476 s 和新版直接计算加速比。
- 整流水线 xctrace 02/03：AGXMetalG17X `setComputePipelineState` 原生 SIGSEGV，调用涉及 Torch MPS reduction/max；最小 max 四 dtype 在普通/profiler/xctrace/两者结合都通过，真实根因未定位。
- attach trace 04：raw trace 约 15 GiB，转换 RSS >12 GiB，工作负载停滞，主动终止。trace 不完整，不算 GPU 验收。
- 普通无 trace run 05：MLX step 由 8–12 s 逐渐到 26/59/79/94 s，主动终止。这证明不能仅归咎于 trace；JSONL 中已完成 reference placement 证据保留。
- 后续 trace 必须限制到有界 Torch-only 阶段，不要再直接录整个 MLX+MPS 全流水线。

## 8. 下一位 agent 的建议顺序

1. 先读本文件、`docs/torch-upgrade-mps-plan.md`、`docs/mlx-migration-roadmap.md`、验证报告；检查当前 branch/status、进程和 ComfyUI queue。保护 dirty 文件。
2. 优先做**固定实际张量的小范围颜色诊断**：保存/对比两版本 DINO 条件、VAE 条件、reference UNet 特征、第一步 denoising 输出；固定实际 timestep/latent/condition，定位差异首次显著放大的边界。若测试跨 Torch/MLX 调度，清楚记录是哪一个网络和调用。先验证小范围证据再决定是否值得跑完整链路。
3. 如仍无法归因，可在相同 fixture/条件下比较混合调度与 Torch-only 的有界单步，评估 MLX parity；不要未经内存评估直接跑全 Torch 全模型。
4. GPU 证据：针对有界 Torch-only DINO/VAE/reference/first-step/超分阶段逐个采集 Metal trace。审计发现 CPU 例外需要真实尺寸数值对照后再修改，不能凭 16 个小算子测试移除 bake workaround。
5. 内存/速度：相同 cache policy、固定实际随机张量、相同输入下串行旧/新配对与重复；监测 swap/框架 active/cache/driver，保留失败记录。未通过质量/trace/内存前不要推广候选默认环境。
6. 这些门槛完成后再按 MLX roadmap 逐步迁移。全 MLX 并不自动更快，需要端到端测量和质量对照。

当前阶段判定：P0/P1 完成；P2 placement/calls 完成但 trace 未过；P3 bounded ops 完成但真实 bake CPU 例外未解；P4 结构通过/颜色失败；P5 配对性能与内存验收未完成。

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

## 10. 已交付资产与保护证据

`/Users/daubert/Github/hy3d-mlx-eval/` 下：

- `shape_quality.glb`：151613 vertices /303234 faces，用户认可白模；SHA256 `af195a0d8dabeb396fcb368dd2740b65be588647d0520184142646c3f9c43a80`。
- `shape_quality_textured.glb`：原始上色，3 变黑；SHA256 `d5aa1ac89fd99db20b3fa2bed3747f58af1dbb67c6acd340b1c12ed62223e8c8`。
- `shape_quality_textured_red3.glb`：用户授权 Python 精确局部修红 3，几何/UV/区域外像素保持；SHA256 `de694c37a22f7b6ffc641769d9a8164538acab3fbe937e8942439b14de0836c9`。
- `shape_textured.glb`：更早历史交付，禁止覆盖；SHA256 `0ac916c04e3a3033897ab0dfac7b016187749494cbd231bfccff3a80168995b0`。
- `demo.png` 是原始参考图。五个文件最近均校验 unchanged。

初始完整快照、权重 hash、旧依赖和原始 dirty 状态在外部 `baseline/manifest.json`、`baseline/status.txt`、`baseline/runtime-overlay.patch`。不要把保护检查误认为新版质量验收。

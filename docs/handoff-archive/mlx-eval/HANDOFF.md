# Hunyuan3D-2.1 MLX 加速评估 — 交接文档

> 生成时间：2026-10-26 16:50（由上一会话 Agent 撰写）
> **2026-10-26 17:45 更新：任务 1/2/3 已全部完成**（MoE 修复、.eval() 修复、
> 上游 PR #1/#2 已提交），最新基准见 §3，修复内容见 §4。

---

## 1. 项目目标

评估第三方仓库 `noahcorona/Hunyuan3D-2.1-macOS` 用 MLX 替换 Hunyuan3D-2.1 的
DiT（去噪网络）和 VAE 体解码器后在 Apple Silicon 上的真实收益，定位性能差异根因，
并给出可行的优化/使用建议。

## 2. 环境与路径（全部已就绪，可直接使用）

| 项 | 路径 / 说明 |
|---|---|
| Python venv | `/Users/daubert/Github/.venvs/hunyuan3d-shape/bin/python` |
| 评估沙盒（主要工作区） | `/Users/daubert/Github/hy3d-mlx-eval/` |
| └ fork 拷贝+MLX 补丁 | `hy3d-mlx-eval/hy3dshape/`（PYTHONPATH 指向此处的 `hy3dshape/`） |
| 第三方仓库（完整克隆） | `/Users/daubert/Github/Hunyuan3D-2.1-macOS`（HEAD `e644522`） |
| 我们自己的 fork（7 commits） | `/Users/daubert/Github/Hunyuan3D-2.1` |
| 模型权重（14GB） | `/Users/daubert/Github/hunyuan3d-artifacts/Hunyuan3D-2.1-modelscope` |
| 硬件 | Apple Silicon，48GB 统一内存 |

**必须的环境变量**（所有实验脚本都依赖）：
```
HY3DGEN_MODELS=/Users/daubert/Github/hunyuan3d-artifacts/Hunyuan3D-2.1-modelscope
PYTHONPATH=/Users/daubert/Github/hy3d-mlx-eval/hy3dshape
# MLX 开关（在评估沙盒的 pipelines.py 中实现）：
HUNYUAN_DISABLE_MLX=1      # 全部走 PyTorch/MPS
HUNYUAN_MLX_DIT=0/1        # DiT 是否用 MLX（默认 1）
HUNYUAN_MLX_DECODER=0/1    # VAE 解码器是否用 MLX（默认 1）
```

## 3. 最终基准（修复后，重复性测试，串行独占，每配置两轮取中位）

**2026-10-26 更新**：MoE 修复 + `.eval()` 修复后（`bench_repeat_fixed.log`）：

| 配置 | R1 | R2 | 中位 | DiT/步 | geo_decoder | 顶点数 |
|---|---|---|---|---|---|---|
| A 纯 MPS | 172.5s | 170.3s | **171.4s** | ~13.4s | 23-36s | 9134/9642 |
| **B MLX DiT + MLX 解码器** | 70.3s* | 74.6s* | **72.5s** ✅ | ~5.6s | 0.7-0.9s | 9174/9772 |
| C MPS DiT + MLX 解码器 | 127.3s | 126.1s | **126.7s** | ~11.7s | 0.8s | 10192/9192 |

\* B 的两轮来自 slack=2.5 定稿后的单独复测；slack=2.0 时 bench_repeat.sh 的两轮为 78.2/69.7s，一致。

**结论（更新）：推荐配置 B（MLX DiT + MLX 解码器）**，稳定快于纯 MPS 2.4x。
相比修复前（655.7s/165.8s，4.4x 方差）方差降到 ~6%。A 配置也从 249.9s →
171.4s（`.eval()` 修复把 MoE 从 train 分支（repeat_interleave+index_put）
切到 eval 分支的附带收益）。

### MoE 隔离实验（修复后，mlx_dit_scaling_clean.json 已覆盖为新数据）

| 路径 | batch 1 | batch 2 | batch 3 |
|---|---|---|---|
| stock（固定容量路由） | 1721ms | 3436ms | **6351ms**（修复前 60612ms，9.5x） |
| dense_moe（对照） | 1835ms | 2442ms | 3603ms |

stock 扩展已近线性（原为 29x 劣化）。

## 4. 根因与修复状态

### 4.1 MLX DiT 的 MoE 路由缺陷 — ✅ 已修复（见上游 PR #2）

原缺陷（`mlx_dit.py` MoEBlock）：`counts = [int(c) for c in count_arr.tolist()]`
每层强制 GPU→CPU 同步 + 8 个专家各跑动态 token 数的矩阵乘。

**修复方案：固定容量 (expert, rank) 槽表 + `put_along_axis` 分发**：
- 容量 = `ceil(B_S * top_k / E * slack)`，默认 slack=2.5（`HUNYUAN_MLX_MOE_SLACK` 可调）；
- 槽位 rank 用纯数组算子（one-hot cumsum + take_along_axis）计算，全程无 `.item()/.tolist()`；
- 溢出槽位（rank≥容量）回退到 shared expert；`HUNYUAN_MLX_MOE_DEBUG=1` 会报告溢出；
- **slack 的确定过程**：真实权重路由高度偏斜，单专家峰值占 27% 槽位（均匀 12.5%）。
  slack=1.25 大量丢 token（parity rel 0.24）；slack=2.0 在真实管线数据上有一层
  丢 113/16388 槽（被 DEBUG 抓到，顶点数 8322 异常偏低）；slack=2.5 真实运行零溢出，
  parity rel ≤ 5.3e-4（fp16 舍入级）。stress 测试（100% 路由到单专家）优雅降级不崩溃。
- 曾经考虑并否决的方案：`mx.gather_mm` 按行选权重（此 MLX 版本实现极慢，1543ms/block，
  每行都在 gather 大权重矩阵），见 `moe_microbench.py`。

**验证结果**：
- `mlx_dit_scaling_clean.py`：stock batch3 60612→6351ms，扩展近线性；
- 端到端 B 配置：655.7/165.8s → 70.3/74.6s，方差 4.4x→~6%；
- `moe_fix_validate.json`：parity/overflow/stress 数据。

### 4.2 上游真实缺陷 — .eval() 缺失 — ✅ 已修复（见上游 PR #1）

- `pipelines.py` `__init__` 补 `self.model.eval()`；
- `moe_layers.py` `moe_infer` 的 `scatter_reduce_(reduce='sum')`（MPS 不实现
  `aten::scatter_reduce.two_out`）替换为语义等价的 `index_add_`（MPS/CUDA/CPU 通吃）；
- **数值证据**（`eval_branch_diff.py` / `eval_branch_diff.json`）：MPS fp32 下
  train 分支 vs eval 分支（含回退）单次 DiT 前向输出**逐位一致**
  （max_abs_diff=0.0，cosine=1.0）——修复不改变数值，只是让预期代码路径真正执行；
- 附带收益：A 配置（纯 MPS）249.9s → 171.4s（train 分支的 repeat_interleave+index_put 被替换）。

### 4.3 上游反馈 — ✅ 已提交（issues 被上游禁用，故以 PR 携带完整报告）

- PR #1 <https://github.com/noahcorona/Hunyuan3D-2.1-macOS/pull/1>：
  `.eval()` 缺失 + MPS `scatter_reduce_` 不兼容（正文含完整 bug 报告与数值证据）；
- PR #2 <https://github.com/noahcorona/Hunyuan3D-2.1-macOS/pull/2>：
  MLX MoE 静态化修复（正文含 before/after 数据表）。
- fork：`AmatsuZero/Hunyuan3D-2.1-macOS`；分支
  `fix/dit-eval-mode-moe-infer`（0144bc0）、`fix/mlx-moe-static-capacity`（9dd5aa1）。
- PR 正文存档：`pr_body_1.md` / `pr_body_2.md`。

### 4.4 MLX 数值一致性（上一轮已完成验证 ✅）

- DiT 单步 fp16：MPS vs MLX 相关系数 **0.9996**（`dit_parity.json`）；
- MLX 解码器输出与 PyTorch 一致（`bench_dit.json`）。

## 5. 已知坑（接手前必读）

1. **并发测量污染**：这台机器上同时跑两个 GPU 任务，数字会失真最多 10x。
   所有 benchmark 必须**串行独占**跑，配置之间 `sleep 20` 释放统一内存。
2. **`nn.Module.__call__` 绕过实例 monkeypatch**：给 `pipe.model.__call__`
   打计时补丁无效（Python 特殊方法查找绕过实例属性）。用 forward hook 或
   patch `.forward`。`profile_stages2.py` 已正确处理。
3. **MPS fp32 vs MLX fp16**：管线 MPS 跑 fp32（DiT ~10GB），MLX 跑 fp16；
   对比时要意识到这是精度不同但都是"管线真实模式"。
4. **配置 D（MLX DiT + MPS 解码器）会 MPS OOM**（24.8GB+37.8GB 超 48GB），勿跑。
5. **网络对 GitHub 大传输不稳定**：smart-HTTP 克隆反复断连。已用 promisor
   部分克隆 + 逐 blob 拉取解决（`fetch_blobs.py`，带哈希校验，443 文件全对）。
   剩 25 个缺失全是 `mini_trainset/preprocessed/` 训练数据二进制，不影响运行。
6. **沙箱安全删除**：`rm` 会被转成移入废纸篓；对仓库目录勿用 `rm -rf`，用
   `mv` 到时间戳备份目录。
7. `mlx_dit_scaling.json` / `mlx_dit_scaling.json`（非 clean 版）和
   `profile_mlx.json`（991s）、早期 `profile_stages*.py` 的 DiT=0 数据
   都是**污染/无效数据**，仅作反面教材保留，勿引用。

## 6. 关键文件索引

```
hy3d-mlx-eval/
├── REPORT.md                  # 详细报告（早期数据部分已被本文件 §3 修正）
├── HANDOFF.md                 # 本文件
├── profile_stages2.py         # 端到端分段计时（正确版，forward hook）
├── bench_repeat.sh            # 串行重复性测试（A/B/C × 2 轮）
├── bench_repeat_fixed.log     # 修复后的重复性测试原始日志（2026-10-26）
├── bench_e2e.sh               # 串行四配置对比（含会 OOM 的 D）
├── mlx_dit_scaling_clean.py   # MoE 归因隔离实验（干净版，现为修复后数据）
├── moe_fix_validate.py/.json  # MoE 修复验证：新旧对拍 parity/overflow/stress
├── moe_microbench.py          # legacy/capacity/gather_mm 三策略微基准
├── eval_branch_diff.py/.json  # MPS 上 MoE train vs eval 分支数值对比（逐位一致）
├── pr_body_1.md / pr_body_2.md  # 上游 PR 正文存档
├── parity_dit.py / bench_dit2.py  # 数值一致性 / 单步性能
├── fetch_blobs.py             # 逐 blob 网络恢复克隆工具
├── profile_{mps_only,mlx_both,mlx_decoder_only}.json  # 最终基准数据
├── dit_parity.json / dit_bench.json / mlx_dit_scaling_clean.json
└── hy3dshape/hy3dshape/       # fork 拷贝 + MLX 补丁（PYTHONPATH 目标）
    ├── pipelines.py           # :249-279 MLX 开关逻辑（_setup_mlx）
    ├── mlx_dit.py             # :217-258 MoEBlock（缺陷所在）
    ├── mlx_dit_utils.py       # 权重加载 / DiT 分发安装
    ├── mlx_vae_decoder.py     # MLX 解码器（32x 加速的实现）
    └── mlx_vae_decoder_utils.py
```

## 7. 待办（2026-10-26 更新）

1. ~~**修复 MLX MoE 路由**~~ ✅ 已完成（§4.1，验收标准全部达成）。
2. ~~**修复 `.eval()` 缺失**~~ ✅ 已完成（§4.2）。
3. ~~**整理上游反馈**~~ ✅ 已提交 PR #1/#2（§4.3）。后续：跟进维护者反馈，
   如需修改在 fork 分支上继续。
4. **可选（未做）**：MLX DiT + `mx.compile` 预编译；batch=1 场景下 MLX DiT
   是否有稳定优势；`HUNYUAN_MLX_MOE_SLACK` 灵敏度扫描（slack 越大专家计算
   越多，2.5→3.0 约 +20% 专家 FLOPs）。
5. **注意**：`mlx_dit_scaling_clean.json` 与 `moe_fix_validate.json` 现为
   修复后的有效数据；`mlx_dit_scaling.json`（非 clean 版）和 `profile_mlx.json`
   仍是污染数据，勿引用。REPORT.md 中早期数据已被本文件 §3 取代。

## 8. PyTorch 2.5.1 不能就地升级，MLX 也不能取代它（2026-10-26 18:28）

**问题。** 形状管线跑在 venv 的 PyTorch **2.5.1** 上。这个版本到 2026-10 已经大约两年。给纹理阶段装 `basicsr` / `realesrgan` / `pymeshlab` 时，它们的依赖是 `torch>=1.7`、没有上限，`uv pip install`（不加 `--no-deps`）会解析到 **torch 2.14.1** 并替换当前环境。已主动停掉那两次安装。当前 `import torch` 仍是 2.5.1。

**为什么先钉住 2.5.1。**

- 形状基准、`.eval()` / `scatter_reduce_` 的 MPS 行为、光栅化器 `custom_rasterizer_kernel.so` 的 rpath，都对着这份 torch 的 `libc10`。
- 换 2.14 之后，MPS 算子覆盖、已编译扩展、已记录的耗时都要重测，不能当成无成本升级。

**MLX 不能取代这份 PyTorch。** 仓库里的 MLX 是嵌在 PyTorch 管线里的热循环替换，不是另一个运行时。

| 阶段 | 谁在算 | 依据 |
|---|---|---|
| 形状 DiT | MLX（已修 MoE） | `hy3dshape/mlx_dit.py`，外层仍是 `MLXDispatchDiT(torch 包装)` |
| 形状 VAE 体解码 | MLX | `mlx_vae_decoder.py` |
| 纹理 inner UNet 去噪第 2–15 步 | MLX（上游自称 SDPA 约 3–5x，**本环境尚未实测**） | `hy3dpaint/mlx_utils.py` `MLXDispatchUNet.forward` |
| 纹理 UNet 第 1 步、reference write（`mode` 含 `w`） | PyTorch MPS | 同上，用来填 outer UNet 的 cache |
| 纹理 VAE / CLIP / DINO / RealESRGAN | PyTorch | `multiview_utils.py`、`image_super_utils.py` |
| 网格减面、UV、光栅化、贴图 inpaint | C++/CPU（pymeshlab、xatlas、自编译扩展） | 与 MLX 无关 |

权重加载也走 `torch.load`（`mlx_utils.py` 的 `load_weights_from_pytorch`）。所以即使用上 MLX UNet，纹理管线仍然需要这份 PyTorch。MLX 是性能手段，不是躲开 2.5.1 的办法。

**当前约定。** 纹理依赖用 `uv pip install --no-deps` 安装，不升级 torch。PyTorch 升级单独开一轮：新 venv、重编光栅化器、重跑形状 A/B 基准，确认 MPS 上 `scatter_reduce_` / `index_add_` 行为没变，再决定是否切换。

## 9. 快速复现

```bash
# 单配置端到端分段计时（约 2.5-11 分钟/次）
export HY3DGEN_MODELS=/Users/daubert/Github/hunyuan3d-artifacts/Hunyuan3D-2.1-modelscope
export PYTHONPATH=/Users/daubert/Github/hy3d-mlx-eval/hy3dshape
HUNYUAN_MLX_DIT=0 HUNYUAN_MLX_DECODER=1 \
  /Users/daubert/Github/.venvs/hunyuan3d-shape/bin/python \
  /Users/daubert/Github/hy3d-mlx-eval/profile_stages2.py

# 全套串行重复性测试（约 35 分钟）
bash /Users/daubert/Github/hy3d-mlx-eval/bench_repeat.sh

# MoE 归因隔离实验（约 15 分钟）
/Users/daubert/Github/.venvs/hunyuan3d-shape/bin/python \
  /Users/daubert/Github/hy3d-mlx-eval/mlx_dit_scaling_clean.py
```

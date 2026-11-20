# 质量白模 + 上色 — 交接文档（2026-10-26 22:50）

> **最新状态（2026-10-26 23:12，Asia/Shanghai）：上色已完成并通过文件验收。**
> `shape_quality_textured.glb` 已生成，3,163,272 bytes；上色 444.4s，整个进程 476.476s。
> 已用 PyMeshLab 替换 Open3D 减面路径。下文 §1–§9 是上次交接时的历史状态，最终结果见 §10；无需重跑。

给接手 Agent：优先读完 **§1 目标**、**§4 阻塞点与推荐修复**、**§6 下一步**。不要重跑形体；用户已确认可以上色。

---

## 1. 目标与验收

| 项 | 说明 |
|---|---|
| 用户目标 | 用 Gradio 档形体参数生成白模，再贴图；整体观感已明显好于低精度 benchmark 网格 |
| 用户已确认 | 2026-10-26 看过 `shape_quality.glb` 与 `demo.png`：**除牌子上的字被建成 3D 浮雕外，其余形体满意，同意先上色** |
| 待交付 | `/Users/daubert/Github/hy3d-mlx-eval/shape_quality_textured.glb`（及同名 `.obj`），**不要覆盖** `shape_textured.glb` |
| 未要求 | 再提 PR、再 benchmark、升级 PyTorch、Gradio 8 视角 / 768 贴图 |

牌子字变 3D：参考图里 “HY3D” 是平面印刷，形状模型在 octree 256 下把字当几何重建了；上色不会削平，只会着色。若以后要更平的牌子，需降 octree 或后处理，**不在当前任务**。

---

## 2. 已完成

### 2.1 质量白模（成功）

- 输入图：`/Users/daubert/Github/hy3d-mlx-eval/demo.png`
- 输出：`/Users/daubert/Github/hy3d-mlx-eval/shape_quality.glb`（5.5MB）
- 参数：MLX DiT + MLX VAE 解码，`num_inference_steps=30`，`octree_resolution=256`，`guidance_scale=5.0`，`num_chunks=8000`，seed `1234`
- 统计：**151613 顶点，303234 面**；256 一档成功，未退到 192/128
- 耗时：约 262s（仅形体子进程）
- 脚本：`/Users/daubert/Github/Hunyuan3D-2.1-macOS/run_shape_quality.py`（父进程按 256→192→128 起子进程）

### 2.2 代码与 Git（macOS 仓库）

- 仓库：`/Users/daubert/Github/Hunyuan3D-2.1-macOS`
- 分支：**`quality-shape-demo`**（从 MoE 修复提交 `9dd5aa1` 分出）
- 本任务相关提交（仅这 3 个，**勿把下面 dirty 文件混进 commit**）：

| SHA | 说明 |
|---|---|
| `7e369cc` | `quality_shape.py` + `tests/test_quality_shape.py` |
| `7787bff` | `run_shape_quality.py` |
| `b93b603` | `run_paint.py`（`--mesh` / `--output-obj`） |

- 规格与计划（只读参考）：
  - `hy3d-mlx-eval/docs/superpowers/specs/2026-10-26-quality-shape-design.md`
  - `hy3d-mlx-eval/docs/superpowers/plans/2026-10-26-quality-shape-implementation.md`
- SDD 进度：`Hunyuan3D-2.1-macOS/.superpowers/sdd/progress.md`（Task 1–3 complete；Task 4 未做完）

### 2.3 低精度历史结果（保留）

- 形体 benchmark：`shape_profile_mlx_both.glb`（~9772 顶点，10 步 / octree 64）
- 已上色低精度：`shape_textured.glb`（约 19 分钟那轮）

### 2.4 贴图管线（曾跑通，配置不变）

- 6 视角、512、remesh、RealESRGAN、inpaint；UNet 第 1 步 PyTorch MPS + 小 SDPA 分块，第 2–15 步 MLX
- 必须：`PYTORCH_ENABLE_MPS_FALLBACK=1`（DINO bicubic MPS 缺算子）
- 参考成功命令见 `hy3d-mlx-eval/HANDOFF.md` §8 与旧 `run_paint.py` 路径；**新命令见 §6**

---

## 3. 环境与路径

| 项 | 值 |
|---|---|
| Python | `/Users/daubert/Github/.venvs/hunyuan3d-shape/bin/python`（3.11，**torch 2.5.1**，勿升级） |
| 形状 MLX 补丁 | `PYTHONPATH=/Users/daubert/Github/hy3d-mlx-eval/hy3dshape` |
| 权重 | `HY3DGEN_MODELS=/Users/daubert/Github/hunyuan3d-artifacts/Hunyuan3D-2.1-modelscope` |
| DINO | `/Users/daubert/Github/hunyuan3d-artifacts/dinov2-giant` |
| RealESRGAN | `/Users/daubert/Github/Hunyuan3D-2.1-macOS/ckpt/RealESRGAN_x4plus.pth` |
| 硬件 | Apple Silicon 48GB 统一内存；**同一时刻只跑一个 GPU 任务**；禁止 MLX DiT + MPS 解码 |

安装依赖仍遵守 `hy3d-mlx-eval/HANDOFF.md` §8：`uv pip install --no-deps`（必要时 `--no-build-isolation`），避免解析到 torch 2.14。

---

## 4. 当前阻塞：上色在 remesh 阶段失败

### 4.1 现象

对 `shape_quality.glb` 跑 macOS 版 `run_paint.py` 时，模型加载成功（~18s），随后在 `remesh_mesh` → `mesh_simplify_trimesh` 崩溃：

- 面数 303234 > 默认 `target_count=40000`
- `trimesh.simplify_quadric_decimation(target_count)` 走 **Open3D**
- 环境最初无 `open3d`；后续即使用 aria2 装上 wheel，**`import open3d` 仍会拉一长串可选依赖**（plotly、dash、narwhals…），最后在 `open3d.ml` 链路上缺 **`sklearn`**，完整 import 仍失败

**上色进程当前不在运行。** `shape_quality_textured.glb` **尚未生成**。

### 4.2 根因（代码位置）

`Hunyuan3D-2.1-macOS/hy3dpaint/utils/simplify_mesh_utils.py`：

1. 已用 pymeshlab 读 glb 并导出中间 obj  
2. 再用 trimesh 加载；若 `face_num > target_count`，调用 `simplify_quadric_decimation` → **依赖 Open3D**

贴图管线**不需要** Open3D 可视化/ML 模块，只需要**减面**。

### 4.3 推荐修复（优先，小 diff）

**用已安装的 pymeshlab 做 quadric 减面**，与上面验证一致（接手 Agent 已在本机验证过）：

```python
# 在 mesh_simplify_trimesh 中，当 face_num > target_count 时：
ms = pymeshlab.MeshSet()
ms.load_new_mesh(inputpath, load_in_a_single_layer=True)  # 或沿用已有 ms 流程
ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target_count))
ms.save_current_mesh(outputpath)
# 可选：再 trimesh.load 校验面数
```

验证命令（已通过）：

```bash
/Users/daubert/Github/.venvs/hunyuan3d-shape/bin/python - <<'PY'
import pymeshlab
ms = pymeshlab.MeshSet()
ms.load_new_mesh("/Users/daubert/Github/hy3d-mlx-eval/shape_quality.glb", load_in_a_single_layer=True)
print("before", ms.current_mesh().face_number())
ms.meshing_decimation_quadric_edge_collapse(targetfacenum=40000)
print("after", ms.current_mesh().face_number())
PY
# 输出：before 303234 → after 40000
```

改完后**不必**为了上色而继续补 Open3D/sklearn 全家桶。若坚持 Open3D 路径，需 `uv pip install scikit-learn` 等并反复试 `import open3d`，成本高且与 PyTorch 升级风险无关但浪费。

### 4.4 Open3D / aria2 侧记（可选清理）

- aria2 RPC：`http://127.0.0.1:6800`，token `fa1bbb5f3150536259ffab60ac5a4a6d`，容器内 `dir=/downloads` 对应主机  
  `/Users/daubert/Github/CrawlerBot/aria2/downloads`
- 已下载：`open3d-0.18.0-...whl`（41MB）、`plotly`、`dash-2.18.2`、`narwhals` 等小 wheel；venv 里 **`open3d==0.18.0` 已装但完整 import 仍失败**（缺 sklearn）
- 若采用 §4.3，Open3D 对上色**非必需**；是否卸载 open3d/plotly/dash 由接手 Agent 自行决定（不影响 pymeshlab 减面）

---

## 5. macOS 仓库 dirty 状态（勿提交）

`git status` 上还有与本次无关的本地修改（partial clone 缺 blob、光栅化器、attn_processor、mlx_utils 等）。**只 commit 与 remesh 修复或明确任务相关的文件**，不要 `git add -A`。

评估目录 `hy3d-mlx-eval` **不是 git 仓库**；产物 glb/obj 放在该目录即可。

---

## 6. 接手 Agent 建议步骤（按序）

1. **修** `hy3dpaint/utils/simplify_mesh_utils.py`（§4.3），在 `quality-shape-demo` 上单独 commit（可选）。
2. 确认无其它 Hunyuan Python GPU 进程：`pgrep -fl run_paint` 应为空。
3. **跑上色**（工作目录 macOS 仓库根）：

```bash
cd /Users/daubert/Github/Hunyuan3D-2.1-macOS
PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
  /Users/daubert/Github/.venvs/hunyuan3d-shape/bin/python -u run_paint.py \
  --mesh /Users/daubert/Github/hy3d-mlx-eval/shape_quality.glb \
  --output-obj /Users/daubert/Github/hy3d-mlx-eval/shape_quality_textured.obj
```

4. 预期：日志出现 `Models Loaded.`，随后可能长时间 silent（第 1 步 SDPA 分块很慢）；**上次低精度贴图约 1117s**，本次 remesh 后约 4 万面，整体仍可能 **20–30+ 分钟**，勿并行第二个 GPU 任务。
5. 成功后检查：

```bash
ls -lh /Users/daubert/Github/hy3d-mlx-eval/shape_quality_textured.glb
# GLB 内应有 materials / textures / images（可参考 plan Task 4 的 Python 片段）
```

6. 把结果路径与耗时告诉用户；若 OOM，参考旧 HANDOFF：仅调 SDPA 分块或延后 MLX 权重加载，**不要**同时跑 shape benchmark。

---

## 7. 与主 HANDOFF 的关系

- MLX MoE / 形状 benchmark / `HANDOFF.md` §3–§9 仍然有效；**不要引用**污染过的 `profile_mlx.json`。
- 本文件只覆盖 **quality-shape-demo 分支 + 待完成的上色**；形状加速任务已闭环，无需再跑 `run_shape_quality.py` 除非用户要新参数。

---

## 8. 快速文件索引

| 路径 | 用途 |
|---|---|
| `hy3d-mlx-eval/shape_quality.glb` | 待上色白模（用户已批） |
| `hy3d-mlx-eval/shape_quality_textured.*` | 目标输出（尚未生成） |
| `Hunyuan3D-2.1-macOS/run_paint.py` | 带 CLI 的贴图入口 |
| `Hunyuan3D-2.1-macOS/run_shape_quality.py` | 质量白模（已完成） |
| `Hunyuan3D-2.1-macOS/quality_shape.py` | 八叉树退档策略 |
| `hy3d-mlx-eval/hy3dshape/` | MLX 形状补丁（PYTHONPATH） |
| `hy3d-mlx-eval/HANDOFF.md` | 原 MLX 评估与 PyTorch 2.5.1 约定 |

---

## 9. 用户原话摘要

- 「效果看这比之前好多了，不过举的牌子的字怎么也变 3D 的了」→ 已解释；用户接受先上色。
- 「如果不好下载，还可以尝试 aria2」→ Open3D/plotly 等已用 aria2；**减面改 pymeshlab 后不必再依赖 Open3D 下载**。

---

## 10. 完成结果（2026-10-26 23:12，Asia/Shanghai）

- 分支保持 `quality-shape-demo`。修复 `hy3dpaint/utils/simplify_mesh_utils.py`：在现有 PyMeshLab MeshSet 中，当面数超过目标时调用 `meshing_decimation_quadric_edge_collapse(targetfacenum=int(target_count))`；保留后续 OBJ/GLB 导出。
- 新增 `tests/test_simplify_mesh.py`，覆盖 Open3D 不可用时的真实减面、输入文件不变，以及等于/低于目标面数时的 OBJ/GLB 输出。
- 验证：`python -m unittest discover -s tests -v` 共 6 项通过；`git diff --check` 通过；真实白模 CPU 减面 `303234 → 40000` 面。
- 按 §6 原命令执行，退出码 0；模型加载 22.6s，上色 444.4s，进程总耗时 476.476s。
- 交付：`/Users/daubert/Github/hy3d-mlx-eval/shape_quality_textured.glb`，另有同名 OBJ、MTL、JPG 及 PBR 贴图配套文件。
- GLB 验收：24,087 顶点（含 UV 接缝拆点）、40,000 三角面、1 个材质、2 张内嵌 2048×2048 贴图；baseColor / metallicRoughness 引用有效，图片可解码，UV 数量与顶点数一致且有限。
- `shape_quality.glb` 与 `shape_textured.glb` SHA-256 均与运行前一致；未重跑形体，PyTorch 仍为 2.5.1，未安装/升级依赖。
- 本次源码/测试改动尚未提交或暂存；无关 dirty 文件保留原样。
- 独立子代理审核因服务 429 未完成；已执行本地代码检查、回归测试和实际上色验收。未另做 3D 查看器视觉验收。
- 日志：`quality-texture-20261026T150444Z.log`
- 进程与计时：`quality-texture-20261026T150444Z.json`
- 文件验收报告：`quality-texture-verification-20261026T150444Z.json`
- 运行前保护文件指纹：`quality-texture-inputs-20261026T150444Z.json`


## 11. 数字 3 局部修色（2026-10-27）

- 用户反馈：原图红色的数字 `3` 在贴图中变黑；实际检查基础颜色贴图确认，非仅灯光/材质显示问题。
- 用户选择保留现有效果，局部修色并另存；随后明确授权使用 Python 精确修色。
- 交付：`/Users/daubert/Github/hy3d-mlx-eval/shape_quality_textured_red3.glb`（4,772,724 bytes）。
- 仅在原始 GLB 内嵌 2048×2048 基础颜色图中修改数字 `3` 的连通像素，参考色来自 demo.png 的红色像素中值 RGB `(233, 17, 29)`；保留边缘抗锯齿。
- 共修改 23,291 像素；遮罩外解码像素完全一致。新颜色贴图以无损 PNG 内嵌；所有原始二进制数据保留，几何、UV、材质参数和金属度/粗糙度贴图字节不变。原 shape_quality_textured.glb SHA-256 未改变。
- 脚本：`fix_red3_texture.py`；验收：`shape_quality_textured_red3_verification.json`；左右前后对比：`shape_quality_textured_red3_comparison.png`。
- 先前 imagegen 尝试改变了贴图分辨率及部分区域外细节，未采用到 GLB；最终产物完全来自经授权的 Python 局部修改。
- 本轮无模型推理/形体重跑，CPU 修色打包及验收约 0.66s。已检查贴图对比；尚无新的 3D 查看器视觉验收。

# P3 交接：真实尺寸 bake 的 MPS 路径（2026-10-27）

这是 P3 阶段交接；总交接仍是 [`HANDOFF-torch-mps-upgrade.md`](HANDOFF-torch-mps-upgrade.md)。

## 结论

P3 的真实尺寸 back-sample bake 验证和最小生产修改已完成。Torch 2.14.1 在 MPS 上可以承载本次 4096×4096、11,175,499 个 UV position 的 gather；旧的“所有非 CUDA 都转 CPU”分支已移除。`torch.isin` 在真实尺寸 MPS 上会触发约 348 GB 的 MPSGraph fused allocation，因此改成等价的布尔 mask 索引。

这只关闭了 back-sample 的旧 CPU workaround。mesh/UV/native 操作、`.cpu().numpy()` 桥接和 inpaint 等显式 CPU/native 边界仍存在；不能据此宣称整个 pipeline 全 GPU，也不能把 P3 结果当成性能提升证明。

## 代码变更

文件：`hy3dpaint/DifferentiableRenderer/MeshRender.py`

- `_on_cpu = self.device == "cpu"`：只有显式 CPU renderer 走 CPU 路径，MPS 保持在 active device。
- `torch.isin(valid_idx, inner_valid_idx)` 替换为 `valid_idx[inner_mask[valid_idx]]`，保持有效像素集合顺序并避开真实尺寸 MPSGraph 超大临时分配。
- 未修改根目录受保护的 `torchvision_fix.py`。

## 证据

外部实验目录：`/Users/daubert/Github/hy3d-upgrade-evidence/20261027/`。小型可复核文件 `p3-bake-mps-01/{summary.json,audit.json,events.jsonl,timings.jsonl}` 已写回该目录；完整严格审计原始输出保留在 `/private/tmp/hy3d-p3-bake-audit-01/`。大型 `.npy`、GLB 和 raw 输出不进入 Git。

### 最小/真实尺寸数值对照

- 12M 元素 MPS advanced indexing 成功。
- 直接使用 `torch.isin` 的完整链在真实尺寸触发：`Failed to allocate MTLBuffer of 348204000000 B for oversized fused alloc.`
- 等价的 `inner_mask[valid_idx]` 链成功；小输入 CPU oracle 与 MPS 的 valid indices、采样 RGB 完全一致。
- 真实 run09 mesh：UV position `11,175,499`；纹理 `4096×4096`；CPU valid count `3,909,758`。CPU/MPS 的 valid indices、sampled RGB/weight/boundary、final texture/cos/boundary 均一致，最大绝对差为 0；MPS 诊断路径约 `0.033 s`，CPU 约 `0.705 s`。

### 生产路径烟测

- Torch `2.14.1`，`PYTORCH_ENABLE_MPS_FALLBACK=0`，device `mps`。
- 输出：texture `[4096,4096,3]`、cos `[4096,4096,1]`、boundary `[16777216]`；全部 finite；未写 GLB。

### 完整严格 audit

目录：`/private/tmp/hy3d-p3-bake-audit-01`，fresh output，未覆盖已交付资产。

| 项目 | 结果 |
| --- | ---: |
| exit / fallback | 0 / 0 |
| steps / bake calls | 15 / 2 |
| load / inference / total | 18.383 / 115.243 / 133.640 s |
| MLX cache threshold | 1 GiB（空闲缓存回收阈值，不是硬上限） |
| MLX peak / active | 5.64 / 2.11 GiB |

与 `candidate-audit-09-up0-loaded` 对照：multiview、enhance、inpaint、OBJ/MTL/JPG/MR 纹理逐字节相同；bake albedo/MR 中间数组最大绝对差分别约 `0.000448`/`0.000638`，各只有 19 个值超过 `1e-5`，没有值超过 `1e-3`；boundary 数组完全一致。两个 GLB 的容器字节 hash 不同，但 vertices、faces、UV、base-color texture、metallic-roughness texture 和 material 结构一致。

## 验证与边界

- 正常 Metal 权限下 `python -m unittest discover -s tests -v`：30/30 passed；`py_compile`、`git diff --check` 通过。受限沙箱无法访问 Metal，MLX 测试收集会在 nanobind 导入处 abort；正常权限复测通过，因此不把沙箱结果视为代码失败。
- P2 named Torch-only Metal trace 已通过；P3 bake 本身尚未做独立 Metal trace，不把这次数值烟测当作 trace 证据。
- 需要继续的门槛：旧/新版完整配对性能与 48 GiB 内存验收、用户设备/独立 3D 查看、默认环境切换评估。
- 下一步是 P5：相同输入、固定实际初始噪声、相同 1 GiB MLX cache policy，旧版与候选版串行冷/热运行并记录 wall time、RSS、MPS/MLX counters、swap；完成匹配重复前不要宣称 Torch 升级带来性能提升。

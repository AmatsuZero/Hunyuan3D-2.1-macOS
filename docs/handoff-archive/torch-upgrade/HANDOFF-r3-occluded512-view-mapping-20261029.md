# R3 occluded512 图像与 latent 视角映射（2026-10-29）

本阶段只读取既有 Torch reference 与 MLX Winograd-disabled opt-in 输出；没有启动
GPU、修改模型代码或覆盖 GLB。唯一权威迁移交接仍为
[`HANDOFF-torch-mps-upgrade.md`](HANDOFF-torch-mps-upgrade.md)。工作区
`/Users/daubert/Github/hy3d-torch-upgrade`，分支 `codex/torch-mps-upgrade`。

## 证据与映射

MLX 输入是 `20261029/r3-reference-winograd-optin-20261029/occluded512-replay/`，
Torch reference 是 `20261028/r3-reference-fp32-resident-01/occluded512/torch-reference/`，
均在 `/Users/daubert/Github/hy3d-upgrade-evidence/` 下。两组各有 12 张
`generated_*.png` 与 15 步 resident `noise_pred`、CFG 输出、latent trace。

运行模型源文件的 `pbr_setting=["albedo","mr"]`，pipeline 的
`(b n_pbr n)` 展平顺序以及 `run_mlx_denoise` 的三分支 CFG 展开共同确定：

- 图像和 scheduler/latent：`flat_index = pbr_index * 6 + view_index`；
  `0..5` 是 albedo，`6..11` 是 MR。
- 36 项 noise：`flat_index = branch_index * 12 + image_index`，三分支依次是
  unconditional、reference、full。全局 step 10–14 的最大 noise 差索引
  `[22/10/34,...]` 因而主要对应 **image 10（MR、view 4）**，并非两个超阈值
  输出；上一阶段“空间坐标接近”不能当成 image 5/11 的根因证据。
- view 5 相机是 `(elev=-90, azim=180, weight=0.05)`；两张超阈值图
  `generated_5`（albedo）和 `generated_11`（MR）都在该视角。

逐图/最终 latent 明细：
`/Users/daubert/Github/hy3d-upgrade-evidence/20261029/r3-reference-winograd-optin-20261029/occluded512-image-latent-mapping.json`，
SHA256 `b4c65694c5674ed5ac8f993cecbb04cd4ce6ab8aef6eff9cf4ffa4a6ad9198b9`。
分支、CFG 和 latent 的 step 0、10–14 明细：
`.../occluded512-branch-step-localization.json`，SHA256
`d7caf7ceade1af91983f6e8a450e1710da7c4f2a5649f4707f264f1e7c789e9a`。
两份 JSON 均先写 `/private/tmp` 再复制至上述证据目录，`cmp` 与 SHA256 一致。

| image | 材质/视角 | 图像最大差 | 超过 32 的像素 | 超阈值 bbox `(x0,y0,x1,y1)` | 最终 latent 最大差 `(ch,y,x)` |
| ---: | --- | ---: | ---: | --- | --- |
| 5 | albedo / view 5 | 43 | 17 | `(206,309,211,333)` | `0.19287` at `(1,39,25)` |
| 10 | MR / view 4 | 9 | 0 | 无 | `0.71191` at `(0,43,35)` |
| 11 | MR / view 5 | 41 | 2 | `(231,334,232,334)` | `0.25391` at `(1,44,31)` |

以 8 倍尺寸做**粗粒度**位置映射时，image 5 的超阈值区域落在 latent
`x=25..26,y=38..41`，该窗口最大差 `0.19287`；image 11 落在
`x=28..29,y=41`，窗口最大差 `0.24902`。VAE 解码有邻域感受野，该除 8
映射不是像素到 latent 的精确逆变换。12 张图的粗粒度 image/latent 差值
Pearson 系数为 `0.25–0.72`，只支持局部相关，不能证明算子因果。
尤其 image 10 的最终 latent 最大差高于 image 5/11，却只有 `9/255`
图像最大差，因此不能单凭全局 latent/noise 最大值预测图像门槛。

## 下一阶段

1. 以 view 5 的 albedo/MR 为首要对象，沿 step 0→14 的同一 image index
   对比 noise 三分支、guided CFG 与 scheduler 后 latent 的局部差，确认
   差异在哪个边界明显放大；保留 image 10/view 4 作为反例对照。
2. 若离线 trace 不能再收窄，才做单 GPU 的局部中间层 capture，优先
   复核 view 5 的 generation 输入/输出与 reference cache 消费边界，
   然后再判断是否需要改变 Winograd fallback 范围。不得仅按全局最大
   noise 坐标选择卷积。
3. occluded512 `43/255 > 32/255` 探索性门槛仍失败，默认 Torch reference、
   Winograd 和 MLX dispatch 保持不变。GPU 前检查 ComfyUI queue 与其他
   推理进程；一次一 GPU、1 GiB MLX idle cache、fresh output。保护
   原始 checkout、旧 venv、已交付 GLB 和根目录 dirty `torchvision_fix.py`。

上一自动新会话 `01a184cf-4ed3-7063-9a1f-b0947aab57ee` 在首条更新后因
`429 Too Many Requests` 失败，未执行本阶段分析或 GPU；因此当前会话继续离线工作。

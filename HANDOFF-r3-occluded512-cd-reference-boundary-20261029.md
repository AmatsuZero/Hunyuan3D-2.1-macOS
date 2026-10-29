# R3 occluded512 C/D reference 残余边界（2026-10-29）

本阶段只读取已归档的四路矩阵 C/D trace，在相同 Winograd-disabled policy 下
分析 MLX fp32 reference 与 Torch reference 的残余 `42/255` 图像差。没有启动 GPU、
修改模型代码或默认路径。

## 固定对照

- C：MLX fp32 reference + `HUNYUAN_MLX_DISABLE_WINOGRAD=1`；D：Torch
  reference + 同一 disabled policy。两路皆为 Torch 2.14.1、MLX fp16
  generation、15 步、同一 initial noise/fixtures/DINO bundle、1 GiB cache。
- 初始 latents、inner sample、DINO/text/reference 输入与 reference timestep
  逐元素 exact；完整 15 类输入合同已在 §19 归档。
- C：`20261029/r3-reference-winograd-optin-20261029/occluded512-replay/`；
  D：同父目录 `occluded512-torchref-winograd-off/`。

## CPU-only 结果

- `reference_write` C/D NRMSE `0.0132010`、max abs `0.0512085`；三条
  reference batch 分支的 NRMSE 相同。
- 16 个 `condition_embed_*.npy` cache 均非 exact；最早归档点
  `down_0_0_0` NRMSE `0.0009781`，最大为 `up_2_2_0` NRMSE
  `0.00740545`。这将首次可观测差异放在 reference/cache 侧，但不能由
  cache 误差大小直接推断最终像素。
- 首个 generation step 的三条 CFG noise 分支：未条件化分支 C/D **exact**，
  另外两条 NRMSE `0.00144444` / `0.000810331`；全量首步 noise NRMSE
  `0.000995092`。下一步未条件化分支才出现 NRMSE `0.001133`，符合
  条件分支先受 reference 差影响、再由 latent 反馈传播的次序。
- 全量 resident noise NRMSE 从 step 0 的 `0.000995` 增至 step 14 的
  `0.007048`；latent 从 step 0 的 `0.000208` 增至 step 14 的 `0.004577`。
  这些是累计轨迹差，不能反推出单个算子错误。
- 12 张图中 C/D 的 image 11 最大差 `42/255`，仅 4 个像素超过
  `32/255`；image 5 最大 `25/255`，image 10 最大 `6/255`。image 10 的
  局部 latent 误差虽高，仍不是最终超阈值图。
- image 5 局部 CFG mean abs 于 step 7 达 `0.148758`；image 11 于
  step 14 达 `0.276260`。两处由三分支 noise 线性组合预测的 residual RMS
  分别为 `0.000397` / `0.001100`。CFG/scheduler 没有新的首发分叉证据。

## 证据与验证

- 外部目录：
  `/Users/daubert/Github/hy3d-upgrade-evidence/20261029/r3-reference-winograd-optin-20261029/`。
- `occluded512-cd-reference-boundary.json` SHA256
  `6bc9400adb2fd4a252a1210b918fbf28409ed2af71c4e31eddc235782ab75922`。
- `occluded512-cd-view5-20261029.json` SHA256
  `a2da3c50b72b3beb2b9bbe41607dab449c7026ce5ae75b2a8b1c0404f99cb067`；
  使用仓库既有 `analyze_occluded512_view5.py` 生成。
- 两份报告及一段 CPU-only 分析脚本的逐文件 manifest SHA256
  `32c463a84a2b4d2909ae4c2c2a2bcac93265b8a6749fed7c87348d82794bc8e9`。
- 本阶段无代码改动；`git diff --check` 通过。根目录受保护 dirty
  `torchvision_fix.py` SHA256 仍为
  `d6139690b099ca3ecef80ddf3f1a25edfdf57f027c79bf8655a8f8aac1c8b493`。
  前一阶段宿主 Metal 全量 unittest 为 `88/88 OK`，本阶段未重复。

## 结论与下一步

C/D 的最早可观测差异在 reference write/cache；首步只有条件化 noise 分支
不同，后续所有分支随 latent 反馈发散。当前还不能把 `up_2_2_0` 或任何单层
cache 认定为 `42/255` 图像差的因果来源。默认 Torch reference、Winograd、
MLX dispatch 保持不变；探索性 `32/255` 门槛仍失败。

下一阶段优先做同一 generation 入参下的 cache 交换/注入实验：只替换一个
reference cache 或完整 cache，比较首步三分支 noise 与 image 11；若要完整
15 步，仍需 fresh output、空 ComfyUI queue、一次只跑一个 GPU，且不得覆盖
已交付 GLB。先确认既有 `probe_fixed_conditions.py` 是否支持安全的 cache 注入，
再设计最小新增 probe。

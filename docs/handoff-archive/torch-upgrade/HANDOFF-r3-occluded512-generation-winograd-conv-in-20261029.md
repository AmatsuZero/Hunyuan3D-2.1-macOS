# R3 occluded512 generation Winograd 边界（2026-10-29，已修正步骤编号）

本阶段在确认 ComfyUI `/queue` 的 `queue_running=[]`、
`queue_pending=[]` 且进程表没有第二个 Hunyuan 推理后，串行运行两次
fresh local-layer capture。两次都使用 15 步、512px、6 视角、guidance 3、
reference/mva scale 1、1 GiB MLX cache 和同一初始噪声；没有写入 GLB。

## 对照

- A：Torch reference，MLX generation 默认 Winograd。
- D：Torch reference，MLX generation 设置
  `HUNYUAN_MLX_DISABLE_WINOGRAD=1`。
- 两侧均为 Torch 2.14.1、`PYTORCH_ENABLE_MPS_FALLBACK=0`、MLX fp16
  generation；reference forward、DINO bundle、fixtures 和 scheduler 相同。
- 初版局部探针把 `forward_mlx` 调用序号误标为扩散 step；其 `step1/3` 文件
  实际对应扩散 step 2/4，不能用于首步因果结论。修正后的 probe 同时钩住
  `_mlx_forward_raw`，重新采集真正的扩散 step 1。
- 为缩小边界，修正后的局部探针采集真正 step 1 的
  `conv_in`、`down_0..down_3`、`mid`、`up_1.2.attn`、`up_2.2.attn`、
  `up_3.2.attn` 以及 `down_0` 首个 ResNet 的 norm/conv，保留 image 5/view 5、image 11/view 5 和 image 10/view 4
  三个局部窗口的三条 CFG 分支。

## 证据归档

外部目录：
`/Users/daubert/Github/hy3d-upgrade-evidence/20261029/r3-reference-winograd-optin-20261029/occluded512-ad-first-step-layer-capture/`

- `torch-reference/`：A fresh capture；`torch-reference-winograd-disabled/`：D
  fresh capture。两侧各 102 个文件，包含 `capture.json`、`local-layers.json`、
  54 个局部 patch 和 12 张生成图。
- `analysis.json` 是 CPU-only 合并报告，SHA256
  `7888b2b3f726ef1745433872d171f1d1322166c0141deb9ab29b038a6c3ddbd9`。
- 两个 fresh capture 的 12 张图分别与既有 A/D 归档逐文件 SHA256 exact；
  两侧目录与 `/private/tmp` 源逐文件 SHA256 exact。

## 初版结果的修正

- fresh A/D 的 `initial-latents`、`inner_sample`、`reference_sample`、
  `reference_write`、DINO hidden states 和 generation text embeddings 全部
  exact；另核对 16 个 reference cache、4 个 position indices 与 4 个 inner
  scalar 输入均 exact。`reference_write` 的跨侧 NRMSE 为 `0`。
- 首步 `noise_pred_0`/`inner_output` 的 A/D NRMSE 为 `0.0008369135`，
  mean absolute error `0.000339919`，max absolute error `0.025390625`，
  最大索引 `[27,2,38,41]`（第 2 CFG 分支、view 3、channel 2）。这与既有
  resident A/D 对照完全一致。
旧版 `analysis.json` 的局部差异仍是有效的后续步观察，但不能称为首步边界。
真正 step 1 的修正结果见下一阶段交接；其中 `conv_in` 与
`down_0_resnet_0.norm1` exact，最早差异在 `down_0_resnet_0.conv1`。

## 代码与验证

`tools/quality_demo/capture_occluded512_local_layers.py` 新增了两个诊断参数：

- `--capture-steps`：选择实际扩散 step；默认仍为原来的 `3 6 10 14`。
- `--detailed-down0`：保存首个 down ResNet 的 norm/conv 局部 patch。
- `--winograd-disabled`：允许 Torch-reference capture 显式声明
  `HUNYUAN_MLX_DISABLE_WINOGRAD=1`；默认环境变量和默认 dispatch 不变。

宿主全量 unittest fresh 结果：`Ran 88 tests ... OK`。
`git diff --check` 通过；受保护 `torchvision_fix.py` SHA256 仍为
`d6139690b099ca3ecef80ddf3f1a25edfdf57f027c79bf8655a8f8aac1c8b493`。

## 边界与下一步

初版探针不足以定位首步；修正后的证据把差异缩小到首个 down ResNet 的
`conv1`。独立单卷积 oracle 证明相同 frozen 输入下 `conv_in` 的默认与
disabled 输出 exact；CPU `conv1` oracle 则显示 disabled 更接近 CPU（NRMSE
约 `0.00032`），默认 Winograd约 `0.00256–0.00725`。尚未证明具体 Metal
kernel 实现或数值精度是唯一根因。`43/255` occluded512 探索性门槛仍失败，
默认 Torch reference、默认 Winograd、默认 MLX dispatch 继续不切换。

下一步优先对首个 down ResNet `conv1` 做完整空间/通道 oracle 与 Metal
kernel 路径核对；继续保持一次一个 GPU 任务，不重复生成形状。

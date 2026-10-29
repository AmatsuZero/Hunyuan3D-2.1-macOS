# R3 occluded512 真正 step 1：`down_0` conv1 边界（2026-10-29）

本阶段修正了上一阶段局部 probe 的步骤编号，并在空 ComfyUI queue 下串行完成
两次 fresh capture。上一版 `forward_mlx` 调用序号从 1 开始，但在 resident
loop 中对应扩散 step 2；真正扩散 step 1 走 `MLXDispatchUNet._mlx_forward_raw`，
因此旧的 `step1/3` 文件实际是 step 2/4，不能支持“首步 `conv_in` 差异”结论。

## 合同与修正

- A：Torch reference / 默认 Winograd；D：Torch reference /
  `HUNYUAN_MLX_DISABLE_WINOGRAD=1`。
- Torch 2.14.1、fallback=0、MLX fp16、同一 fixtures、初始 noise、DINO bundle、
  guidance/reference/mva scale、15 步和 1 GiB cache。
- `capture_occluded512_local_layers.py` 现在同时钩 `_mlx_forward_raw`，并把
  `forward_mlx` 后续调用映射为实际扩散 step；`--detailed-down0` 额外记录
  `down_0_resnet_0` 的 norm/conv 局部 patch。
- 两次运行前 `/queue` 的 running/pending 均为空，进程表没有第二个 Hunyuan
  推理；没有写交付 GLB。

## 证据

外部归档：
`/Users/daubert/Github/hy3d-upgrade-evidence/20261029/r3-reference-winograd-optin-20261029/occluded512-true-step1-winograd-boundary/`

- `default/` 与 `disabled/` 各 87 个文件，12 张图分别与原 A/D 归档逐文件
  exact；两侧目录与 `/private/tmp` 源逐文件 SHA256 exact。
- `analysis.json` SHA256
  `de883e3ebb78a476a912ca372feac1e6daa5d53b4627fd1a75335a6f5f1e746d`。
- `conv1-cpu-oracle.json` SHA256
  `2951471958fd856484da51f384d7ed6ed9488ca5cf553be0f24965408b16df3b`。
- `conv-in-oracle-default.npy` 与 `conv-in-oracle-disabled.npy` SHA256 均为
  `860b508df4f9dec6da7e48fa1d3f1a400ad6f93573f2d9d74b8882a0f80905ce`；两种
  policy 的独立 conv_in oracle 输出逐元素 exact。

## 真正 step 1 结果

- A/D 的 26 个边界数组（initial/inner/reference/DINO/text、16 个
  reference cache、4 个 position index）和 4 个 inner scalar 全部 exact；
  `reference_write` exact。
- `conv_in` 与 `down_0_resnet_0.norm1` 三个局部窗口逐元素 exact。
- 最早非零层为 `down_0_resnet_0.conv1`。image 5/11/10 的局部 NRMSE 分别为
  `0.003438/0.005074/0.004221`；之后的 `down_0` 为
  `0.004579/0.003703/0.004977`。首步 noise NRMSE 仍为
  `0.0008369135`，与既有 resident A/D 对照一致。
- 使用同一 frozen `norm1` patch 和 checkpoint 在局部 valid interior 计算的
  CPU float32 `conv1` oracle：
  Winograd-disabled 相对 CPU NRMSE 为 `0.000327/0.000324/0.000330`，默认
  Winograd 为 `0.002562/0.007253/0.004565`。这排除了权重、输入和 SiLU
  作为首个差异源，并把差异缩到 320→320 的 Winograd 相关卷积路径。
- 独立 full `conv_in` oracle 在默认/disabled 两种 policy 下 exact，纠正了
  上一阶段把后续步 `conv_in` 差异当作首步差异的表述。

## 代码与验证

- 变更文件：`tools/quality_demo/capture_occluded512_local_layers.py`。
- `python3 -m py_compile` 通过；宿主全量 unittest 前一阶段 fresh 结果为
  `88/88 OK`，本阶段只扩展诊断钩子。
- `git diff --check`、受保护 `torchvision_fix.py` SHA256
  `d6139690b099ca3ecef80ddf3f1a25edfdf57f027c79bf8655a8f8aac1c8b493` 保持。

## 边界与下一步

证据支持 Winograd policy 在首个 down ResNet conv1 改变数值轨迹，但还没有
证明具体 Metal kernel 源码错误、最终图像局部像素的直接因果或应切换默认策略。
`43/255` occluded512 探索性门槛仍失败；默认 Torch reference、默认 Winograd、
默认 MLX dispatch 继续不切换。下一阶段优先检查该 conv1 的完整空间/通道
oracle 与 MLX convolution policy 实现来源，避免再扩大到完整 pipeline。

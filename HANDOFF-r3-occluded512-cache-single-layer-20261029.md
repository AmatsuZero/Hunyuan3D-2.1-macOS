# R3 occluded512 单层 reference cache swap（2026-10-29）

本阶段在 §27 完整 16 层 cache swap 的基础上，分别只把 C cache 的一个层替换
成 D cache，验证最早差异层与最大 NRMSE 层是否能单独解释 C/D 的 `42/255`
图像差。默认路径、Torch/MLX dispatch、Winograd policy 和已交付 GLB 均未改动。

## 固定合同

- C reference：MLX fp32；D cache：Torch reference + disabled Winograd 归档。
- generation：MLX fp16，first step + resident loop，Winograd disabled；15 步、
  guidance 3、reference scale 1、MVA scale 1。
- 同一 fixtures、initial noise、DINO/text/position 输入；每次只运行一个 GPU
  任务，ComfyUI queue 启动前为空。
- 合成 cache 目录只含 16 层 `condition_embed_*.npy`：其余 15 层来自 C，目标层
  来自 D；`_load_reference_cache()` 与 runtime `consumer_exact` 均通过。

## 两个单层结果

机器可读摘要：
`/Users/daubert/Github/hy3d-upgrade-evidence/20261029/r3-cache-single-layer-20261029/hy3d-cache-swap-single-layer-summary-20261029.json`
SHA256 `d87dd31abafda625f39cae33d87ab4ed635fd9043c3888ff68c9a0e75e8f522f`。

| 单独替换层 | 首步 noise NRMSE vs C / D | image 11 max vs C / D | 12 图相对 C 最大差 | 结论 |
| --- | ---: | ---: | ---: | --- |
| `down_0_0_0` | `0.0003638 / 0.0010110` | `15 / 38` | `15/255` | 早期层有影响，但不足以复现 C/D |
| `up_2_2_0` | `0.0006963 / 0.0008948` | `13 / 35` | `13/255` | NRMSE 最大层有影响，但不足以复现 C/D |

两次 capture 都有 12 张图、`consumer_exact=true`，冻结输入相对 C/D 全部 exact。
完整 capture SHA256：

- `down_0_0_0`：`1cad247d2b73b29c7cf0d9a4feeca2d2162f18224c94aaac2a3e0773527f214c`
- `up_2_2_0`：`311452c18937d51bacc94db742f46b7cf98b2f9c3495b81aea7100664c30532d`

## 证据归档

目录：`/Users/daubert/Github/hy3d-upgrade-evidence/20261029/r3-cache-single-layer-20261029/`

- 包含两份合成 16 层 cache、两份完整 15 步 capture、摘要和逐文件 manifest。
- manifest 129 项，SHA256
  `06037adcb31686a2870c1ab73d11af6ee326c0228c5c0d84291982b80d64b094`。
- 归档与 `/private/tmp` 原始 capture 逐文件 exact。

## 验证与约束

- §27 的宿主 Metal 全量 unittest：`90/90 OK`；本阶段只新增外部诊断数据，未
  修改默认代码。
- 保护 `torchvision_fix.py` SHA256 仍为
  `d6139690b099ca3ecef80ddf3f1a25edfdf57f027c79bf8655a8f8aac1c8b493`。
- 没有覆盖或重跑形体、没有写入已交付 GLB、没有修改原始 checkout。
- §27 推送后按授权尝试了新 Codex 会话
  `01a185f4-8269-7173-82fd-823e6b1479e8` 与
  `01a18626-6722-79a1-b98c-b270e8703314`；均在开始时因平台 429（第二次
  还出现连接 Cloudflare 403）失败，未执行迁移步骤。本阶段因而在原会话继续。

## 结论与下一步

`down_0_0_0` 与 `up_2_2_0` 各自都能改变首步条件分支和最终图像，但单层替换
只能产生 `13–15/255` 的 image 11 差，不能单独解释完整 `42/255`。当前更符合
多层 cache 交互与 resident 轨迹放大，而非某个单层 cache 的充分因果。

下一阶段优先做这两个层的联合 swap（仍保持 C 其余层），或改做按 block 的
分组 swap；不应据此修改默认 reference producer 或接受颜色门槛。

# R3 occluded512 view 5 step-localization (2026-10-29)

本阶段只读取归档的 Torch reference 与 Winograd-disabled MLX opt-in resident
trace，未启动 GPU、未修改模型代码、未覆盖 GLB。工作区仍为
`/Users/daubert/Github/hy3d-torch-upgrade`，分支 `codex/torch-mps-upgrade`。

## 可复跑分析

脚本 `tools/quality_demo/analyze_occluded512_view5.py` 对 step 0→14 的
36 项 `noise_pred`（branch 0/1/2 = unconditional/reference/full）、CFG 后的
`resident_scheduler_model_output`、以及 scheduler 后 `resident_latents` 做同一
局部窗口的 mean-abs、RMS、max 和 NRMSE 对照。输入为：

- Torch reference：`20261028/r3-reference-fp32-resident-01/occluded512/torch-reference`
- MLX opt-in：`20261029/r3-reference-winograd-optin-20261029/occluded512-replay`
- image 5 窗口：latent `y=38..41,x=25..26`（view 5 / albedo）
- image 11 窗口：latent `y=41..42,x=28..29`（view 5 / MR）
- image 10 窗口：latent `y=42..44,x=34..36`（view 4 / MR，反例）

离线输出写在 `/private/tmp/hy3d-occluded512-view5-step-localization-20261029-v3.json`
（当前运行 SHA256 `a43ed9a906504644e959f18cd0cae93e42042598576949129f6b11a05e13c48f`）。

## 结果

| image | step 0→14 局部 NRMSE 峰值 | CFG 局部 mean-abs 峰值 | latent 局部 mean-abs 峰值 | 观察 |
| ---: | ---: | ---: | ---: | --- |
| 5 / view 5 albedo | 见 JSON；step 6–9 的三分支均明显放大 | step 7 `0.21860` | step 13 `0.08157` | 首次明显放大在 step 3，step 6–9 最强，后段趋稳 |
| 11 / view 5 MR | 见 JSON；step 10 后 full/reference 明显放大 | step 14 `0.32144` | step 14 `0.09807` | step 10 开始放大，step 14 最大，和 image 5 不是同一时间窗 |
| 10 / view 4 MR | 全程逐步升高 | step 14 `0.28879` | step 14 `0.18438` | latent 局部差更大但图像 max 仅 9/255，不能作为超阈值根因 |

逐步表明：view 5 的两个超阈值图像没有共享一个“全局最大 noise 坐标”或单一
step。image 5 的误差先在 step 3、6–9 进入局部；image 11 在 step 10–14
才进入局部。CFG 组合的 step 1–14 代数残差 RMS 不超过 `0.00140`（image 5）、
`0.00149`（image 11）和 `0.00025`（image 10）；step 0 的约 `0.0027–0.0040`
来自 float16 组合舍入。因而当前证据支持“分支误差被 CFG 放大并经 scheduler
累积”，不支持“scheduler 单独产生首次分叉”。这仍是定位证据，不是卷积算子
因果证明。

## 阶段边界与下一步

- 43/255 图像门槛仍失败；默认 Torch reference、默认 Winograd、默认 MLX
  dispatch 不变。
- 2026-10-29 GPU 前检查：ComfyUI `/queue` 返回 `queue_running=[]`、
  `queue_pending=[]`；唯一相关进程是既有服务 PID 3402（无推理任务）。
  `/system_stats` 报告 MPS 设备、总内存 48 GiB、空闲约 20.7 GiB。此次阶段
  未启动推理。
- 下一步若继续，先做单 GPU、fresh output 的局部中间层 capture，优先检查
  view 5 两张图在 step 3/6/10 的 reference cache 消费与 inner UNet 输出，
  并保留 image 10/view 4 反例。
- GPU 实验前必须检查 `http://127.0.0.1:8188/queue` 为空、无其他推理进程，
  只运行一个 GPU 任务并设置 1 GiB MLX idle cache；保护原始 checkout、旧 venv、
  已交付 GLB 与根目录 dirty `torchvision_fix.py`。

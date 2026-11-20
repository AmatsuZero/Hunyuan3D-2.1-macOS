# R3 交接：reference fp32 参数诊断（2026-10-28）

总交接为 `HANDOFF-torch-mps-upgrade.md`。本阶段基于 `7b1c5ec`，针对
`HANDOFF-r3-reference-coverage-20261028.md` 中两个失败输入做单变量诊断。

## 实现与证据

`tools/quality_demo/replay_mlx_reference.py` 增加 `--fp32`：加载同一 checkpoint
后把 MLX model 参数转换为 float32，报告记录 `precision=fp32`。输入保持 capture
的原始 float16 数值，Torch oracle 仍是原来的 fp16 MPS capture；默认 replay 仍为
fp16。此实验只改变 MLX 参数/随后计算的精度，不是全流水线 fp32 对照。

两次 replay 严格串行，输入路径、checkpoint SHA、686 权重键和 16 层结构均与
原失败报告相同；报告保存在已有 case 的新子目录 `mlx-fp32/`，没有覆盖 fp16 证据。

| case | fp16 max cache NRMSE | fp32 max cache NRMSE | fp32 write NRMSE | 原门槛判定 |
| --- | ---: | ---: | ---: | --- |
| `occluded512` | `0.010211126` | `0.006395094` | `0.023045830` | PASS |
| `reference384` | `0.014229643` | `0.006527736` | `0.021965840` | PASS |

根目录：
`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-reference-coverage-01/`

报告：`<case>/mlx-fp32/exact-input-hashed-metrics.json`。验证器在两份报告上均 exit 0，
保留 cache `0.01`、write `0.03` 门槛。相关脚本 `py_compile` 与 `git diff --check` 通过；
本阶段直接用真实两组 replay 验证新增诊断参数；宿主全量 unittest `81/81` 通过。
正式 dispatch 未改动，独立审查仍因服务 429 未完成。

## 结论与下一优先级

两组超限对提高 MLX reference 参数精度敏感；这支持精度造成差异的解释，还未定位
某个具体算子，也不能证明 fp16 kernel 有缺陷。本诊断阶段模型尚无独立的 reference fp32
开关，已有 `HUNYUAN_MLX_FP32=1` 同时影响 generation inner UNet，不能用作默认修复。

下一会话（已在后续阶段完成独立开关的代码切片）：

1. 先核对总交接、当前 HEAD、进程与队列；保护根目录 `torchvision_fix.py` 和交付资产。
2. 独立 reference fp32 配置、fp16 cache consumer 和 reset 合同已完成，详见
   `HANDOFF-r3-reference-fp32-consumer-20261028.md`；下一步是完整 15 步图像、内存
   和性能验收。未完成前不切默认。
3. 完成后写新交接、提交推送，再按用户授权创建新会话。先前接力任务已因 429 终止。

## 复现

按前一覆盖交接的 case 路径运行 `replay_mlx_reference.py`，加 `--fp32` 并使用全新
`--output`；再运行 `validate_mlx_reference_report.py --report <新报告>`。每次仅一个
GPU 推理，原始输入不变，不用新 seed 代替同输入对照。

# R3 reference 时间嵌入局部边界（2026-10-29）

本阶段沿 §30 的时间支路线索，增加 opt-in 探针并完成相同 frozen input、相同权重、
Torch/MLX fp32 的时间嵌入对照。MLX 使用显式 disabled Winograd。默认路径未切换，
没有生成或覆盖 GLB；这是局部数值定位，不是最终图像质量修复。

## 改动

- `hy3dpaint/mlx_unet.py`：可选 `_reference_debug_trace` 捕获时间嵌入两层 Linear
  的输入/输出、首个 ResNet 的 `temb` 与 SiLU 后 `time_emb_proj` 输入。
- `tools/quality_demo/trace_mlx_reference_up2.py`：增加六个探针，Torch/MLX 两侧对齐。
- `tools/quality_demo/oracle_reference_time_embedding.py`：CPU-only float32 oracle；
  对每个后端的**自身捕获输入**重放 Linear/SiLU，检查冻结输入和 checkpoint 哈希。
- `tests/test_reference_time_trace.py`：验证探针开启前后输出 exact、temb 传递和 SiLU 边界。

## 已验证结果

| 边界 | Torch vs MLX NRMSE | Torch 自身输入的 CPU oracle | MLX 自身输入的 CPU oracle |
| --- | ---: | ---: | ---: |
| 时间编码 / `linear_1.input` | exact | exact | exact |
| `time_embedding.linear_1` | `1.43e-8` | `1.57e-8` | `6.36e-9` |
| SiLU 后 / `linear_2.input` | `5.01e-8` | `7.52e-9` | `4.84e-8` |
| `time_embedding.linear_2` | `2.729e-4` | `2.95e-7` | `2.730e-4` |
| 首个 `time_emb_proj` | `6.337e-4` | `5.89e-7` | `3.745e-4` |

两端 `temb` 均与 `linear_2` 输出 exact，ResNet SiLU 对各自输入的 CPU 差仅约
`1e-8`。因此时间编码公式、第一层 Linear 和 SiLU 都不是本样本明显差异的首发点。
已观测的首个明显局部边界是 `time_embedding.linear_2`；`time_emb_proj` 本身也有
独立局部误差，不能把它的全部 `6.337e-4` 都归因于上游 temb。

这仍不能区分具体 MLX Linear/GEMM kernel、累加模式或参数 materialization 问题。
CPU oracle 使用 checkpoint fp16 数值提升到 fp32，和 Torch 控制的权重精度匹配；
后续最小 probe 仍需核对真实 MLX 层的 weight/bias、输入 dtype 和逐元素值。

## 证据与验收

目录：`/Users/daubert/Github/hy3d-upgrade-evidence/20261029/r3-reference-time-embedding-20261029/`

- `trace-torch-fp32/` 与 `trace-mlx-fp32/`：每侧 58 个探针、相同冻结输入/checkpoint 哈希。
- `cpu-oracle/oracle.json` SHA256：
  `b6686f8b15d2bbbffe0dd74c85dc81089525bc1ad55be0ece5c81249dce74c73`。
- 131 项 manifest SHA256：
  `873d1cfbf5616f427016a470e6639441b612888281a5369a03f2de5831ce17d0`。
- 新探针与 §30 的既有 52 个输出：Torch 52/52 exact，MLX 52/52 exact。
- 定向测试 1/1 OK，宿主全量 `unittest discover -s tests -v`：91/91 OK；四个 Python
  文件编译检查与 `git diff --check` 通过。测试日志及源码快照已归档。
- 根目录 `torchvision_fix.py` SHA256 保持
  `d6139690b099ca3ecef80ddf3f1a25edfdf57f027c79bf8655a8f8aac1c8b493`。

本阶段接力会话 `01a186e7-467e-7202-8267-d336618ce9c6` 完成代码与 Torch trace 后
因模型容量错误终止；原会话确认无残留 GPU 进程、ComfyUI queue 空后补齐 MLX trace、
CPU oracle、全量测试及归档。没有并行 GPU 推理。

## 下一阶段

优先做两个有界的 frozen Linear probe：`time_embedding.linear_2`（3×1280 输入、
1280×1280 权重）和首个 `time_emb_proj`（3×1280 输入、320×1280 权重）。
逐元素核对 MLX 参数与 checkpoint 转换，比较 `nn.Linear`、显式 `mx.addmm`/matmul、
MLX CPU 与 Torch CPU fp32；先确认直接调用能复现归档误差，再查 dispatch/累加模式。
每次只改变一个因素，避免为追逐最终像素而盲目替换整个 block。

在局部原因和改法通过对照后，才能重放 16 层 reference cache，再跑完整 15 步。
trace 仍未保存最终 16 层 cache / reference write 数值，不能用探针一致代替正式 consumer
验真。默认路径、Winograd 默认策略和已交付 GLB 均保持不变。

可复制 CPU oracle 命令（输出目录必须不存在）：

```sh
rtk proxy /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python \
  tools/quality_demo/oracle_reference_time_embedding.py \
  --torch-trace /Users/daubert/Github/hy3d-upgrade-evidence/20261029/r3-reference-time-embedding-20261029/trace-torch-fp32 \
  --mlx-trace /Users/daubert/Github/hy3d-upgrade-evidence/20261029/r3-reference-time-embedding-20261029/trace-mlx-fp32 \
  --checkpoint /Users/daubert/Github/hunyuan3d-artifacts/Hunyuan3D-2.1-modelscope/hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin \
  --output /private/tmp/r3-time-oracle-next-fresh
```

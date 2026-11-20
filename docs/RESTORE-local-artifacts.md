# 本地大文件还原说明（hy3d 项目清理）

清理时间：2026-11-20。因本机磁盘空间不足，hy3d 相关目录已删除，等高优先级项目结束后再恢复。

本仓库已保存全部代码改动与实验结论（HANDOFF 文档，见 `docs/handoff-archive/`）。体积大、可重建的资产没有提交，按下述方式还原。

## 1. 模型权重（原约 14G）

原路径：`/Users/daubert/Github/hunyuan3d-artifacts/Hunyuan3D-2.1-modelscope/`

包含三个子目录，均为 Tencent-Hunyuan Hunyuan3D-2.1 官方权重：

- `hunyuan3d-dit-v2-1`（约 6.9G，shape DiT）
- `hunyuan3d-paintpbr-v2-1`（约 6.4G，paint PBR，结构为 SD 2.1 / Zero-Terminal-SNR，含 unet/image_encoder/text_encoder/vae 等）
- `hunyuan3d-vae-v2-1`（约 625M）

还原方式（任选其一，下载到上述原路径即可，脚本里的绝对路径无需改动）：

- HuggingFace：官方仓库 `tencent/Hunyuan3D-2.1`（见本仓库 README 的 Download 链接），用 `huggingface-cli download tencent/Hunyuan3D-2.1 --local-dir <原路径>`；部分仓库需先登录并接受 license。
- ModelScope：搜索 `Hunyuan3D-2.1` 官方模型库，用 `modelscope download` 下载同名子目录到原路径。

注意：`hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin` 是 `hy3dpaint/mlx_utils.py` 里硬编码的 checkpoint 路径之一；如换用 HuggingFace 缓存路径，可设置环境变量 `HUNYUAN_PAINT_UNET_CKPT` 指向实际文件，无需改代码。

## 2. 性能 trace / 诊断产物（原约 46G）

原路径：`/Users/daubert/Github/hy3d-upgrade-evidence/`（按日期分 `20261027/` `20261028/` `20261029/`）

内容为 Xcode Instruments `.trace` 包、`.npy` 中间数组、渲染 PNG 等诊断产物，均可重跑生成脚本复现，**未提交**。每个实验的复现命令、输入精度与结论记录在：

- 本仓库 `docs/handoff-archive/torch-upgrade/HANDOFF-*.md`（Torch/MPS 升级、MLX 一致性实验）
- 本仓库 `docs/handoff-archive/mlx-eval/HANDOFF*.md`（MLX DiT 评测）

需要具体某份 trace 时，按对应 HANDOFF 文档里的脚本与命令重跑即可，无需恢复原始二进制文件。

## 3. 环境与路径变量

实验脚本普遍依赖以下变量，还原后重新设置：

```
HY3DGEN_MODELS=/Users/daubert/Github/hunyuan3d-artifacts/Hunyuan3D-2.1-modelscope
PYTHONPATH=/Users/daubert/Github/hy3d-mlx-eval/hy3dshape
```

硬件基线：Apple Silicon，48GB 统一内存，Python 3.11，MLX 0.32.4。

## 4. 仓库与分支

- `Hunyuan3D-2.1-macOS`（fork: AmatsuZero）— 主开发仓库，分支 `quality-shape-demo`；`codex/torch-mps-upgrade` 分支记录 Torch 升级实验（原 worktree 在 `hy3d-torch-upgrade/`）。
- `Hunyuan3D-2.1`（origin: AmatsuZero）— shape baseline 分支 `macos-shape-baseline`。

所有本地 WIP 与文档均已提交并推送到这两个远端，本地删除后可从远端 clone 恢复。

# R3 MLX 0.32.4 Winograd source 与显式 opt-out（2026-10-29）

本阶段核查当前运行的 MLX 0.32.4 对应的官方 source tag，并在同一 frozen
`down_0_resnet_0.conv1` 输入上验证官方显式 Winograd 开关。没有修改默认 dispatch、
模型代码、原始 checkout 或交付 GLB。

## 版本与来源

- 候选 venv 的 `mlx==0.32.4`、`mlx-metal==0.32.4`，Python wheel 的
  `METADATA`/`WHEEL` 与 `RECORD` 已检查；两包都没有 `direct_url.json` 或 VCS
  commit 字段。
- 本机官方 `https://github.com/ml-explore/mlx` Git 仓库有 `v0.32.4` tag，
  指向 commit `36969c094354086e15d2a327a9c50feb20b9cf11`；
  `mlx/version.h` 为 0.32.4。`mlx-swift` 的 `mlx-0.32.4` 分支把 Cmlx 的
  `mlx` gitlink 锁在同一 commit。该 tag 的
  `mlx/backend/metal/conv.cpp` SHA256 为
  `75a189e9b47395e5e0812f237cb28fbdf31ba2260c09660f5c7d3a24f3feb751`。
- 当前 wheel 的 `mlx/lib/libmlx.dylib` SHA256
  `d8c9393d584e781393c3effccfccd8ba85ad0c70264edaf9f897dca9875375c6`；
  `mlx/core.cpython-311-darwin.so` 通过 `@rpath/libmlx.dylib` 链接到它。
  该 dylib 内可见 `MLX_CONV_WINOGRAD`、
  `MLX_CONV_WINOGRAD_WORKING_SET`、`MLX_CONV_WINOGRAD_TILE_BATCH` 和
  `implicit_gemm_conv_2d_` 字符串。此前只扫描 `core.so` 而找不到变量，不能
  据此否定当前 binary 的 Winograd policy。
- 官方 tag 与已安装版本一致，binary 字符串也与 source 机制一致；wheel 无构建
  commit/`direct_url`，因此**仍不能证明当前 dylib 逐字节由该 tag 编译**。

## 0.32.4 dispatch 与实测

- `v0.32.4` 的 `dispatch_conv_2D_gpu` 对 3×3、stride/dilation 1、通道 32
  对齐且输入足够大的卷积，默认在工作内存许可下尝试 Winograd。
  `(36,64,64,320)` 的 `conv1` 符合条件。
- 现有 `HUNYUAN_MLX_DISABLE_WINOGRAD=1` 使
  `MLX_CONV_WINOGRAD_WORKING_SET=1`；`winograd_batch_step()` 的 3/4 budget
  为 0，返回 0，随后进入 implicit GEMM。0.32.4 的 dispatch 还支持
  `MLX_CONV_WINOGRAD=0` 显式跳过 Winograd。该显式环境开关是 0.32.4 相比
  0.32.3 的新增 source 差异。
- 空 ComfyUI queue、无第二个推理进程下，只运行一次独立 frozen `conv1` GPU
  probe，环境为 `MLX_CONV_WINOGRAD=0` 且 working set 未设。输出形状
  `(36,64,64,320)`；与前一阶段 `working_set=1` 的完整 `conv1` capture
  **逐元素 exact**，NRMSE/max abs 均 0，输出 `.npy` SHA256 均为
  `b9623423f010f4859cdcbdd161cd857fc0d99ed679cd2d940d0e8ce9a9f6b08e`。
  这只证明该卷积的数值等价，未证明完整 15 步流水线等价或图像门槛通过。

## 证据与验证

- 外部 source/binary 摘要：
  `/Users/daubert/Github/hy3d-upgrade-evidence/20261029/r3-reference-winograd-optin-20261029/mlx0324-source-boundary.json`，
  SHA256 `689bf9cd9402abd4234f6ed62d364a45889cdafd3d5f5ad04e03dd7956d468e8`。
- 独立卷积报告：同目录下
  `occluded512-full-down0-20261029/conv1-direct-winograd-off.json`，SHA256
  `47c0451bba3cedb4c8613967e18f6f5d8b7104909acf06d7ecc2c1dcdec488cd`。
- 生成脚本与输出均归档；5 个本阶段文件逐文件回读核验的
  `mlx0324-source-boundary-manifest.json` SHA256 为
  `687d56f8a27fe60a36dbfbbc52b03bdeb8e773f2edc28e7a8b0247b3d8f133a5`。
- 本阶段无仓库代码改动；`git diff --check` 通过，根目录受保护 dirty
  `torchvision_fix.py` 保持 SHA256
  `d6139690b099ca3ecef80ddf3f1a25edfdf57f027c79bf8655a8f8aac1c8b493`。
  前一阶段宿主 Metal 全量 unittest 为 `88/88 OK`，本阶段未重复。

## 结论与下一步

当前 0.32.4 的 policy 机制与 frozen conv1 probe 一致；它解释 A/D 的首步
数值分叉，仍不能把 Winograd 精度差称为 Metal kernel bug，亦不能把
`43/255` 的最终图像差归因到这个单点。默认 Torch reference、Winograd 与
MLX dispatch 继续不切换。

下一阶段应回到图像门槛：在已归档四路矩阵中优先定位同 disabled policy 的
C/D（MLX/Torch reference）残余 `42/255`，区分 reference cache 差与后续
generation 累积；仅在 CPU 归档分析不足时做新的单 GPU 实验。若将来改用
`MLX_CONV_WINOGRAD=0`，需在候选 env 及版本锁中显式验证支持，不能让版本漂移
静默改变 opt-in 语义。

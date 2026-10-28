# R3 交接：reference fp32 真实 resident 输入验证（2026-10-28）

总交接为 `HANDOFF-torch-mps-upgrade.md`；本阶段基于 `18cf308`。

## 当前结论

真实 resident 验证发现新的失败输入。独立 reference fp32 不能被视为覆盖所有输入的
修复；默认路径继续使用 Torch reference，generation 保持 fp16。此前 coverage replay
的 PASS 只对其精确输入成立，不替代本轮结果，也不放宽 cache `0.01` / write `0.03`。

## 输入合同

- 三组 reference：固定 512px；同一 512px reference 加中央白色遮挡块
  `(192,128,320,384)`；未遮挡 reference 缩到 384px。
- generation 始终为 512px、6 views、2 PBR、15 steps、相同 raw initial diffusion noise，
  fallback=0、1 GiB MLX idle-cache threshold、MLX DINO projector/position cache/resident。
- 每组分别运行 Torch reference / MLX reference fp16 / MLX reference fp32，generation
  均为 fp16；另跑一次 fp32 同实例两请求。每次新建进程的输出目录均为 fresh。
- 本轮不注入预生成 DINO/normal/position 条件。三个 VAE posterior 的 noise 用
  `np.random.default_rng(20261027 + encode_index)` 固定（生成 float32、再转 posterior
  dtype），并从当前 reference/controls 生成条件；repeat 前重置 encode index。
- 旧 `r3-reference-coverage-01` 注入 `dino_hidden_states/embeds_normal/embeds_position`，
  ref latents 未注入；因此进入 injection 分支后 reference VAE sampling 使用 ambient
  MPS RNG，而不使用上述 fixed posterior wrapper。相同 seed 不代表这两条路径同输入。
- 遮挡 case 的旧/新 reference timestep、encoder states 完全相同；reference sample
  NRMSE `0.00002702259337`，49152 元素中 4503 个不同，max abs `0.0009765625`。
  旧/新 Torch cache 最大 NRMSE `0.0047341793`（`up_2_2_0`）。这说明输入确有微小
  变化，但尚未定位其放大机制；不能直接归因于某个后端或 kernel。

## 完整结果

原 reference 门槛保持 cache 最大 NRMSE ≤ `0.01`、write NRMSE ≤ `0.03`。
开跑前 `protocol.json` 另声明本轮完整 loop 的**探索性筛查**：逐步 latent 最大 NRMSE
≤ `0.01`，12 图平均像素差 ≤ `0.25/255`、最大像素差 ≤ `32/255`。后者不是此前已有
的生产质量标准；不因本轮输出事后调整，也不替代用户设备/GLB 验收。

| case / reference | cache max NRMSE | write NRMSE | step latent max NRMSE | image mean / max (0–255) | 判定 |
| --- | ---: | ---: | ---: | --- | --- |
| fixed512 / fp16 | 0.00685167 | 0.02634891 | 0.00464408 | 0.04757 / 18 | 两类筛查通过 |
| fixed512 / fp32 | 0.00643489 | 0.02128252 | 0.00465007 | 0.05038 / 17 | 两类筛查通过 |
| occluded512 / fp16 | 0.01709722 | 0.03083737 | 0.01368142 | 0.11117 / 91 | reference、latent、image max FAIL |
| occluded512 / fp32 | 0.01432952 | 0.02617332 | 0.01273196 | 0.10135 / 79 | cache、latent、image max FAIL |
| reference384 / fp16 | 0.00685625 | 0.02502109 | 0.00436956 | 0.05149 / 13 | 两类筛查通过 |
| reference384 / fp32 | 0.00651370 | 0.02178813 | 0.00561421 | 0.07031 / 36 | reference/latent 通过，image max FAIL |

遮挡 fp32 最坏层为 `up_2_2_0`；`up_3_0_0/up_2_0_0/up_2_1_0` 也超过 `0.01`。
384px 的旧/新 reference sample NRMSE `0.00007300685316`，27648 元素中 2943 个变化，
max abs `0.00146484375`；本轮 fp16 的 PASS 不覆盖旧输入的 FAIL。

冻结的 reference sample `.npy` 文件 SHA256：

| case | 旧 coverage | 本轮 Torch reference（fp16/fp32 candidate 输入精确相同） |
| --- | --- | --- |
| occluded512 | `e2e3094ed6be8fe2693c092f2cf4c505d3f39ab5fd7dbbe5d9c82e7486ba7e15` | `b5e9fb3634c1568ae8060365bc6380696af5426819b26c8f7ba7c7aaa6182828` |
| reference384 | `f5d40b953789ae716d75830a7c282fe1405372b528d835c130adadf346381655` | `ae4870a7dd55e25a3a49dcad05f479ebfb24fecfa83413413c7ee0973dd48f89` |

- 12 个 capture 命令全部 exit 0，共 15 次真实完整请求 / 225 个生成步；三组各有
  Torch reference、fp16、fp32 的 45 个 finite float16 trace，共 405 个逐步数组。
- 每个配对的 reference sample/timestep/encoder、initial latent、first inner sample/
  timestep/DINO/encoder 共 8 类输入逐元素精确相同；16 层 consumer cache 均为 fp16。
  所有报告明确 generation dtype 为 fp16，fp32 candidate 的 reference dtype 为 fp32。
- 三组 fp32 同实例 repeat 均为 12 图 `max_image_abs=0`、16 层 cache exact；两次
  实际 reset 的计数分别 `0→0`、`1→0`，cached/pending/producer cache 全部清空。
  无 trace 的 repeat 原图也与各自带 trace 的 fp32 capture 逐元素相同。这只证明
  **同输入两次请求**，未替代不同输入/分辨率 A/B/A 的生命周期验收。
- 三组 albedo/MR 六视角 contact sheet 已核看：整体外观和黑 H/Y/D、红 3 保持；
  该观察不抵消表中的像素/数值 FAIL，也不是独立设备或完整 GLB 验收。
- 最终汇总 `complete=true`、`fp32_quality_passed=false`，exit **1**。这是保留失败
  的预期行为，不是捕获程序 crash；不能把 12 个 capture exit 0 写成质量全部通过。

## 耗时和内存

所有值为单次诊断样本；完整 capture 带逐步 trace，不是无 profiling 的性能验收。

| case | fp16 / fp32 inference s | fp16 / fp32 MLX peak GiB | fp16 / fp32 dispatch load s |
| --- | --- | --- | --- |
| fixed512 | 82.101 / 81.508 | 8.104 / 9.716 | 2.489 / 2.545 |
| occluded512 | 84.147 / 83.297 | 8.104 / 9.716 | 2.450 / 2.666 |
| reference384 | 73.749 / 73.651 | 8.066 / 9.678 | 2.400 / 2.503 |

Torch load-to-MPS、reference wrapper elapsed、repeat elapsed 和 RSS/MPS/system swap
原始计数均见各自 capture。fp32/fp16 inference 比为 `0.993/0.990/0.999`，未观察到
超过本轮 10% 筛查线的回退，不能声称加速；MLX peak 增加约 **19.9–20.0%**。
本轮 20% peak 筛查线未触发，但非常接近边界，且不是 48 GiB 安全内存证明。

MLX peak 在 load 后 reset，随后第一请求与 repeat 的 peak 分别记录（repeat 计数为
两次请求累计最大值）。MPS 是快照而非 peak，RSS high-water 是进程生命周期值，
allocator/process/system 计数重叠、不可相加。reference timer 在调用后同步，其值可能
含此前已排队 MPS 工作，因此只称 wrapper elapsed，不用作纯 reference kernel 归因。

## 证据、验证与代码范围

持久归档：
`/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-reference-fp32-resident-01/`
原捕获根目录：`/private/tmp/hy3d-r3-resident-validation-20261028-01/`。
归档保留原命令路径；通过该目录的相对文件名读取复制后的证据。
已逐文件验证归档的 1130 个 manifest 条目（约 1.37 GB），全部 hash 匹配。
`summary.json` SHA256 为
`e0f0448ffe841e60ba76f7851c7346d5918aa8638cc0ef88edfe81beb4cb4d4b`；
`sha256-manifest.json` SHA256 为
`8085c79ad5d8d568294c99b83d5f8ee32e7ca5d2fa362f8da1d8d6847210b16e`。

- `summary.json/summary.log/protocol.json/commands.json`：全部指标、开跑前门槛与命令。
- `<case>/<mode>/`：精确 reference 输入/16 层 cache、inner 条件、初始 noise、12 张图、
  45 个 trace（repeat 例外）及 actual runtime source SHA；日志在 `<case>/<mode>.log`。
- `precision-and-provenance.json`：每个实际 capture 的 precision、模块来源与 hash；
  checkpoint SHA 为 `675a1b5cd0098b2002637c443946529c03c5cd54427f40245263350feb3dd5b8`。
- `coverage-input-comparison.json`：新旧精确输入、逐层 cache 差异及哈希。
- `frozen-inputs/`：三组原 reference/control PNG、旧 Torch coverage capture、当前
  first inner sample 的 normal/position 通道精确切片。posterior noise 另按已知算法/
  seed/shape 重建保存；这是确定性重建，未冒充 runtime mean/std 的捕获。
- `implementation-snapshot/`、三个运行/汇总/冻结脚本和 `sha256-manifest.json`：
  复现来源与文件完整性。首个 Torch baseline 在追加 reset 观测前加载工具，故没有
  reset 字段；随后 fp32 repeat 均具备完整记录。中途新增的其余字段仅是计量注释和
  RSS 单位归一化，没有改变模型数值运算。
- 宿主 Metal 全量 unittest **83/83** 通过，exit 0；`unittest.log` 记录
  `Ran 83 tests in 0.812s`、`OK`。Python 语法编译和 `git diff --check` 通过。
- native 独立 reviewer 已审查诊断 diff 与汇总 gate，指出的实际 precision 校验缺口
  已补齐并复查通过；这不是独立数值算法/视觉验收。
- 版本化代码只增加 capture 的 load/内存/reference wrapper elapsed/reset 观测；
  本阶段没有修改 UNet、reference 算法、模型动态源码、容差或默认 dispatch。
- 起止 `protection-*.json` 完全相同；原始 checkout status、根目录受保护文件和五个
  交付/参考资产哈希未变。首版临时 shell 引用错误产生的 `10%`/`20%` 空文件已只清理
  这两项；修正的脚本随后完成所有实验，未提交这些临时文件。

复核命令（汇总器当前预期 exit 1，显示 `COMPLETE True FP32_QUALITY False`）：

```sh
rtk proxy /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python \
  /private/tmp/hy3d-r3-resident-validation-20261028-01/summarize_r3_resident_validation.py
rtk proxy /Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python \
  -m unittest discover -s tests -v
rtk proxy git diff --check
```

工作目录仍为 `/Users/daubert/Github/hy3d-torch-upgrade`，分支
`codex/torch-mps-upgrade`；仅推送 `fork`，以 `git log -1` 和
`git ls-remote fork refs/heads/codex/torch-mps-upgrade` 核对最终提交。

## 下一优先级

1. 先读取本轮冻结失败输入与 hash，核对 reference sample/encoder/timestep、16 层
   cache、inner sample/DINO/encoder、初始 noise、参数 precision 和实际模块来源。
2. 对同一失败输入重放 reference 前向，逐层定位 Torch/MLX 首次分叉；再比较旧/新
   sample 的微小变化如何进入 down/mid/up blocks。使用相同 checkpoint，不变更门槛。
   注意 `replay_mlx_reference.py --fp32` 当前只提升模型参数，sample/timestep/encoder
   仍从原 fp16 capture 构造；正式 dispatch 会将三者提升到 `_reference_dtype`，并把
   cache 转回 generation fp16。两者的 precision 合同要分别记录，不能把离线参数
   fp32 结果直接当作集成 reference fp32 的精确复现。
3. 定位后才做小范围修复及回归，再重新跑失败输入的 reference gate 和 15 步图像。
   未通过前不扩大 outer condition/VAE 或默认 dispatch 的迁移范围。
4. 每阶段按已授权流程写交接、推送 fork、创建下一独立任务；GPU 严格串行。

保护总交接 §3：不触碰根目录 dirty `torchvision_fix.py`、原始 checkout、旧 venv 或
已交付 GLB。本阶段没有生成或覆盖 GLB。

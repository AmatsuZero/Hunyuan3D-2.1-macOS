# R3 occluded512 两层 reference cache 联合替换（2026-10-29）

本阶段在单层 `down_0_0_0` / `up_2_2_0` 替换之后，保持 C 的其余 14 层不变，
同时将这两层换成 D，验证多层交互是否足以复现 C/D 的图像差。实验只使用已有
归档的 frozen inputs、initial noise 和 D cache，没有重跑形体，也没有切换默认路径。

## 合同与结果

- 参考：C = MLX fp32 reference；D = Torch reference + Winograd disabled。
- 生成：MLX fp16，15 步，guidance/reference/MVA = `3/1/1`，first-step + resident loop，
  Winograd disabled，MLX idle cache limit = 1 GiB。
- 本轮接手时 ComfyUI `queue_running=[]`、`queue_pending=[]`，没有其他本任务 GPU 推理。
  磁盘上已存在联合完整 capture，因而本轮仅回读验证，没有重复推理；没有将接手时的
  空队列追溯当作原始 capture 启动时的占用证据。
- 联合 cache 只含 16 个条件层，其中 `down_0_0_0` 和 `up_2_2_0` 来自 D，另外 14 层来自 C；
  capture 报告 `consumer_exact=true`。
- 完整 capture 有 12 张图，`steps=15`、`stop_after=0`、fallback=0；15 类冻结输入
  数组相对 C/D/两个单层 capture 全部逐元素 exact。

机器可读证据归档：
`/Users/daubert/Github/hy3d-upgrade-evidence/20261029/r3-cache-joint-layer-20261029/`

- 逐文件归档 manifest：66 项，SHA256 `73df44eb33d711316b37604c594c89b9a89250311908c1af45935ce664dcb1c7`。
- 验证摘要：`verification/summary.json`，SHA256 `72902bd8ed1c9afc4da468b74555f96d5a44c88365ce00785477004def3a9444`。
- 联合完整 capture：`capture/capture.json`，SHA256 `492e9c415d121dd63f63f3ac0034befc5240e32089e0e19aa0274d0e9f2fe9b9`。
- 联合合成 cache：`synthetic-cache/`；校验脚本为 `verify_hy3d_joint_swap.py`。

## 图像和轨迹对照

以下数值是联合输出相对各参考输出的逐像素最大绝对差：

| 图像 | 联合 vs C | 联合 vs D | 联合相对 C 的超 `32/255` 像素 |
| --- | ---: | ---: | ---: |
| image 5 / view 5 | `19/255` | `43/255` | 0 |
| image 11 / view 5 | `17/255` | `36/255` | 0 |

其余图像联合 vs C 最大 `≤17/255`；联合 vs D 的差仍保留。首步 `noise_pred_0`
相对 C 的 NRMSE 为 `0.0007161`，相对 D 为 `0.0009079`。联合与两个单层结果的
image 11 signed difference-in-differences 最大值为 `17`，说明两层交互不是简单的
逐层图像差相加。联合的 `reference_write` 与 D 仍有 NRMSE `0.0132010`（max abs
`0.0512085`），因此 cache 注入没有伪造 producer 输出。

## 结论

两层联合替换仍不足以复现 D 输出：联合相对 D 的 image 5 最大差 `43/255`、
image 11 最大差 `36/255`。原始 C/D 的 image 11 最大差为 `42/255`；这些最大差
不是同一对输出，不能直接相减当作已解释的比例。当前证据继续支持“reference
producer 先产生多层 cache 差异，随后由 generation 轨迹放大”的模型，尚不能把
任一单层或这两个层的组合认定为根因。默认 Torch reference、默认 Winograd、默认
MLX dispatch 和已交付 GLB 均保持不变。

## 下一阶段

优先补齐 **disabled Winograd 下当前 C/D 精度合同**的 reference producer 层级差分：
C 是 MLX fp32 reference，D 是 Torch fp16 reference，而早先
`HANDOFF-r3-reference-down0-conv1-20261028.md` 的 producer trace 是两端 fp32、
默认 Winograd，不能直接用作当前 C/D 的首次分叉证据。先读已有
`trace_mlx_reference_up2.py` 和 coarse/down0 trace 工具，复用已有探针；同一 frozen
input 下比较 reference 输入到 `down_0_0_0` 的前向边界，必要时加入 Torch fp32
作为精度对照，区分半精度舍入与后端差异。`reference_write` 是末端输出，不是
`down_0_0_0` 的上游。已有满足相同合同的证据应优先复用。

若只能继续 consumer 侧实验，改用按连续 block 分组的 cache swap，并提前设定
“是否复现 image 5/11”的停止判据；不要继续无界枚举层组合。下一阶段成功条件是
定位当前 C/D 最早 producer 数值边界并记录精度控制，尚不要求据此切换默认。

仍须遵守：一次只跑一个 GPU 推理；每次 fresh output；不覆盖 GLB；不修改原始
checkout；不提交或清理根目录 `torchvision_fix.py`。

## 验证

- 宿主 Metal 运行时结果已存在并由 `verify_hy3d_joint_swap.py` 重新读取核验。
- 所有冻结输入、16 个 cache 层、capture JSON 和 12 张 PNG 已逐文件复制并 SHA256
  核对到上述外部证据目录。
- `torchvision_fix.py` SHA256 仍为
  `d6139690b099ca3ecef80ddf3f1a25edfdf57f027c79bf8655a8f8aac1c8b493`。
- 本阶段没有修改运行代码；后续会话开始前应重新检查 git 状态、ComfyUI queue 和
  活跃 GPU 进程。
- 独立 native reviewer 启动因平台 `429 Too Many Requests` 失败；没有独立审查通过结论。

复验归档完整性（只读 CPU，无 GPU）：

```sh
rtk proxy python3 - <<'PY'
from pathlib import Path
import hashlib, json
root = Path('/Users/daubert/Github/hy3d-upgrade-evidence/20261029/r3-cache-joint-layer-20261029')
manifest = json.loads((root / 'manifest.json').read_text())
for name, expected in manifest.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
print(len(manifest), 'files verified')
PY
```

# R3 occluded512 local layer capture (2026-10-29)

本阶段在空 ComfyUI queue、无第二 Hunyuan 推理的宿主状态下，串行运行了两次
fresh output capture。原始 checkout、旧 venv、GLB 和根目录 dirty
`torchvision_fix.py` 均未触碰；默认 Torch reference、默认 Winograd、默认 MLX
dispatch 没有切换。

## 运行合同

- worktree: `/Users/daubert/Github/hy3d-torch-upgrade`
- branch: `codex/torch-mps-upgrade`
- frozen fixtures: `/Users/daubert/Github/hy3d-upgrade-evidence/20261028/r3-reference-fp32-resident-01/frozen-inputs/occluded512/fixtures`
- fixed initial noise: `/Users/daubert/Github/hy3d-upgrade-evidence/20261027/color-diagnosis/initial-noise-torch251-seed0.npy`
- 15 steps, 512px, 6 views, guidance 3, reference/mva scale 1, 1 GiB MLX cache
- selected steps: `3, 6, 10, 14`
- selected local windows: image 5/view 5 albedo `y=38..41,x=25..26`; image 11/view 5 MR
  `y=41..42,x=28..29`; image 10/view 4 MR `y=42..44,x=34..36` as counterexample
- selected inner UNet points: `conv_in`, `down_0`, `down_1`, `down_2`, `down_3`,
  `mid`, `up_1.2.attn`, `up_2.2.attn`, `up_3.2.attn`

The Torch-reference side used MLX generation with Torch reference cache
(`HUNYUAN_MLX_REFERENCE_FORWARD=0`, Winograd default). The candidate side used MLX
fp32 reference cache and Winograd-disabled generation
(`HUNYUAN_MLX_REFERENCE_FORWARD=1`, `HUNYUAN_MLX_REFERENCE_FP32=1`,
`HUNYUAN_MLX_DISABLE_WINOGRAD=1`). Both used `PYTORCH_ENABLE_MPS_FALLBACK=0` and
the same DINO bundle.

## Evidence

External archive:
`/Users/daubert/Github/hy3d-upgrade-evidence/20261029/r3-reference-winograd-optin-20261029/occluded512-local-layer-capture/`

- `torch-reference/` and `mlx-reference/` contain `capture.json`, `local-layers.json`,
  selected local `.npy` patches, and all 12 generated images.
- `comparison-boundary.json` SHA256:
  `408cb06964d155ce60ecaa4a4ad82ab5afa969adf12aa9ee4da75365ec1bc0f4`
- `torch-reference/local-layers.json` SHA256:
  `01ab4677f906bd731e10cc43a9671fc46f241732141e57e675d0a71d23e734bf`
- `mlx-reference/local-layers.json` SHA256:
  `0f01863c994d22a4ebd73d4fed11f8092d204dd5fff893ed206a8eb73f624326`

Both instrumented runs produced 12 images byte-for-byte equal to their corresponding
existing archive (`image_exact: torch=true, mlx=true`). Queue was empty after both
runs. The capture elapsed times were `65.138s` and `84.153s`; this single pair is not
a performance conclusion.

## Result and boundary

- The earliest selected layer, `conv_in`, already contains a local difference at
  step 3 (image 5 NRMSE `0.006109`; image 11 `0.001060`; image 10 `0.000873`).
- The fresh captures make the boundary sharper: `initial-latents`, `inner_sample`,
  DINO hidden states, generation text embeddings, reference sample, and reference
  text embeddings are all exact between sides. `reference_write` is already
  different at NRMSE `0.013201`, and the first resident `noise_pred_0` is different
  at NRMSE `0.001212`. The 16 cached reference tensors also differ, with the largest
  cache NRMSE `0.007406` at `up_2_2_0`. Thus the first observed divergence is in the
  Torch reference write versus MLX reference forward/cache boundary, before the
  later generation-side local layer differences; the step-3 `conv_in` difference
  is downstream of that boundary.
- This capture cannot assign the reference boundary difference to one attention or
  convolution subkernel.
- Differences generally grow through the down path. At step 14, image 5 reaches
  `up_3.2.attn` NRMSE `0.052962`, image 11 `0.072279`.
- The counterexample image 10/view 4 reaches a larger step-14 `up_3.2.attn` NRMSE
  `0.117600` while its final image had only `9/255` max difference in the prior
  archive. Thus a large local inner-UNet difference alone does not explain the two
  view-5 pixels over the exploratory image threshold.
- This narrows the evidence to pre-inner-UNet latent differences and subsequent
  spatial amplification. It does not prove a specific convolution, attention
  kernel, or Winograd implementation as the root cause.

## Reproduction and next step

The committed diagnostic is
`tools/quality_demo/capture_occluded512_local_layers.py`. It supports `capture`,
`finalize`, and offline `compare`; every capture requires a fresh output directory.
The first failed attempt (`/private/tmp/hy3d-r3-occluded512-local-torch`) was a script
callback-order bug and produced no accepted evidence; it is not part of the archive.

Next useful work is CPU-only analysis of the already captured `inner_sample` and
three-branch tensors around step 0→3, or a narrowly instrumented first-step capture.
Do not change default dispatch or accept the `43/255` occluded512 image gate from this
local-layer result.

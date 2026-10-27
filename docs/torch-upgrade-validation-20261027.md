# Torch upgrade validation — 2026-10-27

Status: experimental candidate; **do not switch the default environment**.

## Protected baseline and candidate

- Original checkout: `/Users/daubert/Github/Hunyuan3D-2.1-macOS`, `quality-shape-demo`, `5b0070e` plus saved runtime overlay.
- Original Python: `/Users/daubert/Github/.venvs/hunyuan3d-shape/bin/python`, Torch 2.5.1 / Torchvision 0.20.1.
- Candidate checkout: `/Users/daubert/Github/hy3d-torch-upgrade`, `codex/torch-mps-upgrade`.
- Candidate Python: `/Users/daubert/Github/.venvs/hunyuan3d-torch-upgrade/bin/python`, Torch 2.14.1 / Torchvision 0.29.1, Python 3.11.16.
- Evidence root: `/Users/daubert/Github/hy3d-upgrade-evidence/20261027/`.
- Original packages, full dirty status, paint-related overlay, model/input SHA256 manifest: `baseline/` under the evidence root. No approved shape regeneration; no historical GLB overwrite.

Native managed-worktree creation failed because this partial clone lacks training sample blobs. A sparse linked worktree excludes `hy3dshape/tools/mini_trainset/`; inference source was checked out normally.

## Environment gate

Official PyPI metadata was refreshed before installation, including cp311 macOS arm64 wheel hashes (`official-wheel-metadata.json`). Candidate dependency resolution and `uv pip check` passed. The experimental paint lock is [requirements/macos-paint-torch214.txt](../requirements/macos-paint-torch214.txt).

The old environment has a Dash/Werkzeug metadata conflict. The candidate omits unused web/demo packages instead of reproducing that conflict. Imported Lightning was retained at 1.9.5, Torchmetrics at 1.6.0; MLX, Diffusers and Transformers retain their old versions. Sympy increased to satisfy Torch's declared requirement. This is a paint environment, not shape/training/web UI acceptance.

The rasterizer was rebuilt in the candidate source. Its LC_RPATH resolves to the candidate Torch libraries (`native-links.txt`); a real Metal triangle produced 288 covered pixels, finite barycentrics and MPS outputs. Inpaint was compiled separately; xatlas is a versioned package in the candidate. The original extensions were not rebuilt. The paint entry now derives its code and RealESRGAN paths from its own checkout; isolation regression tests cover both paths.

## First strict full run

`candidate-audit-01/audit.json` records a successful full paint run with automatic MPS fallback unset:

| Measurement | Result |
| --- | --- |
| Load | 19.375 s |
| Inference, including instrumentation/snapshots | 206.500 s |
| Total after tool setup/import | 225.892 s |
| Outer UNet | 15 calls |
| Inner Torch generation UNet | 1 call |
| MLX generation UNet | 14 calls |
| DINO Transformer | 1 call |
| VAE encode / decode | 3 / 1 calls |
| RealESRGAN | 12 calls |

CLIP text is bypassed because the checkpoint uses learned text embeddings; image conditioning uses DINO. Direct VAE method calls are audited, including the Gaussian-distribution result of encode. CPU parameters, computational input/output tensors, missing stages and missing MPS cause audit failure. CPU scalar timestep is the explicit input exception.

This first run does **not** complete device acceptance: review found reference-write uses the separate `unet_dual`. A required wrapper was added for that model, and a repeat audit is needed. Placement, successful fallback-disabled execution and call counts also do not replace a correlated Metal trace.

Run 06 subsequently completed the required reference-network placement check as part of another full successful run; the remaining device gate is correlated GPU trace coverage, not that missing wrapper.

The GLB structural check passed (`candidate-glb-verification.json`): 24,087 UV-split vertices, 40,000 faces, finite geometry/UVs, one PBR material, two decodable embedded 2048×2048 images with valid references. All five protected input/historical files retained their hashes.

## Numerical and quality gates

All 16 bounded MPS probes passed with fallback unset: fp32/fp16 SDPA with mask/without mask and noncontiguous layouts, bicubic with/without antialias, indexed gathering and index_add. See `candidate-operator-parity-v2.json`.

The first probe version failed in the **CPU reference**, where Half antialiased bicubic has no kernel. The corrected probe uses a float32 oracle on the same already-rounded half inputs and retains the predefined tolerance (fp32 rtol=1e-4/atol=1e-5; fp16 rtol=1e-2/atol=2e-3). The failure report is retained. Small operator parity does not establish whole-model parity or safe removal of the real-size bake workaround.

**Color gate FAIL:** the raw candidate front multiview already renders H/Y/D red, along with the red 3; the input requires H/Y/D black. This remains visible in the baked atlas. It therefore precedes baking/inpainting. No manual recolor is used for acceptance. Different Torch kernels, random sampling and other environment differences have not yet been isolated as causes.

**Memory gate unresolved:** one-second sampling observed max process RSS 9.745 GiB and system swap growth from 12.05 to 47.73 GiB, with a sampled swap maximum 47.96 GiB. RSS is not total GPU/unified-memory usage; system swap is not attributable solely to this process. These measurements are not a safe-memory/performance acceptance.

## Remaining gates

- The old matching benchmark was interrupted with SIGTERM after over ten minutes without multiview output and substantial swap (`baseline-benchmark-01-status.json`, shell exit 143). A process sample observed footprint 48.1G. This is an interrupted diagnostic run, not a completed speed comparison; historical 476.476 s is not a matched benchmark.
- The first complete Metal trace attempt crashed in AGXMetalG17X `setComputePipelineState`, called by Torch MPS reduction/max (`candidate-audit-02-crash.ips`; trace target SIGSEGV). No Python audit report was produced. Minimal max probes pass in normal execution, with standalone MPS profiler, under xctrace alone, and under xctrace plus profiler for float32/float16/int32/int64. The full-workload cause remains unresolved; it must not be generalized to every MPS reduction. A repeat launches Python directly and preserves target stdout.
- Direct-Python launch reproduced the full-workload trace crash (`candidate-audit-03-crash.ips`). Attaching after initialization avoided the immediate crash but stalled the workload and generated about 15 GiB of raw records; conversion memory exceeded 12 GiB. Run 04 was interrupted (workload 143, recorder 1). Its raw trace is incomplete and is **not accepted GPU evidence**. Future recording must target bounded Torch-only stages. No more whole-pipeline trace attempts in this session.
- Strict ordinary run 05 adds the separate reference-network check and writes each completed/failed event to JSONL immediately, preserving completed stage evidence even if a native failure prevents the final Python report.
- Ordinary run 05 also slowed progressively (MLX steps initially 8–12 s, later 26/59/79/94 s) without any active recorder; it was interrupted with exit 143. Trace overhead alone therefore does not explain the memory/performance instability. The completed reference-network event passed placement checks and remains in `candidate-audit-05/events.jsonl`; the final run contract is incomplete.
- A separate experiment, run 06, sets the MLX **free-cache reclaim threshold** to 1 GiB using `--mlx-cache-limit-mib 1024`. The default CLI leaves MLX policy unchanged. This is not a hard instantaneous memory cap: MLX reclaims excess cache on the next allocation. MLX 0.32.4's installed API docs state that its default cache limit equals its memory limit, itself 1.5×recommended working set on Metal. This policy may compete with Torch for the same 48 GiB pool. Per-stage active/cache/peak counters now persist to `timings.jsonl`; overlapping framework counters must not be summed as total memory. Cache-policy improvements cannot be attributed to the Torch upgrade.
- Matching old/new benchmark runs and repeatability remain incomplete.
- Explicit reference-write audit and correlated Metal GPU execution evidence.
- Color-quality regression diagnosis using saved multiview, super-resolution, bake and inpaint intermediates.
- Real-size bake CPU indexing exception; mesh/UV/inpaint and framework bridges remain CPU/native as documented.
- Default switch only after quality, device/trace, memory and stable performance gates pass. Preserve the original environment throughout.

No full MLX migration has started. This validation provides the baseline and failure fixtures for its later stages.

## Completed cache-budget experiment and final checkpoint

`candidate-audit-06-cache1g/audit.json` records a successful strict full run (exit 0), with automatic fallback set to 0 and the experimental MLX free-cache reclaim threshold at 1 GiB:

| Item | Result |
| --- | --- |
| Load / inference / total | 19.738 / 200.323 / 220.077 s |
| Reference Torch UNet / first generation Torch UNet | 1 / 1 passed calls |
| Outer UNet / MLX generation | 15 / 14 calls |
| DINO / VAE encode / VAE decode / super-resolution | 1 / 3 / 1 / 12 passed calls |
| 14 MLX step times | 7.388–9.482 s; no late-run cliff in this trial |
| Previous MLX cache limit | 48,962,627,174 bytes, approximately 45.6 GiB |
| MLX peak allocator counter | 8.985 GiB |
| Sampled MPS driver counter maximum | 25.698 GiB |
| Sampled process RSS maximum | 11.083 GiB |
| System swap first / last / maximum | 24.714 / 30.710 / 31.147 GiB |

The counters overlap and do not sum to total unified memory. Swap still increased by about 6 GiB, so memory acceptance remains incomplete. The consistent step times support further cache-budget experiments; they do not establish a Torch-upgrade speedup or a stable default policy. Framework cache policy must be matched in future old/new comparisons.

Run 06 GLB structural validation passed (`candidate-06-glb-verification.json`): 40,000 faces, finite geometry/UV, one material and two embedded 2048×2048 textures. All protected assets retained their hashes, and the original checkout's full porcelain status matched the initial snapshot byte for byte. The original environment still reports Torch 2.5.1 / Torchvision 0.20.1.

Color remains **FAIL**: front multiview H/Y/D are red. Runs 01 and 06 are visually similar but not pixel-identical (maximum front-image channel difference 10); numerical identity is not claimed. No new 3D-viewer/user acceptance is claimed.

The same bounded operator probe was run against the preserved old environment with fallback unset: Torch 2.5.1 passed 12/16 and failed both bicubic variants in fp32/fp16 with `aten::upsample_bicubic2d.out` / `aten::_upsample_bicubic2d_aa.out` not implemented for MPS. The candidate passed 16/16. This is concrete improved operator coverage, not proof that all pipeline operations execute exclusively on GPU.

The targeted/full unit suite passed 19 tests, including existing shape/mesh tests, checkout isolation, MPS availability/CPU-placement rejection, direct VAE output inspection, missing-stage rejection, overwrite protection, immediate failed-event persistence and cache-argument validation. Python compilation and `git diff --check` passed. Native reviewer dispatch failed with 429; a separate author review was performed, with no independent review approval claimed. Minor deferred: repeated parameter metadata makes audit reports larger; these instrumented runs are not used for speed claims.

Phase outcome: P0/P1 complete; P2 placement/call checks and named Torch-only Metal trace coverage complete; P3 bounded operator parity complete, real-size bake exception open; P4 structure and fixed-condition color diagnosis passed with user/device acceptance open; P5 matched performance/memory acceptance open. Stop promotion at these gates. Keep broad MLX migration and default version changes pending.

Experimental tooling and this report are committed on the isolated upgrade branch, without changing the default environment. The inherited runtime prerequisites (native linker path, bounded attention workspace and local MLX weight lookup) are recorded separately from newly authored audit changes. The unrelated root `torchvision_fix.py` modification is excluded; the verified entry uses `hy3dpaint/utils/torchvision_fix.py`. Model weights, GLBs, raw traces and machine-specific evidence remain external artifacts, not Git payloads. No default-switch, quality, complete GPU-trace or stable-speed acceptance is implied by committing this checkpoint.

## Controlled-noise color diagnosis

The real paint VAE was tested sequentially in both environments, using the same normalized 512-pixel reference, actual checkpoint, fp16 MPS and disabled automatic fallback. With a shared NumPy posterior-noise fixture, both reconstructions preserve black H/Y/D and red 3. Posterior-mean normalized RMSE is 0.000632; decoded-image normalized RMSE is 0.000350 (maximum channel difference 0.0078125). This bounded test does not establish whole-pipeline parity or exclude a later VAE interaction.

The same-seed noise probes differ across the two Torch versions: correlation 0.007770 for MPS noise and -0.001714 for CPU half noise (shape 1×4×64×64). Consequently, equal seeds in previous full runs did not control the actual random tensors. This is a demonstrated comparison confound, not an established cause of the color error. Evidence: `color-diagnosis/vae-comparison.json` and the old/new reconstruction arrays under the external evidence root above.

The audit CLI now supports diagnostic `--initial-noise PATH`, a raw NumPy fixture supplied to the real Diffusers `prepare_latents` before scheduler scaling. It captures the actual prepared tensor as `initial-latents.npy`, validates shape/dtype/finite values, records fixture SHA256 and restores the original method. Without a fixture, original RNG behavior is retained. Three tests exercise these contracts through the actual inherited Diffusers method; the complete suite passes 22/22.

Run 07 uses old Torch 2.5.1 CPU-generator seed-0 half noise, shape 12×4×64×64, SHA256 `925e9a05002409d327ec0bee2114bd9cad1e729971990b15797d8daa1dda10f2`, with the same candidate environment and 1 GiB MLX cache policy as run 06. Only initial diffusion noise is replayed; VAE sampling and image-conditioning computations remain candidate-generated. This is not a fully paired old/new replay. All outputs use a fresh external directory; delivered assets and defaults remain protected.

Run 07 completed with exit 0 in 226.793 s (load 21.498 s, inference 205.282 s). All required placement events passed: reference Torch UNet 1, first-generation Torch UNet 1, outer UNet 15, DINO 1, VAE encode/decode 3/1, super-resolution 12, plus 14 MLX generation calls. MLX steps ranged from 7.586 to 10.936 s. Captured initial latents exactly match the old raw fixture; the configured UniPC scheduler's initial sigma is 1.0.

**Color still FAIL:** visual inspection of the raw front multiview shows red H/Y/D. Replaying old initial noise alone did not restore the required sign colors. The run-06/run-07 front images differ (maximum channel delta 202, mean absolute delta 5.050 on 0–255 RGB); thus the intervention affects the image but does not correct this failure. No claim that randomness is irrelevant, or that a Torch kernel is the root cause, follows from this test. Next isolate conditioning/reference-network tensors and the first denoising step using fixed inputs before launching more full runs.

Run 07 GLB verification passed: 24,087 UV-split vertices, 40,000 faces, finite geometry/UV, one PBR material and two decodable embedded 2048×2048 textures. SHA256 is `ae28f0603b2423f5def597da7b9db3b77cdc2a6c0816fe7980eee14a5910675e`. All five protected assets and the original checkout's full porcelain status remain unchanged. Evidence: `candidate-07-glb-verification.json`, `candidate-audit-07-old-noise/audit.json`, `initial-latents.npy` and saved intermediates. The complete unit suite passed 22 tests; Python compilation and diff whitespace checks passed. Quality, correlated GPU trace and matched performance/memory acceptance remain open.

## P2 named Torch-only Metal trace closure

On 2026-10-27, five bounded Torch/MPS windows were recorded with `PYTORCH_ENABLE_MPS_FALLBACK=0`, correlated to the launched target PID, and fail-closed on incomplete recorder windows. The raw traces and exported tables remain outside Git under `/Users/daubert/Github/hy3d-upgrade-evidence/20261027/`; only the stage handoff records their paths.

| Window | Target evidence | PyTorch signposts | Metal GPU intervals | Fallback signposts | Result |
| --- | --- | ---: | ---: | ---: | --- |
| DINO + VAE encode (three calls) | `p2-conditions-trace-01` | 2,571 total | 1,314 total | 0 | pass |
| VAE decode | `p2-vae-decode-trace-01` | 441 | 296 | 0 | pass |
| RealESRGAN forward | `p2-superres-trace-01` | 2,445 | 1,055 | 0 | pass |
| Reference UNet | `p2-reference-trace-02` | 2,138 | 784 | 0 | pass |
| First-step outer/inner UNet | `p2-first-step-trace-03` | 5,961 | 9,251 | 0 | pass |

Every accepted window has target exit status 0, matching marker and target PID, and marker begin/end fully inside the trace start/end range. The first-step attempt `p2-first-step-trace-02` is retained as a negative result because the recorder ended five seconds before the marker end; the new summary tool rejects that case. The trace proves GPU execution in these named windows and does not prove that explicit CPU preprocessing, native mesh operations, or framework bridges elsewhere in the pipeline are absent.

The repository suite passed 30/30 under normal Metal permissions after this work. A sandboxed run without Metal access cannot import MLX; that environment failure is not used as acceptance evidence. Independent reviewer dispatch again returned 429, so no independent review approval is claimed.

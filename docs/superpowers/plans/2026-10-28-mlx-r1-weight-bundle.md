# MLX R1 Weight Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the inner UNet checkpoint once offline and load the resulting MLX-native bundle without importing PyTorch at inference time.

**Status:** Completed on 2026-10-28. The implementation and evidence are summarized in `HANDOFF-r1-mlx-weight-bundle-20261028.md`.

**Architecture:** Keep the existing Torch-backed loader as the offline conversion source of truth. Add a focused Torch-free MLX bundle module that validates metadata and loads exact MLX parameter names with strict key/shape checks. Add a converter that writes `.safetensors` plus a JSON manifest; do not change the default hybrid dispatch until the bundle has an independent smoke and parity evidence.

**Tech Stack:** Python 3.11, Torch 2.14.1 for offline conversion only, MLX 0.32.4, `safetensors`, HunyuanInnerUNet.

**Spec:** `docs/mlx-migration-roadmap.md`, R1 “权重与张量合同”.

## Global Constraints

- Runtime loader must not import `torch`, `diffusers`, or `transformers`.
- Offline conversion may use the candidate Torch environment and the existing explicit layout mapping.
- Conv2D weights use OIHW→OHWI; linear/norm tensors preserve the MLX model orientation.
- Bundle metadata must include schema, source checkpoint SHA256, MLX version, dtype, layout convention, and parameter count.
- Strict loading must reject missing, extra, or shape-mismatched parameters.
- Do not run GPU inference, alter the protected checkout, touch root `torchvision_fix.py`, or add model weights to Git.

## Review Focus

- A bundle with one missing or extra key must fail before acceptance.
- A bundle from a different source checkpoint must fail metadata validation.
- A runtime smoke must prove `torch` is absent from `sys.modules` after importing the loader and model.
- A loader must not silently accept a layout/dtype mismatch.
- A converter failure must not leave a manifest claiming a complete bundle.

---

### Task 1: Define the Torch-free bundle contract

**Files:**
- Create: `hy3dpaint/mlx_weight_bundle.py`
- Test: `tests/test_mlx_weight_bundle.py`

**Interfaces:**
- `BUNDLE_SCHEMA = "hy3d-inner-unet-mlx-v1"`
- `build_metadata(source_sha256, dtype, parameter_count) -> dict[str, str]` (reads the installed MLX version)
- `validate_metadata(metadata, expected_source_sha256=None, expected_dtype=None) -> None`
- `load_bundle(path, expected_source_sha256=None, expected_dtype=None) -> tuple[dict[str, mx.array], dict[str, str]]`
- `load_model_weights(model, path, expected_source_sha256=None, expected_dtype=None) -> model`

- [x] **Step 1: Write failing tests** for metadata round-trip, wrong schema/source/dtype rejection, strict missing/extra key rejection, and a subprocess import guard that blocks `torch`.
- [x] **Step 2: Run the focused tests** and confirm they fail because the module does not exist.
- [x] **Step 3: Implement** the Torch-free module using `mx.load(path, return_metadata=True)`, deterministic metadata validation, and `model.load_weights(..., strict=True)`.
- [x] **Step 4: Run the focused tests** under normal Metal permissions and confirm all pass.

### Task 2: Add the offline converter and manifest

**Files:**
- Create: `tools/quality_demo/convert_mlx_weight_bundle.py`
- Test: `tests/test_mlx_bundle_converter.py`

**Interfaces:**
- CLI arguments: `--checkpoint`, `--output`, `--manifest`, optional `--dtype` (`float16`/`float32`).
- Output: one `.safetensors` bundle containing the flattened MLX parameter names and a JSON manifest with source SHA256, source key count, mapped key count, extra source keys, schema and layout.

- [x] **Step 1: Write failing converter tests** with a tiny fake state dict/model mapping and assert the manifest fields and atomic failure behavior.
- [x] **Step 2: Run the tests** and confirm the converter entry point is missing.
- [x] **Step 3: Implement** the converter by reusing the existing `load_weights_from_pytorch` mapping, flattening `HunyuanInnerUNet.parameters()`, writing metadata through `mx.save_safetensors`, and writing the manifest only after the bundle is complete.
- [x] **Step 4: Run focused tests** and a CPU-only checkpoint-key audit; do not generate the 3.7 GiB production bundle unless the audit is complete.

### Task 3: Independent R1 smoke and evidence

**Files:**
- Create: `tools/quality_demo/check_mlx_weight_bundle.py`
- Modify: `docs/mlx-migration-roadmap.md`
- Create: `HANDOFF-r1-mlx-weight-bundle-20261028.md`

- [x] **Step 1:** Add a subprocess smoke that imports only the pure MLX loader/model, loads a test bundle, checks exact key/shape/dtype metadata, and fails if `torch` appears in `sys.modules`.
- [x] **Step 2:** Run the smoke and capture output plus source/config hashes in external evidence; keep the production bundle outside Git.
- [x] **Step 3:** Record the R1 boundary: conversion is complete, current default dispatch remains hybrid until R2/R3 integration, and no full MLX claim is made.
- [x] **Step 4:** Run `py_compile`, focused tests, full unittest under normal Metal permissions, `git diff --check`, then commit/push only code/tests/docs/handoff.

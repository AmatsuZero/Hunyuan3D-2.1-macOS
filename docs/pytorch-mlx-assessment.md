# PyTorch upgrade and MLX coverage assessment

Date: 2026-10-27. Scope: read-only environment/code inspection and official
package/documentation research. No package installation, framework upgrade,
new model inference or native-extension rebuild was performed.

## Conclusion

PyTorch 2.5.1 is the locally validated baseline, not a permanent project
requirement. An upgrade is feasible to evaluate in a separate checkout and
virtual environment. MLX currently accelerates selected inference subgraphs;
it does not remove the surrounding PyTorch dependency. These are two separate
work streams.

The first candidate to evaluate is `torch==2.14.1` with
`torchvision==0.29.1`, which upstream package metadata allows. This is an
evaluation candidate, not a claim that Hunyuan works on it. Keep the known-good
environment available until imports, native builds, numeric checks, memory,
performance and unretouched output quality pass.

## Why 2.5.1 currently stays in place

| Finding | Evidence | Confidence |
| --- | --- | --- |
| No repository-wide Torch version lock | Root requirements omit Torch; shape requirements do not specify a version. README's CUDA example pins 2.5.1, while its macOS example installs Torch without a version. | High |
| The installed Torchvision does impose an exact constraint | Local `torchvision 0.20.1` metadata requires `torch==2.5.1`. Upgrade the pair together. | High |
| The existing rasterizer is coupled to the old environment | `otool -L` shows libc10/libtorch/libtorch_cpu/libtorch_python; `LC_RPATH` points to the existing venv's `torch/lib`. | High |
| Existing outputs validate one specific stack | Successful paint used Torch 2.5.1, Torchvision 0.20.1, MLX/MLX-Metal 0.32.4, Diffusers 0.30.0, Transformers 4.46.0, BasicSR 1.4.2, RealESRGAN 0.3.0 and local patches. | High |
| A newer Torch necessarily fails | No such failure was established; the historical handoff stopped dependency installations before replacing the baseline. | Unknown |

Local commands used `importlib.metadata` without loading models, inspected
README/requirements and the rasterizer setup, and inspected the compiled
Mach-O's linked libraries. The external handoff describes previous dependency
resolution attempts; those are historical records, not upgrade test results.

Root `requirements.txt` is not a complete macOS lockfile: for example it pins
PyMeshLab 2022.2.post3, while the working environment has 2025.7.post1, and it
includes CUDA/Blender dependencies. Reinstalling it blindly does not recreate
the successful environment.

## Where MLX is already used

| Component | Current execution | Work needed to eliminate Torch here |
| --- | --- | --- |
| Shape DiT | MLX compute inside a Torch wrapper | Port outer orchestration, conditioning and tensor interfaces; verify MoE parity. |
| Shape VAE decoder | MLX decoder inside a Torch pipeline | Port the remaining interfaces and extraction pipeline; preserve chunking and shape acceptance. |
| Paint inner UNet steps 2–15 | MLX | Already ported compute; Torch remains at input/output boundaries. |
| Paint reference write and first denoise step | PyTorch/MPS | Port custom cache-producing behavior and verify every attention/cache input. |
| Paint VAE, CLIP, DINO | PyTorch/Diffusers/Transformers | Separate model implementations, weight conversion and output-parity validation. |
| RealESRGAN | PyTorch/BasicSR RRDBNet | Port RRDB inference and preprocessing/precision behavior. |
| Rasterization, baking and inpaint | Torch tensors, native CPU/Metal extensions and image/mesh libraries | Redesign tensor/extension interfaces; Metal rasterization alone does not make the pipeline Torch-free. |
| Checkpoint loading | `torch.load` in several paths | Convert trusted weights to an independent format and remove remaining Torch loaders. |

Direct code references:

- `hy3dshape/hy3dshape/pipelines.py` (`_setup_mlx`), `mlx_dit_utils.py`
  (`MLXDispatchDiT`) and `mlx_vae_decoder_utils.py` implement hybrid wrappers.
- `hy3dpaint/mlx_utils.py:153` loads Torch weights; its `MLXDispatchUNet`
  inherits `torch.nn.Module`, keeps reference-write/first-pass execution in
  Torch, and converts MLX results back to Torch.
- `hy3dpaint/utils/multiview_utils.py:39` loads the Diffusers pipeline and DINO;
  `utils/image_super_utils.py:24` constructs the Torch RealESRGAN model.
- `hy3dpaint/custom_rasterizer/custom_rasterizer/render.py:16` imports both
  Torch and the native extension even when Metal rasterization is selected.

The wrappers are direct code evidence; the amount of porting effort is an
engineering inference. No full-MLX prototype or new performance claim is made.

## Upgrade risks and useful validation

1. **Reproducible baseline first.** Preserve the current venv, asset hashes,
   package inventory and the exact local source overlay. The pushed branch
   does not contain all pre-existing dirty files used by the successful run:
   `hy3dpaint/mlx_utils.py`, `hunyuanpaintpbr/unet/attn_processor.py`,
   `custom_rasterizer/setup.py` and `torchvision_fix.py` remain local changes.
   Shape acceptance also used the external eval directory through PYTHONPATH.
2. **Use a separate checkout as well as a new venv.** An editable rasterizer
   build in the original checkout could replace its `.so` and break the
   baseline despite separate Python environments. Build only in the isolated
   checkout, against that environment's Torch libraries.
3. **Resolve the candidate pair explicitly.** Start with Python 3.11 to avoid
   changing Python simultaneously. Use package constraints for the complete
   compatible set; unrestricted dependency installation is not a lockfile.
   Check imports, dependency consistency, Torchvision compatibility shims and
   all model/checkpoint loaders before running GPU work.
4. **Exercise native extensions and attention.** Rebuild the rasterizer and
   test its output. Compare SDPA with contiguous and transposed inputs, the
   project-specific MPS chunking, cache paths and numerical tolerances.
   Recheck `.eval()`, `index_add_` and the documented CPU fallback operations.
5. **Compare raw artifacts.** Initially reuse the approved mesh for paint to
   avoid regenerating shape just to test painting. Capture multiview albedos,
   pre/post-inpaint textures, material/UV checks, red-3 fidelity, runtime and
   peak memory. The manually recolored artifact is not pipeline acceptance.
   A later shape test should write separate outputs with the same settings.
6. **Keep the hardware boundary.** One GPU task at a time on the 48GB device;
   do not run the known unsafe MLX-DiT + MPS-shape-decoder combination.

Upgrading across 2.6 changes the default `torch.load(weights_only=...)`
behavior. Some active loaders already pass it explicitly; other paths do not.
Audit actual checkpoint formats rather than globally disabling the safer
loader. BasicSR's old `functional_tensor` import already needs a shim on the
current Torchvision, so that compatibility issue is not new proof that a newer
Torch cannot work.

Only add intermediate old versions if a failure needs bisection. Installing
2.6 first is not inherently required to evaluate the current stable candidate.

## Upstream evidence

- [Torch 2.14.1 package](https://pypi.org/project/torch/2.14.1/): current stable
  candidate found during this assessment, with macOS arm64 distributions.
- [Torchvision 0.29.1 metadata](https://pypi.org/pypi/torchvision/0.29.1/json):
  requires `torch>=2.14.0`, which allows 2.14.1. This is a lower bound, not an
  exact pin. Its descriptive compatibility table lagging behind that version
  is not by itself a reason to reject the candidate.
- [PyTorch 2.6 release](https://pytorch.org/blog/pytorch2-6/): documents the
  backward-incompatible `torch.load` default change.
- [MLX framework interoperability](https://ml-explore.github.io/mlx/build/html/usage/numpy.html):
  documents array conversion/DLPack and synchronization constraints. Tensor
  interoperability does not translate a Torch model or custom extension.
- [MLX Stable Diffusion example](https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion):
  illustrates a separately implemented diffusion pipeline, not a universal
  Diffusers backend switch.

Potential DLPack/zero-copy optimization is a separate experiment. The current
wrapper copies through NumPy/CPU; unified memory does not establish that
cross-framework transfers or synchronization are free.

## Decision boundary

Recommended next experiment: isolated paired Torch upgrade with the existing
MLX subgraphs retained. Full MLX removal of Torch is a larger, independent
porting project. Neither upgraded-stack quality nor speed is verified yet;
no evidence ties the red-3 color error specifically to Torch 2.5.1 or MLX.
Color-validation follow-up is tracked in `quality-texture-validation.md`.

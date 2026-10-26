# Shape baseline evidence

- Upstream source commit: `82920d643c0dc2f7bfd7255f45f62d386edfe60c`.
- Input: `assets/demo.png`.
- Runtime: Python 3.11.16, PyTorch 2.5.1, Apple Silicon arm64, MPS available.
- Model files were obtained from ModelScope `AI-ModelScope/Hunyuan3D-2.1`; direct Hugging Face access failed through the configured proxy, so the target Hugging Face revision could not be independently verified in this run.
- CPU and MPS both produced non-empty finite meshes and GLB files.
- Blender 5.2.3 LTS successfully imported both GLB files.
- A 2-step CPU smoke run was not counted as success: surface extraction failed because the marching-cubes level was outside the volume range. The accepted baseline uses 10 diffusion steps.
- MPS initially failed at torchvision antialiased bilinear resize. Setting `antialias=False` removed that failure without enabling global CPU fallback.
- This is a PyTorch CPU/MPS baseline only; no MLX code was used in these measurements.

# Quality texture validation follow-up

## Observed failure

The approved reference has black `H/Y/D` letters and a red `3`. The quality
paint run produced an all-black `HY3D` in its base-color texture. The exported
GLB passed structural checks, but that did not establish color fidelity.

The first stage that lost red is unknown: generated multiview albedos and
pre/post-inpaint atlases were not retained. Do not attribute the failure to
MLX, PyTorch 2.5.1, decimation or lighting without an isolating comparison.

## Delivered baseline

- PyMeshLab decimation: 303,234 input faces to 40,000 output faces, with no
  Open3D import required. `tests/test_simplify_mesh.py` covers reduction and
  at/below-budget exports.
- Paint configuration: 6 views, 512 inference resolution, 15 denoise steps,
  MPS first step and MLX subsequent steps; output textures are 2048 square.
- The local run exited 0; painting took 444.4 seconds and the entire process
  476.476 seconds. The original GLB has 24,087 UV-split vertices, 40,000 faces,
  one PBR material and two embedded images.
- Local acceptance used additional pre-existing working-tree changes and
  external model/assets. This commit does not package the complete environment.

## Sample-specific repair

`tools/quality_demo/fix_red3_texture.py` records the user-approved Python repair:

```sh
python tools/quality_demo/fix_red3_texture.py --asset-dir /path/to/quality-assets
```

The directory must contain `shape_quality_textured.glb` and `demo.png`. The
script checks the exact source GLB SHA-256 because its UV region is specific
to this sample. It requires the existing NumPy, Pillow, SciPy and trimesh
dependencies. Run normally, without Python's `-O` option, to retain its
acceptance assertions. It refuses existing output files.

The script writes `shape_quality_textured_red3.glb`, a lossless base-color PNG,
a before/after atlas crop and a JSON verification report. It derives red from
the reference, updates the digit's connected pixels, and verifies identical
decoded pixels outside the mask, mesh geometry, UVs, other embedded images
and source-file hash. The original binary payloads remain in the GLB; the
replacement base-color PNG is appended. This is an artifact repair, not an
automatic color-fidelity fix in the generation pipeline. Input assets and
generated binaries are not included in Git.

## Pending pipeline acceptance work

- [ ] Save reference image, actual model/backend versions, seed, view cameras,
  prompts, precision, inference settings and source/output hashes for each run.
- [ ] Save generated multiview albedos, pre-inpaint atlas and post-inpaint atlas
  so the first stage losing a reference color can be identified.
- [ ] Add a reference-region color check for the red `3`, black `H/Y/D` and
  light sign background. Measure saturation/hue or perceptual color error on
  aligned, visible semantic regions; whole-image average color is insufficient.
  Define tolerances using accepted examples before enforcing a threshold.
- [ ] Keep structural checks (embedded images/material references, valid UVs,
  finite geometry and face budget) separate from color/appearance acceptance.
- [ ] Review front/oblique rendered views to catch back-face projection,
  lettering, seams and lighting/material errors that an atlas crop can miss.
- [ ] On framework/backend upgrades, compare original, unretouched outputs with
  the same input, seed and settings. Report color, shape, runtime, peak memory,
  fallback operations and numeric parity separately. A manually repaired
  red-3 artifact must not count as raw pipeline color acceptance.

These pipeline items remain open; the current commit records them and provides
the deterministic local artifact repair only.

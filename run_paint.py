"""Texture an existing shape mesh with Hunyuan3D-Paint on macOS.

Uses the local paint checkpoint (no Hugging Face snapshot of the 6GB paint
weights). DINO-v2-giant is still fetched by transformers on first run.

Shape generation is not repeated: pass an existing .glb plus the reference image.
"""
import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PAINT_ROOT = str(REPO_ROOT / "hy3dpaint")
REALESRGAN_CKPT = str(REPO_ROOT / "ckpt/RealESRGAN_x4plus.pth")
RASTER_ROOT = os.path.join(PAINT_ROOT, "custom_rasterizer")
LOCAL_MODELS = "/Users/daubert/Github/hunyuan3d-artifacts/Hunyuan3D-2.1-modelscope"
IMAGE = "/Users/daubert/Github/hy3d-mlx-eval/demo.png"
DEFAULT_MESH = "/Users/daubert/Github/hy3d-mlx-eval/shape_profile_mlx_both.glb"
DEFAULT_OUT_OBJ = "/Users/daubert/Github/hy3d-mlx-eval/shape_textured.obj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Texture an existing Hunyuan3D mesh.")
    parser.add_argument("--mesh", default=DEFAULT_MESH)
    parser.add_argument("--output-obj", default=DEFAULT_OUT_OBJ)
    args = parser.parse_args()
    if not Path(args.mesh).is_file():
        parser.error(f"mesh does not exist: {args.mesh}")
    if Path(args.mesh).suffix.lower() != ".glb":
        parser.error(f"mesh must be a .glb file: {args.mesh}")
    if Path(args.output_obj).suffix.lower() != ".obj":
        parser.error(f"output must end in .obj: {args.output_obj}")
    return args


def load_paint_dependencies():
    sys.path.insert(0, PAINT_ROOT)
    sys.path.insert(0, RASTER_ROOT)

    from utils.torchvision_fix import apply_fix
    apply_fix()

    import huggingface_hub

    _orig_snapshot = huggingface_hub.snapshot_download

    def _snapshot(repo_id, *args, **kwargs):
        if repo_id in ("tencent/Hunyuan3D-2.1", LOCAL_MODELS):
            print(f"  using local paint weights: {LOCAL_MODELS}", flush=True)
            return LOCAL_MODELS
        return _orig_snapshot(repo_id, *args, **kwargs)

    huggingface_hub.snapshot_download = _snapshot

    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    return Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline


def main(mesh_path: str, output_obj: str):
    Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline = load_paint_dependencies()
    t0 = time.perf_counter()
    conf = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
    conf.realesrgan_ckpt_path = REALESRGAN_CKPT
    conf.multiview_pretrained_path = LOCAL_MODELS
    conf.dino_ckpt_path = "/Users/daubert/Github/hunyuan3d-artifacts/dinov2-giant"
    print("loading paint pipeline...", flush=True)
    pipe = Hunyuan3DPaintPipeline(conf)
    print(f"loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    t1 = time.perf_counter()
    out = pipe(
        mesh_path=mesh_path,
        image_path=IMAGE,
        output_mesh_path=output_obj,
        use_remesh=True,
        save_glb=True,
    )
    print(f"textured in {time.perf_counter() - t1:.1f}s -> {out}", flush=True)
    glb = output_obj.replace(".obj", ".glb")
    print(f"glb exists={os.path.exists(glb)} size={os.path.getsize(glb) if os.path.exists(glb) else 0}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    main(args.mesh, args.output_obj)

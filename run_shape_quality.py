import argparse
from pathlib import Path
import subprocess
import sys
import time

from quality_shape import ATTEMPTS, attempt_for_resolution, mesh_is_acceptable


ROOT = Path("/Users/daubert/Github/hy3d-mlx-eval")
MODEL_PATH = Path("/Users/daubert/Github/hunyuan3d-artifacts/Hunyuan3D-2.1-modelscope")
IMAGE_PATH = ROOT / "demo.png"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a quality white mesh with safe fallback.")
    parser.add_argument("--child", type=int, metavar="RESOLUTION")
    return parser


def run_child(resolution: int) -> int:
    attempt = attempt_for_resolution(resolution)
    sys.path.insert(0, str(ROOT / "hy3dshape"))
    import torch
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    started = time.perf_counter()
    output_path = ROOT / attempt.output_name
    output_path.unlink(missing_ok=True)
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        str(MODEL_PATH),
        device="mps",
        dtype=torch.float32,
        use_safetensors=False,
        variant="fp16",
        subfolder="hunyuan3d-dit-v2-1",
    )
    mesh = pipe(
        image=str(IMAGE_PATH),
        num_inference_steps=30,
        guidance_scale=5.0,
        generator=torch.Generator(device="cpu").manual_seed(1234),
        octree_resolution=resolution,
        num_chunks=8000,
        enable_pbar=True,
    )[0]
    if not mesh_is_acceptable(mesh):
        print(
            f"rejected resolution={resolution} vertices="
            f"{len(mesh.vertices) if mesh is not None else 0} faces="
            f"{len(mesh.faces) if mesh is not None else 0}",
            flush=True,
        )
        return 2
    mesh.export(output_path)
    print(
        f"accepted resolution={resolution} output={output_path} "
        f"vertices={len(mesh.vertices)} faces={len(mesh.faces)} "
        f"elapsed_s={time.perf_counter() - started:.1f}",
        flush=True,
    )
    return 0


def run_parent() -> int:
    for attempt in ATTEMPTS:
        command = [sys.executable, str(Path(__file__)), "--child", str(attempt.resolution)]
        print(f"starting resolution={attempt.resolution}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode == 0:
            return 0
        print(
            f"failed resolution={attempt.resolution} returncode={completed.returncode}; "
            "trying next fallback",
            flush=True,
        )
    return 1


def main() -> int:
    args = build_parser().parse_args()
    if args.child is None:
        return run_parent()
    try:
        return run_child(args.child)
    except Exception as error:
        print(f"failed resolution={args.child} error={error!r}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

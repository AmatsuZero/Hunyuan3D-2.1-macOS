"""Isolated paint validation; outputs and intermediates always go to a fresh directory.

Run in a fresh process with PYTORCH_ENABLE_MPS_FALLBACK unset for --mode audit.
Placement audit is not a substitute for an Instruments Metal trace.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time


def install_initial_noise(pipe, fixture_path, capture_path):
    """Replay raw noise before the scheduler's scale; capture actual prepared latents."""
    import inspect
    import numpy as np
    import torch

    original = pipe.prepare_latents
    signature = inspect.signature(original)
    fixture = np.load(fixture_path, allow_pickle=False) if fixture_path is not None else None

    def prepare(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        if fixture is not None:
            shape = (values['batch_size'], values['num_channels_latents'],
                     values['height'] // pipe.vae_scale_factor, values['width'] // pipe.vae_scale_factor)
            if fixture.shape != shape:
                raise ValueError(f'initial noise shape {fixture.shape}, expected {shape}')
            expected_dtype = np.dtype(str(values['dtype']).removeprefix('torch.'))
            if fixture.dtype != expected_dtype:
                raise ValueError(f'initial noise dtype {fixture.dtype}, expected {expected_dtype}')
            if not np.isfinite(fixture).all():
                raise ValueError('initial noise must be finite')
            values['latents'] = torch.from_numpy(fixture)
        result = original(*bound.args, **bound.kwargs)
        np.save(capture_path, result.detach().cpu().numpy())
        return result

    pipe.prepare_latents = prepare
    return original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["audit", "benchmark"], default="audit")
    parser.add_argument("--signposts", action="store_true")
    parser.add_argument("--initial-noise", type=Path,
                        help="diagnostic raw-noise .npy fixture, applied before scheduler scaling")
    def non_negative(value):
        number = int(value)
        if number < 0:
            raise argparse.ArgumentTypeError("cache limit must be non-negative")
        return number
    parser.add_argument("--mlx-cache-limit-mib", type=non_negative,
                        help="experimental free-cache budget; unset preserves MLX default")
    args = parser.parse_args()
    if args.mode == "audit" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        parser.error("strict audit requires PYTORCH_ENABLE_MPS_FALLBACK unset or 0 before import")
    if not args.mesh.is_file():
        parser.error("mesh does not exist")
    if args.initial_noise is not None and not args.initial_noise.is_file():
        parser.error("initial noise fixture does not exist")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out = args.output_dir.resolve()
    shutil.copy2(args.mesh, out / "input.glb")
    # The pipeline writes white_mesh_remesh.obj beside its input mesh.
    # Copying the mesh prevents any write into the delivered assets directory.
    sys.path.insert(0, str(args.checkout.resolve()))
    import run_paint
    import torch
    import numpy as np
    import psutil
    from PIL import Image
    spec = importlib.util.spec_from_file_location(
        "execution_audit", Path(__file__).resolve().parents[2] / "hy3dpaint/utils/execution_audit.py")
    audit_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit_module)
    audit = audit_module.ExecutionAudit(required=[], event_path=out / "events.jsonl")
    timings = []
    samples = []
    stop = threading.Event()

    def monitor():
        process = psutil.Process()
        while not stop.wait(1):
            samples.append({"t": time.perf_counter(), "rss": process.memory_info().rss,
                            "available": psutil.virtual_memory().available,
                            "swap_used": psutil.swap_memory().used})

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    report = {"mode": args.mode, "checkout": str(args.checkout.resolve()),
              "clock_anchor": {"wall_unix_s": time.time(), "monotonic_s": time.perf_counter()},
              "torch": torch.__version__, "torch_file": torch.__file__,
              "fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
              "status": "failed", "timings": timings, "memory_samples": samples,
              "trace_acceptance": "pending separate Metal trace inspection",
              "color_acceptance": "pending; known red-3 failure must not be repaired for acceptance"}
    counters = {}
    restored = []

    def synchronize():
        torch.mps.synchronize()
        import mlx.core as mx
        mx.synchronize()

    def save_value(value, stem):
        if isinstance(value, Image.Image):
            value.save(out / f"{stem}.png")
        elif isinstance(value, dict):
            for key, item in value.items():
                save_value(item, f"{stem}_{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                save_value(item, f"{stem}_{index}")
        elif isinstance(value, (torch.Tensor, np.ndarray)):
            array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
            np.save(out / f"{stem}.npy", array)
            preview = array.squeeze()
            if preview.ndim in (2, 3):
                if np.issubdtype(preview.dtype, np.floating):
                    preview = np.clip(preview * 255, 0, 255).astype(np.uint8)
                if preview.shape[-1:] == (1,):
                    preview = preview[..., 0]
                Image.fromarray(preview.astype(np.uint8)).save(out / f"{stem}.png")

    def wrap_stage(obj, name, label, snapshots=False):
        original = getattr(obj, name)

        def measured(*pos, **kw):
            index = counters.get(label, 0)
            counters[label] = index + 1
            synchronize()
            start = time.perf_counter()
            try:
                result = original(*pos, **kw)
                synchronize()
                timings.append({"stage": label, "call": index, "elapsed_s": time.perf_counter() - start,
                                "mps_allocated": torch.mps.current_allocated_memory(),
                                "mps_driver_allocated": torch.mps.driver_allocated_memory()})
                import mlx.core as mx
                timings[-1].update(mlx_active=mx.get_active_memory(), mlx_cache=mx.get_cache_memory(),
                                   mlx_peak=mx.get_peak_memory())
                with (out / "timings.jsonl").open("a") as stream:
                    stream.write(json.dumps(timings[-1]) + "\n")
                print(f"[stage] {label} #{index}: {timings[-1]['elapsed_s']:.3f}s", flush=True)
                if snapshots:
                    save_value(result, f"{label}_{index}")
                return result
            except Exception as exc:
                timings.append({"stage": label, "call": index, "error": repr(exc)})
                raise

        restored.append((obj, name, original))
        setattr(obj, name, measured)

    monitor_thread.start()
    started = time.perf_counter()
    profiler = None
    try:
        audit.require_mps()
        if args.mlx_cache_limit_mib is not None:
            import mlx.core as mx
            previous = mx.set_cache_limit(args.mlx_cache_limit_mib * 1024**2)
            report["mlx_cache_policy"] = {"limit_bytes": args.mlx_cache_limit_mib * 1024**2,
                                          "previous_limit_bytes": previous,
                                          "scope": "this experimental process only"}
        Config, Pipeline = run_paint.load_paint_dependencies()
        config = Config(max_num_view=6, resolution=512)
        config.realesrgan_ckpt_path = str(args.checkout.resolve() / "ckpt/RealESRGAN_x4plus.pth")
        config.multiview_pretrained_path = run_paint.LOCAL_MODELS
        config.dino_ckpt_path = "/Users/daubert/Github/hunyuan3d-artifacts/dinov2-giant"
        pipeline = Pipeline(config)
        synchronize()
        report["load_s"] = time.perf_counter() - started
        dispatch = pipeline._mlx_dispatch
        if dispatch is None:
            raise audit_module.AuditError("MLX dispatch missing: hybrid baseline cannot silently change backend")
        import mlx.core as mx
        if mx.default_device() != mx.gpu:
            raise audit_module.AuditError("MLX default device is not GPU")
        report["mlx_device"] = str(mx.default_device())
        mv = pipeline.models["multiview_model"]
        pipe = mv.pipeline
        original_prepare = install_initial_noise(pipe, args.initial_noise, out / 'initial-latents.npy')
        restored.append((pipe, 'prepare_latents', original_prepare))
        if args.initial_noise is not None:
            import hashlib
            report['initial_noise_fixture'] = {
                'path': str(args.initial_noise.resolve()),
                'sha256': hashlib.sha256(args.initial_noise.read_bytes()).hexdigest(),
                'scope': 'initial diffusion noise only; VAE sampling/conditions are not replayed'}
        if args.mode == "audit":
            targets = {
                "inner_unet.torch": (dispatch._original, "forward"),
                "outer_unet": (pipe.unet, "forward"),
                "vae.encode": (pipe.vae, "encode"),
                "vae.decode": (pipe.vae, "decode"),
                "super_resolution": (pipeline.models["super_model"].upsampler.model, "forward"),
            }
            if pipe.unet.use_dual_stream and pipe.unet.use_ra:
                targets["reference_unet.torch"] = (pipe.unet.unet_dual, "forward")
            if hasattr(mv, "dino_v2"):
                targets["dino.transformer"] = (mv.dino_v2.dino_v2, "forward")
            if not pipe.unet.use_learned_text_clip:
                targets["clip.text"] = (pipe.text_encoder, "forward")
            else:
                report["clip.text"] = "not used: checkpoint uses learned text embeddings"
            report["clip.image"] = "not used by this checkpoint's paint path; DINO provides image features"
            audit.required = set(targets)
            for label, (module, method) in targets.items():
                audit.wrap(label, module, method, cpu_scalar_paths=(".timestep",))
        wrap_stage(dispatch, "_mlx_forward", "mlx_unet")
        # Wrap forward_one, because special __call__ lookup happens on the class.
        wrap_stage(mv, "forward_one", "multiview", snapshots=True)
        wrap_stage(pipeline.view_processor, "bake_view_selection", "cameras", snapshots=False)
        wrap_stage(pipeline.view_processor, "bake_from_multiview", "bake", snapshots=True)
        wrap_stage(pipeline.view_processor, "texture_inpaint", "inpaint", snapshots=True)
        for label, module in [("vae_encode", pipe.vae), ("vae_decode", pipe.vae)]:
            wrap_stage(module, "encode" if label == "vae_encode" else "decode", label)
        wrap_stage(pipeline.models["super_model"].upsampler, "enhance", "enhance", snapshots=True)
        report["module_paths"] = {name: module.__file__ for name, module in sys.modules.items()
                                  if getattr(module, "__file__", None) and name.split(".")[0] in
                                  ("textureGenPipeline", "utils", "hunyuanpaintpbr", "DifferentiableRenderer", "custom_rasterizer", "custom_rasterizer_kernel", "mlx_utils", "run_paint")}
        for name, file in report["module_paths"].items():
            if not Path(file).resolve().is_relative_to(args.checkout.resolve()):
                raise audit_module.AuditError(f"source contamination: {name}: {file}")
        if args.signposts:
            profiler = torch.mps.profiler.profile(mode="interval,event", wait_until_completed=True)
            profiler.__enter__()
        inference_started = time.perf_counter()
        pipeline(mesh_path=str(out / "input.glb"), image_path=run_paint.IMAGE,
                 output_mesh_path=str(out / "textured.obj"), use_remesh=True, save_glb=True)
        synchronize()
        report["inference_s"] = time.perf_counter() - inference_started
        report["mlx_generation_calls"] = dispatch._gen_call_count
        report["mlx_peak_bytes"] = mx.get_peak_memory()
        report["mlx_active_bytes"] = mx.get_active_memory()
        report["stage_calls"] = counters
        if counters.get("mlx_unet") != 14 or dispatch._gen_call_count != 15:
            raise audit_module.AuditError("expected 15 generation steps, including 14 MLX calls")
        if args.mode == "audit":
            audit.validate()
        report["status"] = "pipeline_passed_pending_quality_and_trace"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_s"] = time.perf_counter() - started
        report["stage_calls"] = counters
        if profiler is not None:
            profiler.__exit__(None, None, None)
        stop.set()
        monitor_thread.join(timeout=2)
        for obj, name, original in reversed(restored):
            setattr(obj, name, original)
        audit.close()
        audit.save(out / "audit.json", **report)
        print(json.dumps({k: v for k, v in report.items() if k not in ("memory_samples", "module_paths", "timings")}, indent=2), flush=True)


if __name__ == "__main__":
    main()

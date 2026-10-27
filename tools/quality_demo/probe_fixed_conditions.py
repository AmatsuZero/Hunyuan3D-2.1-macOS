"""Bounded colour diagnosis with fixed condition tensors.

Captures the real paint conditioning tensors and the first denoising steps of
the multiview path, so two Torch versions can be compared boundary by
boundary. Diagnostic only: it never accepts colour quality, never overwrites
delivered assets and never writes into a delivered asset directory.
"""
import argparse
import hashlib
import importlib
import json
import os
import random
import shutil
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace


def _early_fallback(argv):
    for index, value in enumerate(argv):
        if value == "--fallback":
            return argv[index + 1]
        if value.startswith("--fallback="):
            return value.split("=", 1)[1]
    return None


_FALLBACK = _early_fallback(sys.argv)
if _FALLBACK is not None:
    # Must be set before torch is imported.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = _FALLBACK

import numpy as np  # noqa: E402
import torch  # noqa: E402
from einops import rearrange  # noqa: E402


STAGES = ("fixtures", "conditions", "capture", "compare")
CONDITION_NAMES = ("dino_hidden_states", "dino_pixel_values", "ref_latents",
                   "embeds_normal", "embeds_position")
INJECTABLE = ("dino_hidden_states", "ref_latents", "embeds_normal", "embeds_position")
SIGNPOST_STAGES = ("dino", "vae.encode", "reference", "first_step", "all")


class StopCapture(Exception):
    pass


@contextmanager
def trace_scope(label, selected, report):
    """Bound MPS signposts to one named Torch stage, excluding model loading."""
    if selected not in (label, "all"):
        yield
        return
    torch.mps.synchronize()
    start = {"stage": label, "event": "begin", "pid": os.getpid(), "wall_unix_s": time.time(),
             "monotonic_s": time.perf_counter()}
    report.setdefault("trace_markers", []).append(start)
    print("[trace-marker] " + json.dumps(start), flush=True)
    status = "failed"
    try:
        with torch.mps.profiler.profile(mode="interval,event", wait_until_completed=True):
            yield
            torch.mps.synchronize()
        status = "completed"
    finally:
        end = {"stage": label, "event": "end", "pid": os.getpid(), "status": status,
               "wall_unix_s": time.time(), "monotonic_s": time.perf_counter()}
        report["trace_markers"].append(end)
        print("[trace-marker] " + json.dumps(end), flush=True)


def add_checkout_paths(checkout):
    checkout = Path(checkout).resolve()
    paint = checkout / "hy3dpaint"
    for path in (str(paint / "custom_rasterizer"), str(paint), str(checkout)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return checkout


def _runtime_object_source(name, value):
    """Describe the module that supplied a runtime class or processor."""
    if value is None:
        return None
    cls = type(value)
    module_name = cls.__module__
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    source = {
        "name": name,
        "class": f"{module_name}.{cls.__qualname__}",
        "module": module_name,
        "module_file": module_file,
    }
    if module_file and Path(module_file).is_file():
        source["module_sha256"] = hashlib.sha256(Path(module_file).read_bytes()).hexdigest()
    return source


def runtime_sources(pipe, model_dir=None):
    """Record actual UNet and reference-processor provenance.

    Diffusers may execute model custom code from its dynamic module cache,
    which is distinct from the workspace's similarly named implementation.
    Keeping this in every capture report prevents a local-path assumption from
    being mistaken for the code that actually ran.
    """
    sources = {
        "pipeline": _runtime_object_source("pipeline", pipe),
        "outer_unet": _runtime_object_source("outer_unet", getattr(pipe, "unet", None)),
    }
    outer = getattr(pipe, "unet", None)
    inner = getattr(outer, "unet", None)
    dual = getattr(outer, "unet_dual", None)
    sources["inner_unet"] = _runtime_object_source("inner_unet", inner)
    sources["reference_unet"] = _runtime_object_source("reference_unet", dual)
    reference_processor = None
    for module in (inner, dual, outer):
        if module is None or not hasattr(module, "modules"):
            continue
        for candidate in module.modules():
            attention = getattr(candidate, "attn_refview", None)
            processor = getattr(attention, "processor", None) if attention is not None else None
            if processor is not None:
                reference_processor = processor
                break
        if reference_processor is not None:
            break
    sources["reference_processor"] = _runtime_object_source("reference_processor", reference_processor)
    if model_dir is not None:
        expected_root = Path(model_dir).resolve() / "unet"
        for key, filename in (("outer_unet", "modules.py"),
                              ("reference_processor", "attn_processor.py")):
            expected = expected_root / filename
            record = sources[key]
            record["model_source_file"] = str(expected)
            record["model_source_sha256"] = hashlib.sha256(expected.read_bytes()).hexdigest()
            record["matches_model_source"] = (
                record.get("module_sha256") == record["model_source_sha256"]
            )
    return sources


def save_npy(directory, name, value):
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    np.save(Path(directory) / (name + ".npy"), array)
    return {"name": name, "shape": list(array.shape), "dtype": str(array.dtype)}


def load_npy(directory, name):
    path = Path(directory) / (name + ".npy")
    if not path.exists():
        return None
    return np.load(path, allow_pickle=False)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)


def fixed_posterior(posterior, seed):
    """Return a posterior whose sample() uses a version-independent noise tensor.

    Diffusers samples the VAE posterior with the ambient generator, whose MPS
    stream differs between Torch releases. Fixing the noise isolates the VAE
    compute from the sampler.
    """
    noise = np.random.default_rng(seed).standard_normal(tuple(posterior.mean.shape)).astype(np.float32)
    fixed = torch.from_numpy(noise).to(device=posterior.mean.device, dtype=posterior.mean.dtype)

    class Fixed:
        def __init__(self, posterior, fixed):
            self._posterior = posterior
            self._fixed = fixed

        @property
        def mean(self):
            return self._posterior.mean

        @property
        def std(self):
            return self._posterior.std

        @property
        def logvar(self):
            return self._posterior.logvar

        @property
        def parameters(self):
            return self._posterior.parameters

        def sample(self, generator=None):
            return self.mean + self.std * self._fixed

    return Fixed(posterior, fixed)


def compare_arrays(left, right):
    """Compare two arrays and return relative error metrics for a saved pair."""
    if left.shape != right.shape:
        return {"shape_mismatch": [list(left.shape), list(right.shape)]}
    left64 = left.astype(np.float64)
    right64 = right.astype(np.float64)
    difference = left64 - right64
    left_norm = float(np.linalg.norm(left64))
    right_norm = float(np.linalg.norm(right64))
    result = {"max_abs": float(np.abs(difference).max()) if difference.size else 0.0,
              "mean_abs": float(np.abs(difference).mean()) if difference.size else 0.0}
    if left_norm:
        result["nrmse"] = float(np.linalg.norm(difference) / left_norm)
    if left_norm and right_norm:
        result["cosine"] = float(np.dot(left64.ravel(), right64.ravel()) / (left_norm * right_norm))
    return result


def load_condition_inputs(fixtures):
    from PIL import Image

    fixtures = Path(fixtures)
    reference = Image.open(fixtures / "reference.png").convert("RGB")
    normal_maps = [Image.open(fixtures / ("normal_" + str(index) + ".png")) for index in range(6)]
    position_maps = [Image.open(fixtures / ("position_" + str(index) + ".png")) for index in range(6)]
    return reference, normal_maps, position_maps


def control_tensor(images, device):
    """Reproduce hunyuanpaintpbr.pipeline.convert_pil_list_to_tensor for one view batch."""
    views = []
    for image in images:
        array = np.asarray(image, dtype=np.float32) / 255.0
        views.append(torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous().half().to(device))
    return torch.cat(views, dim=0).unsqueeze(0)


# ------------------------------------------------------------------ fixtures


def stage_fixtures(args):
    from PIL import Image
    import trimesh

    from utils.simplify_mesh_utils import remesh_mesh
    from utils.uvwrap_utils import mesh_uv_wrap
    from utils.pipeline_utils import ViewProcessor
    from DifferentiableRenderer.MeshRender import MeshRender
    from textureGenPipeline import Hunyuan3DPaintConfig

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    mesh_copy = out / "input.glb"
    shutil.copy2(args.mesh, mesh_copy)
    remeshed = out / "white_mesh_remesh.obj"
    remesh_mesh(str(mesh_copy), str(remeshed), target_count=40000)

    mesh = trimesh.load(remeshed)
    mesh = mesh_uv_wrap(mesh)
    config = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
    render = MeshRender(default_resolution=config.render_size, texture_size=config.texture_size,
                        bake_mode=config.bake_mode, raster_mode=config.raster_mode)
    render.load_mesh(mesh=mesh)
    view = ViewProcessor(config, render)
    elevs, azims, weights = view.bake_view_selection(
        config.candidate_camera_elevs, config.candidate_camera_azims,
        config.candidate_view_weights, config.max_selected_view_num)
    normal_maps = view.render_normal_multiview(elevs, azims, use_abs_coor=True)
    position_maps = view.render_position_multiview(elevs, azims)

    modes = {"normal": [], "position": []}
    for index, image in enumerate(normal_maps):
        resized = image.resize((config.resolution, config.resolution))
        modes["normal"].append(resized.mode)
        resized.save(out / ("normal_" + str(index) + ".png"))
    for index, image in enumerate(position_maps):
        resized = image.resize((config.resolution, config.resolution))
        if resized.mode == "L":
            resized = resized.point(lambda value: 255 if value > 1 else 0, mode="1")
        modes["position"].append(resized.mode)
        resized.save(out / ("position_" + str(index) + ".png"))

    reference = Image.open(args.reference)
    reference = reference.resize((config.resolution, config.resolution))
    if reference.mode == "RGBA":
        background = Image.new("RGB", reference.size, (255, 255, 255))
        background.paste(reference, mask=reference.getchannel("A"))
        reference = background
    reference = reference.convert("RGB")
    reference.save(out / "reference.png")

    meta = {"stage": "fixtures", "cameras": {"elevs": elevs, "azims": azims, "weights": weights},
            "resolution": config.resolution, "views": len(normal_maps), "modes": modes,
            "mesh": str(Path(args.mesh).resolve()), "reference": str(Path(args.reference).resolve())}
    (out / "fixtures.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


# ---------------------------------------------------------------- conditions


def stage_conditions(args):
    from diffusers import AutoencoderKL
    from hunyuanpaintpbr.unet.modules import Dino_v2

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    device = "mps"
    seed_everything(0)
    reference, normal_maps, position_maps = load_condition_inputs(args.fixtures)

    report = {"stage": "conditions", "torch": torch.__version__,
              "fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
              "fixtures": str(Path(args.fixtures).resolve()), "records": [], "notes": [],
              "trace_stage": args.trace_stage}

    dino = Dino_v2(args.dino_dir).to(torch.float16).to(device)
    processor = dino.dino_processor
    pixel_values = {}

    def processor_call(*call_args, **call_kwargs):
        result = processor(*call_args, **call_kwargs)
        pixel_values["tensor"] = result.pixel_values
        return result

    dino.dino_processor = processor_call
    original_dino_forward = dino.dino_v2.forward

    def dino_forward(*pos, **kw):
        with trace_scope("dino", args.trace_stage, report):
            return original_dino_forward(*pos, **kw)

    dino.dino_v2.forward = dino_forward
    dino_hidden_states = dino(reference)
    dino.dino_v2.forward = original_dino_forward
    dino.dino_processor = processor
    report["records"].append(save_npy(out, "dino_hidden_states", dino_hidden_states))
    if "tensor" in pixel_values:
        report["records"].append(save_npy(out, "dino_pixel_values", pixel_values["tensor"]))

    vae = AutoencoderKL.from_pretrained(args.model_dir, subfolder="vae", torch_dtype=torch.float16,
                                        local_files_only=True, use_safetensors=False).eval().to(device)
    vae.requires_grad_(False)
    scaling_factor = vae.config.scaling_factor

    def encode(tensor, seed):
        with trace_scope("vae.encode", args.trace_stage, report):
            posterior = vae.encode(tensor).latent_dist
        return fixed_posterior(posterior, seed)

    # Reference image path, matching HunyuanPaintPipeline.__call__.
    images_vae = torch.tensor(np.array(reference) / 255.0)
    images_vae = images_vae.unsqueeze(0).permute(0, 3, 1, 2)
    images_vae = images_vae.to(device=device, dtype=torch.float16)
    ref_latents = encode((images_vae - 0.5) * 2.0, 20261027).sample() * scaling_factor
    report["records"].append(save_npy(out, "ref_latents", ref_latents))

    embeds_normal = encode(rearrange(control_tensor(normal_maps, device), "b n c h w -> (b n) c h w"),
                           20261028).sample() * scaling_factor
    embeds_normal = rearrange(embeds_normal, "(b n) c h w -> b n c h w", b=1, n=6)
    report["records"].append(save_npy(out, "embeds_normal", embeds_normal))

    embeds_position = encode(rearrange(control_tensor(position_maps, device), "b n c h w -> (b n) c h w"),
                             20261029).sample() * scaling_factor
    embeds_position = rearrange(embeds_position, "(b n) c h w -> b n c h w", b=1, n=6)
    report["records"].append(save_npy(out, "embeds_position", embeds_position))

    (out / "conditions.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


# ------------------------------------------------------------------- capture


def stage_capture(args):
    from diffusers import UniPCMultistepScheduler
    from hunyuanpaintpbr.pipeline import HunyuanPaintPipeline
    from hunyuanpaintpbr.unet.modules import Dino_v2

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    device = "mps"
    seed_everything(0)
    reference, normal_maps, position_maps = load_condition_inputs(args.fixtures)
    fixture_meta = json.loads((Path(args.fixtures) / "fixtures.json").read_text())
    cameras = fixture_meta.get("cameras", {})

    report = {"stage": "capture", "torch": torch.__version__,
              "fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
              "fixtures": str(Path(args.fixtures).resolve()), "steps": args.steps,
              "stop_after": args.stop_after, "guidance_scale": args.guidance_scale,
              "reference_scale": args.reference_scale, "mva_scale": args.mva_scale,
              "cameras": cameras,
              "records": [], "notes": [], "trace_stage": args.trace_stage}

    pipe = HunyuanPaintPipeline.from_pretrained(args.model_dir, torch_dtype=torch.float16)
    report["runtime_sources"] = runtime_sources(pipe, args.model_dir)
    mismatches = [name for name in ("outer_unet", "reference_processor")
                  if not report["runtime_sources"][name]["matches_model_source"]]
    if mismatches:
        raise RuntimeError(f"Diffusers dynamic module cache differs from model source: {mismatches}")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.set_progress_bar_config(disable=True)
    pipe.eval()
    pipe = pipe.to(device)
    if args.mlx:
        import mlx.core as mx
        from mlx_utils import setup_mlx_dispatch

        if args.mlx_cache_limit_mib is not None:
            previous_cache_limit = mx.set_cache_limit(args.mlx_cache_limit_mib * 1024 * 1024)
            report["mlx_cache_policy"] = {
                "limit_bytes": args.mlx_cache_limit_mib * 1024 * 1024,
                "previous_limit_bytes": previous_cache_limit,
            }

        mlx_dispatch = setup_mlx_dispatch(pipe)
        if mlx_dispatch is None:
            raise RuntimeError("requested MLX dispatch but setup_mlx_dispatch returned None")
        report["mlx_dispatch"] = {"enabled": True, "fp32": bool(getattr(mlx_dispatch, "_use_fp32", False))}

    injected = {}
    if args.inject:
        for name in INJECTABLE:
            value = load_npy(Path(args.inject), name)
            if value is not None:
                tensor = torch.from_numpy(value)
                # encode_images returns (b, n, c, h, w); the reference latents
                # carry a single view, so restore that axis before injection.
                if name == "ref_latents" and tensor.dim() == 4:
                    tensor = tensor.unsqueeze(1)
                injected[name] = tensor
        report["injected"] = sorted(injected)

    if injected:
        order = ["ref_latents", "embeds_normal", "embeds_position"]
        state = {"count": 0}
        original_encode_images = pipe.encode_images

        def encode_images_injected(images):
            index = state["count"]
            state["count"] += 1
            if index < len(order) and order[index] in injected:
                return injected[order[index]].to(device=device, dtype=torch.float16)
            return original_encode_images(images)

        pipe.encode_images = encode_images_injected
    elif args.ambient_posterior:
        report["notes"].append("VAE posterior sampling left on the ambient MPS RNG")
    else:
        original_encode = pipe.vae.encode
        encode_state = {"count": 0}

        def encode_fixed(*call_args, **call_kwargs):
            posterior = original_encode(*call_args, **call_kwargs).latent_dist
            index = encode_state["count"]
            encode_state["count"] += 1
            return SimpleNamespace(latent_dist=fixed_posterior(posterior, 20261027 + index))

        pipe.vae.encode = encode_fixed

    captured = {}
    original_dual = pipe.unet.unet_dual.forward

    def dual_forward(*call_args, **call_kwargs):
        with trace_scope("reference", args.trace_stage, report):
            result = original_dual(*call_args, **call_kwargs)
        captured["reference_write"] = result[0] if isinstance(result, tuple) else result
        if args.trace_stage == "reference":
            raise StopCapture("reference trace completed")
        return result

    pipe.unet.unet_dual.forward = dual_forward

    original_unet = pipe.unet.forward
    original_inner = pipe.unet.unet.forward

    def inner_forward(*call_args, **call_kwargs):
        with trace_scope("first_step", args.trace_stage, report):
            return original_inner(*call_args, **call_kwargs)

    if args.trace_stage:
        pipe.unet.unet.forward = inner_forward

    def unet_forward(*call_args, **call_kwargs):
        result = original_unet(*call_args, **call_kwargs)
        captured.setdefault("noise_pred", []).append(result[0] if isinstance(result, tuple) else result)
        return result

    pipe.unet.forward = unet_forward

    original_step = pipe.scheduler.step
    step_state = {"count": 0}

    def scheduler_step(*call_args, **call_kwargs):
        captured.setdefault("scheduler_model_output", []).append(call_args[0].detach().cpu())
        captured.setdefault("scheduler_sample", []).append(call_args[2].detach().cpu())
        captured.setdefault("scheduler_timestep", []).append(torch.as_tensor(call_args[1]).detach().cpu())
        result = original_step(*call_args, **call_kwargs)
        latents = result[0] if isinstance(result, tuple) else result
        captured.setdefault("latents", []).append(latents)
        step_state["count"] += 1
        if args.stop_after and step_state["count"] >= args.stop_after:
            raise StopCapture("requested step limit reached")
        return result

    pipe.scheduler.step = scheduler_step

    restore = None
    if args.initial_noise:
        from audit_mps import install_initial_noise

        restore = install_initial_noise(pipe, Path(args.initial_noise), out / "initial-latents.npy")

    kwargs = {"generator": torch.Generator(device="cpu").manual_seed(0), "width": 512, "height": 512,
              "num_in_batch": 6, "images_normal": [normal_maps], "images_position": [position_maps],
              "camera_azims": cameras.get("azims", [0] * len(normal_maps)),
              "camera_elevs": cameras.get("elevs", [0] * len(normal_maps))}
    kwargs["mva_scale"] = args.mva_scale
    if "dino_hidden_states" in injected:
        kwargs["dino_hidden_states"] = injected["dino_hidden_states"].to(device=device, dtype=torch.float16)
    else:
        dino = Dino_v2(args.dino_dir).to(torch.float16).to(device)
        kwargs["dino_hidden_states"] = dino(reference)

    if args.wait_for_file:
        gate = Path(args.wait_for_file)
        print(json.dumps({"wait_for_file": str(gate), "pid": os.getpid()}), flush=True)
        while not gate.exists():
            time.sleep(0.1)
        print(json.dumps({"wait_released": str(gate)}), flush=True)

    started = time.perf_counter()
    result = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = pipe([reference], prompt="high quality", num_inference_steps=args.steps, sync_condition=None,
                          guidance_scale=args.guidance_scale, reference_scale=args.reference_scale, **kwargs)
        except StopCapture:
            report["notes"].append("stopped after the selected trace stage" if args.trace_stage == "reference"
                                   else "stopped after the requested step limit")
        report["elapsed_s"] = time.perf_counter() - started
    report["warnings"] = [str(item.message) for item in caught][:200]

    if result is not None and args.stop_after <= 0:
        images = result.images if hasattr(result, "images") else result
        for index, image in enumerate(images):
            image.save(out / ("generated_" + str(index) + ".png"))
        report["generated_images"] = len(images)

    for name, value in captured.items():
        if isinstance(value, list):
            for index, item in enumerate(value):
                report["records"].append(save_npy(out, name + "_" + str(index), item))
        else:
            report["records"].append(save_npy(out, name, value))

    report["scheduler"] = {
        "class": type(pipe.scheduler).__name__,
        "config": dict(pipe.scheduler.config),
        "timesteps": pipe.scheduler.timesteps.detach().cpu().tolist(),
        "sigmas": pipe.scheduler.sigmas.detach().cpu().tolist(),
    }

    (out / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
    if restore is not None:
        pipe.prepare_latents = restore
    print(json.dumps({key: value for key, value in report.items() if key != "warnings"}, indent=2))


# ------------------------------------------------------------------- compare


def stage_compare(args):
    left_dir = Path(args.left)
    right_dir = Path(args.right)
    results = {}
    for path in sorted(left_dir.glob("*.npy")):
        right = load_npy(right_dir, path.stem)
        if right is None:
            continue
        results[path.stem] = compare_arrays(np.load(path, allow_pickle=False), right)
    report = {"stage": "compare", "left": str(left_dir.resolve()), "right": str(right_dir.resolve()),
              "results": results}
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkout", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--model-dir", default="/Users/daubert/Github/hunyuan3d-artifacts/"
                                             "Hunyuan3D-2.1-modelscope/hunyuan3d-paintpbr-v2-1")
    parser.add_argument("--dino-dir", default="/Users/daubert/Github/hunyuan3d-artifacts/dinov2-giant")
    parser.add_argument("--mesh", default="/Users/daubert/Github/hy3d-mlx-eval/shape_quality.glb")
    parser.add_argument("--reference", default="/Users/daubert/Github/hy3d-mlx-eval/demo.png")
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--initial-noise", type=Path)
    parser.add_argument("--inject", type=Path)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--stop-after", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--reference-scale", type=float, default=None)
    parser.add_argument("--mva-scale", type=float, default=1.0)
    parser.add_argument("--ambient-posterior", action="store_true",
                        help="leave VAE posterior sampling unfixed for sensitivity comparison")
    parser.add_argument("--mlx", action="store_true",
                        help="replace inner UNet with the repository MLX dispatch path")
    parser.add_argument("--mlx-cache-limit-mib", type=int, default=None,
                        help="set MLX idle cache limit for capture diagnostics")
    parser.add_argument("--fallback", choices=("0", "1"), default="0")
    parser.add_argument("--trace-stage", choices=SIGNPOST_STAGES,
                        help="bound MPS signposts to one Torch stage for Metal trace correlation")
    parser.add_argument("--wait-for-file", type=Path,
                        help="wait after model setup until this file exists (attach tracing)")
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.wait_for_file and args.trace_stage != "first_step":
        parser.error("--wait-for-file requires --trace-stage first_step")
    if args.trace_stage:
        allowed = {"conditions": ("dino", "vae.encode", "all"),
                   "capture": ("reference", "first_step", "all")}
        if args.trace_stage not in allowed.get(args.stage, ()):
            parser.error("trace stage must match conditions or capture")
        if args.fallback != "0" or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
            parser.error("trace requires fallback disabled before Torch import")
        if args.mlx or (args.stage == "capture" and args.stop_after != 1):
            parser.error("trace capture requires Torch-only dispatch and --stop-after 1")

    if args.stage == "compare":
        stage_compare(args)
        return
    add_checkout_paths(args.checkout)
    if args.stage == "fixtures":
        stage_fixtures(args)
    elif args.stage == "conditions":
        stage_conditions(args)
    else:
        stage_capture(args)


if __name__ == "__main__":
    main()

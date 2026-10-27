"""Layer-wise DINO parity probe for the Torch upgrade colour diagnosis.

Runs the real DINOv2 conditioning model on a chosen device, captures the
patch embeddings, every transformer layer output and the final states, so two
Torch releases can be compared at each boundary. Diagnostic only.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def _early_fallback(argv):
    for index, value in enumerate(argv):
        if value == "--fallback":
            return argv[index + 1]
        if value.startswith("--fallback="):
            return value.split("=", 1)[1]
    return None


_FALLBACK = _early_fallback(sys.argv)
if _FALLBACK is not None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = _FALLBACK

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dino-dir", default="/Users/daubert/Github/hunyuan3d-artifacts/dinov2-giant")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--fallback", choices=("0", "1"), default="0")
    args = parser.parse_args()

    checkout = Path(args.checkout).resolve()
    paint = checkout / "hy3dpaint"
    for path in (str(paint / "custom_rasterizer"), str(paint), str(checkout)):
        if path not in sys.path:
            sys.path.insert(0, path)

    from PIL import Image
    from hunyuanpaintpbr.unet.modules import Dino_v2

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)

    dino = Dino_v2(args.dino_dir).to(torch.float16).to(args.device)
    reference = Image.open(Path(args.fixtures) / "reference.png").convert("RGB")

    backbone = dino.dino_v2
    layers = backbone.encoder.layer
    captured = {}

    def make_hook(index):
        def hook(module, inputs, output):
            captured["layer_" + str(index)] = output[0] if isinstance(output, tuple) else output
        return hook

    def embed_hook(module, inputs, output):
        captured["embeddings"] = output

    handles = [layer.register_forward_hook(make_hook(index)) for index, layer in enumerate(layers)]
    handles.append(backbone.embeddings.register_forward_hook(embed_hook))
    with torch.no_grad():
        states = dino(reference)
    for handle in handles:
        handle.remove()

    records = []
    for name in sorted(captured, key=lambda key: (key != "embeddings", int(key.split("_")[-1]) if key != "embeddings" else 0)):
        value = captured[name]
        array = value.detach().float().cpu().numpy()
        np.save(out / (name + ".npy"), array)
        records.append({"name": name, "shape": list(array.shape),
                        "absmax": float(np.abs(array).max()),
                        "finite": bool(np.isfinite(array).all())})
    final = states.detach().float().cpu().numpy()
    np.save(out / "final.npy", final)

    report = {"device": args.device, "torch": torch.__version__,
              "fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
              "layers": records, "final_shape": list(final.shape),
              "final_absmax": float(np.abs(final).max())}
    (out / "dino_layers.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"device": args.device, "torch": torch.__version__,
                      "layers": len(records), "final_absmax": float(np.abs(final).max())}, indent=2))


if __name__ == "__main__":
    main()

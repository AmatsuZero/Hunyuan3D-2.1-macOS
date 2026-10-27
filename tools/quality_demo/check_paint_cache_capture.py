#!/usr/bin/env python3
"""Check the real 512px paint reference/cache capture contract."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_LAYERS = (
    "down_0_0_0", "down_0_1_0", "down_1_0_0", "down_1_1_0",
    "down_2_0_0", "down_2_1_0", "mid_0_0", "up_1_0_0", "up_1_1_0",
    "up_1_2_0", "up_2_0_0", "up_2_1_0", "up_2_2_0", "up_3_0_0",
    "up_3_1_0", "up_3_2_0",
)
EXPECTED_POSITIONS = {24576: 512, 6144: 256, 1536: 128, 384: 64}


def check_capture(directory):
    directory = Path(directory)
    raw = (directory / "capture.json").read_bytes()
    report = json.loads(raw)
    contract = report["cache_contract"]
    if tuple(sorted(contract["condition_layers"])) != tuple(sorted(EXPECTED_LAYERS)):
        raise ValueError("unexpected reference condition layer keys")
    reference = {}
    for name in EXPECTED_LAYERS:
        value = np.load(directory / f"condition_embed_{name}.npy", mmap_mode="r",
                        allow_pickle=False)
        if value.ndim != 3 or value.shape[0] != 3 or value.shape[1] <= 0 or value.dtype != np.float16:
            raise ValueError(f"invalid reference cache tensor: {name}")
        reference[name] = value
    position_shapes = {}
    if "position_entries" in contract:
        if set(map(int, contract["position_entries"])) != set(EXPECTED_POSITIONS):
            raise ValueError("unexpected position sequence keys")
        for sequence, resolution in EXPECTED_POSITIONS.items():
            if contract["position_resolutions"][str(sequence)] != resolution:
                raise ValueError(f"wrong voxel resolution at sequence {sequence}")
            value = np.load(directory / f"position_voxel_indices_{sequence}.npy",
                            mmap_mode="r", allow_pickle=False)
            if value.shape != (3, sequence, 3) or value.dtype != np.int64:
                raise ValueError(f"invalid position indices at sequence {sequence}")
            position_shapes[str(sequence)] = list(value.shape)
        encoder = np.load(directory / "generation_encoder_hidden_states.npy",
                          mmap_mode="r", allow_pickle=False)
        if encoder.shape != (36, 77, 1024) or encoder.dtype != np.float16:
            raise ValueError("invalid generation encoder hidden states")
    return {
        "path": str(directory.resolve()),
        "capture_sha256": hashlib.sha256(raw).hexdigest(),
        "reference_layers": len(reference),
        "reference_shapes": {name: list(value.shape) for name, value in reference.items()},
        "position_shapes": position_shapes,
        "generation_encoder_shape": contract.get("generation_encoder_shape"),
    }, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--first-step", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite an existing report")
    first, first_reference = check_capture(args.reference)
    second, second_reference = check_capture(args.first_step)
    if not second["position_shapes"]:
        raise ValueError("first-step capture contains no position indices")
    exact = all(np.array_equal(first_reference[name], second_reference[name])
                for name in EXPECTED_LAYERS)
    report = {"reference_capture": first, "first_step_capture": second,
              "reference_cache_exact": exact}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"reference_layers": first["reference_layers"],
                      "position_entries": len(second["position_shapes"]),
                      "reference_cache_exact": exact}, indent=2))


if __name__ == "__main__":
    main()

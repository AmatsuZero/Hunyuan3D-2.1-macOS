from dataclasses import dataclass


BASELINE_VERTICES = 9772


@dataclass(frozen=True)
class Attempt:
    resolution: int
    output_name: str


ATTEMPTS = (
    Attempt(256, "shape_quality.glb"),
    Attempt(192, "shape_quality_o192.glb"),
    Attempt(128, "shape_quality_o128.glb"),
)


def attempt_for_resolution(resolution: int) -> Attempt:
    for attempt in ATTEMPTS:
        if attempt.resolution == resolution:
            return attempt
    raise ValueError(f"unsupported octree resolution: {resolution}")


def mesh_is_acceptable(mesh: object, minimum_vertices: int = BASELINE_VERTICES) -> bool:
    if mesh is None:
        return False
    vertices = getattr(mesh, "vertices", ())
    faces = getattr(mesh, "faces", ())
    return len(vertices) > minimum_vertices and len(faces) > 0

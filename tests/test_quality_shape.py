import unittest

from quality_shape import ATTEMPTS, attempt_for_resolution, mesh_is_acceptable


class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = [None] * vertices
        self.faces = [None] * faces


class QualityShapeTests(unittest.TestCase):
    def test_attempts_are_in_descending_fallback_order(self):
        self.assertEqual(
            [(attempt.resolution, attempt.output_name) for attempt in ATTEMPTS],
            [
                (256, "shape_quality.glb"),
                (192, "shape_quality_o192.glb"),
                (128, "shape_quality_o128.glb"),
            ],
        )

    def test_lookup_rejects_unknown_resolution(self):
        with self.assertRaises(ValueError):
            attempt_for_resolution(64)

    def test_mesh_requires_faces_and_more_than_baseline_vertices(self):
        self.assertTrue(mesh_is_acceptable(Mesh(vertices=9773, faces=1)))
        self.assertFalse(mesh_is_acceptable(Mesh(vertices=9772, faces=1)))
        self.assertFalse(mesh_is_acceptable(Mesh(vertices=9773, faces=0)))
        self.assertFalse(mesh_is_acceptable(None))


if __name__ == "__main__":
    unittest.main()

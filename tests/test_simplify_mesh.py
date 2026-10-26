from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from hy3dpaint.utils.simplify_mesh_utils import mesh_simplify_trimesh


class MeshSimplifyTests(unittest.TestCase):
    def test_reduces_mesh_without_open3d(self):
        source = trimesh.creation.icosphere(subdivisions=3)
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "source.glb"
            output_path = Path(directory) / "remesh.obj"
            source.export(input_path)
            original = input_path.read_bytes()
            with patch.dict(sys.modules, {"open3d": None}):
                mesh_simplify_trimesh(str(input_path), str(output_path), 400)
            result = trimesh.load(output_path, force="mesh")
            self.assertGreater(len(result.faces), 0)
            self.assertLessEqual(len(result.faces), 400)
            self.assertTrue(np.isfinite(result.vertices).all())
            self.assertEqual(input_path.read_bytes(), original)

    def test_preserves_mesh_at_or_below_target(self):
        source = trimesh.creation.box()
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "source.obj"
            source.export(input_path)
            for target in (12, 20):
                for extension in ("obj", "glb"):
                    with self.subTest(target=target, extension=extension):
                        output_path = Path(directory) / f"remesh-{target}.{extension}"
                        mesh_simplify_trimesh(str(input_path), str(output_path), target)
                        result = trimesh.load(output_path, force="mesh")
                        self.assertEqual(len(result.faces), 12)
                        np.testing.assert_allclose(result.bounds, source.bounds)


if __name__ == "__main__":
    unittest.main()

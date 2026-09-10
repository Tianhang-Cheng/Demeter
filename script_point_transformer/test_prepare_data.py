"""CPU tests: python -m unittest discover -s script_point_transformer -p 'test_*.py'."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
import torch

from prepare_data import inverse_distances, make_splits, prepare_sample, select_splits
from run import stage_inference


class PreparationTests(unittest.TestCase):
    def test_targets_against_brute_force(self):
        rng = np.random.default_rng(7)
        points = {0: rng.normal(size=(13, 3)), 4: rng.normal(size=(21, 3)) + 1,
                  9: rng.normal(size=(3, 3)) + 20}
        retained = {key: p[1:] for key, p in points.items()}
        target, threshold = inverse_distances(points, retained)
        spacing, cross = [], []
        for key, query in retained.items():
            # Fewer than ten points: unavailable neighbors contribute zero,
            # leaving the farthest available neighbor as the maximum.
            spacing.extend(np.sort(cdist(query, query), axis=1)[:, :10].max(axis=1))
            cross.extend(cdist(query, np.concatenate([p for k, p in points.items() if k != key])).min(axis=1))
        expected_t = 2 * np.mean(spacing)
        self.assertAlmostEqual(threshold, expected_t)
        np.testing.assert_allclose(target, expected_t / np.maximum(cross, expected_t), rtol=1e-6)
        self.assertTrue(np.any(target < 1))
        self.assertTrue(np.any(target == 1))

    def test_degenerate_spacing_is_rejected(self):
        points = {0: np.zeros((2, 3)), 1: np.ones((2, 3))}
        with self.assertRaisesRegex(ValueError, "threshold"):
            inverse_distances(points, points)

    def test_split_reproducibility_and_no_plant_leakage(self):
        names = [f"{i}_{side}" for i in range(10) for side in ("i", "o")]
        splits = make_splits(names, 0.2, 42)
        self.assertEqual(set(splits), {"train", "test"})
        self.assertEqual(splits, make_splits(list(reversed(names)), 0.2, 42))
        self.assertEqual(sorted(sum(splits.values(), [])), sorted(names))
        assignment = {name: split for split, group in splits.items() for name in group}
        for i in range(10):
            self.assertEqual(assignment[f"{i}_i"], assignment[f"{i}_o"])

    def test_small_split_requires_explicit_choice(self):
        with self.assertRaises(ValueError):
            make_splits(["1_i", "1_o"], 0.15, 1)
        self.assertEqual(make_splits(["1_i"], 0, 1)["train"], ["1_i"])

    def test_released_split_is_reproducible(self):
        preset = json.loads(Path(__file__).with_name("soybean_split.json").read_text())
        self.assertEqual([len(preset[k]) for k in ("train", "test")], [67, 11])
        self.assertEqual(select_splits(sum(preset.values(), []), None, 2025), preset)

    def test_split_file_rejects_plant_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "split.json"
            path.write_text(json.dumps(dict(train=["1_i"], test=["1_o"])))
            with self.assertRaisesRegex(ValueError, "both splits"):
                select_splits(["1_i", "1_o"], None, 2025, path)

    def make_source(self, root):
        folder = root / "source/example"
        (folder / "raw").mkdir(parents=True)
        (folder / "info").mkdir()
        (folder / "info/class.txt").write_text("0 1\n1 0\n2 1\n3 2\n4 3\n")
        (folder / "info/parent.txt").write_text("0 -> -1\n1 -> 0\n2 -> 0\n3 -> 0\n4 -> 0\n")
        # Nonidentity rotation ensures round-trip tests catch transposition errors.
        q = torch.tensor([np.sqrt(0.5), 0, 0, np.sqrt(0.5)], dtype=torch.float32)
        torch.save({"M_quat_0": q}, folder / "graph.pkl")
        rng = np.random.default_rng(3)
        retained_points = []
        for i in range(5):
            xyz = rng.normal(size=(24, 3)) * 0.1 + [i, 1, 2]
            rgb = np.full_like(xyz, 0.5)
            rgb[0] = 0
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
            cloud.colors = o3d.utility.Vector3dVector(rgb)
            o3d.io.write_point_cloud(str(folder / "raw" / f"{i:03}.ply"), cloud)
            retained_points.extend(xyz[1:])
        return folder, np.asarray(retained_points)

    def test_full_sample_schema_labels_and_coordinates(self):
        with tempfile.TemporaryDirectory() as temp:
            folder, points = self.make_source(Path(temp))
            sample, info = prepare_sample(folder)
            self.assertEqual(sample["coord"].shape, (115, 3))
            np.testing.assert_array_equal(sample["semantic_gt5"].numpy(), np.repeat([2, 0, 1, 3, 4], 23))
            np.testing.assert_array_equal(sample["instance_gt"].numpy(), np.repeat(range(5), 23))
            rotation = np.asarray(info["rotation"])
            np.testing.assert_allclose([0, -1, 0] @ rotation, [1, 0, 0], atol=1e-6)
            restored = (sample["coord"].numpy() * info["radius"]) @ rotation.T + info["bbox_center"]
            np.testing.assert_allclose(restored, points, atol=1e-6)
            np.testing.assert_allclose(np.linalg.norm(sample["normal"], axis=1), 1, atol=1e-5)
            self.assertEqual(sample["inv_dists"].dtype, torch.float32)
            self.assertEqual(sample["semantic_gt5"].dtype, torch.int64)

    def test_cli_and_inference_staging(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder, points = self.make_source(root)
            output = root / "dataset"
            command = [sys.executable, str(Path(__file__).with_name("prepare_data.py")),
                       "--source", str(folder.parent), "--output", str(output),
                       "--test-fraction", "0"]
            subprocess.run(command, check=True, capture_output=True, text=True)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertFalse((output / "val").exists())
            self.assertEqual(manifest["splits"]["train"], ["example"])
            sample = torch.load(output / "train/example.pth", weights_only=True)
            stage_inference(output / "train/example.pth", root / "inference")
            staged = torch.load(root / "inference/normalized_pcd.pth", weights_only=True)
            self.assertEqual(set(staged), {"coord", "color", "normal", "scene_id"})
            torch.testing.assert_close(staged["coord"], sample["coord"])
            cloud = o3d.io.read_point_cloud(str(root / "inference/pcd.ply"))
            np.testing.assert_allclose(np.asarray(cloud.points), points, atol=1e-6)
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("Output must be empty", second.stderr)
            before = (output / "train/example.pth").stat().st_mtime_ns
            subprocess.run(command + ["--resume"], check=True, capture_output=True, text=True)
            self.assertEqual((output / "train/example.pth").stat().st_mtime_ns, before)
            mismatch = subprocess.run(command + ["--resume", "--seed", "9"], capture_output=True, text=True)
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn("Resume settings differ", mismatch.stderr)


if __name__ == "__main__":
    unittest.main()

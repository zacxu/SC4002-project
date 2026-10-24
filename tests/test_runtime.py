"""Verify Phase 1's consequential experiment and portability conventions."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from src.runtime import (
    configuration_hash, create_run_directory, is_improvement, load_protocol,
    project_root, relative_project_path, selection_score,
)


class RuntimeProtocolTests(unittest.TestCase):
    def test_portable_paths_and_original_split_contract(self):
        root = project_root()
        protocol = load_protocol(root)
        self.assertEqual(sum(protocol["dataset"]["partition_counts"].values()), 16521)
        self.assertEqual(relative_project_path(root, protocol["paths"]["dataset"]), root / "data/1.0/data/en-US.jsonl")
        with self.assertRaises(ValueError):
            relative_project_path(root, "../outside")
        with self.assertRaises(ValueError):
            relative_project_path(root, str(root / "data"))

    def test_selection_is_unrounded_mean_and_exact_ties_keep_earliest(self):
        self.assertAlmostEqual(selection_score(0.8, 0.6), 0.7)
        self.assertTrue(is_improvement(0.7, None))
        self.assertFalse(is_improvement(0.7, 0.7))
        self.assertTrue(is_improvement(0.7000000000001, 0.7))
        self.assertFalse(is_improvement(0.6, 0.7))

    def test_invalid_metrics_cannot_select_a_checkpoint(self):
        for invalid in (math.nan, math.inf, -0.1, 70.0, True):
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    selection_score(invalid, 0.5)
                with self.assertRaises(ValueError):
                    is_improvement(invalid, 0.5)

    def test_configuration_hash_is_order_independent_and_content_sensitive(self):
        self.assertEqual(configuration_hash({"a": 1, "b": 2}), configuration_hash({"b": 2, "a": 1}))
        self.assertNotEqual(configuration_hash({"seed": 42}), configuration_hash({"seed": 43}))

    def test_run_directories_preserve_configurations_without_overwriting(self):
        protocol = deepcopy(load_protocol(project_root()))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = create_run_directory(root, "stage1", protocol)
            second = create_run_directory(root, "stage1", protocol)
            self.assertNotEqual(first, second)
            metadata = json.loads((first / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["configuration"], protocol)
            self.assertEqual(metadata["config_sha256"], configuration_hash(protocol))
            with self.assertRaises(ValueError):
                create_run_directory(root, "invalid", protocol)

    def test_source_relative_root_does_not_depend_on_working_directory(self):
        expected = project_root()
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                self.assertEqual(project_root(), expected)
                self.assertEqual(load_protocol(project_root())["project"], "SC4002")
            finally:
                os.chdir(previous)

    def test_run_collision_is_an_error_and_preserves_existing_metadata(self):
        protocol = load_protocol(project_root())
        with tempfile.TemporaryDirectory() as temporary, patch("src.runtime.datetime") as clock:
            clock.now.return_value.strftime.return_value = "fixed_timestamp"
            first = create_run_directory(Path(temporary), "stage1", protocol)
            original = (first / "run.json").read_bytes()
            with self.assertRaises(FileExistsError):
                create_run_directory(Path(temporary), "stage1", protocol)
            self.assertEqual((first / "run.json").read_bytes(), original)


class ReproducibilityTests(unittest.TestCase):
    def test_reseeding_reproduces_all_initial_random_streams(self):
        import numpy as np
        import torch
        from src.runtime import configure_reproducibility, dataloader_generator

        runtime = load_protocol(project_root())["runtime"]

        def sample():
            configure_reproducibility(runtime)
            streams = [random.random(), np.random.random(), torch.rand(4),
                       torch.randperm(20, generator=dataloader_generator(runtime["seed"]))]
            if torch.cuda.is_available():
                streams.append(torch.rand(4, device="cuda").cpu())
            return streams

        first, second = sample(), sample()
        self.assertEqual(first[:2], second[:2])
        for left, right in zip(first[2:], second[2:]):
            self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)

    def test_device_auto_and_explicit_cuda_failure(self):
        import torch
        from src.runtime import select_device

        self.assertEqual(select_device("auto").type, "cuda" if torch.cuda.is_available() else "cpu")
        self.assertEqual(select_device("cpu").type, "cpu")
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(select_device("auto").type, "cpu")
            with self.assertRaises(RuntimeError):
                select_device("cuda")


if __name__ == "__main__":
    unittest.main()

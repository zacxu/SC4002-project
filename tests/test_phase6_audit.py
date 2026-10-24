"""Adversarial evidence checks; no training examples or model inference."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_phase6 import inventory, project_path, run, sha256, validate_final, verify_freeze, verify_registered_runs, verify_document_links


class Phase6AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.summary = json.loads((cls.root / "artifacts/phase5/final_stage2/final_summary.json").read_text(encoding="utf-8"))

    def test_incomplete_or_repeated_final_evaluation_is_rejected(self):
        validate_final(self.summary)
        for key, value in (("status", "running"), ("test_model_evaluations", 2),
                           ("model_weights_unchanged", False), ("checkpoint_bytes_unchanged", False),
                           ("stage2_training_and_tuning_closed", False)):
            with self.subTest(key=key):
                modified = deepcopy(self.summary)
                modified[key] = value
                with self.assertRaises(ValueError):
                    validate_final(modified)

    def test_parameter_cap_and_metric_count_disagreements_are_rejected(self):
        for modification in (lambda x: x["parameter_counts"].update(total=20_000_001),
                             lambda x: x["parameter_counts"].update(trainable=x["parameter_counts"]["total"] + 1),
                             lambda x: x["test"]["metrics"].update(intent_accuracy=float("nan")),
                             lambda x: x["test"]["metrics"].update(correct_intents=2458),
                             lambda x: x["test"]["metrics"].update(slot_tp=2112)):
            modified = deepcopy(self.summary)
            modification(modified)
            with self.assertRaises(ValueError):
                validate_final(modified)

    def test_missing_reload_or_incomplete_original_split_is_rejected(self):
        for modification in (lambda x: x["fresh_process_reload"].update(ordered_predictions_equal=False),
                             lambda x: x["fresh_process_reload"].update(absolute_score_difference=0.00001),
                             lambda x: x["dev"].update(num_examples=2032),
                             lambda x: x.update(selected_epoch=x["epochs_trained"] + 1),
                             lambda x: x["test"].update(limited=True)):
            modified = deepcopy(self.summary)
            modification(modified)
            with self.assertRaises(ValueError):
                validate_final(modified)

    def test_freeze_checks_both_original_and_snapshot_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "src/model.py"
            original.parent.mkdir()
            original.write_bytes(b"verified source")
            directory = root / "final"
            snapshot = directory / "snapshot/src/model.py"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_bytes(original.read_bytes())
            checkpoint = root / "selected.pt"
            checkpoint.write_bytes(b"bookkeeping checkpoint bytes")
            manifest = {"snapshot_file_hashes": {"src/model.py": sha256(original)},
                        "source_hashes": {"src/model.py": sha256(original)}}
            manifest_path = directory / "freeze_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            summary = {"freeze_manifest_sha256": sha256(manifest_path), "checkpoint": "selected.pt",
                       "checkpoint_sha256": sha256(checkpoint)}
            self.assertTrue(verify_freeze(root, directory, summary)["all_bytes_match"])
            for target in (original, snapshot, checkpoint, manifest_path):
                with self.subTest(target=str(target)):
                    previous = target.read_bytes()
                    target.write_bytes(previous + b"altered")
                    with self.assertRaises((ValueError, json.JSONDecodeError)):
                        verify_freeze(root, directory, summary)
                    target.write_bytes(previous)

    def test_inventory_covers_evidence_and_excludes_environment_bytecode_and_self(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = ("src/model.py", "artifacts/runs/candidate/history.csv", "artifacts/runs/candidate/learning_curves.png",
                     "data/original.jsonl", "requirements.lock.txt", ".venv/private", "src/__pycache__/model.pyc",
                     "artifacts/phase6/reference/previous_inventory.json")
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"inventory marker")
            rows = inventory(root)
            self.assertEqual({row["path"] for row in rows}, set(names[:5]))
            self.assertTrue(all(row["size_bytes"] == 16 and len(row["sha256"]) == 64 for row in rows))

    def test_path_escape_and_existing_output_fail_before_reading_or_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for value in ("", "../outside", str(root / "absolute.json")):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    project_path(root, value)
            output = root / "existing"
            output.mkdir()
            marker = output / "preserve.txt"
            marker.write_bytes(b"untouched")
            with self.assertRaises(FileExistsError):
                run(root, output)
            self.assertEqual(marker.read_bytes(), b"untouched")

    def test_unregistered_or_missing_research_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def metadata(name):
                directory = root / "artifacts/runs" / name
                directory.mkdir(parents=True)
                (directory / "run.json").write_text(json.dumps({"purpose": "experiment"}), encoding="utf-8")
                return directory
            metadata("registered")
            expected = {"artifacts/runs/registered"}
            self.assertEqual(verify_registered_runs(root, expected), 1)
            extra = metadata("later_unregistered")
            with self.assertRaisesRegex(ValueError, "extra="):
                verify_registered_runs(root, expected)
            with self.assertRaisesRegex(ValueError, "missing="):
                verify_registered_runs(root, expected | {"artifacts/runs/missing"})
            self.assertEqual((extra / "run.json").read_text(), json.dumps({"purpose": "experiment"}))

    def test_document_links_detect_obsolete_placeholders_and_preserve_encoded_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "existing file.txt").write_text("present", encoding="utf-8")
            document = root / "README.md"
            document.write_text("[file](<existing%20file.txt#anchor>) [external](https://example.com) [section](#local)", encoding="utf-8")
            self.assertEqual(verify_document_links(root)["existing_local_links_checked"], 1)
            document.write_text(document.read_text() + " [obsolete](missing_placeholder.md)", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing_placeholder.md"):
                verify_document_links(root)


if __name__ == "__main__":
    unittest.main()

"""Finalization guards; no assessed run or test model evaluation is performed.

Identity fixtures use the original dev records. Temporary checkpoint byte
fixtures test hashing only and are never loaded as models or training data.
"""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from src.acquisition import file_sha256
from src.data import load_massive
from src.final_evaluation import (
    _validate_checkpoint_payload, _validate_complete_result, _validate_completed_run, _weight_fingerprint, load_selection,
)
from src.runtime import configuration_hash, load_protocol, project_root


class FinalizationGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = project_root()
        protocol = load_protocol(cls.root)
        cls.dev = load_massive(cls.root / protocol["paths"]["dataset"])["dev"]
        cls.predictions = tuple(example.gold_prediction() for example in cls.dev)
        cls.ids = [example.record_id for example in cls.dev]

    @staticmethod
    def selection():
        return {"schema_version": 1, "purpose": "stage1_baseline_selection", "selection_split": "dev",
                "test_used_for_selection": False, "selected_run": "artifacts/runs/unit_fixture",
                "checkpoint": "artifacts/runs/unit_fixture/best.pt", "checkpoint_sha256": "a" * 64,
                "selected_epoch": 2, "epochs_trained": 3, "selection_score": 0.5}

    def payload(self):
        return {"purpose": "experiment", "epoch": 2, "selection_score": 0.5,
                "training_config": {"max_epochs": 3},
                "validation": {"num_examples": 2033, "partitions": ["dev"], "limited": False, "max_batches": None},
                "validation_predictions": [prediction.to_dict() for prediction in self.predictions]}

    def result(self):
        return {"num_examples": 2033, "limited": False, "max_batches": None, "partitions": ["dev"],
                "record_ids": list(self.ids), "predictions": self.predictions}

    def write_selection(self, root, selection=None):
        selection = self.selection() if selection is None else selection
        checkpoint = root / selection["checkpoint"]
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"hash-only fixture, never loaded as a model")
        selection["checkpoint_sha256"] = file_sha256(checkpoint)
        path = root / "artifacts/phase4/selection.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(selection), encoding="utf-8")
        return path, checkpoint, selection

    def test_selection_requires_persisted_dev_only_decision(self):
        for changes in ({"purpose": "diagnostic"}, {"selection_split": "test"}, {"test_used_for_selection": True},
                        {"selected_epoch": 0}, {"epochs_trained": 1}, {"selection_score": float("nan")},
                        {"selection_score": True}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                selection = {**self.selection(), **changes}
                path, _, _ = self.write_selection(root, selection)
                with self.assertRaises(ValueError):
                    load_selection(root, path)

    def test_selection_rejects_missing_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "selection.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required"):
                load_selection(root, path)

    def test_selection_hash_is_checked_before_model_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, checkpoint, selection = self.write_selection(root)
            selected, _, run, selected_checkpoint = load_selection(root, path)
            self.assertEqual(selected, selection)
            self.assertEqual(selected_checkpoint, checkpoint)
            self.assertEqual(run, checkpoint.parent)
            checkpoint.write_bytes(checkpoint.read_bytes() + b" changed")
            with self.assertRaisesRegex(ValueError, "checkpoint bytes"):
                load_selection(root, path)

    def test_selection_rejects_path_escape_and_checkpoint_outside_selected_run(self):
        for changes in ({"selected_run": "../outside"}, {"checkpoint": "../outside.pt"},
                        {"selected_run": "artifacts/phase3/diagnostic"},
                        {"checkpoint": "artifacts/runs/another/best.pt"}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path, _, selection = self.write_selection(root)
                selection.update(changes)
                path.write_text(json.dumps(selection), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_selection(root, path)

    def test_diagnostic_checkpoint_is_rejected_even_with_full_dev_count(self):
        payload = self.payload()
        payload["purpose"] = "diagnostic"
        with self.assertRaisesRegex(ValueError, "diagnostic checkpoint"):
            _validate_checkpoint_payload(self.selection(), payload)

    def test_checkpoint_requires_full_dev_and_unlimited_evaluation(self):
        for changes in ({"num_examples": 2032}, {"partitions": ["train"]}, {"limited": True}, {"max_batches": 64}):
            with self.subTest(changes=changes):
                payload = self.payload()
                payload["validation"].update(changes)
                with self.assertRaisesRegex(ValueError, "complete original dev"):
                    _validate_checkpoint_payload(self.selection(), payload)

    def test_checkpoint_rejects_duplicate_missing_ids_and_changed_selection(self):
        payload = self.payload()
        _validate_checkpoint_payload(self.selection(), payload)
        for mode in ("duplicate", "missing", "epoch", "score", "budget"):
            with self.subTest(mode=mode):
                changed = deepcopy(payload)
                if mode == "duplicate":
                    changed["validation_predictions"][1] = changed["validation_predictions"][0]
                elif mode == "missing":
                    changed["validation_predictions"].pop()
                elif mode == "epoch":
                    changed["epoch"] = 1
                elif mode == "score":
                    changed["selection_score"] += 0.001
                else:
                    changed["training_config"]["max_epochs"] = 2
                with self.assertRaises(ValueError):
                    _validate_checkpoint_payload(self.selection(), changed)

    def test_final_result_requires_exact_original_ids_and_order(self):
        _validate_complete_result(self.result(), self.ids, "dev")
        for mode in ("duplicate", "missing", "reordered", "wrong_partition", "limited", "predictions"):
            with self.subTest(mode=mode):
                changed = self.result()
                if mode == "duplicate":
                    changed["record_ids"][1] = changed["record_ids"][0]
                elif mode == "missing":
                    changed["record_ids"].pop()
                elif mode == "reordered":
                    changed["record_ids"] = changed["record_ids"][::-1]
                elif mode == "wrong_partition":
                    changed["partitions"] = ["train"]
                elif mode == "limited":
                    changed["limited"] = True
                else:
                    changed["predictions"] = self.predictions[::-1]
                with self.assertRaises(ValueError):
                    _validate_complete_result(changed, self.ids, "dev")

    def test_weight_fingerprint_detects_mutation_without_changing_weights(self):
        model = torch.nn.Linear(3, 2)
        previous = {name: value.detach().clone() for name, value in model.state_dict().items()}
        initial = _weight_fingerprint(model)
        self.assertEqual(initial, _weight_fingerprint(model))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, previous[name]))
        with torch.no_grad():
            model.weight[0, 0] += 1
        self.assertNotEqual(initial, _weight_fingerprint(model))

    def run_metadata_fixture(self, run):
        """Only bookkeeping fixtures; no trained model or model metrics implied."""
        selection = {**self.selection(), "epochs_trained": 2}
        protocol = {"fixture": "identity-check-only"}
        payload = {"model_config": {"embedding_dim": 256}, "training_config": {"max_epochs": 2},
                   "source_hashes": {"fixture.py": "a" * 64}, "data_sha256": "b" * 64}
        stage_config = {"stage": "stage1", "model": payload["model_config"], "training": payload["training_config"]}
        effective = {"protocol": protocol, "stage_config": stage_config, "purpose": "experiment"}
        metadata = {"purpose": "experiment", "reportable_experiment": True, "effective_configuration": effective,
                    "config_sha256": configuration_hash(effective), "source_hashes": payload["source_hashes"],
                    "dataset_sha256": payload["data_sha256"]}
        summary = {"purpose": "experiment", "test_evaluated": False, "max_train_batches": None, "max_dev_batches": None,
                   "epochs_trained": 2, "best_epoch": 2, "best_score": 0.5}
        experiment = {**summary, "fresh_process_reload": {"passed": True, "fresh_python_process": True,
                      "ordered_predictions_equal": True, "test_evaluated": False, "num_examples": 2033, "selection_score": 0.5}}
        history = [{"epoch": epoch, "purpose": "experiment", "train_num_examples": 11514, "dev_num_examples": 2033,
                    "dev_intent_accuracy": score, "dev_slot_micro_f1": score, "selection_score": score}
                   for epoch, score in ((1, 0.4), (2, 0.5))]
        for name, data in (("run.json", metadata), ("stage_config.json", stage_config),
                           ("training_summary.json", summary), ("experiment_summary.json", experiment)):
            (run / name).write_text(json.dumps(data), encoding="utf-8")
        (run / "history.jsonl").write_text("\n".join(json.dumps(row) for row in history), encoding="utf-8")
        return selection, payload, protocol, history

    def test_completed_run_rejects_incomplete_or_post_test_summaries_and_reload(self):
        for mode in ("valid", "history", "test_used", "partial_reload", "changed_config"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                run = Path(temporary)
                selection, payload, protocol, history = self.run_metadata_fixture(run)
                if mode == "valid":
                    _validate_completed_run(run, selection, payload, protocol)
                    continue
                if mode == "history":
                    (run / "history.jsonl").write_text(json.dumps(history[0]), encoding="utf-8")
                else:
                    name = "stage_config.json" if mode == "changed_config" else "experiment_summary.json"
                    path = run / name
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if mode == "test_used":
                        data["test_evaluated"] = True
                    elif mode == "partial_reload":
                        data["fresh_process_reload"]["num_examples"] = 64
                    else:
                        data["model"]["embedding_dim"] = 128
                    path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    _validate_completed_run(run, selection, payload, protocol)

    def test_completed_run_rejects_later_checkpoint_on_exact_dev_tie(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            selection, payload, protocol, history = self.run_metadata_fixture(run)
            history[0].update(dev_intent_accuracy=0.5, dev_slot_micro_f1=0.5, selection_score=0.5)
            (run / "history.jsonl").write_text("\n".join(json.dumps(row) for row in history), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "earliest best-scoring"):
                _validate_completed_run(run, selection, payload, protocol)


if __name__ == "__main__":
    unittest.main()

"""Stage 2 selection/freeze guards; these tests never evaluate a test model.

Temporary checkpoint bytes and scores below are bookkeeping fixtures, not
training data, trained artifacts, or reportable task measurements.
"""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.acquisition import file_sha256
from src.data import load_massive
from src.runtime import configuration_hash, load_protocol, project_root
from src.stage2_finalization import (
    _validate_checkpoint_payload, _validate_completed_run, _validate_registered_candidates, finalize_stage2, load_selection,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class Stage2FinalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = project_root()
        protocol = load_protocol(root)
        cls.root, cls.protocol = root, protocol
        dev = load_massive(root / protocol["paths"]["dataset"])["dev"]
        cls.predictions = [item.gold_prediction().to_dict() for item in dev]

    def selection(self):
        return {"schema_version": 1, "stage": "stage2", "selection_split": "dev", "test_used_for_selection": False,
                "selected_run": "artifacts/runs/stage2_fixture", "checkpoint": "artifacts/runs/stage2_fixture/best.pt",
                "checkpoint_sha256": "a" * 64, "selected_epoch": 2, "epochs_trained": 2, "selection_score": 0.5,
                "registered_plan": "artifacts/phase5/experiment_plan.json", "registered_plan_sha256": "b" * 64,
                "selected_candidate": "fixture", "candidates": []}

    def selection_file(self, root):
        selection = self.selection()
        checkpoint = root / selection["checkpoint"]
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"bookkeeping only: never loaded as a model")
        selection["checkpoint_sha256"] = file_sha256(checkpoint)
        path = root / "artifacts/phase5/selection.json"
        write(path, selection)
        return selection, path, checkpoint

    def payload(self):
        payload = {"schema_version": 1, "model_class": "JointStage2", "purpose": "experiment", "epoch": 2,
                   "selection_score": 0.5, "stage_config": {"stage": "stage2", "training": {"max_epochs": 2}},
                   "protocol": {"bookkeeping_fixture": True}, "mappings": {}, "char_to_id": {},
                   "validation": {"num_examples": 2033, "partitions": ["dev"], "limited": False, "max_batches": None},
                   "validation_predictions": deepcopy(self.predictions)}
        for key in ("stage_config", "protocol", "mappings", "char_to_id"):
            payload[key + "_sha256"] = configuration_hash(payload[key])
        return payload

    def test_selection_requires_dev_only_stage2_and_complete_budget(self):
        for change in ({"stage": "stage1"}, {"selection_split": "test"}, {"test_used_for_selection": True},
                       {"selected_epoch": 0}, {"epochs_trained": 1}, {"selection_score": float("nan")},
                       {"selection_score": True}, {"registered_plan_sha256": "invalid"}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                selection, path, _ = self.selection_file(root)
                selection.update(change)
                write(path, selection)
                with self.assertRaises(ValueError):
                    load_selection(root, path)

    def test_selection_hash_checked_before_loading_and_path_escapes_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, path, checkpoint = self.selection_file(root)
            self.assertEqual(load_selection(root, path)[0], selection)
            for change in ({"selected_run": "../outside"}, {"checkpoint": "../outside.pt"},
                           {"selected_run": "artifacts/phase5/diagnostic"},
                           {"checkpoint": "artifacts/runs/other/best.pt"}):
                with self.subTest(change=change):
                    write(path, {**selection, **change})
                    with self.assertRaises(ValueError):
                        load_selection(root, path)
            write(path, selection)
            checkpoint.write_bytes(b"changed after persisted selection")
            with self.assertRaisesRegex(ValueError, "checkpoint bytes"):
                load_selection(root, path)

    def test_selection_requires_registered_plan_and_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, path, _ = self.selection_file(root)
            for key in ("candidates", "registered_plan", "registered_plan_sha256", "selected_candidate"):
                changed = dict(selection)
                changed.pop(key)
                write(path, changed)
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "missing required"):
                    load_selection(root, path)

    def test_finalizer_passes_device_name_to_runner_before_any_inference(self):
        """Exercise the real finalizer prelude and stop at its model loader."""
        class StopBeforeModelLoading(Exception):
            pass

        selection = self.selection()
        run = self.root / selection["selected_run"]
        checkpoint = self.root / selection["checkpoint"]
        selection_path = self.root / "artifacts/phase5/selection.json"
        with patch("src.stage2_finalization.load_selection", return_value=(selection, selection_path, run, checkpoint)), \
                patch("src.stage2_finalization.load_protocol", return_value=self.protocol), \
                patch("src.stage2_finalization.configure_reproducibility"), \
                patch("src.stage2_experiment.load_run", side_effect=StopBeforeModelLoading) as loader:
            with self.assertRaises(StopBeforeModelLoading):
                finalize_stage2(self.root, selection_path, "cpu")
            loader.assert_called_once_with(self.root, checkpoint, "cpu")

    def test_checkpoint_rejects_diagnostic_partial_or_other_model(self):
        _validate_checkpoint_payload(self.selection(), self.payload())
        for change in ({"purpose": "diagnostic"}, {"model_class": "JointBaseline"}, {"schema_version": 2}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                _validate_checkpoint_payload(self.selection(), {**self.payload(), **change})
        for change in ({"num_examples": 2032}, {"partitions": ["test"]}, {"limited": True}, {"max_batches": 64}):
            payload = self.payload()
            payload["validation"].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "complete original dev"):
                _validate_checkpoint_payload(self.selection(), payload)

    def test_checkpoint_rejects_missing_duplicate_ids_or_changed_selection(self):
        for mode in ("missing", "duplicate", "epoch", "score", "budget"):
            payload = self.payload()
            if mode == "missing":
                payload["validation_predictions"].pop()
            elif mode == "duplicate":
                payload["validation_predictions"][1] = payload["validation_predictions"][0]
            elif mode == "epoch":
                payload["epoch"] = 1
            elif mode == "score":
                payload["selection_score"] = 0.6
            else:
                payload["stage_config"]["training"]["max_epochs"] = 3
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                _validate_checkpoint_payload(self.selection(), payload)

    def test_checkpoint_hashes_cover_character_map_and_all_configuration(self):
        for key in ("stage_config", "protocol", "mappings", "char_to_id"):
            payload = self.payload()
            payload[key]["unexpected_changed_entry"] = 123
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "identity hash changed"):
                _validate_checkpoint_payload(self.selection(), payload)

    def run_fixture(self, run, *, score=0.5, identity="fixture", config=None, plan_sha="b" * 64,
                    dataset_sha="c" * 64, source_hashes=None):
        config = config or {"stage": "stage2", "model": {"identity": identity}, "training": {"max_epochs": 2}}
        protocol = {"bookkeeping_fixture": True, "paths": {"dataset": "data/fixture.txt"}}
        hashes = {"train": configuration_hash(["original_train_identity_fixture"]),
                  "dev": configuration_hash(["original_dev_identity_fixture"])}
        payload = {"stage_config": config, "protocol": protocol, "source_hashes": source_hashes or {}, "data_sha256": dataset_sha}
        metadata = {"purpose": "experiment", "stage_config": config, "protocol": protocol,
                    "source_hashes": payload["source_hashes"], "dataset_sha256": dataset_sha,
                    "config_sha256": configuration_hash(config), "registered_plan_sha256": plan_sha,
                    "candidate_id": identity, "created_at_utc": "2026-10-21T12:00:01+00:00"}
        summary = {"purpose": "experiment", "test_evaluated": False, "max_train_batches": None, "max_dev_batches": None,
                   "epochs_trained": 2, "best_epoch": 2, "best_score": score, "training_seconds": 1.0}
        experiment = {**summary, "fresh_process_reload": {"passed": True, "fresh_python_process": True,
                      "ordered_predictions_equal": True, "test_evaluated": False, "num_examples": 2033, "selection_score": score}}
        history = [{"epoch": epoch, "purpose": "experiment", "train_num_examples": 11514, "dev_num_examples": 2033,
                    "dev_intent_accuracy": value, "dev_slot_micro_f1": value, "selection_score": value,
                    "train_record_ids_sha256": hashes["train"], "dev_record_ids_sha256": hashes["dev"]}
                   for epoch, value in ((1, score - 0.1), (2, score))]
        for name, data in (("run.json", metadata), ("stage_config.json", config), ("protocol.json", protocol),
                           ("training_summary.json", summary), ("experiment_summary.json", experiment)):
            write(run / name, data)
        (run / "history.jsonl").write_text("\n".join(json.dumps(row) for row in history), encoding="utf-8")
        return {**self.selection(), "selection_score": score}, payload, protocol, hashes, history

    def test_completed_run_requires_full_history_and_original_membership_hashes(self):
        for mode in ("valid", "missing_epoch", "missing_record", "train_identity", "dev_identity", "wrong_score"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                run = Path(temporary)
                selection, payload, protocol, hashes, history = self.run_fixture(run)
                if mode == "valid":
                    _validate_completed_run(run, selection, payload, protocol, hashes)
                    continue
                if mode == "missing_epoch":
                    history.pop()
                elif mode == "missing_record":
                    history[0]["train_num_examples"] -= 1
                elif mode == "train_identity":
                    history[0]["train_record_ids_sha256"] = "f" * 64
                elif mode == "dev_identity":
                    history[0]["dev_record_ids_sha256"] = "f" * 64
                else:
                    history[0]["selection_score"] += 0.01
                (run / "history.jsonl").write_text("\n".join(json.dumps(row) for row in history), encoding="utf-8")
                with self.assertRaises(ValueError):
                    _validate_completed_run(run, selection, payload, protocol, hashes)

    def test_completed_run_rejects_later_epoch_on_tie(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            selection, payload, protocol, hashes, history = self.run_fixture(run)
            history[0].update(dev_intent_accuracy=0.5, dev_slot_micro_f1=0.5, selection_score=0.5)
            (run / "history.jsonl").write_text("\n".join(json.dumps(row) for row in history), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "earliest best-scoring"):
                _validate_completed_run(run, selection, payload, protocol, hashes)

    def test_completed_run_rejects_post_test_summary_failed_reload_and_config_change(self):
        for mode in ("test", "reload", "config", "config_hash", "budget"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                run = Path(temporary)
                selection, payload, protocol, hashes, _ = self.run_fixture(run)
                name = "run.json" if mode in ("config", "config_hash") else "experiment_summary.json"
                data = json.loads((run / name).read_text(encoding="utf-8"))
                if mode == "test":
                    data["test_evaluated"] = True
                elif mode == "reload":
                    data["fresh_process_reload"]["num_examples"] = 64
                elif mode == "config":
                    data["stage_config"]["model"]["identity"] = "changed"
                elif mode == "config_hash":
                    data["config_sha256"] = "f" * 64
                else:
                    data["epochs_trained"] = 1
                write(run / name, data)
                with self.assertRaises(ValueError):
                    _validate_completed_run(run, selection, payload, protocol, hashes)

    def registered_fixture(self, root, *, tie=False):
        dataset = root / "data/fixture.txt"
        dataset.parent.mkdir(parents=True)
        dataset.write_text("bookkeeping fixture; no task examples or model evaluation", encoding="utf-8")
        configs = [{"stage": "stage2", "model": {"identity": identity}, "training": {"max_epochs": 2}}
                   for identity in ("first", "second")]
        registrations = []
        for identity, config in zip(("first", "second"), configs):
            path = root / f"configs/phase5/{identity}.json"
            write(path, config)
            registrations.append({"candidate_id": identity, "configuration": path.relative_to(root).as_posix(),
                                  "configuration_sha256": file_sha256(path), "config_sha256": configuration_hash(config), "max_epochs": 2})
        plan_document = root / "PHASE5_EXPERIMENT_PLAN.md"
        plan_document.write_text("Bookkeeping-only registered plan document fixture.", encoding="utf-8")
        plan = {"stage": "stage2", "registered_at_utc": "2026-10-21T12:00:00+00:00", "candidates": registrations,
                "plan_document": plan_document.relative_to(root).as_posix(), "plan_document_sha256": file_sha256(plan_document)}
        plan_path = root / "artifacts/phase5/experiment_plan.json"
        write(plan_path, plan)
        candidates = []
        for index, (registered, config) in enumerate(zip(registrations, configs)):
            identity = registered["candidate_id"]
            run = root / f"artifacts/runs/stage2_{identity}"
            score = 0.5 if tie else 0.4 + index * 0.1
            _, _, protocol, hashes, _ = self.run_fixture(run, score=score, identity=identity, config=config,
                                          plan_sha=file_sha256(plan_path), dataset_sha=file_sha256(dataset))
            checkpoint = run / "best.pt"
            checkpoint.write_bytes(f"hash-only fixture {identity}; never loaded as a model".encode())
            Path(str(checkpoint) + ".sha256").write_text(file_sha256(checkpoint), encoding="ascii")
            candidates.append({"candidate_id": identity, "configuration": registered["configuration"],
                               "run": run.relative_to(root).as_posix(), "checkpoint_sha256": file_sha256(checkpoint),
                               "best_epoch": 2, "epochs_trained": 2, "selection_score": score})
        winner = candidates[0 if tie else 1]
        selection = {**self.selection(), "registered_plan_sha256": file_sha256(plan_path), "candidates": candidates,
                     "selected_candidate": winner["candidate_id"], "selected_run": winner["run"],
                     "checkpoint": winner["run"] + "/best.pt", "checkpoint_sha256": winner["checkpoint_sha256"],
                     "selection_score": winner["selection_score"]}
        return selection, protocol, hashes, plan_path

    def test_registered_candidates_require_every_candidate_in_registration_order(self):
        for mode in ("valid", "missing", "reorder", "duplicate", "plan_changed"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                selection, protocol, hashes, plan_path = self.registered_fixture(root)
                if mode == "valid":
                    _validate_registered_candidates(root, selection, protocol, hashes)
                    continue
                if mode == "missing":
                    selection["candidates"].pop()
                elif mode == "reorder":
                    selection["candidates"].reverse()
                elif mode == "duplicate":
                    selection["candidates"][1]["run"] = selection["candidates"][0]["run"]
                else:
                    plan_path.write_text(plan_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
                with self.assertRaises(ValueError):
                    _validate_registered_candidates(root, selection, protocol, hashes)

    def test_registered_candidates_recompute_winner_and_resolve_ties_earliest(self):
        for tie in (False, True):
            with self.subTest(tie=tie), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                selection, protocol, hashes, _ = self.registered_fixture(root, tie=tie)
                _validate_registered_candidates(root, selection, protocol, hashes)
                loser = selection["candidates"][1 if tie else 0]
                selection.update(selected_candidate=loser["candidate_id"], selected_run=loser["run"],
                                 checkpoint_sha256=loser["checkpoint_sha256"], selection_score=loser["selection_score"])
                with self.assertRaisesRegex(ValueError, "earliest registered maximum"):
                    _validate_registered_candidates(root, selection, protocol, hashes)

    def test_registered_plan_document_hash_is_required_checked_and_snapshotted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, protocol, hashes, _ = self.registered_fixture(root)
            plan, evidence = _validate_registered_candidates(root, selection, protocol, hashes)
            document = root / plan["plan_document"]
            self.assertIn(document, evidence)
            document.write_text("Plan narrative changed after registration", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "plan document bytes changed"):
                _validate_registered_candidates(root, selection, protocol, hashes)

    def test_registered_candidates_reject_changed_config_checkpoint_or_plan_binding(self):
        for mode in ("config", "checkpoint", "sidecar", "plan_binding", "identity", "timestamp", "budget", "dataset"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                selection, protocol, hashes, _ = self.registered_fixture(root)
                candidate = selection["candidates"][0]
                run = root / candidate["run"]
                if mode == "config":
                    (root / candidate["configuration"]).write_text("{}", encoding="utf-8")
                elif mode == "checkpoint":
                    (run / "best.pt").write_bytes(b"changed")
                elif mode == "sidecar":
                    (run / "best.pt.sha256").write_text("f" * 64, encoding="ascii")
                elif mode == "budget":
                    candidate["epochs_trained"] = 1
                elif mode == "dataset":
                    (root / protocol["paths"]["dataset"]).write_text("changed", encoding="utf-8")
                else:
                    metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
                    key, value = {"plan_binding": ("registered_plan_sha256", "f" * 64),
                                  "identity": ("candidate_id", "other"),
                                  "timestamp": ("created_at_utc", "2026-10-21T11:59:59+00:00")}[mode]
                    metadata[key] = value
                    write(run / "run.json", metadata)
                with self.assertRaises(ValueError):
                    _validate_registered_candidates(root, selection, protocol, hashes)


if __name__ == "__main__":
    unittest.main()

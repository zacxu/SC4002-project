"""Freeze a dev-selected Stage 1 baseline before one reporting-only test pass.

This module never selects a configuration from test results. The caller first
persists ``selection.json`` after comparing completed dev-only experiments.
The exclusive output directory deliberately prevents repeat final test passes.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import torch
from torch.utils.data import DataLoader

from src.acquisition import file_sha256
from src.audit import EVALUATION_PROFILE, PREPROCESSING_PROFILE
from src.data import EXPECTED_COUNTS, FittedMappings, collate_examples, fit_mappings, load_massive, prepare_split, write_predictions
from src.models import enforce_parameter_cap
from src.runtime import configuration_hash, configure_reproducibility, load_protocol, relative_project_path, select_device, selection_score
from src.training import evaluate_model, load_checkpoint


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def load_selection(root: Path, path: Path) -> tuple[dict, Path, Path, Path]:
    """Validate the persisted dev-only selection without loading any model."""
    root = root.resolve()
    path = path.resolve() if path.is_absolute() else relative_project_path(root, path.as_posix())
    if not path.is_relative_to(root):
        raise ValueError("Selection file must remain inside the project")
    selection = _read_json(path)
    required = {"schema_version", "purpose", "selection_split", "test_used_for_selection",
                "selected_run", "checkpoint", "checkpoint_sha256", "selected_epoch",
                "selection_score", "epochs_trained"}
    if not required <= selection.keys():
        raise ValueError(f"Selection is missing required fields: {sorted(required - selection.keys())}")
    if (selection["schema_version"] != 1 or selection["purpose"] != "stage1_baseline_selection"
            or selection["selection_split"] != "dev" or selection["test_used_for_selection"] is not False):
        raise ValueError("Finalization requires an explicit dev-only Stage 1 baseline selection")
    for key in ("selected_epoch", "epochs_trained"):
        if type(selection[key]) is not int or selection[key] < 1:
            raise ValueError(f"Selection {key} must be a positive integer")
    if selection["selected_epoch"] > selection["epochs_trained"]:
        raise ValueError("Selected epoch exceeds the completed training budget")
    score = selection["selection_score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Selection score must be a finite unrounded fraction")
    digest = selection["checkpoint_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("Selection checkpoint SHA-256 is malformed")
    run = relative_project_path(root, selection["selected_run"])
    checkpoint = relative_project_path(root, selection["checkpoint"])
    if not run.is_relative_to(root / "artifacts/runs") or checkpoint != run / "best.pt":
        raise ValueError("Selected checkpoint must be best.pt inside the selected full experiment")
    if not checkpoint.is_file() or file_sha256(checkpoint) != digest:
        raise ValueError("Selected checkpoint bytes do not match the persisted selection")
    return selection, path, run, checkpoint


def _validate_checkpoint_payload(selection: dict, payload: dict) -> None:
    validation = payload.get("validation", {})
    if payload.get("purpose") != "experiment":
        raise ValueError("A diagnostic checkpoint cannot be finalized as the baseline")
    if (validation.get("num_examples") != EXPECTED_COUNTS["dev"] or validation.get("partitions") != ["dev"]
            or validation.get("limited") is not False or validation.get("max_batches") is not None):
        raise ValueError("The selected checkpoint must come from complete original dev evaluation")
    predictions = payload.get("validation_predictions", [])
    ids = [prediction["record_id"] for prediction in predictions]
    if len(ids) != EXPECTED_COUNTS["dev"] or len(set(ids)) != len(ids):
        raise ValueError("Checkpoint dev predictions have missing or duplicate record identities")
    if payload["epoch"] != selection["selected_epoch"] or abs(payload["selection_score"] - selection["selection_score"]) > 1e-12:
        raise ValueError("Selection epoch or score differs from its checkpoint")
    if payload["training_config"]["max_epochs"] != selection["epochs_trained"]:
        raise ValueError("Selection does not record the checkpoint's complete training budget")


def _validate_completed_run(run: Path, selection: dict, payload: dict, protocol: dict) -> tuple[dict, dict, list[dict]]:
    metadata = _read_json(run / "run.json")
    stage_config = _read_json(run / "stage_config.json")
    summary = _read_json(run / "training_summary.json")
    experiment = _read_json(run / "experiment_summary.json")
    history = [json.loads(line) for line in (run / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    if (metadata.get("purpose") != "experiment" or metadata.get("reportable_experiment") is not True
            or configuration_hash(metadata["effective_configuration"]) != metadata["config_sha256"]):
        raise ValueError("Run metadata does not identify a complete reportable experiment")
    effective = metadata["effective_configuration"]
    if effective["protocol"] != protocol or effective["stage_config"] != stage_config or effective["purpose"] != "experiment":
        raise ValueError("Run configuration differs from its persisted protocol or stage settings")
    if (stage_config.get("stage") != "stage1" or stage_config["model"] != payload["model_config"]
            or stage_config["training"] != payload["training_config"] or metadata["source_hashes"] != payload["source_hashes"]
            or metadata["dataset_sha256"] != payload["data_sha256"]):
        raise ValueError("Checkpoint and completed run have different identities")
    for report in (summary, experiment):
        if (report.get("purpose") != "experiment" or report.get("test_evaluated") is not False
                or report.get("max_train_batches") is not None or report.get("max_dev_batches") is not None
                or report["epochs_trained"] != selection["epochs_trained"]
                or report["best_epoch"] != selection["selected_epoch"]
                or abs(report["best_score"] - selection["selection_score"]) > 1e-12):
            raise ValueError("Run summary does not match completed dev-only selection")
    reload = experiment.get("fresh_process_reload", {})
    if (reload.get("passed") is not True or reload.get("fresh_python_process") is not True
            or reload.get("ordered_predictions_equal") is not True or reload.get("test_evaluated") is not False
            or reload.get("num_examples") != EXPECTED_COUNTS["dev"]
            or abs(reload["selection_score"] - selection["selection_score"]) > 1e-12):
        raise ValueError("The completed experiment lacks a successful full-dev fresh-process reload")
    if len(history) != selection["epochs_trained"] or [row["epoch"] for row in history] != list(range(1, len(history) + 1)):
        raise ValueError("Completed run history is incomplete")
    for row in history:
        if (row.get("purpose") != "experiment" or row["train_num_examples"] != EXPECTED_COUNTS["train"]
                or row["dev_num_examples"] != EXPECTED_COUNTS["dev"]
                or abs(selection_score(row["dev_intent_accuracy"], row["dev_slot_micro_f1"]) - row["selection_score"]) > 1e-12):
            raise ValueError("Every epoch must contain complete train/dev measurements")
    best = max(history, key=lambda row: row["selection_score"])
    if best["epoch"] != selection["selected_epoch"] or abs(best["selection_score"] - selection["selection_score"]) > 1e-12:
        raise ValueError("Selection is not the earliest best-scoring dev checkpoint")
    return metadata, summary, history


def _validate_complete_result(result: dict, expected_ids: list[str], partition: str) -> None:
    if (len(expected_ids) != EXPECTED_COUNTS[partition] or len(set(expected_ids)) != len(expected_ids)
            or result["num_examples"] != EXPECTED_COUNTS[partition] or result.get("limited") is not False
            or result.get("max_batches") is not None or result.get("partitions") != [partition]
            or result["record_ids"] != expected_ids):
        raise ValueError(f"Final evaluation must visit every original {partition} record exactly once in source order")
    if [prediction.record_id for prediction in result["predictions"]] != expected_ids:
        raise ValueError(f"Final {partition} predictions have missing, duplicate, or reordered IDs")


def _weight_fingerprint(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str((str(value.dtype), tuple(value.shape))).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _result_data(result: dict) -> dict:
    return {**{key: result[key] for key in ("loss", "intent_loss", "slot_loss", "num_examples", "num_tokens", "num_batches", "seconds", "limited", "max_batches", "partitions")},
            "metrics": result["metrics"].to_dict(), "record_ids": result["record_ids"]}


def finalize_baseline(root: Path, selection_path: Path, device: str = "cuda") -> dict:
    """Run from the fresh CLI process after all Stage 1 dev tuning has ended."""
    root = root.resolve()
    selection, selection_path, run, checkpoint = load_selection(root, selection_path)
    protocol = load_protocol(root, (run / "protocol.json").relative_to(root).as_posix())
    if protocol["preprocessing"] != PREPROCESSING_PROFILE or protocol["evaluation"] != EVALUATION_PROFILE:
        raise ValueError("The shared frozen preprocessing/evaluation convention changed")
    configure_reproducibility(protocol["runtime"])
    torch.set_default_dtype(torch.float32)
    selected_device = select_device(device)
    audit_directory = relative_project_path(root, protocol["paths"]["phase2"])
    audit = _read_json(audit_directory / "data_audit.json")
    dataset_path = relative_project_path(root, protocol["paths"]["dataset"])
    dataset_sha = file_sha256(dataset_path)
    mapping_path = audit_directory / "preprocessing.json"
    if (audit.get("status") != "passed" or dataset_sha != audit["dataset_sha256"]
            or file_sha256(mapping_path) != audit["preprocessing_sha256"]):
        raise ValueError("Source dataset or mappings differ from the passing Phase 2 audit")
    expected_mappings = FittedMappings.from_dict(_read_json(mapping_path))
    model, mappings, payload = load_checkpoint(checkpoint, selected_device, expected_protocol=protocol,
                                               expected_mappings=expected_mappings, expected_data_sha256=dataset_sha)
    _validate_checkpoint_payload(selection, payload)
    metadata, training_summary, history = _validate_completed_run(run, selection, payload, protocol)
    required_sources = {"src/models.py", "src/training.py", "src/experiment.py", "src/data.py", "src/schema.py",
                        "src/metrics.py", "src/runtime.py", "scripts/run_experiment.py", "scripts/verify_checkpoint.py"}
    if not required_sources <= payload["source_hashes"].keys():
        raise ValueError("The checkpoint is missing required trained-source identities")
    source_hashes = dict(payload["source_hashes"])
    for name, expected in source_hashes.items():
        if file_sha256(relative_project_path(root, name)) != expected:
            raise ValueError(f"Trained source identity changed: {name}")
    for name in ("src/data.py", "src/schema.py", "src/metrics.py"):
        if source_hashes[name] != audit["source_hashes"][name]:
            raise ValueError(f"Shared Phase 2 implementation changed: {name}")
    # Capture helper dependencies and launchers beyond the runner's fixed nine
    # trained-source hashes, without changing those existing run identities.
    all_source_files = sorted((*root.joinpath("src").glob("*.py"), *root.joinpath("scripts").glob("*.py")))
    source_hashes.update({path.relative_to(root).as_posix(): file_sha256(path) for path in all_source_files})
    splits = load_massive(dataset_path)
    if fit_mappings(splits["train"]).to_dict() != mappings.to_dict():
        raise ValueError("Checkpoint mappings do not match a fresh fit on original training records only")
    membership_path = audit_directory / "split_membership.json"
    membership = _read_json(membership_path)
    original_ids = {partition: [example.record_id for example in examples] for partition, examples in splits.items()}
    if membership["dataset_sha256"] != dataset_sha or membership["record_ids"] != original_ids:
        raise ValueError("Original split membership differs from Phase 2")
    counts = enforce_parameter_cap(model, protocol["limits"]["total_parameters"])
    if counts != payload["parameter_counts"]:
        raise ValueError("Frozen model parameter count changed")

    output = root / "artifacts/phase4/final_baseline"
    # An incomplete output requires inspection, not another automatic test pass.
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "snapshot"
    source_paths = [relative_project_path(root, name) for name in source_hashes]
    evidence_paths = [selection_path, checkpoint, Path(str(checkpoint) + ".sha256"), mapping_path, membership_path,
                      audit_directory / "data_audit.json"]
    evidence_paths.extend(run / name for name in ("run.json", "protocol.json", "stage_config.json", "training_summary.json",
                          "experiment_summary.json", "history.jsonl", "history.csv", "learning_curves.png", "best_dev_predictions.jsonl", "reload_check.json"))
    evidence_paths.extend(root / name for name in ("requirements.txt", "requirements-cuda.txt", "requirements.lock.txt",
                          "PHASE4_EXPERIMENT_PLAN.md", "artifacts/phase4/experiment_plan.json") if (root / name).is_file())
    frozen_files = {}
    for source in dict.fromkeys(source_paths + evidence_paths):
        name = source.relative_to(root).as_posix()
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        frozen_files[name] = file_sha256(destination)
        if frozen_files[name] != file_sha256(source):
            raise ValueError(f"Snapshot differs from its source: {name}")

    training = payload["training_config"]
    def loader(partition):
        prepared = prepare_split(splits[partition], mappings)
        return DataLoader(prepared, batch_size=training["batch_size"], shuffle=False, drop_last=False,
                          num_workers=0, collate_fn=collate_examples)
    weights = {"intent_weight": training.get("intent_loss_weight", 1.0), "slot_weight": training.get("slot_loss_weight", 1.0)}
    before_weights = _weight_fingerprint(model)
    dev = evaluate_model(model, loader("dev"), mappings, selected_device, **weights)
    _validate_complete_result(dev, original_ids["dev"], "dev")
    if [prediction.to_dict() for prediction in dev["predictions"]] != payload["validation_predictions"]:
        raise ValueError("Fresh final evaluation changed ordered checkpoint dev predictions")
    score = selection_score(dev["metrics"].intent_accuracy, dev["metrics"].slot_micro_f1)
    score_difference = abs(score - selection["selection_score"])
    if score_difference > 1e-12:
        raise ValueError("Fresh final evaluation failed to reproduce the unrounded selected dev score")
    _write_json(output / "dev_evaluation.json", _result_data(dev))
    write_predictions(output / "dev_predictions.jsonl", dev["predictions"])
    if file_sha256(selection_path) != frozen_files[selection_path.relative_to(root).as_posix()]:
        raise ValueError("Dev-only selection changed while freezing")
    if file_sha256(checkpoint) != selection["checkpoint_sha256"] or _weight_fingerprint(model) != before_weights:
        raise ValueError("Evaluation changed selected checkpoint bytes or model weights")

    manifest = {"schema_version": 1, "stage": "stage1", "status": "frozen_before_first_test_forward",
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "selection": selection,
                "selection_file": selection_path.relative_to(root).as_posix(),
                "selection_file_sha256": frozen_files[selection_path.relative_to(root).as_posix()],
                "checkpoint_sha256": selection["checkpoint_sha256"], "model_weights_sha256": before_weights,
                "dataset_sha256": dataset_sha, "mappings_sha256": payload["mappings_sha256"],
                "protocol_sha256": payload["protocol_sha256"], "training_config_sha256": payload["training_config_sha256"],
                "source_hashes": source_hashes, "snapshot_file_hashes": frozen_files, "parameter_counts": counts,
                "fresh_dev_reload": {"ordered_predictions_equal": True, "selection_score": score,
                                     "absolute_score_difference": score_difference, "num_examples": dev["num_examples"]},
                "stage1_training_and_tuning_closed": True,
                "test_evaluation_started": False,
                "test_policy": "One complete test pass for final reporting only. No subsequent Stage 1 training/tuning or Stage 2 diagnosis from these test outputs.",
                "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
                                "device": str(selected_device), "gpu": torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else None,
                                "fresh_cli_process": True, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}}
    _write_json(output / "freeze_manifest.json", manifest)
    reporting = output / "reporting_only"
    reporting.mkdir()
    _write_json(reporting / "test_started.json", {"started_at_utc": datetime.now(timezone.utc).isoformat(),
                "freeze_manifest_sha256": file_sha256(output / "freeze_manifest.json"), "purpose": "final_reporting_only"})
    # No test record has reached the model before the manifest exists on disk.
    test = evaluate_model(model, loader("test"), mappings, selected_device, allow_test=True, **weights)
    _validate_complete_result(test, original_ids["test"], "test")
    if file_sha256(checkpoint) != selection["checkpoint_sha256"] or _weight_fingerprint(model) != before_weights:
        raise ValueError("Final test evaluation changed selected checkpoint bytes or model weights")
    _write_json(reporting / "test_evaluation.json", {"purpose": "final_reporting_only", **_result_data(test)})
    write_predictions(reporting / "test_predictions.jsonl", test["predictions"])
    rows = []
    for partition, result in (("dev", dev), ("test", test)):
        metrics = result["metrics"]
        rows.append({"stage": "Stage 1", "split": partition, "intent_accuracy_percent": f"{100 * metrics.intent_accuracy:.2f}",
                     "slot_micro_f1_percent": f"{100 * metrics.slot_micro_f1:.2f}", "total_parameters": counts["total"],
                     "trainable_parameters": counts["trainable"], "selected_epoch": selection["selected_epoch"],
                     "epochs_trained": selection["epochs_trained"], "num_examples": result["num_examples"]})
    _write_json(output / "baseline_results.json", {"schema_version": 1, "display_scale": "percent_2dp", "rows": rows,
                "full_precision": {"dev": dev["metrics"].to_dict(), "test": test["metrics"].to_dict()}})
    with (output / "baseline_results.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"schema_version": 1, "status": "complete", "stage": "stage1", "output_directory": str(output),
               "selected_run": selection["selected_run"], "checkpoint": selection["checkpoint"],
               "checkpoint_sha256": selection["checkpoint_sha256"], "frozen_checkpoint": str(snapshot / selection["checkpoint"]),
               "freeze_manifest_sha256": file_sha256(output / "freeze_manifest.json"), "source_hashes": source_hashes,
               "protocol_sha256": payload["protocol_sha256"], "training_config_sha256": payload["training_config_sha256"],
               "dataset_sha256": dataset_sha, "mappings_sha256": payload["mappings_sha256"], "parameter_counts": counts,
               "epochs_trained": selection["epochs_trained"], "selected_epoch": selection["selected_epoch"],
               "selection_score": score, "training_seconds": training_summary["training_seconds"],
               "fresh_process_reload": {"passed": True, "ordered_predictions_equal": True, "absolute_score_difference": score_difference,
                                        "num_examples": dev["num_examples"], "fresh_python_process": True},
               "dev": _result_data(dev), "test": {key: value for key, value in _result_data(test).items() if key != "record_ids"},
               "test_model_evaluations": 1, "test_evaluation_finished": True, "model_weights_unchanged": True,
               "checkpoint_bytes_unchanged": True, "stage1_training_and_tuning_closed": True,
               "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Frozen baseline final results. Test outputs are reporting only; Stage 2 design and error analysis must use development data."}
    _write_json(output / "final_summary.json", summary)
    return summary

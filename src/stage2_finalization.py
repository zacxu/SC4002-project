"""Freeze a fully trained, development-selected Stage 2 joint system.

All registered candidates must have complete original train/dev histories.
Selection and source snapshots are persisted before the first final test
forward pass. The exclusive destination intentionally refuses repeat tests.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys

import torch

from src.acquisition import file_sha256
from src.audit import EVALUATION_PROFILE, PREPROCESSING_PROFILE
from src.data import EXPECTED_COUNTS, FittedMappings, fit_mappings, load_massive, prepare_split, write_predictions
from src.final_evaluation import _read_json, _result_data, _validate_complete_result, _weight_fingerprint, _write_json
from src.models import enforce_parameter_cap
from src.runtime import configuration_hash, configure_reproducibility, load_protocol, relative_project_path, select_device, selection_score


def _valid_score(value) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and 0 <= value <= 1)


def _check_digest(digest: str, name: str) -> None:
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{name} SHA-256 is malformed")


def load_selection(root: Path, path: Path) -> tuple[dict, Path, Path, Path]:
    """Require a persisted dev-only Stage 2 selection and matching bytes."""
    root = root.resolve()
    path = path.resolve() if path.is_absolute() else relative_project_path(root, path.as_posix())
    if not path.is_relative_to(root):
        raise ValueError("Selection must remain inside the project")
    value = _read_json(path)
    required = {"schema_version", "stage", "selection_split", "test_used_for_selection", "selected_run",
                "checkpoint", "checkpoint_sha256", "selected_epoch", "epochs_trained", "selection_score",
                "registered_plan", "registered_plan_sha256", "selected_candidate", "candidates"}
    if not required <= value.keys():
        raise ValueError(f"Selection is missing required fields: {sorted(required - value.keys())}")
    if (value["schema_version"] != 1 or value["stage"] != "stage2" or value["selection_split"] != "dev"
            or value["test_used_for_selection"] is not False):
        raise ValueError("Finalization requires an explicit dev-only Stage 2 selection")
    if any(type(value[key]) is not int or value[key] < 1 for key in ("selected_epoch", "epochs_trained")):
        raise ValueError("Selected epoch and training budget must be positive integers")
    if value["selected_epoch"] > value["epochs_trained"] or not _valid_score(value["selection_score"]):
        raise ValueError("Invalid selected epoch or unrounded selection score")
    _check_digest(value["checkpoint_sha256"], "Checkpoint")
    _check_digest(value["registered_plan_sha256"], "Registered plan")
    run = relative_project_path(root, value["selected_run"])
    checkpoint = relative_project_path(root, value["checkpoint"])
    if not run.is_relative_to(root / "artifacts/runs") or checkpoint != run / "best.pt":
        raise ValueError("Selected checkpoint must be best.pt inside its full experiment")
    if not checkpoint.is_file() or file_sha256(checkpoint) != value["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint bytes differ from the persisted selection")
    return value, path, run, checkpoint


def _validate_checkpoint_payload(selection: dict, payload: dict) -> None:
    if payload.get("schema_version") != 1 or payload.get("model_class") != "JointStage2" or payload.get("purpose") != "experiment":
        raise ValueError("Only a complete Stage 2 experiment checkpoint can be finalized")
    validation = payload.get("validation", {})
    if (validation.get("num_examples") != EXPECTED_COUNTS["dev"] or validation.get("partitions") != ["dev"]
            or validation.get("limited") is not False or validation.get("max_batches") is not None):
        raise ValueError("Checkpoint must come from complete original dev evaluation")
    ids = [row["record_id"] for row in payload.get("validation_predictions", [])]
    if len(ids) != EXPECTED_COUNTS["dev"] or len(set(ids)) != len(ids):
        raise ValueError("Checkpoint dev predictions have missing or duplicate identities")
    if (payload["epoch"] != selection["selected_epoch"] or not _valid_score(payload["selection_score"])
            or abs(payload["selection_score"] - selection["selection_score"]) > 1e-12):
        raise ValueError("Selection epoch or score differs from its checkpoint")
    if payload["stage_config"]["training"]["max_epochs"] != selection["epochs_trained"]:
        raise ValueError("Selection does not record the checkpoint's full training budget")
    for key in ("stage_config", "protocol", "mappings", "char_to_id"):
        if configuration_hash(payload[key]) != payload.get(f"{key}_sha256"):
            raise ValueError(f"Checkpoint {key} identity hash changed")


def _validate_completed_run(run: Path, selection: dict, payload: dict | None, protocol: dict,
                            expected_id_hashes: dict[str, str]) -> tuple[dict, dict, list[dict]]:
    """Validate full histories and reload evidence, including split identities."""
    metadata = _read_json(run / "run.json")
    config = _read_json(run / "stage_config.json")
    saved_protocol = _read_json(run / "protocol.json")
    summary = _read_json(run / "training_summary.json")
    experiment = _read_json(run / "experiment_summary.json")
    history = [json.loads(line) for line in (run / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    if (metadata.get("purpose") != "experiment" or config.get("stage") != "stage2"
            or metadata.get("stage_config") != config or metadata.get("protocol") != protocol or saved_protocol != protocol
            or metadata.get("config_sha256") != configuration_hash(config)):
        raise ValueError("Run metadata differs from its saved Stage 2 configuration or protocol")
    if config["training"]["max_epochs"] != selection["epochs_trained"]:
        raise ValueError("Completed run does not match the registered epoch budget")
    if payload is not None and (payload["stage_config"] != config or payload["protocol"] != protocol
            or metadata.get("source_hashes") != payload["source_hashes"]
            or metadata.get("dataset_sha256") != payload["data_sha256"]):
        raise ValueError("Checkpoint and completed run have different identities")
    for report in (summary, experiment):
        if (report.get("purpose") != "experiment" or report.get("test_evaluated") is not False
                or report.get("max_train_batches") is not None or report.get("max_dev_batches") is not None
                or report.get("epochs_trained") != selection["epochs_trained"]
                or report.get("best_epoch") != selection["selected_epoch"]
                or not _valid_score(report.get("best_score"))
                or abs(report["best_score"] - selection["selection_score"]) > 1e-12):
            raise ValueError("Run summary does not match a completed dev-only selection")
    reload = experiment.get("fresh_process_reload", {})
    if (reload.get("passed") is not True or reload.get("fresh_python_process") is not True
            or reload.get("ordered_predictions_equal") is not True or reload.get("test_evaluated") is not False
            or reload.get("num_examples") != EXPECTED_COUNTS["dev"]
            or not _valid_score(reload.get("selection_score"))
            or abs(reload["selection_score"] - selection["selection_score"]) > 1e-12):
        raise ValueError("Completed experiment lacks a successful full-dev fresh-process reload")
    if len(history) != selection["epochs_trained"] or [row["epoch"] for row in history] != list(range(1, len(history) + 1)):
        raise ValueError("Completed run history is incomplete")
    for row in history:
        if (row.get("purpose") != "experiment" or row.get("train_num_examples") != EXPECTED_COUNTS["train"]
                or row.get("dev_num_examples") != EXPECTED_COUNTS["dev"]
                or row.get("train_record_ids_sha256") != expected_id_hashes["train"]
                or row.get("dev_record_ids_sha256") != expected_id_hashes["dev"]
                or not _valid_score(row.get("selection_score"))
                or abs(selection_score(row["dev_intent_accuracy"], row["dev_slot_micro_f1"]) - row["selection_score"]) > 1e-12):
            raise ValueError("Every epoch must contain complete original train/dev identities and measurements")
    best = max(history, key=lambda row: row["selection_score"])
    if best["epoch"] != selection["selected_epoch"] or abs(best["selection_score"] - selection["selection_score"]) > 1e-12:
        raise ValueError("Selection is not the earliest best-scoring dev checkpoint")
    return metadata, summary, history


def _validate_registered_candidates(root: Path, selection: dict, protocol: dict,
                                    expected_id_hashes: dict[str, str]) -> tuple[dict, list[Path]]:
    """Require every pre-registered candidate, then recompute the dev winner."""
    plan_path = relative_project_path(root, selection["registered_plan"])
    if file_sha256(plan_path) != selection["registered_plan_sha256"]:
        raise ValueError("Registered experiment plan bytes changed")
    plan = _read_json(plan_path)
    plan_document = relative_project_path(root, plan.get("plan_document", ""))
    _check_digest(plan.get("plan_document_sha256"), "Registered plan document")
    if file_sha256(plan_document) != plan["plan_document_sha256"]:
        raise ValueError("Registered plan document bytes changed")
    registrations = plan.get("candidates", [])
    candidates = selection["candidates"]
    if plan.get("stage") != "stage2" or not registrations or len(candidates) != len(registrations):
        raise ValueError("Selection must account for every registered Stage 2 candidate")
    registered_ids = [row["candidate_id"] for row in registrations]
    if len(set(registered_ids)) != len(registered_ids) or [row["candidate_id"] for row in candidates] != registered_ids:
        raise ValueError("Candidate identities/order differ from the registered plan")
    if len({row["run"] for row in candidates}) != len(candidates):
        raise ValueError("Each registered candidate must refer to a distinct completed experiment")
    selected_metadata = _read_json(relative_project_path(root, selection["selected_run"]) / "run.json")
    expected_sources = selected_metadata.get("source_hashes", {})
    evidence = [plan_path, plan_document]
    try:
        registered_at = datetime.fromisoformat(plan["registered_at_utc"])
        if registered_at.tzinfo is None:
            raise ValueError("Registration time lacks a timezone")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Registered plan requires an unambiguous registration timestamp") from error
    for registered, candidate in zip(registrations, candidates):
        if not _valid_score(candidate.get("selection_score")):
            raise ValueError("Candidate selection scores must be unrounded finite fractions")
        config_path = relative_project_path(root, registered["configuration"])
        if (candidate.get("configuration") != registered["configuration"]
                or file_sha256(config_path) != registered["configuration_sha256"]
                or configuration_hash(_read_json(config_path)) != registered["config_sha256"]):
            raise ValueError("Candidate configuration differs from its pre-registered bytes")
        run = relative_project_path(root, candidate["run"])
        if not run.is_relative_to(root / "artifacts/runs"):
            raise ValueError("Registered candidates must be full experiment runs")
        checkpoint = run / "best.pt"
        if (file_sha256(checkpoint) != candidate["checkpoint_sha256"]
                or Path(str(checkpoint) + ".sha256").read_text(encoding="ascii").strip() != candidate["checkpoint_sha256"]):
            raise ValueError("Candidate checkpoint bytes differ from the selection ledger")
        if candidate["epochs_trained"] != registered["max_epochs"]:
            raise ValueError("Candidate did not complete its registered epoch budget")
        if _read_json(config_path) != _read_json(run / "stage_config.json"):
            raise ValueError("Completed candidate used a different registered configuration")
        candidate_selection = {"selected_epoch": candidate["best_epoch"], "epochs_trained": candidate["epochs_trained"],
                               "selection_score": candidate["selection_score"]}
        metadata, _, _ = _validate_completed_run(run, candidate_selection, None, protocol, expected_id_hashes)
        if (metadata.get("registered_plan_sha256") != selection["registered_plan_sha256"]
                or metadata.get("candidate_id") != registered["candidate_id"]
                or metadata.get("source_hashes") != expected_sources):
            raise ValueError("Completed candidate does not refer to the registered plan identity")
        try:
            created_at = datetime.fromisoformat(metadata["created_at_utc"])
            if created_at.tzinfo is None or created_at < registered_at:
                raise ValueError("Training preceded registration")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Every candidate must have been registered before training") from error
        dataset_path = relative_project_path(root, protocol["paths"]["dataset"])
        if metadata.get("dataset_sha256") != file_sha256(dataset_path):
            raise ValueError("Registered candidate used a different source dataset")
        for name, digest in metadata.get("source_hashes", {}).items():
            if file_sha256(relative_project_path(root, name)) != digest:
                raise ValueError(f"Registered candidate trained source changed: {name}")
        evidence.append(config_path)
        evidence.extend(run / name for name in ("run.json", "stage_config.json", "protocol.json", "training_summary.json",
                                                "experiment_summary.json", "history.jsonl"))
    winner = max(candidates, key=lambda row: row["selection_score"])
    if (winner["candidate_id"] != selection["selected_candidate"] or winner["run"] != selection["selected_run"]
            or winner["checkpoint_sha256"] != selection["checkpoint_sha256"]
            or winner["best_epoch"] != selection["selected_epoch"] or winner["epochs_trained"] != selection["epochs_trained"]
            or abs(winner["selection_score"] - selection["selection_score"]) > 1e-12):
        raise ValueError("Selected model is not the earliest registered maximum development score")
    return plan, evidence


def finalize_stage2(root: Path, selection_path: Path, device: str = "cuda") -> dict:
    """Run in a fresh deterministic process after all Stage 2 tuning ends."""
    from src.stage2_experiment import SOURCE_FILES, evaluate, load_run, make_loader

    root = root.resolve()
    selection, selection_path, run, checkpoint = load_selection(root, selection_path)
    protocol = load_protocol(root, (run / "protocol.json").relative_to(root).as_posix())
    if protocol["preprocessing"] != PREPROCESSING_PROFILE or protocol["evaluation"] != EVALUATION_PROFILE:
        raise ValueError("Shared frozen preprocessing/evaluation conventions changed")
    configure_reproducibility(protocol["runtime"])
    torch.set_default_dtype(torch.float32)
    selected_device = select_device(device)
    audit_directory = relative_project_path(root, protocol["paths"]["phase2"])
    audit = _read_json(audit_directory / "data_audit.json")
    dataset = relative_project_path(root, protocol["paths"]["dataset"])
    dataset_sha = file_sha256(dataset)
    mapping_path = audit_directory / "preprocessing.json"
    if (audit.get("status") != "passed" or dataset_sha != audit["dataset_sha256"]
            or file_sha256(mapping_path) != audit["preprocessing_sha256"]):
        raise ValueError("Source data or mappings differ from the passing Phase 2 audit")
    expected_mappings = FittedMappings.from_dict(_read_json(mapping_path))
    model, mappings, char_to_id, payload = load_run(root, checkpoint, str(selected_device))
    _validate_checkpoint_payload(selection, payload)
    if mappings.to_dict() != expected_mappings.to_dict() or payload["protocol"] != protocol or payload["data_sha256"] != dataset_sha:
        raise ValueError("Selected model differs from the shared audited data/protocol")
    if payload["char_to_id"] != char_to_id:
        raise ValueError("Loaded character mapping differs from the selected checkpoint")
    splits = load_massive(dataset)
    if fit_mappings(splits["train"]).to_dict() != mappings.to_dict():
        raise ValueError("Mappings differ from a fresh original train-only fit")
    membership_path = audit_directory / "split_membership.json"
    membership = _read_json(membership_path)
    original_ids = {partition: [example.record_id for example in examples] for partition, examples in splits.items()}
    if membership["dataset_sha256"] != dataset_sha or membership["record_ids"] != original_ids:
        raise ValueError("Original split membership differs from Phase 2")
    id_hashes = {"train": configuration_hash(sorted(original_ids["train"])),
                 "dev": configuration_hash(original_ids["dev"])}
    _, training_summary, _ = _validate_completed_run(run, selection, payload, protocol, id_hashes)
    plan, registered_evidence = _validate_registered_candidates(root, selection, protocol, id_hashes)
    required_sources = set(SOURCE_FILES)
    if not required_sources <= payload["source_hashes"].keys():
        raise ValueError("Checkpoint is missing required trained-source identities")
    source_hashes = dict(payload["source_hashes"])
    for name, expected in source_hashes.items():
        if file_sha256(relative_project_path(root, name)) != expected:
            raise ValueError(f"Trained source identity changed: {name}")
    for name in ("src/data.py", "src/schema.py", "src/metrics.py"):
        if source_hashes[name] != audit["source_hashes"][name]:
            raise ValueError(f"Shared Phase 2 implementation changed: {name}")
    all_sources = sorted((*root.joinpath("src").glob("*.py"), *root.joinpath("scripts").glob("*.py")))
    source_hashes.update({path.relative_to(root).as_posix(): file_sha256(path) for path in all_sources})
    counts = enforce_parameter_cap(model, protocol["limits"]["total_parameters"])
    if counts != payload["parameter_counts"]:
        raise ValueError("Selected model parameter count changed")

    output = root / "artifacts/phase5/final_stage2"
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "snapshot"
    evidence = [selection_path, checkpoint, Path(str(checkpoint) + ".sha256"), mapping_path, membership_path,
                audit_directory / "data_audit.json", *registered_evidence]
    evidence.extend(run / name for name in ("run.json", "protocol.json", "stage_config.json", "training_summary.json",
                    "experiment_summary.json", "history.jsonl", "history.csv", "learning_curves.png", "best_dev_predictions.jsonl", "reload_check.json"))
    evidence.extend(root / name for name in ("requirements.txt", "requirements-cuda.txt", "requirements.lock.txt",
                    "PHASE5_EXPERIMENT_PLAN.md") if (root / name).is_file())
    frozen_files = {}
    for source in dict.fromkeys([relative_project_path(root, name) for name in source_hashes] + evidence):
        name = source.relative_to(root).as_posix()
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        frozen_files[name] = file_sha256(destination)
        if frozen_files[name] != file_sha256(source):
            raise ValueError(f"Snapshot differs from its source: {name}")

    training = payload["stage_config"]["training"]
    def loader(partition):
        return make_loader(prepare_split(splits[partition], mappings), partition,
                           training["batch_size"], protocol["runtime"]["seed"], char_to_id)
    before_weights = _weight_fingerprint(model)
    dev = evaluate(model, loader("dev"), mappings, selected_device)
    _validate_complete_result(dev, original_ids["dev"], "dev")
    if [prediction.to_dict() for prediction in dev["predictions"]] != payload["validation_predictions"]:
        raise ValueError("Fresh final reload changed ordered checkpoint dev predictions")
    score = selection_score(dev["metrics"].intent_accuracy, dev["metrics"].slot_micro_f1)
    difference = abs(score - selection["selection_score"])
    if difference > 1e-12:
        raise ValueError("Fresh final reload changed the unrounded selected dev score")
    _write_json(output / "dev_evaluation.json", _result_data(dev))
    write_predictions(output / "dev_predictions.jsonl", dev["predictions"])
    if file_sha256(selection_path) != frozen_files[selection_path.relative_to(root).as_posix()]:
        raise ValueError("Dev-only selection changed while freezing")
    if file_sha256(checkpoint) != selection["checkpoint_sha256"] or _weight_fingerprint(model) != before_weights:
        raise ValueError("Evaluation changed selected checkpoint bytes or model weights")
    manifest = {"schema_version": 1, "stage": "stage2", "status": "frozen_before_first_test_forward",
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "selection": selection,
                "selection_file": selection_path.relative_to(root).as_posix(),
                "selection_file_sha256": frozen_files[selection_path.relative_to(root).as_posix()],
                "checkpoint_sha256": selection["checkpoint_sha256"], "model_weights_sha256": before_weights,
                "dataset_sha256": dataset_sha, "mappings_sha256": payload["mappings_sha256"],
                "char_to_id_sha256": payload["char_to_id_sha256"], "protocol_sha256": payload["protocol_sha256"],
                "stage_config_sha256": payload["stage_config_sha256"], "source_hashes": source_hashes,
                "snapshot_file_hashes": frozen_files, "parameter_counts": counts,
                "registered_candidates_completed": len(plan["candidates"]),
                "fresh_dev_reload": {"ordered_predictions_equal": True, "selection_score": score,
                                     "absolute_score_difference": difference, "num_examples": dev["num_examples"]},
                "stage2_training_and_tuning_closed": True, "test_evaluation_started": False,
                "test_policy": "One complete final-reporting-only test pass; no subsequent Stage 2 tuning or selection.",
                "environment": {"python": sys.version, "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
                                "device": str(selected_device), "gpu": torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else None,
                                "fresh_cli_process": True, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}}
    _write_json(output / "freeze_manifest.json", manifest)
    reporting = output / "reporting_only"
    reporting.mkdir()
    _write_json(reporting / "test_started.json", {"started_at_utc": datetime.now(timezone.utc).isoformat(),
                "freeze_manifest_sha256": file_sha256(output / "freeze_manifest.json"), "purpose": "final_reporting_only"})
    # Test examples reach the model only after the immutable manifest exists.
    test = evaluate(model, loader("test"), mappings, selected_device, allow_test=True)
    _validate_complete_result(test, original_ids["test"], "test")
    if file_sha256(checkpoint) != selection["checkpoint_sha256"] or _weight_fingerprint(model) != before_weights:
        raise ValueError("Final test evaluation changed selected checkpoint bytes or model weights")
    _write_json(reporting / "test_evaluation.json", {"purpose": "final_reporting_only", **_result_data(test)})
    write_predictions(reporting / "test_predictions.jsonl", test["predictions"])
    rows = []
    for partition, result in (("dev", dev), ("test", test)):
        metrics = result["metrics"]
        rows.append({"stage": "Stage 2", "split": partition, "intent_accuracy_percent": f"{100 * metrics.intent_accuracy:.2f}",
                     "slot_micro_f1_percent": f"{100 * metrics.slot_micro_f1:.2f}", "total_parameters": counts["total"],
                     "trainable_parameters": counts["trainable"], "selected_epoch": selection["selected_epoch"],
                     "epochs_trained": selection["epochs_trained"], "num_examples": result["num_examples"]})
    _write_json(output / "stage2_results.json", {"schema_version": 1, "display_scale": "percent_2dp", "rows": rows,
                "full_precision": {"dev": dev["metrics"].to_dict(), "test": test["metrics"].to_dict()}})
    with (output / "stage2_results.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"schema_version": 1, "status": "complete", "stage": "stage2", "output_directory": str(output),
               "selected_run": selection["selected_run"], "checkpoint": selection["checkpoint"],
               "checkpoint_sha256": selection["checkpoint_sha256"], "frozen_checkpoint": str(snapshot / selection["checkpoint"]),
               "freeze_manifest_sha256": file_sha256(output / "freeze_manifest.json"), "source_hashes": source_hashes,
               "protocol_sha256": payload["protocol_sha256"], "stage_config_sha256": payload["stage_config_sha256"],
               "dataset_sha256": dataset_sha, "mappings_sha256": payload["mappings_sha256"], "char_to_id_sha256": payload["char_to_id_sha256"],
               "parameter_counts": counts, "epochs_trained": selection["epochs_trained"], "selected_epoch": selection["selected_epoch"],
               "selection_score": score, "training_seconds": training_summary["training_seconds"],
               "fresh_process_reload": {"passed": True, "ordered_predictions_equal": True, "absolute_score_difference": difference,
                                        "num_examples": dev["num_examples"], "fresh_python_process": True},
               "dev": _result_data(dev), "test": {key: value for key, value in _result_data(test).items() if key != "record_ids"},
               "test_model_evaluations": 1, "test_evaluation_finished": True, "model_weights_unchanged": True,
               "checkpoint_bytes_unchanged": True, "stage2_training_and_tuning_closed": True,
               "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Frozen Stage 2 final results. Test outputs are reporting only; selection used development data exclusively."}
    _write_json(output / "final_summary.json", summary)
    return summary

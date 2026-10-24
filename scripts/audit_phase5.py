"""Independently audit completed Phase 5 artifacts without model inference.

This reads Stage 2 test predictions only after final_summary.json confirms
completion and freeze/test-start records exist. Torch is used only to inspect
saved CPU tensors with weights_only=True. No model is constructed or executed.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys


REQUIRED_TRAINED_SOURCES = {
    "src/stage2_model.py", "src/stage2_experiment.py", "scripts/run_stage2.py",
    "src/models.py", "src/training.py", "src/experiment.py", "src/final_evaluation.py",
    "src/data.py", "src/schema.py", "src/metrics.py", "src/runtime.py",
    "src/audit.py", "src/acquisition.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def project_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.anchor:
        raise ValueError("Audit evidence paths must be relative to the project")
    path = (root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Audit evidence path escapes the project")
    return path


def timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Evidence timestamp is missing its timezone")
    return result


def identical_metrics(left: dict, right: dict) -> bool:
    keys = {"intent_accuracy", "slot_micro_f1", "slot_tp", "slot_fp", "slot_fn", "num_examples", "correct_intents"}
    if set(left) != keys or set(right) != keys:
        return False
    return all(left[key] == right[key] if isinstance(left[key], int)
               else math.isclose(left[key], right[key], rel_tol=0, abs_tol=1e-12) for key in keys)


def state_fingerprint(state: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in state.items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str((str(value.dtype), tuple(value.shape))).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def audit(root: Path, test_log: Path) -> dict:
    root = root.resolve()
    sys.path.insert(0, str(root))
    from src.data import FittedMappings, load_massive, read_predictions, tokenize_with_offsets
    from src.metrics import evaluate_predictions

    checks = []

    def check(name, condition, **details):
        if not condition:
            raise ValueError("Independent audit failed: " + name)
        checks.append({"check": name, "passed": True, **details})

    final_dir = root / "artifacts/phase5/final_stage2"
    final_path = final_dir / "final_summary.json"
    manifest_path = final_dir / "freeze_manifest.json"
    started_path = final_dir / "reporting_only/test_started.json"
    # Stop here before loading any task labels or Stage 2 test artifacts if the
    # one reporting-only evaluation has not finished.
    final = read_json(final_path)
    check("final_evaluation_already_complete_before_offline_audit",
          final.get("status") == "complete" and final.get("test_evaluation_finished") is True
          and final.get("test_model_evaluations") == 1 and manifest_path.is_file() and started_path.is_file())
    manifest, started = read_json(manifest_path), read_json(started_path)
    check("immutable_pretest_manifest_identity", sha256(manifest_path) == final["freeze_manifest_sha256"]
          == started["freeze_manifest_sha256"] and manifest["status"] == "frozen_before_first_test_forward")
    selection_path = root / "artifacts/phase5/selection.json"
    selection = read_json(selection_path)
    check("persisted_selection_matches_freeze", selection == manifest["selection"]
          and sha256(selection_path) == manifest["selection_file_sha256"])
    check("selection_dev_only", selection["stage"] == "stage2" and selection["selection_split"] == "dev"
          and selection["test_used_for_selection"] is False)
    selected_time = timestamp(next(selection[key] for key in
                              ("committed_at_utc", "selection_committed_at_utc", "selected_at_utc") if key in selection))
    check("selection_freeze_test_completion_chronology", selected_time < timestamp(manifest["frozen_at_utc"])
          < timestamp(started["started_at_utc"]) < timestamp(final["completed_at_utc"]))
    check("final_test_is_reporting_only_and_tuning_closed", started["purpose"] == "final_reporting_only"
          and manifest["test_evaluation_started"] is False and manifest["stage2_training_and_tuning_closed"] is True
          and final["stage2_training_and_tuning_closed"] is True and final["model_weights_unchanged"] is True
          and final["checkpoint_bytes_unchanged"] is True)

    plan_path = project_path(root, selection["registered_plan"])
    plan = read_json(plan_path)
    check("registered_plan_and_document_identity", sha256(plan_path) == selection["registered_plan_sha256"]
          and sha256(project_path(root, plan["plan_document"])) == plan["plan_document_sha256"])
    registrations, candidates = plan["candidates"], selection["candidates"]
    check("exactly_five_bounded_registered_candidates", len(registrations) == len(candidates) == 5
          and plan["total_candidates"] == 5 and plan["total_epochs"] == 100
          and all(item["max_epochs"] == 20 for item in registrations)
          and [item["candidate_id"] for item in candidates] == [item["candidate_id"] for item in registrations]
          and len({item["run"] for item in candidates}) == 5)
    check("plan_uses_development_only", plan["selection_split"] == "dev" and plan["test_used_for_design"] is False)
    registered_time = timestamp(plan["registered_at_utc"])
    corrected_time = timestamp(plan["diagnosis_count_correction_before_training"]["corrected_at_utc"])
    check("documented_count_correction_after_registration", registered_time < corrected_time)

    protocol = read_json(root / "configs/protocol.json")
    dataset = project_path(root, protocol["paths"]["dataset"])
    data_hash = sha256(dataset)
    preprocessing_path = root / "artifacts/phase2/preprocessing.json"
    mappings = FittedMappings.from_dict(read_json(preprocessing_path))
    phase2 = read_json(root / "artifacts/phase2/data_audit.json")
    check("original_dataset_and_mappings_unchanged", data_hash == phase2["dataset_sha256"] == final["dataset_sha256"]
          and sha256(preprocessing_path) == phase2["preprocessing_sha256"])
    splits = load_massive(dataset)
    original_ids = {part: [example.record_id for example in rows] for part, rows in splits.items()}
    membership = read_json(root / "artifacts/phase2/split_membership.json")
    check("every_original_partition_identity_and_order", membership["record_ids"] == original_ids
          and membership["dataset_sha256"] == data_hash
          and {part: len(rows) for part, rows in splits.items()} == {"train": 11514, "dev": 2033, "test": 2974})
    expected_id_hashes = {"train": config_hash(sorted(original_ids["train"])), "dev": config_hash(original_ids["dev"])}
    expected_tokens = {part: sum(len(tokenize_with_offsets(example.utt)) for example in splits[part])
                       for part in ("train", "dev", "test")}
    characters = sorted({character for example in splits["train"] for token in tokenize_with_offsets(example.utt)
                         for character in token.text})
    char_to_id = {"<PAD>": 0, "<UNK>": 1, **{char: index + 2 for index, char in enumerate(characters)}}
    tags = [tag for tag, index in sorted(mappings.tag_to_id.items(), key=lambda item: item[1])]
    intents = set(mappings.intent_to_id)
    slot_types = set(mappings.slot_types)

    def score_predictions(path, partition, label):
        predictions = read_predictions(path)
        check(label + "_complete_prediction_identity_order", [row.record_id for row in predictions] == original_ids[partition])
        correct = tp = fp = fn = 0
        literal = True
        for example, prediction in zip(splits[partition], predictions):
            previous_end = 0
            literal = literal and prediction.intent in intents
            for slot in prediction.slots:
                literal = (literal and slot.slot_type in slot_types and type(slot.start) is int and type(slot.end) is int
                           and previous_end <= slot.start < slot.end <= len(example.utt)
                           and example.utt[slot.start:slot.end] == slot.text)
                previous_end = slot.end
            correct += prediction.intent == example.intent
            predicted_pairs = {(slot.slot_type, slot.text) for slot in prediction.slots}
            gold_pairs = {(slot.slot_type, slot.text) for slot in example.slots}
            tp += len(gold_pairs & predicted_pairs)
            fp += len(predicted_pairs - gold_pairs)
            fn += len(gold_pairs - predicted_pairs)
        check(label + "_literal_spans_and_valid_labels", literal)
        denominator = 2 * tp + fp + fn
        manual = {"intent_accuracy": correct / len(predictions), "slot_micro_f1": 2 * tp / denominator if denominator else 0.0,
                  "slot_tp": tp, "slot_fp": fp, "slot_fn": fn, "num_examples": len(predictions), "correct_intents": correct}
        shared = evaluate_predictions([example.gold_prediction() for example in splits[partition]], predictions).to_dict()
        check(label + "_manual_counts_equal_shared_scorer", identical_metrics(manual, shared))
        return shared, [prediction.to_dict() for prediction in predictions]

    # Load only tensors on CPU; do not construct a model, optimizer, or loader.
    import torch
    expected_sources = None
    candidate_results = []
    selected_state_hash = None
    total_epochs = 0
    for registration, candidate in zip(registrations, candidates):
        prefix = f"candidate_{candidate['candidate_id']}"
        run = project_path(root, candidate["run"])
        config_path = project_path(root, registration["configuration"])
        config = read_json(config_path)
        metadata = read_json(run / "run.json")
        summary = read_json(run / "training_summary.json")
        experiment = read_json(run / "experiment_summary.json")
        history = [json.loads(line) for line in (run / "history.jsonl").read_text(encoding="utf-8").splitlines()]
        created_time = timestamp(metadata["created_at_utc"])
        check(prefix + "_registration_and_correction_before_training", registered_time < corrected_time < created_time < selected_time)
        check(prefix + "_registered_config_and_protocol", sha256(config_path) == registration["configuration_sha256"]
              and config_hash(config) == registration["config_sha256"] == metadata["config_sha256"]
              and config == read_json(run / "stage_config.json") == metadata["stage_config"]
              and protocol == read_json(run / "protocol.json") == metadata["protocol"]
              and metadata["candidate_id"] == candidate["candidate_id"] == registration["candidate_id"]
              and metadata["registered_plan_sha256"] == selection["registered_plan_sha256"]
              and metadata["dataset_sha256"] == data_hash and candidate["configuration"] == registration["configuration"])
        check(prefix + "_whole_original_twenty_epochs", len(history) == config["training"]["max_epochs"]
              == summary["epochs_trained"] == experiment["epochs_trained"] == candidate["epochs_trained"] == 20
              and [row["epoch"] for row in history] == list(range(1, 21))
              and all(row["train_num_examples"] == 11514 and row["dev_num_examples"] == 2033
                      and row["train_record_ids_sha256"] == expected_id_hashes["train"]
                      and row["dev_record_ids_sha256"] == expected_id_hashes["dev"]
                      and row["train_num_tokens"] == expected_tokens["train"]
                      and row["dev_num_tokens"] == expected_tokens["dev"] for row in history))
        total_epochs += len(history)
        check(prefix + "_unrounded_epoch_scores", all(math.isfinite(row["selection_score"])
              and 0 <= row["dev_intent_accuracy"] <= 1 and 0 <= row["dev_slot_micro_f1"] <= 1
              and math.isclose(row["selection_score"], (row["dev_intent_accuracy"] + row["dev_slot_micro_f1"]) / 2,
                               rel_tol=0, abs_tol=1e-12) for row in history))
        check(prefix + "_finite_training_and_evaluation_losses", all(
            all(math.isfinite(row[part + "_" + key]) for part in ("train", "dev")
                for key in ("loss", "intent_loss", "slot_loss")) and math.isfinite(row["max_gradient_norm"])
            for row in history))
        best = max(history, key=lambda row: row["selection_score"])
        check(prefix + "_earliest_best_checkpoint", best["epoch"] == candidate["best_epoch"] == summary["best_epoch"] == experiment["best_epoch"]
              and best["selection_score"] == candidate["selection_score"] == summary["best_score"] == experiment["best_score"])
        check(prefix + "_no_test_during_training", summary["test_evaluated"] is False and experiment["test_evaluated"] is False
              and all(report["purpose"] == "experiment" and report["max_train_batches"] is None and report["max_dev_batches"] is None
                      for report in (summary, experiment)))
        reload = experiment["fresh_process_reload"]
        check(prefix + "_fresh_full_dev_reload", reload == read_json(run / "reload_check.json")
              and reload["passed"] is True and reload["fresh_python_process"] is True
              and reload["ordered_predictions_equal"] is True and reload["test_evaluated"] is False
              and reload["num_examples"] == 2033 and reload["absolute_score_difference"] == 0
              and reload["selection_score"] == best["selection_score"])
        sources = metadata["source_hashes"]
        expected_sources = sources if expected_sources is None else expected_sources
        check(prefix + "_all_thirteen_trained_sources_identical", set(sources) == REQUIRED_TRAINED_SOURCES
              and sources == expected_sources and all(sha256(project_path(root, name)) == digest for name, digest in sources.items()))
        checkpoint = run / "best.pt"
        checkpoint_sha = sha256(checkpoint)
        check(prefix + "_checkpoint_bytes_and_sidecar", checkpoint_sha == candidate["checkpoint_sha256"]
              == Path(str(checkpoint) + ".sha256").read_text(encoding="ascii").strip())
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        check(prefix + "_checkpoint_training_identity", payload["model_class"] == "JointStage2" and payload["purpose"] == "experiment"
              and payload["stage_config"] == config and payload["protocol"] == protocol and payload["mappings"] == mappings.to_dict()
              and payload["char_to_id"] == char_to_id and payload["source_hashes"] == sources and payload["data_sha256"] == data_hash
              and payload["epoch"] == best["epoch"] and payload["selection_score"] == best["selection_score"]
              and all(config_hash(payload[key]) == payload[key + "_sha256"] for key in ("stage_config", "protocol", "mappings", "char_to_id")))
        metrics, prediction_rows = score_predictions(run / "best_dev_predictions.jsonl", "dev", prefix + "_dev")
        check(prefix + "_saved_predictions_match_checkpoint_and_reload", prediction_rows == payload["validation_predictions"]
              and identical_metrics(metrics, payload["validation"]["metrics"]) and identical_metrics(metrics, reload["metrics"])
              and math.isclose((metrics["intent_accuracy"] + metrics["slot_micro_f1"]) / 2, best["selection_score"], rel_tol=0, abs_tol=1e-12))
        state = payload["model_state"]
        architecture = config["model"]
        expected_names = {"embedding.weight", "intent_head.weight", "intent_head.bias", "slot_head.weight", "slot_head.bias"}
        for suffix in ("", "_reverse"):
            for kind in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                expected_names.add(f"encoder.{kind}_l0{suffix}")
        if architecture["use_char"]:
            expected_names.update({"char_encoder.embedding.weight", "char_encoder.convolution.weight", "char_encoder.convolution.bias"})
        if architecture["intent_conditioning"]:
            expected_names.add("intent_embeddings.weight")
        if architecture["use_crf"]:
            expected_names.update({"crf.start_transitions", "crf.end_transitions", "crf.transitions", "crf.start_allowed", "crf.transition_allowed"})
        buffer_names = {"crf.start_allowed", "crf.transition_allowed"} if architecture["use_crf"] else set()
        check(prefix + "_one_joint_system_registered_tensor_inventory", architecture["num_layers"] == 1 and set(state) == expected_names
              and all(bool(torch.isfinite(value).all()) for value in state.values()))
        count = sum(value.numel() for name, value in state.items() if name not in buffer_names)
        hidden, embedding = architecture["hidden_dim"], architecture["embedding_dim"]
        channels = architecture["char_channels"] if architecture["use_char"] else 0
        context = architecture["intent_context_dim"] if architecture["intent_conditioning"] else 0
        formula = (len(mappings.token_to_id) * embedding + 2 * 4 * hidden * (embedding + channels + hidden + 2)
                   + (2 * hidden + 1) * len(intents) + (2 * hidden + context + 1) * len(tags))
        if architecture["use_char"]:
            formula += (len(char_to_id) * architecture["char_embedding_dim"]
                        + channels * architecture["char_embedding_dim"] * architecture["char_kernel_size"] + channels)
        formula += len(intents) * context
        if architecture["use_crf"]:
            formula += len(tags) ** 2 + 2 * len(tags)
            expected_start = torch.tensor([not tag.startswith("I-") for tag in tags], dtype=torch.bool)
            expected_transition = torch.tensor([[not next_tag.startswith("I-") or previous in {"B-" + next_tag[2:], next_tag}
                                                for next_tag in tags] for previous in tags], dtype=torch.bool)
            check(prefix + "_persisted_crf_exact_BIO_constraints", torch.equal(state["crf.start_allowed"], expected_start)
                  and torch.equal(state["crf.transition_allowed"], expected_transition))
        check(prefix + "_all_embeddings_and_heads_in_parameter_cap", count == formula < 20_000_000
              and payload["parameter_counts"] == summary["parameter_counts"] == experiment["parameter_counts"]
              == {"total": count, "trainable": count})
        console = (root / f"artifacts/phase5/candidate{candidate['candidate_id']}_console.txt").read_text(encoding="utf-8")
        check(prefix + "_twenty_console_epochs", [int(number) for number in re.findall(r"^Epoch (\d+):", console, re.MULTILINE)] == list(range(1, 21)))
        candidate_results.append({"candidate_id": candidate["candidate_id"], "run": candidate["run"], "selected_epoch": best["epoch"],
                                  "selection_score": best["selection_score"], "metrics": metrics, "parameter_count": count,
                                  "checkpoint_sha256": checkpoint_sha})
        if candidate["run"] == selection["selected_run"]:
            selected_state_hash = state_fingerprint(state)
        del payload, state
    check("all_one_hundred_original_split_epochs_completed", total_epochs == 100)
    winner = max(candidate_results, key=lambda item: item["selection_score"])
    check("selected_earliest_registered_development_winner", winner["candidate_id"] == selection["selected_candidate"]
          and winner["run"] == selection["selected_run"] == final["selected_run"]
          and winner["selected_epoch"] == selection["selected_epoch"] == final["selected_epoch"]
          and winner["selection_score"] == selection["selection_score"] == final["selection_score"]
          and winner["checkpoint_sha256"] == selection["checkpoint_sha256"] == final["checkpoint_sha256"])
    check("selected_exact_saved_tensor_fingerprint", selected_state_hash == manifest["model_weights_sha256"])

    snapshot = final_dir / "snapshot"
    frozen_files = manifest["snapshot_file_hashes"]
    check("every_new_snapshot_byte_matches_manifest_and_original", all(
        sha256(project_path(snapshot.resolve(), name)) == digest == sha256(project_path(root, name))
        for name, digest in frozen_files.items()), files=len(frozen_files))
    check("new_snapshot_inventory_has_no_unlisted_files", {path.relative_to(snapshot).as_posix() for path in snapshot.rglob("*") if path.is_file()}
          == set(frozen_files))
    check("all_current_and_frozen_sources_match", manifest["source_hashes"] == final["source_hashes"]
          and REQUIRED_TRAINED_SOURCES <= manifest["source_hashes"].keys()
          and all(sha256(project_path(root, name)) == digest == sha256(project_path(snapshot.resolve(), name))
                  for name, digest in manifest["source_hashes"].items()), source_files=len(manifest["source_hashes"]))
    check("frozen_selected_checkpoint_exact", sha256(project_path(snapshot.resolve(), selection["checkpoint"]))
          == sha256(project_path(root, selection["checkpoint"])) == selection["checkpoint_sha256"])

    final_metrics = {}
    for partition in ("dev", "test"):
        directory = final_dir if partition == "dev" else final_dir / "reporting_only"
        metrics, _ = score_predictions(directory / f"{partition}_predictions.jsonl", partition, "final_" + partition)
        evaluation = read_json(directory / f"{partition}_evaluation.json")
        check("final_" + partition + "_all_original_ids_and_metrics", evaluation["record_ids"] == original_ids[partition]
              and evaluation["num_examples"] == len(splits[partition]) and evaluation["partitions"] == [partition]
              and evaluation["num_tokens"] == expected_tokens[partition]
              and evaluation["limited"] is False and evaluation["max_batches"] is None
              and identical_metrics(metrics, evaluation["metrics"]) and identical_metrics(metrics, final[partition]["metrics"]))
        if partition == "test":
            check("test_result_marked_reporting_only", evaluation["purpose"] == "final_reporting_only")
        final_metrics[partition] = metrics
    check("final_dev_matches_selected_candidate", identical_metrics(final_metrics["dev"], winner["metrics"])
          and final["fresh_process_reload"]["passed"] is True and final["fresh_process_reload"]["ordered_predictions_equal"] is True
          and final["fresh_process_reload"]["absolute_score_difference"] == 0 and final["fresh_process_reload"]["num_examples"] == 2033)
    results = read_json(final_dir / "stage2_results.json")
    with (final_dir / "stage2_results.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    check("two_row_final_reporting_table", len(results["rows"]) == len(csv_rows) == 2
          and [row["split"] for row in results["rows"]] == ["dev", "test"]
          and all(identical_metrics(final_metrics[part], results["full_precision"][part]) for part in ("dev", "test"))
          and all(row["intent_accuracy_percent"] == f"{100 * final_metrics[row['split']]['intent_accuracy']:.2f}"
                  and row["slot_micro_f1_percent"] == f"{100 * final_metrics[row['split']]['slot_micro_f1']:.2f}"
                  for row in csv_rows))

    baseline_dir = root / "artifacts/phase4/final_baseline"
    baseline = read_json(baseline_dir / "final_summary.json")
    baseline_manifest = read_json(baseline_dir / "freeze_manifest.json")
    baseline_completion = read_json(root / "artifacts/phase4/completion.json")
    check("baseline_freeze_and_audit_evidence_preserved", sha256(baseline_dir / "freeze_manifest.json") == baseline["freeze_manifest_sha256"]
          and sha256(root / "artifacts/phase4/independent_audit.json") == baseline_completion["independent_audit_sha256"])
    check("all_38_baseline_snapshot_files_and_originals_preserved", len(baseline_manifest["snapshot_file_hashes"]) == 38
          and all(sha256(project_path((baseline_dir / "snapshot").resolve(), name)) == digest == sha256(project_path(root, name))
                  for name, digest in baseline_manifest["snapshot_file_hashes"].items()))
    check("all_17_baseline_sources_preserved", len(baseline_manifest["source_hashes"]) == 17
          and all(sha256(project_path(root, name)) == digest for name, digest in baseline_manifest["source_hashes"].items()))
    check("baseline_results_and_checkpoint_unchanged", baseline["checkpoint_sha256"] == baseline_manifest["checkpoint_sha256"]
          and sha256(project_path(root, baseline["checkpoint"])) == baseline["checkpoint_sha256"]
          and identical_metrics(baseline["dev"]["metrics"], baseline_completion["dev_metrics"])
          and identical_metrics(baseline["test"]["metrics"], baseline_completion["test_metrics"]))

    test_text = test_log.read_text(encoding="utf-8")
    count_matches = re.findall(r"Ran (\d+) tests? in [0-9.]+s", test_text)
    check("final_full_unit_test_suite_passed", bool(count_matches) and int(count_matches[-1]) >= 139
          and re.search(r"\nOK\s*$", test_text) is not None and "FAILED (" not in test_text)
    return {"schema_version": 1, "status": "passed", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Independent saved-artifact audit. Manual and shared aggregate scoring of saved predictions; CPU weights_only tensor inspection. No model construction, forward pass, optimizer update, or additional test-model evaluation.",
            "checks": checks, "checks_passed": len(checks), "candidate_results": candidate_results,
            "selected_candidate": selection["selected_candidate"], "selected_run": selection["selected_run"],
            "final_metrics": final_metrics, "training_epochs_verified": total_epochs,
            "snapshot_files_verified": len(frozen_files), "source_files_verified": len(manifest["source_hashes"]),
            "baseline_snapshot_files_verified": 38, "baseline_source_files_verified": 17,
            "test_model_evaluations_during_audit": 0, "unit_tests_passed": int(count_matches[-1]),
            "evidence_hashes": {"audit_script": sha256(Path(__file__)), "selection": sha256(selection_path),
                                "final_summary": sha256(final_path), "freeze_manifest": sha256(manifest_path),
                                "unit_tests": sha256(test_log)},
            "limits": "Single final test pass and absence of later updates are corroborated by exclusive-output code, saved chronology, immutable hashes, and run records; this audit is not an external process monitor."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--test-log", default="artifacts/phase5/unit_tests_final.txt")
    args = parser.parse_args()
    root = args.project_root.resolve()
    report = audit(root, project_path(root, args.test_log))
    output = root / "artifacts/phase5/independent_audit.json"
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "checks_passed": report["checks_passed"],
                      "unit_tests_passed": report["unit_tests_passed"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

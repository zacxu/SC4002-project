"""Notebook-reusable Phase 2 audit; test inspection stays strictly structural."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any

from src.acquisition import SOURCE_URL, file_sha256
from src.data import (
    EXPECTED_COUNTS, SLOT_IGNORE_INDEX, TOKEN_PATTERN, FittedMappings, collate_examples,
    decode_bio, fit_mappings, gold_bio_tags, load_massive, prepare_split,
    tokenize_with_offsets,
)
from src.metrics import evaluate_predictions
from src.runtime import (
    configuration_hash, configure_reproducibility, dataloader_generator,
    project_root, relative_project_path,
)
from src.schema import IntentSlotPrediction


PREPROCESSING_PROFILE = {
    "version": 1, "tokenizer_pattern": TOKEN_PATTERN, "normalization": "none",
    "vocabulary_fit_partition": "train", "vocabulary_min_frequency": 1,
    "vocabulary_order": "descending_frequency_then_lexicographic",
    "pad_token_id": 0, "unknown_token_id": 1, "slot_encoding": "BIO",
    "slot_ignore_index": SLOT_IGNORE_INDEX, "padding": "dynamic_right", "truncation": "none",
    "invalid_bio_transition": "start_new_span",
}
EVALUATION_PROFILE = {
    "version": 1, "slot_matching": "set_of_exact_type_and_surface_text_per_record",
    "normalization": "none", "offsets": "retained_for_audit_not_scoring_key",
    "duplicate_identical_pairs": "count_once_per_record", "empty_slot_denominator_f1": 0.0,
    "empty_dataset": "error", "record_join": "exact_id_set_no_duplicates",
    "authority": "documented_project_interpretation_of_assignment",
}


def _require(condition: bool, message: str) -> None:
    """Checks remain active even when the interpreter uses optimization flags."""
    if not condition:
        raise ValueError(message)


def _length_summary(values: list[int]) -> dict[str, int | float | str]:
    ordered = sorted(values)
    return {"min": ordered[0], "median": median(ordered),
            "p95": ordered[math.ceil(0.95 * len(ordered)) - 1], "max": ordered[-1],
            "percentile_method": "nearest rank"}


def _training_statistics(examples: tuple, prepared: tuple) -> dict[str, Any]:
    intents = Counter(example.intent for example in examples)
    slots = Counter(slot.slot_type for example in examples for slot in example.slots)
    repeated_pair_examples = sum(
        any(count > 1 for count in Counter((slot.slot_type, slot.text) for slot in example.slots).values())
        for example in examples
    )
    return {
        "partition": "train", "num_examples": len(examples),
        "intent_counts": dict(sorted(intents.items())),
        "slot_occurrence_counts": dict(sorted(slots.items())),
        "utterance_character_lengths": _length_summary([len(example.utt) for example in examples]),
        "utterance_token_lengths": _length_summary([len(example.tokens) for example in prepared]),
        "examples_without_slots": sum(not example.slots for example in examples),
        "examples_with_repeated_identical_type_text_pairs": repeated_pair_examples,
    }


def _audit_loader(partition: str, prepared: tuple, mappings: FittedMappings, seed: int) -> dict[str, Any]:
    # Imported only after run_data_audit has configured the deterministic runtime.
    import torch
    from torch.utils.data import DataLoader

    batch_size = 32
    expected_by_id = {example.example.record_id: example for example in prepared}
    expected_ids = [example.example.record_id for example in prepared]
    loader = DataLoader(
        prepared, batch_size=batch_size, shuffle=partition == "train", drop_last=False,
        num_workers=0, collate_fn=collate_examples,
        generator=dataloader_generator(seed) if partition == "train" else None,
    )
    observed_ids: list[str] = []
    num_batches = 0
    for batch in loader:
        num_batches += 1
        inputs, masks, lengths = batch["input_ids"], batch["attention_mask"], batch["lengths"]
        labels, intents = batch["slot_labels"], batch["intent_labels"]
        _require(inputs.shape == masks.shape == labels.shape, "Padded input/target/mask shape mismatch")
        _require(inputs.dtype == labels.dtype == intents.dtype == lengths.dtype == torch.long,
                 "Input and label tensors must have integer dtype")
        _require(masks.dtype == torch.bool, "Attention masks must be boolean")
        _require(bool((lengths > 0).all()) and len(lengths) == inputs.shape[0], "Invalid unpadded lengths")
        _require(inputs.shape[1] == int(lengths.max()), "Batch was truncated or padded beyond its longest input")
        expected_masks = torch.arange(inputs.shape[1])[None, :] < lengths[:, None]
        _require(torch.equal(masks, expected_masks), "Attention masks do not match unpadded lengths")
        _require(bool((inputs[~masks] == 0).all()), "Padding must use token ID 0")
        _require(bool((labels[~masks] == SLOT_IGNORE_INDEX).all()), "Padded slot targets must be ignored")
        _require(bool(((inputs >= 0) & (inputs < len(mappings.token_to_id))).all()), "Token IDs outside vocabulary")
        _require(bool(((labels[masks] >= 0) & (labels[masks] < len(mappings.tag_to_id))).all()),
                 "Unpadded slot labels outside training mapping")
        _require(bool(((intents >= 0) & (intents < len(mappings.intent_to_id))).all()),
                 "Intent labels outside training mapping")
        _require(len(batch["record_ids"]) == inputs.shape[0], "Batch ID count mismatch")
        for row, record_id in enumerate(batch["record_ids"]):
            expected = expected_by_id[record_id]
            size = len(expected.tokens)
            _require(int(lengths[row]) == size, "An utterance was truncated during batching")
            _require(tuple(inputs[row, :size].tolist()) == expected.token_ids, "Input token IDs changed in collation")
            _require(tuple(labels[row, :size].tolist()) == expected.tag_ids, "Slot targets changed in collation")
            _require(int(intents[row]) == expected.intent_id, "Intent target changed in collation")
            _require(batch["tokens"][row] == expected.tokens, "Token sidecar changed in collation")
            _require(batch["examples"][row] == expected.example, "Record sidecar changed in collation")
            observed_ids.append(record_id)
    _require(len(observed_ids) == len(set(observed_ids)) == len(expected_ids), "A DataLoader dropped or repeated records")
    _require(set(observed_ids) == set(expected_ids), "DataLoader split membership changed")
    _require(num_batches == math.ceil(len(prepared) / batch_size), "Unexpected number of batches")
    if partition != "train":
        _require(observed_ids == expected_ids, "Evaluation loader must preserve source record order")
    return {"verified_records": len(observed_ids), "batch_size": batch_size, "num_batches": num_batches,
            "shuffle": partition == "train", "drop_last": False, "num_workers": 0,
            "full_membership_unique": True, "padding_masks_and_label_ranges": "passed",
            "source_order_preserved": True if partition != "train" else "training shuffle enabled"}


def run_data_audit(root: str | Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Verify complete original splits and save evidence only after every check passes.

    Gold self-scoring below tests only evaluator/representation identity on train
    and dev; it is not model performance. Test labels are consumed exclusively to
    validate loading, mapping compatibility, exact alignment and batching.
    """
    root = project_root(root)
    _require(protocol.get("preprocessing") == PREPROCESSING_PROFILE,
             "Configured preprocessing differs from the implemented frozen v1 profile")
    _require(protocol.get("evaluation") == EVALUATION_PROFILE,
             "Configured evaluation differs from the implemented frozen v1 profile")
    _require(protocol["runtime"]["num_workers"] == 0, "The audited loaders use num_workers=0")
    required_dataset = {"release": "MASSIVE 1.0", "locale": "en-US", "url": SOURCE_URL,
                        "model_input": "utt", "intent_target": "intent", "slots_target": "annot_utt",
                        "intent_types": 60, "slot_types": 55}
    _require(all(protocol["dataset"].get(key) == value for key, value in required_dataset.items()),
             "Dataset configuration differs from the assignment's original English source/target contract")
    configure_reproducibility(protocol["runtime"])
    dataset_path = relative_project_path(root, protocol["paths"]["dataset"])
    source_path = relative_project_path(root, "data/source_manifest.json")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    _require(source["schema_version"] == 1 and source["source"]["url"] == SOURCE_URL,
             "Unexpected dataset source manifest")
    english = source["english"]
    _require(english["relative_path"] == protocol["paths"]["dataset"], "Manifest/dataset path mismatch")
    raw_sha256 = file_sha256(dataset_path)
    _require(raw_sha256 == english["sha256"] and dataset_path.stat().st_size == english["size_bytes"],
             "English source bytes do not match the recorded acquisition manifest")
    archive_path = relative_project_path(root, source["archive"]["relative_path"])
    _require(file_sha256(archive_path) == source["archive"]["sha256"]
             and archive_path.stat().st_size == source["archive"]["size_bytes"],
             "Archive source bytes do not match the acquisition manifest")

    splits = load_massive(dataset_path)
    counts = {partition: len(examples) for partition, examples in splits.items()}
    _require(counts == EXPECTED_COUNTS == protocol["dataset"]["partition_counts"], "Required original split counts failed")
    _require(sum(counts.values()) == 16521, "Expected all 16,521 original English records")
    mappings = fit_mappings(splits["train"])
    _require(len(mappings.intent_to_id) == 60 and len(mappings.slot_types) == 55 and len(mappings.tag_to_id) == 111,
             "Training data must cover all 60 intents, 55 slot types and 111 BIO tags")
    _require(mappings.training_count == counts["train"], "Mappings were not fitted on all training records")
    mapping_data = mappings.to_dict()
    mapping_text = json.dumps(mapping_data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _require(FittedMappings.from_dict(json.loads(mapping_text)) == mappings, "Serialized mappings failed roundtrip")

    prepared_splits = {}
    structural_checks = {}
    decoded_predictions = {}
    for partition, examples in splits.items():
        prepared = prepare_split(examples, mappings)
        prepared_splits[partition] = prepared
        _require(len(prepared) == counts[partition], "Preparation dropped records")
        decoded = []
        for item in prepared:
            example = item.example
            input_only_tokens = tokenize_with_offsets(example.utt)
            _require(item.tokens == input_only_tokens, "Tokens must depend only on utt")
            _require("".join(token.text for token in item.tokens) == "".join(example.utt.split()),
                     "Tokenizer discarded or altered non-whitespace characters")
            _require(len(item.tokens) == len(item.token_ids) == len(item.tag_ids), "Prepared target/input length mismatch")
            _require(item.token_ids == tuple(mappings.token_to_id.get(token.text, 1) for token in input_only_tokens),
                     "Model inputs must be obtained only from utt and the training-fitted vocabulary")
            tags = gold_bio_tags(example, item.tokens)
            slots = decode_bio(example.utt, item.tokens, tags)
            _require(slots == example.slots, f"Gold BIO reconstruction differs for record {example.record_id}")
            if partition != "test":
                decoded.append(IntentSlotPrediction(example.record_id, example.intent, slots))
        if partition != "test":
            decoded_predictions[partition] = tuple(decoded)
        structural_checks[partition] = {
            "verified_records": len(prepared), "exact_annotation_alignment": "passed",
            "input_only_tokenization_and_nonwhitespace_coverage": "passed",
            "exact_bio_type_text_offset_roundtrip": "passed", "train_mapping_compatibility": "passed",
            "no_truncation_or_dropped_records": "passed",
        }

    identity_checks = {}
    for partition in ("train", "dev"):
        result = evaluate_predictions((example.gold_prediction() for example in splits[partition]),
                                      decoded_predictions[partition])
        _require(result.intent_accuracy == 1.0 and result.slot_micro_f1 == 1.0
                 and result.slot_fp == result.slot_fn == 0, "Gold representation/evaluator identity check failed")
        identity_checks[partition] = "passed; gold reconstruction sanity check, not model performance"
    loader_checks = {partition: _audit_loader(partition, prepared_splits[partition], mappings, protocol["runtime"]["seed"])
                     for partition in EXPECTED_COUNTS}
    training_statistics = _training_statistics(splits["train"], prepared_splits["train"])

    membership = {
        "schema_version": 1, "dataset_sha256": raw_sha256, "order": "original file order within each partition",
        "record_ids": {partition: [example.record_id for example in examples] for partition, examples in splits.items()},
    }
    output_dir = relative_project_path(root, protocol["paths"]["phase2"])
    output_dir.mkdir(parents=True, exist_ok=True)
    mappings_path = output_dir / "preprocessing.json"
    membership_path = output_dir / "split_membership.json"
    mappings_path.write_text(mapping_text, encoding="utf-8")
    membership_path.write_text(json.dumps(membership, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema_version": 1, "checked_at_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
        "scope": "Phase 2 data and evaluator correctness only; no model training or model performance measurement.",
        "config_sha256": configuration_hash(protocol), "dataset_sha256": raw_sha256,
        "implemented_preprocessing_profile": PREPROCESSING_PROFILE,
        "implemented_evaluation_profile": EVALUATION_PROFILE,
        "source_manifest_sha256": file_sha256(source_path), "archive_sha256": source["archive"]["sha256"],
        "source": source, "split_counts": counts, "total_records": sum(counts.values()),
        "fit_partition": "train", "training_label_coverage": {"intent_types": len(mappings.intent_to_id),
            "slot_types": len(mappings.slot_types), "bio_tags": len(mappings.tag_to_id), "vocabulary_size": len(mappings.token_to_id)},
        "preprocessing_artifact": str(mappings_path.relative_to(root)), "preprocessing_sha256": file_sha256(mappings_path),
        "split_membership_artifact": str(membership_path.relative_to(root)), "split_membership_sha256": file_sha256(membership_path),
        "structural_checks": structural_checks, "loader_checks": loader_checks,
        "gold_identity_scorer_checks": identity_checks, "training_statistics": training_statistics,
        "test_inspection_scope": "Structural validation only: source/split preservation, annotation offsets, input token coverage, BIO roundtrip, mapping compatibility and batching. No test label/length/OOV distributions or test scores.",
        "model_input_fields": ["input_ids", "attention_mask", "lengths"],
        "target_fields": ["intent_labels", "slot_labels"], "audit_sidecars": ["record_ids", "examples", "tokens"],
        "determinism": {"runtime": protocol["runtime"], "startup_pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
        "source_hashes": {name: file_sha256(root / name) for name in (
            "src/schema.py", "src/data.py", "src/metrics.py", "src/acquisition.py", "src/audit.py", "scripts/audit_data.py")},
        "evidence_limit": "Locally observed SHA-256 values provide provenance/change detection, not an independently published authenticity checksum.",
    }
    report_path = output_dir / "data_audit.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "split_counts": counts, "total_records": report["total_records"],
                      "training_label_coverage": report["training_label_coverage"],
                      "all_split_annotation_and_bio_roundtrips": "passed", "all_split_full_dataloaders": "passed",
                      "train_dev_gold_identity_scorer": "passed (sanity only)",
                      "test_inspection": "structural only; no test scores", "audit_report": str(report_path)}, indent=2))
    return report

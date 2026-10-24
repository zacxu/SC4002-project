"""Read-only acceptance checks for the standalone, genuinely executed notebook.

This auditor does not instantiate a model, run inference, or train.  It checks
visible code/output and reconciles the newly generated evidence with the
original release. Historical scores are comparison observations, not acceptance
targets. Static inspection is a bounded check, not a security sandbox.
"""
from __future__ import annotations

import argparse
import ast
import base64
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import nbformat


EXPECTED_COUNTS = {"train": 11514, "dev": 2033, "test": 2974}
EXPECTED_TOKENS = {"train": 83008, "dev": 14522, "test": 21089}
EXPECTED_PARAMETERS = {"stage1": 4646571, "stage2": 4939498}
DATA_SHA256 = "c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e"
ANNOTATION = re.compile(r"\[([A-Za-z_][A-Za-z_0-9]*)[ \t]*:[ \t]*(.*?)\]", re.DOTALL)
FORBIDDEN_IMPORTS = {"src", "scripts"}
HISTORICAL_PATH = re.compile(r"(?:^|[/\\])(?:configs[/\\]|artifacts[/\\](?:phase[1-6](?:[/\\]|$)|runs[/\\]))")


class ValidationError(ValueError):
    """An acceptance gate failed; existing evidence remains untouched."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def evidence_path(root: Path, name: str) -> Path:
    """Confine every evidence read to its explicit clean execution root."""
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValidationError("Evidence paths must be nonempty POSIX relative paths")
    relative = Path(name)
    if relative.is_absolute() or re.match(r"^[A-Za-z]:", name) or ".." in relative.parts:
        raise ValidationError(f"Unsafe evidence path: {name!r}")
    root = root.resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValidationError(f"Evidence path escapes execution root: {name!r}")
    if not result.is_file():
        raise ValidationError(f"Missing evidence file: {name}")
    return result


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValidationError(f"Empty or blank JSONL record: {path.name}")
    return [json.loads(line) for line in lines]


def _qualified(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _qualified(node.value) + "." + node.attr
    return ""


def validate_code(source: str) -> None:
    """Reject hidden repository dependencies and dynamically loaded source."""
    if re.search(r"^\s*(?:%{1,2}(?:run|load|load_ext)|!\s*(?:python|ipython))\b", source, re.MULTILINE):
        raise ValidationError("Notebook loads external implementation through a magic/shell command")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValidationError(f"Notebook code is not ordinary valid Python: {error}") from error
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
            if node.level:
                raise ValidationError("Relative imports require unsubmitted local modules")
        else:
            modules = []
        if any(module.split(".")[0] in FORBIDDEN_IMPORTS for module in modules):
            raise ValidationError("Notebook imports unsubmitted project modules")
        if any(module in {"importlib", "runpy"} or module.startswith(("importlib.", "runpy.")) for module in modules):
            raise ValidationError("Dynamic module loading prevents inline-source verification")
        if isinstance(node, ast.Attribute) and _qualified(node).startswith("sys.path"):
            raise ValidationError("Notebook modifies or searches sys.path")
        if isinstance(node, ast.Call):
            function = _qualified(node.func)
            if function in {"exec", "eval", "__import__"}:
                raise ValidationError("Notebook dynamically executes hidden implementation")
            # Provenance strings and explanatory docstrings are allowed. Literal
            # paths passed to filesystem/process operations are execution inputs.
            path_operation = function.split(".")[-1] in {
                "Path", "open", "load", "read_text", "read_bytes", "run", "Popen", "check_call", "check_output"
            }
            if path_operation:
                for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
                    for literal in ast.walk(argument):
                        if isinstance(literal, ast.Constant) and isinstance(literal.value, str):
                            value = literal.value
                            if HISTORICAL_PATH.search(value) or re.search(r"[A-Za-z]:[/\\]Users[/\\]", value):
                                raise ValidationError(f"Notebook reads a developer/historical path: {value!r}")


def _manifest_hashes(notebook: Any) -> list[str] | None:
    metadata = notebook.metadata.get("phase7", {})
    manifest = metadata.get("builder_source_manifest", metadata)
    hashes = metadata.get("code_cell_sha256", manifest.get("code_cell_sha256"))
    if hashes is None:
        return None
    if not isinstance(hashes, list) or any(not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{64}", item) for item in hashes):
        raise ValidationError("Invalid code-cell source hash manifest")
    return hashes


def validate_notebook_source(notebook: Any, *, require_manifest: bool = True) -> dict[str, Any]:
    nbformat.validate(notebook)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not cells:
        raise ValidationError("Notebook contains no implementation cells")
    actual = []
    for index, cell in enumerate(cells, 1):
        try:
            validate_code(cell.source)
        except ValidationError as error:
            raise ValidationError(f"Code cell {index}: {error}") from error
        actual.append(hashlib.sha256(cell.source.encode("utf-8")).hexdigest())
    expected = _manifest_hashes(notebook)
    if require_manifest and expected is None:
        raise ValidationError("Notebook is missing its builder code-cell source manifest")
    if expected is not None and actual != expected:
        raise ValidationError("Visible notebook source changed after the builder manifest was recorded")
    return {"code_cells": len(cells), "code_cell_sha256": actual,
            "implementation_sha256": canonical_hash(actual), "source_manifest_matches": expected is not None}


def validate_execution_outputs(notebook: Any) -> dict[str, Any]:
    counts: list[int] = []
    texts: list[str] = []
    pngs = 0
    png_hashes: list[str] = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        count = cell.get("execution_count")
        if type(count) is not int or count <= 0:
            raise ValidationError("Every code cell must retain a positive execution count")
        counts.append(count)
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                raise ValidationError(f"Notebook contains execution error: {output.get('ename')}: {output.get('evalue')}")
            if output.output_type == "stream":
                texts.append(output.get("text", ""))
            data = output.get("data", {})
            texts.append(data.get("text/plain", ""))
            if "image/png" in data:
                try:
                    payload = base64.b64decode("".join(data["image/png"].split()), validate=True)
                except (ValueError, TypeError) as error:
                    raise ValidationError("Malformed inline PNG output") from error
                if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValidationError("Inline plot is not a PNG")
                pngs += 1
                png_hashes.append(hashlib.sha256(payload).hexdigest())
    if counts != sorted(set(counts)):
        raise ValidationError("Code cells were executed out of order or have repeated counts")
    if pngs < 2:
        raise ValidationError("Both stages must retain inline learning-curve PNG output")
    text = "\n".join(texts)
    return {"execution_counts": counts, "inline_png_outputs": pngs,
            "inline_png_sha256": png_hashes, "visible_text": text, "error_outputs": 0}


def load_original(path: Path) -> dict[str, list[dict[str, Any]]]:
    if sha256(path) != DATA_SHA256:
        raise ValidationError("Dataset bytes differ from the frozen original English release")
    splits: dict[str, list[dict[str, Any]]] = {key: [] for key in EXPECTED_COUNTS}
    seen: set[str] = set()
    for row in read_jsonl(path):
        if row.get("locale") != "en-US" or row.get("partition") not in splits or row.get("id") in seen:
            raise ValidationError("Invalid original locale, split, or duplicate ID")
        seen.add(row["id"])
        parts: list[str] = []
        slots: list[dict[str, Any]] = []
        cursor = plain = 0
        for match in ANNOTATION.finditer(row["annot_utt"]):
            literal = row["annot_utt"][cursor:match.start()]
            parts.append(literal)
            plain += len(literal)
            value = match.group(2)
            slots.append({"slot_type": match.group(1), "text": value, "start": plain, "end": plain + len(value)})
            parts.append(value)
            plain += len(value)
            cursor = match.end()
        parts.append(row["annot_utt"][cursor:])
        if "".join(parts) != row["utt"]:
            raise ValidationError("Original annotation does not reconstruct utterance")
        splits[row["partition"]].append({"record_id": row["id"], "utt": row["utt"], "annot_utt": row["annot_utt"],
                                          "intent": row["intent"], "slots": slots})
    if {key: len(rows) for key, rows in splits.items()} != EXPECTED_COUNTS:
        raise ValidationError("Original split membership/counts changed")
    return splits


def fresh_training_mappings(training: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, int]]:
    """Recompute model dictionaries independently from original training only."""
    counts = Counter(match.group() for row in training for match in re.finditer(r"\w+|[^\w\s]", row["utt"]))
    words = {"<PAD>": 0, "<UNK>": 1}
    for token, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])):
        if token not in words:
            words[token] = len(words)
    intents = {name: index for index, name in enumerate(sorted({row["intent"] for row in training}))}
    types = sorted({slot["slot_type"] for row in training for slot in row["slots"]})
    tags = {"O": 0}
    for name in types:
        for prefix in ("B", "I"):
            tags[f"{prefix}-{name}"] = len(tags)
    mappings = {"schema_version": 1, "fit_partition": "train", "training_count": len(training),
                "training_ids_sha256": canonical_hash([row["record_id"] for row in training]),
                "training_records_sha256": canonical_hash([[row["record_id"], row["utt"], row["intent"], row["annot_utt"]] for row in training]),
                "tokenizer_pattern": r"\w+|[^\w\s]", "normalization": "none", "truncation": "none",
                "token_to_id": words, "intent_to_id": intents, "tag_to_id": tags, "slot_types": types,
                "pad_token_id": 0, "unknown_token_id": 1, "slot_ignore_index": -100}
    chars = {"<PAD>": 0, "<UNK>": 1}
    for character in sorted({character for token in counts for character in token}):
        chars[character] = len(chars)
    return mappings, chars


def score_saved_predictions(gold: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not gold or [row["record_id"] for row in predictions] != [row["record_id"] for row in gold]:
        raise ValidationError("Saved predictions do not preserve complete original ID order")
    if len({row["record_id"] for row in predictions}) != len(predictions):
        raise ValidationError("Duplicate prediction IDs")
    correct = tp = fp = fn = 0
    for original, predicted in zip(gold, predictions):
        if not isinstance(predicted.get("intent"), str) or not predicted["intent"]:
            raise ValidationError("Missing predicted intent")
        for slot in predicted["slots"]:
            if type(slot.get("start")) is not int or type(slot.get("end")) is not int:
                raise ValidationError("Predicted slot offsets must be integers")
            if not slot.get("slot_type") or not slot.get("text") or slot["start"] < 0 or slot["end"] <= slot["start"]:
                raise ValidationError("Invalid predicted slot")
            if original["utt"][slot["start"]:slot["end"]] != slot["text"]:
                raise ValidationError("Predicted slot is not a literal original utterance slice")
        target = {(slot["slot_type"], slot["text"]) for slot in original["slots"]}
        actual = {(slot["slot_type"], slot["text"]) for slot in predicted["slots"]}
        correct += predicted["intent"] == original["intent"]
        tp += len(target & actual)
        fp += len(actual - target)
        fn += len(target - actual)
    denominator = 2 * tp + fp + fn
    return {"intent_accuracy": correct / len(gold), "slot_micro_f1": 2 * tp / denominator if denominator else 0.0,
            "slot_tp": tp, "slot_fp": fp, "slot_fn": fn,
            "num_examples": len(gold), "correct_intents": correct}


def check_metrics(recorded: dict[str, Any], actual: dict[str, Any]) -> None:
    for key, expected in actual.items():
        observed = recorded.get(key)
        if isinstance(expected, float):
            valid = isinstance(observed, (float, int)) and math.isfinite(observed) and abs(observed - expected) <= 1e-12
        else:
            valid = type(observed) is int and observed == expected
        if not valid:
            raise ValidationError(f"Saved metric/count mismatch for {key}: {observed!r} versus {expected!r}")


def validate_history(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if [row.get("epoch") for row in rows] != list(range(1, 21)):
        raise ValidationError("A selected configuration must genuinely train all 20 ordered epochs")
    best = -float("inf")
    selected = None
    for row in rows:
        for split in ("train", "dev"):
            if row.get(f"{split}_num_examples") != EXPECTED_COUNTS[split] or row.get(f"{split}_num_tokens") != EXPECTED_TOKENS[split]:
                raise ValidationError("Epoch lacks complete original examples/tokens")
            for suffix in ("loss", "intent_loss", "slot_loss"):
                value = row.get(f"{split}_{suffix}")
                if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                    raise ValidationError("Epoch contains missing/nonfinite/negative loss")
            if f"{split}_num_batches" in row and row[f"{split}_num_batches"] != math.ceil(EXPECTED_COUNTS[split] / 32):
                raise ValidationError("Epoch contains an incomplete original batch traversal")
        if "max_gradient_norm" in row and (not math.isfinite(row["max_gradient_norm"]) or row["max_gradient_norm"] < 0):
            raise ValidationError("Epoch contains nonfinite/negative gradients")
        intent, slot = row.get("dev_intent_accuracy"), row.get("dev_slot_micro_f1")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1 for value in (intent, slot)):
            raise ValidationError("Invalid development task score")
        score = (intent + slot) / 2
        if not math.isclose(row.get("selection_score", float("nan")), score, rel_tol=0, abs_tol=1e-12):
            raise ValidationError("Development selection mean was changed")
        improvement = score > best
        if "selected_checkpoint" in row and row["selected_checkpoint"] is not improvement:
            raise ValidationError("Checkpoint selection does not preserve earliest exact tie")
        if improvement:
            best, selected = score, row["epoch"]
    return {"epochs_trained": 20, "selected_epoch": selected, "selection_score": best}


def validate_execution_evidence(root: Path, completion: dict[str, Any], source: dict[str, Any],
                                *, stage_names: tuple[str, ...] = ("stage1", "stage2")) -> dict[str, Any]:
    """Validate the explicit evidence ledger. No score is a historical target."""
    # The completed notebook writes this ledger with relative paths, so the
    # independent validator never guesses a historical checkpoint location.
    if completion.get("status") != "complete":
        raise ValidationError("Notebook execution ledger is incomplete")
    if completion.get("implementation_sha256") != source["implementation_sha256"]:
        raise ValidationError("Execution ledger is for different visible notebook definitions")
    if not stage_names or len(set(stage_names)) != len(stage_names) or set(stage_names) - EXPECTED_PARAMETERS.keys():
        raise ValidationError("Unknown or repeated requested stage audit")
    data = load_original(evidence_path(root, completion["dataset"]))
    mappings, chars = fresh_training_mappings(data["train"])
    if tuple(map(len, (mappings["token_to_id"], mappings["intent_to_id"], mappings["tag_to_id"], chars))) != (5146, 60, 111, 34):
        raise ValidationError("Original training-only vocabulary/label/character cardinalities changed")
    stages = {}
    for name in stage_names:
        expected_parameters = EXPECTED_PARAMETERS[name]
        entry = completion["stages"][name]
        directory = entry["directory"].rstrip("/")
        summary = read_json(evidence_path(root, entry.get("summary", directory + "/final_summary.json")))
        history = validate_history(read_jsonl(evidence_path(root, directory + "/history.jsonl")))
        for row in read_jsonl(evidence_path(root, directory + "/history.jsonl")):
            for partition in ("train", "dev"):
                ids = [record["record_id"] for record in data[partition]]
                expected = canonical_hash({"ids": sorted(ids) if partition == "train" else ids})
                if row.get(partition + "_record_ids_sha256") != expected:
                    raise ValidationError(f"{name} original {partition} membership/order hash changed during training")
        if summary.get("status") != "complete" or summary.get("test_model_evaluations") != 1:
            raise ValidationError(f"{name} did not record exactly one completed final test pass")
        for alternatives in (("model_weights_unchanged", "weights_unchanged"),
                             ("checkpoint_bytes_unchanged", "checkpoint_unchanged"),
                             ("training_and_tuning_closed", "tuning_closed")):
            if not any(summary.get(key) is True for key in alternatives):
                key = alternatives[0]
                raise ValidationError(f"{name} lacks required integrity/closure evidence: {key}")
        if summary.get("selected_epoch") != history["selected_epoch"] or summary.get("epochs_trained") != 20:
            raise ValidationError(f"{name} selected/trained epochs disagree with actual history")
        counts = summary["parameter_counts"]
        if counts.get("total") != expected_parameters or counts.get("trainable") != expected_parameters:
            raise ValidationError(f"{name} whole-model parameter count differs from fixed selected architecture")
        checkpoint = evidence_path(root, summary["checkpoint"])
        if sha256(checkpoint) != summary.get("checkpoint_sha256"):
            raise ValidationError(f"{name} checkpoint bytes changed")
        sidecar = evidence_path(root, summary["checkpoint"] + ".sha256").read_text(encoding="utf-8").strip().split()[0]
        if sidecar != sha256(checkpoint):
            raise ValidationError(f"{name} checkpoint sidecar hash disagrees")
        import torch
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("purpose") != "notebook_fixed_configuration_reproduction" or payload.get("stage") != name:
            raise ValidationError(f"{name} checkpoint is not a fresh notebook reproduction")
        if payload.get("mappings") != mappings or payload.get("char_to_id") != (chars if name == "stage2" else {}):
            raise ValidationError(f"{name} learned dictionaries were not fitted to original train only")
        if payload.get("epoch") != history["selected_epoch"] or abs(payload.get("selection_score", float("inf")) - history["selection_score"]) > 1e-12:
            raise ValidationError(f"{name} checkpoint is not the selected development maximum")
        if payload.get("data_sha256") != DATA_SHA256:
            raise ValidationError(f"{name} checkpoint refers to different original data")
        if completion.get("code_sha256") is not None and payload.get("code_sha256") != completion["code_sha256"]:
            raise ValidationError(f"{name} checkpoint refers to different notebook definitions")
        for key in ("mappings", "char_to_id", "stage_config", "runtime"):
            if canonical_hash(payload[key]) != payload.get(key + "_sha256"):
                raise ValidationError(f"{name} checkpoint {key} content/hash disagree")
        state = payload.get("model_state", payload.get("model_state_dict"))
        if not isinstance(state, dict) or not state:
            raise ValidationError(f"{name} checkpoint lacks learned tensors")
        learned = sum(value.numel() for value in state.values() if isinstance(value, torch.Tensor) and value.dtype != torch.bool)
        if learned != expected_parameters:
            raise ValidationError(f"{name} checkpoint learned-tensor count differs from complete model count")
        recorded = {}
        for split in ("dev", "test"):
            predictions = read_jsonl(evidence_path(root, directory + f"/{split}_predictions.jsonl"))
            if any(row.get("intent") not in mappings["intent_to_id"]
                   or any(slot.get("slot_type") not in mappings["slot_types"] for slot in row.get("slots", []))
                   for row in predictions):
                raise ValidationError(f"{name} predicts labels outside the original training label inventory")
            actual = score_saved_predictions(data[split], predictions)
            evaluation = read_json(evidence_path(root, directory + f"/{split}_evaluation.json"))
            check_metrics(evaluation["metrics"], actual)
            summary_split = summary["development"] if split == "dev" and "development" in summary else summary[split]
            check_metrics(summary_split["metrics"], actual)
            if evaluation.get("num_tokens") != EXPECTED_TOKENS[split] or evaluation.get("num_batches") != math.ceil(EXPECTED_COUNTS[split] / 32):
                raise ValidationError(f"{name} final {split} traversal was incomplete")
            recorded[split] = actual
            if split == "dev" and predictions != payload.get("validation_predictions"):
                raise ValidationError(f"{name} selected and reloaded development predictions differ")
        reload = summary["fresh_process_reload"]
        if (reload.get("ordered_predictions_equal") is not True
                or abs(reload.get("absolute_score_difference", float("inf"))) > 1e-12
                or reload.get("fresh_python_process") is not True
                or reload.get("test_evaluated") is not False
                or reload.get("num_examples") != EXPECTED_COUNTS["dev"]
                or reload.get("training_pid") == reload.get("verification_pid")):
            raise ValidationError(f"{name} lacks fresh-process exact ordered development parity")
        reload_path = evidence_path(root, directory + "/reload_check.json")
        if read_json(reload_path) != reload:
            raise ValidationError(f"{name} independent fresh-process verification record differs")
        freeze_path = evidence_path(root, directory + "/freeze_manifest.json")
        freeze = read_json(freeze_path)
        started = read_json(evidence_path(root, directory + "/test_started.json"))
        if (freeze.get("test_model_evaluations_at_freeze") != 0 or freeze.get("tuning_closed") is not True
                or started.get("test_model_evaluations") != 1 or started.get("reporting_only") is not True
                or started.get("freeze_manifest_sha256") != sha256(freeze_path)
                or summary.get("freeze_manifest_sha256") != sha256(freeze_path)
                or not (datetime.fromisoformat(reload["verified_at_utc"]) <= datetime.fromisoformat(freeze["frozen_at_utc"])
                        <= datetime.fromisoformat(started["started_at_utc"]) <= datetime.fromisoformat(summary["completed_at_utc"]))):
            raise ValidationError(f"{name} final-test freeze/chronology is inconsistent")
        for basename, field in (("history.jsonl", "history_sha256"),
                                ("training_summary.json", "training_summary_sha256"),
                                ("reload_check.json", "reload_check_sha256")):
            if sha256(evidence_path(root, directory + "/" + basename)) != freeze.get(field):
                raise ValidationError(f"{name} pre-test frozen evidence changed: {basename}")
        if freeze.get("checkpoint_sha256") != sha256(checkpoint):
            raise ValidationError(f"{name} checkpoint differs from pre-test freeze")
        digest = hashlib.sha256()
        for key, tensor in sorted(state.items()):
            tensor = tensor.detach().cpu().contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        if digest.hexdigest() != freeze.get("model_state_sha256"):
            raise ValidationError(f"{name} learned tensors differ from the pre-test weight freeze")
        optimizer = payload.get("optimizer_state", {})
        groups = optimizer.get("param_groups", [])
        if len(groups) != 1 or not groups[0].get("params") or set(groups[0]["params"]) != set(optimizer.get("state", {})):
            raise ValidationError(f"{name} does not retain one complete joint AdamW optimizer")
        expected_steps = history["selected_epoch"] * math.ceil(EXPECTED_COUNTS["train"] / 32)
        if any(float(record.get("step", -1)) != expected_steps for record in optimizer["state"].values()):
            raise ValidationError(f"{name} optimizer steps do not match complete selected training epochs")
        stages[name] = {**history, "parameter_counts": counts, **recorded,
                        "checkpoint_sha256": sha256(checkpoint)}
    if stage_names != ("stage1", "stage2"):
        # During the clean run a completed first stage can be inspected without
        # inventing a final completion artifact or evaluating either model.
        return {"scope": "partial saved-stage audit; not full notebook acceptance", "stages": stages,
                "dataset_sha256": DATA_SHA256, "original_split_counts": EXPECTED_COUNTS}
    reporting = read_json(evidence_path(root, "results.json"))
    rows = reporting.get("rows", [])
    expected_rows = []
    for name in EXPECTED_PARAMETERS:
        label = "Stage-1 system" if name == "stage1" else "Stage-2 system"
        for split in ("dev", "test"):
            metrics = stages[name][split]
            expected_rows.append({"model_version": label, "split": "Validation" if split == "dev" else "Test",
                                  "total_parameters": EXPECTED_PARAMETERS[name], "trainable_parameters": EXPECTED_PARAMETERS[name],
                                  "intent_accuracy_percent": f"{100 * metrics['intent_accuracy']:.2f}",
                                  "slot_micro_f1_percent": f"{100 * metrics['slot_micro_f1']:.2f}",
                                  "epochs_trained": 20, "selected_epoch": stages[name]["selected_epoch"]})
            check_metrics(reporting.get("full_precision", {}).get(label, {}).get(split, {}), metrics)
    if rows != expected_rows:
        raise ValidationError("The four-row report table differs from actual reloaded notebook evaluations")
    with evidence_path(root, "results.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if csv_rows != [{key: str(value) for key, value in row.items()} for row in expected_rows]:
        raise ValidationError("The generated CSV and JSON results tables differ")
    for split in ("dev", "test"):
        for key in ("intent_accuracy", "slot_micro_f1"):
            actual = 100 * (stages["stage2"][split][key] - stages["stage1"][split][key])
            observed = reporting.get("stage2_minus_stage1_percentage_points", {}).get(split, {}).get(key, float("nan"))
            if not math.isfinite(observed) or abs(observed - actual) > 1e-12:
                raise ValidationError("Reported improvement/regression differs from actual notebook results")
    errors = read_json(evidence_path(root, "error_cases.json"))
    cases = errors.get("cases", [])
    if len(cases) < 2 or len({case.get("record_id") for case in cases}) != len(cases):
        raise ValidationError("At least two distinct genuine final-system development errors are required")
    original_by_id = {row["record_id"]: row for row in data["dev"]}
    stage2_directory = completion["stages"]["stage2"]["directory"].rstrip("/")
    predicted_by_id = {row["record_id"]: row for row in read_jsonl(evidence_path(root, stage2_directory + "/dev_predictions.jsonl"))}
    for case in cases:
        original = original_by_id.get(case.get("record_id"))
        predicted = predicted_by_id.get(case.get("record_id"))
        if original is None or predicted is None:
            raise ValidationError("An error case is absent from original development records")
        if (case.get("utt") != original["utt"] or case.get("gold_intent") != original["intent"]
                or case.get("gold_slots") != original["slots"] or case.get("predicted_intent") != predicted["intent"]
                or case.get("predicted_slots") != predicted["slots"]):
            raise ValidationError("An error case does not reproduce genuine gold/prediction facts")
        pair_set = lambda slots: {(slot["slot_type"], slot["text"]) for slot in slots}
        if original["intent"] == predicted["intent"] and pair_set(original["slots"]) == pair_set(predicted["slots"]):
            raise ValidationError("A claimed error case is actually fully correct")
        hypotheses = case.get("possible_causes_labelled_as_inferences", [])
        if not hypotheses or any(not isinstance(value, str) or not value.startswith("Inference:") for value in hypotheses):
            raise ValidationError("Error causes must be explicitly labelled as inferences")
    hashes = completion.get("artifact_sha256", {})
    for name, expected in hashes.items():
        if sha256(evidence_path(root, name)) != expected:
            raise ValidationError(f"Generated execution artifact changed: {name}")
    return {"stages": stages, "artifact_hashes_verified": len(hashes),
            "dataset_sha256": DATA_SHA256, "original_split_counts": EXPECTED_COUNTS,
            "generated_result_rows": len(rows), "verified_genuine_error_cases": len(cases)}


def run(notebook_path: Path, execution_dir: Path | None = None) -> dict[str, Any]:
    notebook = nbformat.read(notebook_path, as_version=4)
    source = validate_notebook_source(notebook)
    outputs = validate_execution_outputs(notebook)
    text = outputs.pop("visible_text")
    definitions = {node.name for cell in notebook.cells if cell.cell_type == "code"
                   for node in ast.walk(ast.parse(cell.source)) if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    required = {"JointBiLSTM", "JointStage2", "BIOCRF", "CharacterCNN", "evaluate_predictions",
                "load_massive", "fit_mappings", "acquire_dataset", "nb_train_stage",
                "nb_verify_development", "nb_finalize_stage"}
    if missing := required - definitions:
        raise ValidationError(f"Notebook omits required inline implementation: {sorted(missing)}")
    for stage in EXPECTED_PARAMETERS:
        epochs = re.findall(rf"{stage} epoch (\d+)/20:", text)
        if [int(epoch) for epoch in epochs] != list(range(1, 21)):
            raise ValidationError(f"Visible {stage} training logs do not retain all 20 ordered epochs")
        fresh_marker = any("FRESH_PROCESS_DEVELOPMENT_VERIFIED" in line
                           and json.loads(line.split("FRESH_PROCESS_DEVELOPMENT_VERIFIED", 1)[1].strip()).get("stage") == stage
                           for line in text.splitlines() if "FRESH_PROCESS_DEVELOPMENT_VERIFIED" in line)
        if not fresh_marker and f"{stage}: fresh-process reload reproduced all 2,033 ordered development predictions exactly" not in text:
            raise ValidationError(f"Visible {stage} output omits fresh-interpreter development verification")
    report = {"schema_version": 1, "status": "passed", "notebook_sha256": sha256(notebook_path),
              "source": source, "outputs": outputs, "model_inference_performed": False}
    if execution_dir is not None:
        completion_path = evidence_path(execution_dir, "completion.json")
        completion = read_json(completion_path)
        report["evidence"] = validate_execution_evidence(execution_dir, completion, source)
        report["completion_sha256"] = sha256(completion_path)
        for name in EXPECTED_PARAMETERS:
            directory = completion["stages"][name]["directory"].rstrip("/")
            if sha256(evidence_path(execution_dir, directory + "/learning_curves.png")) not in outputs["inline_png_sha256"]:
                raise ValidationError(f"The actual new {name} learning-curve image is not retained inline")
        marker = "PHASE7_COMPLETION_JSON:"
        completions = [line.split(marker, 1)[1].strip() for line in text.splitlines() if marker in line]
        if len(completions) != 1 or json.loads(completions[0]) != completion:
            raise ValidationError("Visible notebook completion output differs from its generated execution ledger")
    if not text.strip():
        raise ValidationError("Notebook retained no visible execution evidence")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--execution-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run(arguments.notebook, arguments.execution_dir)
    if arguments.output.exists():
        raise ValidationError("Refusing to overwrite a prior notebook validation report")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(arguments.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

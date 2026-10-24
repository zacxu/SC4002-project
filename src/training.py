"""Joint-task training and checkpoint evidence, reusable in scripts/notebooks.

Diagnostic runs are explicitly separate from assessed full-split experiments.
No function automatically reads or evaluates the test split.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.nn import functional as F

from src.data import EXPECTED_COUNTS, SLOT_IGNORE_INDEX, FittedMappings, decode_bio, write_predictions
from src.metrics import evaluate_predictions
from src.models import BaselineConfig, JointBiLSTM, enforce_parameter_cap
from src.runtime import configuration_hash, is_improvement, selection_score
from src.schema import IntentSlotPrediction


@dataclass(frozen=True)
class LossBundle:
    total: torch.Tensor
    intent: torch.Tensor
    slot: torch.Tensor
    intent_sum: torch.Tensor
    slot_sum: torch.Tensor
    num_examples: int
    num_tokens: int


def compute_losses(outputs: dict, batch: dict, intent_weight: float = 1.0, slot_weight: float = 1.0) -> LossBundle:
    """Intent CE per utterance + slot CE per real token; ignore padded targets."""
    if any(not math.isfinite(value) or value <= 0 for value in (intent_weight, slot_weight)):
        raise ValueError("Both task loss weights must be finite and positive")
    intent_logits, slot_logits = outputs["intent_logits"], outputs["slot_logits"]
    intents, slots, mask = batch["intent_labels"], batch["slot_labels"], batch["attention_mask"]
    if intent_logits.ndim != 2 or slot_logits.ndim != 3 or slots.shape != slot_logits.shape[:2]:
        raise ValueError("Invalid task-logit or target shapes")
    if mask.shape != slots.shape or mask.dtype != torch.bool or intents.shape != (slots.shape[0],):
        raise ValueError("Target shapes and attention mask must agree")
    if intent_logits.shape[0] != slots.shape[0] or not slots.shape[0]:
        raise ValueError("Intent/slot batch sizes must agree and be nonempty")
    if intents.dtype != torch.long or slots.dtype != torch.long:
        raise ValueError("Classification targets must be long tensors")
    if not bool((slots[~mask] == SLOT_IGNORE_INDEX).all()) or bool((slots[mask] < 0).any()):
        raise ValueError("Only padded slot targets may use the ignore index")
    if not bool(torch.isfinite(intent_logits).all()) or not bool(torch.isfinite(slot_logits).all()):
        raise ValueError("Task logits must be finite")
    num_examples, num_tokens = slots.shape[0], int(mask.sum().item())
    if num_tokens == 0:
        raise ValueError("Cannot compute slot loss without real tokens")
    intent_sum = F.cross_entropy(intent_logits, intents, reduction="sum")
    slot_sum = F.cross_entropy(slot_logits.reshape(-1, slot_logits.shape[-1]), slots.reshape(-1),
                               ignore_index=SLOT_IGNORE_INDEX, reduction="sum")
    intent_loss, slot_loss = intent_sum / num_examples, slot_sum / num_tokens
    total = intent_weight * intent_loss + slot_weight * slot_loss
    if not bool(torch.isfinite(total)):
        raise ValueError("Nonfinite task loss")
    return LossBundle(total, intent_loss, slot_loss, intent_sum, slot_sum, num_examples, num_tokens)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _device_batch(batch: dict, device: torch.device) -> dict:
    result = dict(batch)
    for key in ("input_ids", "attention_mask", "intent_labels", "slot_labels"):
        result[key] = batch[key].to(device)
    # pack_padded_sequence consumes CPU lengths; metadata remains outside forward.
    result["lengths"] = batch["lengths"].cpu()
    return result


def _partitions(batch: dict) -> set[str]:
    if not batch.get("examples"):
        raise ValueError("Original-example sidecars are required to enforce split boundaries")
    return {example.partition for example in batch["examples"]}


def _summarize(sums: tuple[float, float], examples: int, tokens: int, batches: int,
               seconds: float, intent_weight: float, slot_weight: float, ids: list[str], limited: bool) -> dict:
    if not examples or not tokens or not batches:
        raise ValueError("Cannot summarize an empty epoch")
    if len(ids) != len(set(ids)) or len(ids) != examples:
        raise ValueError("Epoch repeated or lost example identities")
    intent_loss, slot_loss = sums[0] / examples, sums[1] / tokens
    return {"loss": intent_weight * intent_loss + slot_weight * slot_loss,
            "intent_loss": intent_loss, "slot_loss": slot_loss,
            "num_examples": examples, "num_tokens": tokens, "num_batches": batches,
            "seconds": seconds, "record_ids": ids, "limited": limited}


def _validate_limit(max_batches: int | None) -> None:
    if max_batches is not None and (type(max_batches) is not int or max_batches <= 0):
        raise ValueError("max_batches must be a positive integer or None")


def train_epoch(model, loader, optimizer, device, *, intent_weight: float = 1.0, slot_weight: float = 1.0,
                grad_clip: float = 5.0, max_batches: int | None = None) -> dict:
    _validate_limit(max_batches)
    if not math.isfinite(grad_clip) or grad_clip <= 0:
        raise ValueError("Gradient clipping norm must be finite and positive")
    device = torch.device(device)
    model.train()
    _synchronize(device)
    start = perf_counter()
    intent_sum = slot_sum = 0.0
    examples = tokens = batches = 0
    maximum_norm = 0.0
    ids: list[str] = []
    for batch in loader:
        if max_batches is not None and batches >= max_batches:
            break
        if _partitions(batch) != {"train"}:
            raise ValueError("Parameter updates may consume training records only")
        batch = _device_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["input_ids"], batch["lengths"], batch["attention_mask"])
        losses = compute_losses(outputs, batch, intent_weight, slot_weight)
        losses.total.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
        maximum_norm = max(maximum_norm, float(norm))
        optimizer.step()
        intent_sum += float(losses.intent_sum.detach())
        slot_sum += float(losses.slot_sum.detach())
        examples += losses.num_examples
        tokens += losses.num_tokens
        batches += 1
        ids.extend(batch["record_ids"])
    _synchronize(device)
    result = _summarize((intent_sum, slot_sum), examples, tokens, batches, perf_counter() - start,
                        intent_weight, slot_weight, ids, max_batches is not None)
    result["max_gradient_norm"] = maximum_norm
    return result


def evaluate_model(model, loader, mappings: FittedMappings, device, *, intent_weight: float = 1.0,
                   slot_weight: float = 1.0, allow_test: bool = False, max_batches: int | None = None) -> dict:
    _validate_limit(max_batches)
    device = torch.device(device)
    model.eval()
    inverse_intents = {value: key for key, value in mappings.intent_to_id.items()}
    inverse_tags = {value: key for key, value in mappings.tag_to_id.items()}
    _synchronize(device)
    start = perf_counter()
    intent_sum = slot_sum = 0.0
    examples = tokens = batches = 0
    ids: list[str] = []
    predictions, gold = [], []
    observed_partitions: set[str] = set()
    with torch.inference_mode():
        for batch in loader:
            if max_batches is not None and batches >= max_batches:
                break
            partitions = _partitions(batch)
            if "test" in partitions and (not allow_test or max_batches is not None):
                raise ValueError("Test evaluation requires explicit permission and cannot be partial")
            if "test" in partitions and (len(loader.dataset) != EXPECTED_COUNTS["test"] or loader.drop_last):
                raise ValueError("Test evaluation requires the complete original test loader")
            observed_partitions.update(partitions)
            batch = _device_batch(batch, device)
            outputs = model(batch["input_ids"], batch["lengths"], batch["attention_mask"])
            losses = compute_losses(outputs, batch, intent_weight, slot_weight)
            predicted_intents = outputs["intent_logits"].argmax(-1).cpu().tolist()
            predicted_tags = outputs["slot_logits"].argmax(-1).cpu().tolist()
            for row, example in enumerate(batch["examples"]):
                length = int(batch["lengths"][row])
                tags = tuple(inverse_tags[index] for index in predicted_tags[row][:length])
                slots = decode_bio(example.utt, batch["tokens"][row], tags)
                predictions.append(IntentSlotPrediction(example.record_id, inverse_intents[predicted_intents[row]], slots))
                gold.append(example.gold_prediction())
            intent_sum += float(losses.intent_sum)
            slot_sum += float(losses.slot_sum)
            examples += losses.num_examples
            tokens += losses.num_tokens
            batches += 1
            ids.extend(batch["record_ids"])
    if len(observed_partitions) != 1:
        raise ValueError("Evaluation must contain one nonempty original partition")
    if "test" in observed_partitions and examples != EXPECTED_COUNTS["test"]:
        raise ValueError("Test evaluation did not visit the complete original split")
    metrics = evaluate_predictions(gold, predictions)
    _synchronize(device)
    result = _summarize((intent_sum, slot_sum), examples, tokens, batches, perf_counter() - start,
                        intent_weight, slot_weight, ids, max_batches is not None)
    result.update(metrics=metrics, predictions=tuple(predictions), partitions=sorted(observed_partitions), max_batches=max_batches)
    return result


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".part")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(path: Path, model, optimizer, mappings: FittedMappings, training_config: dict,
                    protocol: dict, epoch: int, validation: dict, *, source_hashes: dict | None = None,
                    data_sha256: str | None = None, purpose: str = "experiment") -> dict:
    if purpose not in {"diagnostic", "experiment"} or type(epoch) is not int or epoch < 1:
        raise ValueError("Invalid checkpoint purpose or epoch")
    if purpose == "experiment" and (validation["num_examples"] != EXPECTED_COUNTS["dev"]
                                     or validation.get("partitions") != ["dev"] or validation["limited"]):
        raise ValueError("Experiment checkpoints require the complete original dev evaluation")
    counts = enforce_parameter_cap(model, protocol["limits"]["total_parameters"])
    model_config, mapping_data = model.config.to_dict(), mappings.to_dict()
    score = selection_score(validation["metrics"].intent_accuracy, validation["metrics"].slot_micro_f1)
    payload = {
        "schema_version": 1, "purpose": purpose, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "epoch": epoch, "selection_score": score,
        "model_class": "JointBiLSTM", "model_config": model_config,
        "model_config_sha256": configuration_hash(model_config),
        "model_dimensions": {"vocab_size": len(mappings.token_to_id), "num_intents": len(mappings.intent_to_id),
                             "num_tags": len(mappings.tag_to_id)},
        "model_state": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
        "optimizer_state": copy.deepcopy(optimizer.state_dict()), "optimizer_class": type(optimizer).__name__,
        "mappings": mapping_data, "mappings_sha256": configuration_hash(mapping_data),
        "training_config": copy.deepcopy(training_config), "training_config_sha256": configuration_hash(training_config),
        "protocol": copy.deepcopy(protocol), "protocol_sha256": configuration_hash(protocol),
        "source_hashes": dict(source_hashes or {}), "data_sha256": data_sha256,
        "parameter_counts": counts, "torch_version": str(torch.__version__),
        "validation": {"loss": validation["loss"], "intent_loss": validation["intent_loss"],
                       "slot_loss": validation["slot_loss"], "num_examples": validation["num_examples"],
                       "metrics": validation["metrics"].to_dict(), "partitions": validation.get("partitions", []),
                       "limited": validation.get("limited", False), "max_batches": validation.get("max_batches")},
        "validation_predictions": [prediction.to_dict() for prediction in validation["predictions"]],
        "resume_support": "Optimizer state retained; exact training resumption is not implemented (RNG/loader states not restored).",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(_file_hash(path) + "\n", encoding="ascii")
    return payload


def load_checkpoint(path: Path, device, *, expected_mappings: FittedMappings | None = None,
                    expected_protocol: dict | None = None, expected_data_sha256: str | None = None):
    path = Path(path)
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file() or _file_hash(path) != sidecar.read_text(encoding="ascii").strip():
        raise ValueError("Checkpoint file hash is missing or does not match")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["schema_version"] != 1 or payload["model_class"] != "JointBiLSTM":
        raise ValueError("Unsupported checkpoint schema or model")
    for key in ("model_config", "mappings", "training_config", "protocol"):
        if configuration_hash(payload[key]) != payload[f"{key}_sha256"]:
            raise ValueError(f"Checkpoint {key} identity mismatch")
    mappings = FittedMappings.from_dict(payload["mappings"])
    if expected_mappings is not None and mappings.to_dict() != expected_mappings.to_dict():
        raise ValueError("Checkpoint mappings differ from expected training-fitted mappings")
    if expected_protocol is not None and configuration_hash(expected_protocol) != payload["protocol_sha256"]:
        raise ValueError("Checkpoint protocol differs from the expected protocol")
    if expected_data_sha256 is not None and expected_data_sha256 != payload["data_sha256"]:
        raise ValueError("Checkpoint dataset hash differs from the expected data")
    dimensions = {"vocab_size": len(mappings.token_to_id), "num_intents": len(mappings.intent_to_id),
                  "num_tags": len(mappings.tag_to_id)}
    if payload["model_dimensions"] != dimensions:
        raise ValueError("Checkpoint model dimensions differ from saved label mappings")
    model = JointBiLSTM(**dimensions, config=BaselineConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    if enforce_parameter_cap(model, payload["protocol"]["limits"]["total_parameters"]) != payload["parameter_counts"]:
        raise ValueError("Checkpoint parameter counts do not match the reconstructed model")
    model.to(torch.device(device)).eval()
    return model, mappings, payload


def verify_reloaded_checkpoint(path: Path, validation_loader, device, *, expected_mappings=None,
                                expected_protocol=None, expected_data_sha256=None) -> dict:
    model, mappings, payload = load_checkpoint(path, device, expected_mappings=expected_mappings,
                                              expected_protocol=expected_protocol,
                                              expected_data_sha256=expected_data_sha256)
    training_config = payload["training_config"]
    result = evaluate_model(model, validation_loader, mappings, device,
                            intent_weight=training_config.get("intent_loss_weight", 1.0),
                            slot_weight=training_config.get("slot_loss_weight", 1.0),
                            max_batches=payload["validation"].get("max_batches"))
    # Diagnostic callers must supply exactly the same finite evaluation view.
    observed = [prediction.to_dict() for prediction in result["predictions"]]
    if observed != payload["validation_predictions"]:
        raise ValueError("Reloaded checkpoint changed ordered validation predictions")
    score = selection_score(result["metrics"].intent_accuracy, result["metrics"].slot_micro_f1)
    difference = abs(score - payload["selection_score"])
    if difference > payload["protocol"]["selection"]["reload_score_atol"]:
        raise ValueError("Reloaded checkpoint did not reproduce the selected validation score")
    return {"passed": True, "epoch": payload["epoch"], "selection_score": score,
            "absolute_score_difference": difference, "ordered_predictions_equal": True,
            "num_examples": result["num_examples"], "purpose": payload["purpose"],
            "metrics": result["metrics"].to_dict()}


def plot_learning_curves(history: list[dict], path: Path, purpose: str) -> None:
    cache = path.parent / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache.resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [record["epoch"] for record in history]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, key, title in zip(axes.flat[:3], ("loss", "intent_loss", "slot_loss"),
                                ("Joint loss", "Intent loss", "Slot loss (real tokens)")):
        axis.plot(epochs, [record[f"train_{key}"] for record in history], label="Train")
        axis.plot(epochs, [record[f"dev_{key}"] for record in history], label="Dev")
        axis.set(title=title, xlabel="Epoch", ylabel="Loss")
        axis.legend()
    for key, label in (("dev_intent_accuracy", "Intent accuracy"), ("dev_slot_micro_f1", "Slot micro F1")):
        axes[1, 1].plot(epochs, [100 * record[key] for record in history], label=label)
    axes[1, 1].set(title="Dev metrics", xlabel="Epoch", ylabel="Percent")
    axes[1, 1].legend()
    figure.suptitle("Diagnostic execution checks (not final results)" if purpose == "diagnostic" else "Stage 1 learning curves")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def fit_model(model, train_loader, dev_loader, optimizer, mappings: FittedMappings, device,
              training_config: dict, protocol: dict, output_dir: Path, *, source_hashes=None,
              data_sha256=None, purpose: str = "experiment", max_train_batches=None, max_dev_batches=None) -> dict:
    if purpose not in {"experiment", "diagnostic"}:
        raise ValueError("Unknown training purpose")
    epochs = training_config["max_epochs"]
    if type(epochs) is not int or epochs < 1:
        raise ValueError("max_epochs must be positive")
    if purpose == "experiment" and (max_train_batches is not None or max_dev_batches is not None):
        raise ValueError("Partial-batch training is allowed only for diagnostics")
    for loader, partition in ((train_loader, "train"), (dev_loader, "dev")):
        if any(item.example.partition != partition for item in loader.dataset):
            raise ValueError(f"Expected original {partition} records")
        if purpose == "experiment" and (len(loader.dataset) != EXPECTED_COUNTS[partition] or loader.drop_last):
            raise ValueError("Experiments must use every original train/dev record")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in ("history.jsonl", "history.csv", "best.pt", "training_summary.json")):
        raise FileExistsError("Refusing to overwrite previous training evidence")
    counts = enforce_parameter_cap(model, protocol["limits"]["total_parameters"])
    print(f"Parameters: total={counts['total']:,}; trainable={counts['trainable']:,}; cap={protocol['limits']['total_parameters']:,}", flush=True)
    weights = {"intent_weight": training_config.get("intent_loss_weight", 1.0),
               "slot_weight": training_config.get("slot_loss_weight", 1.0)}
    history = []
    best_score, best_epoch = None, None
    checkpoint_path = output_dir / "best.pt"
    _synchronize(torch.device(device))
    workflow_start = perf_counter()
    for epoch in range(1, epochs + 1):
        train = train_epoch(model, train_loader, optimizer, device, **weights,
                            grad_clip=training_config.get("gradient_clip_norm", 5.0), max_batches=max_train_batches)
        validation = evaluate_model(model, dev_loader, mappings, device, **weights, max_batches=max_dev_batches)
        if purpose == "experiment":
            for result, loader, partition in ((train, train_loader, "train"), (validation, dev_loader, "dev")):
                expected_ids = {item.example.record_id for item in loader.dataset}
                if result["num_examples"] != EXPECTED_COUNTS[partition] or set(result["record_ids"]) != expected_ids:
                    raise ValueError(f"Experiment epoch did not visit every original {partition} record")
        score = selection_score(validation["metrics"].intent_accuracy, validation["metrics"].slot_micro_f1)
        checkpoint_seconds = 0.0
        selected = is_improvement(score, best_score)
        if selected:
            _synchronize(torch.device(device))
            checkpoint_start = perf_counter()
            save_checkpoint(checkpoint_path, model, optimizer, mappings, training_config, protocol, epoch, validation,
                            source_hashes=source_hashes, data_sha256=data_sha256, purpose=purpose)
            write_predictions(output_dir / "best_dev_predictions.jsonl", validation["predictions"])
            _synchronize(torch.device(device))
            checkpoint_seconds = perf_counter() - checkpoint_start
            best_score, best_epoch = score, epoch
        record = {"epoch": epoch, "purpose": purpose,
                  **{f"train_{key}": train[key] for key in ("loss", "intent_loss", "slot_loss", "num_examples", "num_tokens", "seconds")},
                  **{f"dev_{key}": validation[key] for key in ("loss", "intent_loss", "slot_loss", "num_examples", "num_tokens", "seconds")},
                  "dev_intent_accuracy": validation["metrics"].intent_accuracy,
                  "dev_slot_micro_f1": validation["metrics"].slot_micro_f1, "selection_score": score,
                  "selected_checkpoint": selected, "checkpoint_seconds": checkpoint_seconds,
                  "max_gradient_norm": train["max_gradient_norm"]}
        history.append(record)
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        with (output_dir / "history.csv").open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(record)
        print(f"Epoch {epoch}: train_loss={train['loss']:.6f} dev_loss={validation['loss']:.6f} "
              f"dev_intent={100 * validation['metrics'].intent_accuracy:.2f}% "
              f"dev_slots={100 * validation['metrics'].slot_micro_f1:.2f}% "
              f"selection={100 * score:.2f}%", flush=True)
    _synchronize(torch.device(device))
    training_seconds = perf_counter() - workflow_start
    summary = {"purpose": purpose, "epochs_trained": len(history), "best_epoch": best_epoch,
               "best_score": best_score, "checkpoint": str(checkpoint_path),
               "training_seconds": training_seconds, "parameter_counts": counts,
               "max_train_batches": max_train_batches, "max_dev_batches": max_dev_batches,
               "test_evaluated": False}
    _atomic_json(output_dir / "training_summary.json", summary)
    plot_learning_curves(history, output_dir / "learning_curves.png", purpose)
    print(f"Selected epoch {best_epoch}; validation score={100 * best_score:.2f}%; checkpoint={checkpoint_path.name}", flush=True)
    return {**summary, "history": history}


"""Development-only error analysis using saved predictions and the shared scorer.

This module neither imports a model nor performs inference. Training counts are
descriptive statistics; no development labels enter vocabulary fitting. Offset
categories explain unmatched scorer pairs and do not replace the shared metric.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from src.data import (EXPECTED_COUNTS, FittedMappings, UtteranceExample, fit_mappings,
                      parse_annotations, read_predictions, tokenize_with_offsets)
from src.metrics import evaluate_predictions
from src.schema import IntentSlotPrediction, SlotSpan


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_development(path: Path) -> dict[str, tuple[UtteranceExample, ...]]:
    """Read the original file, constructing labels/examples for train/dev only."""
    examples: dict[str, list[UtteranceExample]] = {"train": [], "dev": []}
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["partition"] not in examples:
            if row["partition"] != "test":
                raise ValueError("Unexpected original partition")
            continue
        if row["locale"] != "en-US" or row["id"] in seen:
            raise ValueError("Unexpected locale or duplicate train/dev record")
        seen.add(row["id"])
        examples[row["partition"]].append(UtteranceExample(
            row["id"], row["partition"], row["utt"], row["intent"], row["annot_utt"],
            parse_annotations(row["utt"], row["annot_utt"])))
    if any(len(values) != EXPECTED_COUNTS[key] for key, values in examples.items()):
        raise ValueError("Incomplete original train/dev partitions")
    return {key: tuple(value) for key, value in examples.items()}


def validate_development(
    development: Sequence[UtteranceExample], predictions: Sequence[IntentSlotPrediction],
    mappings: FittedMappings,
) -> None:
    if not development or any(row.partition != "dev" for row in development):
        raise ValueError("Diagnosis accepts development examples only")
    ids = [row.record_id for row in development]
    if len(set(ids)) != len(ids) or [row.record_id for row in predictions] != ids:
        raise ValueError("Predictions must cover every development ID exactly once in original order")
    for example, prediction in zip(development, predictions):
        if prediction.intent not in mappings.intent_to_id:
            raise ValueError("Prediction intent absent from train-only mappings")
        for slot in prediction.slots:
            if slot.slot_type not in mappings.slot_types or example.utt[slot.start:slot.end] != slot.text:
                raise ValueError("Predicted slot type or exact source slice is invalid")


def unique_pairs(slots: Sequence[SlotSpan]) -> dict[tuple[str, str], SlotSpan]:
    """Earliest occurrence represents duplicate scorer pairs for offset analysis."""
    pairs: dict[tuple[str, str], SlotSpan] = {}
    for slot in sorted(slots, key=lambda item: (item.start, item.end, item.slot_type)):
        pairs.setdefault((slot.slot_type, slot.text), slot)
    return pairs


def unmatched_category(slot: SlotSpan, opposite: Sequence[SlotSpan]) -> str:
    """Mutually exclusive category for one unmatched pair, not a 1:1 alignment.

    A span can explain several opposite errors. Counts sum separately to FP/FN;
    they must not be added as a count of underlying linguistic error events.
    """
    if any(other.start == slot.start and other.end == slot.end and
           other.slot_type != slot.slot_type for other in opposite):
        return "exact_boundary_wrong_type"
    overlap = [other for other in opposite if other.start < slot.end and slot.start < other.end]
    if any(other.slot_type == slot.slot_type for other in overlap):
        return "same_type_wrong_boundary"
    if overlap:
        return "overlap_wrong_type_and_boundary"
    return "no_overlap"


def pair_counts(gold: set[tuple[str, str]], predicted: set[tuple[str, str]]) -> dict[str, int | float]:
    tp, fp, fn = len(gold & predicted), len(predicted - gold), len(gold - predicted)
    return counts_metrics(tp, fp, fn)


def counts_metrics(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def analyze_predictions(
    training: Sequence[UtteranceExample], development: Sequence[UtteranceExample],
    predictions: Sequence[IntentSlotPrediction], mappings: FittedMappings,
) -> dict[str, Any]:
    """Compute reproducible associations; these are not causal intervention tests."""
    if not training or any(row.partition != "train" for row in training):
        raise ValueError("Training support statistics require train-only records")
    if fit_mappings(training).to_dict() != mappings.to_dict():
        raise ValueError("Mappings differ from original train-only fit")
    validate_development(development, predictions, mappings)
    gold = [row.gold_prediction() for row in development]
    overall = evaluate_predictions(gold, predictions).to_dict()
    training_tokens = Counter(token.text for row in training for token in tokenize_with_offsets(row.utt))
    training_intents = Counter(row.intent for row in training)
    training_slots = Counter(key[0] for row in training for key in unique_pairs(row.slots))
    training_surfaces = {key for row in training for key in unique_pairs(row.slots)}
    confusion: Counter[tuple[str, str]] = Counter()
    intent_counts: dict[str, Counter] = defaultdict(Counter)
    slot_counts: dict[str, Counter] = defaultdict(Counter)
    fn_categories: Counter[str] = Counter()
    fp_categories: Counter[str] = Counter()
    slot_surface_recall: dict[str, Counter] = defaultdict(Counter)
    gold_slot_length_recall: dict[str, Counter] = defaultdict(Counter)
    strata: dict[str, list[int]] = defaultdict(list)
    examples: list[dict[str, Any]] = []
    duplicate_gold_pairs = duplicate_predicted_pairs = total_oov_tokens = 0
    for index, (example, predicted) in enumerate(zip(development, predictions)):
        tokens = tokenize_with_offsets(example.utt)
        oov = [token.text for token in tokens if token.text not in mappings.token_to_id]
        rare = [token.text for token in tokens if 0 < training_tokens[token.text] <= 2]
        total_oov_tokens += len(oov)
        strata["oov/" + ("contains_oov" if oov else "no_oov")].append(index)
        strata["rare_tokens/" + ("contains_train_frequency_1_or_2" if rare else "none")].append(index)
        strata["length/" + ("1_to_5" if len(tokens) <= 5 else "6_to_10" if len(tokens) <= 10 else "11_plus")].append(index)
        train_intent_count = training_intents[example.intent]
        strata["intent_support/" + ("1_to_100" if train_intent_count <= 100 else "101_to_300" if train_intent_count <= 300 else "301_plus")].append(index)
        intent_correct = example.intent == predicted.intent
        strata["intent_result/" + ("correct" if intent_correct else "incorrect")].append(index)
        intent_counts[example.intent]["gold"] += 1
        intent_counts[predicted.intent]["predicted"] += 1
        if intent_correct:
            intent_counts[example.intent]["tp"] += 1
        else:
            confusion[(example.intent, predicted.intent)] += 1
        gs, ps = unique_pairs(example.slots), unique_pairs(predicted.slots)
        duplicate_gold_pairs += len(example.slots) - len(gs)
        duplicate_predicted_pairs += len(predicted.slots) - len(ps)
        gkeys, pkeys = set(gs), set(ps)
        missing = [gs[key] for key in sorted(gkeys - pkeys)]
        extra = [ps[key] for key in sorted(pkeys - gkeys)]
        missing_details = [{**slot.to_dict(), "category": unmatched_category(slot, extra)} for slot in missing]
        extra_details = [{**slot.to_dict(), "category": unmatched_category(slot, missing)} for slot in extra]
        fn_categories.update(item["category"] for item in missing_details)
        fp_categories.update(item["category"] for item in extra_details)
        for key in gkeys | pkeys:
            slot_counts[key[0]]["tp" if key in gkeys & pkeys else "fn" if key in gkeys else "fp"] += 1
        for key in gs:
            outcome = "recovered" if key in ps else "missed"
            slot_surface_recall["seen_in_train" if key in training_surfaces else "unseen_in_train"][outcome] += 1
            n_tokens = len(tokenize_with_offsets(key[1]))
            gold_slot_length_recall["one_token" if n_tokens == 1 else "two_tokens" if n_tokens == 2 else "three_plus_tokens"][outcome] += 1
        if not intent_correct or missing or extra:
            examples.append({"record_id": example.record_id, "utt": example.utt,
                             "gold_intent": example.intent, "predicted_intent": predicted.intent,
                             "gold_slots": [slot.to_dict() for slot in example.slots],
                             "predicted_slots": [slot.to_dict() for slot in predicted.slots],
                             "oov_tokens": oov, "rare_train_tokens": rare,
                             "false_negative_pairs": missing_details, "false_positive_pairs": extra_details})
    metrics_by_stratum = {key: evaluate_predictions([gold[i] for i in indices],
                                                   [predictions[i] for i in indices]).to_dict()
                         for key, indices in sorted(strata.items())}
    per_intent = []
    for label in mappings.intent_to_id:
        counts = intent_counts[label]
        tp, support, predicted = counts["tp"], counts["gold"], counts["predicted"]
        per_intent.append({"intent": label, "train_support": training_intents[label], "dev_support": support,
                           "predicted_count": predicted, **counts_metrics(tp, predicted - tp, support - tp)})
    per_slot = [{"slot_type": label, "train_gold_pair_support": training_slots[label],
                 "dev_gold_pair_support": slot_counts[label]["tp"] + slot_counts[label]["fn"],
                 **counts_metrics(slot_counts[label]["tp"], slot_counts[label]["fp"], slot_counts[label]["fn"])}
                for label in mappings.slot_types]
    rare_slot_groups: dict[str, Counter] = defaultdict(Counter)
    for row in per_slot:
        count = row["train_gold_pair_support"]
        group = "1_to_50" if count <= 50 else "51_to_200" if count <= 200 else "201_plus"
        for key in ("tp", "fp", "fn"):
            rare_slot_groups[group][key] += row[key]
    def recall_rows(groups: dict[str, Counter]) -> dict[str, Any]:
        return {key: {"recovered": value["recovered"], "missed": value["missed"],
                      "gold_pair_support": sum(value.values()),
                      "recall": value["recovered"] / sum(value.values())}
                for key, value in sorted(groups.items())}
    if sum(fn_categories.values()) != overall["slot_fn"] or sum(fp_categories.values()) != overall["slot_fp"]:
        raise AssertionError("Diagnostic unmatched-pair counts do not reconcile with shared scorer")
    return {"overall": overall, "total_development_tokens": sum(len(tokenize_with_offsets(row.utt)) for row in development),
            "oov_token_occurrences": total_oov_tokens, "strata": metrics_by_stratum,
            "intent_confusions": [{"gold_intent": pair[0], "predicted_intent": pair[1], "count": count}
                                  for pair, count in sorted(confusion.items(), key=lambda item: (-item[1], item[0]))],
            "per_intent": per_intent, "per_slot_type": per_slot,
            "slot_support_groups": {key: counts_metrics(value["tp"], value["fp"], value["fn"])
                                    for key, value in sorted(rare_slot_groups.items())},
            "slot_surface_recall": recall_rows(slot_surface_recall),
            "gold_slot_length_recall": recall_rows(gold_slot_length_recall),
            "false_negative_categories": dict(sorted(fn_categories.items())),
            "false_positive_categories": dict(sorted(fp_categories.items())),
            "duplicate_gold_pairs_collapsed": duplicate_gold_pairs,
            "duplicate_predicted_pairs_collapsed": duplicate_predicted_pairs,
            "utterances_with_any_error": len(examples), "error_examples": examples,
            "illegal_bio_transitions": {"available": False, "reason": "Saved predictions contain decoded spans, not raw token tags. The decoder repairs orphan/type-changing I tags, so pre-decoding invalid transitions cannot be reconstructed."}}


def run_diagnosis(root: Path) -> dict[str, Any]:
    """Validate immutable baseline evidence before producing a dev-only diagnosis."""
    root = root.resolve()
    selection_path = root / "artifacts/phase4/selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["selection_split"] != "dev" or selection["test_used_for_selection"] is not False:
        raise ValueError("Baseline must have been selected using development data only")
    run = (root / selection["selected_run"]).resolve()
    if not run.is_relative_to(root / "artifacts/runs"):
        raise ValueError("Selected baseline run is outside the project runs directory")
    paths = {"selection": selection_path, "dataset": root / "data/1.0/data/en-US.jsonl",
             "mappings": root / "artifacts/phase2/preprocessing.json",
             "membership": root / "artifacts/phase2/split_membership.json",
             "predictions": root / "artifacts/phase4/final_baseline/dev_predictions.jsonl",
             "selected_run_predictions": run / "best_dev_predictions.jsonl", "history": run / "history.jsonl",
             "diagnosis_source": root / "src/diagnostics.py", "diagnosis_entry_point": root / "scripts/analyze_development.py",
             "shared_data_source": root / "src/data.py", "shared_metrics_source": root / "src/metrics.py",
             "prediction_schema_source": root / "src/schema.py"}
    membership = json.loads(paths["membership"].read_text(encoding="utf-8"))
    if sha256(paths["dataset"]) != membership["dataset_sha256"]:
        raise ValueError("Original dataset hash does not match the audited membership")
    if sha256(paths["predictions"]) != sha256(paths["selected_run_predictions"]):
        raise ValueError("Frozen development predictions differ from selected-run predictions")
    splits = load_training_development(paths["dataset"])
    for name, examples in splits.items():
        if [row.record_id for row in examples] != membership["record_ids"][name]:
            raise ValueError("Original train/dev membership or ordering has changed")
    mappings = FittedMappings.from_dict(json.loads(paths["mappings"].read_text(encoding="utf-8")))
    result = analyze_predictions(splits["train"], splits["dev"], read_predictions(paths["predictions"]), mappings)
    mean = (result["overall"]["intent_accuracy"] + result["overall"]["slot_micro_f1"]) / 2
    if mean != selection["selection_score"]:
        raise ValueError("Recomputed baseline development score differs from committed selection")
    history = [json.loads(line) for line in paths["history"].read_text(encoding="utf-8").splitlines()]
    if [row["epoch"] for row in history] != list(range(1, selection["epochs_trained"] + 1)):
        raise ValueError("Incomplete baseline learning history")
    if any(row["train_num_examples"] != EXPECTED_COUNTS["train"] or
           row["dev_num_examples"] != EXPECTED_COUNTS["dev"] for row in history):
        raise ValueError("Learning history contains incomplete partitions")
    selected = max(history, key=lambda row: row["selection_score"])
    if selected["epoch"] != selection["selected_epoch"] or selected["selection_score"] != mean:
        raise ValueError("Baseline history does not reproduce selection")
    minimum = min(history, key=lambda row: row["dev_loss"])
    result.update({"schema_version": 1, "purpose": "stage2_development_diagnosis",
                   "analysis_partition": "dev", "training_statistics_partition": "train",
                   "test_used": False, "model_inference_performed": False,
                   "baseline_run": selection["selected_run"], "baseline_selected_epoch": selection["selected_epoch"],
                   "inputs": {key: {"path": str(path.relative_to(root)).replace("\\", "/"), "sha256": sha256(path)}
                              for key, path in paths.items()},
                   "learning_curve": {"minimum_development_loss_epoch": minimum,
                                      "selected_epoch": selected, "last_epoch": history[-1],
                                      "interpretation": "Train loss falls while dev cross entropy rises after its minimum. This supports overfitting/overconfidence concerns, but the preregistered metric mean correctly selects epoch 19 rather than minimum loss."},
                   "offset_category_semantics": "Classify each unmatched per-record type/text pair separately using unmatched opposite spans. Priority: exact boundary/wrong type, same type/overlap/wrong boundary, other overlap, no overlap. These counts reconcile separately to FN and FP, are not one-to-one error events, and never alter the frozen project scorer. Duplicate pairs use earliest occurrence for offset description.",
                   "strata_semantics": "Each family partitions development examples. Metrics are shared-scorer micro counts inside each stratum, not per-example average F1. Associations can reflect other correlated difficulty factors; no causal improvement claim is made.",
                   "recommendations": [
                       {"change": "Train-only character CNN features concatenated to word embeddings", "evidence": "Large OOV versus no-OOV development gap and unseen slot surfaces", "target": "OOV representation and slot recovery", "cost": "Additional train-fitted character embedding/CNN and wider BiLSTM input; exact complete-system count must be measured before training", "criterion": "Compare unrounded dev mean and OOV strata against fixed baseline; retain regressions"},
                       {"change": "Constrained BIO linear-chain CRF within the same joint model", "evidence": "Many unmatched gold pairs overlap same-type predictions with incorrect boundaries", "target": "Coherent multi-token slot boundaries", "cost": "111x111 transitions plus 111 start/end values if all registered; decoding is quadratic in tags", "criterion": "Controlled ablation, exact shared slot F1 and mean; do not claim raw illegal BIO counts are known"},
                       {"change": "Condition slot head on soft predicted intent probabilities", "evidence": "Slot type mistakes and reduced slot performance when predicted intent is wrong", "target": "Task consistency and contextual slot disambiguation", "cost": "Small projection/head expansion; use predictions only at train and inference, never gold intent input", "criterion": "Separate ablation before a justified combination; compare both tasks and preserve negative results"}]})
    return result


def render_markdown(result: dict[str, Any]) -> str:
    overall = result["overall"]
    text = ["# Phase 5 baseline development diagnosis", "",
            "This analysis uses the original training partition for support/vocabulary statistics and all 2,033 original development examples for errors. It reads saved baseline development predictions, performs no model inference, and does not inspect test predictions or metrics.", "",
            f"Baseline: epoch {result['baseline_selected_epoch']}; intent accuracy **{100 * overall['intent_accuracy']:.2f}%**; exact slot micro F1 **{100 * overall['slot_micro_f1']:.2f}%**. Slot TP/FP/FN = **{overall['slot_tp']}/{overall['slot_fp']}/{overall['slot_fn']}**. The shared evaluator exactly reproduces the committed baseline selection score.", "",
            "## Representation and support", "", "| Development stratum | Utterances | Intent accuracy | Slot micro F1 |", "| --- | ---: | ---: | ---: |"]
    for key, row in result["strata"].items():
        text.append(f"| {key} | {row['num_examples']} | {100 * row['intent_accuracy']:.2f}% | {100 * row['slot_micro_f1']:.2f}% |")
    text += ["", f"{result['oov_token_occurrences']} of {result['total_development_tokens']} development token occurrences are absent from the train-only vocabulary. Strata are associations, not controlled causal estimates; rare words, long inputs and entity difficulty can overlap.", "",
             "| Gold slot surface | Gold pairs | Recovered | Recall |", "| --- | ---: | ---: | ---: |"]
    for label, row in result["slot_surface_recall"].items():
        text.append(f"| {label} | {row['gold_pair_support']} | {row['recovered']} | {100 * row['recall']:.2f}% |")
    text += ["", "| Gold slot length | Gold pairs | Recovered | Recall |", "| --- | ---: | ---: | ---: |"]
    for label, row in result["gold_slot_length_recall"].items():
        text.append(f"| {label} | {row['gold_pair_support']} | {row['recovered']} | {100 * row['recall']:.2f}% |")
    text += ["", "| Training slot-pair support per type | TP | FP | FN | Slot micro F1 |", "| --- | ---: | ---: | ---: | ---: |"]
    for label, row in result["slot_support_groups"].items():
        text.append(f"| {label} | {row['tp']} | {row['fp']} | {row['fn']} | {100 * row['f1']:.2f}% |")
    text += ["", "The train-unseen slot-surface recall gap and multi-token boundary gap support character features and structured decoding as controlled hypotheses. Low-support intents do not uniformly underperform, and slot support groups are not monotonic. This evidence does not justify blanket class reweighting by itself. Small per-label supports also make individual percentages unstable."]
    text += ["", "## Intent confusions", "", "| Gold intent | Predicted intent | Errors |", "| --- | --- | ---: |"]
    for row in result["intent_confusions"][:15]:
        text.append(f"| {row['gold_intent']} | {row['predicted_intent']} | {row['count']} |")
    text += ["", "## Slot error categories", "", result["offset_category_semantics"], "", "| Category | False negatives | False positives |", "| --- | ---: | ---: |"]
    for category in sorted(set(result["false_negative_categories"]) | set(result["false_positive_categories"])):
        text.append(f"| {category} | {result['false_negative_categories'].get(category, 0)} | {result['false_positive_categories'].get(category, 0)} |")
    text += ["", result["illegal_bio_transitions"]["reason"], "",
             "| Slot type (largest FN counts) | Train support | Dev gold support | TP | FP | FN | F1 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in sorted(result["per_slot_type"], key=lambda r: (-r["fn"], r["slot_type"]))[:15]:
        text.append(f"| {row['slot_type']} | {row['train_gold_pair_support']} | {row['dev_gold_pair_support']} | {row['tp']} | {row['fp']} | {row['fn']} | {100 * row['f1']:.2f}% |")
    curve = result["learning_curve"]
    minimum, selected, last = curve["minimum_development_loss_epoch"], curve["selected_epoch"], curve["last_epoch"]
    text += ["", "## Learning curves", "",
             f"Development combined cross entropy is lowest at epoch {minimum['epoch']} ({minimum['dev_loss']:.6f}), then reaches {selected['dev_loss']:.6f} at selected epoch {selected['epoch']} and {last['dev_loss']:.6f} at epoch {last['epoch']}. Train loss falls from {minimum['train_loss']:.6f} at epoch {minimum['epoch']} to {last['train_loss']:.6f} at epoch {last['epoch']}. This is consistent with overfitting or increasing overconfidence. It does not invalidate epoch {selected['epoch']}: the declared selection metric is the unrounded mean of intent accuracy and slot F1, which is highest there. Extra capacity alone is not an evidence-based remedy.", "",
             "## Genuine baseline examples", "", "Examples below are original development records with saved baseline predictions. Selection demonstrates different observed failure modes; it is not a random estimate of prevalence."]
    chosen: list[dict[str, Any]] = []
    for predicate in (
            lambda row: bool(row["oov_tokens"]) and bool(row["false_negative_pairs"]),
            lambda row: any(s["category"] == "same_type_wrong_boundary" for s in row["false_negative_pairs"]),
            lambda row: any(s["category"] == "exact_boundary_wrong_type" for s in row["false_negative_pairs"]),
            lambda row: row["gold_intent"] != row["predicted_intent"]):
        candidate = next((row for row in result["error_examples"] if predicate(row) and row not in chosen), None)
        if candidate:
            chosen.append(candidate)
    for row in chosen:
        text += ["", f"### Record {row['record_id']}", "", f"Utterance: `{row['utt']}`", "",
                 f"Intent: gold `{row['gold_intent']}`; predicted `{row['predicted_intent']}`.", "",
                 f"Gold slots: `{json.dumps(row['gold_slots'], ensure_ascii=False)}`", "",
                 f"Predicted slots: `{json.dumps(row['predicted_slots'], ensure_ascii=False)}`", "",
                 f"Unseen tokens: `{json.dumps(row['oov_tokens'], ensure_ascii=False)}`. Missing-pair categories: `{json.dumps([s['category'] for s in row['false_negative_pairs']])}`."]
        categories = {slot["category"] for slot in row["false_negative_pairs"]}
        if row["oov_tokens"] and "same_type_wrong_boundary" in categories:
            interpretation = "The predicted slot has the correct type but omits part of its gold surface. An unfamiliar word occurs in the utterance; this motivates testing character features and span coherence without claiming either will recover an unseen title."
        elif "same_type_wrong_boundary" in categories:
            interpretation = "The saved prediction includes a partial same-type overlap, but exact slot scoring rejects incomplete surfaces. Local tag decisions can fragment a span; a structured decoder is a testable remedy, not a guaranteed correction."
        elif "exact_boundary_wrong_type" in categories:
            interpretation = "The surface boundaries are correct and the slot type is wrong. Sequence consistency alone need not fix semantic type distinctions; task context or remaining label ambiguity can matter."
        else:
            interpretation = "The intent is wrong despite correct slot extraction in this case. Better slot decoding alone need not solve intent polarity or utterance-level meaning."
        text += ["", interpretation]
    text += ["", "## Justified controlled experiments", ""]
    for number, row in enumerate(result["recommendations"], 1):
        text += [f"{number}. **{row['change']}.** Evidence: {row['evidence']}. Target: {row['target']}. Cost: {row['cost']}. Acceptance: {row['criterion']}.", ""]
    text += ["Retain one joint model, the original partition/scoring rules, complete parameter accounting below 20M, and development-only selection. Compare one major change at a time before combinations. These are hypotheses rather than promises of improvement; exact budgets and parameter counts belong in the preregistered Stage 2 experiment plan.", "",
             "## Reproduction and evidence", "", "Run `python scripts/analyze_development.py`. Full precision metrics, every intent and slot type, all actual error examples, input hashes, category semantics, and learning-curve records are in [diagnosis.json](artifacts/phase5/diagnosis.json). Source: [diagnostics.py](src/diagnostics.py). The source validates audited dataset identity, exact original train/dev membership/order, a fresh train-only mapping fit, prediction identity/order, valid exact source spans, complete learning history and committed selection-score equality.", ""]
    return "\n".join(text)

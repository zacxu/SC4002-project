"""Shared intent accuracy and exact slot-pair micro F1.

Scoring convention frozen for this project: each utterance has a *set* of
``(slot_type, exact_surface_text)`` pairs. Strings are compared without case,
whitespace, or Unicode normalization. Offsets remain in predictions for audit
and reconstruction but are not part of a scoring key. Repeated identical pairs
within one utterance collapse; the same pair in different utterances counts
separately. This is the project's interpretation of the assignment's requested
set of [type:span] pairs, not a claim about an official MASSIVE evaluator.

Scores are fractions. Multiply by 100 only for presentation. A split with no
gold or predicted slots receives slot F1 0.0; an empty split is an error.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

from src.schema import IntentSlotPrediction


@dataclass(frozen=True)
class EvaluationResult:
    intent_accuracy: float
    slot_micro_f1: float
    slot_tp: int
    slot_fp: int
    slot_fn: int
    num_examples: int
    correct_intents: int

    def to_dict(self) -> dict[str, float | int]:
        """Return full-precision metrics and integer counts for JSON artifacts."""
        return asdict(self)


def _index_records(
    records: Iterable[IntentSlotPrediction], source: str,
) -> dict[str, IntentSlotPrediction]:
    indexed: dict[str, IntentSlotPrediction] = {}
    for record in records:
        if not isinstance(record, IntentSlotPrediction):
            raise TypeError(f"{source} must contain IntentSlotPrediction values")
        if record.record_id in indexed:
            raise ValueError(f"Duplicate {source} record ID: {record.record_id!r}")
        indexed[record.record_id] = record
    return indexed


def evaluate_predictions(
    gold: Iterable[IntentSlotPrediction],
    predictions: Iterable[IntentSlotPrediction],
) -> EvaluationResult:
    """Score a complete split, joining predictions to gold by record ID.

    Every gold ID must have exactly one prediction and no unexpected prediction
    IDs are permitted. Input ordering is irrelevant. Counts are pooled across
    utterances before F1 is computed; slots cannot match across utterances.
    """
    gold_by_id = _index_records(gold, "gold")
    predicted_by_id = _index_records(predictions, "prediction")
    if not gold_by_id:
        raise ValueError("Cannot evaluate an empty gold split")

    missing_ids = gold_by_id.keys() - predicted_by_id.keys()
    unexpected_ids = predicted_by_id.keys() - gold_by_id.keys()
    if missing_ids or unexpected_ids:
        raise ValueError(
            "Prediction IDs do not match gold IDs: "
            f"missing={sorted(missing_ids)!r}, unexpected={sorted(unexpected_ids)!r}"
        )

    correct_intents = slot_tp = slot_fp = slot_fn = 0
    for record_id, gold_record in gold_by_id.items():
        predicted = predicted_by_id[record_id]
        correct_intents += int(predicted.intent == gold_record.intent)
        gold_pairs = {(slot.slot_type, slot.text) for slot in gold_record.slots}
        predicted_pairs = {(slot.slot_type, slot.text) for slot in predicted.slots}
        slot_tp += len(gold_pairs & predicted_pairs)
        slot_fp += len(predicted_pairs - gold_pairs)
        slot_fn += len(gold_pairs - predicted_pairs)

    denominator = 2 * slot_tp + slot_fp + slot_fn
    return EvaluationResult(
        intent_accuracy=correct_intents / len(gold_by_id),
        slot_micro_f1=2 * slot_tp / denominator if denominator else 0.0,
        slot_tp=slot_tp,
        slot_fp=slot_fp,
        slot_fn=slot_fn,
        num_examples=len(gold_by_id),
        correct_intents=correct_intents,
    )


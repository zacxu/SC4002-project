"""Evaluation v2: exact typed character-span occurrences, joined by record ID.

The canonical implementation is shared with both standalone notebooks. Repeated
surface values at distinct positions remain separate occurrences. Exact text,
type and original character boundaries must match; there is no normalization.
Historical v1 text-set results/checkpoint selections are not v2 results.
Scores are fractions; presentation alone multiplies by 100.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

from src.schema import IntentSlotPrediction
from src.comparison_contract import EVALUATOR_VERSION, score_exact_spans


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

    return EvaluationResult(**score_exact_spans(gold_by_id.values(), predicted_by_id.values()))


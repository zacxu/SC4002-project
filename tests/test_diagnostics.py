"""Hand-counted diagnostic cases; these fixtures are never training data."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.data import UtteranceExample, fit_mappings
from src.diagnostics import (analyze_predictions, load_training_development,
                             unmatched_category, unique_pairs, validate_development)
from src.schema import IntentSlotPrediction, SlotSpan


def example(record_id="1", partition="dev", utt="red blue", intent="a", slots=()):
    return UtteranceExample(record_id, partition, utt, intent, utt, tuple(slots))


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.training = (example("t1", "train", slots=(SlotSpan("color", "red", 0, 3),)),
                         example("t2", "train", intent="b", slots=(SlotSpan("shade", "blue", 4, 8),)))
        self.mappings = fit_mappings(self.training)

    def test_refuses_test_and_train_examples_for_diagnosis(self):
        for partition in ("train", "test"):
            row = example(partition=partition)
            with self.assertRaisesRegex(ValueError, "development"):
                validate_development((row,), (row.gold_prediction(),), self.mappings)

    def test_refuses_missing_duplicate_and_reordered_ids(self):
        rows = (example("1"), example("2"))
        for predictions in ((rows[0].gold_prediction(),),
                            (rows[0].gold_prediction(), rows[0].gold_prediction()),
                            (rows[1].gold_prediction(), rows[0].gold_prediction())):
            with self.assertRaisesRegex(ValueError, "original order"):
                validate_development(rows, predictions, self.mappings)

    def test_rejects_non_source_slot_text(self):
        row = example()
        prediction = IntentSlotPrediction("1", "a", (SlotSpan("color", "xyz", 0, 3),))
        with self.assertRaisesRegex(ValueError, "exact source slice"):
            validate_development((row,), (prediction,), self.mappings)

    def test_category_priority_and_overlap(self):
        gold = SlotSpan("color", "red blue", 0, 8)
        self.assertEqual(unmatched_category(gold, [SlotSpan("shade", "red blue", 0, 8)]), "exact_boundary_wrong_type")
        self.assertEqual(unmatched_category(gold, [SlotSpan("color", "red", 0, 3)]), "same_type_wrong_boundary")
        self.assertEqual(unmatched_category(gold, [SlotSpan("shade", "red", 0, 3)]), "overlap_wrong_type_and_boundary")
        self.assertEqual(unmatched_category(gold, [SlotSpan("color", "x", 9, 10)]), "no_overlap")

    def test_repeated_surface_occurrences_are_scored_separately(self):
        slots = (SlotSpan("color", "red", 4, 7), SlotSpan("color", "red", 0, 3))
        self.assertEqual(unique_pairs(slots), {
            ("color", "red", item.start, item.end): item for item in slots})
        row = example(utt="red red", slots=slots)
        prediction = IntentSlotPrediction("1", "a", slots[:1])
        result = analyze_predictions(self.training, (row,), (prediction,), self.mappings)
        self.assertEqual(result["overall"]["slot_tp"], 1)
        self.assertEqual(result["duplicate_gold_pairs_collapsed"], 0)
        self.assertEqual(result["overall"]["slot_fn"], 1)
        self.assertEqual(result["false_negative_categories"], {"no_overlap": 1})

    def test_hand_counted_oov_and_unmatched_counts_reconcile(self):
        rows = (example("1", slots=(SlotSpan("color", "red blue", 0, 8),)),
                example("2", utt="red green", intent="b", slots=(SlotSpan("shade", "green", 4, 9),)))
        predictions = (IntentSlotPrediction("1", "a", (SlotSpan("color", "red", 0, 3),)),
                       IntentSlotPrediction("2", "a", ()))
        result = analyze_predictions(self.training, rows, predictions, self.mappings)
        self.assertEqual(result["overall"]["slot_tp"], 0)
        self.assertEqual(result["overall"]["slot_fp"], 1)
        self.assertEqual(result["overall"]["slot_fn"], 2)
        self.assertEqual(result["false_negative_categories"], {"no_overlap": 1, "same_type_wrong_boundary": 1})
        self.assertEqual(result["oov_token_occurrences"], 1)
        self.assertEqual(result["strata"]["oov/contains_oov"]["num_examples"], 1)
        self.assertEqual(result["strata"]["oov/no_oov"]["intent_accuracy"], 1.0)
        self.assertFalse(result["illegal_bio_transitions"]["available"])

    def test_rejects_vocabulary_fitted_on_other_data(self):
        alternate = fit_mappings((replace(self.training[0], utt="different"), self.training[1]))
        row = example()
        with self.assertRaisesRegex(ValueError, "train-only fit"):
            analyze_predictions(self.training, (row,), (row.gold_prediction(),), alternate)

    def test_training_statistics_reject_nontraining_records(self):
        row = example()
        with self.assertRaisesRegex(ValueError, "train-only records"):
            analyze_predictions((row,), (row,), (row.gold_prediction(),), self.mappings)

    def test_loader_does_not_construct_test_targets(self):
        rows = [{"id": "t", "partition": "train", "locale": "en-US", "utt": "red",
                 "intent": "a", "annot_utt": "red"},
                {"partition": "test", "intentionally_no_label_or_text_fields": True},
                {"id": "d", "partition": "dev", "locale": "en-US", "utt": "blue",
                 "intent": "b", "annot_utt": "blue"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "original.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with patch("src.diagnostics.EXPECTED_COUNTS", {"train": 1, "dev": 1}):
                result = load_training_development(path)
        self.assertEqual(set(result), {"train", "dev"})
        self.assertEqual(result["dev"][0].record_id, "d")

    def test_loader_rejects_incomplete_original_partitions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                load_training_development(path)


if __name__ == "__main__":
    unittest.main()

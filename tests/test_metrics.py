"""Hand-counted evaluator inputs; these are not task training examples."""

from dataclasses import FrozenInstanceError
import unittest

from src.metrics import evaluate_predictions
from src.schema import IntentSlotPrediction, SlotSpan


def span(slot_type="type-A", text="x", start=0):
    return SlotSpan(slot_type, text, start, start + len(text))


def record(record_id="r1", intent="intent-A", slots=()):
    return IntentSlotPrediction(record_id, intent, tuple(slots))


class SharedEvaluatorTests(unittest.TestCase):
    def test_hand_counted_micro_counts_and_intent_accuracy(self):
        gold = [record("r1", slots=(span(), span("type-B", "y", 2))),
                record("r2", "intent-B", (span(), span(start=4))),
                record("r3", "intent-C")]
        predictions = [record("r1", slots=(span(), span("type-B", "z", 2))),
                       record("r2", "intent-A"),
                       record("r3", "intent-C", (span(),))]
        result = evaluate_predictions(gold, predictions)
        self.assertEqual((result.slot_tp, result.slot_fp, result.slot_fn), (1, 2, 2))
        self.assertEqual((result.num_examples, result.correct_intents), (3, 2))
        self.assertAlmostEqual(result.intent_accuracy, 2 / 3)
        self.assertAlmostEqual(result.slot_micro_f1, 1 / 3)
        self.assertEqual(result.to_dict()["slot_fn"], 2)
        self.assertEqual(result.to_dict()["intent_accuracy"], 2 / 3)
        with self.assertRaises(FrozenInstanceError):
            result.slot_tp = 7

    def test_gold_self_scoring_is_perfect_when_slots_exist(self):
        gold = [record("r1", slots=(span(),)), record("r2", "intent-B")]
        result = evaluate_predictions(gold, gold)
        self.assertEqual(result.intent_accuracy, 1.0)
        self.assertEqual(result.slot_micro_f1, 1.0)
        self.assertEqual((result.slot_tp, result.slot_fp, result.slot_fn), (1, 0, 0))

    def test_order_is_irrelevant_and_single_pass_iterables_work(self):
        gold = [record("r1", slots=(span(),)), record("r2", "intent-B")]
        ordered = evaluate_predictions(gold, gold)
        shuffled = evaluate_predictions(iter(gold), reversed(gold))
        self.assertEqual(ordered, shuffled)

    def test_duplicate_pairs_collapse_within_each_record_in_both_inputs(self):
        gold = [record(slots=(span(), span(start=4)))]
        predictions = [record(slots=(span(start=8), span(start=10)))]
        result = evaluate_predictions(gold, predictions)
        self.assertEqual((result.slot_tp, result.slot_fp, result.slot_fn), (1, 0, 0))
        self.assertEqual(result.slot_micro_f1, 1.0)

    def test_matching_pairs_in_different_records_count_separately(self):
        gold = [record("r1", slots=(span(),)), record("r2", slots=(span(),))]
        predictions = [record("r1", slots=(span(),)), record("r2")]
        result = evaluate_predictions(gold, predictions)
        self.assertEqual((result.slot_tp, result.slot_fp, result.slot_fn), (1, 0, 1))
        self.assertAlmostEqual(result.slot_micro_f1, 2 / 3)

    def test_slots_cannot_match_across_records(self):
        gold = [record("r1", slots=(span(),)), record("r2")]
        predictions = [record("r1"), record("r2", slots=(span(),))]
        result = evaluate_predictions(gold, predictions)
        self.assertEqual((result.slot_tp, result.slot_fp, result.slot_fn), (0, 1, 1))

    def test_exact_type_and_full_surface_text_are_both_required(self):
        gold = [record(slots=(span(text="xy"),))]
        for wrong_slot in (span("type-B", "xy"), span(text="x"), span(text="xyz"),
                           span(text="XY"), span(text="xy "), span(text=" xy")):
            with self.subTest(slot=wrong_slot):
                result = evaluate_predictions(gold, [record(slots=(wrong_slot,))])
                self.assertEqual((result.slot_tp, result.slot_fp, result.slot_fn), (0, 1, 1))
                self.assertEqual(result.slot_micro_f1, 0.0)

    def test_unicode_text_is_not_normalized(self):
        gold = [record(slots=(span(text="\u00e9"),))]
        predictions = [record(slots=(span(text="e\u0301"),))]
        result = evaluate_predictions(gold, predictions)
        self.assertEqual((result.slot_tp, result.slot_fp, result.slot_fn), (0, 1, 1))

    def test_intent_comparison_is_exact(self):
        result = evaluate_predictions([record()], [record(intent="Intent-A")])
        self.assertEqual(result.intent_accuracy, 0.0)
        self.assertEqual(result.correct_intents, 0)

    def test_missing_and_extra_slots_have_distinct_counts(self):
        cases = [([record(slots=(span(),))], [record()], (0, 0, 1)),
                 ([record()], [record(slots=(span(),))], (0, 1, 0))]
        for gold, predictions, counts in cases:
            with self.subTest(counts=counts):
                result = evaluate_predictions(gold, predictions)
                self.assertEqual((result.slot_tp, result.slot_fp, result.slot_fn), counts)
                self.assertEqual(result.slot_micro_f1, 0.0)

    def test_all_no_slot_predictions_receive_zero_slot_f1(self):
        result = evaluate_predictions([record("r1"), record("r2")],
                                      [record("r1"), record("r2")])
        self.assertEqual(result.slot_micro_f1, 0.0)
        self.assertEqual(result.intent_accuracy, 1.0)
        self.assertEqual((result.slot_tp, result.slot_fp, result.slot_fn), (0, 0, 0))

    def test_duplicate_gold_or_prediction_ids_are_errors(self):
        for gold, predictions, source in (([record(), record()], [record()], "gold"),
                                         ([record()], [record(), record()], "prediction")):
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError, "Duplicate " + source):
                    evaluate_predictions(gold, predictions)

    def test_missing_or_unexpected_prediction_ids_are_errors(self):
        cases = [([record("r1"), record("r2")], [record("r1")]),
                 ([record("r1")], [record("r1"), record("r2")]),
                 ([record("r1")], [record("r2")]),
                 ([record("r1")], [])]
        for gold, predictions in cases:
            with self.subTest(gold=gold, predictions=predictions):
                with self.assertRaisesRegex(ValueError, "Prediction IDs do not match"):
                    evaluate_predictions(gold, predictions)

    def test_empty_split_cannot_produce_a_misleading_score(self):
        with self.assertRaisesRegex(ValueError, "empty gold split"):
            evaluate_predictions([], [])

    def test_untyped_prediction_inputs_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "gold must contain"):
            evaluate_predictions([{"record_id": "r1"}], [record()])
        with self.assertRaisesRegex(TypeError, "prediction must contain"):
            evaluate_predictions([record()], [{"record_id": "r1"}])


if __name__ == "__main__":
    unittest.main()

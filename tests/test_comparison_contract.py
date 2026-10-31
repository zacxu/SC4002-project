"""Hand-counted occurrence scoring and dataset identity corruption checks."""

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src import comparison_contract as contract


def slot(start=0, text="room", slot_type="place"):
    return {"slot_type": slot_type, "text": text, "start": start, "end": start + len(text)}


def record(record_id="r", slots=(), intent="intent", utt=None):
    result = {"record_id": record_id, "slots": list(slots), "intent": intent}
    if utt is not None:
        result["utt"] = utt
    return result


class ExactOccurrenceScoringTests(unittest.TestCase):
    def test_wrong_occurrence_of_identical_value_is_not_credit(self):
        result = contract.score_exact_spans([record(slots=[slot()], utt="room room")],
                                            [record(slots=[slot(5)])])
        self.assertEqual((result["slot_tp"], result["slot_fp"], result["slot_fn"]), (0, 1, 1))

    def test_repeated_gold_values_at_distinct_positions_count_independently(self):
        result = contract.score_exact_spans([record(slots=[slot(), slot(5)], utt="room room")],
                                            [record(slots=[slot()])])
        self.assertEqual((result["slot_tp"], result["slot_fp"], result["slot_fn"]), (1, 0, 1))
        self.assertAlmostEqual(result["slot_micro_f1"], 2 / 3)

    def test_duplicate_identical_gold_or_prediction_occurrence_is_rejected(self):
        for gold, pred in (([record(slots=[slot(), slot()])], [record()]),
                           ([record()], [record(slots=[slot(), slot()])])):
            with self.assertRaisesRegex(ValueError, "Duplicate identical slot"):
                contract.score_exact_spans(gold, pred)

    def test_raw_text_checks_both_gold_and_predictions(self):
        gold = [record(slots=[slot()], utt="room")]
        with self.assertRaisesRegex(ValueError, "original utterance"):
            contract.score_exact_spans(gold, [record(slots=[slot(text="ROOM")])])
        with self.assertRaisesRegex(ValueError, "original utterance"):
            contract.score_exact_spans([record(slots=[slot(1)], utt="room")], [record()])

    def test_wrong_type_boundary_or_literal_text_cannot_match(self):
        for value in (slot(slot_type="other"), slot(text="roo"), slot(1), slot(text="ROOM")):
            result = contract.score_exact_spans([record(slots=[slot()])], [record(slots=[value])])
            self.assertEqual((result["slot_tp"], result["slot_fp"], result["slot_fn"]), (0, 1, 1))

    def test_unicode_and_whitespace_are_not_normalized(self):
        for gold_text, predicted_text in (("é", "e\u0301"), ("room", "room ")):
            result = contract.score_exact_spans([record(slots=[slot(text=gold_text)])],
                                                [record(slots=[slot(text=predicted_text)])])
            self.assertEqual(result["slot_micro_f1"], 0)

    def test_ids_join_by_identity_not_order_and_pool_micro_counts(self):
        gold = [record("a", [slot()]), record("b", [slot()]), record("c")]
        pred = [record("c", [slot()]), record("b", intent="wrong"), record("a", [slot()])]
        result = contract.score_exact_spans(iter(gold), iter(pred))
        self.assertEqual((result["slot_tp"], result["slot_fp"], result["slot_fn"]), (1, 1, 1))
        self.assertEqual(result["slot_micro_f1"], 0.5)
        self.assertAlmostEqual(result["intent_accuracy"], 2 / 3)

    def test_duplicate_missing_extra_or_empty_id_coverage_rejected(self):
        for gold, pred in (([], []), ([record(), record()], [record()]),
                           ([record()], [record(), record()]), ([record()], []),
                           ([record()], [record("other")]), ([record("")], [record("")])):
            with self.assertRaises(ValueError):
                contract.score_exact_spans(gold, pred)

    def test_duck_typed_schema_objects_share_the_same_result(self):
        obj = SimpleNamespace(record_id="r", intent="intent", slots=(SimpleNamespace(**slot()),))
        self.assertEqual(contract.score_exact_spans([obj], [obj]),
                         contract.score_exact_spans([record(slots=[slot()])], [record(slots=[slot()])]))

    def test_invalid_boolean_or_inconsistent_offsets_rejected(self):
        for changed in ({"start": True}, {"end": 3}, {"start": -1}, {"text": ""}):
            value = {**slot(), **changed}
            with self.assertRaises(ValueError):
                contract.score_exact_spans([record()], [record(slots=[value])])

    def test_all_empty_slots_have_zero_f1_and_valid_intent_accuracy(self):
        result = contract.score_exact_spans([record()], [record()])
        self.assertEqual(result["slot_micro_f1"], 0)
        self.assertEqual(result["intent_accuracy"], 1)


class CharacterAdapterTests(unittest.TestCase):
    def test_tokenization_independent_same_character_occurrence(self):
        utt = "queen's song"
        coarse = contract.bio_to_character_spans(utt, [(0, 7), (8, 12)], ["B-artist", "O"])
        fine = contract.bio_to_character_spans(utt, [(0, 5), (5, 6), (6, 7), (8, 12)],
                                             ["B-artist", "I-artist", "I-artist", "O"])
        self.assertEqual(coarse, fine)
        self.assertEqual(coarse[0]["text"], "queen's")

    def test_orphan_and_type_changed_i_start_new_spans(self):
        self.assertEqual(contract.bio_to_character_spans("a b", [(0, 1), (2, 3)], ["I-x", "I-y"]),
                         [{"slot_type": "x", "text": "a", "start": 0, "end": 1},
                          {"slot_type": "y", "text": "b", "start": 2, "end": 3}])

    def test_tag_lengths_invalid_tags_and_overlapping_offsets_fail(self):
        for offsets, tags in (([(0, 1)], []), ([(0, 1)], ["X-a"]),
                              ([(0, 2), (1, 3)], ["O", "O"]), ([(0, 9)], ["O"])):
            with self.assertRaises(ValueError):
                contract.bio_to_character_spans("abc", offsets, tags)

    def test_offset_objects_and_dicts_supported(self):
        expected = contract.bio_to_character_spans("x", [(0, 1)], ["B-type"])
        for offsets in ([{"start": 0, "end": 1}], [SimpleNamespace(start=0, end=1)]):
            self.assertEqual(contract.bio_to_character_spans("x", offsets, ["B-type"]), expected)

    def test_gold_annotation_reconstruction_preserves_exact_unicode_spaces(self):
        self.assertEqual(contract.parse_annotated_spans("é  x", "[name : é]  x"),
                         [{"slot_type": "name", "text": "é", "start": 0, "end": 1}])
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            contract.parse_annotated_spans("é x", "[name : é]  x")


class VerifiedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "en-US.jsonl"
        self.rows = [{"id": str(i), "partition": split, "locale": "en-US", "utt": "room",
                      "annot_utt": "[place : room]", "intent": "intent"}
                     for i, split in enumerate(("train", "dev", "test"))]
        self.raw = ("\n".join(json.dumps(row) for row in self.rows) + "\n").encode()
        self.hash_patch = patch.object(contract, "DATASET_SHA256", hashlib.sha256(self.raw).hexdigest())
        self.count_patch = patch.object(contract, "EXPECTED_COUNTS", {"train": 1, "dev": 1, "test": 1})
        self.hash_patch.start()
        self.count_patch.start()
        self.addCleanup(self.hash_patch.stop)
        self.addCleanup(self.count_patch.stop)

    def test_identical_data_produces_same_original_partition_and_content_hashes(self):
        self.path.write_bytes(self.raw)
        first = contract.load_verified_massive(self.path)
        second = contract.ensure_verified_massive(self.path)
        self.assertEqual(first, second)
        splits, manifest = first
        self.assertEqual([row["id"] for row in splits["train"]], ["0"])
        self.assertEqual(set(manifest["split_content_sha256"]), {"train", "dev", "test"})
        self.assertEqual(manifest["dataset_sha256"], hashlib.sha256(self.raw).hexdigest())

    def test_counts_alone_cannot_pass_tampered_text_or_membership(self):
        for changed in (self.raw.replace(b'"room"', b'"ROOM"', 1),
                        self.raw.replace(b'"train"', b'"test"', 1)):
            self.path.write_bytes(changed)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                contract.ensure_verified_massive(self.path)
            self.assertEqual(self.path.read_bytes(), changed)

    def test_bad_structure_rejected_even_with_matching_fixture_digest(self):
        for rows in ([{**self.rows[0], "locale": "en-GB"}, *self.rows[1:]],
                     [self.rows[0], {**self.rows[1], "id": "0"}, self.rows[2]],
                     [{**self.rows[0], "annot_utt": "wrong"}, *self.rows[1:]],
                     [{**self.rows[0], "id": 0}, *self.rows[1:]]):
            raw = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
            self.path.write_bytes(raw)
            with patch.object(contract, "DATASET_SHA256", hashlib.sha256(raw).hexdigest()):
                with self.assertRaises(ValueError):
                    contract.load_verified_massive(self.path)

    def archive(self, data, member_name=contract.ENGLISH_MEMBER):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            info = tarfile.TarInfo(member_name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        return buffer.getvalue()

    def test_official_member_download_is_verified_before_exclusive_publication(self):
        response = io.BytesIO(self.archive(self.raw))
        with patch.object(contract, "urlopen", return_value=response) as opened:
            splits, manifest = contract.ensure_verified_massive(self.path)
        self.assertEqual(self.path.read_bytes(), self.raw)
        self.assertEqual(manifest["counts"], {"train": 1, "dev": 1, "test": 1})
        self.assertEqual(opened.call_args.args[0].full_url, contract.SOURCE_URL)
        self.assertFalse(list(self.path.parent.glob("*.part")))

    def test_downloaded_wrong_data_or_missing_member_never_published(self):
        for payload in (self.archive(b"bad"), self.archive(self.raw, "../../en-US.jsonl")):
            with patch.object(contract, "urlopen", return_value=io.BytesIO(payload)):
                with self.assertRaises(ValueError):
                    contract.ensure_verified_massive(self.path)
            self.assertFalse(self.path.exists())
            self.assertFalse(list(self.path.parent.glob("*.part")))


if __name__ == "__main__":
    unittest.main()

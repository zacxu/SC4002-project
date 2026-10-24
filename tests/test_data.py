"""Check preprocessing against unchanged original training records.

Selected fixtures are real MASSIVE 1.0 en-US train IDs. Deliberate corruption
fixtures test validation only; no fixture is added to model training data.
"""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from src.data import (
    EXPECTED_COUNTS, FittedMappings, SLOT_IGNORE_INDEX, Token,
    collate_examples, decode_bio, fit_mappings, gold_bio_tags, load_massive,
    parse_annotations, prepare_example, prepare_split, read_predictions,
    tokenize_with_offsets, write_predictions,
)
from src.runtime import project_root


class OriginalDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = project_root() / "data/1.0/data/en-US.jsonl"
        cls.splits = load_massive(cls.path)
        cls.train = cls.splits["train"]
        cls.by_id = {example.record_id: example for example in cls.train}
        cls.mappings = fit_mappings(cls.train)
        cls.selected = tuple(cls.by_id[key] for key in ("4", "49", "57", "12270", "10235"))

    @staticmethod
    def source_record(example):
        """Preserve the original required values when making temporary JSONL."""
        return {"id": example.record_id, "locale": "en-US", "partition": example.partition,
                "utt": example.utt, "intent": example.intent, "annot_utt": example.annot_utt}

    def load_fixture(self, rows, counts=None, suffix=""):
        counts = {"train": len(rows), "dev": 0, "test": 0} if counts is None else counts
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows) + suffix, encoding="utf-8")
            return load_massive(path, expected_counts=counts)

    def test_original_counts_and_id_partition_membership_are_preserved(self):
        self.assertEqual({key: len(rows) for key, rows in self.splits.items()}, EXPECTED_COUNTS)
        ids = [example.record_id for rows in self.splits.values() for example in rows]
        self.assertEqual(len(ids), len(set(ids)))
        for partition, rows in self.splits.items():
            self.assertTrue(all(example.partition == partition for example in rows))
        # Source order is retained, rather than sorting lexical record IDs.
        rows = [self.source_record(self.by_id[key]) for key in ("49", "4", "57")]
        loaded = self.load_fixture(rows)
        self.assertEqual(tuple(example.record_id for example in loaded["train"]), ("49", "4", "57"))

    def test_loading_rejects_duplicate_ids_instead_of_dropping_records(self):
        original = self.source_record(self.by_id["4"])
        with self.assertRaisesRegex(ValueError, "Duplicate record ID"):
            self.load_fixture([original, original])

    def test_loading_rejects_blank_lines_and_incorrect_split_counts(self):
        rows = [self.source_record(self.by_id["4"])]
        with self.assertRaisesRegex(ValueError, "Blank JSONL record"):
            self.load_fixture(rows, suffix="\n")
        with self.assertRaisesRegex(ValueError, "split count mismatch"):
            self.load_fixture(rows, counts={"train": 2, "dev": 0, "test": 0})

    def test_loading_rejects_incorrect_locale_partition_and_field_types(self):
        for field, replacement in (("locale", "en-GB"), ("partition", "validation"), ("id", 4),
                                   ("utt", ""), ("intent", None)):
            with self.subTest(field=field):
                row = self.source_record(self.by_id["4"])
                row[field] = replacement
                with self.assertRaises(ValueError):
                    self.load_fixture([row])

    def test_exact_character_offsets_include_punctuation_inside_gold_span(self):
        example = self.by_id["49"]
        slots = parse_annotations(example.utt, example.annot_utt)
        self.assertEqual([(slot.slot_type, slot.text, slot.start, slot.end) for slot in slots],
                         [("time_zone", "g. m. t. plus five", 20, 38)])
        self.assertEqual(example.utt[slots[0].start:slots[0].end], slots[0].text)
        tokens = tokenize_with_offsets(example.utt)
        self.assertTrue(any(token.text == "." for token in tokens))
        for token in tokens:
            self.assertEqual(example.utt[token.start:token.end], token.text)

    def test_repeated_surface_values_keep_their_distinct_occurrence_offsets(self):
        example = self.by_id["12270"]
        repeated = [slot for slot in parse_annotations(example.utt, example.annot_utt)
                    if slot.slot_type == "transport_type"]
        self.assertEqual([(slot.text, slot.start, slot.end) for slot in repeated],
                         [("train", 10, 15), ("train", 30, 35)])
        self.assertEqual(decode_bio(example.utt, tokenize_with_offsets(example.utt),
                                    gold_bio_tags(example, tokenize_with_offsets(example.utt))), example.slots)

    def test_adjacent_slot_types_start_separate_bio_spans(self):
        example = self.by_id["57"]
        tokens = tokenize_with_offsets(example.utt)
        tags = gold_bio_tags(example, tokens)
        self.assertEqual(tags[[token.text for token in tokens].index("thai")], "B-food_type")
        self.assertEqual(tags[[token.text for token in tokens].index("takeaways")], "B-order_type")
        self.assertEqual(decode_bio(example.utt, tokens, tags), example.slots)

    def test_annotation_alignment_does_not_normalize_case_or_spacing(self):
        example = self.by_id["49"]
        # Annotation-only corruptions of an original train record must fail.
        for corrupted in (example.annot_utt.upper(), example.annot_utt + " ",
                          example.annot_utt.replace("g. m.", "g.  m.")):
            with self.subTest(annotation=corrupted):
                with self.assertRaisesRegex(ValueError, "reproduce utt exactly"):
                    parse_annotations(example.utt, corrupted)
        self.assertEqual(parse_annotations(example.utt, example.annot_utt), example.slots)

    def test_tokenization_uses_only_raw_utterance_and_preserves_case(self):
        example = self.by_id["49"]
        first = tokenize_with_offsets(example.utt)
        # A gold-label change has no effect on input tokenization or offsets.
        modified = deepcopy(self.source_record(example))
        modified["annot_utt"] = modified["annot_utt"].replace("time_zone", "date")
        self.assertNotEqual(parse_annotations(example.utt, modified["annot_utt"]), example.slots)
        self.assertEqual(tokenize_with_offsets(modified["utt"]), first)
        # Derived uppercase text is only a tokenizer validation input.
        upper = tokenize_with_offsets(example.utt.upper())
        self.assertEqual(tuple(token.text for token in upper), tuple(token.text.upper() for token in first))
        self.assertNotEqual(upper, first)

    def test_selected_original_annotations_round_trip_without_text_loss(self):
        for example in self.selected:
            with self.subTest(record_id=example.record_id):
                tokens = tokenize_with_offsets(example.utt)
                self.assertEqual(decode_bio(example.utt, tokens, gold_bio_tags(example, tokens)), example.slots)
        no_slots = self.by_id["4"]
        self.assertEqual(parse_annotations(no_slots.utt, no_slots.annot_utt), ())
        self.assertEqual(set(gold_bio_tags(no_slots, tokenize_with_offsets(no_slots.utt))), {"O"})

    def test_all_label_mappings_are_fitted_from_original_train(self):
        self.assertEqual(self.mappings.training_count, EXPECTED_COUNTS["train"])
        self.assertEqual(len(self.mappings.intent_to_id), 60)
        self.assertEqual(len(self.mappings.slot_types), 55)
        self.assertEqual(len(self.mappings.tag_to_id), 111)
        self.assertEqual(self.mappings.token_to_id["<PAD>"], 0)
        self.assertEqual(self.mappings.token_to_id["<UNK>"], 1)
        self.assertEqual(self.mappings.tag_to_id["O"], 0)
        self.assertEqual(len(self.mappings.training_ids_sha256), 64)
        self.assertEqual(len(self.mappings.training_records_sha256), 64)

    def test_small_train_fit_maps_unseen_actual_training_words_to_unknown(self):
        training = [self.by_id["4"]]
        mappings = fit_mappings(training)
        unseen = self.by_id["5"]  # Same original intent; raw utterance has an unseen token.
        self.assertNotIn("stop", mappings.token_to_id)
        prepared = prepare_example(unseen, mappings)
        self.assertEqual(prepared.token_ids, (1,))
        self.assertEqual(mappings.training_count, 1)
        self.assertEqual(set(mappings.intent_to_id), {training[0].intent})
        self.assertEqual(mappings.slot_types, ())

    def test_fitting_refuses_nontrain_empty_and_duplicate_sequences(self):
        for rows in ([], [self.splits["dev"][0]], [self.by_id["4"], self.splits["test"][0]],
                     [self.by_id["4"], self.by_id["4"]]):
            with self.subTest(ids=[row.record_id for row in rows]):
                with self.assertRaises(ValueError):
                    fit_mappings(rows)

    def test_small_fit_rejects_labels_not_observed_in_training(self):
        with self.assertRaisesRegex(ValueError, "Label not observed in training"):
            prepare_example(self.by_id["49"], fit_mappings([self.by_id["4"]]))

    def test_mappings_json_round_trip_preserves_ids_and_preprocessing_contract(self):
        payload = json.loads(json.dumps(self.mappings.to_dict()))
        self.assertEqual(FittedMappings.from_dict(payload), self.mappings)
        prepared = prepare_example(self.by_id["49"], FittedMappings.from_dict(payload))
        self.assertEqual(prepared, prepare_example(self.by_id["49"], self.mappings))
        for key, replacement in (("fit_partition", "dev"), ("normalization", "lowercase"),
                                 ("truncation", 20), ("slot_ignore_index", 0)):
            corrupted = deepcopy(payload)
            corrupted[key] = replacement
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    FittedMappings.from_dict(corrupted)
        corrupted = deepcopy(payload)
        corrupted["token_to_id"]["<PAD>"] = len(corrupted["token_to_id"]) + 1
        with self.assertRaises(ValueError):
            FittedMappings.from_dict(corrupted)

    def test_prepare_split_retains_every_input_record_and_offset_sidecar(self):
        prepared = prepare_split(self.selected, self.mappings)
        self.assertEqual(tuple(item.example.record_id for item in prepared),
                         tuple(item.record_id for item in self.selected))
        for item in prepared:
            self.assertEqual(len(item.tokens), len(item.token_ids))
            self.assertEqual(len(item.tokens), len(item.tag_ids))
            tags = tuple({value: key for key, value in self.mappings.tag_to_id.items()}[tag]
                         for tag in item.tag_ids)
            self.assertEqual(decode_bio(item.example.utt, item.tokens, tags), item.example.slots)

    def test_longest_original_train_row_is_not_truncated_and_padding_is_masked(self):
        import torch

        longest = max(self.train, key=lambda example: len(tokenize_with_offsets(example.utt)))
        short, long = prepare_split((self.by_id["4"], longest), self.mappings)
        self.assertEqual(long.example.record_id, "10235")
        self.assertEqual(len(long.tokens), 35)
        batch = collate_examples((short, long))
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 35))
        self.assertEqual(batch["lengths"].tolist(), [len(short.tokens), 35])
        self.assertEqual(batch["attention_mask"].dtype, torch.bool)
        self.assertEqual(batch["attention_mask"].sum(dim=1).tolist(), batch["lengths"].tolist())
        self.assertEqual(batch["input_ids"][1].tolist(), list(long.token_ids))
        self.assertTrue(torch.equal(batch["slot_labels"][1], torch.tensor(long.tag_ids)))
        padding = ~batch["attention_mask"]
        self.assertTrue(torch.all(batch["input_ids"][padding] == 0))
        self.assertTrue(torch.all(batch["slot_labels"][padding] == SLOT_IGNORE_INDEX))
        self.assertTrue(torch.all(batch["slot_labels"][batch["attention_mask"]] >= 0))
        self.assertEqual(batch["record_ids"], [short.example.record_id, long.example.record_id])
        self.assertEqual(batch["tokens"], (short.tokens, long.tokens))
        self.assertEqual(batch["intent_labels"].tolist(), [short.intent_id, long.intent_id])
        with self.assertRaisesRegex(ValueError, "empty batch"):
            collate_examples(())

    def test_orphan_and_type_changing_i_tags_start_new_exact_slots(self):
        example = self.by_id["49"]
        tokens = tokenize_with_offsets(example.utt)
        tags = ["O"] * len(tokens)
        tags[:3] = ["I-time_zone", "I-time_zone", "I-transport_type"]
        slots = decode_bio(example.utt, tokens, tags)
        self.assertEqual([(slot.slot_type, slot.start, slot.end) for slot in slots],
                         [("time_zone", tokens[0].start, tokens[1].end),
                          ("transport_type", tokens[2].start, tokens[2].end)])
        self.assertEqual(slots[0].text, example.utt[tokens[0].start:tokens[1].end])

    def test_bio_decoding_rejects_bad_lengths_tags_and_token_offsets(self):
        example = self.by_id["49"]
        tokens = tokenize_with_offsets(example.utt)
        with self.assertRaisesRegex(ValueError, "tag length"):
            decode_bio(example.utt, tokens, ["O"])
        for invalid in ("X-time_zone", "I-", "B", ""):
            tags = ["O"] * len(tokens)
            tags[0] = invalid
            with self.subTest(tag=invalid):
                with self.assertRaisesRegex(ValueError, "Invalid BIO tag"):
                    decode_bio(example.utt, tokens, tags)
        bad_tokens = (Token(tokens[0].text, 1, tokens[0].end + 1),) + tokens[1:]
        with self.assertRaisesRegex(ValueError, "Token offsets"):
            decode_bio(example.utt, bad_tokens, ["O"] * len(tokens))
        with self.assertRaisesRegex(ValueError, "Token offsets"):
            decode_bio(example.utt, tuple(reversed(tokens)), ["O"] * len(tokens))

    def test_prediction_jsonl_round_trip_preserves_ids_text_and_offsets(self):
        gold = tuple(example.gold_prediction() for example in self.selected)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "predictions.jsonl"
            write_predictions(path, gold)
            self.assertEqual(read_predictions(path), gold)
            with self.assertRaisesRegex(ValueError, "Duplicate prediction record ID"):
                write_predictions(path, (gold[0], gold[0]))
            # Failed writes validate all IDs before replacing an existing artifact.
            self.assertEqual(read_predictions(path), gold)
            duplicate = json.dumps(gold[0].to_dict()) + "\n"
            path.write_text(duplicate * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate prediction record ID"):
                read_predictions(path)


if __name__ == "__main__":
    unittest.main()

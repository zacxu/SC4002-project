"""Stage 2 mathematics and integration; task examples come from original train.

Tiny CRF emissions are numerical fixtures, never extra task-data utterances.
No test-split predictions, labels, metrics, or parameter updates are used.
"""

from dataclasses import FrozenInstanceError, replace
import inspect
import io
import itertools
import json
import unittest

import torch
from torch.nn import functional as F

from src.data import FittedMappings, UtteranceExample, parse_annotations, prepare_split
from src.models import BaselineConfig, JointBiLSTM, enforce_parameter_cap, parameter_counts
from src.runtime import project_root
from src.stage2_model import BIOCRF, JointStage2, Stage2Config, collate_stage2, fit_char_vocab
from src.training import compute_losses


class CRFTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(82)
        self.names = ["O", "B-x", "I-x", "B-y", "I-y"]
        self.crf = BIOCRF(self.names).double()
        with torch.no_grad():
            self.crf.transitions.copy_(torch.randn(5, 5, dtype=torch.double) / 3)
            self.crf.start_transitions.copy_(torch.randn(5, dtype=torch.double) / 3)
            self.crf.end_transitions.copy_(torch.randn(5, dtype=torch.double) / 3)

    def legal(self, path):
        if self.names[path[0]].startswith("I-"):
            return False
        for previous, current in zip(path, path[1:]):
            previous, current = self.names[previous], self.names[current]
            if current.startswith("I-") and previous not in {"B-" + current[2:], current}:
                return False
        return True

    def manual_score(self, emissions, path):
        result = self.crf.start_transitions[path[0]] + self.crf.end_transitions[path[-1]]
        for position, tag in enumerate(path):
            result = result + emissions[position, tag]
            if position:
                result = result + self.crf.transitions[path[position - 1], tag]
        return result

    def test_partition_gold_nll_and_viterbi_match_exhaustive_paths(self):
        emissions = torch.randn(4, 4, 5, dtype=torch.double)
        lengths = torch.tensor([4, 1, 3, 2])
        mask = torch.arange(4).unsqueeze(0) < lengths.unsqueeze(1)
        actual_partitions = self.crf.log_partition(emissions, mask)
        actual_paths = self.crf.decode(emissions, mask)
        gold = torch.full((4, 4), -100, dtype=torch.long)
        expected_scores = []
        for row, length in enumerate(lengths.tolist()):
            paths = [path for path in itertools.product(range(5), repeat=length) if self.legal(path)]
            scores = torch.stack([self.manual_score(emissions[row], path) for path in paths])
            expected_partition = torch.logsumexp(scores, 0)
            torch.testing.assert_close(actual_partitions[row], expected_partition, rtol=1e-12, atol=1e-12)
            self.assertEqual(actual_paths[row, :length].tolist(), list(paths[scores.argmax().item()]))
            self.assertTrue(bool((actual_paths[row, length:] == 0).all()))
            selected = len(paths) // 2
            gold[row, :length] = torch.tensor(paths[selected])
            expected_scores.append(scores[selected])
        expected_scores = torch.stack(expected_scores)
        torch.testing.assert_close(self.crf.gold_score(emissions, gold, mask), expected_scores)
        torch.testing.assert_close(self.crf.neg_log_likelihood(emissions, gold, mask),
                                   actual_partitions - expected_scores)

    def test_exact_legality_even_when_illegal_emissions_are_extreme(self):
        emissions = torch.zeros(2, 3, 5, dtype=torch.double)
        emissions[:, :, 2] = 1e8
        emissions[1, 1, 4] = 2e8
        mask = torch.tensor([[True, True, True], [True, True, False]])
        paths = self.crf.decode(emissions, mask)
        for row, length in enumerate(mask.sum(1).tolist()):
            self.assertTrue(self.legal(paths[row, :length].tolist()))
        self.assertTrue(bool(torch.isfinite(self.crf.log_partition(emissions, mask)).all()))

    def test_backward_is_finite_padding_and_illegal_parameters_have_zero_gradient(self):
        emissions = torch.randn(3, 4, 5, dtype=torch.double, requires_grad=True)
        tags = torch.tensor([[1, 2, 0, 3], [0, -100, -100, -100], [3, 4, -100, -100]])
        mask = tags >= 0
        self.crf.neg_log_likelihood(emissions, tags, mask).sum().backward()
        self.assertTrue(bool(torch.isfinite(emissions.grad).all()))
        self.assertEqual(torch.count_nonzero(emissions.grad[~mask]).item(), 0)
        for parameter in self.crf.parameters():
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
        self.assertEqual(torch.count_nonzero(self.crf.transitions.grad[~self.crf.transition_allowed]).item(), 0)
        self.assertEqual(torch.count_nonzero(self.crf.start_transitions.grad[~self.crf.start_allowed]).item(), 0)

    def test_padded_suffix_cannot_change_partition_loss_or_path(self):
        emissions = torch.randn(2, 3, 5, dtype=torch.double)
        tags = torch.tensor([[1, 2, 0], [3, -100, -100]])
        mask = tags >= 0
        changed = emissions.clone()
        changed[~mask] = torch.randn_like(changed[~mask]) * 1e7
        torch.testing.assert_close(self.crf.log_partition(emissions, mask), self.crf.log_partition(changed, mask), rtol=0, atol=0)
        torch.testing.assert_close(self.crf.neg_log_likelihood(emissions, tags, mask),
                                   self.crf.neg_log_likelihood(changed, tags, mask), rtol=0, atol=0)
        self.assertTrue(torch.equal(self.crf.decode(emissions, mask), self.crf.decode(changed, mask)))

    def test_rejects_illegal_gold_masks_shapes_nonfinite_and_broken_inventory(self):
        emissions = torch.zeros(1, 2, 5, dtype=torch.double)
        mask = torch.ones(1, 2, dtype=torch.bool)
        for tags in (torch.tensor([[2, 2]]), torch.tensor([[1, 4]]), torch.tensor([[0, -100]]), torch.tensor([[0.0, 0.0]])):
            with self.subTest(tags=tags), self.assertRaises(ValueError):
                self.crf.neg_log_likelihood(emissions, tags, mask)
        for invalid in (torch.tensor([[False, True]]), torch.tensor([[False, False]]), mask.long(), mask[:, :1]):
            with self.subTest(mask=invalid), self.assertRaises(ValueError):
                self.crf.decode(emissions, invalid)
        for names in ([], ["O", "B-x"], ["O", "I-x"], ["B-x", "O", "I-x"], ["O", "O"], ["O", "Z-x"]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                BIOCRF(names)
        emissions[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.crf.decode(emissions, mask)

    def test_tied_scores_have_repeatable_legal_decoding(self):
        crf = BIOCRF(self.names)
        emissions = torch.zeros(2, 4, 5)
        mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
        self.assertTrue(torch.equal(crf.decode(emissions, mask), torch.zeros(2, 4, dtype=torch.long)))


class Stage2ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        root = project_root()
        cls.mappings = FittedMappings.from_dict(json.loads((root / "artifacts/phase2/preprocessing.json").read_text(encoding="utf-8")))
        original_train = []
        # Ignore nontraining rows without parsing their targets or annotations.
        with (root / "data/1.0/data/en-US.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["partition"] != "train":
                    continue
                original_train.append(UtteranceExample(row["id"], "train", row["utt"], row["intent"], row["annot_utt"],
                                                       parse_annotations(row["utt"], row["annot_utt"])))
                if len(original_train) == 8:
                    break
        cls.prepared = prepare_split(original_train, cls.mappings)
        cls.characters = fit_char_vocab(cls.prepared)
        cls.names = [tag for tag, index in sorted(cls.mappings.tag_to_id.items(), key=lambda item: item[1])]

    def setUp(self):
        torch.manual_seed(42)
        self.batch = collate_stage2(self.prepared, self.characters)

    def model(self, **overrides):
        config = Stage2Config(embedding_dim=8, hidden_dim=6, char_embedding_dim=4, char_channels=5,
                              intent_context_dim=5, **overrides)
        return JointStage2(len(self.mappings.token_to_id), len(self.mappings.intent_to_id), self.names,
                           len(self.characters), config)

    @staticmethod
    def forward(model, batch):
        return model(batch["input_ids"], batch["lengths"], batch["attention_mask"], batch["char_ids"])

    def test_configuration_strict_roundtrip_and_freezing(self):
        config = Stage2Config()
        self.assertEqual(Stage2Config.from_dict(config.to_dict()), config)
        with self.assertRaises(FrozenInstanceError):
            config.hidden_dim = 1
        for values in ({"embedding_dim": 0}, {"hidden_dim": True}, {"char_kernel_size": 2},
                       {"use_char": 1}, {"use_crf": None}, {"dropout": float("nan")}, {"dropout": True},
                       {"dropout": 1.0}, {"intent_context_dim": -1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Stage2Config(**values)
        with self.assertRaises(ValueError):
            Stage2Config.from_dict({"embedding_dim": 8})
        with self.assertRaises(ValueError):
            Stage2Config.from_dict({**config.to_dict(), "unknown": 2})

    def test_character_vocabulary_is_sorted_training_only_and_rejects_duplicates(self):
        expected = sorted({char for item in self.prepared for token in item.tokens for char in token.text})
        self.assertEqual(list(self.characters), ["<PAD>", "<UNK>"] + expected)
        self.assertEqual(list(self.characters.values()), list(range(len(self.characters))))
        nontrain = replace(self.prepared[0], example=replace(self.prepared[0].example, partition="dev"))
        for records in ([], [nontrain], [self.prepared[0], self.prepared[0]]):
            with self.subTest(records=len(records)), self.assertRaises(ValueError):
                fit_char_vocab(records)

    def test_collation_retains_exact_full_characters_and_unknown_mapping(self):
        for row, item in enumerate(self.prepared):
            for column, token in enumerate(item.tokens):
                expected = [self.characters[char] for char in token.text]
                self.assertEqual(self.batch["char_ids"][row, column, :len(expected)].tolist(), expected)
                self.assertTrue(bool((self.batch["char_ids"][row, column, len(expected):] == 0).all()))
        unknown = collate_stage2(self.prepared, {"<PAD>": 0, "<UNK>": 1})
        self.assertTrue(bool((unknown["char_ids"][unknown["char_ids"] > 0] == 1).all()))
        self.assertTrue(bool((self.batch["char_ids"][~self.batch["attention_mask"]] == 0).all()))
        for invalid in ({"<PAD>": 0, "<UNK>": 0}, {"<PAD>": 0, "<UNK>": 2}, {"<PAD>": 0, "<UNK>": 1, "ab": 2}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                collate_stage2(self.prepared, invalid)

    def test_all_toggles_disabled_are_identical_to_frozen_architecture(self):
        torch.manual_seed(42)
        baseline = JointBiLSTM(len(self.mappings.token_to_id), len(self.mappings.intent_to_id), len(self.names),
                               BaselineConfig(embedding_dim=8, hidden_dim=6, dropout=0.3)).eval()
        torch.manual_seed(42)
        stage2 = self.model().eval()
        self.assertEqual(set(baseline.state_dict()), set(stage2.state_dict()))
        for key in baseline.state_dict():
            torch.testing.assert_close(baseline.state_dict()[key], stage2.state_dict()[key], rtol=0, atol=0)
        with torch.inference_mode():
            original = baseline(self.batch["input_ids"], self.batch["lengths"], self.batch["attention_mask"])
            candidate = self.forward(stage2, self.batch)
        for key in original:
            torch.testing.assert_close(original[key], candidate[key], rtol=0, atol=0)
        torch.testing.assert_close(compute_losses(original, self.batch).total, stage2.loss(candidate, self.batch).total, rtol=0, atol=0)
        self.assertEqual(parameter_counts(baseline), parameter_counts(stage2))

    def test_all_eight_ablations_forward_loss_backward_and_decoding(self):
        for char, crf, conditioning in itertools.product((False, True), repeat=3):
            with self.subTest(char=char, crf=crf, conditioning=conditioning):
                model = self.model(use_char=char, use_crf=crf, intent_conditioning=conditioning)
                outputs = self.forward(model, self.batch)
                self.assertEqual(outputs["intent_logits"].shape, (8, 60))
                self.assertEqual(outputs["slot_logits"].shape, (*self.batch["input_ids"].shape, 111))
                losses = model.loss(outputs, self.batch)
                losses.total.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                for parameter in (model.embedding.weight, model.encoder.weight_ih_l0,
                                  model.encoder.weight_ih_l0_reverse, model.intent_head.weight, model.slot_head.weight):
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
                if char:
                    self.assertGreater(model.char_encoder.embedding.weight.grad.abs().sum().item(), 0)
                decoded = model.decode(outputs, self.batch["attention_mask"])
                self.assertEqual(decoded.shape, self.batch["input_ids"].shape)
                self.assertTrue(bool((decoded[~self.batch["attention_mask"]] == 0).all()))
                if crf:
                    self.assertTrue(bool(model.crf.start_allowed[decoded[:, 0]].all()))
                    transitions = model.crf.transition_allowed[decoded[:, :-1], decoded[:, 1:]]
                    self.assertTrue(bool(transitions[self.batch["attention_mask"][:, 1:]].all()))

    def test_joint_loss_has_explicit_example_and_token_denominators(self):
        model = self.model(use_crf=True, use_char=True, intent_conditioning=True)
        outputs = self.forward(model, self.batch)
        losses = model.loss(outputs, self.batch)
        expected_intent = F.cross_entropy(outputs["intent_logits"], self.batch["intent_labels"], reduction="sum")
        expected_slot = model.crf.neg_log_likelihood(outputs["slot_logits"], self.batch["slot_labels"], self.batch["attention_mask"]).sum()
        torch.testing.assert_close(losses.intent, expected_intent / 8)
        torch.testing.assert_close(losses.slot, expected_slot / int(self.batch["attention_mask"].sum()))
        torch.testing.assert_close(losses.total, losses.intent + losses.slot)

    def test_conditioning_uses_predicted_intent_with_slot_gradient_to_intent_head(self):
        self.assertEqual(list(inspect.signature(JointStage2.forward).parameters),
                         ["self", "input_ids", "lengths", "attention_mask", "char_ids"])
        for conditioned in (False, True):
            model = self.model(intent_conditioning=conditioned).eval()
            outputs = self.forward(model, self.batch)
            F.cross_entropy(outputs["slot_logits"].flatten(0, 1), self.batch["slot_labels"].flatten(), ignore_index=-100).backward()
            if conditioned:
                self.assertGreater(model.intent_head.weight.grad.abs().sum().item(), 0)
                self.assertGreater(model.intent_embeddings.weight.grad.abs().sum().item(), 0)
            else:
                self.assertIsNone(model.intent_head.weight.grad)
            self.assertGreater(model.encoder.weight_ih_l0.grad.abs().sum().item(), 0)

    def test_char_and_token_padding_invariance_alone_and_batched(self):
        model = self.model(use_char=True, use_crf=True, intent_conditioning=True).eval()
        extended = dict(self.batch)
        extended["input_ids"] = F.pad(self.batch["input_ids"], (0, 3))
        extended["attention_mask"] = F.pad(self.batch["attention_mask"], (0, 3))
        extended["char_ids"] = F.pad(self.batch["char_ids"], (0, 9, 0, 3))
        row = int(self.batch["lengths"].argmin())
        alone = collate_stage2([self.prepared[row]], self.characters)
        with torch.inference_mode():
            original = self.forward(model, self.batch)
            padded = self.forward(model, extended)
            isolated = self.forward(model, alone)
            char_features = model.char_encoder(extended["char_ids"], extended["attention_mask"])
        torch.testing.assert_close(original["intent_logits"], padded["intent_logits"], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(original["slot_logits"][self.batch["attention_mask"]],
                                   padded["slot_logits"][extended["attention_mask"]], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(original["intent_logits"][row], isolated["intent_logits"][0], rtol=1e-5, atol=1e-7)
        length = int(self.batch["lengths"][row])
        torch.testing.assert_close(original["slot_logits"][row, :length], isolated["slot_logits"][0], rtol=1e-5, atol=1e-7)
        self.assertTrue(bool((char_features[~extended["attention_mask"]] == 0).all()))
        self.assertTrue(torch.equal(model.decode(original, self.batch["attention_mask"]),
                                    model.decode(padded, extended["attention_mask"])[:, :self.batch["input_ids"].shape[1]]))

    def test_frozen_character_and_word_embeddings_still_count_against_cap(self):
        model = JointStage2(5146, 60, self.names, len(self.characters), Stage2Config(use_char=True, use_crf=True, intent_conditioning=True))
        counts = enforce_parameter_cap(model)
        self.assertGreater(counts["total"], 4646571)
        self.assertLess(counts["total"], 20_000_000)
        frozen = model.embedding.weight.numel() + model.char_encoder.embedding.weight.numel()
        model.embedding.weight.requires_grad_(False)
        model.char_encoder.embedding.weight.requires_grad_(False)
        after = enforce_parameter_cap(model)
        self.assertEqual(after["total"], counts["total"])
        self.assertEqual(after["trainable"], counts["total"] - frozen)
        with self.assertRaises(ValueError):
            enforce_parameter_cap(model, counts["total"] - 1)

    def test_state_dict_config_roundtrip_retains_outputs_loss_and_paths(self):
        model = self.model(use_char=True, use_crf=True, intent_conditioning=True).eval()
        outputs = self.forward(model, self.batch)
        buffer = io.BytesIO()
        torch.save({"config": model.config.to_dict(), "state": model.state_dict()}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        restored = JointStage2(len(self.mappings.token_to_id), 60, self.names, len(self.characters),
                               Stage2Config.from_dict(saved["config"])).eval()
        restored.load_state_dict(saved["state"], strict=True)
        fresh = self.forward(restored, self.batch)
        for key in outputs:
            torch.testing.assert_close(outputs[key], fresh[key], rtol=0, atol=0)
        torch.testing.assert_close(model.loss(outputs, self.batch).total, restored.loss(fresh, self.batch).total, rtol=0, atol=0)
        self.assertTrue(torch.equal(model.decode(outputs, self.batch["attention_mask"]), restored.decode(fresh, self.batch["attention_mask"])))

    def test_padding_embedding_rows_stay_zero_after_joint_update(self):
        model = self.model(use_char=True, use_crf=True, intent_conditioning=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        model.loss(self.forward(model, self.batch), self.batch).total.backward()
        for embedding in (model.embedding, model.char_encoder.embedding):
            self.assertEqual(torch.count_nonzero(embedding.weight.grad[0]).item(), 0)
        optimizer.step()
        for embedding in (model.embedding, model.char_encoder.embedding):
            self.assertEqual(torch.count_nonzero(embedding.weight[0]).item(), 0)

    def test_character_input_requires_exact_prefixes_and_real_token_alignment(self):
        model = self.model(use_char=True)
        with self.assertRaisesRegex(ValueError, "requires"):
            model(self.batch["input_ids"], self.batch["lengths"], self.batch["attention_mask"])
        invalids = [self.batch["char_ids"].float(), self.batch["char_ids"][:, :, :0],
                    torch.zeros_like(self.batch["char_ids"])]
        gap = self.batch["char_ids"].clone()
        gap[0, 0, 0] = 0
        invalids.append(gap)
        out_of_range = self.batch["char_ids"].clone()
        out_of_range[0, 0, 0] = len(self.characters)
        invalids.append(out_of_range)
        for characters in invalids:
            with self.subTest(shape=characters.shape), self.assertRaises(ValueError):
                model(self.batch["input_ids"], self.batch["lengths"], self.batch["attention_mask"], characters)


if __name__ == "__main__":
    unittest.main()

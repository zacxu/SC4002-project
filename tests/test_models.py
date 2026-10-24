"""CPU checks for the baseline's mathematical and input contracts.

ID tensors exercise shapes and masks only; no invented utterances are added to
the task data. The integration test uses unchanged original train examples.
"""

from dataclasses import FrozenInstanceError
import unittest

import torch
from torch import nn

from src.data import collate_examples, fit_mappings, load_massive, prepare_split
from src.models import BaselineConfig, JointBiLSTM, enforce_parameter_cap, parameter_counts
from src.runtime import project_root


class BaselineModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = BaselineConfig(embedding_dim=8, hidden_dim=6, dropout=0.2)
        self.model = JointBiLSTM(25, 4, 7, self.config)
        # The first row is longer so sorting in pack_padded_sequence is tested
        # by the reversed batch in test_unsorted_lengths below.
        self.ids = torch.tensor([[2, 3, 4, 5], [6, 7, 0, 0]])
        self.lengths = torch.tensor([4, 2])
        self.mask = torch.tensor([[True, True, True, True], [True, True, False, False]])

    def test_shapes_both_head_gradients_and_shared_encoder(self):
        output = self.model(self.ids, self.lengths, self.mask)
        self.assertEqual(set(output), {"intent_logits", "slot_logits"})
        self.assertEqual(output["intent_logits"].shape, (2, 4))
        self.assertEqual(output["slot_logits"].shape, (2, 4, 7))
        intent_loss = output["intent_logits"].square().mean()
        slot_loss = output["slot_logits"][self.mask].square().mean()
        for loss in (intent_loss, slot_loss):
            self.model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            self.assertGreater(self.model.embedding.weight.grad.abs().sum().item(), 0)
            self.assertGreater(self.model.encoder.weight_ih_l0.grad.abs().sum().item(), 0)
            self.assertGreater(self.model.encoder.weight_ih_l0_reverse.grad.abs().sum().item(), 0)
        self.model.zero_grad(set_to_none=True)
        (intent_loss + slot_loss).backward()
        self.assertGreater(self.model.intent_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.slot_head.weight.grad.abs().sum().item(), 0)
        self.assertEqual(output["intent_logits"].dtype, torch.float32)

    def test_padding_invariance_alone_batched_and_extra_padding(self):
        self.model.eval()
        with torch.inference_mode():
            batched = self.model(self.ids, self.lengths, self.mask)
            alone = self.model(self.ids[1:2, :2], self.lengths[1:2])
            extended_ids = torch.nn.functional.pad(self.ids, (0, 5))
            extra = self.model(extended_ids, self.lengths)
        torch.testing.assert_close(batched["intent_logits"][1], alone["intent_logits"][0], rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(batched["slot_logits"][1, :2], alone["slot_logits"][0], rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(batched["intent_logits"], extra["intent_logits"], rtol=0, atol=0)
        torch.testing.assert_close(batched["slot_logits"][self.mask], extra["slot_logits"][:, :4][self.mask], rtol=0, atol=0)

    def test_unsorted_lengths_restore_original_batch_order(self):
        self.model.eval()
        with torch.inference_mode():
            normal = self.model(self.ids, self.lengths)
            reverse = self.model(self.ids.flip(0), self.lengths.flip(0))
        for key in normal:
            torch.testing.assert_close(normal[key], reverse[key].flip(0), rtol=1e-5, atol=1e-7)

    def test_pad_embedding_remains_zero_after_optimizer_update(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        output = self.model(self.ids, self.lengths)
        (output["intent_logits"].square().mean() + output["slot_logits"][self.mask].square().mean()).backward()
        self.assertEqual(torch.count_nonzero(self.model.embedding.weight.grad[0]).item(), 0)
        optimizer.step()
        self.assertEqual(torch.count_nonzero(self.model.embedding.weight[0]).item(), 0)

    def test_dropout_is_active_only_during_training(self):
        self.model.train()
        first = self.model(self.ids, self.lengths)
        second = self.model(self.ids, self.lengths)
        self.assertFalse(torch.equal(first["intent_logits"], second["intent_logits"]))
        self.model.eval()
        with torch.inference_mode():
            first = self.model(self.ids, self.lengths)
            second = self.model(self.ids, self.lengths)
        self.assertTrue(torch.equal(first["intent_logits"], second["intent_logits"]))

    def test_invalid_lengths_are_rejected(self):
        for lengths in (torch.tensor([0, 2]), torch.tensor([-1, 2]), torch.tensor([5, 2]),
                        torch.tensor([[4, 2]]), torch.tensor([4]), torch.tensor([4.0, 2.0]),
                        torch.tensor([True, True])):
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                self.model(self.ids, lengths)

    def test_invalid_indices_shapes_and_empty_batches_are_rejected(self):
        for ids in (self.ids.float(), self.ids.bool(), torch.tensor([[2, -1, 4, 5], [6, 7, 0, 0]]),
                    torch.tensor([[2, 25, 4, 5], [6, 7, 0, 0]]), self.ids.flatten(),
                    torch.empty((0, 4), dtype=torch.long), torch.empty((2, 0), dtype=torch.long),
                    torch.tensor([[2, 3, 4, 5], [6, 0, 0, 0]]),
                    torch.tensor([[2, 3, 4, 5], [6, 7, 8, 0]])):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.model(ids, self.lengths)

    def test_masks_must_be_exact_right_padded_prefixes(self):
        invalid_masks = (self.mask[:, :3], self.mask.float(),
                         torch.tensor([[1, 1, 1, 1], [1, 0, 1, 0]]),
                         torch.tensor([[1, 1, 1, 1], [1, 2, 0, 0]]),
                         torch.ones_like(self.ids))
        for mask in invalid_masks:
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                self.model(self.ids, self.lengths, mask)
        output = self.model(self.ids.to(torch.int32), self.lengths.to(torch.int32), self.mask.to(torch.int64))
        self.assertEqual(output["intent_logits"].shape, (2, 4))

    def test_default_parameter_count_and_cap_include_frozen_embeddings(self):
        model = JointBiLSTM(5146, 60, 111)
        counts = enforce_parameter_cap(model)
        self.assertEqual(counts, {"total": 4646571, "trainable": 4646571})
        model.embedding.weight.requires_grad_(False)
        frozen = parameter_counts(model)
        self.assertEqual(frozen["total"], counts["total"])
        self.assertEqual(frozen["trainable"], counts["total"] - 5146 * model.config.embedding_dim)
        with self.assertRaisesRegex(ValueError, "exceeding"):
            enforce_parameter_cap(model, counts["total"] - 1)
        self.assertEqual(enforce_parameter_cap(model, counts["total"]), frozen)
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                enforce_parameter_cap(model, limit)

    def test_configuration_is_frozen_validated_and_serializable(self):
        self.assertEqual(BaselineConfig.from_dict(self.config.to_dict()), self.config)
        with self.assertRaises(FrozenInstanceError):
            self.config.hidden_dim = 10
        for values in ({"embedding_dim": 0}, {"hidden_dim": True}, {"num_layers": -1},
                       {"dropout": 1.0}, {"dropout": -0.1}, {"dropout": float("nan")},
                       {"dropout": True}, {"dropout": "0.2"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                BaselineConfig(**values)
        for values in ({}, dict(self.config.to_dict(), unexpected=True)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                BaselineConfig.from_dict(values)

    def test_invalid_model_dimensions_are_rejected(self):
        for sizes in ((1, 4, 7), (25, 0, 7), (25, 4, -1), (25, True, 7)):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                JointBiLSTM(*sizes)
        with self.assertRaises(TypeError):
            JointBiLSTM(25, 4, 7, config={})

    def test_original_train_batch_integration(self):
        splits = load_massive(project_root() / "data/1.0/data/en-US.jsonl")
        mappings = fit_mappings(splits["train"])
        examples = prepare_split(splits["train"][:4], mappings)
        batch = collate_examples(examples)
        model = JointBiLSTM(len(mappings.token_to_id), len(mappings.intent_to_id), len(mappings.tag_to_id), self.config)
        output = model(batch["input_ids"], batch["lengths"], batch["attention_mask"])
        loss = nn.functional.cross_entropy(output["intent_logits"], batch["intent_labels"])
        loss = loss + nn.functional.cross_entropy(output["slot_logits"].flatten(0, 1), batch["slot_labels"].flatten())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.all(torch.isfinite(p.grad)) for p in model.parameters() if p.grad is not None))


if __name__ == "__main__":
    unittest.main()

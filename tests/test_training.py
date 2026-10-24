"""Training correctness on unchanged original train rows, never assessed runs.

Temporary checkpoints and diagnostic minibatches are isolated from experiment
artifacts. Numerical logits are unit-test inputs, not added task examples.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler

from src.data import (
    SLOT_IGNORE_INDEX, collate_examples, fit_mappings, load_massive,
    prepare_split,
)
from src.models import BaselineConfig, JointBiLSTM
from src.metrics import evaluate_predictions
from src.runtime import load_protocol, project_root
from src.training import (
    compute_losses, evaluate_model, fit_model, load_checkpoint, save_checkpoint,
    train_epoch, verify_reloaded_checkpoint,
)


class InputOnlyProbe(nn.Module):
    """Deterministic logits dependent on real inputs, with no label argument."""

    def __init__(self, num_intents, num_tags):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.2))
        self.num_intents = num_intents
        self.num_tags = num_tags
        self.calls = []

    def forward(self, input_ids, lengths, attention_mask=None):
        self.calls.append((self.training, torch.is_grad_enabled()))
        features = input_ids.remainder(17).to(torch.float32) / 17
        intent_axis = torch.linspace(-1, 1, self.num_intents, device=input_ids.device)
        tag_axis = torch.linspace(-1, 1, self.num_tags, device=input_ids.device)
        return {
            "intent_logits": self.anchor * features.sum(1, keepdim=True) * intent_axis,
            "slot_logits": self.anchor * features.unsqueeze(-1) * tag_axis,
        }


class TrainingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.root = project_root()
        cls.protocol = load_protocol(cls.root)
        splits = load_massive(cls.root / cls.protocol["paths"]["dataset"])
        cls.mappings = fit_mappings(splits["train"])
        by_id = {example.record_id: example for example in splits["train"]}
        cls.original = tuple(by_id[key] for key in ("4", "49", "57", "10235", "12270"))
        cls.prepared = prepare_split(cls.original, cls.mappings)
        cls.dev_prepared = prepare_split(splits["dev"][:2], cls.mappings)
        cls.device = torch.device("cpu")

    def loader(self, batch_size=2, prepared=None):
        return DataLoader(self.prepared if prepared is None else prepared,
                          batch_size=batch_size, shuffle=False,
                          collate_fn=collate_examples)

    def model(self, dropout=0.2):
        torch.manual_seed(42)
        return JointBiLSTM(len(self.mappings.token_to_id),
                          len(self.mappings.intent_to_id),
                          len(self.mappings.tag_to_id),
                          BaselineConfig(embedding_dim=8, hidden_dim=6,
                                         num_layers=1, dropout=dropout))

    def probe(self):
        return InputOnlyProbe(len(self.mappings.intent_to_id),
                              len(self.mappings.tag_to_id))

    @staticmethod
    def training_config(max_epochs=2):
        return {"optimizer": "AdamW", "learning_rate": 0.001,
                "weight_decay": 0.0001, "batch_size": 2,
                "max_epochs": max_epochs, "intent_loss_weight": 1.0,
                "slot_loss_weight": 1.0, "gradient_clip_norm": 5.0}

    def test_losses_match_independently_counted_example_and_token_means(self):
        batch = collate_examples(self.prepared[:3])
        output = self.probe()(batch["input_ids"], batch["lengths"], batch["attention_mask"])
        losses = compute_losses(output, batch, intent_weight=0.3, slot_weight=0.7)
        expected_intent = F.cross_entropy(output["intent_logits"], batch["intent_labels"])
        expected_slot = F.cross_entropy(output["slot_logits"][batch["attention_mask"]],
                                        batch["slot_labels"][batch["attention_mask"]])
        self.assertEqual(losses.num_examples, 3)
        self.assertEqual(losses.num_tokens, int(batch["lengths"].sum()))
        torch.testing.assert_close(losses.intent, expected_intent)
        torch.testing.assert_close(losses.slot, expected_slot)
        torch.testing.assert_close(losses.intent_sum, expected_intent * 3)
        torch.testing.assert_close(losses.slot_sum, expected_slot * losses.num_tokens)
        torch.testing.assert_close(losses.total, 0.3 * expected_intent + 0.7 * expected_slot)

    def test_padded_slot_logits_have_zero_gradient_and_do_not_change_loss(self):
        batch = collate_examples(self.prepared[:3])
        shape = batch["input_ids"].shape
        intent = torch.randn(shape[0], len(self.mappings.intent_to_id), requires_grad=True)
        slot = torch.randn(*shape, len(self.mappings.tag_to_id), requires_grad=True)
        losses = compute_losses({"intent_logits": intent, "slot_logits": slot}, batch)
        losses.total.backward()
        self.assertEqual(torch.count_nonzero(slot.grad[~batch["attention_mask"]]).item(), 0)
        self.assertTrue(torch.isfinite(slot.grad).all())
        self.assertGreater(torch.count_nonzero(slot.grad[batch["attention_mask"]]).item(), 0)
        changed = slot.detach().clone()
        changed[~batch["attention_mask"]] = 1000 * torch.randn_like(changed[~batch["attention_mask"]])
        other = compute_losses({"intent_logits": intent.detach(), "slot_logits": changed}, batch)
        torch.testing.assert_close(losses.total.detach(), other.total)

    def test_loss_rejects_padding_targets_and_ignored_real_token_targets(self):
        batch = collate_examples(self.prepared[:3])
        output = self.probe()(batch["input_ids"], batch["lengths"], batch["attention_mask"])
        corrupted = dict(batch)
        corrupted["slot_labels"] = batch["slot_labels"].clone()
        corrupted["slot_labels"][~batch["attention_mask"]] = 0
        with self.assertRaises(ValueError):
            compute_losses(output, corrupted)
        corrupted["slot_labels"] = batch["slot_labels"].clone()
        corrupted["slot_labels"][0, 0] = SLOT_IGNORE_INDEX
        with self.assertRaises(ValueError):
            compute_losses(output, corrupted)

    def test_epoch_aggregation_uses_global_example_and_token_denominators(self):
        model = self.probe()
        one_batch = evaluate_model(model, self.loader(5), self.mappings, self.device,
                                   intent_weight=0.3, slot_weight=0.7)
        uneven = evaluate_model(model, self.loader(2), self.mappings, self.device,
                                intent_weight=0.3, slot_weight=0.7)
        for key in ("loss", "intent_loss", "slot_loss"):
            self.assertAlmostEqual(one_batch[key], uneven[key], places=5)
        self.assertEqual(uneven["num_examples"], 5)
        self.assertEqual(uneven["num_tokens"], sum(len(row.tokens) for row in self.prepared))
        self.assertEqual(uneven["num_batches"], 3)
        self.assertEqual(one_batch["metrics"], uneven["metrics"])
        self.assertEqual(one_batch["predictions"], uneven["predictions"])

    def test_evaluation_disables_gradients_and_preserves_parameters(self):
        model = self.probe()
        model.train()
        before = deepcopy(model.state_dict())
        result = evaluate_model(model, self.loader(), self.mappings, self.device)
        self.assertTrue(model.calls)
        self.assertTrue(all(not training and not gradients for training, gradients in model.calls))
        self.assertEqual(tuple(row.record_id for row in result["predictions"]),
                         tuple(row.record_id for row in self.original))
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[name]))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_training_updates_parameters_with_finite_losses_and_gradients(self):
        model = self.model(dropout=0.0)
        before = deepcopy(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        result = train_epoch(model, self.loader(), optimizer, self.device)
        self.assertEqual(result["num_examples"], 5)
        self.assertEqual(result["num_batches"], 3)
        self.assertEqual(tuple(result["record_ids"]), tuple(row.record_id for row in self.original))
        for key in ("loss", "intent_loss", "slot_loss", "max_gradient_norm"):
            self.assertTrue(math.isfinite(result[key]), key)
        self.assertTrue(all(torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters() if parameter.grad is not None))
        self.assertTrue(any(not torch.equal(value, before[name])
                            for name, value in model.state_dict().items()))

    def test_training_rejects_nontraining_sidecars(self):
        altered = tuple(replace(row, example=replace(row.example, partition="dev"))
                        for row in self.prepared)
        model = self.model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with self.assertRaises(ValueError):
            train_epoch(model, self.loader(prepared=altered), optimizer, self.device)

    def test_nonfinite_gradients_are_rejected_before_optimizer_update(self):
        model = self.probe()
        before = model.anchor.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        hook = model.anchor.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
        try:
            with self.assertRaises(RuntimeError):
                train_epoch(model, self.loader(), optimizer, self.device)
        finally:
            hook.remove()
        self.assertTrue(torch.equal(model.anchor.detach(), before))
        self.assertEqual(len(optimizer.state), 0)

    def test_nonfinite_logits_cannot_be_logged_as_training_evidence(self):
        batch = collate_examples(self.prepared[:3])
        output = self.probe()(batch["input_ids"], batch["lengths"], batch["attention_mask"])
        for key in ("intent_logits", "slot_logits"):
            with self.subTest(logits=key):
                invalid = {name: values.detach().clone() for name, values in output.items()}
                invalid[key].flatten()[0] = float("nan")
                with self.assertRaises(ValueError):
                    compute_losses(invalid, batch)

    def test_test_partition_is_rejected_by_diagnostic_evaluation(self):
        # Deliberately changed partition metadata tests the guard, without reading
        # actual test labels or producing any held-out test-model result.
        altered = tuple(replace(row, example=replace(row.example, partition="test"))
                        for row in self.prepared)
        with self.assertRaises(ValueError):
            evaluate_model(self.probe(), self.loader(prepared=altered), self.mappings, self.device)
        with self.assertRaises(ValueError):
            evaluate_model(self.probe(), self.loader(prepared=altered), self.mappings,
                           self.device, allow_test=True, max_batches=1)
        with self.assertRaises(ValueError):
            evaluate_model(self.probe(), self.loader(prepared=altered), self.mappings,
                           self.device, allow_test=True)

    def test_saved_checkpoint_restores_fresh_model_mappings_and_exact_predictions(self):
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        train_epoch(model, self.loader(), optimizer, self.device)
        validation = evaluate_model(model, self.loader(), self.mappings, self.device)
        original_state = deepcopy(model.state_dict())
        source_hashes = {"src/models.py": hashlib.sha256((self.root / "src/models.py").read_bytes()).hexdigest()}
        data_hash = hashlib.sha256((self.root / self.protocol["paths"]["dataset"]).read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "selected.pt"
            save_checkpoint(checkpoint, model, optimizer, self.mappings,
                            self.training_config(), self.protocol, 1, validation,
                            source_hashes=source_hashes, data_sha256=data_hash,
                            purpose="diagnostic")
            checkpoint_bytes = checkpoint.read_bytes()
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(10)
            reloaded, mappings, payload = load_checkpoint(
                checkpoint, self.device, expected_mappings=self.mappings,
                expected_protocol=self.protocol, expected_data_sha256=data_hash)
            self.assertIsNot(reloaded, model)
            self.assertFalse(reloaded.training)
            self.assertEqual(mappings.to_dict(), self.mappings.to_dict())
            for name, value in reloaded.state_dict().items():
                self.assertTrue(torch.equal(value, original_state[name]), name)
            evaluated = evaluate_model(reloaded, self.loader(), mappings, self.device)
            self.assertEqual(evaluated["predictions"], validation["predictions"])
            self.assertEqual(evaluated["metrics"], validation["metrics"])
            self.assertEqual(checkpoint.read_bytes(), checkpoint_bytes)
            self.assertEqual(payload["source_hashes"], source_hashes)
            self.assertEqual(payload["data_sha256"], data_hash)
            self.assertEqual(payload["epoch"], 1)
            verified = verify_reloaded_checkpoint(
                checkpoint, self.loader(), self.device,
                expected_mappings=self.mappings, expected_protocol=self.protocol,
                expected_data_sha256=data_hash)
            self.assertIsInstance(verified, dict)

    def test_loading_checkpoint_rejects_wrong_mapping_protocol_and_dataset_identity(self):
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        validation = evaluate_model(model, self.loader(), self.mappings, self.device)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "selected.pt"
            save_checkpoint(checkpoint, model, optimizer, self.mappings,
                            self.training_config(), self.protocol, 1, validation,
                            data_sha256="a" * 64, purpose="diagnostic")
            changed_mappings = replace(self.mappings, training_ids_sha256="b" * 64)
            with self.assertRaises(ValueError):
                load_checkpoint(checkpoint, self.device, expected_mappings=changed_mappings)
            changed_protocol = deepcopy(self.protocol)
            changed_protocol["runtime"]["seed"] += 1
            with self.assertRaises(ValueError):
                load_checkpoint(checkpoint, self.device, expected_protocol=changed_protocol)
            with self.assertRaises(ValueError):
                load_checkpoint(checkpoint, self.device, expected_data_sha256="b" * 64)

    def test_fit_retains_earliest_exact_tie_and_refuses_artifact_overwrite(self):
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        training = train_epoch(model, self.loader(), optimizer, self.device, max_batches=1)
        # The fixed evaluator below isolates checkpoint selection from model
        # quality. These are original dev rows and no dev model score is run.
        dev_loader = self.loader(2, self.dev_prepared)
        predictions = tuple(row.example.gold_prediction() for row in self.dev_prepared)
        metrics = evaluate_predictions(predictions, predictions)
        validation = {"loss": 1.0, "intent_loss": 0.5, "slot_loss": 0.5,
                      "num_examples": 2, "num_tokens": sum(len(row.tokens) for row in self.dev_prepared),
                      "num_batches": 1, "seconds": 0.0,
                      "record_ids": [row.record_id for row in predictions], "limited": True,
                      "predictions": predictions, "metrics": metrics}
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "diagnostic"
            output_dir.mkdir()
            with patch("src.training.train_epoch", return_value=training), \
                    patch("src.training.evaluate_model", return_value=validation), \
                    patch("src.training.save_checkpoint", wraps=save_checkpoint) as saver:
                result = fit_model(
                    model, self.loader(), dev_loader, optimizer, self.mappings,
                    self.device, self.training_config(), self.protocol, output_dir,
                    purpose="diagnostic", max_train_batches=1, max_dev_batches=1)
            self.assertEqual(result["best_epoch"], 1)
            self.assertEqual(result["best_score"], 1.0)
            self.assertEqual(result["epochs_trained"], 2)
            self.assertEqual(saver.call_count, 1)
            checkpoint = Path(result["checkpoint"])
            original = checkpoint.read_bytes()
            with self.assertRaises((ValueError, FileExistsError)):
                fit_model(model, self.loader(), dev_loader, optimizer,
                          self.mappings, self.device, self.training_config(),
                          self.protocol, output_dir, purpose="diagnostic",
                          max_train_batches=1, max_dev_batches=1)
            self.assertEqual(checkpoint.read_bytes(), original)

    def test_corrupted_checkpoint_is_rejected_before_loading_model(self):
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        validation = evaluate_model(model, self.loader(), self.mappings, self.device)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "selected.pt"
            save_checkpoint(checkpoint, model, optimizer, self.mappings,
                            self.training_config(), self.protocol, 1, validation,
                            purpose="diagnostic")
            checkpoint.write_bytes(checkpoint.read_bytes() + b"deliberate unit-test corruption")
            with self.assertRaises((ValueError, RuntimeError)):
                load_checkpoint(checkpoint, self.device)

    def test_experiment_rejects_subset_sampler_before_saving_checkpoint(self):
        # Full original partition metadata makes the nominal dataset lengths
        # valid. Epoch functions are mocked: no full training or dev scoring.
        splits = load_massive(self.root / self.protocol["paths"]["dataset"])
        datasets = {partition: tuple(SimpleNamespace(example=row) for row in splits[partition])
                    for partition in ("train", "dev")}
        for omitted_partition in ("train", "dev"):
            with self.subTest(omitted_partition=omitted_partition), tempfile.TemporaryDirectory() as temporary:
                loaders = {
                    partition: DataLoader(
                        dataset, batch_size=2,
                        sampler=SubsetRandomSampler([0, 1]) if partition == omitted_partition else None)
                    for partition, dataset in datasets.items()
                }
                results = {}
                for partition in ("train", "dev"):
                    visited = splits[partition][:2] if partition == omitted_partition else splits[partition]
                    results[partition] = {"num_examples": len(visited),
                                          "record_ids": [row.record_id for row in visited],
                                          "limited": False}
                model = self.model()
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
                output_dir = Path(temporary) / "run"
                with patch("src.training.train_epoch", return_value=results["train"]), \
                        patch("src.training.evaluate_model", return_value=results["dev"]), \
                        patch("src.training.save_checkpoint") as saver:
                    with self.assertRaisesRegex(ValueError, f"every original {omitted_partition} record"):
                        fit_model(model, loaders["train"], loaders["dev"], optimizer,
                                  self.mappings, self.device, self.training_config(),
                                  self.protocol, output_dir, purpose="experiment")
                saver.assert_not_called()
                self.assertFalse((output_dir / "best.pt").exists())
                self.assertFalse((output_dir / "history.jsonl").exists())


if __name__ == "__main__":
    unittest.main()

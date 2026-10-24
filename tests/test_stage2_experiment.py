"""Training boundaries and aggregation, using unchanged original train records."""
from dataclasses import replace
from pathlib import Path
import unittest
import torch

from src.data import fit_mappings, load_massive, prepare_split
from src.runtime import load_protocol
from src.stage2_experiment import _epoch, evaluate, make_loader, new_model
from src.stage2_model import Stage2Config, fit_char_vocab


class Stage2ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        root = Path(__file__).resolve().parents[1]
        splits = load_massive(root / load_protocol(root)['paths']['dataset'])
        cls.mappings = fit_mappings(splits['train'])
        cls.prepared = prepare_split(splits['train'][:5], cls.mappings)
        cls.chars = fit_char_vocab(cls.prepared)

    def model(self):
        return new_model(self.mappings, self.chars, {'model': Stage2Config(embedding_dim=8, hidden_dim=6,
            use_char=True, use_crf=True, intent_conditioning=True, dropout=0.0).to_dict()}, 'cpu')

    def loader(self, batch_size=2, partition='train'):
        data = tuple(replace(x, example=replace(x.example, partition=partition)) for x in self.prepared)
        return make_loader(data, 'dev', batch_size, 42, self.chars)

    def test_loss_aggregation_and_predictions_ignore_minibatch_boundaries(self):
        model = self.model()
        small = evaluate(model, self.loader(2), self.mappings, 'cpu')
        full = evaluate(model, self.loader(5), self.mappings, 'cpu')
        self.assertAlmostEqual(small['intent_loss'], full['intent_loss'], places=5)
        self.assertAlmostEqual(small['slot_loss'], full['slot_loss'], places=5)
        self.assertEqual(small['predictions'], full['predictions'])
        self.assertEqual(small['num_examples'], 5)

    def test_updates_reject_development_records(self):
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        with self.assertRaisesRegex(ValueError, 'training records only'):
            _epoch(model, self.loader(partition='dev'), self.mappings, 'cpu', optimizer=optimizer)

    def test_test_is_blocked_even_when_permission_has_partial_loader(self):
        for allowed in (False, True):
            with self.assertRaisesRegex(ValueError, 'complete final evaluation'):
                evaluate(self.model(), self.loader(partition='test'), self.mappings, 'cpu', allow_test=allowed)

    def test_one_joint_update_reaches_all_active_components(self):
        model = self.model()
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        result = _epoch(model, self.loader(), self.mappings, 'cpu', optimizer=optimizer)
        for prefix in ('embedding.', 'encoder.', 'char_encoder.', 'crf.', 'intent_embeddings.', 'intent_head.', 'slot_head.'):
            self.assertTrue(any(not torch.equal(before[name], value) for name, value in model.named_parameters() if name.startswith(prefix)), prefix)
        self.assertEqual(result['num_examples'], 5)
        self.assertTrue(result['max_gradient_norm'] > 0)


if __name__ == '__main__':
    unittest.main()

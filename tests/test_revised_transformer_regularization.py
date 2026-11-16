"""CPU fixtures for optional regularization; no MASSIVE task-model training."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.revised_transformer_data import collate_transformer
from src.revised_transformer_model import JointTransformer
from src import revised_transformer_training as training
from tests.test_revised_transformer_model import tiny_config, fixture, TAGS
from tests.test_revised_transformer_training import TRAINING, sample_bundle, FixedLossModel


def loader(bundle, *, shuffle=False):
    return DataLoader(bundle['splits']['train'], batch_size=2, shuffle=shuffle,
        collate_fn=collate_transformer, generator=torch.Generator().manual_seed(42))


def fixture_model(bundle, dropout=0.):
    mappings = bundle['mappings']
    return JointTransformer(len(mappings['word_to_id']), 2, mappings['tag_names'],
        len(mappings['char_to_id']), tiny_config(dropout=dropout))


def fixture_optimizer(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    optimizer._rt_clip = 1.
    return optimizer


class RegularizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        torch.manual_seed(42)

    def test_optional_configuration_is_backward_compatible_and_validated(self):
        training.validate_training_config(TRAINING)
        training.validate_training_config(dict(TRAINING, word_dropout_alpha=0, intent_label_smoothing=0))
        training.validate_training_config(dict(TRAINING, word_dropout_alpha=.25, intent_label_smoothing=.05))
        for name, values in (('word_dropout_alpha', (-1, True, float('inf'), float('nan'))),
                             ('intent_label_smoothing', (-.1, 1, True, float('inf'), float('nan')))):
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    training.validate_training_config(dict(TRAINING, **{name: value}))

    def test_counts_and_expected_rates_use_only_original_training_tokens(self):
        bundle = sample_bundle()
        bundle['mappings']['word_to_id']['rare'] = 3
        bundle['splits']['train'] = deepcopy(bundle['splits']['train'])
        bundle['splits']['train'][0]['word_ids'] = [3]
        noise = training.FrequencyWordDropout.from_bundle(bundle, .25, 42)
        self.assertEqual(noise.metadata['word_counts'], [0, 0, 5, 1])
        self.assertEqual(noise.metadata['training_token_count'], 6)
        self.assertAlmostEqual(noise.probabilities[2].item(), .25 / 5.25)
        self.assertAlmostEqual(noise.probabilities[3].item(), .25 / 1.25)
        self.assertAlmostEqual(noise.metadata['expected_mask_fraction'], (5 * .25 / 5.25 + .2) / 6)
        # Deliberately corrupt held-out IDs: the fitting helper must never inspect them.
        bundle['splits']['dev'] = [{'word_ids': [999] * 1000}]
        bundle['splits']['test'] = [{'word_ids': [999] * 1000}]
        other = training.FrequencyWordDropout.from_bundle(bundle, .25, 42)
        self.assertEqual(noise.metadata, other.metadata)
        self.assertEqual(noise.metadata['fit_partition'], 'train')

    def test_masking_preserves_characters_targets_positions_and_excludes_pad_unk(self):
        bundle = sample_bundle()
        noise = training.FrequencyWordDropout.from_bundle(bundle, 1e20, 42)
        batch = collate_transformer(bundle['splits']['train'])
        batch['word_ids'][0, 0] = 1
        before = deepcopy(batch)
        words, evidence = noise.apply(batch['word_ids'], batch['mask'])
        self.assertEqual(evidence, {'masked_tokens': 5, 'eligible_tokens': 5})
        self.assertTrue(torch.equal(words[batch['mask']], torch.ones(6, dtype=torch.long)))
        self.assertTrue(torch.equal(words[~batch['mask']], torch.zeros(3, dtype=torch.long)))
        for name in ('word_ids', 'char_ids', 'slot_ids', 'intent_ids', 'mask'):
            self.assertTrue(torch.equal(batch[name], before[name]), name)
        self.assertEqual(batch['rows'], before['rows'])

    def test_private_rng_is_reproducible_and_never_consumes_global_draws(self):
        bundle = sample_bundle()
        left = training.FrequencyWordDropout.from_bundle(bundle, 3., 42)
        right = training.FrequencyWordDropout.from_bundle(bundle, 3., 42)
        words, mask = torch.full((20, 3), 2), torch.ones((20, 3), dtype=torch.bool)
        global_before = torch.get_rng_state().clone()
        private_before = left.rng_state_sha256()
        actual, evidence = left.apply(words, mask)
        self.assertTrue(torch.equal(global_before, torch.get_rng_state()))
        expected, other = right.apply(words, mask)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(evidence, other)
        self.assertNotEqual(private_before, left.rng_state_sha256())
        self.assertEqual(left.rng_state_sha256(), right.rng_state_sha256())

    def test_zero_noise_has_identity_and_no_random_draws(self):
        bundle = sample_bundle()
        noise = training.FrequencyWordDropout.from_bundle(bundle, 0., 42)
        batch = collate_transformer(bundle['splits']['train'])
        private_before, global_before = noise.rng_state_sha256(), torch.get_rng_state().clone()
        words, evidence = noise.apply(batch['word_ids'], batch['mask'])
        self.assertIs(words, batch['word_ids'])
        self.assertEqual(evidence, {'masked_tokens': 0, 'eligible_tokens': 6})
        self.assertEqual(private_before, noise.rng_state_sha256())
        self.assertTrue(torch.equal(global_before, torch.get_rng_state()))

    def test_word_noise_exposes_unk_embedding_to_task_gradient(self):
        bundle = sample_bundle()
        noise = training.FrequencyWordDropout.from_bundle(bundle, 1e20, 42)
        batch = collate_transformer(bundle['splits']['train'])
        model = fixture_model(bundle).train()
        state_names = tuple(model.state_dict())
        params = sum(parameter.numel() for parameter in model.parameters())
        outputs = model(batch['word_ids'], batch['mask'], batch['char_ids'])
        model.loss(outputs, batch['intent_ids'], batch['slot_ids'], batch['mask'])['total'].backward()
        self.assertEqual(model.embedding.weight.grad[1].abs().sum().item(), 0.)
        model.zero_grad(set_to_none=True)
        words, _ = noise.apply(batch['word_ids'], batch['mask'])
        outputs = model(words, batch['mask'], batch['char_ids'])
        model.loss(outputs, batch['intent_ids'], batch['slot_ids'], batch['mask'])['total'].backward()
        self.assertGreater(model.embedding.weight.grad[1].abs().sum().item(), 0.)
        self.assertEqual(tuple(model.state_dict()), state_names)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), params)

    def test_smoothing_matches_uniform_target_ce_and_keeps_crf_unchanged(self):
        model = JointTransformer(12, 3, TAGS, 9, tiny_config()).train()
        words, mask, chars, intents, slots = fixture()
        outputs = model(words, mask, chars)
        ordinary = model.loss(outputs, intents, slots, mask)
        zero = model.loss(outputs, intents, slots, mask, intent_label_smoothing=0.)
        smoothed = model.loss(outputs, intents, slots, mask, intent_label_smoothing=.05)
        log_probs = F.log_softmax(outputs['intent_logits'], dim=-1)
        expected = .95 * F.nll_loss(log_probs, intents, reduction='sum') - .05 * log_probs.mean(-1).sum()
        torch.testing.assert_close(smoothed['intent_sum'], expected)
        self.assertTrue(torch.equal(ordinary['slot_sum'], smoothed['slot_sum']))
        self.assertTrue(torch.equal(ordinary['total'], zero['total']))
        model.eval()
        evaluation = model.loss(outputs, intents, slots, mask, intent_label_smoothing=.05)
        self.assertTrue(torch.equal(evaluation['total'], ordinary['total']))

    def test_evaluation_ignores_training_noise_and_smoothing_even_for_legacy_fixture(self):
        bundle = sample_bundle()
        noise = training.FrequencyWordDropout.from_bundle(bundle, 1e20, 42)
        before = noise.rng_state_sha256()
        ordinary = training._rt_epoch(FixedLossModel(), loader(bundle), bundle, 'cpu')
        supplied = training._rt_epoch(FixedLossModel(), loader(bundle), bundle, 'cpu',
            word_dropout=noise, intent_label_smoothing=.05)
        self.assertEqual(before, noise.rng_state_sha256())
        for key in ('loss', 'intent_loss', 'slot_loss', 'predictions', 'metrics', 'record_order_sha256'):
            self.assertEqual(ordinary[key], supplied[key], key)

    def test_epoch_records_actual_order_counts_and_effective_regularization(self):
        bundle = sample_bundle()
        model = fixture_model(bundle)
        noise = training.FrequencyWordDropout.from_bundle(bundle, 1e20, 42)
        data_loader = loader(bundle, shuffle=True)
        expected_order = [row['record_id'] for batch in loader(bundle, shuffle=True) for row in batch['rows']]
        result = training._rt_epoch(model, data_loader, bundle, 'cpu', optimizer=fixture_optimizer(model),
            word_dropout=noise, intent_label_smoothing=.05)
        self.assertEqual(result['word_dropout_masked_tokens'], 6)
        self.assertEqual(result['word_dropout_eligible_tokens'], 6)
        self.assertEqual(result['word_dropout_alpha'], 1e20)
        self.assertEqual(result['intent_label_smoothing'], .05)
        self.assertEqual(result['record_order_sha256'], training.rt_json_hash(expected_order))
        self.assertEqual(result['record_ids_sha256'], training.rt_json_hash(sorted(expected_order)))
        self.assertEqual(result['word_dropout_rng_state_sha256'], noise.rng_state_sha256())

    def test_disabled_options_exactly_match_historical_training_weights_and_rng(self):
        # Use the preserved notebook implementation, not a reimplementation of the reference.
        path = Path(__file__).resolve().parents[1] / 'NLP_Final_project_revised.ipynb'
        notebook = json.loads(path.read_text(encoding='utf-8'))
        source = next(''.join(cell['source']) for cell in notebook['cells']
            if cell['cell_type'] == 'code' and 'def _rt_epoch(' in ''.join(cell['source']))
        definition = next(node for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == '_rt_epoch')
        self.assertNotIn('word_dropout', [arg.arg for arg in definition.args.kwonlyargs])
        scope = vars(training).copy()
        exec(compile(ast.Module(body=[definition], type_ignores=[]), '<historical-epoch>', 'exec'), scope)
        bundle = sample_bundle()
        original = fixture_model(bundle, dropout=.2)
        revised = deepcopy(original)
        torch.manual_seed(37)
        expected = scope['_rt_epoch'](original, loader(bundle, shuffle=True), bundle, 'cpu',
            optimizer=fixture_optimizer(original))
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(37)
        actual = training._rt_epoch(revised, loader(bundle, shuffle=True), bundle, 'cpu',
            optimizer=fixture_optimizer(revised), word_dropout=None, intent_label_smoothing=0.)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for name, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, revised.state_dict()[name]), name)
        for key in ('loss', 'intent_loss', 'slot_loss', 'maximum_gradient_norm', 'record_ids_sha256'):
            self.assertEqual(actual[key], expected[key], key)


if __name__ == '__main__':
    unittest.main()

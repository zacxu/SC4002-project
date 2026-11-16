"""Patience/schedule auditing with tiny CPU fixtures, never task-model training."""
import ast
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from src import revised_transformer_training as training
from src.revised_transformer_data import collate_transformer
from tests.test_revised_transformer_training import TRAINING, sample_bundle, FixedLossModel


def config(**overrides):
    return dict(TRAINING, **({'max_epochs': 60, 'early_stopping_patience': 8,
        'early_stopping_min_epochs': 20, 'early_stopping_min_delta': .0005,
        'schedule_epochs': 20} | overrides))


class PlateauFixture(FixedLossModel):
    """Constant predictions with optimizer-visible parameters and real RNG draws."""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(3, 2)
        self.offset = nn.Parameter(torch.ones(2))
        self.dropout = nn.Dropout(.2)

    def forward(self, words, mask, chars):
        outputs = super().forward(words, mask, chars)
        zero = 0 * (self.dropout(self.embedding(words)).sum() + self.offset.sum())
        return {key: value + zero for key, value in outputs.items()}

    def loss(self, outputs, intents, slots, mask):
        result = super().loss(outputs, intents, slots, mask)
        zero = 0 * outputs['intent_logits'].sum()
        for key in ('total', 'intent_sum', 'slot_sum'):
            result[key] = result[key] + zero
        return result


def old_run_stage():
    notebook = json.loads((Path(__file__).resolve().parents[1] / 'NLP_Final_project_revised.ipynb')
                          .read_text(encoding='utf-8'))
    source = next(''.join(cell['source']) for cell in notebook['cells']
        if cell['cell_type'] == 'code' and 'def run_stage(' in ''.join(cell['source']))
    definition = next(node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == 'run_stage')
    scope = vars(training).copy()
    exec(compile(ast.Module(body=[definition], type_ignores=[]), '<historical-run-stage>', 'exec'), scope)
    return scope['run_stage']


def run_fixture(training_config, *, historical=False):
    """Exercise real loop/checkpoint/history writes; stub external provenance and process launch."""
    bundle, trace = sample_bundle(), []
    real_epoch, real_hash = training._rt_epoch, training.rt_file_hash
    with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
        root = Path(temporary)
        dataset = root / 'fixture.jsonl'
        dataset.write_text('CPU bookkeeping fixture only', encoding='utf-8')

        def fixture_loader(data, partition, batch_size, seed, shuffle=False):
            return DataLoader(data['splits'][partition], batch_size=batch_size, shuffle=shuffle,
                collate_fn=collate_transformer, generator=torch.Generator().manual_seed(seed))

        def epoch(*args, **kwargs):
            result = real_epoch(*args, **kwargs)
            if kwargs.get('optimizer') is not None:
                trace.append({'state': training._rt_cpu_copy(args[0].state_dict()),
                              'rng': torch.get_rng_state().clone()})
            return result

        stack.enter_context(patch.object(training, '_rt_loader', side_effect=fixture_loader))
        stack.enter_context(patch.object(training, '_rt_model', side_effect=lambda bundle, cfg, device, matrix=None:
                                         PlateauFixture().to(device)))
        stack.enter_context(patch.object(training, 'rt_seed', side_effect=lambda runtime: torch.manual_seed(runtime['seed'])))
        stack.enter_context(patch.object(training, 'notebook_identity', return_value='reviewed'))
        stack.enter_context(patch.object(training, 'rt_file_hash', side_effect=lambda path:
            training.DATASET_SHA256 if Path(path) == dataset else real_hash(path)))
        stack.enter_context(patch.object(training, '_rt_epoch', side_effect=epoch))
        stack.enter_context(patch.object(training, '_rt_curves'))
        stack.enter_context(patch.object(training, '_rt_fresh_verifier', return_value={'passed': True,
            'fixture_note': 'Fresh-process launch stubbed; parent checkpoint reload remains real'}))
        runner = old_run_stage() if historical else training.run_stage
        with redirect_stdout(io.StringIO()):
            result = runner('stage2', {'model': {}, 'training': training_config}, {'seed': 42},
                {'evaluate_test': False}, bundle, None, 'cpu', root / 'run', dataset, 'reviewed', root / 'fixture.ipynb')
        payload = torch.load(root / 'run/best.pt', map_location='cpu', weights_only=True)
        history = [json.loads(line) for line in (root / 'run/history.jsonl').read_text().splitlines()]
        run = json.loads((root / 'run/run.json').read_text())
        prefix = root / 'run/schedule_horizon_evidence.json'
        prefix = json.loads(prefix.read_text()) if prefix.exists() else None
        return result, payload, history, trace, run, prefix


class EarlyStoppingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_config_defaults_and_boundary_guards(self):
        training.validate_training_config(TRAINING)
        training.validate_training_config(config())
        old = training.DevelopmentPatience(TRAINING)
        self.assertFalse(old.policy['enabled'])
        self.assertEqual(old.policy['schedule_epochs'], TRAINING['max_epochs'])
        for key, value in [('early_stopping_patience', -1), ('early_stopping_patience', True),
            ('early_stopping_min_epochs', 61), ('early_stopping_min_epochs', 2.),
            ('early_stopping_min_delta', -1), ('early_stopping_min_delta', 1.01),
            ('early_stopping_min_delta', float('nan')), ('schedule_epochs', 0),
            ('schedule_epochs', 61), ('schedule_epochs', True), ('schedule_epochs', 20.)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                training.validate_training_config(config(**{key: value}))

    def test_old_twenty_epoch_schedule_is_bitwise_identical_and_tail_is_constant(self):
        horizon = 20 * 181
        original = [training.learning_rate_factor(step, horizon, .1, .1) for step in range(1, horizon + 1)]
        extended = [training.scheduled_learning_rate_factor(step, horizon, .1, .1)
                    for step in range(1, 60 * 181 + 1)]
        self.assertEqual(original, extended[:horizon])
        self.assertEqual(extended[horizon:], [.1] * (40 * 181))
        for step, steps in [(0, horizon), (True, horizon), (1, 0), (1, 1.)]:
            with self.assertRaises(ValueError):
                training.scheduled_learning_rate_factor(step, steps, .1, .1)

    def test_plateau_stops_at_minimum_epoch_without_resetting_counter(self):
        tracker = training.DevelopmentPatience(config())
        states = [tracker.update(epoch, .8) for epoch in range(1, 21)]
        self.assertTrue(all(not row['early_stopping_eligible'] for row in states[:19]))
        self.assertEqual(states[-1]['early_stopping_bad_epochs'], 19)
        self.assertEqual(states[-1]['termination_reason'], 'patience')
        self.assertTrue(states[-1]['early_stopping_triggered'])
        end = tracker.termination(20 * 181, 181)
        self.assertEqual((end['epochs_trained'], end['optimizer_steps'], end['maximum_optimizer_steps']), (20, 3620, 10860))
        self.assertTrue(end['stopped_early'])

    def test_exact_patience_counts_after_last_meaningful_improvement(self):
        tracker = training.DevelopmentPatience(config(early_stopping_min_epochs=0))
        for epoch in range(1, 8):
            state = tracker.update(epoch, .7 + epoch / 100)
            self.assertIsNone(state['termination_reason'])
        for epoch in range(8, 15):
            self.assertIsNone(tracker.update(epoch, .77)['termination_reason'])
        last = tracker.update(15, .77)
        self.assertEqual(last['early_stopping_anchor_epoch'], 7)
        self.assertEqual(last['early_stopping_bad_epochs'], 8)
        self.assertEqual(last['termination_reason'], 'patience')
        with self.assertRaisesRegex(ValueError, 'cannot continue'):
            tracker.update(16, .9)

    def test_cumulative_subthreshold_gains_reset_anchor_and_ties_never_reset(self):
        tracker = training.DevelopmentPatience(config(early_stopping_min_epochs=0, early_stopping_patience=4))
        states = [tracker.update(epoch, score) for epoch, score in enumerate([.8, .8002, .8004, .8005, .8005], 1)]
        self.assertEqual([state['early_stopping_meaningful_improvement'] for state in states], [True, False, False, True, False])
        self.assertEqual([state['early_stopping_bad_epochs'] for state in states], [0, 1, 2, 0, 1])
        zero = training.DevelopmentPatience(config(early_stopping_min_delta=0.))
        zero.update(1, .8)
        self.assertFalse(zero.update(2, .8)['early_stopping_meaningful_improvement'])

    def test_actual_best_is_independent_of_patience_anchor(self):
        tracker = training.DevelopmentPatience(config(early_stopping_min_epochs=0, early_stopping_patience=3))
        scores = [.8, .8002, .8004, .8003]
        states = [tracker.update(epoch, score) for epoch, score in enumerate(scores, 1)]
        self.assertEqual(states[-1]['termination_reason'], 'patience')
        self.assertEqual(states[-1]['early_stopping_anchor_epoch'], 1)
        self.assertEqual(max(enumerate(scores, 1), key=lambda item: item[1])[0], 3)

    def test_max_budget_takes_precedence_and_disabled_policy_does_not_stop(self):
        tracker = training.DevelopmentPatience(config(max_epochs=9, schedule_epochs=3,
            early_stopping_min_epochs=0, early_stopping_patience=8))
        states = [tracker.update(epoch, .8) for epoch in range(1, 10)]
        self.assertEqual(states[-1]['termination_reason'], 'max_epochs')
        self.assertFalse(states[-1]['early_stopping_triggered'])
        disabled = training.DevelopmentPatience(config(max_epochs=9, schedule_epochs=3,
            early_stopping_min_epochs=0, early_stopping_patience=0))
        states = [disabled.update(epoch, .8) for epoch in range(1, 10)]
        self.assertFalse(any(row['early_stopping_eligible'] or row['early_stopping_triggered'] for row in states))
        self.assertTrue(all(row['termination_reason'] is None for row in states[:-1]))
        self.assertEqual(states[-1]['early_stopping_bad_epochs'], 8)

    def test_nonconsecutive_invalid_score_and_incomplete_termination_are_rejected(self):
        for epoch, score in [(2, .8), (True, .8), (1, float('nan')), (1, 1.1), (1, True)]:
            with self.subTest(epoch=epoch, score=score), self.assertRaises(ValueError):
                training.DevelopmentPatience(config()).update(epoch, score)
        tracker = training.DevelopmentPatience(config())
        tracker.update(1, .8)
        with self.assertRaises(ValueError):
            tracker.termination(181, 181)
        tracker = training.DevelopmentPatience(dict(TRAINING, max_epochs=1))
        tracker.update(1, .8)
        with self.assertRaises(ValueError):
            tracker.termination(180, 181)

    def test_real_stage_loop_stops_after_complete_epochs_and_records_prefix_evidence(self):
        summary, payload, history, trace, run, prefix = run_fixture(config(max_epochs=8,
            early_stopping_min_epochs=4, early_stopping_patience=2, early_stopping_min_delta=1., schedule_epochs=2))
        self.assertEqual(len(history), 4)
        self.assertEqual(summary['epochs_trained'], 4)
        self.assertEqual(summary['termination']['reason'], 'patience')
        self.assertEqual(summary['termination']['optimizer_steps'], 8)
        self.assertEqual(summary['termination']['maximum_optimizer_steps'], 16)
        self.assertEqual([row['optimizer_step'] for row in history], [2, 4, 6, 8])
        self.assertEqual(history[-1]['termination_reason'], 'patience')
        self.assertEqual(payload['epoch'], 1)  # exact-score ties retain the earliest checkpoint
        self.assertEqual(payload['stopping_policy'], run['stopping_policy'])
        self.assertEqual(summary['stopping_policy'], run['stopping_policy'])
        self.assertEqual(payload['early_stopping_state']['early_stopping_anchor_epoch'], 1)
        self.assertEqual(prefix['epoch'], 2)
        self.assertEqual(prefix['current_model_state_sha256'], training.rt_state_dict_hash(trace[1]['state']))
        self.assertEqual(prefix['prefix_best_model_state_sha256'], training.rt_state_dict_hash(payload['model_state']))
        self.assertEqual(prefix['prefix_best_dev_predictions_sha256'], training.rt_json_hash(payload['dev_predictions']))
        self.assertEqual(prefix['prefix_best_checkpoint_sha256'], summary['checkpoint_sha256'])
        self.assertEqual(history[2]['learning_rates'], history[3]['learning_rates'])
        self.assertEqual(summary['test_model_evaluations'], 0)

    def test_no_options_preserve_historical_full_training_states_and_rng_bitwise(self):
        settings = dict(TRAINING, max_epochs=3)
        old = run_fixture(settings, historical=True)
        new = run_fixture(settings)
        self.assertEqual(len(old[3]), len(new[3]))
        for original, current in zip(old[3], new[3]):
            self.assertTrue(torch.equal(original['rng'], current['rng']))
            for name, value in original['state'].items():
                self.assertTrue(torch.equal(value, current['state'][name]), name)
        for left, right in zip(old[2], new[2]):
            for name in ('train_loss', 'dev_loss', 'selection_score', 'optimizer_step', 'learning_rates', 'selected'):
                self.assertEqual(left[name], right[name], name)
        self.assertEqual(old[0]['dev']['metrics'], new[0]['dev']['metrics'])
        self.assertEqual(new[0]['termination']['reason'], 'max_epochs')
        self.assertEqual(new[0]['epochs_trained'], 3)
        self.assertIsNone(new[5])


if __name__ == '__main__':
    unittest.main()

"""Read-only evidence checks for longer-run comparisons; no training/inference."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import analyze_transformer_early_stopping as analysis
from scripts import analyze_transformer_regularization as common
from tests.test_transformer_regularization_analysis import fixture_data, write_json


def prefix_fixture(best_epoch=20):
    history = []
    for epoch in range(1, 21):
        accuracy = .5+min(epoch, best_epoch)/100
        if epoch > best_epoch:
            accuracy -= .01
        history.append({'epoch': epoch, 'dev_intent_accuracy': accuracy,
            'dev_slot_micro_f1': .5, 'selection_score': .5*(accuracy+.5),
            'train_loss': 1./epoch, 'dev_loss': 1.2, 'train_record_order_sha256': str(epoch),
            'train_word_dropout_rng_state_sha256': str(epoch)+'rng',
            'train_seconds': 1., 'dev_seconds': .1, 'learning_rates': [.0001, .0005]})
    best = max(history, key=lambda row: row['selection_score'])
    metrics = {'intent_accuracy': best['dev_intent_accuracy'], 'slot_micro_f1': .5,
               'slot_tp': 50, 'slot_fp': 50, 'slot_fn': 50, 'num_examples': 100,
               'correct_intents': 50+best_epoch}
    reference = {'metrics': metrics, 'model_state_sha256': 'a'*64, 'initial_model_state_sha256': 'b'*64}
    predictions = [{'record_id': 'fixture', 'intent': 'intent', 'slots': []}]
    horizon = {'prefix_best_epoch': best['epoch'], 'prefix_best_selection_score': best['selection_score'],
        'prefix_best_model_state_sha256': reference['model_state_sha256'], 'prefix_best_dev_metrics': metrics,
        'prefix_best_dev_predictions_sha256': common.json_hash(predictions),
        'initial_model_state_sha256': reference['initial_model_state_sha256'],
        'current_model_state_sha256': reference['model_state_sha256'] if best_epoch == 20 else 'c'*64,
        'current_dev_metrics': metrics, 'current_dev_predictions_sha256': common.json_hash(predictions)}
    return history, horizon, reference, predictions


class EarlyStoppingResultsTests(unittest.TestCase):
    def test_exact_prefix_allows_only_timing_and_new_bookkeeping_changes(self):
        history, horizon, reference, predicted = prefix_fixture()
        candidate = deepcopy(history)
        for row in candidate:
            row['train_seconds'] *= 2
            row['dev_seconds'] *= 3
            row['early_stopping_bad_epochs'] = 0
        result = analysis.verify_prefix(history, candidate, horizon, reference, predicted)
        self.assertEqual(result['status'], 'passed')
        self.assertTrue(result['epoch20_current_state_comparable_to_preserved_checkpoint'])
        self.assertIn('train_word_dropout_rng_state_sha256', result['exact_history_fields'])

    def test_prefix_rejects_truncation_loss_rng_and_learning_rate_drift(self):
        history, horizon, reference, predicted = prefix_fixture()
        with self.assertRaisesRegex(ValueError, 'complete original and candidate 20-epoch prefix'):
            analysis.verify_prefix(history, history[:19], horizon, reference, predicted)
        for key, value in [('train_loss', 9.), ('dev_loss', 9.),
                ('train_record_order_sha256', 'changed'), ('train_word_dropout_rng_state_sha256', 'changed'),
                ('learning_rates', [.0001, .0006])]:
            with self.subTest(field=key):
                changed = deepcopy(history)
                changed[8][key] = value
                with self.assertRaisesRegex(ValueError, 'First-20 trajectory differs at epoch 9'):
                    analysis.verify_prefix(history, changed, horizon, reference, predicted)

    def test_prefix_binds_historical_state_predictions_and_earliest_winner(self):
        history, horizon, reference, predicted = prefix_fixture()
        for key, value in [('prefix_best_epoch', 19), ('prefix_best_model_state_sha256', 'd'*64),
                           ('prefix_best_dev_predictions_sha256', 'e'*64), ('initial_model_state_sha256', 'f'*64),
                           ('current_model_state_sha256', 'c'*64)]:
            with self.subTest(field=key):
                changed = {**horizon, key: value}
                with self.assertRaises(ValueError):
                    analysis.verify_prefix(history, history, changed, reference, predicted)

    def test_epoch18_reference_does_not_claim_epoch20_tensor_comparison(self):
        history, horizon, reference, predicted = prefix_fixture(best_epoch=18)
        result = analysis.verify_prefix(history, history, horizon, reference, predicted)
        self.assertEqual(result['prefix_best_epoch'], 18)
        self.assertFalse(result['epoch20_current_state_comparable_to_preserved_checkpoint'])
        self.assertTrue(result['prefix_best_model_state_identical'])

    def test_train_mapping_reconstruction_does_not_fit_development_tokens(self):
        data = fixture_data()
        mappings = analysis.train_mappings(data)
        self.assertEqual(mappings['word_to_id']['<UNK>'], 1)
        self.assertNotIn('oslo', mappings['word_to_id'])
        self.assertNotIn('zebra', mappings['word_to_id'])
        self.assertNotIn('other', mappings['intent_to_id'])
        self.assertEqual(mappings['tag_names'], ['O', 'B-place', 'I-place'])

    def test_registered_controls_preserve_other_training_settings(self):
        reference = {'model_configuration': {'width': 512}, 'runtime': {'seed': 42},
            'initial_model_state_sha256': 'same', 'training_configuration': {
                'max_epochs': 20, 'learning_rate': .0005, 'word_dropout_alpha': .25, 'intent_label_smoothing': .05}}
        run = {'stage_config': {'model': reference['model_configuration'], 'training': {
                **reference['training_configuration'], **analysis.EXPECTED_POLICY}},
            'runtime': reference['runtime'], 'initial_model_state_sha256': 'same'}
        protocol = {'stage2_policy': deepcopy(analysis.EXPECTED_POLICY)}
        analysis.verify_run_settings(run, reference, protocol)
        changed = deepcopy(run); changed['stage_config']['training']['learning_rate'] = .001
        with self.assertRaisesRegex(ValueError, 'beyond the registered budget controls'):
            analysis.verify_run_settings(changed, reference, protocol)
        changed = deepcopy(run); changed['stage_config']['training']['early_stopping_patience'] = 5
        with self.assertRaisesRegex(ValueError, '60/20/8/0.0005'):
            analysis.verify_run_settings(changed, reference, protocol)

    def test_checkpoint_configuration_normalizes_tuples_but_rejects_value_and_type_drift(self):
        run = {'code_sha256': 'source', 'stage_config': {'model': {'char_kernels': [3, 4, 5]},
            'training': {'learning_rate': .0005}}, 'runtime': {'deterministic_algorithms': True, 'seed': 42}}
        summary = {'code_sha256': 'source'}
        selected = {'selected_epoch': 18, 'selection_score': .8}
        predictions, metrics = [], {'intent_accuracy': .8}
        payload = {'purpose': 'fresh_joint_transformer', 'stage': 'stage2',
            'dataset_sha256': common.DATASET_SHA256, 'evaluator_version': common.EVALUATOR_VERSION,
            'code_sha256': 'source', 'stage_config': deepcopy(run['stage_config']), 'runtime': deepcopy(run['runtime']),
            'stage_config_sha256': common.json_hash(run['stage_config']), 'epoch': 18, 'selection_score': .8,
            'dev_predictions': predictions, 'dev_metrics': metrics}
        payload['stage_config']['model']['char_kernels'] = (3, 4, 5)
        analysis.validate_checkpoint_header(payload, run, summary, selected, predictions, metrics)
        changed = deepcopy(payload); changed['stage_config']['model']['char_kernels'] = (5, 4, 3)
        changed_lr = deepcopy(payload); changed_lr['stage_config']['training']['learning_rate'] = .001
        changed_bool = deepcopy(payload); changed_bool['runtime']['deterministic_algorithms'] = 1
        for candidate in (changed, changed_lr, changed_bool):
            with self.assertRaisesRegex(ValueError, 'Selected CPU checkpoint'):
                analysis.validate_checkpoint_header(candidate, run, summary, selected, predictions, metrics)

    def test_candidate_discovery_rejects_duplicate_complete_locations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'seed_42/full_notebook/transformer_run_artifacts/stage2'
            path.mkdir(parents=True)
            write_json(path/'final_summary.json', {'status': 'complete'})
            self.assertEqual(analysis.candidate_stage(root, 42), path)
            second = root/'seed_42/extended/stage2'
            second.mkdir(parents=True)
            write_json(second/'final_summary.json', {'status': 'complete'})
            with self.assertRaisesRegex(ValueError, 'Duplicate completed'):
                analysis.candidate_stage(root, 42)

    def test_test_evaluation_and_wrong_dataset_fail_before_checkpoint_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'stage2'
            path.mkdir()
            run = {'stage': 'stage2', 'runtime': {'seed': 42}, 'run_options': {'evaluate_test': False},
                   'dataset_sha256': common.DATASET_SHA256, 'evaluator_version': common.EVALUATOR_VERSION}
            base = {'status': 'complete', 'stage': 'stage2', 'mode': 'development_only',
                    'test_model_evaluations': 0, 'test': None,
                    'dataset_sha256': common.DATASET_SHA256, 'evaluator_version': common.EVALUATOR_VERSION}
            write_json(path/'run.json', run)
            write_json(path/'training_summary.json', {})
            (path/'history.jsonl').write_text('{}\n', encoding='utf-8')
            for changed in ({**base, 'mode': 'final', 'test_model_evaluations': 1},
                            {**base, 'dataset_sha256': 'wrong dataset'}):
                write_json(path/'final_summary.json', changed)
                with patch('torch.load', side_effect=AssertionError('No checkpoint read allowed')):
                    with self.assertRaises(ValueError):
                        analysis.audit_candidate(Path(directory), path, 42, {}, {}, [], [], {}, {}, {})


if __name__ == '__main__':
    unittest.main()

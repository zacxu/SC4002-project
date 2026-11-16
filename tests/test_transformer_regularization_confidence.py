"""Confidence math and DEV-only collection checks on CPU fixtures."""
from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts import analyze_transformer_regularization_confidence as confidence
from src.revised_transformer_data import collate_transformer
from src.revised_transformer_training import rt_state_hash
from tests.test_revised_transformer_training import sample_bundle, FixedLossModel


class SentinelModel(FixedLossModel):
    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.tensor([2.]))


class ConfidenceTests(unittest.TestCase):
    def test_checkpoint_settings_allow_only_json_equivalent_tuple_list_representation(self):
        payload = {'stage_config': {'model': {'char_kernels': (3, 4, 5)},
            'training': {'learning_rate': .0005, 'intent_label_smoothing': .05}},
            'runtime': {'seed': 17, 'deterministic_algorithms': True, 'precision': 'float32'}}
        run = deepcopy(payload)
        run['stage_config']['model']['char_kernels'] = [3, 4, 5]
        confidence.validate_checkpoint_settings(payload, run)
        for group, key, value in [('model', 'char_kernels', [3, 5, 4]),
                                  ('training', 'learning_rate', .0004),
                                  ('training', 'intent_label_smoothing', 0.),
                                  ('runtime', 'seed', 42),
                                  ('runtime', 'deterministic_algorithms', 1)]:
            changed = deepcopy(run)
            (changed['runtime'] if group == 'runtime' else changed['stage_config'][group])[key] = value
            with self.subTest(group=group, key=key), self.assertRaisesRegex(ValueError, 'effective settings differ'):
                confidence.validate_checkpoint_settings(payload, changed)
        changed = deepcopy(run)
        changed['stage_config']['training']['unexpected_option'] = 0
        with self.assertRaisesRegex(ValueError, 'effective settings differ'):
            confidence.validate_checkpoint_settings(payload, changed)

    def test_calibration_matches_hand_calculation_and_population_weighting(self):
        probabilities = np.asarray([[.8, .2], [.4, .6], [.9, .1], [.1, .9]])
        result, predicted, values = confidence.calibration_metrics(np.log(probabilities), np.zeros(4, dtype=int))
        self.assertEqual(predicted.tolist(), [0, 1, 0, 1])
        np.testing.assert_allclose(values, [.8, .6, .9, .9])
        self.assertEqual(result['intent_accuracy'], .5)
        self.assertEqual(result['intent_errors'], 2)
        self.assertAlmostEqual(result['negative_log_likelihood'], -sum(math.log(p) for p in [.8, .4, .9, .1]) / 4)
        self.assertAlmostEqual(result['multiclass_brier_score'], .61)
        self.assertAlmostEqual(result['mean_confidence'], .8)
        self.assertAlmostEqual(result['ece_15'], .4)
        self.assertEqual(sum(row['count'] for row in result['reliability_bins']), 4)
        self.assertEqual(result['top2_accuracy'], 1.)
        self.assertEqual(result['top3_accuracy'], 1.)

    def test_extreme_wrong_logits_are_finite_and_confidence_one_is_in_last_bin(self):
        result, _, _ = confidence.calibration_metrics([[1000., -1000.]], np.array([1]))
        self.assertEqual(result['negative_log_likelihood'], 2000.)
        self.assertEqual(result['multiclass_brier_score'], 2.)
        self.assertEqual(result['ece_15'], 1.)
        self.assertEqual(result['incorrect_confidence_ge_0_90'], 1)
        self.assertEqual(result['incorrect_confidence_ge_0_99'], 1)
        self.assertEqual(result['reliability_bins'][-1]['count'], 1)

    def test_top_k_ties_are_stable_and_logit_translation_leaves_metrics_unchanged(self):
        logits, targets = np.array([[1., 1., 1., 0.]]), np.array([2])
        result, predicted, _ = confidence.calibration_metrics(logits, targets)
        shifted, _, _ = confidence.calibration_metrics(logits + 1024., targets)
        self.assertEqual(result, shifted)
        self.assertEqual(predicted.tolist(), [0])
        self.assertEqual(result['top2_accuracy'], 0.)
        self.assertEqual(result['top3_accuracy'], 1.)

    def test_invalid_logits_targets_or_bins_fail_closed(self):
        invalid = [([], [], 15), ([[float('nan'), 1]], [0], 15),
            ([[1, 2]], [.5], 15), ([[1, 2]], [2], 15), ([[1, 2]], [0], 0)]
        for logits, targets, bins in invalid:
            with self.subTest(logits=logits), self.assertRaises(ValueError):
                confidence.calibration_metrics(logits, np.asarray(targets), bins)

    def test_development_collection_reproduces_predictions_without_gradients_or_mutation(self):
        bundle = sample_bundle()
        data = DataLoader(bundle['splits']['dev'], batch_size=2, collate_fn=collate_transformer)
        expected = [dict(bundle['gold_by_id'][row['record_id']], intent='a') for row in bundle['splits']['dev']]
        model = SentinelModel().train()
        before = rt_state_hash(model)
        with patch('torch.optim.AdamW', side_effect=AssertionError('No optimizer allowed')):
            logits, targets, predictions = confidence.collect_dev_logits(model, data, bundle, 'cpu', expected)
        self.assertEqual(predictions, expected)
        self.assertEqual(logits.shape, (3, 2))
        self.assertEqual(logits.dtype, np.float32)
        self.assertEqual(targets.tolist(), [0, 1, 0])
        self.assertFalse(model.training)
        self.assertIsNone(model.parameter.grad)
        self.assertEqual(before, rt_state_hash(model))

    def test_collection_rejects_wrong_predictions_or_non_development_population(self):
        bundle = sample_bundle()
        data = DataLoader(bundle['splits']['dev'], batch_size=2, collate_fn=collate_transformer)
        expected = [deepcopy(bundle['gold_by_id'][row['record_id']]) for row in bundle['splits']['dev']]
        with self.assertRaisesRegex(ValueError, 'exactly reproduce'):
            confidence.collect_dev_logits(SentinelModel(), data, bundle, 'cpu', expected)
        other = deepcopy(bundle['splits']['dev'])
        other[0]['record_id'] = 'train_only_id'
        data = DataLoader(other, batch_size=2, collate_fn=collate_transformer)
        with self.assertRaisesRegex(ValueError, 'only the original ordered development'):
            confidence.collect_dev_logits(SentinelModel(), data, bundle, 'cpu', expected)

    def test_partial_comparison_is_rejected_before_any_model_construction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'protocol.json').write_text('{"protocol":{"source_sha256":{}}}')
            with patch.object(confidence, 'file_hash', return_value='hash'), \
                 patch.object(confidence, 'analyze', return_value={'status': 'partial'}), \
                 patch.object(confidence, 'audit_checkpoint', side_effect=AssertionError('No model allowed')):
                with self.assertRaisesRegex(ValueError, 'Finish every registered'):
                    confidence.assess(root, root / 'data.jsonl', 'cpu')


if __name__ == '__main__':
    unittest.main()

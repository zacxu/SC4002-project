"""Bookkeeping fixtures for revised training; no task-model training is run."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.revised_transformer_data import collate_transformer
from src.revised_transformer_model import RevisedTransformerConfig, JointTransformer, enforce_parameter_cap
from src.revised_transformer_training import (capture_configuration, notebook_configuration,
    notebook_identity, learning_rate_factor, validate_training_config, _rt_cpu_copy,
    _rt_epoch, _rt_loader, _rt_checkpoint, _rt_load_checkpoint, rt_json_hash,
    report_results)
from src.comparison_contract import DATASET_SHA256, EVALUATOR_VERSION


TRAINING = {'max_epochs': 2, 'batch_size': 2, 'learning_rate': .0005,
    'embedding_learning_rate': .0001, 'weight_decay': .01, 'gradient_clip_norm': 1,
    'warmup_fraction': .1, 'min_lr_ratio': .1}


def document():
    values = {'RUNTIME': {'seed': 42}, 'STAGE1_CONFIG': {'training': TRAINING},
              'STAGE2_CONFIG': {'training': TRAINING}, 'RUN_OPTIONS': {'evaluate_test': False}}
    cells = [{'cell_type': 'code', 'source': 'PROTECTED = 1\n', 'metadata': {}},
             {'cell_type': 'code', 'source': '\n'.join(name + ' = ' + repr(value) for name, value in values.items()),
              'metadata': {'tags': ['configuration']}}]
    return {'cells': cells, 'metadata': {'revised_transformer': {
        'code_cell_sha256': [hashlib.sha256(cell['source'].encode()).hexdigest() for cell in cells],
        'configuration_cell_indices': [1]}}}, values


def sample_bundle():
    rows = []
    for index, length in enumerate((1, 3, 2)):
        utt = ' '.join(['a'] * length)
        rows.append({'record_id': str(index), 'utt': utt, 'token_offsets': [(2 * j, 2 * j + 1) for j in range(length)],
            'word_ids': [2] * length, 'char_ids': [[2]] * length, 'intent_id': int(index == 1),
            'slot_ids': [0] * length})
    return {'splits': {'dev': rows, 'train': rows, 'test': rows}, 'gold_by_id': {
        row['record_id']: {'record_id': row['record_id'], 'utt': row['utt'],
            'intent': 'b' if row['intent_id'] else 'a', 'slots': []} for row in rows},
        'mappings': {'word_to_id': {'<PAD>': 0, '<UNK>': 1, 'a': 2},
            'char_to_id': {'<PAD>': 0, '<UNK>': 1, 'a': 2}, 'intent_to_id': {'a': 0, 'b': 1},
            'tag_names': ['O', 'B-x', 'I-x'], 'slot_types': ['x']}}


class FixedLossModel(nn.Module):
    def forward(self, words, mask, chars):
        return {'intent_logits': torch.tensor([[1., 0.]]).expand(words.shape[0], -1),
                'slot_logits': torch.zeros((*words.shape, 3))}

    def loss(self, outputs, intents, slots, mask):
        examples, tokens = mask.shape[0], int(mask.sum())
        return {'total': torch.tensor(5.), 'intent_sum': torch.tensor(2. * examples),
                'slot_sum': torch.tensor(3. * tokens), 'num_examples': examples, 'num_tokens': tokens}

    def decode(self, outputs, mask):
        return torch.zeros_like(mask, dtype=torch.long)


class RevisedTrainingTests(unittest.TestCase):
    def test_warmup_cosine_boundaries_and_monotonic_decay(self):
        values = [learning_rate_factor(step, 20, .2, .1) for step in range(1, 21)]
        self.assertEqual(values[0], .25)
        self.assertEqual(values[3], 1.)
        self.assertAlmostEqual(values[-1], .1)
        self.assertTrue(all(left >= right for left, right in zip(values[3:], values[4:])))
        with self.assertRaises(ValueError): learning_rate_factor(0, 20, .2, .1)

    def test_training_numeric_and_schema_guards(self):
        validate_training_config(TRAINING)
        for name, value in [('max_epochs', True), ('batch_size', 0), ('weight_decay', -1),
                            ('learning_rate', float('nan')), ('warmup_fraction', 1), ('min_lr_ratio', 2)]:
            config = dict(TRAINING, **{name: value})
            with self.subTest(name=name), self.assertRaises(ValueError): validate_training_config(config)
        with self.assertRaises(ValueError): validate_training_config(dict(TRAINING, optimizer='SGD'))

    def test_configuration_only_edits_change_identity_without_rehashing_metadata(self):
        doc, values = document()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'notebook.ipynb'; path.write_text(json.dumps(doc))
            before = notebook_identity(path)
            values['STAGE1_CONFIG']['training'] = dict(TRAINING, max_epochs=4)
            doc['cells'][1]['source'] = '\n'.join(name + ' = ' + repr(value) for name, value in values.items())
            path.write_text(json.dumps(doc))
            self.assertEqual(notebook_configuration(path), values)
            self.assertNotEqual(before, notebook_identity(path))

    def test_definition_edit_or_extra_configuration_code_is_rejected(self):
        for key in ('protected', 'extra'):
            doc, values = document()
            if key == 'protected': doc['cells'][0]['source'] += '# changed\n'
            else: doc['cells'][1]['source'] += '\nimport os\n'
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'notebook.ipynb'; path.write_text(json.dumps(doc))
                with self.assertRaises(ValueError): notebook_identity(path)

    def test_unsaved_settings_cannot_be_recorded_as_executed(self):
        doc, values = document()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'notebook.ipynb'; path.write_text(json.dumps(doc))
            values['RUN_OPTIONS']['evaluate_test'] = True
            with self.assertRaisesRegex(ValueError, 'Save'):
                capture_configuration(path, *values.values(), Path(temporary) / 'snapshot.json')

    def test_cpu_checkpoint_copy_has_no_live_tensor_alias(self):
        original = {'x': torch.tensor([1.]), 'nested': [torch.tensor([2.])]}
        saved = _rt_cpu_copy(original)
        original['x'].fill_(7); original['nested'][0].fill_(9)
        self.assertEqual(saved['x'].item(), 1)
        self.assertEqual(saved['nested'][0].item(), 2)

    def test_global_denominators_and_complete_order_are_preserved(self):
        bundle = sample_bundle()
        loader = DataLoader(bundle['splits']['dev'], batch_size=2, collate_fn=collate_transformer)
        result = _rt_epoch(FixedLossModel(), loader, bundle, 'cpu')
        self.assertEqual((result['num_examples'], result['num_tokens'], result['num_batches']), (3, 6, 2))
        self.assertEqual((result['intent_loss'], result['slot_loss'], result['loss']), (2., 3., 5.))
        self.assertAlmostEqual(result['metrics']['intent_accuracy'], 2 / 3)
        self.assertEqual([row['record_id'] for row in result['predictions']], ['0', '1', '2'])

    def test_original_split_population_guard_is_not_silently_disabled(self):
        with self.assertRaisesRegex(ValueError, 'every original'):
            _rt_loader(sample_bundle(), 'dev', 2, 42)

    def test_checkpoint_checksum_configuration_and_mapping_guards(self):
        bundle = sample_bundle()
        config = RevisedTransformerConfig.original_baseline().to_dict()
        config.update(word_dim=4, d_model=8, n_heads=2, n_layers=1, d_ff=16)
        model = JointTransformer(3, 2, ['O', 'B-x', 'I-x'], 3,
                                 config=RevisedTransformerConfig.from_dict(config))
        payload = {'evaluator_version': EVALUATOR_VERSION, 'dataset_sha256': DATASET_SHA256,
            'code_sha256': 'reviewed', 'mappings': bundle['mappings'], 'stage_config': {'model': config},
            'model_state': _rt_cpu_copy(model.state_dict()), 'parameter_counts': enforce_parameter_cap(model)}
        payload['stage_config_sha256'] = rt_json_hash(payload['stage_config'])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'best.pt'; _rt_checkpoint(path, payload)
            restored, _ = _rt_load_checkpoint(path, bundle, 'cpu', DATASET_SHA256, 'reviewed')
            self.assertTrue(all(torch.equal(model.state_dict()[key], restored.state_dict()[key]) for key in model.state_dict()))
            with self.assertRaisesRegex(ValueError, 'scorer/data/code'):
                _rt_load_checkpoint(path, bundle, 'cpu', DATASET_SHA256, 'changed')
            path.write_bytes(path.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                _rt_load_checkpoint(path, bundle, 'cpu', DATASET_SHA256, 'reviewed')

    def test_development_report_contains_two_rows_and_only_real_saved_errors(self):
        bundle = sample_bundle(); metrics = {'intent_accuracy': .5, 'slot_micro_f1': 0.}
        summary = {'dev': {'metrics': metrics}, 'test': None, 'parameter_counts': {'total': 10, 'trainable': 10},
                   'epochs_trained': 2, 'selected_epoch': 1}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / 'stage2').mkdir()
            predictions = [dict(bundle['gold_by_id']['0'], intent='b')]
            (root / 'stage2/dev_predictions.jsonl').write_text('\n'.join(json.dumps(row) for row in predictions))
            result = report_results({'stage1': summary, 'stage2': deepcopy(summary)}, root, bundle)
            self.assertEqual(len(result['rows']), 2)
            self.assertEqual({row['split'] for row in result['rows']}, {'dev'})
            cases = json.loads((root / 'error_cases.json').read_text())['cases']
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0]['record_id'], '0')


if __name__ == '__main__': unittest.main()

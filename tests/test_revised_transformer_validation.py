"""Adversarial saved-evidence fixtures; never train or forward a task model."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.validate_revised_transformer import (DATA_SHA256, ValidationError,
    check_metrics, configuration_values, evidence, file_hash, json_hash,
    manual_metrics, state_evidence, validate_history, validate_mode_artifacts)


def record(slots=(), key='one', intent='a', utt='room room'):
    return {'record_id': key, 'intent': intent, 'utt': utt, 'slots': list(slots)}


def slot(start=0, text='room', label='place'):
    return {'slot_type': label, 'text': text, 'start': start, 'end': start + len(text)}


class RevisedTransformerValidationTests(unittest.TestCase):
    def test_wrong_repeated_surface_occurrence_is_not_a_match(self):
        result = manual_metrics([record([slot()])], [record([slot(5)])])
        self.assertEqual((result['slot_tp'], result['slot_fp'], result['slot_fn']), (0, 1, 1))

    def test_repeated_gold_occurrences_do_not_collapse(self):
        result = manual_metrics([record([slot(), slot(5)])], [record([slot()])])
        self.assertEqual((result['slot_tp'], result['slot_fp'], result['slot_fn']), (1, 0, 1))
        self.assertAlmostEqual(result['slot_micro_f1'], 2 / 3)

    def test_wrong_type_or_boundary_gets_no_partial_credit(self):
        gold = [record([slot()])]
        for span in (slot(label='name'), slot(text='roo')):
            with self.subTest(span=span):
                metrics = manual_metrics(gold, [record([span])])
                self.assertEqual((metrics['slot_tp'], metrics['slot_fp'], metrics['slot_fn']), (0, 1, 1))

    def test_duplicate_missing_extra_and_out_of_order_records_fail(self):
        gold = [record(key='one'), record(key='two')]
        for predicted in ([record(key='one')], [record(key='one'), record(key='one')],
                          [record(key='one'), record(key='extra')], [record(key='two'), record(key='one')]):
            with self.subTest(predicted=predicted), self.assertRaises(ValidationError):
                manual_metrics(gold, predicted)
        self.assertEqual(manual_metrics(gold, list(reversed(gold)), require_order=False)['intent_accuracy'], 1)

    def test_invalid_offsets_literal_text_and_duplicate_spans_fail(self):
        cases = [record([slot(text='ROOM')]), record([slot(6)]), record([slot(), slot()])]
        bad = slot(); bad['start'] = True
        cases.append(record([bad]))
        for predicted in cases:
            with self.subTest(predicted=predicted), self.assertRaises(ValidationError):
                manual_metrics([record([slot()])], [predicted])

    def test_zero_slots_fixed_zero_policy_and_empty_split_error(self):
        self.assertEqual(manual_metrics([record()], [record()])['slot_micro_f1'], 0)
        with self.assertRaises(ValidationError):
            manual_metrics([], [])

    def test_metric_comparison_rejects_changed_counts_nonfinite_and_boolean(self):
        actual = manual_metrics([record([slot()])], [record([slot()])])
        check_metrics(deepcopy(actual), actual)
        for key, value in (('slot_tp', 2), ('intent_accuracy', float('nan')),
                           ('slot_micro_f1', .99), ('correct_intents', True)):
            recorded = {**actual, key: value}
            with self.subTest(key=key), self.assertRaises(ValidationError):
                check_metrics(recorded, actual)

    def test_configuration_is_literal_and_contains_all_four_dicts(self):
        good = '\n'.join(name + ' = {}' for name in ('RUNTIME', 'STAGE1_CONFIG', 'STAGE2_CONFIG', 'RUN_OPTIONS'))
        self.assertEqual(len(configuration_values(good)), 4)
        for text in (good + '\nimport os', good + '\nOTHER = {}', good + '\nRUN_OPTIONS = {}',
                     good.replace('RUN_OPTIONS = {}', 'RUN_OPTIONS = dict()')):
            with self.subTest(text=text), self.assertRaises(ValidationError):
                configuration_values(text)

    def history_fixture(self):
        split = {'train': [{'id': str(i)} for i in range(5)], 'dev': [{'id': str(i + 5)} for i in range(3)]}
        tokens, training = {'train': 9, 'dev': 5}, {'max_epochs': 3, 'batch_size': 2}
        rows = []
        for epoch, score in enumerate((.5, .6, .6), 1):
            row = {'epoch': epoch, 'dev_intent_accuracy': score, 'dev_slot_micro_f1': score,
                   'selection_score': score, 'selected': epoch < 3, 'optimizer_step': epoch * 3}
            for part, batch in (('train', 3), ('dev', 2)):
                row.update({part + '_num_examples': len(split[part]), part + '_num_tokens': tokens[part],
                            part + '_num_batches': batch, part + '_loss': 2., part + '_intent_loss': 1.,
                            part + '_slot_loss': 1., part + '_record_ids_sha256': json_hash([r['id'] for r in split[part]])})
            rows.append(row)
        return rows, training, split, tokens

    def test_configured_history_keeps_earliest_tied_epoch(self):
        rows, training, split, tokens = self.history_fixture()
        result = validate_history(rows, training, split, tokens)
        self.assertEqual(result, {'epochs_trained': 3, 'selected_epoch': 2, 'selection_score': .6})

    def test_incomplete_or_tampered_history_fails(self):
        rows, training, split, tokens = self.history_fixture()
        for key, value in (('train_num_tokens', 8), ('dev_record_ids_sha256', 'wrong'),
                           ('selected', True), ('optimizer_step', 8), ('train_loss', float('inf'))):
            wrong = deepcopy(rows); wrong[-1][key] = value
            with self.subTest(key=key), self.assertRaises(ValidationError):
                validate_history(wrong, training, split, tokens)
        with self.assertRaises(ValidationError):
            validate_history(rows[:-1], training, split, tokens)

    def test_all_float_parameter_tensors_count_even_without_grad_and_bool_masks_do_not(self):
        state = {'embedding.weight': torch.zeros(2, 3), 'head.bias': torch.ones(2),
                 'crf.start_allowed': torch.tensor([True, False])}
        total, fingerprint = state_evidence(state, {'total': 8, 'trainable': 8})
        self.assertEqual(total, 8)
        self.assertEqual(len(fingerprint), 64)
        with self.assertRaises(ValidationError):
            state_evidence(state, {'total': 2, 'trainable': 2})
        with self.assertRaisesRegex(ValidationError, 'Boolean CRF'):
            state_evidence({**state, 'unaccounted.weight': torch.ones(3, dtype=torch.bool)}, {'total': 8, 'trainable': 8})
        oversized = {'weight': torch.zeros(1).expand(20_000_001)}
        with self.assertRaisesRegex(ValidationError, 'cap'):
            state_evidence(oversized, {'total': 20_000_001, 'trainable': 20_000_001})

    def test_evidence_paths_cannot_escape_the_run_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / 'ok.json').write_text('{}')
            self.assertEqual(evidence(root, 'ok.json'), (root / 'ok.json').resolve())
            for name in ('../ok.json', 'C:/outside', '/outside', 'a\\b', 'missing'):
                with self.subTest(name=name), self.assertRaises(ValidationError):
                    evidence(root, name)

    def freeze_fixture(self, root):
        directory = root / 'stage1'; directory.mkdir()
        (directory / 'history.jsonl').write_text('registered history\n')
        freeze = {'test_model_evaluations': 0, 'tuning_closed': True, 'checkpoint_sha256': 'checkpoint',
                  'model_state_sha256': 'weights', 'history_sha256': file_hash(directory / 'history.jsonl'),
                  'selected_epoch': 2, 'code_sha256': 'code', 'dataset_sha256': DATA_SHA256,
                  'frozen_at_utc': '2026-11-02T12:00:02+00:00'}
        (directory / 'freeze_manifest.json').write_text(json.dumps(freeze))
        started = {'test_model_evaluations': 1, 'freeze_sha256': file_hash(directory / 'freeze_manifest.json'),
                   'started_at_utc': '2026-11-02T12:00:03+00:00'}
        (directory / 'test_started.json').write_text(json.dumps(started))
        summary = {'completed_at_utc': '2026-11-02T12:00:04+00:00'}
        reload = {'verified_at_utc': '2026-11-02T12:00:01+00:00'}
        return summary, reload, started

    def test_final_evaluation_must_follow_a_matching_pretest_freeze(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); summary, reload, started = self.freeze_fixture(root)
            validate_mode_artifacts(root, 'stage1', 'final', summary, reload, 'checkpoint', 'weights', {'selected_epoch': 2}, 'code')
            started['started_at_utc'] = '2026-11-02T12:00:00+00:00'
            (root / 'stage1/test_started.json').write_text(json.dumps(started))
            with self.assertRaisesRegex(ValidationError, 'chronology'):
                validate_mode_artifacts(root, 'stage1', 'final', summary, reload, 'checkpoint', 'weights', {'selected_epoch': 2}, 'code')

    def test_development_mode_rejects_any_test_or_freeze_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / 'stage1').mkdir()
            validate_mode_artifacts(root, 'stage1', 'development_only', {'test': None}, {}, '', '', {}, '')
            (root / 'stage1/test_started.json').write_text('{}')
            with self.assertRaisesRegex(ValidationError, 'test/freeze'):
                validate_mode_artifacts(root, 'stage1', 'development_only', {'test': None}, {}, '', '', {}, '')


if __name__ == '__main__':
    unittest.main()

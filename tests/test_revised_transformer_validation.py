"""Adversarial saved-evidence fixtures; never train or forward a task model."""
from copy import deepcopy
import ast
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.validate_revised_transformer import (DATA_SHA256, ValidationError,
    check_metrics, configuration_values, evidence, file_hash, json_hash,
    manual_metrics, state_evidence, validate_history, validate_history_csv,
    validate_mode_artifacts, validate_regularization, stopping_policy,
    expected_learning_rates, validate_stopping_evidence, validate_schedule_horizon,
    DEFINITION_MODULES, validate_definition_layout)


def record(slots=(), key='one', intent='a', utt='room room'):
    return {'record_id': key, 'intent': intent, 'utt': utt, 'slots': list(slots)}


def slot(start=0, text='room', label='place'):
    return {'slot_type': label, 'text': text, 'start': start, 'end': start + len(text)}


def definition_fixture(version=2):
    cells, provenance = [], {}
    sha = lambda value: hashlib.sha256(value.encode('utf-8')).hexdigest()
    for index, name in enumerate(sorted(DEFINITION_MODULES)):
        fragments = [f'VALUE_{index} = {index}\n']
        if version == 2 and name == 'src/revised_transformer_training.py':
            fragments = ['from __future__ import annotations\n"Module documentation."\n' + fragments[0],
                         'from __future__ import annotations\ndef train_marker():\n    return 1\n']
        nodes = [ast.parse(f'VALUE_{index} = {index}').body[0]]
        if len(fragments) == 2:
            nodes += ast.parse('def train_marker():\n    return 1').body
        provenance[name] = {'file_sha256': sha(fragments[0]), 'inline_sha256': sha('\n\n'.join(fragments))}
        if version == 2:
            provenance[name].update(inline_cell_sha256=[sha(text) for text in fragments],
                semantic_ast_sha256=sha(ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)))
        cells.extend({'cell_type': 'code', 'source': text, 'metadata': {
            'tags': ['definitions'], **({'source_module': name} if version == 2 else {})}} for text in fragments)
    metadata = {'source_provenance': provenance}
    if version == 2:
        metadata['definition_layout_version'] = version
    return cells, metadata


class RevisedTransformerValidationTests(unittest.TestCase):
    def test_original_whole_module_definition_layout_remains_valid(self):
        cells, metadata = definition_fixture(version=1)
        validate_definition_layout(cells, metadata)
        cells[0]['source'] += 'changed = True\n'
        with self.assertRaisesRegex(ValidationError, 'inline definitions'):
            validate_definition_layout(cells, metadata)

    def test_split_definition_layout_checks_all_three_hash_levels(self):
        cells, metadata = definition_fixture()
        validate_definition_layout(cells, metadata)
        for key, value in [('inline_cell_sha256', ['0'*64]), ('inline_sha256', '0'*64),
                           ('semantic_ast_sha256', '0'*64)]:
            changed = deepcopy(metadata)
            changed['source_provenance']['src/revised_transformer_training.py'][key] = value
            with self.subTest(field=key), self.assertRaises(ValidationError):
                validate_definition_layout(cells, changed)

    def test_split_definition_layout_rejects_missing_duplicate_and_unaccounted_fragments(self):
        original, metadata = definition_fixture()
        index = next(i for i, cell in enumerate(original)
                     if cell['metadata']['source_module'] == 'src/revised_transformer_training.py')
        variants = []
        changed = deepcopy(original); del changed[index]; variants.append(changed)
        changed = deepcopy(original); changed.insert(index, deepcopy(changed[index])); variants.append(changed)
        changed = deepcopy(original); changed[index], changed[index+1] = changed[index+1], changed[index]; variants.append(changed)
        changed = deepcopy(original); changed.insert(0, changed.pop(index)); variants.append(changed)
        changed = deepcopy(original); del changed[0]['metadata']['source_module']; variants.append(changed)
        changed = deepcopy(original); changed[0]['metadata']['source_module'] = 'src/unknown.py'; variants.append(changed)
        changed = deepcopy(original); changed[0]['metadata']['tags'] = []; variants.append(changed)
        for i, cells in enumerate(variants):
            with self.subTest(case=i), self.assertRaises(ValidationError):
                validate_definition_layout(cells, metadata)

    def test_split_definition_layout_rejects_module_and_version_downgrade(self):
        cells, original = definition_fixture()
        for version in (1, 0, 3, True):
            changed = deepcopy(original); changed['definition_layout_version'] = version
            with self.subTest(version=version), self.assertRaises(ValidationError):
                validate_definition_layout(cells, changed)
        changed = deepcopy(original)
        changed['source_provenance']['src/unexpected.py'] = changed['source_provenance'].pop('src/stage2_model.py')
        with self.assertRaisesRegex(ValidationError, 'module membership'):
            validate_definition_layout(cells, changed)

    def test_split_definition_layout_keeps_canonical_contract_whole_and_unchanged(self):
        cells, metadata = definition_fixture()
        name = 'src/comparison_contract.py'
        metadata['source_provenance'][name]['file_sha256'] = '0'*64
        with self.assertRaisesRegex(ValidationError, 'one unchanged cell'):
            validate_definition_layout(cells, metadata)

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
        tokens = {'train': 9, 'dev': 5}
        training = {'max_epochs': 3, 'batch_size': 2, 'learning_rate': .0005,
                    'embedding_learning_rate': .0001, 'warmup_fraction': 0., 'min_lr_ratio': .1}
        rows = []
        for epoch, score in enumerate((.5, .6, .6), 1):
            row = {'epoch': epoch, 'dev_intent_accuracy': score, 'dev_slot_micro_f1': score,
                   'selection_score': score, 'selected': epoch < 3, 'optimizer_step': epoch * 3,
                   'learning_rates': [rate * (.775, .325, .1)[epoch - 1] for rate in (.0001, .0005)]}
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

    def early_stopping_fixture(self):
        original, training, split, tokens = self.history_fixture()
        training.update(max_epochs=10, early_stopping_patience=3, early_stopping_min_epochs=4,
                        early_stopping_min_delta=.01, schedule_epochs=3)
        # Sub-delta gains accumulate until epoch 3; later tiny gains still win checkpoints.
        scores = (.5, .505, .51, .512, .515, .517)
        anchors = (.5, .5, .51, .51, .51, .51)
        anchor_epochs, bad = (1, 1, 3, 3, 3, 3), (0, 1, 0, 1, 2, 3)
        rows = []
        for epoch, score in enumerate(scores, 1):
            row = deepcopy(original[0])
            row.update(epoch=epoch, dev_intent_accuracy=score, dev_slot_micro_f1=score,
                selection_score=score, selected=True, optimizer_step=3*epoch,
                learning_rates=[rate * (.775 if epoch == 1 else .325 if epoch == 2 else .1)
                                for rate in (.0001, .0005)],
                early_stopping_anchor_score=anchors[epoch-1], early_stopping_anchor_epoch=anchor_epochs[epoch-1],
                early_stopping_bad_epochs=bad[epoch-1], early_stopping_meaningful_improvement=epoch in (1, 3),
                early_stopping_eligible=epoch >= 4, early_stopping_triggered=epoch == 6,
                termination_reason='patience' if epoch == 6 else None)
            rows.append(row)
        return rows, training, split, tokens

    def test_early_stop_uses_cumulative_anchor_and_saves_actual_best_below_delta(self):
        rows, training, split, tokens = self.early_stopping_fixture()
        result = validate_history(rows, training, split, tokens)
        self.assertEqual(result, {'epochs_trained': 6, 'selected_epoch': 6, 'selection_score': .517})
        self.assertEqual(rows[-1]['early_stopping_anchor_epoch'], 3)

    def test_early_stop_rejects_premature_truncation_continued_training_and_bad_counters(self):
        rows, training, split, tokens = self.early_stopping_fixture()
        with self.assertRaisesRegex(ValidationError, 'ended before'):
            validate_history(rows[:-1], training, split, tokens)
        extended = deepcopy(rows)
        extended.append(dict(extended[-1], epoch=7, optimizer_step=21, selected=False))
        with self.assertRaisesRegex(ValidationError, 'continues after'):
            validate_history(extended, training, split, tokens)
        for key, value in (('early_stopping_bad_epochs', 0), ('early_stopping_anchor_epoch', 5),
                           ('early_stopping_anchor_score', .515), ('early_stopping_triggered', False),
                           ('early_stopping_eligible', 1), ('termination_reason', 'max_epochs')):
            changed = deepcopy(rows); changed[-1][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValidationError, 'stopping state'):
                validate_history(changed, training, split, tokens)
        changed = deepcopy(rows); del changed[0]['early_stopping_triggered']
        with self.assertRaisesRegex(ValidationError, 'stopping state'):
            validate_history(changed, training, split, tokens)

    def test_minimum_epoch_gates_stopping_without_resetting_the_counter_and_ties_never_reset(self):
        rows, training, split, tokens = self.early_stopping_fixture()
        training.update(early_stopping_patience=2, early_stopping_min_epochs=6, early_stopping_min_delta=0.)
        for epoch, row in enumerate(rows, 1):
            row.update(dev_intent_accuracy=.5, dev_slot_micro_f1=.5, selection_score=.5, selected=epoch == 1,
                early_stopping_anchor_score=.5, early_stopping_anchor_epoch=1,
                early_stopping_bad_epochs=epoch-1, early_stopping_meaningful_improvement=epoch == 1,
                early_stopping_eligible=epoch == 6)
        result = validate_history(rows, training, split, tokens)
        self.assertEqual(result['selected_epoch'], 1)
        self.assertEqual(rows[-1]['early_stopping_bad_epochs'], 5)

    def test_maximum_budget_precedes_patience_and_disabled_policy_requires_full_budget(self):
        rows, training, split, tokens = self.early_stopping_fixture()
        training['max_epochs'] = 6
        rows[-1].update(early_stopping_triggered=False, termination_reason='max_epochs')
        self.assertEqual(validate_history(rows, training, split, tokens)['epochs_trained'], 6)
        training['early_stopping_patience'] = 0
        for row in rows:
            row.update(early_stopping_eligible=False, early_stopping_triggered=False)
        self.assertEqual(validate_history(rows, training, split, tokens)['epochs_trained'], 6)
        with self.assertRaisesRegex(ValidationError, 'early stopping disabled'):
            validate_history(rows[:-1], training, split, tokens)

    def test_schedule_matches_warmup_and_holds_floor_beyond_twenty_epoch_horizon(self):
        training = {'max_epochs': 60, 'schedule_epochs': 20, 'learning_rate': .0005,
                    'embedding_learning_rate': .0001, 'warmup_fraction': .1, 'min_lr_ratio': .1}
        for epoch, factor in ((1, .5), (2, 1.), (20, .1), (21, .1), (60, .1)):
            actual = expected_learning_rates(training, 5, epoch)
            with self.subTest(epoch=epoch):
                for observed, base in zip(actual, (.0001, .0005)):
                    self.assertAlmostEqual(observed, base*factor, places=15)
        rows, configuration, split, tokens = self.early_stopping_fixture()
        rows[3]['learning_rates'] = [.0001, .0005]
        with self.assertRaisesRegex(ValidationError, 'learning rates differ'):
            validate_history(rows, configuration, split, tokens)
        for key, value in (('schedule_epochs', 61), ('early_stopping_patience', True),
                           ('early_stopping_min_epochs', 61), ('early_stopping_min_delta', float('nan'))):
            with self.subTest(key=key), self.assertRaises(ValidationError):
                stopping_policy(dict(training, **{key: value}))

    def test_termination_metadata_and_selected_checkpoint_state_are_independently_checked(self):
        rows, training, split, tokens = self.early_stopping_fixture()
        selected = validate_history(rows, training, split, tokens)
        policy = stopping_policy(training)
        termination = {'reason': 'patience', 'epochs_trained': 6, 'max_epochs': 10,
            'optimizer_steps': 18, 'maximum_optimizer_steps': 30, 'schedule_epochs': 3, 'schedule_steps': 9,
            'last_meaningful_improvement_epoch': 3, 'meaningful_improvement_anchor_score': .51,
            'non_improving_epochs': 3, 'stopped_early': True}
        state = {key: value for key, value in rows[-1].items()
                 if key.startswith('early_stopping_') or key == 'termination_reason'}
        records = [{'stopping_policy': deepcopy(policy)} for _ in range(4)]
        records[1]['termination'] = deepcopy(termination)
        records[2]['early_stopping_state'] = deepcopy(state)
        records[3]['termination'] = deepcopy(termination)
        result = validate_stopping_evidence(rows, training, 5, selected, *records)
        self.assertEqual(result['evidence'], 'independently_reconstructed')
        for index, key, value in ((1, 'optimizer_steps', 30), (3, 'last_meaningful_improvement_epoch', 6),
                                  (3, 'reason', 'max_epochs'), (3, 'stopped_early', 1)):
            changed = deepcopy(records); changed[index]['termination'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValidationError, 'Termination evidence'):
                validate_stopping_evidence(rows, training, 5, selected, *changed)
        changed = deepcopy(records); changed[2]['early_stopping_state']['early_stopping_anchor_score'] = .517
        with self.assertRaisesRegex(ValidationError, 'Selected checkpoint stopping state'):
            validate_stopping_evidence(rows, training, 5, selected, *changed)
        with self.assertRaisesRegex(ValidationError, 'lack stopping/schedule evidence'):
            validate_stopping_evidence(rows, training, 5, selected, {}, {}, {}, {})
        legacy, legacy_config, legacy_split, legacy_tokens = self.history_fixture()
        chosen = validate_history(legacy, legacy_config, legacy_split, legacy_tokens)
        self.assertEqual(validate_stopping_evidence(legacy, legacy_config, 5, chosen, {}, {}, {}, {})['evidence'],
                         'legacy_full_budget')

    def schedule_horizon_fixture(self, path):
        rows, training, _, _ = self.early_stopping_fixture()
        gold = [record([slot()], key='d' + str(i)) for i in range(10)]
        predicted = deepcopy(gold)
        for row in predicted[7:]:
            row['intent'] = 'wrong'
            row['slots'] = [slot(label='wrong')]
        metrics = manual_metrics(gold, predicted)
        for epoch, row in enumerate(rows, 1):
            score = (.5, .6, .7)[min(epoch, 3) - 1]
            row.update(dev_num_examples=10, dev_num_tokens=20, dev_num_batches=5,
                dev_record_ids_sha256=json_hash([r['record_id'] for r in gold]),
                dev_intent_accuracy=score, dev_slot_micro_f1=score, selection_score=score,
                selected=epoch <= 3, early_stopping_anchor_score=score,
                early_stopping_anchor_epoch=min(epoch, 3), early_stopping_meaningful_improvement=epoch <= 3,
                early_stopping_bad_epochs=max(0, epoch - 3), train_record_order_sha256=json_hash(['0', '1', '2', '3', '4']))
        raw = ''.join(json.dumps(row) + '\n' for row in rows).encode('utf-8')
        (path / 'history.jsonl').write_bytes(raw)
        run = {'code_sha256': 'b' * 64, 'initial_model_state_sha256': 'c' * 64}
        payload = {'dev_metrics': metrics, 'dev_predictions': predicted}
        selected = {'epochs_trained': 6, 'selected_epoch': 3, 'selection_score': .7}
        horizon = {'schema_version': 1, 'stage': 'stage2', 'epoch': 3, 'schedule_epochs': 3,
            'maximum_epochs': 10, 'code_sha256': run['code_sha256'], 'dataset_sha256': DATA_SHA256,
            'evaluator_version': 'exact_character_span_v2', 'initial_model_state_sha256': run['initial_model_state_sha256'],
            'optimizer_step': 9, 'current_model_state_sha256': 'a' * 64,
            'current_dev_metrics': metrics, 'current_dev_predictions_sha256': json_hash(predicted),
            'prefix_best_epoch': 3, 'prefix_best_selection_score': .7, 'prefix_best_model_state_sha256': 'a' * 64,
            'prefix_best_dev_metrics': metrics, 'prefix_best_dev_predictions_sha256': json_hash(predicted),
            'prefix_best_checkpoint_sha256': 'd' * 64,
            'prefix_history_sha256': hashlib.sha256(b''.join(raw.splitlines(keepends=True)[:3])).hexdigest(),
            'training_record_order_sha256': rows[2]['train_record_order_sha256']}
        (path / 'schedule_horizon_evidence.json').write_text(json.dumps(horizon), encoding='utf-8')
        return (path, 'stage2', rows, training, 5, gold, run, payload, selected, 'a' * 64, 'd' * 64), horizon

    def test_schedule_horizon_binds_prefix_winner_to_unchanged_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            args, horizon = self.schedule_horizon_fixture(path)
            result = validate_schedule_horizon(*args)
            self.assertTrue(result['required'])
            self.assertTrue(result['prefix_best_bound_to_final_checkpoint'])
            later = deepcopy(list(args))
            later[3]['max_epochs'] = 6
            later[2][-1].update(dev_intent_accuracy=.8, dev_slot_micro_f1=.8, selection_score=.8, selected=True,
                early_stopping_anchor_score=.8, early_stopping_anchor_epoch=6, early_stopping_bad_epochs=0,
                early_stopping_meaningful_improvement=True, early_stopping_triggered=False, termination_reason='max_epochs')
            (path / 'history.jsonl').write_bytes(''.join(json.dumps(row) + '\n' for row in later[2]).encode('utf-8'))
            (path / 'schedule_horizon_evidence.json').write_text(json.dumps(dict(horizon, maximum_epochs=6)), encoding='utf-8')
            later[8] = dict(later[8], selected_epoch=6, selection_score=.8)
            later[7]['dev_predictions'][7] = deepcopy(later[5][7])
            later[7]['dev_metrics'] = manual_metrics(later[5], later[7]['dev_predictions'])
            later[9], later[10] = 'e' * 64, 'f' * 64
            split = {'train': [{'id': str(i)} for i in range(5)],
                     'dev': [{'id': row['record_id']} for row in later[5]]}
            self.assertEqual(validate_history(later[2], later[3], split, {'train': 9, 'dev': 20})['selected_epoch'], 6)
            self.assertFalse(validate_schedule_horizon(*later)['prefix_best_bound_to_final_checkpoint'])

    def test_schedule_horizon_rejects_missing_provenance_metric_winner_and_hash_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            args, horizon = self.schedule_horizon_fixture(path)
            target = path / 'schedule_horizon_evidence.json'
            for key, value in (('epoch', 2), ('optimizer_step', 8), ('code_sha256', 'e' * 64),
                    ('dataset_sha256', 'e' * 64), ('evaluator_version', 'wrong'),
                    ('initial_model_state_sha256', 'e' * 64), ('prefix_best_epoch', 2),
                    ('prefix_best_selection_score', .6), ('prefix_history_sha256', 'e' * 64),
                    ('training_record_order_sha256', 'e' * 64), ('current_model_state_sha256', 'malformed'),
                    ('prefix_best_checkpoint_sha256', 'e' * 64),
                    ('current_dev_metrics', dict(horizon['current_dev_metrics'], intent_accuracy=.6)),
                    ('prefix_best_dev_metrics', dict(horizon['prefix_best_dev_metrics'], slot_tp=8))):
                target.write_text(json.dumps(dict(horizon, **{key: value})), encoding='utf-8')
                with self.subTest(key=key), self.assertRaises(ValidationError):
                    validate_schedule_horizon(*args)
            target.unlink()
            with self.assertRaisesRegex(ValidationError, 'Missing or unsafe evidence'):
                validate_schedule_horizon(*args)

    def test_schedule_horizon_hashes_original_raw_prefix_and_rejects_unexpected_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            args, _ = self.schedule_horizon_fixture(path)
            history = path / 'history.jsonl'
            raw = history.read_bytes()
            history.write_bytes(raw.replace(b'\n', b' \n', 1))
            with self.assertRaisesRegex(ValidationError, 'history-prefix hash'):
                validate_schedule_horizon(*args)
            history.write_bytes(raw)
            no_horizon = list(args)
            no_horizon[3] = dict(no_horizon[3], schedule_epochs=10)
            with self.assertRaisesRegex(ValidationError, 'Unexpected schedule-horizon evidence'):
                validate_schedule_horizon(*no_horizon)
            (path / 'schedule_horizon_evidence.json').unlink()
            self.assertEqual(validate_schedule_horizon(*no_horizon)['status'], 'not_applicable')

    def test_history_csv_roundtrip_preserves_disabled_rng_null_and_active_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'history.csv'
            for state_hash in (None, 'a' * 64):
                row = {'epoch': 1, 'train_word_dropout_rng_state_sha256': state_hash,
                    'train_word_dropout_alpha': .25 if state_hash else 0.,
                    'selected': True, 'learning_rates': [.0001, .0005], 'train_num_tokens': 81050}
                # Match the real trainer: the same native values go to JSONL and DictWriter.
                history = [json.loads(json.dumps(row))]
                with path.open('w', encoding='utf-8', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)
                with self.subTest(state_hash=state_hash):
                    validate_history_csv(path, history)
                altered = dict(row, train_word_dropout_rng_state_sha256='None' if state_hash is None else 'b' * 64)
                with path.open('w', encoding='utf-8', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(altered)
                with self.subTest(tampered_state_hash=state_hash), self.assertRaisesRegex(ValidationError, 'histories differ'):
                    validate_history_csv(path, history)

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

    def regularization_fixture(self, alpha=.25, epsilon=.05):
        training = {'word_dropout_alpha': alpha, 'intent_label_smoothing': epsilon}
        runtime = {'seed': 42}
        split = {'train': [{'id': 'one', 'utt': 'a a b'}, {'id': 'two', 'utt': 'a'}]}
        mappings = {'word_to_id': {'<PAD>': 0, '<UNK>': 1, 'a': 2, 'b': 3}}
        counts = [0, 0, 3, 1]
        frequency = {'alpha': alpha, 'seed': 42, 'fit_partition': 'train',
            'training_record_ids_sha256': json_hash(['one', 'two']),
            'word_to_id_sha256': json_hash(mappings['word_to_id']), 'word_counts': counts,
            'word_counts_sha256': json_hash(counts), 'training_token_count': 4,
            'eligible_training_token_count': 4,
            'expected_mask_fraction': sum(n * alpha / (n + alpha) for n in counts[2:]) / 4} if alpha else None
        regularization = {'word_dropout_alpha': alpha, 'intent_label_smoothing': epsilon,
            'word_dropout': frequency, 'intent_smoothing_scope': 'training_only; dev/test unsmoothed'}
        records = [{'regularization': deepcopy(regularization), 'initial_model_state_sha256': 'a' * 64}
                   for _ in range(3)]
        records[-1]['word_dropout_rng_state'] = torch.Generator().manual_seed(42).get_state() if alpha else None
        history = [{'train_word_dropout_alpha': alpha, 'train_intent_label_smoothing': epsilon,
            'train_word_dropout_masked_tokens': int(bool(alpha)), 'train_word_dropout_eligible_tokens': 4,
            'train_record_order_sha256': json_hash(['two', 'one'])}]
        return (training, runtime, split, mappings, {'train': 4}, history, *records)

    def test_regularization_evidence_reconstructs_train_counts_and_accepts_legacy_disabled(self):
        for alpha, epsilon in ((.25, .05), (.25, 0.), (0., .05), (0., 0.)):
            with self.subTest(alpha=alpha, epsilon=epsilon):
                result = validate_regularization(*self.regularization_fixture(alpha, epsilon))
                self.assertEqual(result['evidence'], 'verified_train_only')
        args = list(self.regularization_fixture(0., 0.))
        args[0] = {}; args[5] = []; args[-3:] = [{}, {}, {}]
        self.assertEqual(validate_regularization(*args)['evidence'], 'legacy_disabled')

    def test_regularization_rejects_missing_or_transductive_frequency_evidence(self):
        args = list(self.regularization_fixture())
        args[-3:] = [{}, {}, {}]
        with self.assertRaisesRegex(ValidationError, 'lacks saved evidence'):
            validate_regularization(*args)
        for key, wrong in (('fit_partition', 'dev'), ('seed', 17), ('word_counts', [0, 0, 2, 2]),
                           ('training_record_ids_sha256', 'wrong'), ('expected_mask_fraction', .9)):
            args = list(self.regularization_fixture())
            for record in args[-3:]:
                record['regularization']['word_dropout'][key] = wrong
            with self.subTest(key=key), self.assertRaises(ValidationError):
                validate_regularization(*args)

    def test_regularization_rejects_invalid_epoch_masks_and_evaluation_regularization(self):
        for key, wrong in (('train_word_dropout_masked_tokens', 5), ('train_word_dropout_masked_tokens', True),
                           ('train_word_dropout_eligible_tokens', 3), ('train_record_order_sha256', 'missing'),
                           ('dev_word_dropout_masked_tokens', 1), ('dev_intent_label_smoothing', .05)):
            args = list(self.regularization_fixture())
            args[5][0][key] = wrong
            with self.subTest(key=key), self.assertRaises(ValidationError):
                validate_regularization(*args)
        args = list(self.regularization_fixture(0., .05))
        args[5][0]['train_word_dropout_masked_tokens'] = 1
        with self.assertRaisesRegex(ValidationError, 'Disabled word dropout'):
            validate_regularization(*args)


if __name__ == '__main__':
    unittest.main()

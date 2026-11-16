"""Check evidence integrity and paired statistics without model training/inference."""
from copy import deepcopy
import ast
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import analyze_transformer_regularization as analysis


def row(key, text, intent='travel', spans=()):
    return {'record_id': key, 'utt': text, 'intent': intent, 'slots': [
        {'slot_type': kind, 'text': text[start:end], 'start': start, 'end': end}
        for kind, start, end in spans]}


def fixture_data():
    training = [row('t1', 'new york', spans=[('place', 0, 8)]),
                row('t2', 'boston jersey', spans=[('place', 0, 6)])]
    gold = [row('d1', 'New York', spans=[('place', 0, 8)]),
            row('d2', 'new jersey', spans=[('place', 0, 10)]),
            row('d3', 'oslo', spans=[('place', 0, 4)]),
            row('d4', 'boston'), row('d5', 'zebra', intent='other')]
    return {'train': training, 'dev': gold}


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def write_run(path, data):
    """Synthetic bookkeeping fixture; its bytes are not a PyTorch checkpoint."""
    path.mkdir(parents=True)
    predictions = deepcopy(data['dev'])
    metrics = analysis.score_exact_spans(data['dev'], predictions)
    score = .5 * (metrics['intent_accuracy'] + metrics['slot_micro_f1'])
    training = {'max_epochs': 2, 'batch_size': 2, 'learning_rate': .001,
                'word_dropout_alpha': 0., 'intent_label_smoothing': 0.}
    history = [{'epoch': i, 'train_num_examples': len(data['train']),
        'dev_num_examples': len(data['dev']), 'optimizer_step': i,
        'dev_intent_accuracy': metrics['intent_accuracy'], 'dev_slot_micro_f1': metrics['slot_micro_f1'],
        'selection_score': score, 'train_seconds': 1., 'dev_seconds': .1,
        'train_record_order_sha256': f'epoch{i}'} for i in (1, 2)]
    for entry in history:
        for part, counts in analysis.dataset_accounting(data).items():
            entry.update({part+'_'+key: value for key, value in counts.items()})
            entry[part+'_num_batches'] = math.ceil(counts['num_examples']/training['batch_size'])
    checkpoint = path/'best.pt'
    checkpoint.write_bytes(b'synthetic-checkpoint-do-not-load')
    checksum = analysis.file_hash(checkpoint)
    (path/'best.pt.sha256').write_text(checksum, encoding='utf-8')
    counts = {'total': 123, 'trainable': 123}
    reload = {'passed': True, 'fresh_python_process': True, 'ordered_predictions_equal': True,
        'metrics': metrics, 'num_examples': len(data['dev']), 'test_evaluated': False,
        'checkpoint_sha256': checksum, 'dataset_sha256': analysis.DATASET_SHA256,
        'code_sha256': 'same-code'}
    summary = {'status': 'complete', 'mode': 'development_only', 'epochs_trained': 2,
        'selected_epoch': 1, 'selection_score': score,
        'dev': {'metrics': metrics, 'record_ids_sha256': analysis.json_hash([r['record_id'] for r in data['dev']])},
        'test': None, 'test_model_evaluations': 0, 'parameter_counts': counts,
        'checkpoint': 'best.pt', 'checkpoint_sha256': checksum, 'fresh_process_reload': reload,
        'model_weights_unchanged': True, 'checkpoint_bytes_unchanged': True,
        'dataset_sha256': analysis.DATASET_SHA256, 'evaluator_version': analysis.EVALUATOR_VERSION,
        'code_sha256': 'same-code'}
    run = {'runtime': {'seed': 42}, 'stage_config': {'model': {'width': 8}, 'training': training},
        'parameter_counts': counts, 'dataset_sha256': analysis.DATASET_SHA256,
        'evaluator_version': analysis.EVALUATOR_VERSION, 'code_sha256': 'same-code',
        'initial_model_state_sha256': 'same-initial-state'}
    write_json(path/'run.json', run)
    write_json(path/'final_summary.json', summary)
    write_json(path/'fresh_reload.json', reload)
    (path/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history), encoding='utf-8')
    (path/'dev_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions), encoding='utf-8')
    return summary, run


def write_notebook_snapshot(path):
    run = {'stage_config': {'model': {'kernels': [3, 4]}, 'training': {'max_epochs': 2}},
           'runtime': {'seed': 42}, 'run_options': {'evaluate_test': False}}
    configuration = {'STAGE2_CONFIG': deepcopy(run['stage_config']), 'STAGE1_CONFIG': {},
                     'RUNTIME': run['runtime'], 'RUN_OPTIONS': run['run_options']}
    configuration['STAGE2_CONFIG']['model']['kernels'] = (3, 4)
    sources = ['def marker():\n    return 1\n',
               '\n'.join(name+' = '+repr(value) for name, value in configuration.items())]
    digests = [hashlib.sha256(value.encode()).hexdigest() for value in sources]
    document = {'cells': [{'cell_type': 'code', 'source': value,
                 'metadata': {'tags': ['definitions'] if i == 0 else ['configuration']},
                 'outputs': [], 'execution_count': None} for i, value in enumerate(sources)],
        'metadata': {'revised_transformer': {'code_cell_sha256': digests,
            'configuration_cell_indices': [1], 'source_provenance': {
                'src/example.py': {'file_sha256': 'a'*64, 'inline_sha256': digests[0]}}}}}
    write_json(path, document)
    run['code_sha256'] = analysis.json_hash(digests)
    registration = {'code_sha256': run['code_sha256'], 'notebook_sha256_before_training': analysis.file_hash(path)}
    return document, run, registration, {'source_sha256': {'src/example.py': 'a'*64}}


def write_split_notebook_snapshot(path):
    document, run, registration, specification = write_notebook_snapshot(path)
    configuration = document['cells'][-1]
    modules = {
        'src/example.py': ["from __future__ import annotations\n\"\"\"Module documentation.\"\"\"\ndef marker():\n    return 1\n",
                           'from __future__ import annotations\ndef second():\n    return marker() + 1\n'],
        'src/other.py': ['def third():\n    return 3\n']}
    provenance, cells = {}, []
    for index, (name, fragments) in enumerate(modules.items()):
        nodes = [node for source in fragments for node in ast.parse(source).body
            if not (isinstance(node, ast.ImportFrom) and node.module == '__future__')
            and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                     and isinstance(node.value.value, str))]
        sha = lambda value: hashlib.sha256(value.encode()).hexdigest()
        provenance[name] = {'file_sha256': ('a' if index == 0 else 'b')*64,
            'inline_sha256': sha('\n\n'.join(fragments)), 'inline_cell_sha256': [sha(s) for s in fragments],
            'semantic_ast_sha256': sha(ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False))}
        cells.extend({'cell_type': 'code', 'source': source, 'metadata': {
            'tags': ['definitions'], 'source_module': name}, 'outputs': [], 'execution_count': None}
            for source in fragments)
    document['cells'] = cells + [configuration]
    document['metadata']['revised_transformer'].update(definition_layout_version=2, source_provenance=provenance)
    specification['source_sha256'] = {name: value['file_sha256'] for name, value in provenance.items()}
    refresh_notebook_registration(path, document, run, registration)
    return document, run, registration, specification


def refresh_notebook_registration(path, document, run, registration):
    """Re-sign fixture code so layout tests reach the independent provenance gate."""
    cells = [cell for cell in document['cells'] if cell['cell_type'] == 'code']
    metadata = document['metadata']['revised_transformer']
    metadata['code_cell_sha256'] = [hashlib.sha256(cell['source'].encode()).hexdigest() for cell in cells]
    metadata['configuration_cell_indices'] = [i for i, cell in enumerate(cells)
        if 'configuration' in cell.get('metadata', {}).get('tags', [])]
    run['code_sha256'] = registration['code_sha256'] = analysis.json_hash(metadata['code_cell_sha256'])
    write_json(path, document)
    registration['notebook_sha256_before_training'] = analysis.file_hash(path)


def confidence_fixture(root):
    """Two complete synthetic seed cohorts with independently hashed confidence evidence."""
    data = fixture_data()
    run_root, dataset = root/'experiment', root/'dataset.jsonl'
    dataset.write_text('synthetic dataset bytes', encoding='utf-8')
    for seed in (17, 42):
        for arm in analysis.ARMS:
            stage = run_root/f'seed_{seed}'/arm/'stage2'
            _, run = write_run(stage, data)
            run['runtime']['seed'] = seed
            run['stage_config']['training'].update(dict(zip(analysis.REGULARIZATION_KEYS,
                analysis.EXPECTED_SETTINGS[arm])))
            write_json(stage/'run.json', run)
    identity = {'dataset_sha256': analysis.DATASET_SHA256, 'evaluator_version': analysis.EVALUATOR_VERSION,
        'train_dev_ordered_id_sha256': {'dev': analysis.json_hash([r['record_id'] for r in data['dev']])},
        'counts': {'train': 2, 'dev': 5}, 'train_intent_count': 3}
    with patch.object(analysis, 'load_train_dev', return_value=(data, identity)):
        report = analysis.analyze(run_root, dataset, replicates=3)
    sources = []
    for name in ('src/example.py', 'scripts/analyze_transformer_regularization_confidence.py',
                 'scripts/analyze_transformer_regularization.py', 'scripts/validate_revised_transformer.py'):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# '+name, encoding='utf-8')
        sources.append(path)
    protocol = {'protocol': {'source_sha256': {'src/example.py': analysis.file_hash(sources[0])}}}
    write_json(run_root/'protocol.json', protocol)
    sources.extend([run_root/'protocol.json', dataset])
    for record in report['runs']:
        directory = run_root/'notebook_sources'/f"seed_{record['seed']}"/record['arm']
        directory.mkdir(parents=True)
        for name in ('registration.json', 'NLP_Final_project_revised_regularized.ipynb'):
            path = directory/name
            path.write_text('synthetic source bytes', encoding='utf-8')
            sources.append(path)
    report['registered_protocol'] = protocol
    report['confirmation'] = {'complete': True, 'registered': True, 'policy': 'Synthetic complete cohort',
        'screen_arms_improving_both': [], 'confirmation_only_summary': analysis.aggregate_matched_seeds(
            [row for row in report['runs'] if row['seed'] != 42])}
    rows = []
    for record in report['runs']:
        calibration = {'num_examples': 5, 'num_classes': 3, 'intent_errors': 0,
            'intent_accuracy': 1., 'negative_log_likelihood': .5 if record['seed'] == 42 else .7,
            'ece_15': .1, 'ece_num_bins': 15, 'multiclass_brier_score': .3, 'mean_confidence': .9,
            'mean_incorrect_confidence': None, 'incorrect_confidence_ge_0_90': 0,
            'incorrect_confidence_ge_0_99': 0, 'top2_accuracy': 1., 'top3_accuracy': 1.}
        rows.append({**{key: record[key] for key in ('seed','arm','run_path','selected_epoch',
            'parameter_counts','checkpoint_sha256','model_state_sha256','initial_model_state_sha256',
            'ordered_dev_predictions_sha256')}, 'calibration': calibration,
            'logits_shape': [5, 3], 'logits_sha256': 'a'*64,
            'saved_intent_and_slot_predictions_reproduced': True, 'saved_unsmoothed_intent_loss_reproduced': True,
            'model_state_unchanged': True, 'checkpoint_bytes_unchanged': True,
            'dev_model_evaluations': 1, 'test_model_evaluations': 0})
    confidence = {'schema_version': 1, 'status': 'complete', 'runs': rows,
        'dataset_sha256': analysis.DATASET_SHA256, 'evaluator_version': analysis.EVALUATOR_VERSION,
        'sources_unchanged': True, 'test_model_evaluations': 0, 'optimizer_created': False,
        'sources_before_and_after_sha256': {str(p.resolve()): analysis.file_hash(p) for p in sources},
        'completed_at_utc': '2026-11-11T00:00:00+00:00', 'device': 'cpu',
        'metric_definition': {'ece_15': '15 equal confidence bins'}, 'limitations': ['Synthetic test evidence']}
    path = root/'confidence.json'
    write_json(path, confidence)
    return report, path, confidence, run_root, dataset


def archive_fixture(root, name='scripts/example.py'):
    run_root = root/'study'
    archive = run_root/'registered_sources'
    original = b"raise AssertionError('Archived evidence must never execute')\n"
    digest = hashlib.sha256(original).hexdigest()
    current, saved = root/name, archive/name
    current.parent.mkdir(parents=True, exist_ok=True)
    saved.parent.mkdir(parents=True, exist_ok=True)
    current.write_text('# Current independently revised implementation\n', encoding='utf-8')
    saved.write_bytes(original)
    write_json(run_root/'protocol.json', {'protocol': {'source_sha256': {name: digest}}})
    manifest = {'protocol_sha256': analysis.file_hash(run_root/'protocol.json'), 'files': {name: digest}}
    write_json(archive/'manifest.json', manifest)
    return run_root, current, saved, digest, manifest


class RegularizationAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture_data()
        self.gold = self.data['dev']
        self.strata = analysis.fit_strata(self.data['train'], self.gold)
        self.identity = {'train_dev_ordered_id_sha256': {
            'dev': analysis.json_hash([r['record_id'] for r in self.gold])}}

    def test_original_dataset_is_pinned_without_reading_test_targets(self):
        # The synthetic test row deliberately has no utterance/intent/annotation.
        rows = [dict(id='t', partition='train', locale='en-US', utt='a', annot_utt='a', intent='x'),
                dict(id='d', partition='dev', locale='en-US', utt='b', annot_utt='b', intent='x'),
                dict(id='q', partition='test', locale='en-US')]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'data.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
            with patch.object(analysis, 'DATASET_SHA256', analysis.file_hash(path)), \
                    patch.object(analysis, 'EXPECTED_COUNTS', {'train': 1, 'dev': 1, 'test': 1}):
                data, identity = analysis.load_train_dev(path)
                self.assertEqual(set(data), {'train', 'dev'})
                self.assertEqual(identity['counts']['test'], 1)
                path.write_text(path.read_text()+'\n', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'pinned'):
                    analysis.load_train_dev(path)

    def test_protocol_bound_archive_resolves_bytes_without_executing_them(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(analysis, 'ROOT', Path(directory)):
            root = Path(directory)
            run_root, current, saved, digest, _ = archive_fixture(root)
            evidence = analysis.resolve_registered_source(run_root, 'scripts/example.py', digest)
            self.assertEqual(evidence['location'], 'archive')
            self.assertEqual(Path(evidence['resolved_path']), saved)
            self.assertEqual(evidence, analysis.resolve_recorded_source(run_root, current, digest))
            current.write_bytes(saved.read_bytes())
            self.assertEqual(analysis.resolve_registered_source(run_root, 'scripts/example.py', digest)['location'], 'current')

    def test_archive_rejects_wrong_protocol_hash_manifest_hash_and_file_bytes(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(analysis, 'ROOT', Path(directory)):
            run_root, _, saved, digest, manifest = archive_fixture(Path(directory))
            path = saved.parents[1]/'manifest.json'
            changed = deepcopy(manifest); changed['protocol_sha256'] = 'different'
            write_json(path, changed)
            with self.assertRaisesRegex(ValueError, 'original protocol'):
                analysis.resolve_registered_source(run_root, 'scripts/example.py', digest)
            changed = deepcopy(manifest); changed['files']['scripts/example.py'] = 'b'*64
            write_json(path, changed)
            with self.assertRaisesRegex(ValueError, 'exact expected source hash'):
                analysis.resolve_registered_source(run_root, 'scripts/example.py', digest)
            write_json(path, manifest)
            saved.write_text('# tampered archived source', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Archived source bytes'):
                analysis.resolve_registered_source(run_root, 'scripts/example.py', digest)

    def test_archive_does_not_allow_traversal_or_dataset_evidence_fallback(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(analysis, 'ROOT', Path(directory)):
            run_root, _, _, digest, _ = archive_fixture(Path(directory))
            for name in ('../scripts/example.py', 'data/dataset.jsonl', str(Path(directory)/'scripts/example.py')):
                with self.assertRaisesRegex(ValueError, 'Invalid registered source'):
                    analysis.resolve_registered_source(run_root, name, digest)
            with self.assertRaisesRegex(ValueError, 'stale or changed'):
                analysis.resolve_recorded_source(run_root, Path(directory)/'data/dataset.jsonl', digest)

    def test_historical_confidence_refuses_changed_runtime_before_inference(self):
        from scripts import analyze_transformer_regularization_confidence as confidence
        with tempfile.TemporaryDirectory() as directory, patch.object(analysis, 'ROOT', Path(directory)), \
                patch.object(confidence, 'ROOT', Path(directory)):
            run_root, _, _, _, _ = archive_fixture(Path(directory), 'src/revised_transformer_training.py')
            with patch.object(confidence, 'audit_checkpoint', side_effect=AssertionError('No inference allowed')):
                with self.assertRaisesRegex(ValueError, 'Cannot rerun historical confidence inference'):
                    confidence.assess(run_root, Path(directory)/'unused_dataset.jsonl', device='cpu')

    def test_train_fitted_novelty_separates_known_combination_from_oov(self):
        profiles = analysis.stratified_metrics(self.gold, self.gold, self.strata)
        self.assertEqual([v['gold_support'] for v in profiles['gold_typed_value_recall'].values()], [1, 1, 1])
        self.assertEqual(self.strata['utterance_groups']['slot_train_oov'], [2])
        self.assertEqual(self.strata['utterance_groups']['non_slot_train_oov'], [4])
        self.assertEqual(self.strata['utterance_groups']['no_gold_slots'], [3, 4])

    def test_exact_metrics_preserve_occurrences_joint_and_spurious_slots(self):
        predicted = deepcopy(self.gold)
        predicted[1]['slots'] = []
        predicted[3]['slots'] = deepcopy(self.data['train'][1]['slots'])
        predicted[4]['intent'] = 'travel'
        result = analysis.stratified_metrics(self.gold, predicted, self.strata)
        self.assertEqual((result['metrics']['slot_tp'], result['metrics']['slot_fp'], result['metrics']['slot_fn']), (2, 1, 1))
        self.assertEqual(result['metrics']['fully_correct_utterances'], 2)
        self.assertEqual(result['metrics']['intent_errors'], 1)
        self.assertEqual(result['utterance_groups']['no_gold_slots']['slot_fp'], 1)
        self.assertEqual(result['gold_typed_value_recall']['unseen_value_known_words']['correct'], 0)

    def test_paired_identical_predictions_have_zero_point_and_interval(self):
        result = analysis.paired_comparison(self.gold, self.gold, self.gold, replicates=123)
        for values in result['differences'].values():
            self.assertEqual(values['percentage_points'], 0.)
            self.assertEqual(values['nominal_95_percentile_interval_pp'], [0., 0.])
        self.assertFalse(result['both_required_metrics_improved'])

    def test_paired_bootstrap_is_reproducible_and_counts_regressions(self):
        before, after = deepcopy(self.gold), deepcopy(self.gold)
        before[0]['intent'] = 'wrong'
        before[1]['slots'] = []
        after[2]['slots'] = []
        after[4]['intent'] = 'wrong'
        result = analysis.paired_comparison(self.gold, before, after, replicates=143, seed=7)
        self.assertEqual(result, analysis.paired_comparison(self.gold, before, after, replicates=143, seed=7))
        self.assertEqual(result['intent_errors_corrected'], 1)
        self.assertEqual(result['intent_errors_introduced'], 1)
        self.assertEqual(result['gold_slot_occurrences_recovered'], 1)
        self.assertEqual(result['gold_slot_occurrences_lost'], 1)
        self.assertAlmostEqual(result['differences']['slot_micro_f1']['percentage_points'], 0.)

    def test_reordered_or_duplicate_prediction_rows_are_rejected(self):
        for values in (self.gold[::-1], self.gold+[self.gold[0]]):
            with self.assertRaisesRegex(ValueError, 'exactly once in order'):
                analysis.paired_comparison(self.gold, self.gold, values, replicates=5)

    def test_complete_bookkeeping_load_and_tampering_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'stage2'
            summary, _ = write_run(path, self.data)
            result, predictions, _ = analysis.load_run(path, 42, 'baseline', self.data, self.identity, self.strata)
            self.assertEqual(result['metrics']['slot_micro_f1'], 1.)
            self.assertEqual(result['epoch_train_record_order_sha256'], ['epoch1', 'epoch2'])
            self.assertEqual(predictions, self.gold)
            changed = deepcopy(summary)
            changed['dev']['metrics']['slot_tp'] = 2
            write_json(path/'final_summary.json', changed)
            with self.assertRaisesRegex(ValueError, 'strict recomputation'):
                analysis.load_run(path, 42, 'baseline', self.data, self.identity, self.strata)
            write_json(path/'final_summary.json', summary)
            (path/'best.pt').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'checkpoint bytes changed'):
                analysis.load_run(path, 42, 'baseline', self.data, self.identity, self.strata)

    def test_test_artifacts_and_late_tie_selection_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'stage2'
            summary, _ = write_run(path, self.data)
            summary['selected_epoch'] = 2
            write_json(path/'final_summary.json', summary)
            with self.assertRaisesRegex(ValueError, 'earliest maximum'):
                analysis.load_run(path, 42, 'baseline', self.data, self.identity, self.strata)
            (path/'test_predictions.jsonl').write_text('Never read these labels.', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Unexpected test artifacts'):
                analysis.load_run(path, 42, 'baseline', self.data, self.identity, self.strata)

    def test_epoch_membership_tokens_and_batches_are_checked_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'stage2'
            summary, run = write_run(path, self.data)
            history = analysis.read_jsonl(path/'history.jsonl')
            for key, value in [('train_record_ids_sha256', 'different-membership'),
                    ('dev_record_ids_sha256', 'different-order'), ('train_num_tokens', 999),
                    ('dev_num_tokens', 999), ('train_num_batches', 999), ('dev_num_batches', 999)]:
                with self.subTest(field=key):
                    changed = deepcopy(history)
                    changed[0][key] = value
                    with self.assertRaisesRegex(ValueError, 'History'):
                        analysis.validate_history(changed, summary, run['stage_config']['training'], self.data)

    def test_standalone_fresh_reload_must_equal_embedded_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'stage2'
            summary, _ = write_run(path, self.data)
            changed = deepcopy(summary['fresh_process_reload'])
            changed['checkpoint_sha256'] = 'changed'
            write_json(path/'fresh_reload.json', changed)
            with self.assertRaisesRegex(ValueError, 'Standalone fresh reload'):
                analysis.load_run(path, 42, 'baseline', self.data, self.identity, self.strata)

    def test_registered_notebook_checks_code_protected_cells_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'snapshot.ipynb'
            document, run, registration, protocol = write_notebook_snapshot(path)
            evidence = analysis.validate_notebook_snapshot(path, run, registration, protocol, check_original_bytes=True)
            self.assertTrue(evidence['protected_cells_verified'])
            changed = deepcopy(document)
            changed['cells'][0]['source'] += '\n# changed implementation\n'
            write_json(path, changed)
            with self.assertRaisesRegex(ValueError, 'Protected notebook'):
                analysis.validate_notebook_snapshot(path, run, registration, protocol, check_original_bytes=False)
            write_json(path, document)
            wrong_registration = {**registration, 'code_sha256': 'unregistered'}
            with self.assertRaisesRegex(ValueError, 'Actual notebook code identity'):
                analysis.validate_notebook_snapshot(path, run, wrong_registration, protocol, check_original_bytes=False)
            changed = deepcopy(document)
            changed['metadata']['revised_transformer']['source_provenance']['src/example.py']['file_sha256'] = 'b'*64
            write_json(path, changed)
            with self.assertRaisesRegex(ValueError, 'registered implementation sources'):
                analysis.validate_notebook_snapshot(path, run, registration, protocol, check_original_bytes=False)

    def test_notebook_outputs_can_change_but_script_snapshot_bytes_cannot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'snapshot.ipynb'
            document, run, registration, protocol = write_notebook_snapshot(path)
            document['cells'][0]['outputs'] = [{'output_type': 'stream', 'text': 'Genuine output\n'}]
            document['cells'][0]['execution_count'] = 1
            write_json(path, document)
            evidence = analysis.validate_notebook_snapshot(path, run, registration, protocol, check_original_bytes=False)
            self.assertEqual(evidence['code_sha256'], registration['code_sha256'])
            with self.assertRaisesRegex(ValueError, 'snapshot bytes changed'):
                analysis.validate_notebook_snapshot(path, run, registration, protocol, check_original_bytes=True)
            run['stage_config']['training']['max_epochs'] = 3
            with self.assertRaisesRegex(ValueError, 'recorded training run'):
                analysis.validate_notebook_snapshot(path, run, registration, protocol, check_original_bytes=False)

    def test_split_notebook_fragments_bind_order_combined_text_and_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'snapshot.ipynb'
            document, run, registration, protocol = write_split_notebook_snapshot(path)
            result = analysis.validate_notebook_snapshot(path, run, registration, protocol, check_original_bytes=True)
            self.assertTrue(result['source_provenance_verified'])
            for key, value in [('inline_cell_sha256', ['0'*64, '1'*64]),
                               ('inline_sha256', '0'*64), ('semantic_ast_sha256', '0'*64),
                               ('file_sha256', '0'*64)]:
                changed = deepcopy(document)
                changed['metadata']['revised_transformer']['source_provenance']['src/example.py'][key] = value
                refresh_notebook_registration(path, changed, run, registration)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    analysis.validate_notebook_snapshot(path, run, registration, protocol, check_original_bytes=True)

    def test_split_notebook_rejects_missing_extra_and_duplicate_origins_or_fragments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'snapshot.ipynb'
            original, run, registration, protocol = write_split_notebook_snapshot(path)
            variants = []
            changed = deepcopy(original); del changed['cells'][0]['metadata']['source_module']; variants.append(changed)
            changed = deepcopy(original); changed['cells'][0]['metadata']['source_module'] = 'src/unknown.py'; variants.append(changed)
            changed = deepcopy(original); del changed['cells'][2]; variants.append(changed)
            changed = deepcopy(original); changed['cells'].insert(1, deepcopy(changed['cells'][0])); variants.append(changed)
            changed = deepcopy(original); changed['cells'][1], changed['cells'][2] = changed['cells'][2], changed['cells'][1]; variants.append(changed)
            changed = deepcopy(original); changed['cells'][-1]['metadata']['source_module'] = 'src/example.py'; variants.append(changed)
            for index, changed in enumerate(variants):
                refresh_notebook_registration(path, changed, run, registration)
                with self.subTest(case=index), self.assertRaises(ValueError):
                    analysis.validate_notebook_snapshot(path, run, registration, protocol, check_original_bytes=True)

    def test_split_notebook_rejects_reordered_fragments_and_relabelled_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'snapshot.ipynb'
            original, run, registration, protocol = write_split_notebook_snapshot(path)
            changed = deepcopy(original)
            changed['cells'][0], changed['cells'][1] = changed['cells'][1], changed['cells'][0]
            refresh_notebook_registration(path, changed, run, registration)
            with self.assertRaisesRegex(ValueError, 'fragment order'):
                analysis.validate_notebook_snapshot(path, run, registration, protocol, check_original_bytes=True)
            for version in (1, 3, True):
                changed = deepcopy(original)
                changed['metadata']['revised_transformer']['definition_layout_version'] = version
                refresh_notebook_registration(path, changed, run, registration)
                with self.subTest(version=version), self.assertRaises(ValueError):
                    analysis.validate_notebook_snapshot(path, run, registration, protocol, check_original_bytes=True)

    def test_full_notebook_discovery_and_duplicate_run_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            arm = Path(directory)/'combined'
            path = arm/'transformer_run_artifacts'/'stage2'
            write_run(path, self.data)
            self.assertEqual(analysis.discover_stage_path(arm), path)
            write_run(arm/'stage2', self.data)
            with self.assertRaisesRegex(ValueError, 'Multiple completed'):
                analysis.discover_stage_path(arm)

    def test_matched_seed_summary_excludes_incomplete_seeds(self):
        runs = []
        for seed in (42, 17):
            for arm in analysis.ARMS:
                runs.append({'seed': seed, 'arm': arm, 'metrics': dict(zip(analysis.METRIC_NAMES,
                    [.8 + (arm == 'combined')*.01 + (seed == 17)*.02, .7, .6]))})
        runs.append({'seed': 101, 'arm': 'combined', 'metrics': dict.fromkeys(analysis.METRIC_NAMES, 1.)})
        result = analysis.aggregate_matched_seeds(runs)
        self.assertEqual(result['complete_four_arm_seeds'], [17, 42])
        self.assertAlmostEqual(result['arms']['combined']['metrics']['intent_accuracy']['mean'], .82)
        self.assertEqual(result['arms']['combined']['seeds_improving_both_required_metrics'], [])
        self.assertAlmostEqual(result['arms']['combined']['paired_seed_differences_pp']['intent_accuracy']['mean'], 1.)
        with self.assertRaisesRegex(ValueError, 'Duplicated arm'):
            analysis.aggregate_matched_seeds(runs+[runs[0]])

    def test_matched_protocol_rejects_learning_rate_or_order_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'stage2'
            write_run(path, self.data)
            baseline, _, _ = analysis.load_run(path, 42, 'baseline', self.data, self.identity, self.strata)
            candidate = deepcopy(baseline)
            candidate['training_configuration']['word_dropout_alpha'] = .25
            analysis.validate_matched_protocol(baseline, candidate)
            candidate['training_configuration']['learning_rate'] = .1
            with self.assertRaisesRegex(ValueError, 'declared regularizers'):
                analysis.validate_matched_protocol(baseline, candidate)
            candidate = deepcopy(baseline)
            candidate['epoch_train_record_order_sha256'].reverse()
            with self.assertRaisesRegex(ValueError, 'record orders'):
                analysis.validate_matched_protocol(baseline, candidate)

    def test_full_aggregation_and_outputs_from_saved_arm_fixtures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm in analysis.ARMS:
                stage = root/'seed_42'/arm/'stage2'
                _, run = write_run(stage, self.data)
                run['stage_config']['training'].update(dict(zip(analysis.REGULARIZATION_KEYS,
                    analysis.EXPECTED_SETTINGS[arm])))
                write_json(stage/'run.json', run)
            with patch.object(analysis, 'load_train_dev', return_value=(self.data, {
                    **self.identity, 'dataset_sha256': analysis.DATASET_SHA256,
                    'evaluator_version': analysis.EVALUATOR_VERSION, 'counts': {'train': 2, 'dev': 5}})):
                report = analysis.analyze(root, root/'unused.jsonl', replicates=101)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(len(report['runs']), 4)
            self.assertEqual(len(report['per_seed_paired_development_comparisons']), 3)
            self.assertFalse(report['test_evaluation_performed'])
            self.assertNotIn('confidence_analysis', report)
            analysis.write_outputs(report, root/'output', root/'report.md')
            self.assertEqual(json.loads((root/'output/results.json').read_text())['status'], 'complete')
            self.assertEqual(len((root/'output/results.csv').read_text().splitlines()), 5)
            self.assertIn('Complete four-arm seed sets: **[42]**.', (root/'report.md').read_text(encoding='utf-8'))
            self.assertIn('No new test result is claimed here', (root/'report.md').read_text(encoding='utf-8'))

    def test_optional_confidence_merges_exact_cohort_and_formats_mean_sd(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(analysis, 'ROOT', Path(directory)):
            report, path, confidence, run_root, dataset = confidence_fixture(Path(directory))
            analysis.attach_confidence(report, path, run_root, dataset)
            merged = report['confidence_analysis']
            self.assertFalse(merged['merge_performed_inference'])
            self.assertEqual(merged['source_sha256'], analysis.file_hash(path))
            nll = merged['across_seed_summary']['baseline']['metrics']['negative_log_likelihood']
            self.assertEqual(nll['n'], 2)
            self.assertAlmostEqual(nll['mean'], .6)
            self.assertAlmostEqual(nll['sample_standard_deviation'], math.sqrt(.02))
            self.assertEqual(merged['across_seed_summary']['baseline']['metrics']['mean_incorrect_confidence']['n'], 0)
            markdown = analysis.markdown_report(report)
            self.assertIn('0.6000 ± 0.1414', markdown)
            self.assertIn('Intent calibration and confidence', markdown)
            self.assertIn('saved development confidence evidence', markdown)

    def test_optional_confidence_missing_file_and_partial_report_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(analysis, 'ROOT', Path(directory)):
            report, path, _, run_root, dataset = confidence_fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, 'does not exist'):
                analysis.attach_confidence(report, path.with_name('missing.json'), run_root, dataset)
            report['status'] = 'partial'
            with self.assertRaisesRegex(ValueError, 'complete registered comparison'):
                analysis.attach_confidence(report, path, run_root, dataset)

    def test_optional_confidence_rejects_other_runs_predictions_and_dataset(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(analysis, 'ROOT', Path(directory)):
            report, path, confidence, run_root, dataset = confidence_fixture(Path(directory))
            cases = []
            changed = deepcopy(confidence); changed['runs'].pop(); cases.append(changed)
            changed = deepcopy(confidence); changed['runs'].append(deepcopy(changed['runs'][0])); cases.append(changed)
            changed = deepcopy(confidence); changed['runs'][0]['seed'] = 101; cases.append(changed)
            changed = deepcopy(confidence); changed['dataset_sha256'] = 'different'; cases.append(changed)
            changed = deepcopy(confidence); changed['runs'][0]['checkpoint_sha256'] = 'different'; cases.append(changed)
            changed = deepcopy(confidence); changed['runs'][0]['ordered_dev_predictions_sha256'] = 'different'; cases.append(changed)
            changed = deepcopy(confidence); changed['runs'][0]['calibration']['intent_errors'] = 1; cases.append(changed)
            for changed in cases:
                write_json(path, changed)
                with self.assertRaises(ValueError):
                    analysis.attach_confidence(deepcopy(report), path, run_root, dataset)

    def test_optional_confidence_rejects_incomplete_and_stale_source_provenance(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(analysis, 'ROOT', Path(directory)):
            report, path, confidence, run_root, dataset = confidence_fixture(Path(directory))
            changed = deepcopy(confidence)
            changed['sources_before_and_after_sha256'].pop(str(dataset.resolve()))
            write_json(path, changed)
            with self.assertRaisesRegex(ValueError, 'source provenance is incomplete'):
                analysis.attach_confidence(report, path, run_root, dataset)
            write_json(path, confidence)
            (run_root/'protocol.json').write_text('changed study registration', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'stale or changed'):
                analysis.attach_confidence(report, path, run_root, dataset)


if __name__ == '__main__':
    unittest.main()

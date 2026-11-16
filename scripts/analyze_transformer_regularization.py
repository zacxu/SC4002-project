"""Analyze completed regularization arms from saved development evidence only.

No model is constructed, trained, or evaluated. The original file is hash-checked,
but only train/development annotations are parsed. Test predictions are never read.
The report keeps per-seed paired comparisons separate from across-seed variation.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.comparison_contract import (DATASET_SHA256, EVALUATOR_VERSION, EXPECTED_COUNTS,
    SOURCE_URL, parse_annotated_spans, score_exact_spans)
from src.revised_transformer_data import transformer_tokens

ARMS = ('baseline', 'word_dropout', 'label_smoothing', 'combined')
REGULARIZATION_KEYS = ('word_dropout_alpha', 'intent_label_smoothing')
EXPECTED_SETTINGS = {'baseline': (0., 0.), 'word_dropout': (.25, 0.),
                     'label_smoothing': (0., .05), 'combined': (.25, .05)}
METRIC_NAMES = ('intent_accuracy', 'slot_micro_f1', 'fully_correct_utterance_rate')
CALIBRATION_METRICS = ('negative_log_likelihood', 'ece_15', 'multiclass_brier_score',
    'mean_confidence', 'mean_incorrect_confidence', 'incorrect_confidence_ge_0_90',
    'incorrect_confidence_ge_0_99', 'top2_accuracy', 'top3_accuracy')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve_registered_source(run_root, name, expected_sha256):
    """Verify source evidence in place or in its protocol-bound archive; never execute it."""
    relative = Path(name)
    require(not relative.is_absolute() and relative.parts and relative.parts[0] in ('src', 'scripts')
            and relative.suffix == '.py' and '..' not in relative.parts
            and re.fullmatch('[0-9a-f]{64}', expected_sha256) is not None,
            'Invalid registered source name/hash')
    current = ROOT/relative
    require(current.resolve().is_relative_to(ROOT.resolve()), 'Registered source escapes repository')
    if current.is_file() and not current.is_symlink() and file_hash(current) == expected_sha256:
        return {'logical_path': relative.as_posix(), 'location': 'current',
                'resolved_path': str(current.resolve()), 'sha256': expected_sha256}
    root = Path(run_root).resolve()
    archive = root/'registered_sources'
    require(archive.is_dir() and not archive.is_symlink() and archive.resolve().is_relative_to(root),
            'Current source differs and verified archive is unavailable: '+name)
    manifest_path = archive/'manifest.json'
    require(manifest_path.is_file() and not manifest_path.is_symlink(), 'Missing regular source archive manifest')
    manifest = read_json(manifest_path)
    require(manifest.get('protocol_sha256') == file_hash(root/'protocol.json'),
            'Source archive is not bound to this original protocol')
    require(manifest.get('files', {}).get(relative.as_posix()) == expected_sha256,
            'Archive manifest does not register the exact expected source hash')
    archived = archive/relative
    require(archived.resolve().is_relative_to(archive.resolve()) and archived.is_file()
            and not archived.is_symlink() and file_hash(archived) == expected_sha256,
            'Archived source bytes are missing, escaped, or changed')
    return {'logical_path': relative.as_posix(), 'location': 'archive',
            'resolved_path': str(archived.resolve()), 'sha256': expected_sha256,
            'archive_manifest_sha256': file_hash(manifest_path), 'archive_protocol_sha256': manifest['protocol_sha256']}


def resolve_recorded_source(run_root, path, expected_sha256):
    """Archive fallback is permitted only for recorded Python implementation sources."""
    path = Path(path).resolve()
    if path.is_file() and file_hash(path) == expected_sha256:
        return {'location': 'current', 'resolved_path': str(path), 'sha256': expected_sha256}
    require(path.is_relative_to(ROOT.resolve()), 'Recorded evidence changed outside this repository')
    relative = path.relative_to(ROOT.resolve())
    require(relative.parts[0] in ('src', 'scripts') and relative.suffix == '.py',
            'Confidence source/protocol/dataset snapshots are stale or changed')
    return resolve_registered_source(run_root, relative.as_posix(), expected_sha256)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    require(bool(lines) and all(line.strip() for line in lines), 'Empty/blank JSONL evidence: ' + str(path))
    return [json.loads(line) for line in lines]


def slot_keys(row):
    return {(s['slot_type'], s['start'], s['end'], s['text']) for s in row['slots']}


def discover_stage_path(arm_path):
    """Accept script and full-notebook outputs; never choose between duplicate runs."""
    candidates = [Path(arm_path)/'stage2', Path(arm_path)/'transformer_run_artifacts'/'stage2']
    completed = []
    for path in candidates:
        summaries = [p for p in (path/'final_summary.json', path/'summary.json') if p.is_file()]
        if summaries and read_json(summaries[0]).get('status') == 'complete':
            completed.append(path)
    require(len(completed) <= 1, 'Multiple completed stage2 locations exist for ' + str(arm_path))
    return completed[0] if completed else None


def dataset_accounting(data):
    return {part: {'num_examples': len(rows),
        'num_tokens': sum(len(transformer_tokens(row['utt'])[0]) for row in rows),
        'record_ids_sha256': json_hash(sorted(row['record_id'] for row in rows)
                                      if part == 'train' else [row['record_id'] for row in rows])}
        for part, rows in data.items()}


def validate_definition_layout(cells, metadata, expected_sources):
    """Bind every inline fragment to its declared module without executing code."""
    provenance = metadata['source_provenance']
    require(set(provenance) == set(expected_sources), 'Notebook source-provenance membership differs')
    version = metadata.get('definition_layout_version', 1)
    require(type(version) is int and version in (1, 2), 'Unsupported notebook definition layout')
    definitions = [cell for cell in cells if 'definitions' in cell.get('metadata', {}).get('tags', [])]
    source = lambda cell: cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    digest = lambda value: hashlib.sha256(value.encode('utf-8')).hexdigest()
    if version == 1:
        require(Counter(digest(source(cell)) for cell in definitions)
                == Counter(value['inline_sha256'] for value in provenance.values()),
                'Actual inline definition cells differ from source provenance')
        return
    groups, previous = {}, None
    for cell in cells:
        details = cell.get('metadata', {})
        if 'definitions' not in details.get('tags', []):
            require('source_module' not in details, 'Module origin on a non-definition cell')
            continue
        name = details.get('source_module')
        require(isinstance(name, str) and name in provenance, 'Missing or unexpected definition module origin')
        if name != previous:
            require(name not in groups, 'Duplicate noncontiguous definition module origin')
            groups[name] = []
            previous = name
        groups[name].append(source(cell))
    require(set(groups) == set(provenance), 'Missing inline definition module')
    for name, fragments in groups.items():
        hashes = provenance[name]
        actual = [digest(value) for value in fragments]
        require(len(set(actual)) == len(actual), 'Duplicate inline definition fragment')
        require(hashes.get('inline_cell_sha256') == actual, 'Inline definition fragment order or hashes differ')
        require(hashes.get('inline_sha256') == digest('\n\n'.join(fragments)),
                'Combined inline definition hash differs')
        body = [node for value in fragments for node in ast.parse(value).body
                if not (isinstance(node, ast.ImportFrom) and node.module == '__future__')
                and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                         and isinstance(node.value.value, str))]
        semantic = digest(ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False))
        require(hashes.get('semantic_ast_sha256') == semantic, 'Inline definition semantic AST hash differs')
        if name == 'src/comparison_contract.py':
            require(len(fragments) == 1 and actual[0] == hashes['file_sha256'],
                    'Canonical definition contract must remain one unchanged cell')


def validate_notebook_snapshot(path, run, registration, specification, *, check_original_bytes):
    """Read and statically inspect the registered notebook; never execute its code."""
    document = read_json(path)
    cells = [cell for cell in document['cells'] if cell['cell_type'] == 'code']
    metadata = document['metadata']['revised_transformer']
    expected, registered = metadata['code_cell_sha256'], metadata['configuration_cell_indices']
    tagged = [i for i, cell in enumerate(cells) if 'configuration' in cell.get('metadata', {}).get('tags', [])]
    require(len(expected) == len(cells) and len(registered) == 1 and registered == tagged,
            'Notebook protected-cell/configuration manifest differs')
    actual, configuration = [], {}
    for i, cell in enumerate(cells):
        source = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
        tree = ast.parse(source)
        digest = hashlib.sha256(source.encode('utf-8')).hexdigest()
        actual.append(digest)
        if i not in registered:
            require(digest == expected[i], 'Protected notebook implementation changed')
            continue
        for statement in tree.body:
            require(isinstance(statement, ast.Assign) and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name), 'Notebook configuration is not literal assignments')
            name = statement.targets[0].id
            require(name in {'RUNTIME', 'STAGE1_CONFIG', 'STAGE2_CONFIG', 'RUN_OPTIONS'}
                    and name not in configuration, 'Unexpected/repeated notebook configuration')
            configuration[name] = ast.literal_eval(statement.value)
            require(isinstance(configuration[name], dict), 'Notebook configuration must be dictionaries')
    require(set(configuration) == {'RUNTIME', 'STAGE1_CONFIG', 'STAGE2_CONFIG', 'RUN_OPTIONS'},
            'Incomplete notebook configuration')
    require(json_hash(actual) == registration['code_sha256'] == run['code_sha256'],
            'Actual notebook code identity differs from registered/trained code')
    # Notebook literals may use tuples, which the JSON run record stores as lists.
    require(json_hash(configuration['STAGE2_CONFIG']) == json_hash(run['stage_config'])
            and json_hash(configuration['RUNTIME']) == json_hash(run['runtime'])
            and json_hash(configuration['RUN_OPTIONS']) == json_hash(run['run_options']),
            'Actual notebook configuration differs from the recorded training run')
    provenance = metadata['source_provenance']
    expected_sources = {name for name in specification['source_sha256'] if name.startswith('src/')}
    validate_definition_layout(cells, metadata, expected_sources)
    for name, hashes in provenance.items():
        require(hashes['file_sha256'] == specification['source_sha256'][name],
                'Notebook provenance differs from registered implementation sources')
    if check_original_bytes:
        require(file_hash(path) == registration['notebook_sha256_before_training'],
                'Unexecuted notebook snapshot bytes changed after registration')
    return {'path': str(path), 'code_sha256': json_hash(actual), 'current_file_sha256': file_hash(path),
            'protected_cells_verified': True, 'source_provenance_verified': True,
            'registered_original_bytes_verified': check_original_bytes}


def validate_registered_notebook(root, record, specification):
    directory = Path(root)/'notebook_sources'/f"seed_{record['seed']}"/record['arm']
    registration = read_json(directory/'registration.json')
    require(registration['seed'] == record['seed'] and registration['arm'] == record['arm'],
            'Notebook registration arm/seed differs')
    path = directory/'NLP_Final_project_revised_regularized.ipynb'
    run_path = Path(record['run_path'])
    run = read_json(run_path/'run.json')
    full_notebook = run_path.parent.name == 'transformer_run_artifacts'
    evidence = {'registration_sha256': file_hash(directory/'registration.json'),
        'source_snapshot': validate_notebook_snapshot(path, run, registration, specification,
                                                       check_original_bytes=not full_notebook)}
    if full_notebook:
        executed_path = run_path.parent.parent/path.name
        evidence['executed_snapshot'] = validate_notebook_snapshot(executed_path, run, registration,
                                                    specification, check_original_bytes=False)
    return evidence


def load_train_dev(path):
    """Pin all raw bytes; never access test intent or annotation targets."""
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Expected original regular dataset file')
    require(file_hash(path) == DATASET_SHA256, 'Dataset differs from pinned original MASSIVE English')
    rows = {'train': [], 'dev': []}
    counts = Counter()
    ids = set()
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        part, key = row['partition'], row['id']
        require(part in EXPECTED_COUNTS and row['locale'] == 'en-US' and key not in ids,
                'Original partition/locale/ID contract differs')
        counts[part] += 1
        ids.add(key)
        if part in rows:
            rows[part].append({'record_id': key, 'utt': row['utt'], 'intent': row['intent'],
                              'slots': parse_annotated_spans(row['utt'], row['annot_utt'])})
    require(dict(counts) == EXPECTED_COUNTS, 'Original split counts differ')
    return rows, {'dataset_sha256': DATASET_SHA256, 'evaluator_version': EVALUATOR_VERSION,
        'source_url': SOURCE_URL, 'counts': dict(counts),
        'train_dev_ordered_id_sha256': {part: json_hash([r['record_id'] for r in values])
                                      for part, values in rows.items()},
        'train_dev_accounting': dataset_accounting(rows),
        'train_intent_count': len({row['intent'] for row in rows['train']}),
        'label_scope': 'Only training/development labels parsed; test labels and predictions are not analyzed'}


def validate_predictions(gold, predictions):
    require([r['record_id'] for r in predictions] == [r['record_id'] for r in gold],
            'Predictions must cover every original development record exactly once in order')
    return score_exact_spans(gold, predictions)


def contribution_array(gold, predictions):
    validate_predictions(gold, predictions)
    rows = []
    for target, predicted in zip(gold, predictions):
        g, p = slot_keys(target), slot_keys(predicted)
        correct = target['intent'] == predicted['intent']
        rows.append([int(correct), len(g & p), len(p - g), len(g - p), int(correct and g == p)])
    return np.asarray(rows, dtype=np.int64)


def metrics_from_counts(counts, n):
    den = 2 * counts[..., 1] + counts[..., 2] + counts[..., 3]
    f1 = np.divide(2 * counts[..., 1], den, out=np.zeros_like(den, dtype=float), where=den != 0)
    return np.stack([counts[..., 0] / n, f1, counts[..., 4] / n], axis=-1)


def paired_comparison(gold, baseline, candidate, *, replicates=10000, seed=20261111):
    require(type(replicates) is int and replicates >= 1, 'Bootstrap replicates must be positive')
    first, second = contribution_array(gold, baseline), contribution_array(gold, candidate)
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, replicates, 100):
        indices = rng.integers(0, len(gold), size=(min(100, replicates - start), len(gold)))
        samples.append(100 * (metrics_from_counts(second[indices].sum(axis=1), len(gold))
                             - metrics_from_counts(first[indices].sum(axis=1), len(gold))))
    samples = np.concatenate(samples)
    point = 100 * (metrics_from_counts(second.sum(axis=0), len(gold))
                   - metrics_from_counts(first.sum(axis=0), len(gold)))
    corrected = int(((first[:, 0] == 0) & (second[:, 0] == 1)).sum())
    regressed = int(((first[:, 0] == 1) & (second[:, 0] == 0)).sum())
    recovered = lost = 0
    for g, a, b in zip(gold, baseline, candidate):
        target, before, after = slot_keys(g), slot_keys(a), slot_keys(b)
        recovered += len((target & after) - before)
        lost += len((target & before) - after)
    return {'differences': {name: {'percentage_points': float(point[i]),
        'nominal_95_percentile_interval_pp': [float(v) for v in np.percentile(samples[:, i], [2.5, 97.5])]}
        for i, name in enumerate(METRIC_NAMES)},
        'intent_errors_corrected': corrected, 'intent_errors_introduced': regressed,
        'gold_slot_occurrences_recovered': recovered, 'gold_slot_occurrences_lost': lost,
        'both_required_metrics_improved': bool(point[0] > 0 and point[1] > 0),
        'bootstrap': {'replicates': replicates, 'seed': seed, 'unit': 'paired whole development utterances',
            'caution': 'Conditional on selected checkpoints on their selection split; excludes seed variation, selection bias and multiplicity adjustment'}}


def fit_strata(training, development):
    words = {t.casefold() for row in training for t in transformer_tokens(row['utt'])[0]}
    values = {(s['slot_type'], s['text'].casefold()) for row in training for s in row['slots']}
    gold_groups = {'seen_typed_value': set(), 'unseen_value_known_words': set(), 'unseen_value_oov_words': set()}
    utterance_groups = {'no_train_oov': [], 'non_slot_train_oov': [], 'slot_train_oov': [], 'no_gold_slots': []}
    for i, row in enumerate(development):
        tokens, offsets = transformer_tokens(row['utt'])
        unknown = [offset for token, offset in zip(tokens, offsets) if token.casefold() not in words]
        overlap = any(a < slot['end'] and slot['start'] < b for a, b in unknown for slot in row['slots'])
        utterance_groups['slot_train_oov' if overlap else 'non_slot_train_oov' if unknown else 'no_train_oov'].append(i)
        if not row['slots']:
            utterance_groups['no_gold_slots'].append(i)
        for slot in row['slots']:
            key = (row['record_id'], *next(iter(slot_keys({'slots': [slot]}))))
            seen = (slot['slot_type'], slot['text'].casefold()) in values
            unseen_word = any(token.casefold() not in words for token in transformer_tokens(slot['text'])[0])
            group = 'seen_typed_value' if seen else 'unseen_value_oov_words' if unseen_word else 'unseen_value_known_words'
            gold_groups[group].add(key)
    return {'words': words, 'typed_values': values, 'gold_groups': gold_groups,
            'utterance_groups': utterance_groups}


def stratified_metrics(gold, predictions, strata):
    metrics = validate_predictions(gold, predictions)
    tp, fp, fn = (metrics[k] for k in ('slot_tp', 'slot_fp', 'slot_fn'))
    metrics.update(slot_precision=tp/(tp+fp) if tp+fp else 0.,
        slot_recall=tp/(tp+fn) if tp+fn else 0.,
        intent_errors=len(gold)-metrics['correct_intents'])
    joint = sum(g['intent'] == p['intent'] and slot_keys(g) == slot_keys(p) for g, p in zip(gold, predictions))
    metrics['fully_correct_utterances'] = joint
    metrics['fully_correct_utterance_rate'] = joint / len(gold)
    correct_slots = {(g['record_id'], *slot) for g, p in zip(gold, predictions)
                     for slot in slot_keys(g) & slot_keys(p)}
    value_recall = {name: {'gold_support': len(keys), 'correct': len(keys & correct_slots),
                         'recall': len(keys & correct_slots)/len(keys) if keys else 0.}
                    for name, keys in strata['gold_groups'].items()}
    predicted_values = {'seen_typed_value': Counter(), 'unseen_typed_value': Counter()}
    for g, p in zip(gold, predictions):
        target = slot_keys(g)
        for slot in p['slots']:
            group = 'seen_typed_value' if (slot['slot_type'], slot['text'].casefold()) in strata['typed_values'] else 'unseen_typed_value'
            predicted_values[group]['predicted_support'] += 1
            predicted_values[group]['correct'] += next(iter(slot_keys({'slots': [slot]}))) in target
    value_precision = {name: {'predicted_support': c['predicted_support'], 'correct': c['correct'],
        'precision': c['correct']/c['predicted_support'] if c['predicted_support'] else 0.}
        for name, c in predicted_values.items()}
    groups = {}
    for name, indices in strata['utterance_groups'].items():
        if not indices:
            groups[name] = {'num_examples': 0}
            continue
        selected_gold, selected_pred = [gold[i] for i in indices], [predictions[i] for i in indices]
        groups[name] = score_exact_spans(selected_gold, selected_pred)
        groups[name]['utterances_with_any_predicted_slot'] = sum(bool(p['slots']) for p in selected_pred)
    return {'metrics': metrics, 'gold_typed_value_recall': value_recall,
            'predicted_typed_value_precision': value_precision, 'utterance_groups': groups}


def validate_history(history, summary, config, data):
    accounting = dataset_accounting(data)
    train_count = accounting['train']['num_examples']
    epochs = config['max_epochs']
    require([r['epoch'] for r in history] == list(range(1, epochs + 1)), 'Incomplete/duplicated epoch history')
    require(summary['epochs_trained'] == epochs, 'Summary training budget differs')
    for i, row in enumerate(history, 1):
        for part, counts in accounting.items():
            require(all(row[part+'_'+key] == value for key, value in counts.items()),
                    'History original membership/example/token accounting differs for '+part)
            require(row[part+'_num_batches'] == math.ceil(counts['num_examples']/config['batch_size']),
                    'History batch count differs for '+part)
            if part == 'dev' and part+'_record_order_sha256' in row:
                require(row[part+'_record_order_sha256'] == counts['record_ids_sha256'],
                        'History original development order differs')
        require(row['optimizer_step'] == i * math.ceil(train_count / config['batch_size']),
                'Optimizer step count differs from full registered budget')
        for name in ('dev_intent_accuracy', 'dev_slot_micro_f1', 'selection_score', 'train_seconds', 'dev_seconds'):
            require(math.isfinite(row[name]), 'Nonfinite history value')
        require(abs(row['selection_score'] - .5*(row['dev_intent_accuracy'] + row['dev_slot_micro_f1'])) <= 1e-12,
                'Selection rule differs from the declared mean development score')
    winner = max(history, key=lambda r: r['selection_score'])
    require(summary['selected_epoch'] == winner['epoch'] and abs(summary['selection_score']-winner['selection_score']) <= 1e-12,
            'Checkpoint is not the earliest maximum development score')
    for field in ('intent_accuracy', 'slot_micro_f1'):
        require(abs(summary['dev']['metrics'][field] - winner['dev_'+field]) <= 1e-12,
                'Final development score differs from selected history epoch')


def load_run(path, seed, arm, data, identity, strata):
    path = Path(path)
    summaries = [p for p in (path/'final_summary.json', path/'summary.json') if p.is_file()]
    require(bool(summaries), 'Missing completed summary')
    summary = read_json(summaries[0])
    require(summary['status'] == 'complete' and summary['mode'] == 'development_only', 'Run is not complete development-only evidence')
    require(summary.get('test') is None and summary.get('test_model_evaluations') == 0,
            'Regularization comparison must not include test evaluation')
    require(not list(path.glob('test_*')), 'Unexpected test artifacts in development-only experiment')
    require(summary['dataset_sha256'] == DATASET_SHA256 and summary['evaluator_version'] == EVALUATOR_VERSION,
            'Run dataset/evaluator identity differs')
    run = read_json(path/'run.json')
    require(run['runtime']['seed'] == seed, 'Directory seed differs from recorded seed')
    require(run['dataset_sha256'] == DATASET_SHA256 and run['evaluator_version'] == EVALUATOR_VERSION,
            'Run provenance differs from the pinned data/scorer')
    config = run['stage_config']['training']
    observed = tuple(float(config.get(key, 0.)) for key in REGULARIZATION_KEYS)
    require(observed == EXPECTED_SETTINGS[arm], 'Arm name differs from registered regularization settings')
    history_path = path/'history.jsonl' if (path/'history.jsonl').is_file() else path/'history.json'
    history = read_jsonl(history_path) if history_path.suffix == '.jsonl' else read_json(history_path)
    validate_history(history, summary, config, data)
    predictions_path = path/'dev_predictions.jsonl'
    predictions = read_jsonl(predictions_path)
    analyzed = stratified_metrics(data['dev'], predictions, strata)
    base_metrics = score_exact_spans(data['dev'], predictions)
    require(base_metrics == summary['dev']['metrics'], 'Saved metrics differ from strict recomputation')
    count = summary['parameter_counts']
    require(type(count['total']) is int and type(count['trainable']) is int
            and 0 < count['trainable'] <= count['total'] <= 20_000_000,
            'Invalid complete parameter count or cap exceeded')
    require(count == run['parameter_counts'], 'Parameter evidence disagrees')
    checkpoint = path / summary['checkpoint']
    require(checkpoint.parent.resolve() == path.resolve() and checkpoint.is_file(), 'Unsafe/missing checkpoint path')
    require(file_hash(checkpoint) == summary['checkpoint_sha256'], 'Selected checkpoint bytes changed')
    require((path/(checkpoint.name+'.sha256')).read_text().strip() == summary['checkpoint_sha256'],
            'Checkpoint checksum sidecar differs')
    reload = summary['fresh_process_reload']
    require(read_json(path/'fresh_reload.json') == reload,
            'Standalone fresh reload record differs from summary')
    require(reload['passed'] and reload['fresh_python_process'] and reload['ordered_predictions_equal'],
            'Fresh-process development reproduction was not verified')
    require(reload['metrics'] == base_metrics and reload['num_examples'] == len(data['dev']),
            'Fresh reload evidence differs')
    require(reload.get('test_evaluated') is False and reload['dataset_sha256'] == DATASET_SHA256
            and reload['checkpoint_sha256'] == summary['checkpoint_sha256'],
            'Fresh reload provenance/test boundary differs')
    require(summary['code_sha256'] == run['code_sha256'] == reload['code_sha256'],
            'Run, final summary and fresh verification code hashes differ')
    require(summary['dev']['record_ids_sha256'] == identity['train_dev_ordered_id_sha256']['dev'],
            'Saved development record-order hash differs')
    require(summary['model_weights_unchanged'] and summary['checkpoint_bytes_unchanged'], 'Evaluation changed model/checkpoint')
    files = [summaries[0], path/'run.json', history_path, predictions_path, path/'fresh_reload.json']
    embedding_path = path.parent/'embedding_initialization.json'
    embedding = read_json(embedding_path) if embedding_path.is_file() else None
    if embedding is not None:
        require(embedding['seed'] == seed and embedding['fit_partition'] == 'train'
                and embedding['full_source_table_retained_for_prediction'] is False,
                'Embedding provenance violates seed/training-only/table contract')
    result = {'seed': seed, 'arm': arm, 'run_path': str(path), **analyzed,
        'parameter_counts': count, 'epochs_trained': summary['epochs_trained'], 'selected_epoch': summary['selected_epoch'],
        'selection_score': summary['selection_score'], 'training_configuration': config,
        'model_configuration': run['stage_config']['model'], 'runtime': run['runtime'],
        'initial_model_state_sha256': run.get('initial_model_state_sha256'),
        'epoch_train_record_order_sha256': [r.get('train_record_order_sha256') for r in history],
        'epoch_word_dropout_masked_tokens': [r.get('train_word_dropout_masked_tokens') for r in history],
        'epoch_word_dropout_rng_state_sha256': [r.get('train_word_dropout_rng_state_sha256') for r in history],
        'regularization_metadata': run.get('regularization'),
        'code_sha256': run['code_sha256'],
        'embedding_source': run.get('run_options', {}).get('embedding_source'),
        'embedding_initialization': embedding,
        'training_workflow_seconds': summary.get('training_workflow_seconds'),
        'measured_epoch_train_dev_seconds': sum(r['train_seconds']+r['dev_seconds'] for r in history),
        'peak_allocated_cuda_bytes': summary.get('peak_allocated_cuda_bytes'),
        'model_state_sha256': summary.get('model_state_sha256'),
        'checkpoint_sha256': summary['checkpoint_sha256'], 'prediction_sha256': file_hash(predictions_path),
        'ordered_dev_predictions_sha256': json_hash(predictions),
        'source_evidence_sha256': {p.name: file_hash(p) for p in files},
        'fresh_development_reload_verified': True, 'test_model_evaluations': 0}
    return result, predictions, history


def validate_matched_protocol(baseline, candidate):
    require(baseline['seed'] == candidate['seed'] and baseline['model_configuration'] == candidate['model_configuration'],
            'Paired runs differ in seed or architecture')
    common = lambda config: {k: v for k, v in config.items() if k not in REGULARIZATION_KEYS}
    require(common(baseline['training_configuration']) == common(candidate['training_configuration']),
            'Paired runs differ beyond the declared regularizers')
    require(baseline['runtime'] == candidate['runtime'] and baseline['embedding_source'] == candidate['embedding_source'],
            'Paired runtime or embedding source differs')
    a, b = baseline.get('initial_model_state_sha256'), candidate.get('initial_model_state_sha256')
    if a is not None and b is not None:
        require(a == b, 'Paired initial model states differ')
    a, b = baseline.get('epoch_train_record_order_sha256'), candidate.get('epoch_train_record_order_sha256')
    if a and b and all(a) and all(b):
        require(a == b, 'Paired epoch training record orders differ')
    a, b = baseline.get('embedding_initialization'), candidate.get('embedding_initialization')
    if a and b:
        for key in ('matrix_sha256', 'source_archive_sha256'):
            require(a[key] == b[key], 'Paired embedding initializations differ')


def mean_sd(values):
    values = list(values)
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'sample_standard_deviation': statistics.stdev(values) if len(values) > 1 else None}


def aggregate_matched_seeds(runs):
    by_seed = {}
    for run in runs:
        require(run['arm'] not in by_seed.get(run['seed'], {}), 'Duplicated arm within seed')
        by_seed.setdefault(run['seed'], {})[run['arm']] = run
    complete_seeds = sorted(seed for seed, values in by_seed.items() if set(values) == set(ARMS))
    summaries = {}
    for arm in ARMS:
        records = [by_seed[seed][arm] for seed in complete_seeds]
        summaries[arm] = {'seeds': complete_seeds, 'num_seeds': len(records),
            'metrics': {metric: mean_sd(row['metrics'][metric] for row in records) for metric in METRIC_NAMES}}
        if arm != 'baseline':
            summaries[arm]['paired_seed_differences_pp'] = {metric: mean_sd(
                100*(by_seed[seed][arm]['metrics'][metric]-by_seed[seed]['baseline']['metrics'][metric])
                for seed in complete_seeds) for metric in METRIC_NAMES}
            summaries[arm]['seeds_improving_both_required_metrics'] = [seed for seed in complete_seeds if all(
                by_seed[seed][arm]['metrics'][m] > by_seed[seed]['baseline']['metrics'][m]
                for m in METRIC_NAMES[:2])]
    winner = max(ARMS, key=lambda arm: .5*sum(summaries[arm]['metrics'][m]['mean'] for m in METRIC_NAMES[:2])) if complete_seeds else None
    return {'complete_four_arm_seeds': complete_seeds, 'arms': summaries,
        'highest_mean_development_score_arm': winner,
        'caution': 'Only seeds with all four completed arms enter this table. Means are per-seed model metrics, not an ensemble or pooled repeated predictions. One seed has no estimated seed standard deviation.'}


def attach_confidence(report, path, run_root, dataset):
    """Bind optional saved DEV diagnostics to this exact complete study; no inference."""
    path = Path(path)
    require(path.is_file(), 'Explicit confidence file does not exist: '+str(path))
    require(report['status'] == 'complete' and report['confirmation'].get('complete') is True,
            'Confidence merge requires the complete registered comparison')
    confidence = read_json(path)
    require(confidence.get('schema_version') == 1 and confidence.get('status') == 'complete'
            and confidence.get('dataset_sha256') == report['dataset']['dataset_sha256']
            and confidence.get('evaluator_version') == report['dataset']['evaluator_version'],
            'Confidence dataset/evaluator/schema/status differs from this study')
    require(confidence.get('sources_unchanged') is True and confidence.get('test_model_evaluations') == 0
            and confidence.get('optimizer_created') is False, 'Confidence read-only/test boundary differs')
    runs = {(r['seed'], r['arm']): r for r in report['runs']}
    observed = [(r['seed'], r['arm']) for r in confidence['runs']]
    require(len(observed) == len(set(observed)) and set(observed) == set(runs),
            'Confidence seed/arm set differs or contains duplicates')
    # Include the protocol, diagnostic/analyzer code and every registered notebook:
    # matching performance alone is insufficient evidence that this is the same run.
    specification = report['registered_protocol']['protocol']
    required_sources = {str((ROOT/name).resolve()) for name in specification['source_sha256']}
    required_sources.update(str((ROOT/name).resolve()) for name in (
        'scripts/analyze_transformer_regularization_confidence.py',
        'scripts/analyze_transformer_regularization.py', 'scripts/validate_revised_transformer.py'))
    required_sources.update((str((Path(run_root)/'protocol.json').resolve()), str(Path(dataset).resolve())))
    for seed, arm in runs:
        directory = Path(run_root)/'notebook_sources'/f'seed_{seed}'/arm
        required_sources.update(str((directory/name).resolve()) for name in (
            'registration.json', 'NLP_Final_project_revised_regularized.ipynb'))
    source_hashes = {str(Path(name).resolve()): digest
                     for name, digest in confidence['sources_before_and_after_sha256'].items()}
    require(required_sources <= source_hashes.keys(), 'Confidence source provenance is incomplete')
    source_resolution = {name: resolve_recorded_source(run_root, name, digest)
                         for name, digest in source_hashes.items()}
    for record in confidence['runs']:
        expected = runs[record['seed'], record['arm']]
        require(Path(record['run_path']).resolve() == Path(expected['run_path']).resolve(),
                'Confidence artifact directory differs from this study')
        for key in ('selected_epoch', 'parameter_counts', 'checkpoint_sha256', 'model_state_sha256',
                    'initial_model_state_sha256', 'ordered_dev_predictions_sha256'):
            require(record[key] == expected[key], 'Confidence run identity differs for '+key)
        require(all(record.get(key) is True for key in ('saved_intent_and_slot_predictions_reproduced',
            'saved_unsmoothed_intent_loss_reproduced', 'model_state_unchanged', 'checkpoint_bytes_unchanged'))
            and record.get('dev_model_evaluations') == 1 and record.get('test_model_evaluations') == 0,
            'Confidence reproduction/mutation/split audit differs')
        calibration = record['calibration']
        require(calibration['num_examples'] == expected['metrics']['num_examples']
                and calibration['num_classes'] == report['dataset']['train_intent_count']
                and calibration['intent_errors'] == expected['metrics']['intent_errors']
                and abs(calibration['intent_accuracy']-expected['metrics']['intent_accuracy']) <= 1e-12,
                'Confidence population or intent predictions differ')
        require(record['logits_shape'] == [calibration['num_examples'], calibration['num_classes']]
                and re.fullmatch('[0-9a-f]{64}', record['logits_sha256']) is not None,
                'Confidence logits shape/hash is malformed')
        require(calibration['ece_num_bins'] == 15, 'Confidence ECE must use the registered 15 bins')
        for key in CALIBRATION_METRICS:
            value = calibration[key]
            if key == 'mean_incorrect_confidence' and calibration['intent_errors'] == 0:
                require(value is None, 'Error-free confidence mean must be unavailable')
            else:
                require(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) and value >= 0, 'Nonfinite/invalid confidence metric: '+key)
        for key in ('ece_15', 'mean_confidence', 'top2_accuracy', 'top3_accuracy'):
            require(calibration[key] <= 1, 'Confidence probability out of range: '+key)
        require(calibration['mean_incorrect_confidence'] is None or calibration['mean_incorrect_confidence'] <= 1,
                'Incorrect confidence out of range')
        require(calibration['multiclass_brier_score'] <= 2
                and calibration['intent_accuracy'] <= calibration['top2_accuracy'] <= calibration['top3_accuracy'],
                'Confidence Brier/top-k values are inconsistent')
        require(all(type(calibration[key]) is int for key in ('incorrect_confidence_ge_0_90', 'incorrect_confidence_ge_0_99'))
                and calibration['incorrect_confidence_ge_0_99'] <= calibration['incorrect_confidence_ge_0_90'] <= calibration['intent_errors'],
                'High-confidence error counts are inconsistent')
    complete_seeds = report['across_seed_summary']['complete_four_arm_seeds']
    require(all({arm for seed, arm in runs if seed == selected} == set(ARMS) for selected in complete_seeds)
            and set(complete_seeds) == {seed for seed, arm in runs}, 'Confidence requires all four arms per seed')
    by_arm = {arm: {'seeds': complete_seeds, 'metrics': {metric: mean_sd(
            row['calibration'][metric] for row in confidence['runs'] if row['arm'] == arm
            and row['calibration'][metric] is not None) for metric in CALIBRATION_METRICS}}
              for arm in ARMS}
    report['confidence_analysis'] = {'source_path': str(path.resolve()), 'source_sha256': file_hash(path),
        'completed_at_utc': confidence['completed_at_utc'], 'device': confidence['device'],
        'metric_definition': confidence['metric_definition'], 'runs': confidence['runs'],
        'sources_before_and_after_sha256': confidence['sources_before_and_after_sha256'],
        'source_resolution': source_resolution,
        'across_seed_summary': by_arm, 'test_model_evaluations': 0, 'merge_performed_inference': False,
        'limitations': confidence['limitations'] + [
            'Means and sample SD use the same complete matched seed set as the performance report.',
            'Incorrect-confidence means exclude seeds with zero errors; n is recorded. This is not probability calibration fitting.',
            'The reporter merges explicitly supplied saved evidence; it does not perform additional inference.']}


def analyze(run_root, dataset, *, replicates=10000, bootstrap_seed=20261111, confidence=None):
    root = Path(run_root)
    data, identity = load_train_dev(dataset)
    strata = fit_strata(data['train'], data['dev'])
    runs, predictions, skipped, histories = [], {}, [], {}
    seed_dirs = sorted((p for p in root.glob('seed_*') if p.is_dir() and re.fullmatch(r'seed_\d+', p.name)),
                       key=lambda p: int(p.name.split('_')[1]))
    require(bool(seed_dirs), 'No seed directories found')
    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.split('_')[1])
        for arm in ARMS:
            path = discover_stage_path(seed_dir/arm)
            if path is None:
                skipped.append({'seed': seed, 'arm': arm, 'reason': 'No completed run summary'})
                continue
            record, predicted, history = load_run(path, seed, arm, data, identity, strata)
            runs.append(record)
            predictions[seed, arm] = predicted
            histories[seed, arm] = history
    require(bool(runs), 'No complete development runs are available')
    lookup = {(r['seed'], r['arm']): r for r in runs}
    comparisons = []
    for seed in sorted({r['seed'] for r in runs}):
        if (seed, 'baseline') not in lookup:
            continue
        for arm in ARMS[1:]:
            if (seed, arm) not in lookup:
                continue
            validate_matched_protocol(lookup[seed, 'baseline'], lookup[seed, arm])
            comparisons.append({'seed': seed, 'candidate': arm, 'reference': 'baseline',
                **paired_comparison(data['dev'], predictions[seed, 'baseline'], predictions[seed, arm],
                                    replicates=replicates, seed=bootstrap_seed + seed)})
        if (seed, 'word_dropout') in lookup and (seed, 'combined') in lookup:
            for field in ('epoch_word_dropout_masked_tokens', 'epoch_word_dropout_rng_state_sha256'):
                require(lookup[seed, 'word_dropout'][field] == lookup[seed, 'combined'][field],
                        'Paired word-dropout streams differ')
    protocol_path = root/'protocol.json'
    protocol = read_json(protocol_path) if protocol_path.is_file() else None
    confirmation = {'registered': protocol is not None}
    if protocol is not None:
        specification = protocol['protocol']
        require(specification['dataset_sha256'] == DATASET_SHA256
                and specification['evaluator_version'] == EVALUATOR_VERSION
                and specification['test_evaluation'] is False, 'Experiment protocol differs')
        require(specification['arms'] == {arm: dict(zip(REGULARIZATION_KEYS, values))
                for arm, values in EXPECTED_SETTINGS.items()}, 'Registered arm coefficients differ')
        for name in ('src/comparison_contract.py', 'src/revised_transformer_data.py'):
            require(file_hash(ROOT/name) == specification['source_sha256'][name],
                    'Historical scoring/tokenization requires unchanged current implementation: '+name)
        source_resolution = {name: resolve_registered_source(root, name, digest)
                             for name, digest in specification['source_sha256'].items()}
        for record in runs:
            record['registered_notebook_audit'] = validate_registered_notebook(root, record, specification)
        require(all(r['seed'] in specification['registered_seeds']
                    and r['epochs_trained'] == specification['epochs'] for r in runs),
                'Completed run seed or budget was not registered')
        require(all(r['initial_model_state_sha256'] and r['embedding_initialization']
                    and all(r['epoch_train_record_order_sha256']) for r in runs),
                'Registered experiment is missing initialization or order evidence')
        trigger_arms = [row['candidate'] for row in comparisons
                        if row['seed'] == 42 and row['both_required_metrics_improved']]
        screen_complete = all((42, arm) in lookup for arm in ARMS)
        required_seeds = [seed for seed in specification['registered_seeds'] if seed != 42] if trigger_arms else []
        confirmation = {'registered': True, 'policy': specification['confirmation_policy'],
            'screen_seed': 42, 'screen_complete': screen_complete,
            'screen_arms_improving_both': trigger_arms,
            'confirmation_required': bool(trigger_arms), 'required_seeds': required_seeds,
            'missing_seed_arms': [{'seed': seed, 'arm': arm} for seed in required_seeds for arm in ARMS
                                  if (seed, arm) not in lookup],
            'confirmation_only_summary': aggregate_matched_seeds([r for r in runs if r['seed'] != 42])}
        confirmation['complete'] = screen_complete and not confirmation['missing_seed_arms']
    report = {'schema_version': 1, 'status': 'complete' if not skipped else 'partial',
        'scope': 'Saved train/development experiment evidence; no model inference, training or test scoring',
        'dataset': identity, 'runs': runs, 'incomplete_or_absent_arms': skipped,
        'per_seed_paired_development_comparisons': comparisons,
        'across_seed_summary': aggregate_matched_seeds(runs),
        'registered_protocol': protocol, 'confirmation': confirmation,
        'registered_source_resolution': source_resolution if protocol is not None else {},
        'novelty_definition': 'Seen means (slot type, casefolded literal value) occurs in original train; words also fitted on train. Official exact scoring is unchanged.',
        'analysis_configuration': {'bootstrap_replicates': replicates, 'bootstrap_seed_base': bootstrap_seed},
        'test_evaluation_performed': False,
        'limitations': ['Development selected checkpoints and is also used to assess proposals; bootstrap intervals omit selection bias and multiplicity adjustment.',
            'Per-seed paired bootstrap measures utterance sampling conditional on checkpoints; across-seed standard deviation describes a different uncertainty.',
            'The previously evaluated official test set has not been used or rescored by this analysis.',
            'Do not treat a higher mean score with a decline in one metric as improvement in both.',
            'Checkpoint bytes and recorded total parameter counts are verified; this results analyzer does not instantiate/count model tensors.']}
    if confirmation.get('registered') and not confirmation['complete']:
        report['status'] = 'partial'
    execution_plan_path = root/'confirmation_execution_plan.json'
    if execution_plan_path.is_file():
        report['confirmation_execution_plan'] = {'path': str(execution_plan_path),
            'sha256': file_hash(execution_plan_path), 'plan': read_json(execution_plan_path)}
    parity_path = ROOT/'results/transformer_regularization/historical_control_parity.json'
    if parity_path.is_file() and (42, 'baseline') in lookup:
        parity = read_json(parity_path)
        baseline = lookup[42, 'baseline']
        if parity.get('status') == 'passed' and parity.get('new_checkpoint_sha256') == baseline['checkpoint_sha256']:
            require(parity['model_state_sha256'] == baseline['model_state_sha256'],
                    'Historical-control audit selected model hash differs')
            report['historical_control_parity'] = {'evidence_path': str(parity_path.relative_to(ROOT)),
                'evidence_sha256': file_hash(parity_path), 'audit': parity}
    if confidence is not None:
        attach_confidence(report, confidence, root, dataset)
    return report


def format_mean_sd(record):
    if record['mean'] is None:
        return 'Unavailable'
    value = f"{100*record['mean']:.2f}%"
    if record['sample_standard_deviation'] is not None:
        value += f" ± {100*record['sample_standard_deviation']:.2f}"
    return value


def format_calibration_mean_sd(record, *, scale=1., digits=3, suffix=''):
    if record['mean'] is None:
        return 'Unavailable'
    value = f"{scale*record['mean']:.{digits}f}{suffix}"
    if record['sample_standard_deviation'] is not None:
        value += f" ± {scale*record['sample_standard_deviation']:.{digits}f}"
    return value


def markdown_report(report):
    seed_summary = report['across_seed_summary']
    lines = ['# Transformer regularization: measured development results', '',
        'This report is generated from completed saved runs and strict rescoring of their development predictions. '
        'It compares the revised Transformer with rare-word dropout, intent label smoothing, and their combination. '
        'All model-selection and comparison data here are development data; this analysis performs no model inference or test evaluation.', '',
        f"Analysis status: **{report['status']}**. Completed runs: **{len(report['runs'])}**. "
        f"Complete four-arm seed sets: **{seed_summary['complete_four_arm_seeds']}**.", '',
        '## Every completed run', '',
        '| Seed | Arm | Intent | Slot F1 | Slot precision | Slot recall | Joint exact | Selected / trained epochs | Total parameters | Epoch train + dev seconds |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |']
    for row in report['runs']:
        m = row['metrics']
        values = [f'{100*m[k]:.2f}%' for k in ('intent_accuracy','slot_micro_f1','slot_precision','slot_recall','fully_correct_utterance_rate')]
        lines.append(f"| {row['seed']} | {row['arm']} | " + ' | '.join(values) +
            f" | {row['selected_epoch']} / {row['epochs_trained']} | {row['parameter_counts']['total']:,} | {row['measured_epoch_train_dev_seconds']:.2f} |")
    lines += ['', 'Each reported checkpoint is the earliest maximum of the registered unrounded mean development intent accuracy and exact slot micro-F1. '
        'Every original training/development example was retained in every recorded epoch. Training loss heights are not directly comparable when label smoothing changes the objective.', '',
        '## Matched seeds', '',
        '| Arm | Seeds | Intent mean ± seed SD | Slot F1 mean ± seed SD | Joint exact mean ± seed SD |',
        '| --- | --- | --- | --- | --- |']
    for arm in ARMS:
        record = seed_summary['arms'][arm]
        lines.append(f"| {arm} | {record['seeds']} | " + ' | '.join(format_mean_sd(record['metrics'][m]) for m in METRIC_NAMES) + ' |')
    lines += ['', seed_summary['caution'], '']
    confirmation = report['confirmation']
    if confirmation.get('registered'):
        lines += [f"Registered confirmation policy: {confirmation['policy']}", '',
            f"Seed-42 arms improving both metrics: **{confirmation['screen_arms_improving_both']}**. "
            f"Confirmation complete: **{confirmation['complete']}**. "
            'The seed-42 screen and its conditional follow-up are disclosed; this was not an unconditional three-seed experiment.', '']
        followup = confirmation['confirmation_only_summary']
        if followup['complete_four_arm_seeds']:
            lines += ['Confirmation seeds alone (excluding the screening seed):', '',
                '| Arm | Seeds | Intent mean ± seed SD | Slot F1 mean ± seed SD |',
                '| --- | --- | --- | --- |']
            for arm in ARMS:
                record = followup['arms'][arm]
                lines.append(f"| {arm} | {record['seeds']} | " + ' | '.join(
                    format_mean_sd(record['metrics'][metric]) for metric in METRIC_NAMES[:2]) + ' |')
            lines.append('')
    winner = seed_summary['highest_mean_development_score_arm']
    if winner:
        lines += [f"The highest mean development selection score belongs to **{winner}**. This ranking uses complete matched seed sets, not the best individual seed.", '']
        if winner != 'baseline':
            deltas = seed_summary['arms'][winner]['paired_seed_differences_pp']
            both = all(deltas[m]['mean'] > 0 for m in METRIC_NAMES[:2])
            lines += [f"Against baseline, its mean intent difference is **{deltas['intent_accuracy']['mean']:+.2f} points** and slot-F1 difference is **{deltas['slot_micro_f1']['mean']:+.2f} points**. "
                + ('Both required mean metrics improved.' if both else 'This is a task tradeoff; both required mean metrics did not improve.'), '']
    lines += ['## Paired development comparisons', '',
        '| Seed | Candidate minus baseline | Intent points [nominal 95% interval] | Slot-F1 points [nominal 95% interval] | Joint exact points [nominal 95% interval] | Intent corrected / introduced |',
        '| ---: | --- | --- | --- | --- | --- |']
    for row in report['per_seed_paired_development_comparisons']:
        values = []
        for metric in METRIC_NAMES:
            d = row['differences'][metric]
            a, b = d['nominal_95_percentile_interval_pp']
            values.append(f"{d['percentage_points']:+.2f} [{a:+.2f}, {b:+.2f}]")
        lines.append(f"| {row['seed']} | {row['candidate']} | " + ' | '.join(values) + f" | {row['intent_errors_corrected']} / {row['intent_errors_introduced']} |")
    lines += ['', f"Intervals use {report['analysis_configuration']['bootstrap_replicates']:,} paired whole-utterance resamples per comparison. "
        'They condition on these development-selected checkpoints, omit selection bias and multiplicity adjustment, and do not estimate training-seed variation. '
        'Report regressions and small/uncertain changes alongside gains.', '',
        '## Novel slot values and unknown words', '', report['novelty_definition'], '',
        '| Seed | Arm | Seen value recall | New value / known words recall | New value / OOV words recall | Unseen predicted-value precision | No-slot false positives |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in report['runs']:
        profiles = row['gold_typed_value_recall']
        values = [f"{100*profiles[k]['recall']:.2f}% ({profiles[k]['correct']}/{profiles[k]['gold_support']})"
                  for k in ('seen_typed_value','unseen_value_known_words','unseen_value_oov_words')]
        p = row['predicted_typed_value_precision']['unseen_typed_value']
        no_slots = row['utterance_groups']['no_gold_slots']
        lines.append(f"| {row['seed']} | {row['arm']} | " + ' | '.join(values) +
            f" | {100*p['precision']:.2f}% | {no_slots.get('slot_fp',0)} |")
    lines += ['', 'The JSON evidence also contains utterance-level no-OOV, non-slot-OOV and slot-OOV strata, full counts, hashes, runtime/configuration values and paired recovered/lost gold slot occurrences. '
        'Subgroup movements are descriptive: improved OOV recall does not compensate for a decrease in overall exact slot F1.', '']
    if report.get('confidence_analysis'):
        confidence = report['confidence_analysis']
        source = Path(confidence['source_path'])
        try:
            source_link = '../'+source.relative_to(ROOT).as_posix()
        except ValueError:
            source_link = source.as_posix()
        lines += ['## Intent calibration and confidence', '',
            'These additional diagnostics use the same selected checkpoints and original development utterances. '
            'They merge explicitly supplied saved confidence evidence; this reporting step performs no inference. '
            'Values are means ± sample standard deviations across the same complete matched seeds listed above. '
            'They include the screening seed when present; the confirmation policy remains conditional.', '',
            '| Arm | NLL (nats) | ECE15 (%) | Multiclass Brier | Mean confidence (%) | Mean incorrect confidence (%) |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for arm in ARMS:
            metrics = confidence['across_seed_summary'][arm]['metrics']
            values = [format_calibration_mean_sd(metrics[key], scale=scale, digits=digits)
                      for key, scale, digits in [('negative_log_likelihood',1,4), ('ece_15',100,2),
                        ('multiclass_brier_score',1,4), ('mean_confidence',100,2), ('mean_incorrect_confidence',100,2)]]
            lines.append('| '+arm+' | '+' | '.join(values)+' |')
        lines += ['', 'Lower NLL, ECE and Brier indicate better probability quality under their stated definitions. '
            'Confidence itself is descriptive: lower confidence alone is not a performance improvement. '
            'Percentage standard deviations are percentage points; NLL uses ordinary unsmoothed gold-intent probabilities. '
            'Incorrect-confidence means exclude error-free seeds, with the contributing n retained in JSON.', '',
            '| Arm | Incorrect predictions ≥90% confidence | Incorrect predictions ≥99% confidence | Top-2 accuracy (%) | Top-3 accuracy (%) |',
            '| --- | ---: | ---: | ---: | ---: |']
        for arm in ARMS:
            metrics = confidence['across_seed_summary'][arm]['metrics']
            values = [format_calibration_mean_sd(metrics[key], scale=scale, digits=2)
                      for key, scale in [('incorrect_confidence_ge_0_90',1), ('incorrect_confidence_ge_0_99',1),
                                         ('top2_accuracy',100), ('top3_accuracy',100)]]
            lines.append('| '+arm+' | '+' | '.join(values)+' |')
        lines += ['', 'High-confidence error entries are per-seed utterance counts, averaged across seeds. '
            'ECE uses 15 equally spaced confidence bins and depends on binning/sample size. '
            'Better calibration or top-k accuracy does not establish higher top-1 intent accuracy or slot F1. '
            'Development selected these checkpoints, so these remain development diagnostics.', '',
            f"Source: [saved development confidence evidence](<{source_link}>) "
            f"(SHA-256 `{confidence['source_sha256']}`; completed `{confidence['completed_at_utc']}`, device `{confidence['device']}`). "
            'The merger verifies identical complete seed/arm sets, checkpoint/state/ordered-prediction hashes, intent accuracy/error counts, '
            'and exact recorded diagnostic, protocol, dataset and registered notebook sources. '
            'Historical Python source bytes may resolve to the protocol-bound source archive; these files are verified as evidence and never executed. '
            'The JSON report retains individual-run diagnostics, metric definitions and the full source-hash manifest.', '']
    lines += [
        '## Evidence and scope', '',
        f"- Dataset SHA-256: `{report['dataset']['dataset_sha256']}`.",
        f"- Evaluator: `{report['dataset']['evaluator_version']}`; original counts {report['dataset']['counts']}.",
        '- JSON and CSV: `results/transformer_regularization/results.json` and `results/transformer_regularization/results.csv`.',
        '- Saved checkpoint hashes, complete epoch histories, original ordered development IDs, exact scores and fresh-process reproduction records are checked before inclusion.',
        '- Source resolution is recorded in JSON. Changed Python implementation files require exact archived bytes bound to the original protocol; the current scoring contract/tokenizer must still match. Archives are never executed by this analyzer.',
        '- All declared whole-model counts must be at most 20M; this analysis does not reinstantiate the model.',
        '- Timings are descriptive workflow measurements. The seed-42 combined run also executes Stage 1 before Stage 2, unlike script-only arms; confirmation arms may share GPU capacity concurrently. These are not controlled speed benchmarks and support no speed-gain claim.',
        '- The official test set was previously exposed in the earlier frozen comparison. No new test result is claimed here; future evaluation must disclose that history.', '']
    if report.get('historical_control_parity'):
        lines += ['The separate [historical-control parity audit](../results/transformer_regularization/historical_control_parity.json) '
            'records exact agreement between the fresh seed-42 zero-regularizer control and the preserved revised baseline '
            'over all 20 epochs of losses, development metrics and schedules, including the selected learned state and saved development prediction bytes. '
            'Checkpoint file hashes differ because provenance metadata changed. This analyzer binds that audit to the current baseline checkpoint/state hashes.', '']
    if report.get('confirmation_execution_plan'):
        lines += ['Concurrent scheduling details are retained in '
            '[confirmation_execution_plan.json](../artifacts/transformer_regularization/confirmation_execution_plan.json) '
            'and embedded with their file hash in the JSON evidence. GPU contention affects timing comparability.', '']
    if report['incomplete_or_absent_arms']:
        lines += ['Incomplete or absent arms are excluded, not counted as failed performance results:', '']
        lines.extend(f"- Seed {row['seed']}, {row['arm']}: {row['reason']}." for row in report['incomplete_or_absent_arms'])
        lines.append('')
    return '\n'.join(lines)


def write_outputs(report, output_dir, markdown_path):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output/'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    rows = []
    for run in report['runs']:
        rows.append({'seed': run['seed'], 'arm': run['arm'], 'split': 'dev',
            **{name+'_percent': f"{100*run['metrics'][name]:.2f}" for name in ('intent_accuracy','slot_micro_f1','slot_precision','slot_recall','fully_correct_utterance_rate')},
            'total_parameters': run['parameter_counts']['total'], 'trainable_parameters': run['parameter_counts']['trainable'],
            'epochs_trained': run['epochs_trained'], 'selected_epoch': run['selected_epoch'],
            'measured_epoch_train_dev_seconds': f"{run['measured_epoch_train_dev_seconds']:.6f}"})
    with (output/'results.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path = Path(markdown_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_report(report), encoding='utf-8', newline='\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=ROOT/'artifacts/transformer_regularization')
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/1.0/data/en-US.jsonl')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/transformer_regularization')
    parser.add_argument('--markdown', type=Path, default=ROOT/'docs/TRANSFORMER_REGULARIZATION_RESULTS.md')
    parser.add_argument('--bootstrap-replicates', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=20261111)
    parser.add_argument('--confidence', type=Path, default=None,
        help='Explicit saved confidence.json to merge after all required runs; omitted means no confidence loading/inference.')
    args = parser.parse_args()
    report = analyze(args.run_root, args.dataset, replicates=args.bootstrap_replicates,
                     bootstrap_seed=args.bootstrap_seed, confidence=args.confidence)
    write_outputs(report, args.output_dir, args.markdown)
    print(json.dumps({'status': report['status'], 'completed_runs': len(report['runs']),
        'matched_seeds': report['across_seed_summary']['complete_four_arm_seeds'],
        'highest_mean_development_score_arm': report['across_seed_summary']['highest_mean_development_score_arm'],
        'output_dir': str(args.output_dir), 'markdown': str(args.markdown)}))


if __name__ == '__main__':
    main()

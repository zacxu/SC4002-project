"""Independently validate revised-Transformer saved evidence without inference.

The auditor reads inline source, original data, logs, predictions and CPU tensors.
It never constructs a model, performs a forward pass, updates weights or selects
using test scores. Development-only acceptance is not final-test acceptance.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

import nbformat


DATA_SHA256 = 'c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e'
EVALUATOR = 'exact_character_span_v2'
COUNTS = {'train': 11514, 'dev': 2033, 'test': 2974}
TOKEN = re.compile(r"\w+(?:['\u2019\-]\w+)*|[^\w\s]", re.UNICODE)
ANNOTATION = re.compile(r'\[([A-Za-z_][A-Za-z_0-9]*)[ \t]*:[ \t]*(.*?)\]', re.DOTALL)
CONFIG_NAMES = {'RUNTIME', 'STAGE1_CONFIG', 'STAGE2_CONFIG', 'RUN_OPTIONS'}


class ValidationError(ValueError):
    """A saved-evidence gate failed."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def evidence(root, name):
    require(isinstance(name, str) and bool(name) and '\\' not in name,
            'Evidence names must be nonempty relative POSIX paths')
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts and not re.match(r'^[A-Za-z]:', name),
            'Evidence path must stay inside its run root')
    resolved = (root / relative).resolve()
    require(resolved.is_relative_to(root.resolve()) and resolved.is_file(), 'Missing or unsafe evidence: ' + name)
    return resolved


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    rows = Path(path).read_text(encoding='utf-8').splitlines()
    require(bool(rows) and all(row.strip() for row in rows), 'JSONL evidence is empty or contains blank rows')
    return [json.loads(row) for row in rows]


def configuration_values(source):
    result = {}
    try:
        statements = ast.parse(source).body
        for node in statements:
            require(isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name), 'Configuration requires literal assignments only')
            name = node.targets[0].id
            require(name in CONFIG_NAMES and name not in result, 'Unexpected or duplicate configuration name')
            value = ast.literal_eval(node.value)
            require(isinstance(value, dict), 'Configuration values must be dictionaries')
            result[name] = value
    except (SyntaxError, TypeError, ValueError) as error:
        if isinstance(error, ValidationError):
            raise
        raise ValidationError('Invalid literal configuration') from error
    require(set(result) == CONFIG_NAMES, 'All four configuration dictionaries are required')
    return result


def validate_source(path):
    document = nbformat.read(path, as_version=4)
    nbformat.validate(document)
    cells = [cell for cell in document.cells if cell.cell_type == 'code']
    metadata = document.metadata.get('revised_transformer', {})
    expected, registered = metadata.get('code_cell_sha256'), metadata.get('configuration_cell_indices')
    require(isinstance(expected, list) and len(expected) == len(cells), 'Missing notebook code manifest')
    require(all(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) for value in expected),
            'Malformed notebook code manifest')
    tagged = [i for i, cell in enumerate(cells) if 'configuration' in cell.metadata.get('tags', [])]
    require(isinstance(registered, list) and len(registered) == 1 and registered == tagged
            and type(registered[0]) is int, 'Expected one registered editable configuration cell')
    actual = []
    for index, cell in enumerate(cells):
        try:
            tree = ast.parse(cell.source)
        except SyntaxError as error:
            raise ValidationError('Notebook contains invalid Python') from error
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                require(not node.level and not (node.module or '').split('.')[0] in {'src', 'scripts'},
                        'Notebook imports unsubmitted project implementation')
            elif isinstance(node, ast.Import):
                require(not any(alias.name.split('.')[0] in {'src', 'scripts'} for alias in node.names),
                        'Notebook imports unsubmitted project implementation')
        digest = hashlib.sha256(cell.source.encode('utf-8')).hexdigest()
        actual.append(digest)
        if index not in registered:
            require(digest == expected[index], 'Protected notebook source changed')
    config = configuration_values(cells[registered[0]].source)
    shared = [cell for cell in cells if 'comparison_contract' in cell.metadata.get('tags', [])]
    require(len(shared) == 1, 'Exactly one inline canonical evaluator/data contract is required')
    canonical = Path(__file__).resolve().parents[1] / 'src/comparison_contract.py'
    require(shared[0].source == canonical.read_text(encoding='utf-8'), 'Inline canonical contract differs from reviewed source')
    contract = document.metadata.get('comparison_contract', {})
    require(contract.get('evaluator_version') == EVALUATOR
            and contract.get('source_sha256') == hashlib.sha256(shared[0].source.encode('utf-8')).hexdigest(),
            'Canonical contract source/evaluator identity differs')
    counts = [cell.get('execution_count') for cell in cells]
    require(counts == list(range(1, len(cells) + 1)), 'Notebook code cells were not all executed once in order')
    visible, pngs = [], []
    for cell in cells:
        for output in cell.get('outputs', []):
            require(output.get('output_type') != 'error', 'Notebook retains an execution error')
            if output.get('text'):
                visible.append(output['text'])
            if 'image/png' in output.get('data', {}):
                import base64
                pngs.append(hashlib.sha256(base64.b64decode(output['data']['image/png'])).hexdigest())
    return {'notebook_sha256': file_hash(path), 'code_sha256': json_hash(actual),
            'configuration_sha256': json_hash(config), 'configuration': config,
            'code_cells': len(cells), 'visible_text': '\n'.join(visible), 'inline_png_sha256': pngs,
            'canonical_contract_sha256': contract['source_sha256']}


def parse_gold(row):
    pieces, slots = [], []
    annotated, offset, cursor = row['annot_utt'], 0, 0
    for match in ANNOTATION.finditer(annotated):
        before = annotated[cursor:match.start()]
        pieces.append(before); offset += len(before)
        text = match.group(2)
        require(bool(text), 'Empty original annotated span')
        slots.append({'slot_type': match.group(1), 'text': text, 'start': offset, 'end': offset + len(text)})
        pieces.append(text); offset += len(text); cursor = match.end()
    pieces.append(annotated[cursor:])
    require(''.join(pieces) == row['utt'], 'Annotations do not reconstruct original raw utterance')
    return {'record_id': row['id'], 'utt': row['utt'], 'intent': row['intent'], 'slots': slots}


def verified_data(path):
    require(file_hash(path) == DATA_SHA256, 'Original dataset SHA256 differs')
    raw = read_jsonl(path); split = {name: [] for name in COUNTS}; seen = set()
    for row in raw:
        require(row.get('locale') == 'en-US' and row.get('partition') in split, 'Unexpected locale/partition')
        require(isinstance(row.get('id'), str) and row['id'] and row['id'] not in seen, 'Duplicate/invalid original ID')
        seen.add(row['id']); split[row['partition']].append(row)
    require({name: len(rows) for name, rows in split.items()} == COUNTS, 'Original partition counts differ')
    identity = {'evaluator_version': EVALUATOR, 'dataset_sha256': DATA_SHA256,
        'source_url': 'https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz',
        'archive_member': '1.0/data/en-US.jsonl', 'counts': COUNTS,
        'split_id_sha256': {name: json_hash([row['id'] for row in rows]) for name, rows in split.items()},
        'split_content_sha256': {name: json_hash(rows) for name, rows in split.items()},
        'identity_scope': 'Pinned original bytes; original row order and every record field retained'}
    gold = {name: [parse_gold(row) for row in rows] for name, rows in split.items()}
    words = sorted({match.group().casefold() for row in split['train'] for match in TOKEN.finditer(row['utt'])})
    chars = sorted({char for row in split['train'] for match in TOKEN.finditer(row['utt']) for char in match.group()})
    intents = sorted({row['intent'] for row in split['train']})
    types = sorted({slot['slot_type'] for row in gold['train'] for slot in row['slots']})
    mappings = {'word_to_id': {word: i for i, word in enumerate(['<PAD>', '<UNK>'] + words)},
        'char_to_id': {char: i for i, char in enumerate(['<CPAD>', '<CUNK>'] + chars)},
        'intent_to_id': {intent: i for i, intent in enumerate(intents)},
        'tag_names': ['O'] + [prefix + '-' + name for name in types for prefix in ('B', 'I')],
        'slot_types': types}
    token_counts = {name: sum(len(list(TOKEN.finditer(row['utt']))) for row in rows) for name, rows in split.items()}
    return split, gold, identity, mappings, token_counts


def _span_keys(record, utterance):
    keys = set()
    for slot in record.get('slots', []):
        start, end, text, label = (slot.get(key) for key in ('start', 'end', 'text', 'slot_type'))
        require(type(start) is int and type(end) is int and 0 <= start < end <= len(utterance),
                'Invalid character span boundaries')
        require(isinstance(text, str) and text and utterance[start:end] == text
                and isinstance(label, str) and label, 'Span is not an exact original typed text slice')
        key = (label, start, end, text)
        require(key not in keys, 'Duplicate identical span occurrence')
        keys.add(key)
    return keys


def manual_metrics(gold, predictions, *, require_order=True):
    require(bool(gold), 'Cannot score an empty split')
    reference = {row['record_id']: row for row in gold}
    require(len(reference) == len(gold), 'Duplicate gold IDs')
    observed = {}
    for row in predictions:
        record_id = row.get('record_id')
        require(isinstance(record_id, str) and record_id and record_id not in observed, 'Duplicate or invalid prediction ID')
        require(isinstance(row.get('intent'), str) and row['intent'], 'Invalid predicted intent')
        observed[record_id] = row
    require(set(reference) == set(observed), 'Missing or unexpected prediction IDs')
    if require_order:
        require(list(reference) == list(observed), 'Prediction order differs from original records')
    correct = tp = fp = fn = 0
    for record_id, expected in reference.items():
        predicted = observed[record_id]
        if 'utt' in predicted:
            require(predicted['utt'] == expected['utt'], 'Prediction raw utterance differs')
        target, actual = _span_keys(expected, expected['utt']), _span_keys(predicted, expected['utt'])
        correct += predicted['intent'] == expected['intent']
        tp += len(target & actual); fp += len(actual - target); fn += len(target - actual)
    denominator = 2 * tp + fp + fn
    return {'intent_accuracy': correct / len(gold), 'slot_micro_f1': 2 * tp / denominator if denominator else 0,
            'slot_tp': tp, 'slot_fp': fp, 'slot_fn': fn, 'num_examples': len(gold), 'correct_intents': correct}


def check_metrics(recorded, actual):
    for key, expected in actual.items():
        observed = recorded.get(key)
        valid = (type(observed) is int and observed == expected) if type(expected) is int else (
            isinstance(observed, (int, float)) and not isinstance(observed, bool)
            and math.isfinite(observed) and abs(observed - expected) <= 1e-12)
        require(valid, 'Metric/count mismatch: ' + key)


def validate_history(rows, training, split, token_counts):
    epochs, batch = training['max_epochs'], training['batch_size']
    require(type(epochs) is int and epochs > 0 and type(batch) is int and batch > 0, 'Invalid epoch/batch budget')
    require([row.get('epoch') for row in rows] == list(range(1, epochs + 1)), 'Incomplete configured epoch budget')
    best, winner = None, None
    for row in rows:
        for name in ('train', 'dev'):
            require(row.get(name + '_num_examples') == len(split[name])
                    and row.get(name + '_num_tokens') == token_counts[name]
                    and row.get(name + '_num_batches') == math.ceil(len(split[name]) / batch),
                    'Incomplete original ' + name + ' traversal')
            ids = [item['id'] for item in split[name]]
            require(row.get(name + '_record_ids_sha256') == json_hash(sorted(ids) if name == 'train' else ids),
                    'Original record membership/order hash differs')
            for component in ('loss', 'intent_loss', 'slot_loss'):
                require(isinstance(row.get(name + '_' + component), (int, float))
                        and math.isfinite(row[name + '_' + component]), 'Nonfinite epoch loss')
            require(abs(row[name + '_loss'] - row[name + '_intent_loss'] - row[name + '_slot_loss']) <= 1e-10,
                    'Joint loss does not equal the declared normalized components')
        for key in ('dev_intent_accuracy', 'dev_slot_micro_f1'):
            require(isinstance(row.get(key), (int, float)) and 0 <= row[key] <= 1, 'Invalid development metric')
        score = .5 * (row['dev_intent_accuracy'] + row['dev_slot_micro_f1'])
        require(row.get('selection_score') == score, 'Development selection mean differs')
        selected = best is None or score > best
        require(row.get('selected') is selected, 'Earliest development checkpoint-selection flags differ')
        if selected:
            best, winner = score, row['epoch']
        require(row.get('optimizer_step') == row['epoch'] * math.ceil(len(split['train']) / batch),
                'Optimizer-step history does not cover every original training batch')
    return {'epochs_trained': epochs, 'selected_epoch': winner, 'selection_score': best}


def state_evidence(state, counts):
    import torch
    require(isinstance(state, dict) and state and all(isinstance(value, torch.Tensor) for value in state.values()),
            'Checkpoint must contain ordinary state tensors')
    require(all(value.dtype == torch.bool or value.is_floating_point() for value in state.values()),
            'Unexpected nonlearned/nonfloating checkpoint tensor')
    require(all(value.dtype != torch.bool or name in {'crf.start_allowed', 'crf.transition_allowed'}
                for name, value in state.items()), 'Only declared Boolean CRF buffers may be excluded from parameter counts')
    total = sum(value.numel() for value in state.values() if value.dtype != torch.bool)
    require(type(counts.get('total')) is int and counts['total'] == total and 0 < total <= 20_000_000,
            'Complete learned parameter count/cap differs')
    require(counts.get('trainable') == total, 'Declared all-trainable model count differs')
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        require(bool(torch.isfinite(value).all()), 'Checkpoint contains nonfinite learned weights')
        digest.update(name.encode('utf-8')); digest.update(str(value.dtype).encode('ascii'))
        digest.update(str(tuple(value.shape)).encode('ascii')); digest.update(value.numpy().tobytes())
    return total, digest.hexdigest()


def validate_mode_artifacts(root, stage, mode, summary, reload, checksum,
                            tensor_sha, selected, code_sha256):
    if mode == 'development_only':
        require(summary.get('test') is None, 'Development-only run contains test summary')
        require(not any((root / stage / name).exists() for name in
            ('freeze_manifest.json', 'test_started.json', 'test_predictions.jsonl', 'test_evaluation.json')),
            'Development-only run contains test/freeze artifacts')
        return
    require(mode == 'final', 'Unknown evaluation mode')
    freeze = read_json(evidence(root, stage + '/freeze_manifest.json'))
    started = read_json(evidence(root, stage + '/test_started.json'))
    require(freeze.get('test_model_evaluations') == 0 and freeze.get('tuning_closed') is True
            and started.get('test_model_evaluations') == 1
            and started.get('freeze_sha256') == file_hash(root / stage / 'freeze_manifest.json')
            and freeze.get('checkpoint_sha256') == checksum and freeze.get('model_state_sha256') == tensor_sha
            and freeze.get('history_sha256') == file_hash(root / stage / 'history.jsonl')
            and freeze.get('selected_epoch') == selected['selected_epoch']
            and freeze.get('code_sha256') == code_sha256 and freeze.get('dataset_sha256') == DATA_SHA256,
            'Pre-test frozen evidence changed')
    require(datetime.fromisoformat(reload['verified_at_utc']) <= datetime.fromisoformat(freeze['frozen_at_utc'])
            <= datetime.fromisoformat(started['started_at_utc']) <= datetime.fromisoformat(summary['completed_at_utc']),
            'Freeze/test chronology is inconsistent')


def run(notebook, run_root):
    notebook, root = Path(notebook).resolve(), Path(run_root).resolve()
    source = validate_source(notebook)
    config = source['configuration']
    snapshot = read_json(evidence(root, 'run_configuration.json'))
    require(json_hash({name: snapshot.get(name) for name in CONFIG_NAMES}) == source['configuration_sha256']
            and snapshot.get('configuration_sha256') == source['configuration_sha256']
            and snapshot.get('code_sha256') == source['code_sha256'], 'Effective configuration/source snapshot differs')
    mode = 'final' if config['RUN_OPTIONS'].get('evaluate_test') is True else 'development_only'
    completion = read_json(evidence(root, 'completion.json'))
    require(completion.get('status') == 'complete' and completion.get('mode') == mode
            and completion.get('evaluator_version') == EVALUATOR, 'Incomplete or inconsistent completion mode')
    require(completion.get('code_sha256') == source['code_sha256']
            and completion.get('configuration_sha256') == source['configuration_sha256']
            and completion.get('dataset_sha256') == DATA_SHA256, 'Completion identity differs')
    split, gold, identity, mappings, token_counts = verified_data(evidence(root, 'data/1.0/data/en-US.jsonl'))
    require(read_json(evidence(root, 'dataset_identity.json')) == identity
            and completion.get('dataset_identity') == identity, 'Verified dataset manifests differ')
    require(read_json(evidence(root, 'mappings.json')) == mappings, 'Mappings were not fitted on original train only')
    stages, predictions_by_stage, rows_expected = {}, {}, []
    for stage in ('stage1', 'stage2'):
        stage_config = config['STAGE1_CONFIG' if stage == 'stage1' else 'STAGE2_CONFIG']
        history = read_jsonl(evidence(root, stage + '/history.jsonl'))
        selected = validate_history(history, stage_config['training'], split, token_counts)
        with evidence(root, stage + '/history.csv').open(encoding='utf-8', newline='') as handle:
            history_csv = list(csv.DictReader(handle))
        require(history_csv == [{key: str(value) for key, value in row.items()} for row in history],
                'CSV and JSONL epoch histories differ')
        summary = read_json(evidence(root, stage + '/final_summary.json'))
        trained = read_json(evidence(root, stage + '/training_summary.json'))
        run_metadata = read_json(evidence(root, stage + '/run.json'))
        require(json_hash(run_metadata.get('stage_config')) == json_hash(stage_config)
                and run_metadata.get('runtime') == config['RUNTIME']
                and run_metadata.get('code_sha256') == source['code_sha256'], 'Registered stage settings differ')
        require(summary.get('status') == 'complete' and summary.get('mode') == mode
                and summary.get('test_model_evaluations') == int(mode == 'final'), 'Incorrect stage mode/test-pass count')
        require(completion.get('stages', {}).get(stage) == summary, 'Completion does not reproduce actual stage summary')
        for key, value in selected.items():
            require(summary.get(key) == value and trained.get(key) == value, 'Selected/trained history differs from summary')
        require(summary.get('model_weights_unchanged') is True and summary.get('checkpoint_bytes_unchanged') is True,
                'Missing unchanged-weight/checkpoint evidence')
        require(summary.get('code_sha256') == source['code_sha256']
                and summary.get('dataset_sha256') == DATA_SHA256 and summary.get('evaluator_version') == EVALUATOR,
                'Stage summary source/data/evaluator differs')
        checkpoint = evidence(root, stage + '/' + summary.get('checkpoint', ''))
        checksum = file_hash(checkpoint)
        require(summary.get('checkpoint_sha256') == checksum
                and evidence(root, stage + '/' + summary['checkpoint'] + '.sha256').read_text().strip() == checksum,
                'Checkpoint bytes or checksum changed')
        import torch
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        require(payload.get('purpose') == 'fresh_joint_transformer' and payload.get('stage') == stage,
                'Checkpoint is not a new joint Transformer run')
        require(payload.get('code_sha256') == source['code_sha256'] and payload.get('dataset_sha256') == DATA_SHA256
                and payload.get('evaluator_version') == EVALUATOR and payload.get('mappings') == mappings,
                'Checkpoint source/data/evaluator/train-only mappings differ')
        require(payload.get('stage_config_sha256') == json_hash(stage_config)
                and json_hash(payload.get('stage_config')) == json_hash(stage_config)
                and payload.get('runtime') == config['RUNTIME'], 'Checkpoint actual settings differ')
        require(payload.get('epoch') == selected['selected_epoch']
                and payload.get('selection_score') == selected['selection_score'], 'Checkpoint is not earliest best development epoch')
        counts = summary['parameter_counts']
        total, tensor_sha = state_evidence(payload['model_state'], counts)
        require(payload.get('parameter_counts') == counts and trained.get('parameter_counts') == counts
                and run_metadata.get('parameter_counts') == counts, 'Whole-system count differs across evidence')
        require(tuple(payload['model_state']['embedding.weight'].shape)
                == (len(mappings['word_to_id']), stage_config['model']['word_dim']), 'Task word-embedding table differs')
        if stage_config['model']['use_crf']:
            names = mappings['tag_names']
            expected_start = torch.tensor([not name.startswith('I-') for name in names])
            expected_transitions = torch.tensor([[not following.startswith('I-')
                or previous in {'B-' + following[2:], following} for following in names] for previous in names])
            require('crf.start_allowed' in payload['model_state'] and 'crf.transition_allowed' in payload['model_state']
                    and torch.equal(payload['model_state']['crf.start_allowed'], expected_start)
                    and torch.equal(payload['model_state']['crf.transition_allowed'], expected_transitions),
                    'Stored CRF legality masks differ from the systematic training BIO inventory')
        require(summary.get('model_state_sha256') == tensor_sha, 'Checkpoint learned tensors differ from saved fingerprint')
        reload = read_json(evidence(root, stage + '/fresh_reload.json'))
        require(reload == summary.get('fresh_process_reload') and reload.get('passed') is True
                and reload.get('fresh_python_process') is True and reload.get('ordered_predictions_equal') is True
                and reload.get('test_evaluated') is False and reload.get('num_examples') == COUNTS['dev']
                and reload.get('training_pid') == payload.get('training_pid')
                and reload.get('verification_pid') != payload.get('training_pid')
                and reload.get('checkpoint_sha256') == checksum and reload.get('code_sha256') == source['code_sha256']
                and reload.get('dataset_sha256') == DATA_SHA256, 'Missing exact independent-process reload evidence')
        actual_metrics = {}
        partitions = ('dev', 'test') if mode == 'final' else ('dev',)
        for partition in partitions:
            predictions = read_jsonl(evidence(root, stage + '/' + partition + '_predictions.jsonl'))
            require(all(row.get('intent') in mappings['intent_to_id']
                    and all(slot.get('slot_type') in mappings['slot_types'] for slot in row.get('slots', []))
                    for row in predictions), 'Predicted labels are outside the train inventory')
            metrics = manual_metrics(gold[partition], predictions)
            evaluation = read_json(evidence(root, stage + '/' + partition + '_evaluation.json'))
            check_metrics(evaluation.get('metrics', {}), metrics)
            check_metrics(summary.get(partition, {}).get('metrics', {}), metrics)
            require(evaluation.get('num_examples') == COUNTS[partition]
                    and evaluation.get('num_tokens') == token_counts[partition]
                    and evaluation.get('num_batches') == math.ceil(COUNTS[partition] / stage_config['training']['batch_size'])
                    and evaluation.get('record_ids_sha256') == identity['split_id_sha256'][partition], 'Incomplete final traversal')
            actual_metrics[partition] = metrics
            if partition == 'dev':
                require(predictions == payload.get('dev_predictions'), 'Selected and reloaded ordered predictions differ')
                check_metrics(payload.get('dev_metrics', {}), metrics); check_metrics(reload.get('metrics', {}), metrics)
                predictions_by_stage[stage] = predictions
                require(.5 * (metrics['intent_accuracy'] + metrics['slot_micro_f1']) == selected['selection_score'],
                        'Selected epoch does not reproduce its unrounded development score')
            rows_expected.append({'model_version': stage, 'split': partition,
                'total_parameters': total, 'trainable_parameters': counts['trainable'],
                'intent_accuracy_percent': f'{100 * metrics["intent_accuracy"]:.2f}',
                'slot_micro_f1_percent': f'{100 * metrics["slot_micro_f1"]:.2f}',
                'epochs_trained': selected['epochs_trained'], 'selected_epoch': selected['selected_epoch']})
        validate_mode_artifacts(root, stage, mode, summary, reload, checksum,
                                tensor_sha, selected, source['code_sha256'])
        optimizer = payload.get('optimizer_state', {})
        groups, state = optimizer.get('param_groups', []), optimizer.get('state', {})
        parameter_ids = [key for group in groups for key in group.get('params', [])]
        require(len(groups) == 2 and len(parameter_ids) == len(set(parameter_ids))
                and set(parameter_ids) == set(state), 'Checkpoint lacks one complete joint two-group AdamW state')
        expected_steps = selected['selected_epoch'] * math.ceil(COUNTS['train'] / stage_config['training']['batch_size'])
        require(all(float(record.get('step', -1)) == expected_steps for record in state.values()), 'Selected optimizer-step count differs')
        require(file_hash(root / stage / 'learning_curves.png') in source['inline_png_sha256'], 'Actual learning curve is not retained inline')
        visible_epochs = [int(value) for value in re.findall(rf'{stage} epoch (\d+)/{selected["epochs_trained"]}:', source['visible_text'])]
        require(visible_epochs == list(range(1, selected['epochs_trained'] + 1)), 'Visible epoch logs do not retain full configured training')
        require(completion.get('new_model_test_evaluations', {}).get(stage) == int(mode == 'final'), 'Completion test-pass count differs')
        stages[stage] = {**selected, 'parameter_counts': counts, 'metrics': actual_metrics, 'checkpoint_sha256': checksum}
        del payload
    reporting = read_json(evidence(root, 'results.json'))
    require(reporting.get('evaluator_version') == EVALUATOR and reporting.get('dataset_sha256') == DATA_SHA256
            and reporting.get('rows') == rows_expected, 'Generated result rows differ from actual checkpoints')
    with evidence(root, 'results.csv').open(encoding='utf-8', newline='') as handle:
        csv_rows = list(csv.DictReader(handle))
    require(csv_rows == [{key: str(value) for key, value in row.items()} for row in rows_expected], 'CSV and JSON results differ')
    for stage, details in stages.items():
        for partition, metrics in details['metrics'].items():
            check_metrics(reporting.get('full_precision', {}).get(stage, {}).get(partition, {}), metrics)
    for partition in stages['stage1']['metrics']:
        for metric in ('intent_accuracy', 'slot_micro_f1'):
            delta = 100 * (stages['stage2']['metrics'][partition][metric] - stages['stage1']['metrics'][partition][metric])
            require(abs(reporting.get('stage2_minus_stage1_percentage_points', {}).get(partition, {}).get(metric, float('inf')) - delta) <= 1e-12,
                    'Reported improvement/regression differs from actual metrics')
    winner = max(stages, key=lambda name: stages[name]['selection_score'])
    require(reporting.get('development_winner') == winner, 'System winner was not selected on development only')
    errors = read_json(evidence(root, 'error_cases.json'))
    predicted = {row['record_id']: row for row in predictions_by_stage['stage2']}
    reference = {row['record_id']: row for row in gold['dev']}
    genuine_ids = [key for key in predicted if (lambda value: value['correct_intents'] != 1
                   or value['slot_fp'] or value['slot_fn'])(manual_metrics([reference[key]], [predicted[key]]))]
    cases = errors.get('cases', [])
    require([case.get('record_id') for case in cases] == genuine_ids[:4], 'Error cases do not match genuine revised development failures')
    for case in cases:
        key = case['record_id']
        require(case.get('gold') == reference[key] and case.get('prediction') == predicted[key]
                and case.get('utt') == reference[key]['utt'], 'Error case was changed or fabricated')
        require(isinstance(case.get('explanation'), str) and case['explanation'].startswith('Inference:'),
                'Error explanation must be labelled as an inference')
    return {'schema_version': 1, 'status': 'passed', 'mode': mode,
        'scope': 'Independent source/data/history/checkpoint/prediction audit; development-only is not final-test acceptance',
        'model_construction_performed': False, 'model_forward_performed': False, 'model_training_performed': False,
        'notebook_sha256': source['notebook_sha256'], 'code_sha256': source['code_sha256'],
        'configuration_sha256': source['configuration_sha256'], 'canonical_contract_sha256': source['canonical_contract_sha256'],
        'code_cells': source['code_cells'], 'dataset_sha256': DATA_SHA256, 'counts': COUNTS,
        'stages': stages, 'generated_result_rows': len(rows_expected), 'genuine_development_error_cases': len(cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite a prior acceptance record')
    report = run(args.notebook, args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'status': report['status'], 'mode': report['mode'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()

"""Fresh, auditable training for the two joint Transformer systems.

These ordinary definitions are also inlined in the revised notebook. A stage
always trains its complete configured budget, selects on development only,
reloads in a separate interpreter, and optionally freezes before one test pass.
"""
from __future__ import annotations

import ast
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.comparison_contract import (DATASET_SHA256, EVALUATOR_VERSION, EXPECTED_COUNTS,
    bio_to_character_spans, load_verified_massive, score_exact_spans)
from src.revised_transformer_data import build_data, collate_transformer
from src.revised_transformer_model import JointTransformer, RevisedTransformerConfig, enforce_parameter_cap


def rt_json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def rt_file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rt_write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                         + '\n', encoding='utf-8', newline='\n')
    temporary.replace(path)


def notebook_configuration(path):
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    cells = [cell for cell in document['cells'] if cell['cell_type'] == 'code']
    metadata = document['metadata']['revised_transformer']
    expected, registered = metadata['code_cell_sha256'], metadata['configuration_cell_indices']
    tagged = [i for i, cell in enumerate(cells) if 'configuration' in cell.get('metadata', {}).get('tags', [])]
    if registered != tagged or len(registered) != 1 or len(expected) != len(cells):
        raise ValueError('Expected one registered editable configuration cell')
    configuration = {}
    for index, cell in enumerate(cells):
        source = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
        if index not in registered:
            if hashlib.sha256(source.encode('utf-8')).hexdigest() != expected[index]:
                raise ValueError('Protected notebook implementation changed; rebuild and review it')
            continue
        for statement in ast.parse(source).body:
            if (not isinstance(statement, ast.Assign) or len(statement.targets) != 1
                    or not isinstance(statement.targets[0], ast.Name)):
                raise ValueError('Configuration permits only literal dictionary assignments')
            name = statement.targets[0].id
            if name not in {'RUNTIME', 'STAGE1_CONFIG', 'STAGE2_CONFIG', 'RUN_OPTIONS'} or name in configuration:
                raise ValueError('Unexpected or repeated configuration name')
            try:
                value = ast.literal_eval(statement.value)
            except (ValueError, TypeError) as error:
                raise ValueError('Use literal configuration values') from error
            if not isinstance(value, dict):
                raise ValueError('Every configuration must be a dictionary')
            configuration[name] = value
    if set(configuration) != {'RUNTIME', 'STAGE1_CONFIG', 'STAGE2_CONFIG', 'RUN_OPTIONS'}:
        raise ValueError('All four configuration dictionaries are required')
    return configuration


def notebook_identity(path):
    notebook_configuration(path)
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    sources = [cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
               for cell in document['cells'] if cell['cell_type'] == 'code']
    return rt_json_hash([hashlib.sha256(source.encode('utf-8')).hexdigest() for source in sources])


def capture_configuration(path, runtime, stage1_config, stage2_config, run_options, output_path):
    values = {'RUNTIME': runtime, 'STAGE1_CONFIG': stage1_config,
              'STAGE2_CONFIG': stage2_config, 'RUN_OPTIONS': run_options}
    if notebook_configuration(path) != values:
        raise ValueError('Save configuration edits before restarting and running all cells')
    if type(run_options.get('evaluate_test')) is not bool:
        raise ValueError('evaluate_test must be Boolean')
    snapshot = {**values, 'code_sha256': notebook_identity(path),
                'configuration_sha256': rt_json_hash(values)}
    rt_write_json(output_path, snapshot)
    return snapshot


def rt_seed(runtime):
    seed = runtime.get('seed', 42)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed must be a nonnegative 32-bit integer')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG', ':4096:8') != ':4096:8':
        raise ValueError('Start the kernel with CUBLAS_WORKSPACE_CONFIG=:4096:8')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_default_dtype(torch.float32)


def rt_sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(torch.device(device))


def rt_state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(tensor.dtype).encode('ascii'))
        digest.update(str(tuple(tensor.shape)).encode('ascii'))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _rt_cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _rt_cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rt_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_rt_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def validate_training_config(config):
    required = {'max_epochs', 'batch_size', 'learning_rate', 'embedding_learning_rate',
                'weight_decay', 'gradient_clip_norm', 'warmup_fraction', 'min_lr_ratio'}
    if set(config) != required:
        raise ValueError('Unexpected training configuration fields')
    for name in ('max_epochs', 'batch_size'):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(name + ' must be a positive integer')
    for name in required - {'max_epochs', 'batch_size'}:
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(name + ' must be a finite number')
    if any(config[name] <= 0 for name in ('learning_rate', 'embedding_learning_rate', 'gradient_clip_norm')):
        raise ValueError('Learning rates and gradient clipping must be positive')
    if config['weight_decay'] < 0 or not 0 <= config['warmup_fraction'] < 1 or not 0 <= config['min_lr_ratio'] <= 1:
        raise ValueError('Invalid weight decay or schedule fraction')


def learning_rate_factor(step, total_steps, warmup_fraction, minimum_ratio):
    if not 1 <= step <= total_steps:
        raise ValueError('Schedule step must be inside the full training budget')
    warmup = min(total_steps, int(total_steps * warmup_fraction))
    if warmup and step <= warmup:
        return step / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return minimum_ratio + (1 - minimum_ratio) * .5 * (1 + math.cos(math.pi * progress))


def _rt_loader(bundle, partition, batch_size, seed, shuffle=False):
    rows = bundle['splits'][partition]
    if len(rows) != EXPECTED_COUNTS[partition] or len({row['record_id'] for row in rows}) != len(rows):
        raise ValueError('Expected every original ' + partition + ' record exactly once')
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, drop_last=False,
                      num_workers=0, collate_fn=collate_transformer,
                      generator=torch.Generator().manual_seed(seed))


def _rt_model(bundle, config, device, embedding_matrix=None):
    mappings = bundle['mappings']
    model = JointTransformer(len(mappings['word_to_id']), len(mappings['intent_to_id']),
        mappings['tag_names'], len(mappings['char_to_id']),
        config=RevisedTransformerConfig.from_dict(config), embedding_matrix=embedding_matrix)
    enforce_parameter_cap(model, 20_000_000)
    return model.to(device)


def _rt_epoch(model, loader, bundle, device, *, optimizer=None, schedule=None):
    training = optimizer is not None
    model.train(training)
    intent_sum = slot_sum = 0.0
    examples = tokens = batches = 0
    observed, predictions = [], []
    maximum_gradient_norm = 0.0
    id_to_intent = {value: key for key, value in bundle['mappings']['intent_to_id'].items()}
    tags = bundle['mappings']['tag_names']
    rt_sync(device)
    started = perf_counter()
    with torch.enable_grad() if training else torch.inference_mode():
        for original in loader:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                     for key, value in original.items()}
            if training:
                optimizer.zero_grad(set_to_none=True)
                if schedule:
                    schedule()
            outputs = model(batch['word_ids'], batch['mask'], batch['char_ids'])
            losses = model.loss(outputs, batch['intent_ids'], batch['slot_ids'], batch['mask'])
            if not bool(torch.isfinite(losses['total'])):
                raise ValueError('Nonfinite joint loss')
            if training:
                losses['total'].backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), optimizer._rt_clip, error_if_nonfinite=True)
                maximum_gradient_norm = max(maximum_gradient_norm, float(norm))
                optimizer.step()
            else:
                intents = outputs['intent_logits'].argmax(-1).cpu().tolist()
                decoded = model.decode(outputs, batch['mask']).cpu().tolist()
                for index, row in enumerate(original['rows']):
                    length = len(row['word_ids'])
                    predictions.append({'record_id': row['record_id'], 'utt': row['utt'],
                        'intent': id_to_intent[intents[index]], 'slots': bio_to_character_spans(
                            row['utt'], row['token_offsets'], [tags[tag] for tag in decoded[index][:length]])})
            observed.extend(row['record_id'] for row in original['rows'])
            intent_sum += float(losses['intent_sum'].detach())
            slot_sum += float(losses['slot_sum'].detach())
            examples += losses['num_examples']
            tokens += losses['num_tokens']
            batches += 1
    expected = [row['record_id'] for row in loader.dataset]
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise ValueError('An epoch dropped or repeated original records')
    if not training and observed != expected:
        raise ValueError('Evaluation must retain original source order')
    rt_sync(device)
    result = {'loss': intent_sum / examples + slot_sum / tokens,
        'intent_loss': intent_sum / examples, 'slot_loss': slot_sum / tokens,
        'num_examples': examples, 'num_tokens': tokens, 'num_batches': batches,
        'seconds': perf_counter() - started, 'record_ids_sha256': rt_json_hash(sorted(observed) if training else observed)}
    if training:
        result['maximum_gradient_norm'] = maximum_gradient_norm
    if not training:
        result['predictions'] = predictions
        result['metrics'] = score_exact_spans([bundle['gold_by_id'][key] for key in observed], predictions)
    return result


def _rt_evidence(result):
    return {key: value for key, value in result.items() if key != 'predictions'}


def _rt_predictions(path, predictions):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def _rt_checkpoint(path, payload):
    temporary = Path(path).with_name(Path(path).name + '.part')
    torch.save(payload, temporary)
    temporary.replace(path)
    Path(str(path) + '.sha256').write_text(rt_file_hash(path) + '\n', encoding='ascii', newline='\n')


def _rt_load_checkpoint(path, bundle, device, dataset_sha256, code_sha256):
    path = Path(path)
    if Path(str(path) + '.sha256').read_text().strip() != rt_file_hash(path):
        raise ValueError('Checkpoint bytes differ from its checksum')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload['evaluator_version'] != EVALUATOR_VERSION or payload['dataset_sha256'] != dataset_sha256
            or payload['code_sha256'] != code_sha256 or payload['mappings'] != bundle['mappings']):
        raise ValueError('Checkpoint scorer/data/code/train-only mappings differ')
    if rt_json_hash(payload['stage_config']) != payload['stage_config_sha256']:
        raise ValueError('Checkpoint effective configuration differs from its recorded identity')
    model = _rt_model(bundle, payload['stage_config']['model'], device)
    model.load_state_dict(payload['model_state'], strict=True)
    if enforce_parameter_cap(model) != payload['parameter_counts']:
        raise ValueError('Reconstructed whole-system parameter count changed')
    return model, payload


def verify_checkpoint(checkpoint, dataset_path, code_sha256, device, fresh_process=False):
    original_sha = rt_file_hash(checkpoint)
    raw, manifest = load_verified_massive(dataset_path)
    bundle = build_data(raw)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    rt_seed(payload['runtime'])
    model, payload = _rt_load_checkpoint(checkpoint, bundle, device, manifest['dataset_sha256'], code_sha256)
    loader = _rt_loader(bundle, 'dev', payload['stage_config']['training']['batch_size'], payload['runtime']['seed'])
    dev = _rt_epoch(model, loader, bundle, device)
    if dev['metrics'] != payload['dev_metrics'] or dev['predictions'] != payload['dev_predictions']:
        raise ValueError('Reloaded checkpoint does not reproduce every selected development prediction')
    if rt_file_hash(checkpoint) != original_sha:
        raise ValueError('Verification changed checkpoint bytes')
    report = {'passed': True, 'fresh_python_process': fresh_process,
        'training_pid': payload['training_pid'], 'verification_pid': os.getpid(),
        'ordered_predictions_equal': True, 'num_examples': len(loader.dataset),
        'metrics': dev['metrics'], 'test_evaluated': False, 'checkpoint_sha256': original_sha,
        'code_sha256': code_sha256, 'dataset_sha256': manifest['dataset_sha256'],
        'verified_at_utc': datetime.now(timezone.utc).isoformat()}
    if fresh_process and report['training_pid'] == report['verification_pid']:
        raise ValueError('Independent verification requires a distinct interpreter')
    rt_write_json(Path(checkpoint).parent / 'fresh_reload.json', report)
    return report


def _rt_fresh_verifier(notebook_path, checkpoint, dataset_path, code_sha256, device, timeout):
    if notebook_identity(notebook_path) != code_sha256:
        raise ValueError('Saved notebook changed since this run began')
    document = json.loads(Path(notebook_path).read_text(encoding='utf-8'))
    definitions = [cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
        for cell in document['cells'] if cell['cell_type'] == 'code'
        and 'definitions' in cell.get('metadata', {}).get('tags', [])]
    source = 'from __future__ import annotations\n\n' + '\n\n'.join(
        item.replace('from __future__ import annotations\n', '') for item in definitions)
    source += "\nif __name__ == '__main__':\n    verify_checkpoint(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], fresh_process=True)\n"
    worker = Path(checkpoint).parent / 'fresh_verifier.py'
    worker.write_text(source, encoding='utf-8', newline='\n')
    environment = os.environ.copy()
    environment.update(PYTHONHASHSEED='42', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONIOENCODING='utf-8')
    environment.pop('PYTHONPATH', None)
    completed = subprocess.run([sys.executable, '-B', str(worker.resolve()), str(Path(checkpoint).resolve()),
        str(Path(dataset_path).resolve()), code_sha256, str(device)], cwd=worker.parent,
        env=environment, capture_output=True, text=True, encoding='utf-8', timeout=timeout)
    (worker.parent / 'fresh_verifier_console.txt').write_text(completed.stdout + completed.stderr,
                                                            encoding='utf-8', newline='\n')
    if completed.returncode:
        raise RuntimeError('Independent reload failed: ' + completed.stderr[-3000:])
    return json.loads((worker.parent / 'fresh_reload.json').read_text(encoding='utf-8'))


def _rt_curves(history, output):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, key in zip(axes.flat[:3], ('loss', 'intent_loss', 'slot_loss')):
        for partition in ('train', 'dev'):
            axis.plot([row['epoch'] for row in history], [row[partition + '_' + key] for row in history], label=partition)
        axis.set(xlabel='Epoch', title=key.replace('_', ' ')); axis.legend()
    for key in ('intent_accuracy', 'slot_micro_f1'):
        axes[1, 1].plot([row['epoch'] for row in history], [100 * row['dev_' + key] for row in history], label=key)
    axes[1, 1].set(xlabel='Epoch', ylabel='Percent', title='Development metrics'); axes[1, 1].legend()
    figure.tight_layout(); figure.savefig(output, dpi=140); plt.close(figure)


def run_stage(stage, stage_config, runtime, run_options, bundle, embedding_matrix, device,
              output_dir, dataset_path, code_sha256, notebook_path):
    """Run one complete, exclusive train/select/reload/(optional)test workflow."""
    stage_config, runtime, run_options = copy.deepcopy((stage_config, runtime, run_options))
    validate_training_config(stage_config['training'])
    if type(run_options.get('evaluate_test')) is not bool:
        raise ValueError('evaluate_test must be Boolean')
    if notebook_identity(notebook_path) != code_sha256 or rt_file_hash(dataset_path) != DATASET_SHA256:
        raise ValueError('Reviewed notebook or verified original data changed')
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    rt_seed(runtime)
    device = torch.device(device)
    model = _rt_model(bundle, stage_config['model'], device, embedding_matrix)
    counts = enforce_parameter_cap(model)
    training = stage_config['training']
    loaders = {part: _rt_loader(bundle, part, training['batch_size'], runtime['seed'], part == 'train') for part in ('train', 'dev')}
    word_parameters = list(model.embedding.parameters())
    word_ids = {id(parameter) for parameter in word_parameters}
    others = [parameter for parameter in model.parameters() if id(parameter) not in word_ids]
    optimizer = torch.optim.AdamW([{'params': word_parameters, 'lr': training['embedding_learning_rate']},
        {'params': others, 'lr': training['learning_rate']}], weight_decay=training['weight_decay'])
    optimizer._rt_clip = training['gradient_clip_norm']
    bases = [group['lr'] for group in optimizer.param_groups]
    total_steps, step = len(loaders['train']) * training['max_epochs'], 0

    def schedule():
        nonlocal step
        step += 1
        factor = learning_rate_factor(step, total_steps, training['warmup_fraction'], training['min_lr_ratio'])
        for group, base in zip(optimizer.param_groups, bases):
            group['lr'] = base * factor

    rt_write_json(output / 'run.json', {'stage': stage, 'stage_config': stage_config,
        'runtime': runtime, 'run_options': run_options, 'code_sha256': code_sha256,
        'dataset_sha256': DATASET_SHA256, 'evaluator_version': EVALUATOR_VERSION,
        'parameter_counts': counts, 'training_pid': os.getpid(),
        'created_at_utc': datetime.now(timezone.utc).isoformat()})
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    history, best_score, best_epoch = [], None, None
    checkpoint = output / 'best.pt'
    rt_sync(device); start = perf_counter()
    for epoch in range(1, training['max_epochs'] + 1):
        train = _rt_epoch(model, loaders['train'], bundle, device, optimizer=optimizer, schedule=schedule)
        dev = _rt_epoch(model, loaders['dev'], bundle, device)
        score = .5 * (dev['metrics']['intent_accuracy'] + dev['metrics']['slot_micro_f1'])
        selected = best_score is None or score > best_score
        row = {'epoch': epoch, **{'train_' + key: value for key, value in train.items()},
            **{'dev_' + key: value for key, value in dev.items() if key not in ('predictions', 'metrics')},
            'dev_intent_accuracy': dev['metrics']['intent_accuracy'], 'dev_slot_micro_f1': dev['metrics']['slot_micro_f1'],
            'selection_score': score, 'selected': selected, 'optimizer_step': step,
            'learning_rates': [group['lr'] for group in optimizer.param_groups]}
        history.append(row)
        with (output / 'history.jsonl').open('a', encoding='utf-8', newline='\n') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        with (output / 'history.csv').open('a', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if epoch == 1: writer.writeheader()
            writer.writerow(row)
        if selected:
            payload = {'schema_version': 1, 'stage': stage, 'purpose': 'fresh_joint_transformer',
                'stage_config': copy.deepcopy(stage_config), 'stage_config_sha256': rt_json_hash(stage_config),
                'runtime': copy.deepcopy(runtime), 'mappings': copy.deepcopy(bundle['mappings']),
                'model_state': _rt_cpu_copy(model.state_dict()), 'optimizer_state': _rt_cpu_copy(optimizer.state_dict()),
                'parameter_counts': counts, 'epoch': epoch, 'selection_score': score,
                'dev_metrics': dev['metrics'], 'dev_predictions': dev['predictions'],
                'code_sha256': code_sha256, 'dataset_sha256': DATASET_SHA256,
                'evaluator_version': EVALUATOR_VERSION, 'training_pid': os.getpid()}
            _rt_checkpoint(checkpoint, payload)
            best_score, best_epoch = score, epoch
        print(f'{stage} epoch {epoch:02d}/{training["max_epochs"]}: train={train["loss"]:.5f}; dev={dev["loss"]:.5f}; '
              f'intent={100 * dev["metrics"]["intent_accuracy"]:.2f}; slot={100 * dev["metrics"]["slot_micro_f1"]:.2f}; '
              f'mean={100 * score:.2f}; selected={selected}', flush=True)
    rt_sync(device); seconds = perf_counter() - start
    if len(history) != training['max_epochs'] or step != total_steps:
        raise ValueError('The complete registered training budget did not finish')
    winner = max(history, key=lambda row: row['selection_score'])
    if winner['epoch'] != best_epoch or winner['selection_score'] != best_score:
        raise ValueError('Checkpoint selection differs from the earliest development maximum')
    training_summary = {'status': 'trained_not_tested', 'epochs_trained': len(history), 'selected_epoch': best_epoch,
        'selection_score': best_score, 'training_workflow_seconds': seconds, 'parameter_counts': counts,
        'timing_scope': 'Training, epoch development scoring, checkpoints and history; excludes setup, curves, fresh reload and final evaluation',
        'peak_allocated_cuda_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}
    rt_write_json(output / 'training_summary.json', training_summary)
    _rt_curves(history, output / 'learning_curves.png')
    del optimizer, model
    if device.type == 'cuda': torch.cuda.empty_cache()
    reload = _rt_fresh_verifier(notebook_path, checkpoint, dataset_path, code_sha256, device,
                               run_options.get('verification_timeout_seconds', 600))
    model, payload = _rt_load_checkpoint(checkpoint, bundle, device, DATASET_SHA256, code_sha256)
    dev = _rt_epoch(model, loaders['dev'], bundle, device)
    if dev['predictions'] != payload['dev_predictions'] or dev['metrics'] != payload['dev_metrics']:
        raise ValueError('Parent reload differs from independently verified development checkpoint')
    _rt_predictions(output / 'dev_predictions.jsonl', dev['predictions'])
    rt_write_json(output / 'dev_evaluation.json', _rt_evidence(dev))
    before, checkpoint_before = rt_state_hash(model), rt_file_hash(checkpoint)
    test = None
    if run_options['evaluate_test']:
        freeze = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'selected_epoch': best_epoch,
            'checkpoint_sha256': checkpoint_before, 'model_state_sha256': before,
            'history_sha256': rt_file_hash(output / 'history.jsonl'), 'test_model_evaluations': 0,
            'code_sha256': code_sha256, 'dataset_sha256': DATASET_SHA256, 'tuning_closed': True}
        rt_write_json(output / 'freeze_manifest.json', freeze)
        rt_write_json(output / 'test_started.json', {'started_at_utc': datetime.now(timezone.utc).isoformat(),
            'freeze_sha256': rt_file_hash(output / 'freeze_manifest.json'), 'test_model_evaluations': 1})
        test = _rt_epoch(model, _rt_loader(bundle, 'test', training['batch_size'], runtime['seed']), bundle, device)
        _rt_predictions(output / 'test_predictions.jsonl', test['predictions'])
        rt_write_json(output / 'test_evaluation.json', _rt_evidence(test))
    if before != rt_state_hash(model) or checkpoint_before != rt_file_hash(checkpoint):
        raise ValueError('Reload/final evaluation changed learned weights or checkpoint bytes')
    summary = {'status': 'complete', 'stage': stage, 'mode': 'final' if test is not None else 'development_only',
        **training_summary, 'status': 'complete', 'dev': _rt_evidence(dev), 'test': _rt_evidence(test) if test else None,
        'checkpoint': checkpoint.name, 'checkpoint_sha256': checkpoint_before, 'model_state_sha256': before,
        'fresh_process_reload': reload, 'test_model_evaluations': int(test is not None),
        'model_weights_unchanged': True, 'checkpoint_bytes_unchanged': True,
        'code_sha256': code_sha256, 'dataset_sha256': DATASET_SHA256, 'evaluator_version': EVALUATOR_VERSION,
        'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    rt_write_json(output / 'final_summary.json', summary)
    del model
    if device.type == 'cuda': torch.cuda.empty_cache()
    return summary


def report_results(summaries, output_root, bundle):
    output = Path(output_root)
    rows, full_precision = [], {}
    for stage, summary in summaries.items():
        full_precision[stage] = {}
        for split in ('dev', 'test'):
            if summary.get(split) is None: continue
            metrics = summary[split]['metrics']; full_precision[stage][split] = metrics
            rows.append({'model_version': stage, 'split': split,
                'total_parameters': summary['parameter_counts']['total'],
                'trainable_parameters': summary['parameter_counts']['trainable'],
                'intent_accuracy_percent': f'{100 * metrics["intent_accuracy"]:.2f}',
                'slot_micro_f1_percent': f'{100 * metrics["slot_micro_f1"]:.2f}',
                'epochs_trained': summary['epochs_trained'], 'selected_epoch': summary['selected_epoch']})
    report = {'evaluator_version': EVALUATOR_VERSION, 'dataset_sha256': DATASET_SHA256,
              'rows': rows, 'full_precision': full_precision}
    if len(summaries) == 2:
        first, last = list(summaries)
        report['stage2_minus_stage1_percentage_points'] = {
            split: {key: 100 * (full_precision[last][split][key] - full_precision[first][split][key])
                    for key in ('intent_accuracy', 'slot_micro_f1')}
            for split in full_precision[first] if split in full_precision[last]}
        report['development_winner'] = max(summaries, key=lambda stage:
            .5 * (full_precision[stage]['dev']['intent_accuracy'] + full_precision[stage]['dev']['slot_micro_f1']))
    rt_write_json(output / 'results.json', report)
    with (output / 'results.csv').open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    stage = list(summaries)[-1]
    predicted = [json.loads(line) for line in (output / stage / 'dev_predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    cases = []
    for prediction in predicted:
        gold = bundle['gold_by_id'][prediction['record_id']]
        score = score_exact_spans([gold], [prediction])
        if score['correct_intents'] != 1 or score['slot_fp'] or score['slot_fn']:
            cases.append({'record_id': prediction['record_id'], 'utt': gold['utt'],
                'gold': gold, 'prediction': prediction,
                'explanation': 'Inference: inspect training support, unfamiliar spelling and boundary/type/context ambiguity; one example does not prove a cause.'})
        if len(cases) == 4: break
    rt_write_json(output / 'error_cases.json', {'source': 'Actual revised development predictions',
        'cases': cases, 'note': 'All available real cases are retained if fewer than two; no errors are invented.'})
    return report

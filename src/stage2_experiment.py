"""Stage 2 experiments with original splits and the frozen shared scorer.

The Stage 1 sources are deliberately reused without modification. Character
features derive only from utterance text; all candidate training is fresh.
"""
from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from src.acquisition import file_sha256
from src.data import EXPECTED_COUNTS, FittedMappings, decode_bio, write_predictions
from src.experiment import _load_context
from src.final_evaluation import _result_data, _validate_complete_result
from src.metrics import evaluate_predictions
from src.models import enforce_parameter_cap
from src.runtime import configuration_hash, dataloader_generator, is_improvement, selection_score
from src.schema import IntentSlotPrediction
from src.stage2_model import Stage2Config, JointStage2, collate_stage2, fit_char_vocab
from src.training import _atomic_json, _synchronize, _summarize


SOURCE_FILES = (
    'src/stage2_model.py', 'src/stage2_experiment.py', 'scripts/run_stage2.py',
    'src/models.py', 'src/training.py', 'src/experiment.py', 'src/final_evaluation.py',
    'src/data.py', 'src/schema.py', 'src/metrics.py', 'src/runtime.py',
    'src/audit.py', 'src/acquisition.py',
)


def load_context(root, protocol, config, device):
    if config.get('schema_version') != 1 or config.get('stage') != 'stage2':
        raise ValueError('Expected a Stage 2 configuration')
    architecture = Stage2Config.from_dict(config['model'])
    if any(config['training'].get(key) != 1.0 for key in ('intent_loss_weight', 'slot_loss_weight')):
        raise ValueError('These controlled experiments use equal positive task loss weights')
    baseline_shape = {key: getattr(architecture, key) for key in ('embedding_dim', 'hidden_dim', 'num_layers', 'dropout')}
    validation_config = {**config, 'stage': 'stage1', 'model': baseline_shape}
    device, mappings, prepared, dataset_hash = _load_context(root, protocol, validation_config, device)
    chars = fit_char_vocab(prepared['train'])
    membership = json.loads((root / 'artifacts/phase2/split_membership.json').read_text(encoding='utf-8'))
    for part in ('train', 'dev'):
        if [x.example.record_id for x in prepared[part]] != membership['record_ids'][part]:
            raise ValueError('Original partition membership changed')
    return device, mappings, prepared, dataset_hash, chars


def make_loader(prepared, partition, batch_size, seed, char_to_id):
    return DataLoader(prepared, batch_size=batch_size, shuffle=partition == 'train', drop_last=False,
                      num_workers=0, generator=dataloader_generator(seed),
                      collate_fn=partial(collate_stage2, char_to_id=char_to_id))


def new_model(mappings, chars, config, device):
    tags = [name for name, index in sorted(mappings.tag_to_id.items(), key=lambda pair: pair[1])]
    return JointStage2(len(mappings.token_to_id), len(mappings.intent_to_id), tags, len(chars),
                       Stage2Config.from_dict(config['model'])).to(device)


def _to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) and key != 'lengths' else value
            for key, value in batch.items()}


def _forward(model, batch):
    return model(batch['input_ids'], batch['lengths'], batch['attention_mask'], batch['char_ids'])


def _epoch(model, loader, mappings, device, *, optimizer=None, grad_clip=5.0, allow_test=False):
    training = optimizer is not None
    model.train(training)
    device = torch.device(device)
    _synchronize(device)
    start = perf_counter()
    sums = [0.0, 0.0]
    examples = tokens = batches = 0
    max_norm = 0.0
    ids, predictions, gold, partitions = [], [], [], set()
    inverse_intents = {value: key for key, value in mappings.intent_to_id.items()}
    inverse_tags = {value: key for key, value in mappings.tag_to_id.items()}
    with torch.enable_grad() if training else torch.inference_mode():
        for original in loader:
            batch_parts = {item.partition for item in original['examples']}
            if training and batch_parts != {'train'}:
                raise ValueError('Updates require original training records only')
            if 'test' in batch_parts and (training or not allow_test or len(loader.dataset) != EXPECTED_COUNTS['test'] or loader.drop_last):
                raise ValueError('Test requires explicitly authorized complete final evaluation')
            partitions.update(batch_parts)
            batch = _to_device(original, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = _forward(model, batch)
            loss = model.loss(outputs, batch)
            if not bool(torch.isfinite(loss.total)):
                raise ValueError('Nonfinite joint loss')
            if training:
                loss.total.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                max_norm = max(max_norm, float(norm))
                optimizer.step()
            else:
                intents = outputs['intent_logits'].argmax(-1).cpu().tolist()
                paths = model.decode(outputs, batch['attention_mask']).cpu().tolist()
                for row, example in enumerate(batch['examples']):
                    tags = tuple(inverse_tags[x] for x in paths[row][:int(batch['lengths'][row])])
                    spans = decode_bio(example.utt, batch['tokens'][row], tags)
                    predictions.append(IntentSlotPrediction(example.record_id, inverse_intents[intents[row]], spans))
                    gold.append(example.gold_prediction())
            sums[0] += float(loss.intent_sum.detach())
            sums[1] += float(loss.slot_sum.detach())
            examples += loss.num_examples
            tokens += loss.num_tokens
            batches += 1
            ids.extend(batch['record_ids'])
    if len(partitions) != 1:
        raise ValueError('An epoch must contain exactly one original partition')
    if 'test' in partitions and examples != EXPECTED_COUNTS['test']:
        raise ValueError('Incomplete test traversal')
    _synchronize(device)
    result = _summarize(tuple(sums), examples, tokens, batches, perf_counter() - start, 1.0, 1.0, ids, False)
    result.update(partitions=sorted(partitions), max_batches=None, max_gradient_norm=max_norm)
    if not training:
        result.update(metrics=evaluate_predictions(gold, predictions), predictions=tuple(predictions))
    return result


def evaluate(model, loader, mappings, device, allow_test=False):
    return _epoch(model, loader, mappings, device, allow_test=allow_test)


def _save(path, model, optimizer, mappings, chars, config, protocol, epoch, dev, sources, data_hash, purpose):
    payload = {'schema_version': 1, 'model_class': 'JointStage2', 'purpose': purpose,
               'stage_config': config, 'protocol': protocol, 'mappings': mappings.to_dict(), 'char_to_id': chars,
               'model_state': {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
               'optimizer_state': copy.deepcopy(optimizer.state_dict()), 'epoch': epoch,
               'selection_score': selection_score(dev['metrics'].intent_accuracy, dev['metrics'].slot_micro_f1),
               'validation': _result_data(dev), 'validation_predictions': [p.to_dict() for p in dev['predictions']],
               'parameter_counts': enforce_parameter_cap(model, protocol['limits']['total_parameters']),
               'source_hashes': sources, 'data_sha256': data_hash}
    for key in ('stage_config', 'protocol', 'mappings', 'char_to_id'):
        payload[key + '_sha256'] = configuration_hash(payload[key])
    temporary = path.with_suffix('.part')
    torch.save(payload, temporary)
    temporary.replace(path)
    Path(str(path) + '.sha256').write_text(file_sha256(path) + '\n', encoding='ascii')


def load_run(root, checkpoint, device):
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_relative_to(root.resolve()):
        raise ValueError('Checkpoint outside project')
    if file_sha256(checkpoint) != Path(str(checkpoint) + '.sha256').read_text(encoding='ascii').strip():
        raise ValueError('Checkpoint checksum mismatch')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload.get('schema_version') != 1 or payload.get('model_class') != 'JointStage2':
        raise ValueError('Unsupported Stage 2 checkpoint')
    for key in ('stage_config', 'protocol', 'mappings', 'char_to_id'):
        if configuration_hash(payload[key]) != payload[key + '_sha256']:
            raise ValueError('Checkpoint configuration identity mismatch: ' + key)
    for name, digest in payload['source_hashes'].items():
        if file_sha256(root / name) != digest:
            raise ValueError('Trained source identity changed: ' + name)
    device, mappings, _, data_hash, chars = load_context(root, payload['protocol'], payload['stage_config'], device)
    if mappings.to_dict() != payload['mappings'] or chars != payload['char_to_id'] or data_hash != payload['data_sha256']:
        raise ValueError('Checkpoint training-only mappings or dataset identity changed')
    model = new_model(mappings, chars, payload['stage_config'], device)
    model.load_state_dict(payload['model_state'], strict=True)
    model.eval()
    if enforce_parameter_cap(model, payload['protocol']['limits']['total_parameters']) != payload['parameter_counts']:
        raise ValueError('Checkpoint parameter count changed')
    return model, mappings, chars, payload


def verify(root, checkpoint, device):
    model, mappings, chars, payload = load_run(root, checkpoint, device)
    device, _, prepared, _, _ = load_context(root, payload['protocol'], payload['stage_config'], device)
    loader = make_loader(prepared['dev'], 'dev', payload['stage_config']['training']['batch_size'],
                         payload['protocol']['runtime']['seed'], chars)
    dev = evaluate(model, loader, mappings, device)
    _validate_complete_result(dev, [x.example.record_id for x in prepared['dev']], 'dev')
    if [x.to_dict() for x in dev['predictions']] != payload['validation_predictions']:
        raise ValueError('Fresh-process ordered development predictions changed')
    score = selection_score(dev['metrics'].intent_accuracy, dev['metrics'].slot_micro_f1)
    difference = abs(score - payload['selection_score'])
    if difference > payload['protocol']['selection']['reload_score_atol']:
        raise ValueError('Reloaded development selection score changed')
    report = {'passed': True, 'fresh_python_process': True, 'ordered_predictions_equal': True,
              'selection_score': score, 'absolute_score_difference': difference, 'num_examples': dev['num_examples'],
              'metrics': dev['metrics'].to_dict(), 'test_evaluated': False}
    _atomic_json(Path(checkpoint).parent / 'reload_check.json', report)
    return report


def plot_curves(history, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, key in zip(axes.flat[:3], ('loss', 'intent_loss', 'slot_loss')):
        for split in ('train', 'dev'):
            axis.plot([r['epoch'] for r in history], [r[split + '_' + key] for r in history], label=split)
        axis.set(title=key.replace('_', ' '), xlabel='Epoch', ylabel='Loss')
        axis.legend()
    for key, label in (('dev_intent_accuracy', 'Intent'), ('dev_slot_micro_f1', 'Slot F1')):
        axes[1, 1].plot([r['epoch'] for r in history], [100*r[key] for r in history], label=label)
    axes[1, 1].set(title='Development metrics', xlabel='Epoch', ylabel='Percent')
    axes[1, 1].legend()
    figure.suptitle('Stage 2: intent CE and slot CE or CRF NLL per real token')
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def train(root, protocol, config, device):
    if (root / 'artifacts/phase5/final_stage2/freeze_manifest.json').exists():
        raise ValueError('Stage 2 tuning is closed after final freeze')
    device, mappings, prepared, data_hash, chars = load_context(root, protocol, config, device)
    # The persisted bounded plan must precede every reportable candidate.
    plan_path = root / 'artifacts/phase5/experiment_plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    registered = [item for item in plan['candidates'] if item['config_sha256'] == configuration_hash(config)]
    if len(registered) != 1:
        raise ValueError('Candidate not uniquely registered before training')
    candidate = registered[0]
    if (file_sha256(root / candidate['configuration']) != candidate['configuration_sha256']
            or config['training']['max_epochs'] != candidate['max_epochs']
            or file_sha256(root / plan['plan_document']) != plan['plan_document_sha256']):
        raise ValueError('Registered configuration, epoch budget or plan document changed')
    for previous in (root / 'artifacts/runs').glob('stage2_*/run.json'):
        existing = json.loads(previous.read_text(encoding='utf-8'))
        if (existing.get('registered_plan_sha256') == file_sha256(plan_path)
                and existing.get('candidate_id') == candidate['candidate_id']):
            raise ValueError('Candidate already attempted; inspect prior evidence before any explicit recovery')
    config_hash = configuration_hash(config)
    output = root / 'artifacts/runs' / ('stage2_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + config_hash[:8])
    output.mkdir(parents=True, exist_ok=False)
    sources = {name: file_sha256(root / name) for name in SOURCE_FILES}
    metadata = {'purpose': 'experiment', 'stage_config': config, 'protocol': protocol,
                'dataset_sha256': data_hash, 'source_hashes': sources, 'config_sha256': config_hash,
                'registered_plan_sha256': file_sha256(plan_path), 'candidate_id': registered[0]['candidate_id'],
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'environment': {'python': sys.version, 'torch': str(torch.__version__), 'torch_cuda': torch.version.cuda,
                    'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                    'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}}
    for name, value in (('run.json', metadata), ('protocol.json', protocol), ('stage_config.json', config)):
        _atomic_json(output / name, value)
    model = new_model(mappings, chars, config, device)
    counts = enforce_parameter_cap(model, protocol['limits']['total_parameters'])
    training = config['training']
    loaders = {part: make_loader(prepared[part], part, training['batch_size'], protocol['runtime']['seed'], chars) for part in ('train', 'dev')}
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'], weight_decay=training['weight_decay'])
    print(f'RUN_DIRECTORY={output}', flush=True)
    print(f'Parameters total/trainable: {counts}', flush=True)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    history, best_score, best_epoch = [], None, None
    checkpoint = output / 'best.pt'
    _synchronize(device)
    start = perf_counter()
    for epoch in range(1, training['max_epochs'] + 1):
        training_result = _epoch(model, loaders['train'], mappings, device, optimizer=optimizer,
                                 grad_clip=training['gradient_clip_norm'])
        dev = evaluate(model, loaders['dev'], mappings, device)
        for result, part in ((training_result, 'train'), (dev, 'dev')):
            expected = [x.example.record_id for x in prepared[part]]
            if result['num_examples'] != EXPECTED_COUNTS[part] or set(result['record_ids']) != set(expected):
                raise ValueError('Epoch did not visit every original record: ' + part)
        _validate_complete_result(dev, [x.example.record_id for x in prepared['dev']], 'dev')
        score = selection_score(dev['metrics'].intent_accuracy, dev['metrics'].slot_micro_f1)
        selected = is_improvement(score, best_score)
        if selected:
            _save(checkpoint, model, optimizer, mappings, chars, config, protocol, epoch, dev, sources, data_hash, 'experiment')
            write_predictions(output / 'best_dev_predictions.jsonl', dev['predictions'])
            best_score, best_epoch = score, epoch
        keys = ('loss', 'intent_loss', 'slot_loss', 'num_examples', 'num_tokens', 'seconds')
        row = {'epoch': epoch, 'purpose': 'experiment', **{'train_' + k: training_result[k] for k in keys},
               **{'dev_' + k: dev[k] for k in keys}, 'dev_intent_accuracy': dev['metrics'].intent_accuracy,
               'dev_slot_micro_f1': dev['metrics'].slot_micro_f1, 'selection_score': score, 'selected_checkpoint': selected,
               'max_gradient_norm': training_result['max_gradient_norm'],
               'train_record_ids_sha256': configuration_hash(sorted(training_result['record_ids'])),
               'dev_record_ids_sha256': configuration_hash(dev['record_ids'])}
        history.append(row)
        with (output / 'history.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        with (output / 'history.csv').open('a', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        print(f"Epoch {epoch}: train={row['train_loss']:.5f} dev={row['dev_loss']:.5f} intent={100*row['dev_intent_accuracy']:.2f}% slots={100*row['dev_slot_micro_f1']:.2f}% mean={100*score:.2f}%", flush=True)
    _synchronize(device)
    summary = {'purpose': 'experiment', 'epochs_trained': len(history), 'best_epoch': best_epoch,
               'best_score': best_score, 'checkpoint': str(checkpoint), 'output_directory': str(output),
               'training_seconds': perf_counter() - start, 'parameter_counts': counts, 'max_train_batches': None,
               'max_dev_batches': None, 'test_evaluated': False,
               'peak_cuda_memory_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}
    _atomic_json(output / 'training_summary.json', summary)
    plot_curves(history, output / 'learning_curves.png')
    command = [sys.executable, str(root / 'scripts/run_stage2.py'), '--mode', 'verify', '--project-root', str(root),
               '--checkpoint', str(checkpoint), '--device', str(device)]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True)
    (output / 'fresh_reload_process.txt').write_text(completed.stdout + completed.stderr, encoding='utf-8')
    if completed.returncode:
        raise RuntimeError('Fresh-process reload failed: ' + str(output))
    summary['fresh_process_reload'] = json.loads((output / 'reload_check.json').read_text(encoding='utf-8'))
    _atomic_json(output / 'experiment_summary.json', summary)
    return summary


def diagnostic(root, protocol, config, device):
    device, mappings, prepared, _, chars = load_context(root, protocol, config, device)
    model = new_model(mappings, chars, config, device)
    counts = enforce_parameter_cap(model, protocol['limits']['total_parameters'])
    batch = _to_device(next(iter(make_loader(prepared['train'], 'train', 32, protocol['runtime']['seed'], chars))), device)
    model.eval()
    with torch.inference_mode():
        initial_outputs = _forward(model, batch)
        initial = model.loss(initial_outputs, batch)
        padded = dict(batch)
        for key in ('input_ids', 'attention_mask'):
            padded[key] = torch.nn.functional.pad(batch[key], (0, 4))
        padded['char_ids'] = torch.nn.functional.pad(batch['char_ids'], (0, 4, 0, 4))
        result = _forward(model, padded)
        torch.testing.assert_close(initial_outputs['intent_logits'], result['intent_logits'], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(initial_outputs['slot_logits'], result['slot_logits'][:, :-4], atol=1e-6, rtol=1e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=config['training']['weight_decay'])
    trajectory = []
    for step in range(100):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(_forward(model, batch), batch)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        if step in (0, 9, 24, 49, 99):
            trajectory.append({'step': step+1, 'intent_loss': float(loss.intent.detach()), 'slot_loss': float(loss.slot.detach())})
    model.eval()
    with torch.inference_mode():
        final_outputs = _forward(model, batch)
        final = model.loss(final_outputs, batch)
        restored = new_model(mappings, chars, config, device)
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        restored.eval()
        for key, value in _forward(restored, batch).items():
            torch.testing.assert_close(value, final_outputs[key], atol=0, rtol=0)
    if not (final.intent < initial.intent / 2 and final.slot < initial.slot / 2):
        raise ValueError('Both task losses must decrease on the original training minibatch')
    report = {'passed': True, 'purpose': 'diagnostic', 'reportable_experiment': False, 'test_evaluated': False,
              'parameter_counts': counts, 'stage_config': config, 'padding_invariance': True,
              'state_roundtrip_logits_exact': True, 'steps': 100, 'record_ids': batch['record_ids'],
              'initial': {'intent_loss': float(initial.intent), 'slot_loss': float(initial.slot)},
              'final': {'intent_loss': float(final.intent), 'slot_loss': float(final.slot)}, 'trajectory': trajectory}
    directory = root / 'artifacts/phase5/diagnostics'
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_json(directory / (configuration_hash(config)[:8] + '.json'), report)
    return report

"""Visible notebook-local execution for the two previously selected systems.

This module is copied into the notebook as ordinary definitions, alongside the
unchanged data, scoring, model and epoch algorithms. It needs no project audit,
historic checkpoint or candidate-selection file at execution time. Every run is
fresh and exclusive; generated files are evidence of that notebook execution.
"""
from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from src.acquisition import file_sha256
from src.comparison_contract import EVALUATOR_VERSION
from src.data import EXPECTED_COUNTS, FittedMappings, collate_examples, fit_mappings, write_predictions
from src.models import BaselineConfig, JointBiLSTM, enforce_parameter_cap
from src.runtime import configuration_hash, configure_reproducibility, dataloader_generator, is_improvement, selection_score
from src.stage2_model import Stage2Config, fit_char_vocab
from src.stage2_experiment import make_loader as s2_make_loader, new_model as s2_new_model, _epoch as s2_epoch, evaluate as s2_evaluate
from src.training import train_epoch, evaluate_model


def nb_now():
    return datetime.now(timezone.utc).isoformat()


def nb_write_json(path, value):
    """Write UTF-8 JSON atomically, rejecting NaN and Infinity."""
    path = Path(path)
    temporary = path.with_name(path.name + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def nb_sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(torch.device(device))


def nb_cpu_copy(value):
    """Snapshot all model/optimizer tensors, without aliases to live state."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: nb_cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [nb_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(nb_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def nb_state_sha256(model):
    """Hash state tensor names, shapes, dtypes and exact contiguous CPU bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(tensor.dtype).encode('ascii'))
        digest.update(str(tuple(tensor.shape)).encode('ascii'))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def nb_validate_config(stage, stage_config, runtime):
    """Validate supported editable settings without changing the data/scorer."""
    if stage not in ('stage1', 'stage2') or stage_config.get('stage') != stage:
        raise ValueError('Unknown or inconsistent model stage')
    training = stage_config.get('training', {})
    required = {'optimizer', 'learning_rate', 'weight_decay', 'batch_size', 'max_epochs',
                'intent_loss_weight', 'slot_loss_weight', 'gradient_clip_norm'}
    if stage_config.get('schema_version') != 1 or set(training) != required:
        raise ValueError('Unsupported training configuration fields')
    if training['optimizer'] != 'AdamW' or any(type(training[key]) is bool or training[key] != 1.0
            for key in ('intent_loss_weight', 'slot_loss_weight')):
        raise ValueError('This workflow supports AdamW and equal task weights of 1.0')
    for key in ('batch_size', 'max_epochs'):
        if type(training[key]) is not int or training[key] <= 0:
            raise ValueError(key + ' must be a positive integer')
    for key in ('learning_rate', 'weight_decay', 'gradient_clip_norm'):
        value = training[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or value < 0 or (key != 'weight_decay' and value == 0)):
            raise ValueError(key + ' must be finite and positive (weight decay may be zero)')
    if stage == 'stage1':
        BaselineConfig.from_dict(stage_config['model'])
    else:
        Stage2Config.from_dict(stage_config['model'])
    if type(runtime.get('seed')) is not int or not 0 <= runtime['seed'] < 2**32:
        raise ValueError('Seed must be a 32-bit nonnegative integer')
    for key, expected in {'num_workers': 0, 'deterministic_algorithms': True,
                          'cudnn_deterministic': True, 'cudnn_benchmark': False,
                          'allow_tf32': False, 'cublas_workspace_config': ':4096:8',
                          'precision': 'float32'}.items():
        if runtime.get(key) != expected:
            raise ValueError('Deterministic runtime changed: ' + key)


def nb_validate_prepared(prepared, partition):
    if len(prepared) != EXPECTED_COUNTS[partition] or any(item.example.partition != partition for item in prepared):
        raise ValueError('Expected every record in the original ' + partition + ' split')
    ids = [item.example.record_id for item in prepared]
    if len(set(ids)) != len(ids):
        raise ValueError('Repeated original record identity')
    return ids


def nb_validate_result(result, prepared, partition):
    """Assert counts, real-token denominators and original record traversal."""
    expected = nb_validate_prepared(prepared, partition)
    observed = result['record_ids']
    if result['num_examples'] != len(expected) or len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise ValueError('Incomplete or repeated original ' + partition + ' traversal')
    if result['num_tokens'] != sum(len(item.tokens) for item in prepared) or result.get('limited', False):
        raise ValueError('Incorrect real-token denominator or limited traversal')
    if any(not math.isfinite(result[key]) for key in ('loss', 'intent_loss', 'slot_loss')):
        raise ValueError('Nonfinite epoch losses')
    if partition != 'train':
        if observed != expected or result.get('partitions') != [partition]:
            raise ValueError('Evaluation order differs from the original split')
        if [prediction.record_id for prediction in result['predictions']] != expected:
            raise ValueError('Ordered evaluation predictions do not cover the original split')
        if result['metrics'].num_examples != len(expected):
            raise ValueError('Scoring denominator does not match traversal')


def nb_make_loader(stage, prepared, partition, batch_size, seed, char_to_id):
    if stage == 'stage2':
        return s2_make_loader(prepared, partition, batch_size, seed, char_to_id)
    return DataLoader(prepared, batch_size=batch_size, shuffle=partition == 'train',
                      drop_last=False, num_workers=0, collate_fn=collate_examples,
                      generator=dataloader_generator(seed))


def nb_new_model(stage, mappings, char_to_id, stage_config, device):
    if stage == 'stage2':
        return s2_new_model(mappings, char_to_id, stage_config, device)
    return JointBiLSTM(len(mappings.token_to_id), len(mappings.intent_to_id), len(mappings.tag_to_id),
                       BaselineConfig.from_dict(stage_config['model'])).to(device)


def nb_evaluate(stage, model, loader, mappings, device, allow_test=False):
    if stage == 'stage2':
        return s2_evaluate(model, loader, mappings, device, allow_test=allow_test)
    return evaluate_model(model, loader, mappings, device, allow_test=allow_test)


def nb_result_data(result):
    keys = ('loss', 'intent_loss', 'slot_loss', 'num_examples', 'num_tokens', 'num_batches',
            'seconds', 'partitions', 'limited', 'max_batches')
    return {**{key: result[key] for key in keys if key in result},
            'record_ids': list(result['record_ids']), 'metrics': result['metrics'].to_dict()}


def nb_save_checkpoint(path, stage, model, optimizer, mappings, char_to_id, stage_config,
                       runtime, epoch, validation, data_sha256, code_sha256):
    """Save selected weights and an independent CPU snapshot of AdamW state."""
    payload = {'schema_version': 1, 'purpose': 'notebook_fixed_configuration_reproduction',
               'evaluator_version': EVALUATOR_VERSION,
               'stage': stage, 'model_class': type(model).__name__, 'created_at_utc': nb_now(),
               'training_pid': os.getpid(), 'stage_config': copy.deepcopy(stage_config),
               'runtime': copy.deepcopy(runtime), 'mappings': mappings.to_dict(),
               'char_to_id': dict(char_to_id), 'epoch': epoch,
               'selection_score': selection_score(validation['metrics'].intent_accuracy, validation['metrics'].slot_micro_f1),
               'parameter_counts': enforce_parameter_cap(model, 20_000_000),
               'model_state': nb_cpu_copy(model.state_dict()),
               'optimizer_state': nb_cpu_copy(optimizer.state_dict()), 'optimizer_class': 'AdamW',
               'validation': nb_result_data(validation),
               'validation_predictions': [prediction.to_dict() for prediction in validation['predictions']],
               'data_sha256': data_sha256, 'code_sha256': code_sha256,
               'resume_support': 'Optimizer snapshot retained; exact RNG/loader resumption is not implemented.'}
    for key in ('stage_config', 'runtime', 'mappings', 'char_to_id'):
        payload[key + '_sha256'] = configuration_hash(payload[key])
    path = Path(path)
    temporary = path.with_name(path.name + '.part')
    torch.save(payload, temporary)
    temporary.replace(path)
    Path(str(path) + '.sha256').write_text(file_sha256(path) + '\n', encoding='ascii')


def nb_load_checkpoint(checkpoint, mappings, char_to_id, device, data_sha256, code_sha256):
    checkpoint = Path(checkpoint)
    sidecar = Path(str(checkpoint) + '.sha256')
    if not sidecar.is_file() or sidecar.read_text(encoding='ascii').strip() != file_sha256(checkpoint):
        raise ValueError('Checkpoint checksum is missing or changed')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload.get('evaluator_version') != EVALUATOR_VERSION:
        raise ValueError('Checkpoint belongs to a different evaluator; fresh development selection is required')
    if payload.get('schema_version') != 1 or payload.get('purpose') != 'notebook_fixed_configuration_reproduction':
        raise ValueError('Expected a fresh notebook checkpoint')
    nb_validate_config(payload['stage'], payload['stage_config'], payload['runtime'])
    for key in ('stage_config', 'runtime', 'mappings', 'char_to_id'):
        if configuration_hash(payload[key]) != payload[key + '_sha256']:
            raise ValueError('Checkpoint identity mismatch: ' + key)
    expected_chars = dict(char_to_id or {}) if payload['stage'] == 'stage2' else {}
    if payload['mappings'] != mappings.to_dict() or payload['char_to_id'] != expected_chars:
        raise ValueError('Checkpoint differs from fresh training-only mappings')
    if payload['data_sha256'] != data_sha256 or payload['code_sha256'] != code_sha256:
        raise ValueError('Dataset or executed notebook code changed since training')
    model = nb_new_model(payload['stage'], mappings, expected_chars, payload['stage_config'], device)
    if type(model).__name__ != payload['model_class']:
        raise ValueError('Checkpoint model class changed')
    model.load_state_dict(payload['model_state'], strict=True)
    model.eval()
    counts = enforce_parameter_cap(model, 20_000_000)
    if counts != payload['parameter_counts']:
        raise ValueError('Reconstructed parameter count changed')
    return model, payload


def nb_plot_learning_curves(history, path, stage):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    epochs = [record['epoch'] for record in history]
    for axis, key in zip(axes.flat[:3], ('loss', 'intent_loss', 'slot_loss')):
        for partition in ('train', 'dev'):
            axis.plot(epochs, [record[partition + '_' + key] for record in history], label=partition)
        axis.set(title=key.replace('_', ' '), xlabel='Epoch', ylabel='Loss')
        axis.legend()
    for key, label in (('dev_intent_accuracy', 'Intent accuracy'), ('dev_slot_micro_f1', 'Slot micro F1')):
        axes[1, 1].plot(epochs, [100 * record[key] for record in history], label=label)
    axes[1, 1].set(title='Development metrics', xlabel='Epoch', ylabel='Percent')
    axes[1, 1].legend()
    figure.suptitle(stage.replace('stage', 'Stage ') + ' — fresh fixed-configuration notebook training')
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return str(path)


def nb_train_stage(stage, prepared, mappings, char_to_id, stage_config, runtime,
                   device, output_dir, data_sha256, code_sha256):
    """Train one fresh selected system, retaining every epoch and earliest best."""
    nb_validate_config(stage, stage_config, runtime)
    for partition in ('train', 'dev'):
        nb_validate_prepared(prepared[partition], partition)
    if fit_mappings([item.example for item in prepared['train']]).to_dict() != mappings.to_dict():
        raise ValueError('Vocabulary and label mappings must fit original training records only')
    chars = dict(char_to_id or {}) if stage == 'stage2' else {}
    if stage == 'stage2' and fit_char_vocab(prepared['train']) != chars:
        raise ValueError('Character vocabulary must fit original training text only')
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    nb_write_json(output / 'run.json', {'stage': stage, 'created_at_utc': nb_now(),
                  'stage_config': stage_config, 'runtime': runtime, 'data_sha256': data_sha256,
                  'code_sha256': code_sha256, 'evaluator_version': EVALUATOR_VERSION, 'training_pid': os.getpid(),
                  'python': sys.version, 'torch': str(torch.__version__), 'torch_cuda': torch.version.cuda,
                  'device': str(device), 'gpu': torch.cuda.get_device_name(device) if torch.device(device).type == 'cuda' else None,
                  'selection': 'maximum unrounded mean(dev intent accuracy, dev slot micro F1), earliest exact tie',
                  'fixed_configuration_reproduction': True, 'test_model_evaluations': 0})
    # Reset before construction; model, loader and optimizer creation match the
    # original scripts. The same train-loader generator stays alive all epochs.
    configure_reproducibility(runtime)
    torch.set_default_dtype(torch.float32)
    model = nb_new_model(stage, mappings, chars, stage_config, device)
    counts = enforce_parameter_cap(model, 20_000_000)
    training = stage_config['training']
    loaders = {partition: nb_make_loader(stage, prepared[partition], partition, training['batch_size'],
                                        runtime['seed'], chars) for partition in ('train', 'dev')}
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'], weight_decay=training['weight_decay'])
    print(f'{stage}: parameters total={counts["total"]:,}; trainable={counts["trainable"]:,}; cap=20,000,000', flush=True)
    if torch.device(device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    history, best_score, best_epoch = [], None, None
    checkpoint = output / 'best.pt'
    nb_sync(device)
    start = perf_counter()
    for epoch in range(1, training['max_epochs'] + 1):
        if (output / 'freeze_manifest.json').exists():
            raise ValueError('Training is closed after the model freeze')
        if stage == 'stage1':
            train = train_epoch(model, loaders['train'], optimizer, device, grad_clip=training['gradient_clip_norm'])
        else:
            train = s2_epoch(model, loaders['train'], mappings, device, optimizer=optimizer,
                             grad_clip=training['gradient_clip_norm'])
        dev = nb_evaluate(stage, model, loaders['dev'], mappings, device)
        nb_validate_result(train, prepared['train'], 'train')
        nb_validate_result(dev, prepared['dev'], 'dev')
        score = selection_score(dev['metrics'].intent_accuracy, dev['metrics'].slot_micro_f1)
        improved = is_improvement(score, best_score)
        checkpoint_seconds = 0.0
        if improved:
            nb_sync(device)
            saving = perf_counter()
            nb_save_checkpoint(checkpoint, stage, model, optimizer, mappings, chars, stage_config,
                               runtime, epoch, dev, data_sha256, code_sha256)
            write_predictions(output / 'best_dev_predictions.jsonl', dev['predictions'])
            nb_sync(device)
            checkpoint_seconds = perf_counter() - saving
            best_score, best_epoch = score, epoch
        keys = ('loss', 'intent_loss', 'slot_loss', 'num_examples', 'num_tokens', 'num_batches', 'seconds')
        row = {'epoch': epoch, **{f'train_{key}': train[key] for key in keys},
               **{f'dev_{key}': dev[key] for key in keys},
               'dev_intent_accuracy': dev['metrics'].intent_accuracy, 'dev_slot_micro_f1': dev['metrics'].slot_micro_f1,
               'selection_score': score, 'selected_checkpoint': improved, 'checkpoint_seconds': checkpoint_seconds,
               'max_gradient_norm': train['max_gradient_norm'],
               'train_record_ids_sha256': configuration_hash({'ids': sorted(train['record_ids'])}),
               'dev_record_ids_sha256': configuration_hash({'ids': dev['record_ids']})}
        history.append(row)
        with (output / 'history.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        with (output / 'history.csv').open('a', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        print(f'{stage} epoch {epoch:02d}/{training["max_epochs"]}: train_loss={train["loss"]:.6f}; '
              f'dev_loss={dev["loss"]:.6f}; intent={100 * dev["metrics"].intent_accuracy:.2f}%; '
              f'slot_F1={100 * dev["metrics"].slot_micro_f1:.2f}%; mean={100 * score:.2f}%; '
              f'train={train["num_examples"]}/{train["num_tokens"]} records/tokens; '
              f'dev={dev["num_examples"]}/{dev["num_tokens"]}; '
              f'seconds(train/dev)={train["seconds"]:.3f}/{dev["seconds"]:.3f}; selected={improved}', flush=True)
    nb_sync(device)
    workflow_seconds = perf_counter() - start
    summary = {'schema_version': 1, 'stage': stage, 'status': 'training_complete_not_tested',
               'epochs_trained': len(history), 'best_epoch': best_epoch, 'best_score': best_score,
               'checkpoint': f'{stage}/best.pt', 'checkpoint_sha256': file_sha256(checkpoint),
               'training_seconds': workflow_seconds, 'training_workflow_seconds': workflow_seconds,
               'timing_scope': f'{len(history)} training epochs, epoch development evaluation, selected checkpoint writes and epoch evidence; excludes setup, plotting and final evaluations',
               'parameter_counts': counts, 'data_sha256': data_sha256, 'code_sha256': code_sha256,
               'peak_allocated_cuda_bytes': torch.cuda.max_memory_allocated(torch.device(device)) if torch.device(device).type == 'cuda' else None,
               'test_model_evaluations': 0, 'completed_at_utc': nb_now()}
    nb_write_json(output / 'training_summary.json', summary)
    nb_plot_learning_curves(history, output / 'learning_curves.png', stage)
    print(f'{stage}: selected epoch {best_epoch}/{training["max_epochs"]}; development mean={100 * best_score:.2f}%; '
          f'checkpoint={checkpoint.name}; training workflow={workflow_seconds:.2f} seconds', flush=True)
    # The returned value contains evidence only, not the last epoch's model.
    del optimizer, model
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    return summary


def nb_assert_selected_history(checkpoint, payload):
    output = Path(checkpoint).parent
    history = [json.loads(line) for line in (output / 'history.jsonl').read_text(encoding='utf-8').splitlines()]
    summary = json.loads((output / 'training_summary.json').read_text(encoding='utf-8'))
    epochs = payload['stage_config']['training']['max_epochs']
    if [row['epoch'] for row in history] != list(range(1, epochs + 1)) or summary['epochs_trained'] != epochs:
        raise ValueError('Every configured epoch must complete before finalization')
    for row in history:
        if row['train_num_examples'] != EXPECTED_COUNTS['train'] or row['dev_num_examples'] != EXPECTED_COUNTS['dev']:
            raise ValueError('Partial experiment history')
        if row['selection_score'] != selection_score(row['dev_intent_accuracy'], row['dev_slot_micro_f1']):
            raise ValueError('History selection metric mismatch')
    best = max(history, key=lambda row: row['selection_score'])
    if best['epoch'] != payload['epoch'] or best['selection_score'] != payload['selection_score']:
        raise ValueError('Checkpoint is not the earliest maximum unrounded development mean')
    if summary['best_epoch'] != payload['epoch'] or summary['best_score'] != payload['selection_score']:
        raise ValueError('Training summary and checkpoint selection disagree')
    if summary['checkpoint_sha256'] != file_sha256(checkpoint):
        raise ValueError('Selected checkpoint changed after training')
    return history, summary


def nb_verify_development(checkpoint, prepared_dev, mappings, char_to_id, device,
                          data_sha256, code_sha256, fresh_python_process=False):
    """Reload in a distinct interpreter and match every ordered dev prediction."""
    checkpoint = Path(checkpoint)
    output = checkpoint.parent
    if (output / 'freeze_manifest.json').exists() or (output / 'reload_check.json').exists():
        raise FileExistsError('Refusing to replace verification or verify after final freeze')
    model, payload = nb_load_checkpoint(checkpoint, mappings, char_to_id, device, data_sha256, code_sha256)
    if not fresh_python_process or os.getpid() == payload['training_pid']:
        raise ValueError('A distinct fresh Python interpreter is required for notebook reload verification')
    nb_assert_selected_history(checkpoint, payload)
    loader = nb_make_loader(payload['stage'], prepared_dev, 'dev', payload['stage_config']['training']['batch_size'],
                            payload['runtime']['seed'], payload['char_to_id'])
    result = nb_evaluate(payload['stage'], model, loader, mappings, device)
    nb_validate_result(result, prepared_dev, 'dev')
    ordered = [prediction.to_dict() for prediction in result['predictions']]
    score = selection_score(result['metrics'].intent_accuracy, result['metrics'].slot_micro_f1)
    if ordered != payload['validation_predictions'] or score != payload['selection_score']:
        raise ValueError('Fresh-process selected development predictions or unrounded score changed')
    report = {'passed': True, 'stage': payload['stage'], 'fresh_python_process': True,
              'verification_pid': os.getpid(), 'training_pid': payload['training_pid'],
              'python_executable': sys.executable, 'verified_at_utc': nb_now(),
              'ordered_predictions_equal': True, 'absolute_score_difference': 0.0,
              'selection_score': score, 'epoch': payload['epoch'], 'num_examples': result['num_examples'],
              'metrics': result['metrics'].to_dict(), 'test_evaluated': False,
              'checkpoint_sha256': file_sha256(checkpoint), 'data_sha256': data_sha256, 'code_sha256': code_sha256}
    nb_write_json(output / 'reload_check.json', report)
    print(f'{payload["stage"]}: fresh-process reload reproduced all {result["num_examples"]:,} ordered development predictions exactly', flush=True)
    return report


def nb_finalize_stage(checkpoint, prepared_dev, prepared_test, mappings, char_to_id,
                      device, data_sha256, code_sha256, *, evaluate_test=True):
    """Persist a model freeze, then perform one complete final test evaluation."""
    checkpoint = Path(checkpoint)
    output = checkpoint.parent
    forbidden = ('freeze_manifest.json', 'test_started.json', 'dev_evaluation.json', 'dev_predictions.jsonl',
                 'test_evaluation.json', 'test_predictions.jsonl', 'final_summary.json')
    if any((output / name).exists() for name in forbidden):
        raise FileExistsError('This stage is frozen or final evaluation has already started; refusing repetition')
    model, payload = nb_load_checkpoint(checkpoint, mappings, char_to_id, device, data_sha256, code_sha256)
    history, training_summary = nb_assert_selected_history(checkpoint, payload)
    verification = json.loads((output / 'reload_check.json').read_text(encoding='utf-8'))
    if (verification.get('passed') is not True or verification.get('fresh_python_process') is not True
            or verification.get('test_evaluated') is not False or verification.get('ordered_predictions_equal') is not True
            or verification.get('num_examples') != EXPECTED_COUNTS['dev']
            or verification.get('verification_pid') == verification.get('training_pid')
            or verification.get('training_pid') != payload['training_pid']
            or verification.get('checkpoint_sha256') != file_sha256(checkpoint)
            or verification.get('code_sha256') != code_sha256 or verification.get('data_sha256') != data_sha256
            or verification.get('absolute_score_difference') != 0.0
            or verification.get('selection_score') != payload['selection_score']):
        raise ValueError('A passing fresh-interpreter full-development verification is required before test')
    stage = payload['stage']
    loaders = {partition: nb_make_loader(stage, prepared, partition, payload['stage_config']['training']['batch_size'],
                                         payload['runtime']['seed'], payload['char_to_id'])
               for partition, prepared in ((('dev', prepared_dev), ('test', prepared_test)) if evaluate_test else (('dev', prepared_dev),))}
    if evaluate_test:
        nb_validate_prepared(prepared_test, 'test')
    dev = nb_evaluate(stage, model, loaders['dev'], mappings, device)
    nb_validate_result(dev, prepared_dev, 'dev')
    score = selection_score(dev['metrics'].intent_accuracy, dev['metrics'].slot_micro_f1)
    if ([prediction.to_dict() for prediction in dev['predictions']] != payload['validation_predictions']
            or score != payload['selection_score']):
        raise ValueError('Final reloaded development evaluation differs from the selected checkpoint')
    counts = enforce_parameter_cap(model, 20_000_000)
    weights_before = nb_state_sha256(model)
    checkpoint_before = file_sha256(checkpoint)
    if not evaluate_test:
        nb_write_json(output / 'dev_evaluation.json', nb_result_data(dev))
        write_predictions(output / 'dev_predictions.jsonl', dev['predictions'])
        summary = {'schema_version': 1, 'stage': stage, 'status': 'complete',
                   'mode': 'development_only', 'evaluator_version': EVALUATOR_VERSION,
                   'completed_at_utc': nb_now(), 'selected_epoch': payload['epoch'],
                   'epochs_trained': len(history), 'checkpoint': f'{stage}/best.pt',
                   'checkpoint_sha256': checkpoint_before, 'model_state_sha256': weights_before,
                   'parameter_counts': counts, 'dev': nb_result_data(dev), 'test': None,
                   'training_workflow_seconds': training_summary['training_workflow_seconds'],
                   'fresh_process_reload': verification, 'ordered_development_predictions_equal': True,
                   'absolute_development_score_difference': 0.0, 'test_model_evaluations': 0,
                   'model_weights_unchanged': nb_state_sha256(model) == weights_before,
                   'checkpoint_bytes_unchanged': file_sha256(checkpoint) == checkpoint_before,
                   'training_and_tuning_closed': False, 'test_evaluation_finished': False,
                   'data_sha256': data_sha256, 'code_sha256': code_sha256}
        nb_write_json(output / 'final_summary.json', summary)
        print(f'{stage}: development-only run complete; NO test model evaluation performed', flush=True)
        del model
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()
        return summary
    manifest = {'schema_version': 1, 'stage': stage, 'frozen_at_utc': nb_now(),
                'checkpoint': checkpoint.name, 'checkpoint_sha256': checkpoint_before,
                'model_state_sha256': weights_before, 'selected_epoch': payload['epoch'],
                'epochs_trained': len(history), 'selection_score': score, 'parameter_counts': counts,
                'stage_config_sha256': payload['stage_config_sha256'], 'runtime_sha256': payload['runtime_sha256'],
                'mappings_sha256': payload['mappings_sha256'], 'char_to_id_sha256': payload['char_to_id_sha256'],
                'data_sha256': data_sha256, 'code_sha256': code_sha256,
                'history_sha256': file_sha256(output / 'history.jsonl'),
                'training_summary_sha256': file_sha256(output / 'training_summary.json'),
                'reload_check_sha256': file_sha256(output / 'reload_check.json'),
                'test_model_evaluations_at_freeze': 0, 'tuning_closed': True}
    nb_write_json(output / 'freeze_manifest.json', manifest)
    nb_write_json(output / 'dev_evaluation.json', nb_result_data(dev))
    write_predictions(output / 'dev_predictions.jsonl', dev['predictions'])
    nb_write_json(output / 'test_started.json', {'stage': stage, 'started_at_utc': nb_now(),
                  'test_model_evaluations': 1, 'reporting_only': True,
                  'freeze_manifest_sha256': file_sha256(output / 'freeze_manifest.json')})
    # The sole model pass over original test records is below this persisted
    # freeze. No test metric influences a checkpoint, epoch or model choice.
    test = nb_evaluate(stage, model, loaders['test'], mappings, device, allow_test=True)
    nb_validate_result(test, prepared_test, 'test')
    if nb_state_sha256(model) != weights_before or file_sha256(checkpoint) != checkpoint_before:
        raise ValueError('Frozen weights or checkpoint changed during final evaluation')
    nb_write_json(output / 'test_evaluation.json', nb_result_data(test))
    write_predictions(output / 'test_predictions.jsonl', test['predictions'])
    summary = {'schema_version': 1, 'stage': stage, 'status': 'complete', 'mode': 'final',
               'evaluator_version': EVALUATOR_VERSION, 'completed_at_utc': nb_now(),
               'selected_epoch': payload['epoch'], 'epochs_trained': len(history),
               'checkpoint': f'{stage}/best.pt', 'checkpoint_sha256': checkpoint_before,
               'parameter_counts': counts, 'dev': nb_result_data(dev), 'test': nb_result_data(test),
               'training_workflow_seconds': training_summary['training_workflow_seconds'],
               'fresh_process_reload': verification, 'ordered_development_predictions_equal': True,
               'absolute_development_score_difference': 0.0, 'test_model_evaluations': 1,
               'model_weights_unchanged': True, 'checkpoint_bytes_unchanged': True,
               'training_and_tuning_closed': True, 'test_evaluation_finished': True,
               'freeze_manifest_sha256': file_sha256(output / 'freeze_manifest.json'),
               'data_sha256': data_sha256, 'code_sha256': code_sha256}
    nb_write_json(output / 'final_summary.json', summary)
    print(f'{stage}: selected epoch {payload["epoch"]}/{len(history)}; reloaded development intent={100 * dev["metrics"].intent_accuracy:.2f}%, '
          f'slot F1={100 * dev["metrics"].slot_micro_f1:.2f}%; '
          f'final test intent={100 * test["metrics"].intent_accuracy:.2f}%, '
          f'slot F1={100 * test["metrics"].slot_micro_f1:.2f}%; one frozen test pass', flush=True)
    del model
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
    return summary

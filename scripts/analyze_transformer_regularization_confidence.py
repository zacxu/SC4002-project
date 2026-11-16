"""Read-only selected-checkpoint calibration on original DEV only.

Run after the registered comparison and conditional confirmations finish. No
optimizer, training traversal, test model inference, or checkpoint writes occur.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_transformer_regularization import (ARMS, analyze, file_hash,
    read_json, read_jsonl, require, resolve_registered_source)
from scripts.validate_revised_transformer import validate_regularization, state_evidence
from src.comparison_contract import DATASET_SHA256, EVALUATOR_VERSION, bio_to_character_spans, score_exact_spans
from src.revised_transformer_data import build_data
from src.revised_transformer_training import (_rt_load_checkpoint, _rt_loader,
    notebook_identity, rt_json_hash, rt_seed, rt_state_hash, validate_training_config)


def calibration_metrics(logits, targets, bins=15):
    """Calculate uncalibrated intent diagnostics using stable float64 math."""
    logits, targets = np.asarray(logits, dtype=np.float64), np.asarray(targets)
    require(logits.ndim == 2 and logits.shape[0] > 0 and logits.shape[1] >= 2
            and np.isfinite(logits).all(), 'Expected finite nonempty intent logits')
    require(targets.shape == (len(logits),) and np.issubdtype(targets.dtype, np.integer)
            and (targets >= 0).all() and (targets < logits.shape[1]).all(), 'Invalid intent targets')
    require(type(bins) is int and bins > 0, 'Calibration bin count must be positive')
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probabilities = np.exp(log_probs)
    predicted = logits.argmax(axis=1)
    correct = predicted == targets
    confidence = probabilities[np.arange(len(logits)), predicted]
    order = np.argsort(-logits, axis=1, kind='stable')
    assignment = np.minimum((confidence * bins).astype(np.int64), bins - 1)
    reliability = []
    ece = 0.
    for index in range(bins):
        selected = assignment == index
        count = int(selected.sum())
        accuracy = float(correct[selected].mean()) if count else None
        mean = float(confidence[selected].mean()) if count else None
        if count:
            ece += count / len(logits) * abs(accuracy - mean)
        reliability.append({'lower': index / bins, 'upper': (index + 1) / bins,
            'count': count, 'accuracy': accuracy, 'mean_confidence': mean})
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(logits)), targets] = 1.
    metrics = {'num_examples': len(logits), 'num_classes': logits.shape[1],
        'intent_accuracy': float(correct.mean()), 'intent_errors': int((~correct).sum()),
        'negative_log_likelihood': float(-log_probs[np.arange(len(logits)), targets].mean()),
        'multiclass_brier_score': float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        'ece_15' if bins == 15 else 'ece': float(ece), 'ece_num_bins': bins,
        'mean_confidence': float(confidence.mean()),
        'mean_incorrect_confidence': float(confidence[~correct].mean()) if (~correct).any() else None,
        'incorrect_confidence_ge_0_90': int(((~correct) & (confidence >= .90)).sum()),
        'incorrect_confidence_ge_0_99': int(((~correct) & (confidence >= .99)).sum()),
        'top2_accuracy': float((order[:, :min(2, logits.shape[1])] == targets[:, None]).any(axis=1).mean()),
        'top3_accuracy': float((order[:, :min(3, logits.shape[1])] == targets[:, None]).any(axis=1).mean()),
        'reliability_bins': reliability}
    return metrics, predicted, confidence


def collect_dev_logits(model, loader, bundle, device, expected_predictions):
    """One unsmoothed inference pass; preserve every original ordered dev row."""
    expected_ids = [row['record_id'] for row in bundle['splits']['dev']]
    require([row['record_id'] for row in loader.dataset] == expected_ids,
            'Confidence inference accepts only the original ordered development loader')
    require([row['record_id'] for row in expected_predictions] == expected_ids,
            'Saved development predictions are incomplete or unordered')
    model.eval()
    names = {value: key for key, value in bundle['mappings']['intent_to_id'].items()}
    tags = bundle['mappings']['tag_names']
    observed, targets, logits, predictions = [], [], [], []
    with torch.inference_mode():
        for original in loader:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                     for key, value in original.items()}
            output = model(batch['word_ids'], batch['mask'], batch['char_ids'])
            require(bool(torch.isfinite(output['intent_logits']).all()), 'Nonfinite checkpoint logits')
            intents = output['intent_logits'].argmax(-1).cpu().tolist()
            decoded = model.decode(output, batch['mask']).cpu().tolist()
            logits.append(output['intent_logits'].float().cpu().numpy())
            targets.extend(original['intent_ids'].tolist())
            for index, row in enumerate(original['rows']):
                observed.append(row['record_id'])
                predictions.append({'record_id': row['record_id'], 'utt': row['utt'],
                    'intent': names[intents[index]], 'slots': bio_to_character_spans(row['utt'],
                        row['token_offsets'], [tags[tag] for tag in decoded[index][:len(row['word_ids'])]])})
    require(observed == expected_ids and len(observed) == len(set(observed)),
            'Confidence inference dropped, repeated, or reordered development examples')
    require(predictions == expected_predictions, 'Checkpoint does not exactly reproduce saved intent and slot predictions')
    return np.concatenate(logits), np.asarray(targets, dtype=np.int64), predictions


def train_dev_bundle(dataset):
    """Fit train-only mappings while deliberately leaving the test split empty."""
    require(file_hash(dataset) == DATASET_SHA256, 'Original dataset bytes changed')
    split = {'train': [], 'dev': [], 'test': []}
    for line in Path(dataset).read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row['partition'] in ('train', 'dev'):
            split[row['partition']].append(row)
    bundle = build_data(split)
    require(not bundle['splits']['test'], 'Test rows must not enter the inference bundle')
    return split, bundle


def validate_checkpoint_settings(payload, run):
    """Compare canonical JSON settings, preserving all values and field names.

    PyTorch preserves model tuples while JSON records the same sequences as
    lists. This is the same canonical identity used by the training checkpoint;
    no coefficient, field, sequence order, numeric type, or runtime value is dropped.
    """
    require(all(rt_json_hash(payload[name]) == rt_json_hash(run[name])
                for name in ('stage_config', 'runtime')),
            'Loaded checkpoint runtime or effective settings differ')


def audit_checkpoint(path, bundle, raw_split, record, device):
    """Validate selected payload against completed-run evidence before inference."""
    path = Path(path)
    run, summary = read_json(path / 'run.json'), read_json(path / 'final_summary.json')
    trained = read_json(path / 'training_summary.json')
    history = read_jsonl(path / 'history.jsonl')
    checkpoint = path / summary['checkpoint']
    snapshot = read_json(path.parent / 'run_configuration.json')
    require(snapshot['STAGE2_CONFIG'] == run['stage_config'] and snapshot['RUNTIME'] == run['runtime']
            and snapshot['RUN_OPTIONS'] == run['run_options'] and snapshot['code_sha256'] == run['code_sha256'],
            'Saved executed configuration differs from the completed stage')
    script_completion = path.parent / 'script_completion.json'
    completion = read_json(script_completion if script_completion.is_file() else path.parent / 'completion.json')
    require(completion['status'] == 'complete' and completion['code_sha256'] == run['code_sha256'],
            'Parent script/notebook workflow did not finish with this implementation')
    if script_completion.is_file():
        require(completion['test_model_evaluations'] == 0 and completion['seed'] == run['runtime']['seed'],
                'Script completion seed or test boundary differs')
    else:
        require(completion['mode'] == 'development_only' and completion['stages']['stage2'] == summary,
                'Notebook completion mode or Stage 2 summary differs')
    validate_training_config(run['stage_config']['training'])
    rt_seed(run['runtime'])
    model, payload = _rt_load_checkpoint(checkpoint, bundle, device, DATASET_SHA256, record['code_sha256'])
    require(payload.get('purpose') == 'fresh_joint_transformer' and payload.get('stage') == 'stage2',
            'Expected a completed fresh joint Stage 2 checkpoint')
    validate_checkpoint_settings(payload, run)
    require(payload['epoch'] == summary['selected_epoch'] and payload['selection_score'] == summary['selection_score'],
            'Loaded checkpoint differs from selected development epoch')
    regularization = validate_regularization(run['stage_config']['training'], run['runtime'], raw_split,
        bundle['mappings'], bundle['stats']['token_counts'], history, run, trained, payload)
    for name in ('parameter_counts', 'regularization', 'initial_model_state_sha256'):
        require(payload.get(name) == summary.get(name) == trained.get(name) == run.get(name),
                'Completed evidence differs for ' + name)
    _, state_hash = state_evidence(payload['model_state'], summary['parameter_counts'])
    require(state_hash == rt_state_hash(model) == summary['model_state_sha256'],
            'Loaded learned tensors differ from the selected checkpoint fingerprint')
    expected = read_jsonl(path / 'dev_predictions.jsonl')
    evaluation, reload = read_json(path / 'dev_evaluation.json'), read_json(path / 'fresh_reload.json')
    require(expected == payload['dev_predictions'], 'Checkpoint and final saved predictions differ')
    metrics = score_exact_spans([bundle['gold_by_id'][row['record_id']] for row in expected], expected)
    require(metrics == payload['dev_metrics'] == summary['dev']['metrics'] == evaluation['metrics'],
            'Selected checkpoint and final development metrics differ')
    require(reload == summary['fresh_process_reload'] and reload['training_pid'] == payload['training_pid']
            and reload['training_pid'] != reload['verification_pid'], 'Independent reload identity differs')
    require(evaluation == summary['dev'], 'Final development traversal evidence differs')
    require(evaluation['num_examples'] == len(bundle['splits']['dev'])
            and evaluation['num_tokens'] == bundle['stats']['token_counts']['dev'], 'Development population differs')
    optimizer = payload['optimizer_state']
    parameter_ids = [index for group in optimizer['param_groups'] for index in group['params']]
    expected_steps = payload['epoch'] * math.ceil(len(bundle['splits']['train']) / run['stage_config']['training']['batch_size'])
    require(len(optimizer['param_groups']) == 2 and len(parameter_ids) == len(set(parameter_ids))
            and set(parameter_ids) == set(optimizer['state'])
            and all(float(item['step']) == expected_steps for item in optimizer['state'].values()),
            'Selected optimizer evidence is incomplete or inconsistent')
    # Optimizer evidence is read only; no optimizer object is ever instantiated.
    del payload
    return model, run, summary, expected, regularization


def assess(run_root, dataset, device='cuda'):
    root, dataset = Path(run_root).resolve(), Path(dataset).resolve()
    protocol_path = root / 'protocol.json'
    require(protocol_path.is_file(), 'A registered complete regularization protocol is required')
    protocol = read_json(protocol_path)['protocol']
    frozen = protocol['source_sha256']
    source_resolution = {path: resolve_registered_source(root, path, digest)
                         for path, digest in frozen.items()}
    require(all(value['location'] == 'current' for path, value in source_resolution.items()
                if path.startswith('src/')),
            'Cannot rerun historical confidence inference after runtime implementation changes. '
            'Archives are evidence only and are not executed. Reproduce the historical report using '
            'analyze_transformer_regularization.py --confidence results/transformer_regularization/confidence.json.')
    require(device in ('cpu', 'cuda'), 'Use cpu or cuda for the read-only diagnostic')
    require(device != 'cuda' or torch.cuda.is_available(), 'Requested CUDA device is unavailable')
    analysis_paths = ['scripts/analyze_transformer_regularization_confidence.py',
        'scripts/analyze_transformer_regularization.py', 'scripts/validate_revised_transformer.py']
    source_before = {str(ROOT / path): file_hash(ROOT / path) for path in set(frozen) | set(analysis_paths)}
    source_before.update({record['resolved_path']: record['sha256'] for record in source_resolution.values()})
    source_before[str(protocol_path)] = file_hash(protocol_path)
    source_before[str(dataset)] = file_hash(dataset)
    # Strict evidence analysis precedes any model construction/inference. One
    # bootstrap replicate is sufficient here: its intervals are neither used nor reported.
    evidence = analyze(root, dataset, replicates=1)
    require(evidence['status'] == 'complete' and evidence['confirmation'].get('complete') is True,
            'Finish every registered required seed/arm before confidence inference')
    for seed in {record['seed'] for record in evidence['runs']}:
        require({record['arm'] for record in evidence['runs'] if record['seed'] == seed} == set(ARMS),
                'Each discovered seed must contain all four completed arms')
    for record in evidence['runs']:
        source_dir = root / 'notebook_sources' / f"seed_{record['seed']}" / record['arm']
        notebook = source_dir / 'NLP_Final_project_revised_regularized.ipynb'
        registration = read_json(source_dir / 'registration.json')
        require(registration['seed'] == record['seed'] and registration['arm'] == record['arm']
                and registration['code_sha256'] == record['code_sha256'] == notebook_identity(notebook),
                'Registered notebook source or per-arm code identity changed')
        # The full notebook executor intentionally copies genuine cell outputs
        # back into this file; code identity remains fixed but raw bytes change.
        if Path(record['run_path']).parent.name != 'transformer_run_artifacts':
            require(registration['notebook_sha256_before_training'] == file_hash(notebook),
                    'Unexecuted script-arm source notebook bytes changed')
        source_before[str(notebook)] = file_hash(notebook)
        source_before[str(source_dir / 'registration.json')] = file_hash(source_dir / 'registration.json')
    raw_split, bundle = train_dev_bundle(dataset)
    results = []
    for record in evidence['runs']:
        path = Path(record['run_path'])
        model, run, summary, expected, regularization = audit_checkpoint(path, bundle, raw_split, record, device)
        checkpoint = path / summary['checkpoint']
        checkpoint_before, state_before = file_hash(checkpoint), rt_state_hash(model)
        loader = _rt_loader(bundle, 'dev', run['stage_config']['training']['batch_size'], run['runtime']['seed'])
        logits, targets, predictions = collect_dev_logits(model, loader, bundle, device, expected)
        calibration, _, _ = calibration_metrics(logits, targets)
        require(abs(calibration['intent_accuracy'] - summary['dev']['metrics']['intent_accuracy']) <= 1e-12,
                'Logit-derived intent accuracy differs from the saved development result')
        require(abs(calibration['negative_log_likelihood'] - summary['dev']['intent_loss']) <= 2e-6,
                'Logit-derived ordinary NLL differs from unsmoothed development CE')
        require(state_before == rt_state_hash(model) and checkpoint_before == file_hash(checkpoint),
                'Confidence inference changed learned state or checkpoint bytes')
        results.append({'seed': record['seed'], 'arm': record['arm'], 'run_path': str(path),
            'selected_epoch': summary['selected_epoch'], 'calibration': calibration,
            'regularization': regularization, 'parameter_counts': summary['parameter_counts'],
            'checkpoint_sha256': checkpoint_before, 'model_state_sha256': state_before,
            'initial_model_state_sha256': run['initial_model_state_sha256'],
            'ordered_dev_predictions_sha256': rt_json_hash(predictions),
            'logits_sha256': hashlib.sha256(logits.astype('<f4', copy=False).tobytes()).hexdigest(),
            'logits_shape': list(logits.shape), 'logits_dtype': 'float32, little-endian C order for hash',
            'saved_intent_and_slot_predictions_reproduced': True,
            'saved_unsmoothed_intent_loss_reproduced': True,
            'model_state_unchanged': True, 'checkpoint_bytes_unchanged': True,
            'dev_model_evaluations': 1, 'test_model_evaluations': 0})
        print(f"CONFIDENCE {record['seed']} {record['arm']}: NLL={calibration['negative_log_likelihood']:.6f} "
              f"ECE15={100*calibration['ece_15']:.3f}%", flush=True)
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    require(all(file_hash(path) == digest for path, digest in source_before.items()),
            'Registered sources, diagnostic sources, protocol, or dataset changed during inference')
    return {'schema_version': 1, 'status': 'complete', 'runs': results,
        'scope': 'Selected frozen checkpoints, one original ordered development pass per model; no training or test inference',
        'dataset_sha256': DATASET_SHA256, 'evaluator_version': EVALUATOR_VERSION,
        'device': device, 'torch_version': str(torch.__version__), 'numpy_version': np.__version__,
        'metric_definition': {'probabilities': 'Stable float64 softmax of original float32 intent logits; no temperature fitting',
            'nll': 'Mean ordinary unsmoothed negative log probability of the gold intent, in nats',
            'ece_15': '15 equally spaced confidence bins [lower,upper), final bin includes 1; population-weighted absolute accuracy-confidence difference',
            'brier': 'Mean sum of squared probability errors across all intent classes',
            'top_k_ties': 'Descending logit, then ascending train-fitted class index'},
        'sources_before_and_after_sha256': source_before, 'sources_unchanged': True,
        'registered_source_resolution': source_resolution,
        'test_model_evaluations': 0, 'optimizer_created': False,
        'limitations': ['Development both selected the checkpoints and assesses calibration; these are not held-out test claims.',
            'Raw logits were not stored during training: their argmax and decoded slots are exactly checked against saved predictions; their NLL is checked against saved ordinary dev CE.',
            'ECE depends on binning and sample size; improved calibration alone does not imply higher intent accuracy or slot F1.'],
        'completed_at_utc': datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=ROOT / 'artifacts/transformer_regularization')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/1.0/data/en-US.jsonl')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/transformer_regularization/confidence.json')
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(ROOT / 'results') or output.is_relative_to(ROOT / 'artifacts'),
            'Write diagnostic results only beneath project results/ or artifacts/')
    require(not output.exists(), 'Confidence output already exists; choose a fresh output filename')
    result = assess(args.run_root, args.dataset, args.device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('CONFIDENCE_COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()

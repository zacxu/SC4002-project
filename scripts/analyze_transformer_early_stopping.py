"""Audit fresh longer runs against preserved 20-epoch combined-model evidence.

Only original training/development labels, saved predictions, logs and CPU
checkpoint tensors are read. No model is instantiated and no inference occurs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_transformer_regularization as common
from scripts import validate_revised_transformer as validator

EXPECTED_POLICY = {'max_epochs': 60, 'schedule_epochs': 20, 'early_stopping_min_epochs': 20,
    'early_stopping_patience': 8, 'early_stopping_min_delta': .0005,
    'word_dropout_alpha': .25, 'intent_label_smoothing': .05}
CONTROL_KEYS = {'max_epochs', 'schedule_epochs', 'early_stopping_min_epochs',
                'early_stopping_patience', 'early_stopping_min_delta'}


def candidate_stage(root, seed):
    paths = [Path(root)/f'seed_{seed}'/tail for tail in (
        'full_notebook/transformer_run_artifacts/stage2', 'extended/stage2', 'stage2')]
    complete = [path for path in paths if (path/'final_summary.json').is_file()
                and common.read_json(path/'final_summary.json').get('status') == 'complete']
    common.require(len(complete) <= 1, 'Duplicate completed candidate stage paths')
    return complete[0] if complete else None


def train_mappings(data):
    tokens = [token for row in data['train'] for token in common.transformer_tokens(row['utt'])[0]]
    kinds = sorted({slot['slot_type'] for row in data['train'] for slot in row['slots']})
    return {'word_to_id': {v: i for i, v in enumerate(['<PAD>', '<UNK>']+sorted({t.casefold() for t in tokens}))},
            'char_to_id': {v: i for i, v in enumerate(['<CPAD>', '<CUNK>']+sorted(set(''.join(tokens))))},
            'intent_to_id': {v: i for i, v in enumerate(sorted({r['intent'] for r in data['train']}))},
            'tag_names': ['O']+[prefix+'-'+kind for kind in kinds for prefix in ('B', 'I')],
            'slot_types': kinds}


def verify_prefix(reference_history, candidate_history, horizon, reference, reference_predictions):
    """Compare all recorded old trajectory values except elapsed wall time."""
    length = EXPECTED_POLICY['schedule_epochs']
    common.require(len(reference_history) == length and len(candidate_history) >= length,
                   'A complete original and candidate 20-epoch prefix is required')
    checked = set()
    for before, after in zip(reference_history, candidate_history[:length]):
        for field, value in before.items():
            if field in {'train_seconds', 'dev_seconds'}:
                continue
            common.require(field in after and after[field] == value,
                f"First-20 trajectory differs at epoch {before['epoch']}, field {field}")
            checked.add(field)
    best = max(reference_history, key=lambda row: row['selection_score'])
    canonical_metrics = {key: reference['metrics'][key] for key in (
        'intent_accuracy', 'slot_micro_f1', 'slot_tp', 'slot_fp', 'slot_fn', 'num_examples', 'correct_intents')}
    checks = {
        'prefix_best_epoch': best['epoch'], 'prefix_best_selection_score': best['selection_score'],
        'prefix_best_model_state_sha256': reference['model_state_sha256'],
        'prefix_best_dev_metrics': canonical_metrics,
        'prefix_best_dev_predictions_sha256': common.json_hash(reference_predictions),
        'initial_model_state_sha256': reference['initial_model_state_sha256']}
    for field, expected in checks.items():
        common.require(horizon.get(field) == expected, 'Historical prefix parity differs for '+field)
    current_comparable = best['epoch'] == length
    if current_comparable:
        for field, expected in {'current_model_state_sha256': reference['model_state_sha256'],
                'current_dev_metrics': canonical_metrics,
                'current_dev_predictions_sha256': common.json_hash(reference_predictions)}.items():
            common.require(horizon.get(field) == expected, 'Historical horizon state differs for '+field)
    return {'status': 'passed', 'epochs': length, 'exact_history_fields': sorted(checked),
        'ignored_history_fields': ['train_seconds', 'dev_seconds'],
        'prefix_best_epoch': best['epoch'], 'prefix_best_model_state_identical': True,
        'prefix_best_ordered_predictions_identical': True,
        'epoch20_current_state_comparable_to_preserved_checkpoint': current_comparable,
        'scope': 'Every original first20 history value except timing matches exactly; selected first20 learned state/predictions match. '
                 'An old epoch20 learned state is available only when the historical selected checkpoint was epoch20.'}


def verify_run_settings(run, reference, specification):
    training = run['stage_config']['training']
    common.require(specification['stage2_policy'] == EXPECTED_POLICY
                   and all(training.get(key) == value for key, value in EXPECTED_POLICY.items()),
                   'Candidate differs from the registered 60/20/8/0.0005 policy')
    common.require(common.json_hash(run['stage_config']['model']) == common.json_hash(reference['model_configuration'])
                   and common.json_hash(run['runtime']) == common.json_hash(reference['runtime']),
                   'Candidate architecture/runtime differs from reference')
    retained = lambda config: {key: value for key, value in config.items() if key not in CONTROL_KEYS}
    common.require(common.json_hash(retained(training)) == common.json_hash(retained(reference['training_configuration'])),
                   'Candidate changes optimization or regularization beyond the registered budget controls')
    common.require(run['initial_model_state_sha256'] == reference['initial_model_state_sha256'],
                   'Fresh candidate initialization differs from historical paired reference')


def validate_checkpoint_header(payload, run, summary, selected, predictions, metrics):
    """Canonical JSON preserves tuple/list serialization while rejecting value/type drift."""
    common.require(payload['purpose'] == 'fresh_joint_transformer' and payload['stage'] == 'stage2'
                   and payload['dataset_sha256'] == common.DATASET_SHA256
                   and payload['evaluator_version'] == common.EVALUATOR_VERSION
                   and payload['code_sha256'] == summary['code_sha256'] == run['code_sha256']
                   and common.json_hash(payload['stage_config']) == common.json_hash(run['stage_config'])
                   and common.json_hash(payload['runtime']) == common.json_hash(run['runtime'])
                   and payload['stage_config_sha256'] == common.json_hash(run['stage_config'])
                   and payload['epoch'] == selected['selected_epoch']
                   and payload['selection_score'] == selected['selection_score']
                   and payload['dev_predictions'] == predictions and payload['dev_metrics'] == metrics,
                   'Selected CPU checkpoint does not match registered/saved development evidence')


def audit_candidate(root, path, seed, specification, reference, reference_history, reference_predictions,
                    data, identity, strata):
    import torch
    run = common.read_json(path/'run.json')
    summary = common.read_json(path/'final_summary.json')
    trained = common.read_json(path/'training_summary.json')
    history = common.read_jsonl(path/'history.jsonl')
    common.require(summary.get('status') == 'complete' and summary.get('mode') == 'development_only'
                   and summary.get('test_model_evaluations') == 0 and summary.get('test') is None
                   and run.get('stage') == summary.get('stage') == 'stage2'
                   and run['runtime']['seed'] == seed and run['run_options']['evaluate_test'] is False,
                   'Expected a complete same-seed development-only Stage2 run')
    common.require(not any((path/name).exists() for name in
        ('test_predictions.jsonl', 'test_evaluation.json', 'test_started.json', 'freeze_manifest.json')),
        'Candidate contains unexpected test/freeze evidence')
    for record in (run, summary):
        common.require(record['dataset_sha256'] == common.DATASET_SHA256
                       and record['evaluator_version'] == common.EVALUATOR_VERSION,
                       'Candidate dataset/evaluator identity differs')
    verify_run_settings(run, reference, specification)
    registration_dir = Path(root)/'notebook_sources'/f'seed_{seed}'
    registration = common.read_json(registration_dir/'registration.json')
    common.require(registration['seed'] == seed and registration.get('test_evaluation') is False,
                   'Notebook registration differs')
    notebook = registration_dir/'NLP_Final_project_revised_early_stopping.ipynb'
    full_notebook = path.parent.name == 'transformer_run_artifacts'
    source = common.validate_notebook_snapshot(notebook, run, registration, specification,
                                               check_original_bytes=not full_notebook)
    if full_notebook:
        common.validate_notebook_snapshot(path.parent.parent/notebook.name, run, registration, specification,
                                          check_original_bytes=False)
    snapshot = common.read_json(path.parent/'run_configuration.json')
    common.require(all(snapshot[key] == expected for key, expected in (
        ('STAGE2_CONFIG', run['stage_config']), ('RUNTIME', run['runtime']),
        ('RUN_OPTIONS', run['run_options']), ('code_sha256', run['code_sha256']))),
        'Executed configuration differs from registered stage settings')
    split = {part: [{'id': row['record_id'], 'utt': row['utt']} for row in rows] for part, rows in data.items()}
    tokens = {part: record['num_tokens'] for part, record in common.dataset_accounting(data).items()}
    selected = validator.validate_history(history, run['stage_config']['training'], split, tokens)
    validator.validate_history_csv(path/'history.csv', history)
    common.require(all(record[key] == value for record in (summary, trained) for key, value in selected.items()),
                   'Selected/trained epoch or score differs from independently checked history')
    predictions = common.read_jsonl(path/'dev_predictions.jsonl')
    metrics = common.validate_predictions(data['dev'], predictions)
    common.require(metrics == summary['dev']['metrics'], 'Saved candidate metrics differ from exact rescoring')
    evaluation = common.read_json(path/'dev_evaluation.json')
    common.require(evaluation == summary['dev'] and evaluation['num_examples'] == len(data['dev'])
                   and evaluation['num_tokens'] == tokens['dev']
                   and evaluation['record_ids_sha256'] == identity['train_dev_ordered_id_sha256']['dev'],
                   'Final development traversal differs from original complete data')
    checkpoint = path/summary['checkpoint']
    common.require(checkpoint.parent.resolve() == path.resolve() and checkpoint.is_file(), 'Unsafe/missing checkpoint')
    checkpoint_hash = common.file_hash(checkpoint)
    common.require(checkpoint_hash == summary['checkpoint_sha256']
                   == (path/(checkpoint.name+'.sha256')).read_text().strip(), 'Selected checkpoint bytes differ')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    validate_checkpoint_header(payload, run, summary, selected, predictions, metrics)
    counts = summary['parameter_counts']
    common.require(counts == reference['parameter_counts'] and all(record['parameter_counts'] == counts
                   for record in (run, trained, payload)), 'Whole-model parameter counts differ')
    total, state_hash = validator.state_evidence(payload['model_state'], counts)
    common.require(total == 19_748_595 and state_hash == summary['model_state_sha256'],
                   'Learned tensors/count differ from verified whole-model evidence')
    mappings = train_mappings(data)
    common.require(payload['mappings'] == mappings == common.read_json(path.parent/'mappings.json'),
                   'Mappings differ from reconstruction using original train only')
    common.require(all(row['intent'] in mappings['intent_to_id'] and all(
        slot['slot_type'] in mappings['slot_types'] for slot in row['slots']) for row in predictions),
        'Predicted labels are outside the training inventory')
    optimizer = payload['optimizer_state']
    parameter_ids = [key for group in optimizer['param_groups'] for key in group['params']]
    expected_steps = selected['selected_epoch']*math.ceil(len(data['train'])/run['stage_config']['training']['batch_size'])
    common.require(len(optimizer['param_groups']) == 2 and len(parameter_ids) == len(set(parameter_ids))
                   and set(parameter_ids) == set(optimizer['state'])
                   and all(float(value['step']) == expected_steps for value in optimizer['state'].values()),
                   'Selected optimizer state does not match the selected full-epoch training budget')
    validator.check_learning_rates([group['lr'] for group in optimizer['param_groups']],
        validator.expected_learning_rates(run['stage_config']['training'],
            math.ceil(len(data['train'])/run['stage_config']['training']['batch_size']), selected['selected_epoch']))
    validator.validate_regularization(run['stage_config']['training'], run['runtime'], split, mappings,
                                      tokens, history, run, trained, payload)
    stopping = validator.validate_stopping_evidence(history, run['stage_config']['training'], len(data['train']),
                                                    selected, run, trained, payload, summary)
    horizon = validator.validate_schedule_horizon(path, 'stage2', history, run['stage_config']['training'],
        len(data['train']), data['dev'], run, payload, selected, state_hash, checkpoint_hash)
    common.require(horizon['required'] and horizon['evidence'] is not None, 'Missing required first20 horizon evidence')
    prefix = verify_prefix(reference_history, history, horizon['evidence'], reference, reference_predictions)
    reload = common.read_json(path/'fresh_reload.json')
    common.require(reload == summary['fresh_process_reload'] and all(reload.get(key) is True for key in
        ('passed', 'fresh_python_process', 'ordered_predictions_equal'))
        and reload['test_evaluated'] is False and reload['metrics'] == metrics
        and reload['num_examples'] == len(data['dev']) and reload['training_pid'] == payload['training_pid']
        and reload['verification_pid'] != reload['training_pid'] and reload['checkpoint_sha256'] == checkpoint_hash
        and reload['code_sha256'] == run['code_sha256'] and reload['dataset_sha256'] == common.DATASET_SHA256,
        'Independent development reload evidence differs')
    common.require(summary.get('model_weights_unchanged') is True and summary.get('checkpoint_bytes_unchanged') is True,
                   'Evaluation lacks unchanged model/checkpoint verification')
    embedding = common.read_json(path.parent/'embedding_initialization.json')
    for key in ('matrix_sha256', 'source_archive_sha256', 'seed', 'fit_partition', 'task_embedding_parameters'):
        common.require(embedding[key] == reference['embedding_initialization'][key], 'Embedding initialization differs: '+key)
    completion_path = path.parent/('completion.json' if full_notebook else 'script_completion.json')
    completion = common.read_json(completion_path)
    common.require(completion['status'] == 'complete' and completion['code_sha256'] == run['code_sha256'],
                   'Parent workflow did not complete this registered notebook implementation')
    if full_notebook:
        common.require(completion['mode'] == 'development_only' and completion['stages']['stage2'] == summary,
                       'Notebook completion differs from candidate evidence')
    else:
        common.require(completion['seed'] == seed and completion['test_model_evaluations'] == 0,
                       'Script completion seed/test boundary differs')
    del payload
    return {'seed': seed, 'run_path': str(path), 'reference_path': reference['run_path'],
        **common.stratified_metrics(data['dev'], predictions, strata), **selected,
        'parameter_counts': counts, 'checkpoint_sha256': checkpoint_hash, 'model_state_sha256': state_hash,
        'prediction_sha256': common.file_hash(path/'dev_predictions.jsonl'),
        'ordered_dev_predictions_sha256': common.json_hash(predictions), 'stopping': stopping,
        'horizon_audit': horizon, 'prefix_parity': prefix, 'notebook_source_audit': source,
        'training_workflow_seconds': summary['training_workflow_seconds'],
        'epoch_train_dev_seconds': sum(row['train_seconds']+row['dev_seconds'] for row in history),
        'reference': {'metrics': reference['metrics'], 'selected_epoch': reference['selected_epoch'],
            'selection_score': reference['selection_score'], 'epochs_trained': reference['epochs_trained'],
            'checkpoint_sha256': reference['checkpoint_sha256'], 'model_state_sha256': reference['model_state_sha256']},
        'selection_score_gain_pp': 100*(selected['selection_score']-reference['selection_score']),
        'evidence_sha256': {name: common.file_hash(path/name) for name in ('run.json','final_summary.json',
            'history.jsonl','history.csv','training_summary.json','fresh_reload.json','dev_evaluation.json',
            'dev_predictions.jsonl','schedule_horizon_evidence.json')}, 'test_model_evaluations': 0}, predictions


def analyze(root, dataset, reference_root=None, replicates=10000):
    root = Path(root).resolve()
    protocol = common.read_json(root/'protocol.json')
    specification = protocol['protocol']
    common.require(specification['dataset_sha256'] == common.DATASET_SHA256
                   and specification['evaluator_version'] == common.EVALUATOR_VERSION
                   and specification['test_evaluation'] is False and specification['stage2_policy'] == EXPECTED_POLICY
                   and specification['reference_arm'] == 'combined', 'Different early-stopping experiment protocol')
    declared_reference = (ROOT/specification['reference_root']).resolve()
    reference_root = Path(reference_root).resolve() if reference_root is not None else declared_reference
    common.require(reference_root == declared_reference, 'Reference root differs from registered historical comparison')
    source_resolution = {name: common.resolve_registered_source(root, name, digest)
                         for name, digest in specification['source_sha256'].items()}
    old_protocol = common.read_json(reference_root/'protocol.json')['protocol']
    reference_sources = {name: common.resolve_registered_source(reference_root, name, digest)
                         for name, digest in old_protocol['source_sha256'].items()}
    for name in ('src/comparison_contract.py','src/revised_transformer_data.py'):
        common.require(common.file_hash(ROOT/name) == specification['source_sha256'][name]
                       == old_protocol['source_sha256'][name], 'Scorer/tokenizer changed across the comparison')
    protected = common.read_json(root/'protected_inputs.json')
    common.require(all(common.file_hash(ROOT/name) == digest for name, digest in protected.items()),
                   'An earlier protected input changed')
    data, identity = common.load_train_dev(dataset)
    strata = common.fit_strata(data['train'], data['dev'])
    runs, missing = [], []
    for seed in specification['registered_seeds']:
        path = candidate_stage(root, seed)
        if path is None:
            missing.append(seed)
            continue
        reference_path = common.discover_stage_path(reference_root/f'seed_{seed}'/'combined')
        common.require(reference_path is not None, 'Historical combined reference is incomplete')
        reference, before, history = common.load_run(reference_path, seed, 'combined', data, identity, strata)
        common.validate_registered_notebook(reference_root, reference, old_protocol)
        row, after = audit_candidate(root, path, seed, specification, reference, history, before, data, identity, strata)
        row['paired_development_comparison'] = common.paired_comparison(data['dev'], before, after,
            replicates=replicates, seed=20261111+seed)
        runs.append(row)
    common.require(bool(runs), 'No completed early-stopping candidate is available')
    screen = next((row for row in runs if row['seed'] == 42), None)
    confirmation_required = screen is not None and screen['selection_score_gain_pp'] > 0
    required = specification['registered_seeds'] if confirmation_required else [42]
    complete = screen is not None and all(seed in {row['seed'] for row in runs} for seed in required)
    decision_path = root/'confirmation_decision.json'
    if decision_path.is_file() and screen:
        decision = common.read_json(decision_path)
        common.require(decision['run_confirmation'] is confirmation_required
                       and decision['reference_first20_score'] == screen['reference']['selection_score']
                       and decision['extended_score'] == screen['selection_score']
                       and decision['test_evaluation'] is False, 'Registered follow-up decision differs from audited seed42 result')
    summary = {key: {'candidate': common.mean_sd(row['metrics'][key] for row in runs),
        'reference': common.mean_sd(row['reference']['metrics'][key] for row in runs),
        'paired_difference_pp': common.mean_sd(100*(row['metrics'][key]-row['reference']['metrics'][key]) for row in runs)}
        for key in common.METRIC_NAMES}
    return {'schema_version': 1, 'status': 'complete' if complete else 'partial',
        'scope': 'Fresh longer combined runs versus their preserved 20-epoch references; CPU evidence audit only, no inference',
        'protocol': protocol, 'dataset': identity, 'source_resolution': source_resolution,
        'historical_source_resolution': reference_sources, 'runs': runs, 'missing_seeds': missing,
        'confirmation': {'policy': specification['confirmation_policy'], 'required': confirmation_required,
            'required_seeds': required, 'completed_seeds': [row['seed'] for row in runs], 'complete': complete},
        'across_seed_summary': summary, 'bootstrap_replicates': replicates,
        'analysis_sources_sha256': {name: common.file_hash(ROOT/name) for name in (
            'scripts/analyze_transformer_early_stopping.py','scripts/analyze_transformer_regularization.py',
            'scripts/validate_revised_transformer.py')}, 'test_evaluation_performed': False,
        'limitations': ['Development chooses checkpoints, triggers confirmation and measures the resulting gains.',
            'Paired bootstrap conditions on selected checkpoints and omits checkpoint-selection and multiple-comparison effects.',
            'Only seeds actually completed enter paired means; a partial report is not the final registered comparison.',
            'More checkpoints provide additional chances to improve on the selection split; no new held-out/test claim is made.',
            'Runtime measures are descriptive and may reflect different concurrent GPU workloads.']}


def markdown_report(report):
    lines = ['# Transformer early-stopping results', '',
        f"Status: **{report['status']}**. Completed seeds: **{report['confirmation']['completed_seeds']}**. "
        'These fresh combined-model runs use a maximum of 60 epochs, a minimum of 20, patience of 8 epochs and min_delta=0.0005 '
        '(0.05 percentage points on the joint development score). The original first 20 epochs of the learning-rate schedule '
        'is retained exactly, followed by its floor. All comparisons use saved development evidence.', '',
        '| Seed | Trained / selected epoch | Stop reason | Reference intent | Extended intent | Reference slot F1 | Extended slot F1 | Joint-score gain (points) |',
        '| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in report['runs']:
        before, after = row['reference']['metrics'], row['metrics']
        lines.append(f"| {row['seed']} | {row['epochs_trained']} / {row['selected_epoch']} | {row['stopping']['termination']['reason']} | "
            f"{100*before['intent_accuracy']:.2f}% | {100*after['intent_accuracy']:.2f}% | "
            f"{100*before['slot_micro_f1']:.2f}% | {100*after['slot_micro_f1']:.2f}% | {row['selection_score_gain_pp']:+.3f} |")
    lines += ['', '## Paired seed means', '', '| Metric | Reference mean ± SD | Extended mean ± SD | Mean difference (points) |',
        '| --- | --- | --- | ---: |']
    for key, value in report['across_seed_summary'].items():
        lines.append(f"| {key} | {common.format_mean_sd(value['reference'])} | {common.format_mean_sd(value['candidate'])} | "
            f"{value['paired_difference_pp']['mean']:+.3f} |")
    lines += ['', 'The SD is the sample standard deviation across completed seeds, not a confidence interval. '
        'A one-seed run has no estimated seed SD. Intent and slot changes are reported separately; '
        'a higher mean selection score can trade one metric against the other.', '',
        '## Paired development uncertainty', '',
        '| Seed | Intent difference [nominal 95% interval] | Slot-F1 difference [nominal 95% interval] | Joint-exact difference [nominal 95% interval] |',
        '| ---: | --- | --- | --- |']
    for row in report['runs']:
        values = []
        for key in common.METRIC_NAMES:
            result = row['paired_development_comparison']['differences'][key]
            low, high = result['nominal_95_percentile_interval_pp']
            values.append(f"{result['percentage_points']:+.3f} [{low:+.3f}, {high:+.3f}]")
        lines.append(f"| {row['seed']} | "+' | '.join(values)+' |')
    lines += ['', f"Intervals use {report['bootstrap_replicates']:,} paired whole-development-utterance resamples. "
        'They condition on development-selected checkpoints, exclude selection/multiplicity effects, and are not training-seed uncertainty.', '',
        '## Verification and interpretation', '',
        '- Every original history value for the first 20 epochs except train/dev elapsed time matches its preserved same-seed combined reference exactly.',
        '- The evidence saved at epoch 20 matches the earlier best epoch within that budget, learned-tensor hash, complete metrics and canonical ordered prediction hash.',
        '- Current epoch-20 state can also be compared where the old selected checkpoint was epoch 20; otherwise only its saved best state within the first 20 epochs is available.',
        '- Initial model/embedding identities, original data, train-only mappings, parameter tensors, fresh development reload and exact character-span scores are verified.',
        '- The independent validator reconstructs patience counters, the fixed-horizon/floor learning rates, terminal reason, and earliest actual maximum separately from the min_delta anchor.',
        '- Checkpoint tensors are read on CPU; this analyzer performs no model construction, training, forward pass, test scoring or checkpoint writes.', '',
        f"Confirmation rule: {report['confirmation']['policy']} Confirmation required: **{report['confirmation']['required']}**; "
        f"complete: **{report['confirmation']['complete']}**.", '',
        'Detailed counts, novelty/OOV recall, precision/recall, joint exactness, paired recovered/lost errors, timing, '
        'source hashes and per-seed stopping/prefix audits are in `results/transformer_early_stopping/results.json`. '
        'The earlier regularization results remain separate and unchanged.', '']
    lines.extend('- '+value for value in report['limitations'])
    return '\n'.join(lines)+'\n'


def write_outputs(report, output, markdown):
    output, markdown = Path(output), Path(markdown)
    output.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    (output/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    rows = [{'seed': row['seed'], 'epochs_trained': row['epochs_trained'], 'selected_epoch': row['selected_epoch'],
        'stop_reason': row['stopping']['termination']['reason'],
        **{key+'_percent': f"{100*row['metrics'][key]:.4f}" for key in common.METRIC_NAMES},
        'selection_score_gain_pp': f"{row['selection_score_gain_pp']:.6f}"} for row in report['runs']]
    with (output/'results.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    markdown.write_text(markdown_report(report), encoding='utf-8', newline='\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=ROOT/'artifacts/transformer_early_stopping')
    parser.add_argument('--reference-root', type=Path, default=None)
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/1.0/data/en-US.jsonl')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/transformer_early_stopping')
    parser.add_argument('--markdown', type=Path, default=ROOT/'docs/TRANSFORMER_EARLY_STOPPING_RESULTS.md')
    parser.add_argument('--bootstrap-replicates', type=int, default=10000)
    args = parser.parse_args()
    report = analyze(args.run_root, args.dataset, args.reference_root, args.bootstrap_replicates)
    write_outputs(report, args.output_dir, args.markdown)
    print(json.dumps({'status': report['status'], 'completed_seeds': report['confirmation']['completed_seeds'],
        'confirmation_complete': report['confirmation']['complete'], 'output_dir': str(args.output_dir)}))


if __name__ == '__main__':
    main()

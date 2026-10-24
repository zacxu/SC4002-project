"""Commit the final Stage 2 choice after every registered dev-only run completes."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from src.acquisition import file_sha256
    from src.runtime import configuration_hash, load_protocol
    from src.stage2_finalization import _validate_registered_candidates
    output = root / 'artifacts/phase5'
    if (output / 'selection.json').exists():
        raise FileExistsError('Development selection is already committed')
    plan_path = output / 'experiment_plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    plan_hash = file_sha256(plan_path)
    metadata_paths = list((root / 'artifacts/runs').glob('stage2_*/run.json'))
    rows = []
    for registration in plan['candidates']:
        matches = []
        for path in metadata_paths:
            metadata = json.loads(path.read_text(encoding='utf-8'))
            if metadata.get('registered_plan_sha256') == plan_hash and metadata.get('candidate_id') == registration['candidate_id']:
                matches.append(path.parent)
        if len(matches) != 1:
            raise ValueError('Exactly one attempted run is required for each registered candidate')
        run = matches[0]
        summary = json.loads((run / 'experiment_summary.json').read_text(encoding='utf-8'))
        metrics = summary['fresh_process_reload']['metrics']
        rows.append({'candidate_id': registration['candidate_id'], 'name': registration['name'],
            'run': run.relative_to(root).as_posix(), 'configuration': registration['configuration'],
            'checkpoint_sha256': file_sha256(run / 'best.pt'), 'best_epoch': summary['best_epoch'],
            'epochs_trained': summary['epochs_trained'], 'selection_score': summary['best_score'],
            'dev_intent_accuracy': metrics['intent_accuracy'], 'dev_slot_micro_f1': metrics['slot_micro_f1'],
            'total_parameters': summary['parameter_counts']['total'], 'trainable_parameters': summary['parameter_counts']['trainable'],
            'training_seconds': summary['training_seconds'], 'peak_cuda_memory_allocated_bytes': summary['peak_cuda_memory_allocated_bytes']})
    winner = max(rows, key=lambda row: row['selection_score'])
    selection = {'schema_version': 1, 'stage': 'stage2', 'selection_split': 'dev', 'test_used_for_selection': False,
        'registered_plan': plan_path.relative_to(root).as_posix(), 'registered_plan_sha256': plan_hash,
        'selected_candidate': winner['candidate_id'], 'selected_configuration': winner['configuration'],
        'selected_run': winner['run'], 'checkpoint': winner['run'] + '/best.pt',
        'checkpoint_sha256': winner['checkpoint_sha256'], 'selected_epoch': winner['best_epoch'],
        'epochs_trained': winner['epochs_trained'], 'selection_score': winner['selection_score'], 'candidates': rows,
        'committed_at_utc': datetime.now(timezone.utc).isoformat(), 'completed_candidates': len(rows),
        'total_epochs': sum(row['epochs_trained'] for row in rows),
        'total_training_seconds': sum(row['training_seconds'] for row in rows),
        'baseline_dev_score': plan['baseline_dev_score'],
        'dev_mean_improvement_percentage_points': 100 * (winner['selection_score'] - plan['baseline_dev_score'])}
    membership = json.loads((root / 'artifacts/phase2/split_membership.json').read_text(encoding='utf-8'))['record_ids']
    ids = {'train': configuration_hash(sorted(membership['train'])), 'dev': configuration_hash(membership['dev'])}
    _validate_registered_candidates(root, selection, load_protocol(root), ids)
    with (output / 'selection.json').open('x', encoding='utf-8') as handle:
        json.dump(selection, handle, indent=2, allow_nan=False)
        handle.write('\n')
    with (output / 'candidate_comparison.json').open('x', encoding='utf-8') as handle:
        json.dump({'selection_split': 'dev', 'candidates': rows}, handle, indent=2, allow_nan=False)
        handle.write('\n')
    with (output / 'candidate_comparison.csv').open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(selection, indent=2))


if __name__ == '__main__':
    main()

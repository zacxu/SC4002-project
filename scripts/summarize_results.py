"""Generate the four required report rows only after both stages are frozen."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from src.acquisition import file_sha256
    records, sources, full_precision = [], {}, {}
    for stage, directory in (('Stage 1', root / 'artifacts/phase4/final_baseline'),
                              ('Stage 2', root / 'artifacts/phase5/final_stage2')):
        path = directory / 'final_summary.json'
        summary = json.loads(path.read_text(encoding='utf-8'))
        if (summary['status'] != 'complete' or summary['test_model_evaluations'] != 1
                or not summary['model_weights_unchanged'] or not summary['checkpoint_bytes_unchanged']
                or file_sha256(directory / 'freeze_manifest.json') != summary['freeze_manifest_sha256']):
            raise ValueError('Both stages must have completed frozen final reporting')
        sources[stage] = {'path': path.relative_to(root).as_posix(), 'sha256': file_sha256(path)}
        full_precision[stage] = {}
        for partition in ('dev', 'test'):
            result = summary[partition]
            metrics = result['metrics']
            full_precision[stage][partition] = metrics
            records.append({'stage': stage, 'split': partition,
                'intent_accuracy_percent': f"{100 * metrics['intent_accuracy']:.2f}",
                'slot_micro_f1_percent': f"{100 * metrics['slot_micro_f1']:.2f}",
                'total_parameters': summary['parameter_counts']['total'],
                'trainable_parameters': summary['parameter_counts']['trainable'],
                'selected_epoch': summary['selected_epoch'], 'epochs_trained': summary['epochs_trained'],
                'num_examples': result['num_examples'], 'training_workflow_seconds': summary['training_seconds']})
    differences = {part: {metric + '_percentage_points': 100 * (full_precision['Stage 2'][part][metric] - full_precision['Stage 1'][part][metric])
                         for metric in ('intent_accuracy', 'slot_micro_f1')} for part in ('dev', 'test')}
    report = {'schema_version': 1, 'purpose': 'final_reporting_only_after_both_stages_frozen', 'rows': records,
              'full_precision': full_precision, 'stage2_minus_stage1': differences, 'sources': sources,
              'scope': 'Script results; final report must use genuine Phase 7 notebook reproduction outputs.'}
    output = root / 'artifacts/phase5'
    json_path, csv_path = output / 'stage_comparison.json', output / 'stage_comparison.csv'
    if json_path.exists() or csv_path.exists():
        raise FileExistsError('Preserve the previously generated final reporting table')
    with json_path.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    with csv_path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

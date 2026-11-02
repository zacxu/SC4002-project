"""Development-only paired analysis of genuine saved Transformer predictions."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.comparison_contract import load_verified_massive, score_exact_spans
from src.revised_transformer_data import build_data


def analyze(run_root):
    run_root = Path(run_root)
    raw, identity = load_verified_massive(run_root / 'data/1.0/data/en-US.jsonl')
    bundle = build_data(raw)
    gold = [bundle['gold_by_id'][row['record_id']] for row in bundle['splits']['dev']]
    predictions, sources = {}, {}
    for stage in ('stage1', 'stage2'):
        path = run_root / stage / 'dev_predictions.jsonl'
        predictions[stage] = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if [row['record_id'] for row in predictions[stage]] != [row['record_id'] for row in gold]:
            raise ValueError('Saved predictions do not cover original development records in order')
        sources[stage] = hashlib.sha256(path.read_bytes()).hexdigest()
    groups = {'No-Train-OOV': [], 'NonSlot-Train-OOV': [], 'Slot-Train-OOV': []}
    for index, row in enumerate(bundle['splits']['dev']):
        unknown = [offset for token, offset in zip(row['word_ids'], row['token_offsets']) if token == 1]
        overlap = any(start < slot['end'] and slot['start'] < end
                      for start, end in unknown for slot in gold[index]['slots'])
        groups['Slot-Train-OOV' if overlap else 'NonSlot-Train-OOV' if unknown else 'No-Train-OOV'].append(index)
    strata = {}
    for name, indices in groups.items():
        strata[name] = {'num_examples': len(indices)}
        for stage in predictions:
            strata[name][stage] = score_exact_spans([gold[i] for i in indices], [predictions[stage][i] for i in indices])
    paired = {'intent_corrected': 0, 'intent_regressed': 0, 'fully_correct_stage1': 0,
              'fully_correct_stage2': 0, 'slot_occurrences_recovered': 0, 'slot_occurrences_lost': 0}
    def keys(record):
        return {(s['slot_type'], s['start'], s['end'], s['text']) for s in record['slots']}
    for target, before, after in zip(gold, predictions['stage1'], predictions['stage2']):
        first, second = before['intent'] == target['intent'], after['intent'] == target['intent']
        paired['intent_corrected'] += int(not first and second)
        paired['intent_regressed'] += int(first and not second)
        g, p, q = keys(target), keys(before), keys(after)
        paired['fully_correct_stage1'] += int(first and g == p)
        paired['fully_correct_stage2'] += int(second and g == q)
        paired['slot_occurrences_recovered'] += len((g & q) - p)
        paired['slot_occurrences_lost'] += len((g & p) - q)
    return {'status': 'complete', 'scope': 'Saved development predictions only; no model inference or test scoring',
        'dataset': identity, 'prediction_sha256': sources,
        'overall': {stage: score_exact_spans(gold, rows) for stage, rows in predictions.items()},
        'oov_strata': strata, 'paired_outcomes': paired,
        'limitations': 'One seed; groups overlap in linguistic difficulty and do not establish causal effects.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'status': report['status'], 'overall': report['overall'], 'output': str(args.output)}))

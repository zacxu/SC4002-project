"""Plot completed development-only regularization evidence; no model inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ['baseline', 'word_dropout', 'label_smoothing', 'combined']
LABELS = ['Control', 'Word dropout', 'Intent smoothing', 'Both']
COLORS = ['#64748B', '#2563EB', '#D97706', '#059669']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'results/transformer_regularization/results.json')
    parser.add_argument('--confidence', type=Path, default=ROOT/'results/transformer_regularization/confidence.json')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/assets/transformer_regularization.png')
    args = parser.parse_args()
    report = json.loads(args.results.read_text(encoding='utf-8'))
    confidence = json.loads(args.confidence.read_text(encoding='utf-8'))
    if report['status'] != 'complete' or confidence['status'] != 'complete':
        raise ValueError('Plot the completed comparison, not a partial screen')
    rows = {(r['seed'], r['arm']):r for r in report['runs']}
    calibrated = {(r['seed'], r['arm']):r for r in confidence['runs']}
    seeds = sorted({seed for seed, _ in rows})
    expected = {(seed, arm) for seed in seeds for arm in ARMS}
    if set(rows) != expected or set(calibrated) != expected:
        raise ValueError('Require the same complete matched seed/arm sets')
    if len(seeds) < 2:
        raise ValueError('This figure requires multiple matched seeds')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'artifacts/transformer_regularization/figure_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2, 2, figsize=(12.5,8.4), layout='constrained')
    fig.suptitle('Stage 2 regularization: measured development results', fontsize=16, fontweight='bold')
    x = np.arange(4)
    aggregate = {}
    for ax, metric, title in [(axes[0,0], 'intent_accuracy', 'Intent accuracy (higher is better)'),
                               (axes[0,1], 'slot_micro_f1', 'Exact slot micro-F1 (higher is better)')]:
        values = np.array([[100*rows[seed,arm]['metrics'][metric] for seed in seeds] for arm in ARMS])
        mean, sd = values.mean(axis=1), values.std(axis=1, ddof=1)
        span = max(1., float((mean+sd).max() - (mean-sd).min()))
        for i in x:
            ax.errorbar(i, mean[i], yerr=sd[i], fmt='o', color=COLORS[i],
                        capsize=5, markersize=8, linewidth=2)
            ax.scatter(i+np.linspace(-.10,.10,len(seeds)), values[i],
                       color=COLORS[i], marker='.', s=26, alpha=.5)
            ax.annotate(f'{mean[i]:.2f}', (i,mean[i]+sd[i]), xytext=(0,8),
                        textcoords='offset points', ha='center', fontsize=10)
        ax.set(xticks=x, xticklabels=LABELS, xlim=(-.45,3.45),
               ylim=(max(0.,float((mean-sd).min())-.22*span), min(100.,float((mean+sd).max())+.45*span)),
               ylabel='Percent', title=title)
        ax.grid(axis='y', alpha=.2)
        ax.tick_params(axis='x', labelsize=9)
        aggregate[metric] = {arm:{'mean_percent':float(mean[i]),'sample_sd_percentage_points':float(sd[i])}
                             for i,arm in enumerate(ARMS)}

    groups = ['seen_typed_value','unseen_value_known_words','unseen_value_oov_words']
    group_labels = ['Seen type/value', 'New value;\nknown words', 'New value;\nOOV words']
    for i,arm in enumerate(ARMS):
        values = np.array([[100*rows[seed,arm]['gold_typed_value_recall'][g]['recall']
                            for seed in seeds] for g in groups])
        means, sd = values.mean(axis=1), values.std(axis=1,ddof=1)
        axes[1,0].bar(np.arange(3)+(i-1.5)*.18, means, .18, yerr=sd,
                      color=COLORS[i], capsize=2, label=LABELS[i])
    axes[1,0].set(xticks=np.arange(3),xticklabels=group_labels, ylim=(0,105),
                  ylabel='Exact span recall (%)',title='Generalization to unfamiliar slot values')
    axes[1,0].legend(loc='upper center',bbox_to_anchor=(.5,-.14), ncol=2, frameon=False, fontsize=9)
    axes[1,0].grid(axis='y',alpha=.15)

    values = np.array([[100*calibrated[seed,arm]['calibration']['ece_15'] for seed in seeds] for arm in ARMS])
    means, sd = values.mean(axis=1), values.std(axis=1,ddof=1)
    bars = axes[1,1].bar(x,means,color=COLORS,width=.58,yerr=sd,capsize=4)
    for i,bar in enumerate(bars):
        axes[1,1].annotate(f'{means[i]:.2f}', (i,means[i]+sd[i]), xytext=(0,5),
                           textcoords='offset points', ha='center', fontsize=10)
    axes[1,1].set(xticks=x,xticklabels=LABELS,ylim=(0,float((means+sd).max())*1.25),
                  ylabel='ECE (percentage points)',title='Intent calibration error (lower is better)')
    axes[1,1].tick_params(axis='x',labelsize=9)
    axes[1,1].grid(axis='y',alpha=.15)
    fig.supxlabel(f'Matched seeds {", ".join(map(str,seeds))}; means and sample SD. Small dots show individual seeds.\n'
                  'All results use development-selected checkpoints and the same original development split; no new test evaluation.', fontsize=9)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=170,facecolor='white')
    plt.close(fig)
    provenance = {'status':'complete','seeds':seeds,'scope':'Measured development results only',
                  'aggregate_task_metrics':aggregate,'source_sha256':{
                      str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in [args.results.resolve(),args.confidence.resolve()]},
                  'figure_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest()}
    (args.results.parent/'figure_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'figure':str(args.output),'seeds':seeds}))


if __name__ == '__main__':
    main()

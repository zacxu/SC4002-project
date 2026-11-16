"""Plot the audited longer-run trajectories; no model inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'results/transformer_early_stopping/results.json')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/assets/transformer_early_stopping.png')
    args=parser.parse_args()
    report=json.loads(args.results.read_text(encoding='utf-8'))
    if report['status']!='complete' or not report['confirmation']['complete']:
        raise ValueError('Plot the completed registered comparison')
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'artifacts/transformer_early_stopping/plot_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained',sharex=True)
    colors={42:'#2563EB',17:'#059669',101:'#D97706'}
    fields=[('dev_intent_accuracy','Intent accuracy'),('dev_slot_micro_f1','Exact slot micro-F1'),
            ('selection_score','Monitored joint score')]
    sources={}
    for run in report['runs']:
        path=Path(run['run_path'])/'history.jsonl'
        digest=sha(path)
        if digest!=run['evidence_sha256']['history.jsonl']:
            raise ValueError('History differs from the audited report')
        sources[str(path.relative_to(ROOT))]=digest
        rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if len(rows)!=run['epochs_trained']:raise ValueError('Incomplete history')
        visible=[row for row in rows if row['epoch']>=10]
        selected=rows[run['selected_epoch']-1]
        old=rows[run['reference']['selected_epoch']-1]
        label=f"Seed {run['seed']}: selected {run['selected_epoch']}, stopped {run['epochs_trained']}"
        for ax,(field,title) in zip(axes,fields):
            color=colors[run['seed']]
            ax.plot([r['epoch'] for r in visible],[100*r[field] for r in visible],
                    color=color,lw=1.6,label=label)
            ax.scatter(old['epoch'],100*old[field],color=color,marker='o',facecolors='white',s=48,zorder=4)
            ax.scatter(selected['epoch'],100*selected[field],color=color,marker='*',s=110,zorder=5)
            ax.scatter(rows[-1]['epoch'],100*rows[-1][field],color=color,marker='x',s=40,zorder=4)
    for ax,(_,title) in zip(axes,fields):
        ax.axvline(20,color='#475569',ls='--',lw=1,alpha=.7)
        ax.set(title=title,xlabel='Completed epoch (showing epoch 10 onward)',ylabel='Percent')
        ax.grid(alpha=.2)
    fig.suptitle('Revised Stage 2 Transformer: training beyond the original 20 epochs',fontsize=15,fontweight='bold')
    axes[-1].legend(loc='lower right',frameon=True,fontsize=9)
    fig.supxlabel('Dashed line: original budget / start of fixed low learning rates. '
                  'Hollow circle: original selected checkpoint; star: new selected checkpoint; x: stopped epoch.\n'
                  'All selections and comparisons use development data. Patience is 8 epochs with min_delta 0.05 joint-score points.',fontsize=9)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=170,facecolor='white');plt.close(fig)
    record={'status':'complete','scope':'Audited development trajectories only; no inference',
        'results_sha256':sha(args.results),'history_sha256':sources,'figure_sha256':sha(args.output)}
    (args.results.parent/'figure_provenance.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'figure':str(args.output),'seeds':[r['seed'] for r in report['runs']]}))


if __name__=='__main__':main()

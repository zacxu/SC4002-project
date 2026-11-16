"""Fresh development-only Stage 2 extension with a fixed first-20-epoch schedule.

Seed 42 executes both stages of the standalone notebook. Confirmation seeds
execute Stage 2 from that notebook's embedded definitions in fresh processes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pprint
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_transformer_regularization import (sha, read, write, checked_child,
    prepare_initialization, worker)

SEEDS = [42, 17, 101]
POLICY = {'max_epochs':60, 'schedule_epochs':20, 'early_stopping_min_epochs':20,
          'early_stopping_patience':8, 'early_stopping_min_delta':.0005,
          'word_dropout_alpha':.25, 'intent_label_smoothing':.05}
SOURCE_FILES = ['src/comparison_contract.py', 'src/revised_transformer_data.py',
    'src/stage2_model.py', 'src/revised_transformer_model.py', 'src/revised_transformer_training.py',
    'scripts/build_revised_transformer_notebook.py', 'scripts/execute_revised_transformer.py',
    'scripts/run_transformer_regularization.py', 'scripts/run_transformer_early_stopping.py']


def verify_frozen(root):
    protocol = read(root/'protocol.json')['protocol']
    if any(sha(ROOT/name) != digest for name,digest in protocol['source_sha256'].items()):
        raise ValueError('Implementation differs from the registered experiment; use a new root')
    if any(sha(ROOT/name) != digest for name,digest in read(root/'protected_inputs.json').items()):
        raise ValueError('A protected earlier input changed')
    return protocol


def create_protected_inputs(root):
    paths = ['NLP_Final_project.ipynb','NLP_Final_project_revised.ipynb',
        'NLP_Final_project_revised_regularized.ipynb','main.ipynb','src/comparison_contract.py',
        'data/1.0/data/en-US.jsonl','results/transformer_regularization/results.json',
        'results/transformer_regularization/confidence.json','artifacts/transformer_regularization/protocol.json']
    for seed in SEEDS:
        stage = ROOT/f'artifacts/transformer_regularization/seed_{seed}/combined'
        stage /= 'transformer_run_artifacts/stage2' if seed == 42 else 'stage2'
        paths.extend((stage/name).relative_to(ROOT).as_posix()
                     for name in ['best.pt','history.jsonl','final_summary.json','dev_predictions.jsonl'])
    write(root/'protected_inputs.json',{name:sha(ROOT/name) for name in paths})


def register(root, dataset, archive):
    from src.comparison_contract import DATASET_SHA256, EVALUATOR_VERSION
    specification = {'schema_version':1,'dataset_sha256':DATASET_SHA256,
        'evaluator_version':EVALUATOR_VERSION,'registered_seeds':SEEDS,'stage2_policy':POLICY,
        'reference_root':'artifacts/transformer_regularization','reference_arm':'combined',
        'selection':'earliest maximum unrounded mean of dev intent accuracy and exact slot micro-F1',
        'test_evaluation':False,'parameter_cap':20000000,
        'dataset_path':str(dataset),'embedding_archive_sha256':sha(archive),
        'source_sha256':{name:sha(ROOT/name) for name in SOURCE_FILES},
        'confirmation_policy':'Run seed42 first; if its selected joint development score exceeds its own '
            'first20 maximum, run both registered confirmation seeds17 and101 regardless of their results.',
        'schedule':'Same first20 epochs as reference: 10% warmup then cosine to10% peak; hold floor afterward.'}
    if (root/'protocol.json').exists():
        if read(root/'protocol.json')['protocol'] != specification:
            raise ValueError('Existing registration differs; choose a new output root')
    else:
        write(root/'protocol.json',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),
                                    'protocol':specification})
        archive_root = root/'registered_sources'
        for name,digest in specification['source_sha256'].items():
            target = archive_root/name
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as handle:handle.write((ROOT/name).read_bytes())
            if sha(target) != digest:raise ValueError('Source archive copy differs')
        write(archive_root/'manifest.json',{'files':specification['source_sha256'],
            'protocol_sha256':sha(root/'protocol.json')})
    return verify_frozen(root)


def build_snapshot(root, seed, dataset, archive):
    import nbformat
    from scripts.build_revised_transformer_notebook import build
    from src.revised_transformer_training import notebook_configuration, notebook_identity
    target = root/f'notebook_sources/seed_{seed}/NLP_Final_project_revised_early_stopping.ipynb'
    if target.parent.exists():raise FileExistsError('An arm has already been started: '+str(target.parent))
    build(output=target,word_dropout_alpha=.25,intent_label_smoothing=.05,early_stopping=True)
    config=notebook_configuration(target)
    config['RUNTIME']['seed']=seed
    config['RUN_OPTIONS'].update(evaluate_test=False,dataset_path=str(dataset),embedding_archive=str(archive))
    document=nbformat.read(target,as_version=4)
    cell=next(c for c in document.cells if 'configuration' in c.metadata.get('tags',[]))
    cell.source='\n\n'.join(name+' = '+pprint.pformat(value,sort_dicts=False) for name,value in config.items())+'\n'
    nbformat.write(document,target)
    if any(config['STAGE2_CONFIG']['training'][k] != v for k,v in POLICY.items()):
        raise ValueError('Built notebook differs from registered policy')
    write(target.parent/'registration.json',{'seed':seed,'code_sha256':notebook_identity(target),
        'notebook_sha256_before_training':sha(target),'notebook':str(target),
        'configuration':config,'test_evaluation':False})
    return target


def reference_stage(seed):
    base=ROOT/f'artifacts/transformer_regularization/seed_{seed}/combined'
    return base/('transformer_run_artifacts/stage2' if seed==42 else 'stage2')


def verify_reference_prefix(stage, seed):
    from src.revised_transformer_training import rt_json_hash
    old=reference_stage(seed)
    previous=read(old/'final_summary.json')
    history=[json.loads(line) for line in (stage/'history.jsonl').read_text().splitlines()]
    reference=[json.loads(line) for line in (old/'history.jsonl').read_text().splitlines()]
    if len(history)<20 or len(reference)!=20:raise ValueError('Missing complete first20 history')
    for actual,expected in zip(history[:20],reference):
        for key,value in expected.items():
            if key not in ('train_seconds','dev_seconds') and actual.get(key)!=value:
                raise ValueError(f'First20 trajectory differs at epoch{expected["epoch"]}: {key}')
    horizon=read(stage/'schedule_horizon_evidence.json')
    predictions=[json.loads(line) for line in (old/'dev_predictions.jsonl').read_text().splitlines()]
    checks={'prefix_best_epoch':previous['selected_epoch'],
        'prefix_best_selection_score':previous['selection_score'],
        'prefix_best_model_state_sha256':previous['model_state_sha256'],
        'prefix_best_dev_metrics':previous['dev']['metrics'],
        'prefix_best_dev_predictions_sha256':rt_json_hash(predictions),
        'initial_model_state_sha256':previous['initial_model_state_sha256']}
    if any(horizon.get(key)!=value for key,value in checks.items()):
        raise ValueError('First20 selected weights/predictions do not reproduce the preserved reference')
    return {'status':'passed','seed':seed,'first20_trajectory_identical_except_timing':True,
        'prefix_best_weights_and_predictions_identical':True,'reference_checkpoint_sha256':sha(old/'best.pt'),
        'reference_model_state_sha256':previous['model_state_sha256'],
        'prefix_best_epoch':previous['selected_epoch'],'prefix_best_selection_score':previous['selection_score']}


def run_controller(args):
    root=args.output_root.resolve();dataset=args.dataset.resolve();archive=args.embedding_archive.resolve()
    if not root.is_relative_to(ROOT/'artifacts'):raise ValueError('Choose an output root beneath project artifacts')
    if len(set(args.seeds)) != len(args.seeds):raise ValueError('Seeds must be unique')
    root.mkdir(parents=True,exist_ok=True)
    if not (root/'protected_inputs.json').exists():create_protected_inputs(root)
    register(root,dataset,archive)
    for seed in args.seeds:
        verify_frozen(root)
        if seed != 42:
            decision=root/'confirmation_decision.json'
            if not decision.exists() or not read(decision)['run_confirmation']:
                raise ValueError('Complete the registered seed42 screen before confirmation seeds')
        initialization=prepare_initialization(root,seed,dataset,archive)
        old_initial=ROOT/f'artifacts/transformer_regularization/seed_{seed}/initialization/embedding_initialization.json'
        if read(initialization/'embedding_initialization.json')['matrix_sha256'] != read(old_initial)['matrix_sha256']:
            raise ValueError('Embedding initialization does not match the preserved reference')
        notebook=build_snapshot(root,seed,dataset,archive)
        directory=root/f'seed_{seed}'/('full_notebook' if seed==42 else 'extended')
        print(f'STARTING seed={seed}; policy={POLICY}',flush=True)
        if seed==42:
            command=[sys.executable,'scripts/execute_revised_transformer.py','--notebook',str(notebook),
                     '--execution-dir',str(directory),'--dataset-cache',str(dataset),'--embedding-cache',str(archive)]
        else:
            command=[sys.executable,str(Path(__file__).resolve()),'--worker','--notebook',str(notebook),
                     '--worker-output',str(directory),'--initialization',str(initialization)]
        checked_child(command,root,root/f'logs/seed_{seed}.txt')
        stage=directory/('transformer_run_artifacts/stage2' if seed==42 else 'stage2')
        summary=read(stage/'final_summary.json')
        if summary['test_model_evaluations'] or summary['parameter_counts']['total'] != 19748595:
            raise ValueError('Unexpected test pass or parameter count')
        prefix=verify_reference_prefix(stage,seed)
        write(root/f'seed_{seed}/prefix_pairing_audit.json',prefix)
        if seed==42:
            write(root/'confirmation_decision.json',{'decided_at_utc':datetime.now(timezone.utc).isoformat(),
                'run_confirmation':summary['selection_score']>prefix['prefix_best_selection_score'],
                'reference_first20_score':prefix['prefix_best_selection_score'],'extended_score':summary['selection_score'],
                'registered_confirmation_seeds':[17,101],'test_evaluation':False})
        print('SEED_COMPLETE',seed,json.dumps({'epochs':summary['epochs_trained'],
            'selected_epoch':summary['selected_epoch'],'dev':summary['dev']['metrics']}),flush=True)
    verify_frozen(root)
    print('REQUESTED_SEEDS_COMPLETE; earlier evidence unchanged; no test forwards',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=ROOT/'artifacts/transformer_early_stopping')
    parser.add_argument('--seeds',type=int,nargs='+',choices=SEEDS,default=[42])
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/1.0/data/en-US.jsonl')
    parser.add_argument('--embedding-archive',type=Path,default=ROOT/'data/pretrained/nlpl_model_6.zip')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--notebook',type=Path)
    parser.add_argument('--worker-output',type=Path)
    parser.add_argument('--initialization',type=Path)
    args=parser.parse_args()
    if args.worker:worker(args)
    else:run_controller(args)


if __name__=='__main__':main()

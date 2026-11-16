"""Run the registered Stage 2 regularization comparison on development only.

Each arm starts fresh, uses a separate process and the standalone notebook's
embedded definitions, and retains the complete standard run_stage evidence.
No test model evaluation is exposed by this command.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pprint
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARMS = {
    'baseline': (0.0, 0.0),
    'word_dropout': (0.25, 0.0),
    'label_smoothing': (0.0, 0.05),
    'combined': (0.25, 0.05),
}
REGISTERED_SEEDS = [42, 17, 101]
SOURCE_FILES = [
    'src/comparison_contract.py', 'src/revised_transformer_data.py',
    'src/stage2_model.py', 'src/revised_transformer_model.py',
    'src/revised_transformer_training.py',
    'scripts/build_revised_transformer_notebook.py',
    'scripts/execute_revised_transformer.py',
    'scripts/run_transformer_regularization.py',
]
PROTECTED_FILES = [
    'NLP_Final_project.ipynb', 'NLP_Final_project_revised.ipynb', 'main.ipynb',
    'src/comparison_contract.py', 'data/1.0/data/en-US.jsonl',
    'artifacts/revised_transformer/normalized_qualification/transformer_run_artifacts/stage2/best.pt',
    'artifacts/notebook_comparison_review/final_test/results.json',
]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def environment(output):
    result = dict(os.environ, PYTHONHASHSEED='42', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                  PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
                  MPLCONFIGDIR=str(output / 'matplotlib_cache'))
    result.pop('PYTHONPATH', None)
    return result


def checked_child(command, output, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment(output),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', bufsize=1)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait(timeout=30)
            raise
    if code:
        raise RuntimeError(f'Worker failed with exit code {code}; see {log_path}')


def prepare_initialization(root, seed, dataset, archive):
    import numpy as np
    from src.comparison_contract import load_verified_massive
    from src.revised_transformer_data import build_data, task_embeddings
    from src.revised_transformer_training import rt_json_hash
    directory = root / f'seed_{seed}' / 'initialization'
    if directory.exists():
        manifest = read(directory / 'embedding_initialization.json')
        matrix = np.load(directory / 'task_embedding.npy', allow_pickle=False)
        if (hashlib.sha256(matrix.tobytes()).hexdigest() != manifest['matrix_sha256']
                or manifest['seed'] != seed or manifest['source_archive_sha256'] != sha(archive)):
            raise ValueError('Existing initialization cache identity differs')
        return directory
    raw, identity = load_verified_massive(dataset)
    bundle = build_data(raw)
    matrix, manifest = task_embeddings(bundle['mappings']['word_to_id'], source='word2vec',
                                      seed=seed, dimension=300, archive_path=archive, download=False)
    directory.mkdir(parents=True, exist_ok=False)
    np.save(directory / 'task_embedding.npy', matrix, allow_pickle=False)
    manifest['mappings_sha256'] = rt_json_hash(bundle['mappings'])
    manifest['dataset_sha256'] = identity['dataset_sha256']
    write(directory / 'embedding_initialization.json', manifest)
    return directory


def build_arm_notebook(path, seed, arm, dataset, archive):
    import nbformat
    from scripts.build_revised_transformer_notebook import build
    from src.revised_transformer_training import notebook_configuration, notebook_identity
    alpha, epsilon = ARMS[arm]
    build(output=path, word_dropout_alpha=alpha, intent_label_smoothing=epsilon)
    configuration = notebook_configuration(path)
    configuration['RUNTIME']['seed'] = seed
    configuration['RUN_OPTIONS'].update(evaluate_test=False, dataset_path=str(dataset), embedding_archive=str(archive))
    document = nbformat.read(path, as_version=4)
    for cell in document.cells:
        if cell.cell_type == 'code' and 'configuration' in cell.metadata.get('tags', []):
            cell.source = '\n\n'.join(k + ' = ' + pprint.pformat(v, sort_dicts=False)
                                      for k, v in configuration.items()) + '\n'
    nbformat.validate(document)
    path.write_text(nbformat.writes(document), encoding='utf-8', newline='\n')
    return notebook_identity(path)


def completed_arm_evidence(root, seed, arm):
    output = root / f'seed_{seed}' / arm
    stage = output / 'stage2'
    run_root = output
    if not stage.exists():
        run_root = output / 'transformer_run_artifacts'
        stage = run_root / 'stage2'
    if not (stage / 'final_summary.json').exists():
        return None
    run = read(stage / 'run.json')
    history = [json.loads(line) for line in (stage / 'history.jsonl').read_text(encoding='utf-8').splitlines()]
    return {
        'embedding_matrix_sha256':read(run_root/'embedding_initialization.json')['matrix_sha256'],
        'initial_model_state_sha256':run['initial_model_state_sha256'],
        'epoch_train_record_order_sha256':[row['train_record_order_sha256'] for row in history],
        'epoch_masked_tokens':[row['train_word_dropout_masked_tokens'] for row in history],
        'epoch_mask_rng_sha256':[row['train_word_dropout_rng_state_sha256'] for row in history],
    }


def audit_pairing(root, seed):
    records = {arm:value for arm in ARMS if (value := completed_arm_evidence(root, seed, arm)) is not None}
    if not records:
        return
    first = next(iter(records.values()))
    for arm, value in records.items():
        for field in ['embedding_matrix_sha256','initial_model_state_sha256','epoch_train_record_order_sha256']:
            if value[field] != first[field]:
                raise ValueError(f'Paired experiment mismatch for seed={seed} arm={arm}: {field}')
    if {'word_dropout','combined'} <= records.keys():
        for field in ['epoch_masked_tokens','epoch_mask_rng_sha256']:
            if records['word_dropout'][field] != records['combined'][field]:
                raise ValueError('Word-dropout stochastic stream differs between paired active arms')
    write(root/f'seed_{seed}'/'pairing_audit.json', {'status':'passed','seed':seed,'arms':records})


def worker(args):
    import numpy as np
    import torch
    notebook, output = args.notebook.resolve(), args.worker_output.resolve()
    document = read(notebook)
    module = types.ModuleType('_sc4002_regularization_notebook')
    sys.modules[module.__name__] = module
    namespace = vars(module)
    for cell in document['cells']:
        if cell['cell_type'] == 'code' and 'definitions' in cell.get('metadata', {}).get('tags', []):
            source = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
            exec(compile(source, str(notebook), 'exec'), namespace)
    config = namespace['notebook_configuration'](notebook)
    if config['RUN_OPTIONS']['evaluate_test'] is not False:
        raise ValueError('This comparison is development-only')
    dataset = Path(config['RUN_OPTIONS']['dataset_path'])
    raw, identity = namespace['load_verified_massive'](dataset)
    bundle = namespace['build_data'](raw)
    initialization = args.initialization.resolve()
    manifest = read(initialization / 'embedding_initialization.json')
    matrix = np.load(initialization / 'task_embedding.npy', allow_pickle=False)
    if (hashlib.sha256(matrix.tobytes()).hexdigest() != manifest['matrix_sha256']
            or namespace['rt_json_hash'](bundle['mappings']) != manifest['mappings_sha256']
            or identity['dataset_sha256'] != manifest['dataset_sha256']
            or config['RUNTIME']['seed'] != manifest['seed']):
        raise ValueError('Initialization does not match seed/data/train-only mappings')
    output.mkdir(parents=True, exist_ok=False)
    snapshot = namespace['capture_configuration'](notebook, config['RUNTIME'], config['STAGE1_CONFIG'],
        config['STAGE2_CONFIG'], config['RUN_OPTIONS'], output / 'run_configuration.json')
    write(output / 'embedding_initialization.json', manifest)
    write(output / 'dataset_identity.json', identity)
    write(output / 'mappings.json', bundle['mappings'])
    write(output / 'data_statistics.json', bundle['stats'])
    write(output / 'runtime.json', {'python':sys.version, 'torch':str(torch.__version__),
                                  'gpu':torch.cuda.get_device_name(), 'runtime':config['RUNTIME']})
    summary = namespace['run_stage']('stage2', config['STAGE2_CONFIG'], config['RUNTIME'],
        config['RUN_OPTIONS'], bundle, matrix, 'cuda', output / 'stage2', dataset,
        snapshot['code_sha256'], notebook)
    write(output / 'script_completion.json', {
        'status':'complete', 'scope':'Stage 2 only using notebook embedded definitions; not a full notebook execution',
        'seed':config['RUNTIME']['seed'], 'notebook':str(notebook),
        'code_sha256':snapshot['code_sha256'], 'summary':'stage2/final_summary.json',
        'test_model_evaluations':summary['test_model_evaluations'],
    })
    print('ARM_COMPLETE', json.dumps(summary['dev']['metrics']), flush=True)


def controller(args):
    from src.comparison_contract import DATASET_SHA256, EVALUATOR_VERSION
    root = args.output_root.resolve()
    if not root.is_relative_to(ROOT / 'artifacts') or root == ROOT / 'artifacts':
        raise ValueError('Use a dedicated directory under this project artifacts/')
    dataset, archive = args.dataset_cache.resolve(), args.embedding_cache.resolve()
    if sha(dataset) != DATASET_SHA256:
        raise ValueError('Dataset identity mismatch')
    if not archive.is_file():
        raise FileNotFoundError(archive)
    protocol = {
        'schema_version':1, 'dataset_sha256':DATASET_SHA256, 'evaluator_version':EVALUATOR_VERSION,
        'registered_seeds':REGISTERED_SEEDS, 'arms':{
            name:dict(word_dropout_alpha=values[0], intent_label_smoothing=values[1])
            for name,values in ARMS.items()},
        'epochs':20, 'selection':'earliest maximum unrounded mean of dev intent accuracy and exact slot micro-F1',
        'test_evaluation':False, 'parameter_cap':20_000_000,
        'source_sha256':{path:sha(ROOT/path) for path in SOURCE_FILES},
        'embedding_archive_sha256':sha(archive),
        'confirmation_policy':'Seed 42 screen; run all four arms at registered seeds 17 and 101 if any changed arm improves both required metrics over its fresh seed-42 baseline. Report all completed arms, including regressions.',
    }
    manifest_path = root / 'protocol.json'
    if manifest_path.exists():
        if read(manifest_path)['protocol'] != protocol:
            raise ValueError('Protocol/source changed: register a new output root rather than mix runs')
        protected = read(root / 'protected_inputs.json')
        if any(sha(ROOT/p) != h for p,h in protected.items()):
            raise ValueError('Historical protected input changed')
    else:
        if root.exists():
            raise FileExistsError('Unregistered output directory already exists')
        root.mkdir(parents=True)
        write(manifest_path, {'registered_at_utc':datetime.now(timezone.utc).isoformat(), 'protocol':protocol})
        write(root / 'protected_inputs.json', {p:sha(ROOT/p) for p in PROTECTED_FILES})
    for seed in args.seeds:
        initialization = prepare_initialization(root, seed, dataset, archive)
        for arm in args.arms:
            if any(sha(ROOT/path) != value for path,value in protocol['source_sha256'].items()):
                raise ValueError('Registered source changed before the next arm')
            output = root / f'seed_{seed}' / arm
            if output.exists():
                raise FileExistsError(f'Never overwrite or silently resume an arm: {output}')
            source_dir = root / 'notebook_sources' / f'seed_{seed}' / arm
            source_dir.mkdir(parents=True, exist_ok=False)
            notebook = source_dir / 'NLP_Final_project_revised_regularized.ipynb'
            code_hash = build_arm_notebook(notebook, seed, arm, dataset, archive)
            write(source_dir/'registration.json', {'seed':seed, 'arm':arm, 'code_sha256':code_hash,
                'notebook_sha256_before_training':sha(notebook), 'registered_at_utc':datetime.now(timezone.utc).isoformat()})
            print(f'STARTING seed={seed} arm={arm} alpha={ARMS[arm][0]} epsilon={ARMS[arm][1]}', flush=True)
            if args.execute_combined_notebook and seed == 42 and arm == 'combined':
                command = [sys.executable,'-u',str(ROOT/'scripts/execute_revised_transformer.py'),
                    '--notebook',str(notebook),'--execution-dir',str(output),
                    '--dataset-cache',str(dataset),'--embedding-cache',str(archive)]
            else:
                command = [sys.executable,'-u',str(Path(__file__).resolve()),'--worker',
                    '--notebook',str(notebook),'--worker-output',str(output),'--initialization',str(initialization)]
            checked_child(command, root, root/'logs'/f'seed_{seed}_{arm}.txt')
            stage_dir = output / 'stage2'
            if not stage_dir.exists():
                stage_dir = output/'transformer_run_artifacts/stage2'
            summary = read(stage_dir/'final_summary.json')
            if summary['status'] != 'complete' or summary['test_model_evaluations'] != 0:
                raise ValueError('An arm did not finish the development-only workflow')
            audit_pairing(root, seed)
            print('FINISHED',seed,arm,json.dumps(summary['dev']['metrics']),flush=True)
    protected = read(root/'protected_inputs.json')
    if any(sha(ROOT/p) != h for p,h in protected.items()):
        raise ValueError('Historical protected input changed')
    if any(sha(ROOT/path) != value for path,value in protocol['source_sha256'].items()):
        raise ValueError('Registered source changed during the comparison')
    print('REQUESTED_ARMS_COMPLETE; historical inputs unchanged; no test forwards',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=ROOT/'artifacts/transformer_regularization')
    parser.add_argument('--seeds',type=int,nargs='+',choices=REGISTERED_SEEDS,default=[42])
    parser.add_argument('--arms',nargs='+',choices=list(ARMS),default=list(ARMS))
    parser.add_argument('--dataset-cache',type=Path,default=ROOT/'data/1.0/data/en-US.jsonl')
    parser.add_argument('--embedding-cache',type=Path,default=ROOT/'data/pretrained/nlpl_model_6.zip')
    parser.add_argument('--execute-combined-notebook',action='store_true',help='Run both notebook stages for seed-42 combined, retaining genuine notebook outputs.')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--notebook',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--worker-output',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--initialization',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)) or len(args.arms) != len(set(args.arms)):
        raise ValueError('Seeds and arms must be unique')
    if args.worker:
        worker(args)
    else:
        controller(args)


if __name__ == '__main__':
    main()

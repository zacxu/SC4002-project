"""Execute only the standalone revised notebook in a fresh, isolated directory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.revised_transformer_training import notebook_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path, default=ROOT / 'NLP_Final_project_revised.ipynb')
    parser.add_argument('--execution-dir', type=Path, required=True)
    parser.add_argument('--dataset-cache', type=Path)
    parser.add_argument('--embedding-cache', type=Path)
    parser.add_argument('--cell-timeout', type=int, default=-1)
    args = parser.parse_args()
    source, directory = args.notebook.resolve(), args.execution_dir.resolve()
    if not directory.is_relative_to(ROOT) or directory == ROOT or directory.exists():
        raise ValueError('Choose a new execution directory inside this project')
    if args.cell_timeout != -1 and args.cell_timeout <= 0:
        raise ValueError('cell-timeout must be positive or -1')
    identity = notebook_identity(source)
    document = nbformat.read(source, as_version=4)
    for cell in document.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    directory.mkdir(parents=True)
    target = directory / source.name
    target.write_text(nbformat.writes(document), encoding='utf-8', newline='\n')
    environment = dict(os.environ, PYTHONHASHSEED='42', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1',
        IPYTHONDIR=str(directory / '.ipython'), JUPYTER_RUNTIME_DIR=str(directory / '.jupyter_runtime'))
    environment.pop('PYTHONPATH', None)
    for flag, variable in ((args.dataset_cache, 'SC4002_DATASET_CACHE'), (args.embedding_cache, 'SC4002_WORD2VEC_CACHE')):
        environment.pop(variable, None)
        if flag:
            if not flag.resolve().is_file():
                raise FileNotFoundError(flag)
            environment[variable] = str(flag.resolve())
    report_path = directory.parent / (directory.name + '_execution.json')
    report = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_notebook': str(source), 'code_sha256': identity, 'initial_files': [target.name],
        'dataset_cache': environment.get('SC4002_DATASET_CACHE'),
        'embedding_cache': environment.get('SC4002_WORD2VEC_CACHE'), 'completed_cells': []}
    def save():
        target.write_text(nbformat.writes(document), encoding='utf-8', newline='\n')
        report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
    def completed(cell, cell_index, **kwargs):
        report['completed_cells'].append({'index': cell_index, 'execution_count': cell.get('execution_count')})
        save()
        print('CELL_FINISHED', cell_index, cell.get('execution_count'), flush=True)
    kernels = KernelSpecManager(kernel_dirs=[str(Path(sys.prefix) / 'share/jupyter/kernels')])
    if Path(kernels.get_kernel_spec('sc4002').argv[0]).resolve() != Path(sys.executable).resolve():
        raise ValueError('The sc4002 kernel must use this project Python environment')
    manager = KernelManager(kernel_name='sc4002', kernel_spec_manager=kernels)
    client = NotebookClient(document, km=manager, kernel_name='sc4002', timeout=args.cell_timeout,
        allow_errors=False, resources={'metadata': {'path': str(directory)}}, on_cell_executed=completed)
    save()
    try:
        client.execute(cwd=str(directory), env=environment, cleanup_kc=True)
        ledger = json.loads((directory / 'transformer_run_artifacts/completion.json').read_text(encoding='utf-8'))
        if ledger['status'] != 'complete': raise ValueError('Completion ledger is not complete')
        document.metadata['revised_transformer']['status'] = 'executed_' + ledger['mode']
        report.update(status='complete', mode=ledger['mode'], finished_at_utc=datetime.now(timezone.utc).isoformat())
        save()
        shutil.copyfile(target, source)
        print(json.dumps({'status': 'complete', 'notebook': str(source), 'run': str(directory), 'mode': ledger['mode']}), flush=True)
    except BaseException as error:
        report.update(status='failed', error=str(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()

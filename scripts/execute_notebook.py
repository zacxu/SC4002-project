"""Execute main.ipynb in a fresh directory using the declared local kernel.

The clean directory initially contains only a copy of the unexecuted notebook.
All data, checkpoints, logs and outputs must be produced by its cells. Failures
retain their real partial notebook and evidence; completed outputs are copied
back only after every cell succeeds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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

# Script entry point support, independent of the user's current directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.notebook_support import nb_source_identity, nb_notebook_configuration


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--notebook", type=Path, default=root / "main.ipynb")
    parser.add_argument("--execution-dir", type=Path, default=root / "artifacts/phase7/clean_execution")
    parser.add_argument('--cell-timeout', type=int, default=900,
                        help='Maximum seconds per cell; use -1 for long training cells without a limit')
    args = parser.parse_args()
    if args.cell_timeout != -1 and args.cell_timeout <= 0:
        raise ValueError('--cell-timeout must be positive or -1')
    source, directory = args.notebook.resolve(), args.execution_dir.resolve()
    if not directory.is_relative_to(root) or directory == root or directory.exists():
        raise ValueError("Use a fresh execution directory inside this workspace")
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    actual_identity = nb_source_identity(source)
    configuration = nb_notebook_configuration(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / "main.ipynb"
    target.write_text(nbformat.writes(notebook), encoding='utf-8', newline='\n')
    report_path = directory.parent / (directory.name + "_execution.json")
    report = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_notebook_sha256": digest(source), "effective_code_sha256": actual_identity,
              "run_options": configuration['RUN_OPTIONS'], "execution_directory": directory.relative_to(root).as_posix(),
              "initial_files": sorted(path.name for path in directory.iterdir()),
              "kernel_name": "sc4002", "kernel_interpreter": sys.executable,
              "startup_environment": {"PYTHONHASHSEED": "42", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
              "source_modules_or_existing_checkpoints_copied": False, "completed_cells": []}
    environment = os.environ.copy()
    environment.update(PYTHONHASHSEED="42", CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    # Keep kernel history/profile writes inside its isolated execution directory.
    environment['IPYTHONDIR'] = str(directory / '.ipython')
    environment['JUPYTER_RUNTIME_DIR'] = str(directory / '.jupyter_runtime')
    environment.pop("PYTHONPATH", None)
    kernels = KernelSpecManager(kernel_dirs=[str(Path(sys.prefix) / "share/jupyter/kernels")])
    specification = kernels.get_kernel_spec("sc4002")
    if Path(specification.argv[0]).resolve() != Path(sys.executable).resolve():
        raise ValueError("The selected kernel does not use this project environment")
    manager = KernelManager(kernel_name="sc4002", kernel_spec_manager=kernels)

    def save_report():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def after_cell(cell, cell_index, **kwargs):
        report["completed_cells"].append({"index": cell_index, "type": cell.cell_type,
                                          "execution_count": cell.get("execution_count"),
                                          "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        target.write_text(nbformat.writes(notebook), encoding='utf-8', newline='\n')
        save_report()
        print("CELL_FINISHED", cell_index, cell.get("execution_count"), flush=True)

    client = NotebookClient(notebook, km=manager, kernel_name="sc4002", timeout=args.cell_timeout,
                            allow_errors=False, resources={"metadata": {"path": str(directory)}},
                            on_cell_executed=after_cell, record_timing=True)
    save_report()
    try:
        client.execute(cwd=str(directory), env=environment, cleanup_kc=True)
    except BaseException as error:
        target.write_text(nbformat.writes(notebook), encoding='utf-8', newline='\n')
        report.update(status="failed", error=str(error), traceback=traceback.format_exc(),
                      finished_at_utc=datetime.now(timezone.utc).isoformat())
        save_report()
        raise
    target.write_text(nbformat.writes(notebook), encoding='utf-8', newline='\n')
    completion_path = directory / "run_artifacts/completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion["status"] != "complete":
        raise ValueError("Notebook did not produce a successful completion ledger")
    report.update({"status": "complete", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                   "executed_notebook_sha256": digest(target), "completion_sha256": digest(completion_path),
                   "code_cells_executed": len([cell for cell in notebook.cells if cell.cell_type == "code"]),
                   "outputs_preserved": True})
    save_report()
    shutil.copyfile(target, source)
    print(json.dumps({"status": "complete", "notebook": str(source), "execution_directory": str(directory),
                      "execution_report": str(report_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()

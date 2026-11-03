"""Execute rerun notebooks one at a time, in order, and stop at the first failure.

Finished runs are skipped. A run that stopped part-way is set aside and restarted, because the
notebook creates its artifact folder exclusively and never resumes.

Usage: python run_reruns.py r0_word2vec_base r1_word2vec_sam ...
"""
import asyncio
import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, DeadKernelError

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT.parent / "src" / "guard.py"


def log(message):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}", flush=True)


def finished(artifacts):
    path = artifacts / "completion.json"
    return path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"


def run(name):
    folder = ROOT / "runs" / name
    artifacts = folder / "transformer_run_artifacts"
    if finished(artifacts):
        log(f"{name}: already complete, skipping")
        return True
    if artifacts.exists():
        aside = folder / f"failed_{datetime.now():%Y%m%d_%H%M%S}"
        artifacts.rename(aside)
        log(f"{name}: set aside an unfinished attempt as {aside.name}")
    notebook = nbformat.read(folder / "NLP_Final_project_revised.ipynb", as_version=4)
    client = NotebookClient(notebook, timeout=None, kernel_name="sc4002", resources={"metadata": {"path": str(folder)}})
    log(f"{name}: started")
    try:
        client.execute()
    except (CellExecutionError, DeadKernelError) as error:
        nbformat.write(notebook, folder / "executed_failed.ipynb")
        log(f"{name}: FAILED: {str(error)[-1500:]}")
        return False
    nbformat.write(notebook, folder / "executed.ipynb")
    log(f"{name}: finished, complete={finished(artifacts)}")
    return finished(artifacts)


def main(names):
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        # no idle sleep while this process lives; it ends with the process and changes no setting
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    os.environ.update(PYTHONHASHSEED="42", CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONIOENCODING="utf-8",
                      SC4002_GUARD_PATH=str(GUARD))
    for name in names:
        if not run(name):
            log(f"stopping: {name} did not complete")
            return 1
    log("all requested runs complete")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Freeze the dev-selected Stage 2 system before its reporting-only test pass."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    sys.path.insert(0, str(root))
    # Do not import Torch before deterministic interpreter startup variables.
    from src.runtime import load_protocol, relative_project_path
    path = args.selection.resolve() if args.selection.is_absolute() else relative_project_path(root, args.selection.as_posix())
    if not path.is_relative_to(root):
        raise ValueError("Selection must remain inside the project")
    selection = json.loads(path.read_text(encoding="utf-8"))
    run = relative_project_path(root, selection["selected_run"])
    protocol = load_protocol(root, (run / "protocol.json").relative_to(root).as_posix())
    settings = {"PYTHONHASHSEED": str(protocol["runtime"]["seed"]),
                "CUBLAS_WORKSPACE_CONFIG": protocol["runtime"]["cublas_workspace_config"]}
    if any(os.environ.get(key) != value for key, value in settings.items()):
        environment = os.environ.copy()
        environment.update(settings)
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                                   env=environment, check=False)
        raise SystemExit(completed.returncode)
    from src.stage2_finalization import finalize_stage2
    print(json.dumps(finalize_stage2(root, path, args.device), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

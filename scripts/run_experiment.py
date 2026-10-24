"""Run baseline correctness diagnostics or a separate full training experiment.

Launches a child interpreter when needed so hash and CUDA determinism settings
take effect before importing Torch. All output paths remain inside the project.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("diagnostic", "train"), required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", default="configs/protocol.json")
    parser.add_argument("--config", default="configs/stage1.json")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    sys.path.insert(0, str(root))

    # runtime imports only the standard library at module import time.
    from src.runtime import load_protocol, relative_project_path

    protocol = load_protocol(root, args.protocol)
    with relative_project_path(root, args.config).open(encoding="utf-8") as handle:
        stage_config = json.load(handle)

    matplotlib_directory = root / "artifacts/setup/matplotlib"
    matplotlib_directory.mkdir(parents=True, exist_ok=True)
    expected_environment = {
        "PYTHONHASHSEED": str(protocol["runtime"]["seed"]),
        "CUBLAS_WORKSPACE_CONFIG": protocol["runtime"]["cublas_workspace_config"],
        "MPLCONFIGDIR": str(matplotlib_directory),
    }
    if any(os.environ.get(key) != value for key, value in expected_environment.items()):
        child_environment = os.environ.copy()
        child_environment.update(expected_environment)
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            env=child_environment,
            check=False,
        )
        raise SystemExit(completed.returncode)

    # Import Torch-dependent code only after the process environment is ready.
    from src.experiment import run_diagnostics, run_full_experiment

    runner = run_diagnostics if args.mode == "diagnostic" else run_full_experiment
    summary = runner(root, protocol, stage_config, args.device)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

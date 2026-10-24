"""Verify complete Phase 2 data preparation and save auditable evidence."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="configs/protocol.json")
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    sys.path.insert(0, str(root))
    from src.runtime import load_protocol

    protocol = load_protocol(root, args.config)
    runtime = protocol["runtime"]
    expected_environment = {"PYTHONHASHSEED": str(runtime["seed"]),
                            "CUBLAS_WORKSPACE_CONFIG": runtime["cublas_workspace_config"]}
    if any(os.environ.get(key) != value for key, value in expected_environment.items()):
        child_environment = os.environ.copy()
        child_environment.update(expected_environment)
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                                   env=child_environment, check=False)
        raise SystemExit(completed.returncode)

    from src.audit import run_data_audit
    run_data_audit(root, protocol)


if __name__ == "__main__":
    main()

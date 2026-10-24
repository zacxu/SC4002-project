"""Reload a selected Stage 1 checkpoint in a fresh Python process; dev only."""

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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", default="configs/protocol.json")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root))
    from src.runtime import load_protocol
    protocol = load_protocol(root, args.protocol)
    settings = {"PYTHONHASHSEED": str(protocol["runtime"]["seed"]),
                "CUBLAS_WORKSPACE_CONFIG": protocol["runtime"]["cublas_workspace_config"]}
    if any(os.environ.get(key) != value for key, value in settings.items()):
        environment = os.environ.copy()
        environment.update(settings)
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env=environment, check=False)
        raise SystemExit(result.returncode)
    from src.experiment import verify_checkpoint_file
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    report = verify_checkpoint_file(root, protocol, checkpoint.resolve(), args.device)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

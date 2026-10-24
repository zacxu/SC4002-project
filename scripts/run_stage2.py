"""Fresh-process Stage 2 diagnostics, full training, or dev-only verification."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('train', 'verify', 'diagnostic'), required=True)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--config')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', default='cuda', choices=('cpu', 'cuda', 'auto'))
    args = parser.parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root))
    from src.runtime import load_protocol, relative_project_path
    if args.mode == 'verify':
        if args.checkpoint is None:
            parser.error('--checkpoint required for verification')
        checkpoint = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
        protocol = load_protocol(root, (checkpoint.parent / 'protocol.json').relative_to(root).as_posix())
    else:
        if not args.config:
            parser.error('--config required for training/diagnostics')
        protocol = load_protocol(root)
    environment = {'PYTHONHASHSEED': str(protocol['runtime']['seed']),
                   'CUBLAS_WORKSPACE_CONFIG': protocol['runtime']['cublas_workspace_config'],
                   'MPLCONFIGDIR': str(root / 'artifacts/setup/matplotlib')}
    if any(os.environ.get(k) != v for k, v in environment.items()):
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env={**os.environ, **environment})
        raise SystemExit(result.returncode)
    from src.stage2_experiment import train, verify, diagnostic
    if args.mode == 'verify':
        report = verify(root, checkpoint, args.device)
    else:
        config = json.loads(relative_project_path(root, args.config).read_text(encoding='utf-8'))
        report = (train if args.mode == 'train' else diagnostic)(root, protocol, config, args.device)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

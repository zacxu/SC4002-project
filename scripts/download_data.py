"""Acquire and verify original MASSIVE 1.0 English data, preserving its partitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    args = parser.parse_args()
    # Import source from the checkout containing this wrapper; --project-root
    # controls output placement and never depends on the caller's directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.acquisition import acquire_dataset

    manifest = acquire_dataset(args.project_root)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

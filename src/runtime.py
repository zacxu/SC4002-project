"""Portable Phase 1 configuration and experiment-runtime conventions.

Third-party imports occur inside functions so deterministic environment variables
can be configured before importing Torch. No dataset or model is loaded here.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any


def project_root(explicit_root: str | Path | None = None) -> Path:
    """Resolve from source location, never from the caller's working directory."""
    root = Path(explicit_root) if explicit_root is not None else Path(__file__).resolve().parents[1]
    return root.expanduser().resolve()


def relative_project_path(root: Path, value: str) -> Path:
    """Resolve a configured relative path and reject escape from the project."""
    path = Path(value)
    if path.anchor or not value.strip():
        raise ValueError(f"Expected a nonempty project-relative path: {value!r}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Configured path escapes the project: {value!r}")
    return resolved


def load_protocol(root: Path, config: str = "configs/protocol.json") -> dict[str, Any]:
    with relative_project_path(root, config).open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    if protocol["schema_version"] != 1:
        raise ValueError("Unsupported protocol schema")
    for value in protocol["paths"].values():
        relative_project_path(root, value)
    if protocol["dataset"]["partition_counts"] != {"train": 11514, "dev": 2033, "test": 2974}:
        raise ValueError("Original MASSIVE English split counts must be preserved")
    if protocol["dataset"]["model_input"] != "utt":
        raise ValueError("Only utt may be used as model input")
    if protocol["runtime"]["precision"] != "float32":
        raise ValueError("The initial runtime protocol uses float32")
    seed = protocol["runtime"]["seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be a 32-bit nonnegative integer")
    if protocol["selection"]["weights"] != {"intent_accuracy": 0.5, "slot_micro_f1": 0.5}:
        raise ValueError("Selection weights must match the documented mean score")
    for key, expected in {"split": "dev", "metric": "mean_intent_accuracy_slot_micro_f1",
                          "metric_scale": "fraction", "mode": "max",
                          "tie_break": "earliest_epoch", "epoch_index_base": 1}.items():
        if protocol["selection"][key] != expected:
            raise ValueError(f"Unsupported selection setting: {key}")
    return protocol


def configure_reproducibility(runtime: dict[str, Any]) -> None:
    """Apply the documented seed/determinism policy before CUDA initialization."""
    expected = runtime["cublas_workspace_config"]
    previous = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if previous is not None and previous != expected:
        raise RuntimeError("Conflicting CUBLAS_WORKSPACE_CONFIG; restart with the protocol setting")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected
    import numpy as np
    import torch

    seed = runtime["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(runtime["deterministic_algorithms"], warn_only=False)
    torch.backends.cudnn.deterministic = runtime["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = runtime["cudnn_benchmark"]
    torch.backends.cuda.matmul.allow_tf32 = runtime["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = runtime["allow_tf32"]


def select_device(requested: str = "auto") -> Any:
    import torch

    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("Device must be auto, cuda, or cpu")
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cuda" if (requested == "cuda" or requested == "auto" and available) else "cpu")


def dataloader_generator(seed: int) -> Any:
    """Use with num_workers=0 in the initial experiment protocol."""
    import torch

    return torch.Generator().manual_seed(seed)


def selection_score(intent_accuracy: float, slot_micro_f1: float) -> float:
    """Mean of unrounded fractional dev metrics, not displayed percentages."""
    values = (intent_accuracy, slot_micro_f1)
    if any(isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Selection metrics must be finite fractions in [0, 1]")
    return (intent_accuracy + slot_micro_f1) / 2


def is_improvement(candidate: float, best: float | None) -> bool:
    """Exact ties retain the earlier checkpoint; reload tolerance is unrelated."""
    for value in (candidate,) if best is None else (candidate, best):
        if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Checkpoint scores must be finite fractions in [0, 1]")
    return best is None or candidate > best


def configuration_hash(configuration: dict[str, Any]) -> str:
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def create_run_directory(root: Path, stage: str, configuration: dict[str, Any]) -> Path:
    """Create an immutable run directory; never replace an existing run."""
    if stage not in {"stage1", "stage2"}:
        raise ValueError("Stage must be stage1 or stage2")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    digest = configuration_hash(configuration)
    run_path = relative_project_path(root, configuration["paths"]["runs"]) / f"{stage}_{stamp}_{digest[:8]}"
    run_path.mkdir(parents=True, exist_ok=False)
    metadata = {"stage": stage, "created_at_utc": stamp, "config_sha256": digest, "configuration": configuration}
    (run_path / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return run_path


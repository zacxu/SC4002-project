"""Compare the one-epoch smoke runs: the patched code with SAM off must match the original exactly."""
import hashlib
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"


def stage(run, name):
    folder = RUNS / run / "transformer_run_artifacts" / name
    summary = json.loads((folder / "final_summary.json").read_text(encoding="utf-8"))
    history = json.loads((folder / "history.jsonl").read_text(encoding="utf-8").splitlines()[0])
    return {"intent": summary["dev"]["metrics"]["intent_accuracy"], "slot": summary["dev"]["metrics"]["slot_micro_f1"],
            "train_loss": history["train_loss"], "epoch_s": round(history["train_seconds"], 1),
            "workflow_s": round(summary["training_workflow_seconds"], 1),
            "weights": summary["model_state_sha256"][:16],
            "preds": hashlib.sha256((folder / "dev_predictions.jsonl").read_bytes()).hexdigest()[:16]}


rows = {run: {name: stage(run, name) for name in ("stage1", "stage2")}
        for run in ("smoke_original", "smoke_patched_rho0", "smoke_patched_sam")}
for run, stages in rows.items():
    for name, values in stages.items():
        print(run, name, values)
for name in ("stage1", "stage2"):
    print(name, "patched rho 0 identical to original:", rows["smoke_original"][name] | {"epoch_s": 0, "workflow_s": 0}
          == rows["smoke_patched_rho0"][name] | {"epoch_s": 0, "workflow_s": 0})

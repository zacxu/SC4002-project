"""One-time test evaluation of one rerun's Stage 2, from the checkpoint chosen on dev.

Follows the notebook's own final-test steps (cell 9, evaluate_test branch) without retraining or testing Stage 1.
Refuses to run if test_<run>_stage2/ exists. That folder is created only after the notebook copy, dataset and
checkpoint match the run and dev reproduces exactly, and it is created before test is read.
Once r0 and r1 both have test predictions, also writes SAM's paired effect to test_sam_effect.json.

The notebook needs its hash seed, which -I would ignore; -P -s give the same protection on imports:
PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -P -s -B -X utf8 tools/run_test_stage2.py r0
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_results import ROOT, gold_spans, groups, score
from verify_patches import definitions

RUNS = {"r0": "r0_word2vec_base", "r1": "r1_word2vec_sam"}
REASONS = {"r0": "user approved one evaluation of r0 Stage 2, the no-SAM control for r1; nothing is selected from it",
           "r1": "user approved one evaluation of r1 Stage 2, chosen on dev"}
DATASET = ROOT / "cache" / "massive" / "en-US.jsonl"
DRAWS = 5000


def output(key):
    return ROOT / f"test_{key}_stage2"


def artifacts(key):
    return ROOT / "runs" / RUNS[key] / "transformer_run_artifacts"


def evaluate(key):
    target, notebook, checkpoint = output(key), ROOT / "runs" / RUNS[key] / "NLP_Final_project_revised.ipynb", artifacts(key) / "stage2" / "best.pt"
    if target.exists():
        raise SystemExit(f"{target} exists: the one-time test has already run")
    if sys.flags.ignore_environment or os.environ.get("PYTHONHASHSEED") != "42":
        raise SystemExit("Start with PYTHONHASHSEED=42 and without -I or -E")
    ns = definitions(notebook)
    code_sha256 = ns["notebook_identity"](notebook)
    completion = json.loads((artifacts(key) / "completion.json").read_text(encoding="utf-8"))
    if code_sha256 != completion["code_sha256"]:
        raise ValueError("Notebook copy differs from the one that trained this checkpoint")
    raw, manifest = ns["load_verified_massive"](DATASET)
    bundle = ns["build_data"](raw)
    device = torch.device("cuda")
    ns["rt_seed"](torch.load(checkpoint, map_location="cpu", weights_only=True)["runtime"])
    model, payload = ns["_rt_load_checkpoint"](checkpoint, bundle, device, manifest["dataset_sha256"], code_sha256)
    batch_size, seed = payload["stage_config"]["training"]["batch_size"], payload["runtime"]["seed"]
    dev = ns["_rt_epoch"](model, ns["_rt_loader"](bundle, "dev", batch_size, seed), bundle, device)
    if dev["metrics"] != payload["dev_metrics"] or dev["predictions"] != payload["dev_predictions"]:
        raise ValueError("Checkpoint does not reproduce its selected dev predictions; test not touched")
    print("Dev reproduced exactly from the checkpoint:", dev["metrics"], flush=True)
    state_before, file_before = ns["rt_state_hash"](model), ns["rt_file_hash"](checkpoint)
    target.mkdir(exist_ok=False)
    ns["rt_write_json"](target / "test_started.json", {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "reason": REASONS[key],
        "checkpoint_sha256": file_before, "model_state_sha256": state_before, "selected_epoch": payload.get("selected_epoch"), "test_model_evaluations": 1})
    test = ns["_rt_epoch"](model, ns["_rt_loader"](bundle, "test", batch_size, seed), bundle, device)
    ns["_rt_predictions"](target / "test_predictions.jsonl", test["predictions"])
    ns["rt_write_json"](target / "test_evaluation.json", ns["_rt_evidence"](test))
    if state_before != ns["rt_state_hash"](model) or file_before != ns["rt_file_hash"](checkpoint):
        raise ValueError("Evaluation changed the weights or the checkpoint")
    return test


def test_gold():
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]
    return {row["id"]: {"intent": row["intent"], "utt": row["utt"], "spans": gold_spans(row)} for row in rows if row["partition"] == "test"}


def per_utterance(gold, predictions, keys):
    table = []
    for key in keys:
        predicted = {(s["slot_type"], s["start"], s["end"], s["text"]) for s in predictions[key]["slots"]}
        spans = gold[key]["spans"]
        table.append((predictions[key]["intent"] == gold[key]["intent"], len(predicted & spans), len(predicted - spans), len(spans - predicted), predicted == spans))
    return np.array(table, dtype=np.float64)


def resampled(table, draws):
    """Intent accuracy and slot F1 for each row of index draws."""
    sums = np.stack([table[draw].sum(axis=0) for draw in draws])
    return 100 * sums[:, 0] / draws.shape[1], 100 * 2 * sums[:, 1] / np.maximum(1, 2 * sums[:, 1] + sums[:, 2] + sums[:, 3])


def interval(values):
    return [round(float(np.percentile(values, 2.5)), 2), round(float(np.percentile(values, 97.5)), 2)]


def report(key, test):
    gold = test_gold()
    predictions = {p["record_id"]: p for p in test["predictions"]}
    keys = list(gold)
    intent, slot = score(gold, predictions, keys)
    if abs(intent - 100 * test["metrics"]["intent_accuracy"]) > 1e-9 or abs(slot - 100 * test["metrics"]["slot_micro_f1"]) > 1e-9:
        raise ValueError("Recomputed test scores differ from the notebook's")
    membership = groups(gold, artifacts(key))
    table = per_utterance(gold, predictions, keys)
    boot_intent, boot_slot = resampled(table, np.random.default_rng(42).integers(len(keys), size=(DRAWS, len(keys))))
    result = {"run": RUNS[key], "stage": "stage2", "split": "test", "intent": round(intent, 2), "slot_f1": round(slot, 2), "joint": round((intent + slot) / 2, 2),
              "exact_match": round(100 * float(np.mean(table[:, 0] * table[:, 4])), 2), "intent_ci95": interval(boot_intent), "slot_f1_ci95": interval(boot_slot),
              "joint_ci95": interval((boot_intent + boot_slot) / 2), "bootstrap_draws": DRAWS, "groups": {}}
    for group in ("no_oov", "nonslot_oov", "slot_oov"):
        subset = [item for item in keys if membership[item] == group]
        group_intent, group_slot = score(gold, predictions, subset)
        result["groups"][group] = {"n": len(subset), "intent": round(group_intent, 2), "slot_f1": round(group_slot, 2)}
    (output(key) / "test_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


def sam_effect():
    """r1 minus r0 on test: the same utterances are drawn for both runs."""
    gold = test_gold()
    loaded = {key: {p["record_id"]: p for p in map(json.loads, (output(key) / "test_predictions.jsonl").read_text(encoding="utf-8").splitlines())} for key in ("r0", "r1")}
    membership = groups(gold, artifacts("r0"))
    rng = np.random.default_rng(42)
    result = {"comparison": "r1 Word2Vec + SAM minus r0 Word2Vec, Stage 2, test", "bootstrap_draws": DRAWS, "groups": {}}
    for group in ("all", "no_oov", "nonslot_oov", "slot_oov"):
        keys = [key for key in gold if group == "all" or membership[key] == group]
        draws = rng.integers(len(keys), size=(DRAWS, len(keys)))
        first, second = (resampled(per_utterance(gold, loaded[key], keys), draws) for key in ("r1", "r0"))
        point_first, point_second = score(gold, loaded["r1"], keys), score(gold, loaded["r0"], keys)
        intent, slot = first[0] - second[0], first[1] - second[1]
        result["groups"][group] = {"n": len(keys), "intent": round(point_first[0] - point_second[0], 2), "intent_ci95": interval(intent),
                                   "slot_f1": round(point_first[1] - point_second[1], 2), "slot_f1_ci95": interval(slot),
                                   "joint": round((point_first[0] + point_first[1] - point_second[0] - point_second[1]) / 2, 2), "joint_ci95": interval((intent + slot) / 2)}
    (ROOT / "test_sam_effect.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    report(sys.argv[1], evaluate(sys.argv[1]))
    if all((output(key) / "test_predictions.jsonl").is_file() for key in RUNS):
        sam_effect()

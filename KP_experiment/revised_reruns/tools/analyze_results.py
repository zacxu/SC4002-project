"""Three checks behind the result tables: paired bootstrap on dev, late learning curves, and the UNK row.

The bootstrap resamples dev utterances, with the same draws for both runs in a pair. It measures how much a
gap depends on which utterances are in dev. It says nothing about seed noise, which one seed cannot show.
Dev only. Checkpoints are this project's own and are read with weights_only=True.

Usage: python -I analyze_results.py
"""
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_results import FULL, ROOT, dev_rows, groups

DRAWS = 5000
PAIRS = [("SAM on Word2Vec", "r1_word2vec_sam", "r0_word2vec_base"), ("SAM on fastText", "r4_fasttext_sam", "r2_fasttext_base"),
         ("SAM on GloVe", "r5_glove_sam", "r3_glove_base"), ("fastText vs Word2Vec, no SAM", "r2_fasttext_base", "r0_word2vec_base"),
         ("GloVe vs Word2Vec, no SAM", "r3_glove_base", "r0_word2vec_base"), ("fastText+SAM vs Word2Vec+SAM", "r4_fasttext_sam", "r1_word2vec_sam"),
         ("GloVe+SAM vs Word2Vec+SAM", "r5_glove_sam", "r1_word2vec_sam"), ("fastText+SAM vs baseline", "r4_fasttext_sam", "r0_word2vec_base"),
         ("GloVe+SAM vs baseline", "r5_glove_sam", "r0_word2vec_base")]


def per_utterance(dev, keys, run, stage):
    folder = ROOT / "runs" / run / "transformer_run_artifacts" / stage
    predictions = {p["record_id"]: p for p in map(json.loads, (folder / "dev_predictions.jsonl").read_text(encoding="utf-8").splitlines())}
    rows = []
    for key in keys:
        predicted = {(s["slot_type"], s["start"], s["end"], s["text"]) for s in predictions[key]["slots"]}
        rows.append((predictions[key]["intent"] == dev[key]["intent"], len(predicted & dev[key]["spans"]),
                     len(predicted - dev[key]["spans"]), len(dev[key]["spans"] - predicted)))
    return np.array(rows, dtype=np.float64)


def scores(table, draws):
    """Intent accuracy and slot F1 for each row of index draws."""
    sums = table[draws].sum(axis=1)
    intent = 100 * sums[:, 0] / draws.shape[1]
    slot = 100 * 2 * sums[:, 1] / np.maximum(1, 2 * sums[:, 1] + sums[:, 2] + sums[:, 3])
    return intent, slot


def bootstrap(dev, membership):
    keys = list(dev)
    oov_keys = [key for key in keys if membership[key] == "slot_oov"]
    rng = np.random.default_rng(42)
    all_draws = rng.integers(len(keys), size=(DRAWS, len(keys)))
    oov_draws = rng.integers(len(oov_keys), size=(DRAWS, len(oov_keys)))
    print(f"Paired bootstrap, {DRAWS} draws, 95% intervals of the difference (first minus second)")
    for stage in ("stage1", "stage2"):
        joint, oov = {}, {}
        for run in FULL:
            intent, slot = scores(per_utterance(dev, keys, run, stage), all_draws)
            joint[run] = (intent + slot) / 2
            oov[run] = scores(per_utterance(dev, oov_keys, run, stage), oov_draws)[1]
        for label, first, second in PAIRS:
            a, b = joint[first] - joint[second], oov[first] - oov[second]
            print(f"{stage} {label:30} joint {a.mean():+.2f} [{np.percentile(a, 2.5):+.2f}, {np.percentile(a, 97.5):+.2f}]"
                  f"  slot-OOV F1 {b.mean():+.2f} [{np.percentile(b, 2.5):+.2f}, {np.percentile(b, 97.5):+.2f}]")


def curves():
    print("\nLearning curves: dev slot F1 at epochs 5, 10, 15, 18, 19, 20; last train loss; selected epoch")
    for stage in ("stage1", "stage2"):
        for run in FULL:
            lines = (ROOT / "runs" / run / "transformer_run_artifacts" / stage / "history.jsonl").read_text(encoding="utf-8").splitlines()
            history = [json.loads(line) for line in lines]
            points = " ".join(f"{100 * history[epoch - 1]['dev_slot_micro_f1']:6.2f}" for epoch in (5, 10, 15, 18, 19, 20))
            best = max(history, key=lambda row: row["selection_score"])["epoch"]
            print(f"{stage} {run:18} {points}  loss {history[-1]['train_loss']:.3f}  selected {best}")


def unk_row():
    print("\nWord table in the selected checkpoint: UNK row norm against the mean norm of train words")
    for stage in ("stage1", "stage2"):
        for run in FULL:
            checkpoint = torch.load(ROOT / "runs" / run / "transformer_run_artifacts" / stage / "best.pt", map_location="cpu", weights_only=True)
            state = checkpoint.get("model_state_dict", checkpoint.get("model_state", checkpoint))
            name = next(key for key, value in state.items() if value.ndim == 2 and value.shape[1] == 300 and value.shape[0] > 5000)
            norms = state[name].norm(dim=1)
            print(f"{stage} {run:18} {name}: UNK {float(norms[1]):.2f}, words {float(norms[2:].mean()):.2f}")


def main():
    dev = dev_rows()
    bootstrap(dev, groups(dev, ROOT / "runs" / FULL[0] / "transformer_run_artifacts"))
    curves()
    unk_row()


if __name__ == "__main__":
    main()

"""Collect dev results of finished reruns into results_summary.csv and print comparison tables.

Groups follow the notebook's analysis: an utterance is Slot-Train-OOV if a word unseen in train falls
inside a gold slot, NonSlot-Train-OOV if unseen words fall only outside slots, No-Train-OOV otherwise.
Dev only: no test data is read beyond what the dataset file holds, and the test rows are skipped.

Usage: python -I collect_results.py [run names...]     default: the six full runs
"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[0]).resolve().parents[1]
FULL = ["r0_word2vec_base", "r1_word2vec_sam", "r2_fasttext_base", "r3_glove_base", "r4_fasttext_sam", "r5_glove_sam"]
ANNOTATION = re.compile(r"\[([A-Za-z_][A-Za-z_0-9]*)[ \t]*:[ \t]*(.*?)\]", re.DOTALL)
TIE_MARGIN = 0.8    # project rule: one-seed dev joint gaps below this are ties
RECORDED = {"stage1": (84.16, 69.56), "stage2": (85.83, 77.76)}     # notebook's recorded RTX 3060 Ti run


def gold_spans(row):
    spans, cursor, plain = set(), 0, 0
    for match in ANNOTATION.finditer(row["annot_utt"]):
        plain += match.start() - cursor
        spans.add((match.group(1), plain, plain + len(match.group(2)), match.group(2)))
        plain += len(match.group(2))
        cursor = match.end()
    return spans


def dev_rows():
    rows = [json.loads(line) for line in (ROOT / "cache/massive/en-US.jsonl").read_text(encoding="utf-8").splitlines()]
    return {row["id"]: {"intent": row["intent"], "utt": row["utt"], "spans": gold_spans(row)}
            for row in rows if row["partition"] == "dev"}


def groups(dev, artifacts):
    pattern = re.compile(json.loads((artifacts / "data_statistics.json").read_text(encoding="utf-8"))["tokenizer_pattern"], re.UNICODE)
    vocabulary = json.loads((artifacts / "mappings.json").read_text(encoding="utf-8"))["word_to_id"]
    result = {}
    for key, row in dev.items():
        unseen = [m for m in pattern.finditer(row["utt"]) if m.group(0).casefold() not in vocabulary]
        in_slot = any(m.end() > start and m.start() < end for m in unseen for _, start, end, _ in row["spans"])
        result[key] = "slot_oov" if in_slot else ("nonslot_oov" if unseen else "no_oov")
    return result


def score(dev, predictions, keys):
    correct = tp = fp = fn = 0
    for key in keys:
        predicted = {(s["slot_type"], s["start"], s["end"], s["text"]) for s in predictions[key]["slots"]}
        correct += predictions[key]["intent"] == dev[key]["intent"]
        tp += len(predicted & dev[key]["spans"])
        fp += len(predicted - dev[key]["spans"])
        fn += len(dev[key]["spans"] - predicted)
    return 100 * correct / len(keys), 100 * 2 * tp / max(1, 2 * tp + fp + fn)


def collect(run, dev):
    artifacts = ROOT / "runs" / run / "transformer_run_artifacts"
    embedding = json.loads((artifacts / "embedding_initialization.json").read_text(encoding="utf-8"))
    config = json.loads((artifacts / "run_configuration.json").read_text(encoding="utf-8"))
    membership = groups(dev, artifacts)
    rows = []
    for stage in ("stage1", "stage2"):
        folder = artifacts / stage
        summary = json.loads((folder / "final_summary.json").read_text(encoding="utf-8"))
        predictions = {p["record_id"]: p for p in map(json.loads, (folder / "dev_predictions.jsonl").read_text(encoding="utf-8").splitlines())}
        metrics = summary["dev"]["metrics"]
        intent, slot = 100 * metrics["intent_accuracy"], 100 * metrics["slot_micro_f1"]
        check_intent, check_slot = score(dev, predictions, list(dev))
        if abs(check_intent - intent) > 1e-9 or abs(check_slot - slot) > 1e-9:
            raise ValueError(f"{run} {stage}: recomputed scores differ from the notebook's")
        row = {"run": run, "stage": stage, "embedding": embedding["source"],
               "sam_rho": config[stage.upper() + "_CONFIG"]["training"]["sam_rho"] if "sam_rho" in config[stage.upper() + "_CONFIG"]["training"] else "",
               "intent": round(intent, 2), "slot_f1": round(slot, 2), "joint": round((intent + slot) / 2, 2),
               "selected_epoch": summary["selected_epoch"], "epochs": summary["epochs_trained"],
               "seconds": round(summary["training_workflow_seconds"]), "peak_vram_mb": round(summary["peak_allocated_cuda_bytes"] / 2 ** 20),
               "params": summary["parameter_counts"]["total"], "coverage": round(100 * embedding.get("pretrained_coverage", 0), 1),
               "mean_vector_norm": round(embedding["mean_vector_norm"], 2) if "mean_vector_norm" in embedding else ""}
        for group in ("no_oov", "nonslot_oov", "slot_oov"):
            keys = [key for key, value in membership.items() if value == group]
            row[group + "_n"] = len(keys)
            row[group + "_intent"], row[group + "_slot_f1"] = (round(value, 2) for value in score(dev, predictions, keys))
        rows.append(row)
    return rows


def main(runs):
    dev = dev_rows()
    rows = [row for run in runs for row in collect(run, dev)]
    target = ROOT / "results_summary.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", target)
    print(f"{'run':20} {'stage':7} {'intent':>7} {'slot':>7} {'joint':>7} {'epoch':>5} {'sec':>5} {'vram':>5} {'cov':>5} {'norm':>5}"
          f" {'n/slotOOV':>9} {'slotOOV-F1':>10} {'noOOV-F1':>9}")
    for r in rows:
        print(f"{r['run']:20} {r['stage']:7} {r['intent']:7.2f} {r['slot_f1']:7.2f} {r['joint']:7.2f} {r['selected_epoch']:5} {r['seconds']:5} {r['peak_vram_mb']:5}"
              f" {r['coverage']:5} {r['mean_vector_norm']:>5} {r['slot_oov_n']:9} {r['slot_oov_slot_f1']:10.2f} {r['no_oov_slot_f1']:9.2f}")
    print("\nGroup sizes:", {g: rows[0][g + "_n"] for g in ("no_oov", "nonslot_oov", "slot_oov")}, "(the notebook reports 1669 / 149 / 215)")
    print("Notebook's recorded RTX 3060 Ti run (intent, slot):", RECORDED)
    index = {(r["run"], r["stage"]): r for r in rows}
    pairs = [("SAM on Word2Vec", "r1_word2vec_sam", "r0_word2vec_base"), ("SAM on fastText", "r4_fasttext_sam", "r2_fasttext_base"),
             ("SAM on GloVe", "r5_glove_sam", "r3_glove_base"), ("fastText vs Word2Vec, no SAM", "r2_fasttext_base", "r0_word2vec_base"),
             ("GloVe vs Word2Vec, no SAM", "r3_glove_base", "r0_word2vec_base"), ("fastText+SAM vs Word2Vec+SAM", "r4_fasttext_sam", "r1_word2vec_sam"),
             ("GloVe+SAM vs Word2Vec+SAM", "r5_glove_sam", "r1_word2vec_sam"), ("fastText+SAM vs baseline", "r4_fasttext_sam", "r0_word2vec_base"),
             ("GloVe+SAM vs baseline", "r5_glove_sam", "r0_word2vec_base")]
    print("\nDifferences (first minus second), points; |joint| below", TIE_MARGIN, "is a tie on one seed")
    for label, first, second in pairs:
        for stage in ("stage1", "stage2"):
            if (first, stage) in index and (second, stage) in index:
                a, b = index[first, stage], index[second, stage]
                joint = a["joint"] - b["joint"]
                print(f"{label:30} {stage}: intent {a['intent'] - b['intent']:+.2f}  slot {a['slot_f1'] - b['slot_f1']:+.2f}  joint {joint:+.2f}"
                      f"  slot-OOV F1 {a['slot_oov_slot_f1'] - b['slot_oov_slot_f1']:+.2f}  {'TIE' if abs(joint) < TIE_MARGIN else ''}")


if __name__ == "__main__":
    main(sys.argv[1:] or FULL)

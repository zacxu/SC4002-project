"""Exact-span slot micro-F1, intent accuracy, OOV subgroup tables, bootstrap, matching."""
from collections import Counter

import numpy as np
import pandas as pd
import torch

from .config import DEVICE, SEED
from .data import to_bio, parse_annotated_utterance

GROUPS = ["No-Train-OOV", "NonSlot-Train-OOV", "Slot-Train-OOV"]


def bio_to_spans(tags):
    """(slot_type, start, end_exclusive); a stray I-X opens a new span (hand-off rule)."""
    spans, cur, start = [], None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        if tag == "O":
            if cur is not None:
                spans.append((cur, start, i))
            cur = start = None
            continue
        prefix, typ = tag.split("-", 1)
        if prefix == "B" or cur != typ:
            if cur is not None:
                spans.append((cur, start, i))
            cur, start = typ, i
    return spans


def slot_counts(gold_lists, pred_lists):
    TP = FP = FN = 0
    for g, p in zip(gold_lists, pred_lists):
        g, p = Counter(bio_to_spans(g)), Counter(bio_to_spans(p))
        TP += sum((g & p).values())
        FP += sum((p - g).values())
        FN += sum((g - p).values())
    return TP, FP, FN


def micro_f1(gold_lists, pred_lists):
    TP, FP, FN = slot_counts(gold_lists, pred_lists)
    denom = 2 * TP + FP + FN
    return 0.0 if denom == 0 else 100.0 * 2 * TP / denom


def to_device(batch):
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def predict(model, loader, assets):
    """Runs the model over a split. Gold and predicted tags are returned as BIO strings."""
    model.eval()
    rows, losses = [], []
    id2tag = model.id2tag
    for raw in loader:
        b = to_device(raw)
        il, sl = model(b)
        losses.append(model.loss(il, sl, b).item())
        pred_int = il.argmax(-1).cpu().tolist()
        pred_tags = model.decode_tags(sl, b["attention_mask"])
        lengths = raw["attention_mask"].long().sum(1).tolist()
        for i, n in enumerate(lengths):
            rows.append({
                "id": raw["ids"][i],
                "gold_intent_id": int(raw["intent_ids"][i]),
                "pred_intent_id": pred_int[i],
                "gold_tags": [to_bio(id2tag[x]) for x in raw["slot_ids"][i, :n].tolist()],
                "pred_tags": pred_tags[i],
            })
    frame = pd.DataFrame(rows)
    out = metrics_from_frame(frame)
    out["loss"] = float(np.mean(losses))
    out["pred_frame"] = frame
    return out


def metrics_from_frame(frame):
    acc = 100.0 * float(np.mean(frame["gold_intent_id"].to_numpy() == frame["pred_intent_id"].to_numpy()))
    f1 = micro_f1(frame["gold_tags"].tolist(), frame["pred_tags"].tolist())
    exact = 100.0 * float(np.mean([(gi == pi) and (bio_to_spans(g) == bio_to_spans(p)) for gi, pi, g, p in zip(
        frame["gold_intent_id"], frame["pred_intent_id"], frame["gold_tags"], frame["pred_tags"])]))
    return {"n": len(frame), "intent_accuracy": acc, "slot_micro_f1": f1,
            "joint_score": (acc + f1) / 2.0, "exact_match": exact}


def oov_table(frame, oov_meta):
    merged = frame.merge(oov_meta, on="id", how="left", validate="one_to_one")
    rows = []
    for g in GROUPS:
        sub = merged[merged["train_oov_group"] == g]
        rows.append({"group": g, **{k: v for k, v in metrics_from_frame(sub).items() if k != "joint_score"}})
    rows.append({"group": "Overall", **{k: v for k, v in metrics_from_frame(merged).items() if k != "joint_score"}})
    return pd.DataFrame(rows), merged


def paired_bootstrap(base, other, group=None, n_boot=1000, seed=SEED):
    """CI of (other - base) slot F1 and intent accuracy on the same utterances."""
    a = base.set_index("id")
    b = other.set_index("id").loc[a.index]
    if group is not None:
        keep = (a["train_oov_group"] == group).to_numpy()
        a, b = a[keep], b[keep]
    rng = np.random.default_rng(seed)

    def counts(fr):
        return np.array([slot_counts([g], [p]) for g, p in zip(fr["gold_tags"], fr["pred_tags"])])

    ca, cb = counts(a), counts(b)
    ia = (a["gold_intent_id"] == a["pred_intent_id"]).to_numpy()
    ib = (b["gold_intent_id"] == b["pred_intent_id"]).to_numpy()

    def f1(c):
        tp, fp, fn = c.sum(0)
        return 100.0 * 2 * tp / max(2 * tp + fp + fn, 1)

    n = len(a)
    d_f1, d_int = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d_f1.append(f1(cb[idx]) - f1(ca[idx]))
        d_int.append(100.0 * (ib[idx].mean() - ia[idx].mean()))
    return {"group": group or "Overall", "n": n,
            "slot_f1_gain": f1(cb) - f1(ca),
            "slot_f1_ci_low": float(np.percentile(d_f1, 2.5)), "slot_f1_ci_high": float(np.percentile(d_f1, 97.5)),
            "intent_gain": 100.0 * (ib.mean() - ia.mean()),
            "intent_ci_low": float(np.percentile(d_int, 2.5)), "intent_ci_high": float(np.percentile(d_int, 97.5))}


def matched_control(merged):
    """Difficulty-matched OOV gap. Each Slot-Train-OOV utterance is matched on length and slot count,
    standardised as in the hand-off, to ALL No-Train-OOV utterances at the minimum distance, each with
    weight 1/k. Picking one row (the hand-off's method) let one arbitrary utterance stand in for up to
    23 queries, which made the gap mostly tie-breaking noise, so old and new gaps are not comparable."""
    slot = merged[merged["train_oov_group"] == "Slot-Train-OOV"]
    pool = merged[merged["train_oov_group"] == "No-Train-OOV"]
    feats = ["n_tokens", "n_gold_slots"]
    x_slot, x_pool = slot[feats].to_numpy(dtype=float), pool[feats].to_numpy(dtype=float)
    sd = np.vstack([x_slot, x_pool]).std(axis=0)
    sd[sd == 0] = 1.0
    z_pool = x_pool / sd
    weights = np.zeros(len(pool))
    keys, counts = np.unique(x_slot, axis=0, return_counts=True)
    for key, c in zip(keys, counts):
        d = ((z_pool - key / sd) ** 2).sum(1)
        tied = np.where(d <= d.min() + 1e-12)[0]
        weights[tied] += c / len(tied)
    per = np.array([slot_counts([g], [p]) for g, p in zip(pool["gold_tags"], pool["pred_tags"])], dtype=float)
    tp, fp, fn = (per * weights[:, None]).sum(0)
    f_m = 100.0 * 2 * tp / max(2 * tp + fp + fn, 1e-9)
    f_s = metrics_from_frame(slot)["slot_micro_f1"]
    return {"matched_no_oov_f1": f_m, "slot_oov_f1": f_s, "gap": f_m - f_s,
            "avg_tokens": float(x_slot[:, 0].mean()), "avg_slots": float(x_slot[:, 1].mean()),
            "matched_avg_tokens": float((x_pool[:, 0] * weights).sum() / weights.sum()),
            "matched_avg_slots": float((x_pool[:, 1] * weights).sum() / weights.sum())}


def error_cases(merged, assets, split_df):
    """Utterances the model gets wrong, with tokens, gold and predicted analyses."""
    lookup = split_df.set_index("id")
    rows = []
    for r in merged.itertuples(index=False):
        int_ok = r.gold_intent_id == r.pred_intent_id
        gold_spans, pred_spans = bio_to_spans(r.gold_tags), bio_to_spans(r.pred_tags)
        slot_ok = Counter(gold_spans) == Counter(pred_spans)
        if int_ok and slot_ok:
            continue
        toks = parse_annotated_utterance(lookup.loc[r.id, "annot_utt"])["tokens"]
        rows.append({
            "id": r.id, "group": r.train_oov_group, "utt": lookup.loc[r.id, "utt"],
            "gold_intent": assets.intent_labels[r.gold_intent_id],
            "pred_intent": assets.intent_labels[r.pred_intent_id],
            "gold_slots": [(t, " ".join(toks[s:e])) for t, s, e in gold_spans],
            "pred_slots": [(t, " ".join(toks[s:e])) for t, s, e in pred_spans],
            "error": ("intent" if not int_ok else "") + ("+" if not int_ok and not slot_ok else "") + ("slot" if not slot_ok else ""),
        })
    return pd.DataFrame(rows)

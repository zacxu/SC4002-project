"""Q6: train-derived intent -> slot-type compatibility, and what a hard mask would do on dev."""
from common import *

train, m, intents = load_all()
_, a1, _ = load_all(A1_FRAME)
cnt = {}
for r in train.itertuples(index=False):
    c = cnt.setdefault(r.intent, Counter())
    for t, _, _ in bio_to_spans(r.bio):
        c[t] += 1
all_types = sorted({t for c in cnt.values() for t in c})
print("types in train:", len(all_types), "| intents:", len(cnt))
n_types = {i: len(cnt.get(i, {})) for i in intents}
vals = np.array(list(n_types.values()))
cells = int(vals.sum())
print(f"intent-type cells seen: {cells} of {60 * len(all_types)} = {100 * cells / (60 * len(all_types)):.1f}% | "
      f"types per intent: min {vals.min()}, median {np.median(vals):.0f}, mean {vals.mean():.1f}, max {vals.max()}")
tr_n = train["intent"].value_counts()
rows = []
for i in intents:
    c = cnt.get(i, Counter())
    rows.append({"intent": i, "train_utts": int(tr_n[i]), "slots": sum(c.values()), "n_types": len(c),
                 "types": "; ".join(f"{k}:{v}" for k, v in c.most_common())})
M = pd.DataFrame(rows)
save(M, "q6_intent_slot_matrix.csv")
show(M.drop(columns="types").sort_values("n_types", ascending=False).head(8))
show(M.drop(columns="types").sort_values("n_types").head(8))
tcount = Counter(t for c in cnt.values() for t in c)
print("intents per type: top", tcount.most_common(12))
print("singleton cells (seen once in train):", sum(1 for c in cnt.values() for v in c.values() if v == 1),
      "| cells seen <=2:", sum(1 for c in cnt.values() for v in c.values() if v <= 2))


def allowed(intent, t, min_count=1):
    return cnt.get(intent, {}).get(t, 0) >= min_count


# ---- dev gold slots under the gold intent
rows = []
for r in m.itertuples(index=False):
    for t, s, e in bio_to_spans(r.bio):
        rows.append({"id": r.id, "type": t, "value": " ".join(r.toks[s:e]).casefold(), "intent": r.gold_intent, "pred_intent": r.pred_intent,
                     "n_gold": cnt.get(r.gold_intent, {}).get(t, 0), "n_pred": cnt.get(r.pred_intent, {}).get(t, 0), "utt": r.utt})
GS = pd.DataFrame(rows)
print("\ndev gold slots:", len(GS))
rows = []
for k in (1, 2, 3, 5):
    v = GS["n_gold"] < k
    rows.append({"min_train_count": k, "blocked_with_gold_intent": int(v.sum()), "pct": round(100 * v.mean(), 2),
                 "blocked_with_S1_pred_intent": int((GS["n_pred"] < k).sum()), "pct_pred": round(100 * (GS["n_pred"] < k).mean(), 2)})
B = pd.DataFrame(rows)
show(B)
save(B, "q6_gold_blocked.csv")
v = GS[GS["n_gold"] == 0]
print("gold slots never seen with gold intent, top (intent,type):", Counter(zip(v["intent"], v["type"])).most_common(25))
show(v[["intent", "type", "value", "utt"]].head(25))
save(v[["id", "intent", "type", "value", "utt"]], "q6_gold_slots_never_seen.csv")

# ---- Stage-1 false positives vs predicted / gold intent
def fp_table(frame, label):
    rows = []
    for r in frame.itertuples(index=False):
        g, p = Counter(bio_to_spans(r.gold_tags)), Counter(bio_to_spans(r.pred_tags))
        for (t, s, e), c in (p - g).items():
            rows.append({"kind": "fp", "type": t, "pred_never": cnt.get(r.pred_intent, {}).get(t, 0) == 0,
                         "gold_never": cnt.get(r.gold_intent, {}).get(t, 0) == 0, "int_ok": r.int_ok})
        for (t, s, e), c in (p & g).items():
            rows.append({"kind": "tp", "type": t, "pred_never": cnt.get(r.pred_intent, {}).get(t, 0) == 0,
                         "gold_never": cnt.get(r.gold_intent, {}).get(t, 0) == 0, "int_ok": r.int_ok})
    D = pd.DataFrame(rows)
    fp, tp = D[D["kind"] == "fp"], D[D["kind"] == "tp"]
    print(f"\n[{label}] FP spans {len(fp)} | type never seen with predicted intent: {int(fp['pred_never'].sum())} = {100 * fp['pred_never'].mean():.1f}% "
          f"| never seen with gold intent: {int(fp['gold_never'].sum())} = {100 * fp['gold_never'].mean():.1f}%")
    for flag, name in ((True, "intent right"), (False, "intent wrong")):
        s = fp[fp["int_ok"] == flag]
        s2 = tp[tp["int_ok"] == flag]
        print(f"   {name}: FP {len(s)}, blocked by predicted-intent mask {int(s['pred_never'].sum())} = {100 * s['pred_never'].mean():.1f}% "
              f"| TP {len(s2)}, blocked {int(s2['pred_never'].sum())} = {100 * s2['pred_never'].mean():.1f}%")
    print(f"   TP spans {len(tp)} | type never seen with predicted intent: {int(tp['pred_never'].sum())} = {100 * tp['pred_never'].mean():.1f}%")
    print("   blocked FP by type:", fp[fp["pred_never"]]["type"].value_counts().head(10).to_dict())
    return D


def strip_tags(tags, intent, k):
    """Deletes predicted spans whose type was seen fewer than k times with `intent`."""
    out = ["O"] * len(tags)
    for t, s, e in bio_to_spans(tags):
        if cnt.get(intent, {}).get(t, 0) >= k:
            out[s] = "B-" + t
            for i in range(s + 1, e):
                out[i] = "I-" + t
    return out


def sim(frame, label):
    base = slot_counts(frame["gold_tags"], frame["pred_tags"])
    b_f1 = f1_from(*base)
    rows = [{"model": label, "mask": "none", "k": 0, "tp": base[0], "fp": base[1], "fn": base[2], "f1": round(b_f1, 2), "gain": 0.0}]
    for name, col in (("gold intent (oracle)", "gold_intent"), ("predicted intent", "pred_intent")):
        for k in (1, 2, 3):
            pr = [strip_tags(t, i, k) for t, i in zip(frame["pred_tags"], frame[col])]
            c = slot_counts(frame["gold_tags"], pr)
            rows.append({"model": label, "mask": name, "k": k, "tp": c[0], "fp": c[1], "fn": c[2], "f1": round(f1_from(*c), 2),
                         "gain": round(f1_from(*c) - b_f1, 2)})
    return rows


D1 = fp_table(m, "Stage 1")
D2 = fp_table(a1, "A1_ft_recipe")
S = pd.DataFrame(sim(m, "Stage 1") + sim(a1, "A1_ft_recipe"))
print("\nPost-hoc span deletion (a lower bound for masked decoding, which could also re-label a blocked span):")
show(S)
save(S, "q6_mask_simulation.csv")

# ---- reverse direction: do predicted slots flag intent errors
print("\nReverse direction: predicted slots vs predicted intent")
for frame, label in ((m, "Stage 1"), (a1, "A1_ft_recipe")):
    flag = np.array([any(cnt.get(i, {}).get(t, 0) == 0 for t, _, _ in bio_to_spans(tags)) for i, tags in zip(frame["pred_intent"], frame["pred_tags"])])
    ok = frame["int_ok"].to_numpy()
    print(f"[{label}] utterances with a predicted slot incompatible with the predicted intent: {int(flag.sum())} = {100 * flag.mean():.1f}% | "
          f"intent wrong among flagged {100 * (~ok[flag]).mean():.1f}% vs unflagged {100 * (~ok[~flag]).mean():.1f}% | "
          f"share of intent errors flagged {100 * flag[~ok].mean():.1f}%")
# gold slot set vs predicted intent, an upper bound on what slot evidence could veto
gold_flag = np.array([any(cnt.get(i, {}).get(t, 0) == 0 for t, _, _ in bio_to_spans(tags)) for i, tags in zip(m["pred_intent"], m["gold_tags"])])
ok = m["int_ok"].to_numpy()
print(f"[gold slots vs Stage-1 predicted intent] flagged {int(gold_flag.sum())} | among intent errors: {int(gold_flag[~ok].sum())} of {int((~ok).sum())} = {100 * gold_flag[~ok].mean():.1f}% "
      f"| among intent-correct: {int(gold_flag[ok].sum())} of {int(ok.sum())} = {100 * gold_flag[ok].mean():.1f}%")
n_cand = []
for tags in m["gold_tags"]:
    ts = {t for t, _, _ in bio_to_spans(tags)}
    if ts:
        n_cand.append(sum(all(cnt.get(i, {}).get(t, 0) > 0 for t in ts) for i in intents))
print(f"utterances with gold slots: {len(n_cand)} | intents compatible with the gold slot set: mean {np.mean(n_cand):.1f}, median {np.median(n_cand):.0f}, "
      f"exactly one: {sum(x == 1 for x in n_cand)}, zero: {sum(x == 0 for x in n_cand)}")
mem("end")

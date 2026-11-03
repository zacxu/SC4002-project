"""Q2: Stage-1 intent confusions on dev, ambiguity evidence, intent accuracy per OOV group."""
from common import *

train, m, intents = load_all()
m = m.merge(pd.read_pickle(OUT / "q1_dev_flags.pkl"), on="id")
_, a1, _ = load_all(A1_FRAME)
m = m.merge(a1[["id", "pred_intent", "int_ok"]].rename(columns={"pred_intent": "a1_pred", "int_ok": "a1_ok"}), on="id")
err = m[~m["int_ok"]].copy()
N = len(err)
print(f"Stage-1 intent errors: {N} of {len(m)} ({100 * N / len(m):.2f}%) | A1 errors: {int((~m['a1_ok']).sum())}")

tr_n = train["intent"].value_counts()
dv_n = m["intent"].value_counts()
t_int = {}
for r in train.itertuples(index=False):
    t_int.setdefault(r.tmpl, Counter())[r.intent] += 1
pair_tmpl = Counter()
for c in t_int.values():
    ks = sorted(c)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            pair_tmpl[(ks[i], ks[j])] += 1

und = Counter(tuple(sorted(x)) for x in zip(err["gold_intent"], err["pred_intent"]))
dire = Counter(zip(err["gold_intent"], err["pred_intent"]))
rows, cum = [], 0
for (a, b), c in und.most_common(20):
    cum += c
    sub = err[((err["gold_intent"] == a) & (err["pred_intent"] == b)) | ((err["gold_intent"] == b) & (err["pred_intent"] == a))]
    rows.append({"pair": f"{a} | {b}", "n": c, "share_pct": round(100 * c / N, 1), "cum_pct": round(100 * cum / N, 1),
                 "a_to_b": dire[(a, b)], "b_to_a": dire[(b, a)], "same_scenario": a.split("_")[0] == b.split("_")[0],
                 "train_a": int(tr_n[a]), "train_b": int(tr_n[b]), "dev_a": int(dv_n.get(a, 0)), "dev_b": int(dv_n.get(b, 0)),
                 "tmpl_shared_train": pair_tmpl[(a, b)],
                 "a1_also_wrong": int((~sub["a1_ok"]).sum()), "a1_same_label": int((sub["a1_pred"] == sub["pred_intent"]).sum()),
                 "no_oov": int((sub["train_oov_group"] == "No-Train-OOV").sum()),
                 "nonslot_oov": int((sub["train_oov_group"] == "NonSlot-Train-OOV").sum()),
                 "slot_oov": int((sub["train_oov_group"] == "Slot-Train-OOV").sum())})
tbl = pd.DataFrame(rows)
show(tbl)
save(tbl, "q2_top20_confused_pairs.csv")
print("distinct unordered pairs:", len(und), "| singletons:", sum(v == 1 for v in und.values()))
print("top 20 directed:", dire.most_common(20))

# ---- scenario level
gs = err["gold_intent"].str.split("_").str[0]
ps = err["pred_intent"].str.split("_").str[0]
same = (gs == ps)
print(f"errors inside the same scenario: {int(same.sum())} = {100 * same.mean():.1f}% | across scenarios: {int((~same).sum())}")
print(f"scenario-level accuracy of Stage 1: {100 * (m['gold_intent'].str.split('_').str[0] == m['pred_intent'].str.split('_').str[0]).mean():.2f}")
scen_err = Counter(zip(gs, ps))
print("top cross-scenario error flows:", [(k, v) for k, v in scen_err.most_common(40) if k[0] != k[1]][:15])
print("scenario gold-side error counts:", Counter(gs).most_common(18))

# ---- all errors of the top pairs, for reading
print("\n=== errors of the top pairs ===")
ex_rows = []
for (a, b), c in und.most_common(5):
    sub = err[((err["gold_intent"] == a) & (err["pred_intent"] == b)) | ((err["gold_intent"] == b) & (err["pred_intent"] == a))]
    print(f"\n--- {a} | {b}: {c}")
    for r in sub.itertuples(index=False):
        print(f"  gold={r.gold_intent:26s} pred={r.pred_intent:26s} a1={'ok ' if r.a1_ok else 'bad'} {r.train_oov_group[:7]:7s} | {r.annot_utt}")
        ex_rows.append({"pair": f"{a} | {b}", "gold": r.gold_intent, "pred": r.pred_intent, "a1_ok": r.a1_ok,
                        "group": r.train_oov_group, "annot_utt": r.annot_utt})
save(pd.DataFrame(ex_rows), "q2_top5_pair_errors.csv")

# ---- ambiguity indicators over all errors
print("\n=== ambiguity indicators over all intent errors ===")
pred_in_tmpl = [p in t_int.get(k, {}) for k, p in zip(err["tmpl"], err["pred_intent"])]
gold_in_tmpl = [g in t_int.get(k, {}) for k, g in zip(err["tmpl"], err["gold_intent"])]
print("template seen in train:", int(err["t_seen"].sum()), "| template seen with the predicted label:", sum(pred_in_tmpl),
      "| with the gold label:", sum(gold_in_tmpl))
print("errors also made by A1:", int((~err["a1_ok"]).sum()), f"= {100 * (~err['a1_ok']).mean():.1f}%",
      "| A1 agrees on the same wrong label:", int((err["a1_pred"] == err["pred_intent"]).sum()))

# ---- intent accuracy per group, error share, direct standardisation
print("\n=== intent accuracy per OOV group ===")
no = m[m["train_oov_group"] == "No-Train-OOV"]
acc_by_int = no.groupby("gold_intent")["int_ok"].mean()
ov = no["int_ok"].mean()
len_bucket = lambda s: pd.cut(s, [0, 4, 8, 12, 100], labels=["1-4", "5-8", "9-12", "13+"])
m["lb"] = len_bucket(m["n_tok"])
acc_by_len = m[m["train_oov_group"] == "No-Train-OOV"].groupby("lb", observed=True)["int_ok"].mean()
rows = []
for g in GROUPS + ["Overall"]:
    s = m if g == "Overall" else m[m["train_oov_group"] == g]
    exp_int = s["gold_intent"].map(acc_by_int).fillna(ov).mean()
    exp_len = s["lb"].map(acc_by_len).astype(float).mean()
    rows.append({"group": g, "n": len(s), "errors": int((~s["int_ok"]).sum()),
                 "error_share_pct": round(100 * (~s["int_ok"]).sum() / N, 1), "S1_acc": round(100 * s["int_ok"].mean(), 2),
                 "A1_acc": round(100 * s["a1_ok"].mean(), 2),
                 "expected_acc_from_intent_mix": round(100 * exp_int, 2), "expected_acc_from_length_mix": round(100 * exp_len, 2),
                 "mean_tokens": round(s["n_tok"].mean(), 2)})
gt = pd.DataFrame(rows)
show(gt)
save(gt, "q2_intent_by_group.csv")
for g in ("NonSlot-Train-OOV", "Slot-Train-OOV"):
    s = m[m["train_oov_group"] == g]
    print(g, "top intents:", s["gold_intent"].value_counts().head(8).to_dict())
    print(g, "error flows:", Counter(zip(s[~s["int_ok"]]["gold_intent"], s[~s["int_ok"]]["pred_intent"])).most_common(8))
lost = {g: float((no["int_ok"].mean() - m[m["train_oov_group"] == g]["int_ok"].mean()) * (m["train_oov_group"] == g).sum())
        for g in ("NonSlot-Train-OOV", "Slot-Train-OOV")}
print("utterances lost vs No-OOV accuracy:", {k: round(v, 1) for k, v in lost.items()},
      "| overall points:", round(100 * sum(lost.values()) / len(m), 2))
adj = {}
for g in ("NonSlot-Train-OOV", "Slot-Train-OOV"):
    s = m[m["train_oov_group"] == g]
    adj[g] = float((s["gold_intent"].map(acc_by_int).fillna(ov).mean() - s["int_ok"].mean()) * len(s))
print("utterances lost vs intent-mix-standardised expectation:", {k: round(v, 1) for k, v in adj.items()},
      "| overall points:", round(100 * sum(adj.values()) / len(m), 2))

# ---- per intent
rows = []
for it in intents:
    gi = m[m["gold_intent"] == it]
    pi = m[m["pred_intent"] == it]
    rows.append({"intent": it, "train": int(tr_n[it]), "dev": len(gi), "tp": int(gi["int_ok"].sum()),
                 "recall": round(100 * gi["int_ok"].mean(), 1) if len(gi) else None,
                 "precision": round(100 * pi["int_ok"].mean(), 1) if len(pi) else None,
                 "errors_gold_side": int((~gi["int_ok"]).sum()), "errors_pred_side": int((~pi["int_ok"]).sum())})
pi = pd.DataFrame(rows).sort_values("errors_gold_side", ascending=False)
show(pi)
save(pi, "q2_per_intent.csv")
print("gold-side error share of top 10 intents:", round(100 * pi["errors_gold_side"].head(10).sum() / N, 1))
mem("end")

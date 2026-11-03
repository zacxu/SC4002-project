"""Q3: Stage-1 slot error taxonomy on dev, per group and per type. Every FP and FN is assigned to one category."""
from common import *

train, m, intents = load_all()
train_lex = {t.casefold() for toks in train["toks"] for t in toks if is_lexical(t)}
tr_type_n = Counter(t for b in train["bio"] for t, _, _ in bio_to_spans(b))


def overlap(a, b):
    return a[1] < b[2] and b[1] < a[2]


G_CATS = ["exact", "wrong_type_same_span", "boundary_same_type", "overlap_other", "missed"]
P_CATS = ["exact", "wrong_type_same_span", "boundary_same_type", "overlap_other", "spurious"]


def classify(gold, pred):
    gs, ps = set(gold), set(pred)
    assert len(gs) == len(gold) and len(ps) == len(pred)
    ex = gs & ps
    gl, pl = gs - ex, ps - ex
    gout, pout = [(g, "exact", None) for g in ex], [(p, "exact", None) for p in ex]
    for p in pl:
        ov = [g for g in gl if overlap(p, g)]
        if any(g[1] == p[1] and g[2] == p[2] for g in ov):
            pout.append((p, "wrong_type_same_span", [g for g in ov if g[1] == p[1] and g[2] == p[2]][0]))
        elif any(g[0] == p[0] for g in ov):
            pout.append((p, "boundary_same_type", [g for g in ov if g[0] == p[0]][0]))
        elif ov:
            pout.append((p, "overlap_other", ov[0]))
        else:
            pout.append((p, "spurious", None))
    for g in gl:
        ov = [p for p in pl if overlap(p, g)]
        if any(p[1] == g[1] and p[2] == g[2] for p in ov):
            gout.append((g, "wrong_type_same_span", [p for p in ov if p[1] == g[1] and p[2] == g[2]][0]))
        elif any(p[0] == g[0] for p in ov):
            gout.append((g, "boundary_same_type", [p for p in ov if p[0] == g[0]][0]))
        elif ov:
            gout.append((g, "overlap_other", ov[0]))
        else:
            gout.append((g, "missed", None))
    return gout, pout


grows, prows = [], []
for r in m.itertuples(index=False):
    gout, pout = classify(bio_to_spans(r.gold_tags), bio_to_spans(r.pred_tags))
    toks = [t.casefold() for t in r.toks]
    for s, c, o in gout:
        has_oov = any(is_lexical(t) and t not in train_lex for t in toks[s[1]:s[2]])
        grows.append(dict(id=r.id, group=r.train_oov_group, int_ok=r.int_ok, type=s[0], start=s[1], end=s[2], cat=c,
                          value=" ".join(toks[s[1]:s[2]]), oth=None if o is None else (o[0], " ".join(toks[o[1]:o[2]])),
                          has_oov=has_oov, n=s[2] - s[1], oth_len=None if o is None else o[2] - o[1],
                          oth_start=None if o is None else o[1], oth_end=None if o is None else o[2], intent=r.gold_intent,
                          pred_intent=r.pred_intent))
    for s, c, o in pout:
        prows.append(dict(id=r.id, group=r.train_oov_group, int_ok=r.int_ok, type=s[0], start=s[1], end=s[2], cat=c,
                          value=" ".join(toks[s[1]:s[2]]), oth=None if o is None else (o[0], " ".join(toks[o[1]:o[2]])),
                          n=s[2] - s[1], intent=r.gold_intent, pred_intent=r.pred_intent))
G = pd.DataFrame(grows)
P = pd.DataFrame(prows)
TP = int((G["cat"] == "exact").sum())
FN = int((G["cat"] != "exact").sum())
FP = int((P["cat"] != "exact").sum())
print(f"reconcile: TP {TP} FP {FP} FN {FN} | F1 {f1_from(TP, FP, FN):.2f}")
assert (TP, FP, FN) == slot_counts(m["gold_tags"], m["pred_tags"])
G.to_pickle(OUT / "q3_gold_spans.pkl")
P.to_pickle(OUT / "q3_pred_spans.pkl")


# ---- overall and per group
def tax(gs, ps, name):
    g = gs["cat"].value_counts()
    p = ps["cat"].value_counts()
    tp, fn, fp = int(g.get("exact", 0)), len(gs) - int(g.get("exact", 0)), len(ps) - int(p.get("exact", 0))
    out = {"group": name, "gold": len(gs), "pred": len(ps), "F1": round(f1_from(tp, fp, fn), 2)}
    for c in G_CATS:
        out["g_" + c] = int(g.get(c, 0))
        out["g%_" + c] = round(100 * g.get(c, 0) / max(len(gs), 1), 1)
    for c in P_CATS:
        out["p_" + c] = int(p.get(c, 0))
        out["p%_" + c] = round(100 * p.get(c, 0) / max(len(ps), 1), 1)
    return out


rows = [tax(G[G["group"] == g], P[P["group"] == g], g) for g in GROUPS] + [tax(G, P, "Overall")]
T = pd.DataFrame(rows)
save(T, "q3_taxonomy_by_group.csv")
print("\nGold view (share of gold spans):")
show(T[["group", "gold", "F1"] + ["g_" + c for c in G_CATS] + ["g%_" + c for c in G_CATS]])
print("\nPrediction view (share of predicted spans):")
show(T[["group", "pred"] + ["p_" + c for c in P_CATS] + ["p%_" + c for c in P_CATS]])

# share of each error category that falls into each group
print("\nShare of each overall error category per group (pct of category, by side):")
rows = []
for side, D, cats in (("gold", G, G_CATS), ("pred", P, P_CATS)):
    for c in cats:
        d = D[D["cat"] == c]["group"].value_counts()
        rows.append({"side": side, "cat": c, "n": int(d.sum()), **{g: round(100 * d.get(g, 0) / max(d.sum(), 1), 1) for g in GROUPS}})
show(pd.DataFrame(rows))
save(pd.DataFrame(rows), "q3_category_by_group.csv")

# ---- what-if F1 if a category vanished
print("\nWhat-if F1 if one error category were fully repaired:")
rows = []
base = f1_from(TP, FP, FN)
gc, pc = G["cat"].value_counts(), P["cat"].value_counts()
for c in ("wrong_type_same_span", "boundary_same_type", "overlap_other"):
    tp2 = TP + gc.get(c, 0)
    f = f1_from(tp2, FP - pc.get(c, 0), FN - gc.get(c, 0))
    rows.append({"repair": c, "gold_side": int(gc.get(c, 0)), "pred_side": int(pc.get(c, 0)), "F1": round(f, 2), "gain": round(f - base, 2)})
f = f1_from(TP, FP - pc.get("spurious", 0), FN)
rows.append({"repair": "spurious removed", "gold_side": 0, "pred_side": int(pc.get("spurious", 0)), "F1": round(f, 2), "gain": round(f - base, 2)})
f = f1_from(TP + gc.get("missed", 0), FP, FN - gc.get("missed", 0))
rows.append({"repair": "missed found", "gold_side": int(gc.get("missed", 0)), "pred_side": 0, "F1": round(f, 2), "gain": round(f - base, 2)})
W = pd.DataFrame(rows)
show(W)
save(W, "q3_what_if_by_category.csv")

# OOV what-if: Slot-Train-OOV group lifted to the No-Train-OOV F1
g_s = G[G["group"] == "Slot-Train-OOV"]
p_s = P[P["group"] == "Slot-Train-OOV"]
tp_s, fn_s, fp_s = int((g_s["cat"] == "exact").sum()), int((g_s["cat"] != "exact").sum()), int((p_s["cat"] != "exact").sum())
for tgt in (71.51, 77.78):
    gold_s = len(g_s)
    new_tp = tgt / 100 * gold_s
    f = f1_from(TP - tp_s + new_tp, FP - fp_s + (gold_s - new_tp), FN - fn_s + (gold_s - new_tp))
    print(f"what-if Slot-Train-OOV group reaches F1 {tgt}: overall F1 {f:.2f} (gain {f - base:.2f}, joint gain {(f - base) / 2:.2f})")
print("Slot-Train-OOV group holds", len(g_s), "of", len(G), "gold spans =", round(100 * len(g_s) / len(G), 1), "%")

# ---- spans containing a Train-OOV token
print("\nGold spans by OOV content:")
rows = []
for name, s in (("span has Train-OOV token", G[G["has_oov"]]),
                ("span without OOV, in Slot-Train-OOV utt", G[(~G["has_oov"]) & (G["group"] == "Slot-Train-OOV")]),
                ("span without OOV, other utts", G[(~G["has_oov"]) & (G["group"] != "Slot-Train-OOV")])):
    v = s["cat"].value_counts()
    rows.append({"subset": name, "n": len(s), **{c: round(100 * v.get(c, 0) / max(len(s), 1), 1) for c in G_CATS}})
show(pd.DataFrame(rows))
save(pd.DataFrame(rows), "q3_oov_span_rates.csv")

# ---- slot quality given intent correctness
print("\nSlot F1 by Stage-1 intent correctness:")
rows = []
for flag in (True, False):
    g_, p_ = G[G["int_ok"] == flag], P[P["int_ok"] == flag]
    rows.append(tax(g_, p_, "intent right" if flag else "intent wrong"))
show(pd.DataFrame(rows)[["group", "gold", "pred", "F1", "g%_exact", "g%_missed", "p%_spurious", "g%_wrong_type_same_span", "g%_boundary_same_type"]])

# ---- boundary detail
print("\nBoundary errors, same type (gold view):")
b = G[G["cat"] == "boundary_same_type"]
kind = []
for r in b.itertuples(index=False):
    a, z, oa, oz = r.start, r.end, r.oth_start, r.oth_end
    kind.append("pred inside gold" if (oa >= a and oz <= z) else "pred contains gold" if (oa <= a and oz >= z) else "shifted")
print(Counter(kind))
print("by type:", b["type"].value_counts().head(12).to_dict())
show(b.assign(pred=b["oth"].map(lambda x: x[1]))[["type", "value", "pred"]].head(30))

# ---- wrong type confusions
print("\nWrong type, same span: top confusions (gold -> pred):")
wt = G[G["cat"] == "wrong_type_same_span"]
wc = Counter(zip(wt["type"], wt["oth"].map(lambda x: x[0])))
und = Counter()
for (a, b_), v in wc.items():
    und[tuple(sorted((a, b_)))] += v
print("directed:", wc.most_common(20))
print("unordered:", und.most_common(15))
save(pd.DataFrame([{"gold": a, "pred": b_, "n": v} for (a, b_), v in wc.most_common()]), "q3_wrong_type_confusions.csv")

print("\nSpurious, top (type, value):", Counter(zip(P[P["cat"] == "spurious"]["type"], P[P["cat"] == "spurious"]["value"])).most_common(25))
print("Spurious by type:", P[P["cat"] == "spurious"]["type"].value_counts().head(15).to_dict())
print("\nMissed, top (type, value):", Counter(zip(G[G["cat"] == "missed"]["type"], G[G["cat"] == "missed"]["value"])).most_common(25))
print("Missed by type:", G[G["cat"] == "missed"]["type"].value_counts().head(15).to_dict())

# ---- per type
rows = []
for t in sorted(set(G["type"]) | set(P["type"])):
    g_, p_ = G[G["type"] == t], P[P["type"] == t]
    tp = int((g_["cat"] == "exact").sum())
    fp = len(p_) - int((p_["cat"] == "exact").sum())
    fn = len(g_) - tp
    rows.append({"type": t, "train_spans": tr_type_n.get(t, 0), "support": len(g_), "pred": len(p_), "tp": tp, "fp": fp, "fn": fn,
                 "precision": round(100 * tp / max(len(p_), 1), 1), "recall": round(100 * tp / max(len(g_), 1), 1),
                 "f1": round(f1_from(tp, fp, fn), 1), "oov_span_pct": round(100 * g_["has_oov"].mean(), 1) if len(g_) else None,
                 **{c: int((g_["cat"] == c).sum()) for c in G_CATS[1:]}, "spurious": int((p_["cat"] == "spurious").sum())})
PT = pd.DataFrame(rows).sort_values("support", ascending=False)
save(PT, "q3_per_type.csv")
print("\nTop 15 types by dev support:")
show(PT.head(15))
print("share of gold spans in top 15:", round(100 * PT.head(15)["support"].sum() / len(G), 1))
print("\nWorst 10 types with support >= 10:")
worst = PT[PT["support"] >= 10].sort_values(["f1", "support"]).head(10)
show(worst)
save(worst, "q3_worst_types.csv")
print("types with support < 10:", int((PT["support"] < 10).sum()), "| their gold spans:", int(PT[PT["support"] < 10]["support"].sum()))
fp_only = PT[PT["support"] == 0]
print("types predicted but no gold on dev:", fp_only[["type", "pred"]].to_dict("records"))

# ---- illegal BIO in predictions
print("\nIllegal BIO in Stage-1 predictions:")
stray_o, stray_type = 0, 0
rep_drop, rep_merge = [], []
span_from_stray = []
for r in m.itertuples(index=False):
    tags = list(r.pred_tags)
    prev = "O"
    fixed_drop, fixed_merge = [], []
    cur_type = None  # type of the open span in the merge repair
    drop_mode = False
    for t in tags:
        if t.startswith("I-"):
            typ = t[2:]
            if prev == "O":
                stray_o += 1
            elif prev[2:] != typ:
                stray_type += 1
        prev = t
    # repair 1: a stray I opens nothing, its run is dropped until the next B or O
    open_type = None
    for t in tags:
        if t == "O":
            fixed_drop.append("O"); open_type = None
        elif t.startswith("B-"):
            fixed_drop.append(t); open_type = t[2:]
        else:
            if open_type == t[2:]:
                fixed_drop.append(t)
            else:
                fixed_drop.append("O"); open_type = None
    rep_drop.append(fixed_drop)
    # repair 2: an I after any open span continues that span with the opener's type
    open_type = None
    for t in tags:
        if t == "O":
            fixed_merge.append("O"); open_type = None
        elif t.startswith("B-"):
            fixed_merge.append(t); open_type = t[2:]
        else:
            if open_type is None:
                fixed_merge.append("B-" + t[2:]); open_type = t[2:]
            else:
                fixed_merge.append("I-" + open_type)
    rep_merge.append(fixed_merge)
n_i = sum(t.startswith("I-") for p in m["pred_tags"] for t in p)
print(f"I- tags predicted: {n_i} | after O: {stray_o} | after a different type: {stray_type}")
for name, rep in (("drop stray I runs", rep_drop), ("merge I into open span", rep_merge)):
    tp, fp, fn = slot_counts(m["gold_tags"], rep)
    print(f"  repair '{name}': TP {tp} FP {fp} FN {fn} F1 {f1_from(tp, fp, fn):.2f} (gain {f1_from(tp, fp, fn) - base:.2f})")
mem("end")

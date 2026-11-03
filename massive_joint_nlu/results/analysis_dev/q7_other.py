"""Q7: casing, punctuation, annotation consistency, length, rare intents, selection noise (train + dev only)."""
from common import *

train, m, intents = load_all()
_, a1, _ = load_all(A1_FRAME)
m = m.merge(pd.read_pickle(OUT / "q4_linear_dev_pred.pkl"), on="id")
G = pd.read_pickle(OUT / "q3_gold_spans.pkl")
P = pd.read_pickle(OUT / "q3_pred_spans.pkl")
ALL = {"train": train, "dev": m}

# ---- casing and characters
print("=== casing and characters ===")
for name, df in ALL.items():
    up = df["utt"].map(lambda s: any(c.isupper() for c in s))
    nasc = df["utt"].map(lambda s: any(ord(c) > 127 for c in s))
    digit = df["utt"].map(lambda s: any(c.isdigit() for c in s))
    print(f"{name}: utterances with uppercase {int(up.sum())} | non-ASCII {int(nasc.sum())} | with digits {int(digit.sum())} = {100 * digit.mean():.1f}% | annot differs from utt in tokens: 0 by assertion")
nas = Counter(c for s in pd.concat([train["utt"], m["utt"]]) for c in s if ord(c) > 127)
print("non-ASCII chars:", nas.most_common(20))

# ---- punctuation tokens
print("\n=== punctuation tokens ===")
for name, df in ALL.items():
    pc = Counter(t for toks in df["toks"] for t in toks if not is_lexical(t))
    tot = sum(len(t) for t in df["toks"])
    utts = df["toks"].map(lambda ts: any(not is_lexical(t) for t in ts))
    insl = sum(1 for toks, sm in zip(df["toks"], df["smask"]) for t, s in zip(toks, sm) if not is_lexical(t) and s)
    print(f"{name}: non-lexical tokens {sum(pc.values())} = {100 * sum(pc.values()) / tot:.2f}% of tokens | utterances with one {int(utts.sum())} = {100 * utts.mean():.1f}% | inside gold slots {insl} | top {pc.most_common(12)}")
    apos = sum(1 for toks in df["toks"] for t in toks if "'" in t or "’" in t)
    hyph = sum(1 for toks in df["toks"] for t in toks if "-" in t)
    curly = sum(1 for toks in df["toks"] for t in toks if "’" in t)
    print(f"   tokens with apostrophe {apos} (curly {curly}) | with hyphen {hyph}")
ends = m["toks"].map(lambda t: t[-1] if t else "")
print("dev utterances ending in punctuation:", Counter(e for e in ends if not is_lexical(e)).most_common(6))
print("dev tokens with apostrophe, top:", Counter(t.casefold() for toks in m["toks"] for t in toks if "'" in t).most_common(15))

# dotted abbreviations
G["has_dot"] = G["value"].str.contains(r"\.", regex=True)
for flag, name in ((True, "gold span contains '.'"), (False, "gold span has no '.'")):
    s = G[G["has_dot"] == flag]
    print(f"{name}: n {len(s)} | exact hit {100 * (s['cat'] == 'exact').mean():.1f}% | boundary {100 * (s['cat'] == 'boundary_same_type').mean():.1f}% | missed {100 * (s['cat'] == 'missed').mean():.1f}%")
print("dotted spans by type:", G[G["has_dot"]]["type"].value_counts().head(8).to_dict())
for dot_form, plain_form in ((r"\ba \. m\b", r"\bam\b"), (r"\bp \. m\b", r"\bpm\b")):
    a = sum(bool(re.search(dot_form, k)) for k in train["key"])
    b = sum(bool(re.search(plain_form, k)) for k in train["key"])
    a2 = sum(bool(re.search(dot_form, k)) for k in m["key"])
    b2 = sum(bool(re.search(plain_form, k)) for k in m["key"])
    print(f"forms {dot_form} vs {plain_form}: train {a} vs {b} | dev {a2} vs {b2}")

# wake word
for w in ("olly",):
    for name, df in ALL.items():
        has = df["toks"].map(lambda ts: any(t.casefold() == w for t in ts))
        print(f"wake word '{w}' in {name}: {int(has.sum())} = {100 * has.mean():.1f}%")
has = m["toks"].map(lambda ts: any(t.casefold() == "olly" for t in ts))
tp, fp, fn = slot_counts(m[has]["gold_tags"], m[has]["pred_tags"])
tp2, fp2, fn2 = slot_counts(m[~has]["gold_tags"], m[~has]["pred_tags"])
print(f"dev with olly: S1 intent {100 * m[has]['int_ok'].mean():.2f} F1 {f1_from(tp, fp, fn):.2f} | without: intent {100 * m[~has]['int_ok'].mean():.2f} F1 {f1_from(tp2, fp2, fn2):.2f}")
tagged = sum(1 for toks, sm in zip(pd.concat([train["toks"], m["toks"]]), pd.concat([train["smask"], m["smask"]])) for t, s in zip(toks, sm) if t.casefold() == "olly" and s)
print("olly tokens inside a gold slot (train+dev):", tagged)

# ---- annotation sanity
print("\n=== annotation shape ===")
for name, df in ALL.items():
    empty = df["annot_utt"].str.contains(r"\[[^\[\]:]+:\s*\]", regex=True).sum()
    adj_same = 0
    n_spans = 0
    for b in df["bio"]:
        sp = bio_to_spans(b)
        n_spans += len(sp)
        adj_same += sum(1 for x, y in zip(sp, sp[1:]) if x[0] == y[0] and x[2] == y[1])
    noslot = df["bio"].map(lambda b: all(t == "O" for t in b))
    print(f"{name}: spans {n_spans} ({n_spans / len(df):.2f}/utt) | empty-value slots {int(empty)} | adjacent same-type spans {adj_same} | utterances with no slot {int(noslot.sum())} = {100 * noslot.mean():.1f}%")
for name, df in ALL.items():
    sl = Counter(s[2] - s[1] for b in df["bio"] for s in bio_to_spans(b))
    tot = sum(sl.values())
    print(f"{name} span lengths:", {k: round(100 * v / tot, 1) for k, v in sorted(sl.items())[:8]})

# ---- value consistency, scanning every train utterance for each train slot value
print("\n=== same value inside and outside slots ===")
MAXL = 6
in_type = {}
for r in train.itertuples(index=False):
    cf = [t.casefold() for t in r.toks]
    for t, s, e in bio_to_spans(r.bio):
        if e - s <= MAXL:
            in_type.setdefault(tuple(cf[s:e]), Counter())[t] += 1


def scan(df, values):
    out = {}
    for r in df.itertuples(index=False):
        cf = [t.casefold() for t in r.toks]
        sp = bio_to_spans(r.bio)
        ex = {(s, e): t for t, s, e in sp}
        tagged_tok = [x != "O" for x in r.bio]
        n = len(cf)
        for s in range(n):
            for L in range(1, MAXL + 1):
                if s + L > n:
                    break
                v = tuple(cf[s:s + L])
                if v not in values:
                    continue
                d = out.setdefault(v, {"exact": Counter(), "outside": 0, "partial": 0})
                if (s, s + L) in ex:
                    d["exact"][ex[(s, s + L)]] += 1
                elif not any(tagged_tok[s:s + L]):
                    d["outside"] += 1
                else:
                    d["partial"] += 1
    return out


sc_tr = scan(train, in_type)
sc_dv = scan(m, in_type)
rows = []
for v, d in sc_tr.items():
    ex = sum(d["exact"].values())
    if ex == 0:
        continue
    top_t, top_n = d["exact"].most_common(1)[0]
    dv = sc_dv.get(v, {"exact": Counter(), "outside": 0, "partial": 0})
    rows.append({"value": " ".join(v), "len": len(v), "top_type": top_t, "tr_exact": ex, "tr_top_type": top_n, "tr_other_type": ex - top_n,
                 "tr_outside": d["outside"], "tr_partial": d["partial"], "dv_exact": sum(dv["exact"].values()),
                 "dv_top_type": dv["exact"].get(top_t, 0), "dv_outside": dv["outside"], "dv_partial": dv["partial"],
                 "types": dict(d["exact"].most_common(4))})
V = pd.DataFrame(rows)
V["tr_n"] = V["tr_exact"] + V["tr_outside"] + V["tr_partial"]
V["tagged_share"] = (V["tr_exact"] / V["tr_n"]).round(3)
save(V.sort_values("tr_n", ascending=False), "q7_value_consistency.csv")
print("train slot values (<=6 tokens):", len(V), "| seen as a slot at least twice:", int((V["tr_exact"] >= 2).sum()),
      "| with more than one type:", int((V["tr_other_type"] > 0).sum()),
      "| both slot and plain-text occurrences:", int((V["tr_outside"] > 0).sum()))
b = V[V["tr_n"] >= 5]
mix = b[(b["tr_exact"] > 0) & (b["tr_outside"] > 0)]
print(f"values with >=5 train occurrences: {len(b)} | also outside a slot: {len(mix)} | with tagged share between 20% and 80%: {int(((b['tagged_share'] >= .2) & (b['tagged_share'] <= .8)).sum())}")
print("\nchecked closed-class time and date values:")
probe = ["five am", "nine am", "seven am", "eight am", "six am", "ten am", "today", "tomorrow", "tonight", "yesterday", "this week", "next week", "this morning", "morning", "evening", "night", "afternoon", "monday", "friday", "weekend", "next", "now", "noon", "midnight", "daily", "every day", "week", "month", "hour", "minute"]
show(V[V["value"].isin(probe)].sort_values("tr_n", ascending=False)[["value", "top_type", "tr_exact", "tr_other_type", "tr_outside", "tr_partial", "tagged_share", "dv_exact", "dv_top_type", "dv_outside", "dv_partial", "types"]])
print("\nlargest mixed values (minority count >= 5), date/time/place style first by count:")
V["minority"] = V[["tr_exact", "tr_outside"]].min(axis=1)
show(V[V["tr_exact"] >= 3].sort_values("minority", ascending=False).head(30)[["value", "top_type", "tr_exact", "tr_other_type", "tr_outside", "tr_partial", "tagged_share", "dv_exact", "dv_outside", "types"]])
print("\nvalues with most type disagreement in train:")
show(V.sort_values("tr_other_type", ascending=False).head(20)[["value", "top_type", "tr_exact", "tr_other_type", "types", "dv_exact"]])

# ---- dev gold spans by the history of their value in train
print("\n=== dev gold spans by value history in train ===")
g_vals = {tuple(v.split(" ")) for v in G["value"]}
hist = scan(train, g_vals)   # exact/outside/partial occurrence of the exact string in train
label = []
for r in G.itertuples(index=False):
    v = tuple(r.value.split(" "))
    h = hist.get(v)
    if r.has_oov:
        label.append("value has Train-OOV token")
    elif h is None and len(v) > MAXL:
        label.append("longer than 6 tokens")
    elif h is None:
        label.append("string never seen in train, words known")
    elif h["exact"].get(r.type, 0) > 0:
        label.append("seen as a slot of this type")
    elif sum(h["exact"].values()) > 0:
        label.append("seen as a slot of another type only")
    else:
        label.append("seen only as plain text or inside a longer slot")
G["hist"] = label
rows = []
for k, s in G.groupby("hist"):
    rows.append({"value history": k, "n": len(s), "share_pct": round(100 * len(s) / len(G), 1), "exact_hit_pct": round(100 * (s["cat"] == "exact").mean(), 1),
                 "wrong_type_pct": round(100 * (s["cat"] == "wrong_type_same_span").mean(), 1), "boundary_pct": round(100 * (s["cat"] == "boundary_same_type").mean(), 1),
                 "missed_pct": round(100 * (s["cat"] == "missed").mean(), 1)})
HV = pd.DataFrame(rows).sort_values("n", ascending=False)
show(HV)
save(HV, "q7_gold_spans_by_value_history.csv")

# ---- Stage-1 errors that look like annotation inconsistency
print("\n=== Stage-1 false positives vs train evidence ===")
sp = P[P["cat"] == "spurious"]
in_tr = [in_type.get(tuple(v.split(" ")), Counter()).get(t, 0) for v, t in zip(sp["value"], sp["type"])]
print(f"spurious spans {len(sp)} | exact (value,type) is a gold slot in train: {sum(x > 0 for x in in_tr)} = {100 * np.mean([x > 0 for x in in_tr]):.1f}% | at least 3 times: {sum(x >= 3 for x in in_tr)}")
sp_t = sp.assign(tr=in_tr)
show(sp_t[sp_t["tr"] >= 3].groupby(["type", "value"]).agg(n=("id", "size"), train_as_slot=("tr", "first")).reset_index().sort_values("n", ascending=False).head(25))
ms = G[G["cat"] == "missed"]
print("missed spans:", len(ms), "| value history:", ms["hist"].value_counts().to_dict())

# ---- boundary convention
print("\n=== boundary convention ===")
FW = {"the", "a", "an", "my", "at", "on", "in", "of", "for", "to", "this", "that", "next", "last", "by", "from", "and", "around", "about", "me", "i", "s", "'s", "."}
b = G[G["cat"] == "boundary_same_type"].copy()
kinds = []
for r in b.itertuples(index=False):
    g = (int(r.start), int(r.end))
    p = (int(r.oth_start), int(r.oth_end))
    gt = r.value.split(" ")
    pt = r.oth[1].split(" ")
    if p[0] >= g[0] and p[1] <= g[1]:
        lead = gt[: p[0] - g[0]]
        trail = gt[len(gt) - (g[1] - p[1]):] if g[1] > p[1] else []
        diff = lead + trail
        kinds.append(("pred too short", all(x in FW for x in diff), " ".join(diff)))
    elif p[0] <= g[0] and p[1] >= g[1]:
        lead = pt[: g[0] - p[0]]
        trail = pt[len(pt) - (p[1] - g[1]):] if p[1] > g[1] else []
        diff = lead + trail
        kinds.append(("pred too long", all(x in FW for x in diff), " ".join(diff)))
    else:
        kinds.append(("shifted", False, ""))
b["k"], b["fw"], b["diff"] = zip(*kinds)
print(b.groupby(["k", "fw"]).size().to_string())
print("difference tokens when only function words or dots:", Counter(b[b["fw"]]["diff"]).most_common(25))
print("difference tokens otherwise:", Counter(b[~b["fw"]]["diff"]).most_common(25))
fw_fix = int(b["fw"].sum())
TP, FP, FN = slot_counts(m["gold_tags"], m["pred_tags"])
print(f"if only function-word/dot boundary errors were repaired: +{fw_fix} TP -> F1 {f1_from(TP + fw_fix, FP - fw_fix, FN - fw_fix):.2f}")

# same core value annotated with and without a leading function word, train
core_in, core_out = Counter(), Counter()
for r in train.itertuples(index=False):
    cf = [t.casefold() for t in r.toks]
    for t, s, e in bio_to_spans(r.bio):
        if cf[s] in FW and e - s >= 2:
            core_in[(t, cf[s], tuple(cf[s + 1:e]))] += 1
        if s > 0 and cf[s - 1] in FW and r.bio[s - 1] == "O":
            core_out[(t, cf[s - 1], tuple(cf[s:e]))] += 1
both = [(k, core_in[k], core_out[k]) for k in core_in if core_out.get(k, 0) > 0]
both.sort(key=lambda x: -min(x[1], x[2]))
tot_in = sum(core_in.values())
print(f"train spans starting with a function word: {tot_in} | (type, word, core) patterns annotated both ways: {len(both)} | spans inside those patterns: {sum(x[1] + x[2] for x in both)}")
print([(k[0], k[1] + " + " + " ".join(k[2]), i, o) for k, i, o in both[:20]])
by_word = {}
for (t, w, core), c in core_in.items():
    by_word.setdefault((t, w), [0, 0])[0] += c
for (t, w, core), c in core_out.items():
    by_word.setdefault((t, w), [0, 0])[1] += c
rows = [{"type": t, "word": w, "inside_span": a, "outside_span": b_, "inside_share_pct": round(100 * a / (a + b_), 1)} for (t, w), (a, b_) in by_word.items() if a >= 5 and b_ >= 5]
BW = pd.DataFrame(rows).sort_values("inside_span", ascending=False)
show(BW.head(25))
save(BW, "q7_boundary_convention.csv")

# ---- length
print("\n=== length ===")
for name, df in ALL.items():
    q = df["n_tok"].quantile([.05, .25, .5, .75, .95, .99, 1]).to_dict()
    print(name, "tokens:", {k: int(v) for k, v in q.items()}, "| mean", round(df["n_tok"].mean(), 2), "| >=20:", int((df["n_tok"] >= 20).sum()), "| >=30:", int((df["n_tok"] >= 30).sum()))
bins = [0, 1, 2, 4, 8, 12, 20, 100]
labels = ["1", "2", "3-4", "5-8", "9-12", "13-20", "21+"]
m["lb"] = pd.cut(m["n_tok"], bins, labels=labels)
a1m = a1.set_index("id").loc[m["id"]]
m["a1_ok"] = a1m["int_ok"].to_numpy()
rows = []
for l in labels:
    s = m[m["lb"] == l]
    tp, fp, fn = slot_counts(s["gold_tags"], s["pred_tags"])
    rows.append({"tokens": l, "n": len(s), "share_pct": round(100 * len(s) / len(m), 1), "train_share_pct": round(100 * pd.cut(train["n_tok"], bins, labels=labels).eq(l).mean(), 1),
                 "S1_intent": round(100 * s["int_ok"].mean(), 1), "A1_intent": round(100 * s["a1_ok"].mean(), 1), "linear_intent": round(100 * s["lin_ok"].mean(), 1),
                 "S1_slot_f1": round(f1_from(tp, fp, fn), 1), "gold_spans": tp + fn, "errors_share_pct": round(100 * (~s["int_ok"]).sum() / (~m["int_ok"]).sum(), 1)})
LT = pd.DataFrame(rows)
show(LT)
save(LT, "q7_by_length.csv")

# ---- rare intents and types
print("\n=== rare intents and types ===")
tr_n = train["intent"].value_counts()
dv_n = m["intent"].value_counts()
rows = []
for i in tr_n.sort_values().index[:14]:
    s = m[m["gold_intent"] == i]
    rows.append({"intent": i, "train": int(tr_n[i]), "dev": len(s), "S1_recall": round(100 * s["int_ok"].mean(), 1) if len(s) else None,
                 "linear_recall": round(100 * s["lin_ok"].mean(), 1) if len(s) else None})
RI = pd.DataFrame(rows)
show(RI)
save(RI, "q7_rare_intents.csv")
ps = tr_n / tr_n.sum()
pd_ = (dv_n / dv_n.sum()).reindex(ps.index).fillna(0)
print("total variation distance train vs dev intent mix:", round(0.5 * (ps - pd_).abs().sum(), 3))
print("intents with train<100:", int((tr_n < 100).sum()), "covering dev:", int(m["gold_intent"].isin(tr_n[tr_n < 100].index).sum()),
      "| their S1 acc:", round(100 * m[m["gold_intent"].isin(tr_n[tr_n < 100].index)]["int_ok"].mean(), 2))
r_by = m.groupby("gold_intent")["int_ok"].mean()
print("macro recall S1:", round(100 * r_by.mean(), 2), "| linear:", round(100 * m.groupby("gold_intent")["lin_ok"].mean().mean(), 2))
print("dev support <5 intents:", dv_n[dv_n < 5].to_dict())
tt = Counter(s[0] for b_ in train["bio"] for s in bio_to_spans(b_))
td = Counter(s[0] for b_ in m["bio"] for s in bio_to_spans(b_))
rare_t = sorted(tt, key=lambda k: tt[k])[:14]
print("rarest slot types (train, dev):", [(k, tt[k], td.get(k, 0)) for k in rare_t])
print("types absent from dev:", sorted(set(tt) - set(td)))
print("top 5 types' share of dev gold spans:", round(100 * sum(v for _, v in td.most_common(5)) / sum(td.values()), 1), td.most_common(5))
tr_lab = {t for b_ in train["bio"] for t in b_}
print("I- labels in train but never B-: ", sorted(t for t in tr_lab if t.startswith("I-") and "B-" + t[2:] not in tr_lab), "| single-token-only types (B- without I-):", sorted(t[2:] for t in tr_lab if t.startswith("B-") and "I-" + t[2:] not in tr_lab))

# ---- selection noise
print("\n=== dev sampling noise ===")
per = np.array([slot_counts([g], [p]) for g, p in zip(m["gold_tags"], m["pred_tags"])])
per_a1 = np.array([slot_counts([g], [p]) for g, p in zip(a1m["gold_tags"], a1m["pred_tags"])])
ia = m["int_ok"].to_numpy().astype(float)
ib = a1m["int_ok"].to_numpy().astype(float)
rng = np.random.default_rng(42)
res = []
n = len(m)


def f1v(c):
    tp, fp, fn = c
    return 100 * 2 * tp / max(2 * tp + fp + fn, 1)


for _ in range(2000):
    idx = rng.integers(0, n, n)
    a_i, b_i = 100 * ia[idx].mean(), 100 * ib[idx].mean()
    a_f, b_f = f1v(per[idx].sum(0)), f1v(per_a1[idx].sum(0))
    res.append((a_i, a_f, (a_i + a_f) / 2, b_i, b_f, (b_i + b_f) / 2, b_i - a_i, b_f - a_f, (b_i + b_f - a_i - a_f) / 2))
res = np.array(res)
names = ["S1 intent", "S1 slot", "S1 joint", "A1 intent", "A1 slot", "A1 joint", "A1-S1 intent", "A1-S1 slot", "A1-S1 joint"]
rows = [{"quantity": k, "mean": round(res[:, i].mean(), 2), "se": round(res[:, i].std(), 2), "ci_low": round(np.percentile(res[:, i], 2.5), 2), "ci_high": round(np.percentile(res[:, i], 97.5), 2)} for i, k in enumerate(names)]
NS = pd.DataFrame(rows)
show(NS)
save(NS, "q7_bootstrap_noise.csv")
mem("end")

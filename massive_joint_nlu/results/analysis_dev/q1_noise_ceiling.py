"""Q1: label noise and intent ceiling from exact twins and slot-abstracted templates (train + dev only)."""
from itertools import combinations

from common import *

train, m, intents = load_all()
rng = np.random.default_rng(42)


def maj(counter):
    return sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))[0]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(100 * (c - h) / d, 1), round(100 * (c + h) / d, 1))


def stats_for_groups(group_counters):
    """group_counters: list of Counter(label->count), groups of size >= 2.
    pairwise disagreement R_NN (instance weighted), resubstitution error (optimistic), leave-one-out error (pessimistic),
    and the Cover-Hart lower bound on the Bayes error implied by R_NN with K=60."""
    tot = pair_agree = resub_ok = loo_ok = 0.0
    per_group = []
    for c in group_counters:
        n = sum(c.values())
        pa = sum(v * (v - 1) for v in c.values()) / (n * (n - 1))
        ok_res = max(c.values())
        ok_loo = 0.0
        for lab, v in c.items():
            oth = dict(c)
            oth[lab] = v - 1
            top = max(oth.values())
            tied = [k for k, x in oth.items() if x == top]
            ok_loo += v * ((lab in tied) / len(tied))
        tot += n
        pair_agree += n * pa
        resub_ok += ok_res
        loo_ok += ok_loo
        per_group.append((n, pa, ok_res, ok_loo))
    arr = np.array(per_group)

    def rnn_of(a):
        return 1 - (a[:, 0] * a[:, 1]).sum() / a[:, 0].sum()

    r = rnn_of(arr)
    k = 60
    low = (k - 1) / k * (1 - np.sqrt(max(0.0, 1 - k * r / (k - 1))))
    boot = [rnn_of(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(2000)]
    return {"groups": len(arr), "instances": int(tot),
            "R_NN_pct": 100 * r, "R_NN_ci": (100 * np.percentile(boot, 2.5), 100 * np.percentile(boot, 97.5)),
            "bayes_acc_low_from_RNN": 100 * (1 - r), "bayes_acc_high_from_RNN": 100 * (1 - low),
            "resub_acc_pct(optimistic)": 100 * resub_ok / tot, "loo_acc_pct(pessimistic)": 100 * loo_ok / tot}


# ------------------------------------------------------------------ exact twins
g_int, g_bio, g_both = {}, {}, {}
for r in train.itertuples(index=False):
    g_int.setdefault(r.key, Counter())[r.intent] += 1
    g_bio.setdefault(r.key, Counter())[tuple(r.bio)] += 1
    g_both.setdefault(r.key, Counter())[(r.intent, tuple(r.bio))] += 1

m["twin"] = m["key"].isin(g_int)
tw = m[m["twin"]].copy()
print("=== dev twins (casefold-identical utterance in train) ===")
print("dev twins:", len(tw), f"({100 * len(tw) / len(m):.2f}% of dev), unique dev keys:", tw["key"].nunique(),
      "| string-level casefold check:",
      int(m["utt"].str.casefold().str.strip().isin(set(train["utt"].str.casefold().str.strip())).sum()))
rows = []
for r in tw.itertuples(index=False):
    ci, cb, cj = g_int[r.key], g_bio[r.key], g_both[r.key]
    b = tuple(r.bio)
    rows.append(dict(
        id=r.id, n_train=sum(ci.values()),
        int_any=r.intent in ci, int_all=set(ci) == {r.intent}, int_maj=maj(ci)[0] == r.intent,
        int_tie=sorted(ci.values())[-1] == (sorted(ci.values())[-2] if len(ci) > 1 else -1),
        bio_any=b in cb, bio_all=set(cb) == {b}, bio_maj=maj(cb)[0] == b,
        both_any=(r.intent, b) in cj, both_all=set(cj) == {(r.intent, b)}, both_maj=maj(ci)[0] == r.intent and maj(cb)[0] == b,
        has_slot=any(t != "O" for t in r.bio) or any(t != "O" for k in cb for t in k)))
T = pd.DataFrame(rows)
tw = tw.merge(T, on="id")
n = len(tw)
for c, name in [("int_all", "intent: all train twins agree"), ("int_maj", "intent: equals train majority"),
                ("int_any", "intent: some train twin agrees"),
                ("bio_all", "BIO: all train twins agree"), ("bio_maj", "BIO: equals train majority"),
                ("bio_any", "BIO: some train twin agrees"),
                ("both_all", "intent+BIO: all agree"), ("both_maj", "intent+BIO: both majority"),
                ("both_any", "intent+BIO: some twin agrees on both")]:
    k = int(tw[c].sum())
    print(f"  {name:40s} {k:4d}/{n} = {100 * k / n:5.1f}%  CI95 {wilson(k, n)}")
print("  ties in majority intent:", int(tw["int_tie"].sum()), "| twins with train group size >1:", int((tw["n_train"] > 1).sum()))
ws = tw[tw["has_slot"]]
n2 = len(ws)
for c, name in [("bio_all", "BIO all agree"), ("bio_maj", "BIO majority agree"), ("bio_any", "BIO any")]:
    print(f"  [twins with a slot in dev or a twin, n={n2}] {name}: {int(ws[c].sum())} = {100 * ws[c].mean():.1f}%")
pred_maj = [list(maj(g_bio[k])[0]) for k in tw["key"]]
tp, fp, fn = slot_counts(tw["bio"], pred_maj)
print(f"  span micro-F1 dev gold vs train-majority annotation on twins: {f1_from(tp, fp, fn):.2f} (TP {tp} FP {fp} FN {fn})")

# stage 1 on twin vs non twin
rows = []
for name, s in [("twin", m[m["twin"]]), ("no twin", m[~m["twin"]])]:
    tp, fp, fn = slot_counts(s["gold_tags"], s["pred_tags"])
    rows.append({"subset": name, "n": len(s), "mean_tokens": round(s["n_tok"].mean(), 2),
                 "S1_intent": round(100 * s["int_ok"].mean(), 2), "S1_slot_f1": round(f1_from(tp, fp, fn), 2),
                 "gold_slots_per_utt": round(s["bio"].map(lambda b: sum(t.startswith("B-") for t in b)).mean(), 2)})
print(pd.DataFrame(rows).to_string(index=False))
twin_table = pd.DataFrame(rows)
save(twin_table, "q1_twin_vs_rest.csv")

# twin where label disagrees: does stage 1 follow the train label or the dev label
dis = tw[~tw["int_maj"]]
print("dev twins disagreeing with train majority:", len(dis),
      "| Stage 1 predicts dev gold:", int((dis["pred_intent"] == dis["gold_intent"]).sum()),
      "| predicts train majority:", int(sum(maj(g_int[k])[0] == p for k, p in zip(dis["key"], dis["pred_intent"]))))
ex = dis.assign(train_counts=[dict(g_int[k]) for k in dis["key"]])[["utt", "intent", "pred_intent", "train_counts"]]
save(ex, "q1_dev_twin_intent_disagreements.csv")
show(ex.head(25))
agree = tw[tw["int_maj"]]
print("Stage-1 intent accuracy on twins agreeing with train majority:", round(100 * agree["int_ok"].mean(), 2),
      "n", len(agree))

# ------------------------------------------------------------------ train-internal duplicates
print("\n=== train-internal duplicates ===")
dup_keys = [k for k, c in g_int.items() if sum(c.values()) >= 2]
n_rows = sum(sum(g_int[k].values()) for k in dup_keys)
conf_int = [k for k in dup_keys if len(g_int[k]) > 1]
conf_bio = [k for k in dup_keys if len(g_bio[k]) > 1]
conf_either = [k for k in dup_keys if len(g_int[k]) > 1 or len(g_bio[k]) > 1]
print("unique train keys:", len(g_int), "| duplicate groups (n>=2):", len(dup_keys), "| rows in them:", n_rows)
print("groups with conflicting intents:", len(conf_int), f"({100 * len(conf_int) / len(dup_keys):.1f}%)",
      "| rows:", sum(sum(g_int[k].values()) for k in conf_int))
print("groups with conflicting annotation:", len(conf_bio), f"({100 * len(conf_bio) / len(dup_keys):.1f}%)",
      "| rows:", sum(sum(g_int[k].values()) for k in conf_bio))
print("groups conflicting on either:", len(conf_either), f"({100 * len(conf_either) / len(dup_keys):.1f}%)")
print("rows in minority intent within duplicate groups:", sum(sum(g_int[k].values()) - maj(g_int[k])[1] for k in dup_keys))
print("rows in minority annotation within duplicate groups:", sum(sum(g_bio[k].values()) - maj(g_bio[k])[1] for k in dup_keys))
print("duplicate groups by size:", dict(sorted(Counter(min(sum(g_int[k].values()), 6) for k in dup_keys).items())))
top = sorted(conf_int, key=lambda k: -sum(g_int[k].values()))[:20]
tbl = pd.DataFrame([{"utt": k, "n": sum(g_int[k].values()), "intents": dict(g_int[k].most_common())} for k in top])
show(tbl)
save(pd.DataFrame([{"utt": k, "n": sum(g_int[k].values()), "intents": dict(g_int[k].most_common())} for k in conf_int]),
     "q1_train_duplicate_intent_conflicts.csv")
save(pd.DataFrame([{"utt": k, "n": sum(g_bio[k].values()),
                    "annotations": [(" ".join(a), v) for a, v in g_bio[k].most_common()]} for k in conf_bio]),
     "q1_train_duplicate_annotation_conflicts.csv")
pair_c = Counter()
for k in conf_int:
    for a, b in combinations(sorted(g_int[k]), 2):
        pair_c[(a, b)] += 1
print("top intent pairs in train duplicate conflicts:", pair_c.most_common(12))

print("\n--- Bayes bracket on exact duplicates ---")
res_dup = {}
res_dup["train duplicates"] = stats_for_groups([g_int[k] for k in dup_keys])
pool = {}
for k, c in g_int.items():
    pool[k] = Counter(c)
for r in m.itertuples(index=False):
    pool.setdefault(r.key, Counter())[r.intent] += 1
res_dup["train+dev duplicates"] = stats_for_groups([c for c in pool.values() if sum(c.values()) >= 2])
# dev twins only: groups that actually contain a dev row, to reflect the dev-side mix
res_dup["groups containing a dev row"] = stats_for_groups([pool[k] for k in tw["key"].unique()])
for name, v in res_dup.items():
    print(name, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()})

# dev-internal duplicates
gd = m.groupby("key")["intent"].agg(list)
dd = gd[gd.map(len) >= 2]
print("dev-internal duplicate groups:", len(dd), "rows", int(dd.map(len).sum()),
      "| with conflicting intents:", int(dd.map(lambda x: len(set(x)) > 1).sum()))

# ------------------------------------------------------------------ templates
print("\n=== templates (slot values replaced by type, casefolded) ===")
t_int = {}
for r in train.itertuples(index=False):
    t_int.setdefault(r.tmpl, Counter())[r.intent] += 1
has_slot = {k: ("<" in k and ">" in k) for k in t_int}
sz = np.array([sum(c.values()) for c in t_int.values()])
mult = [k for k, c in t_int.items() if len(c) > 1]
print("train rows:", len(train), "| unique templates:", len(t_int), "| singletons:", int((sz == 1).sum()),
      "| templates n>=2:", int((sz >= 2).sum()))
print("templates with >1 intent:", len(mult), f"= {100 * len(mult) / (sz >= 2).sum():.1f}% of templates with n>=2",
      "| train rows inside them:", sum(sum(t_int[k].values()) for k in mult),
      f"= {100 * sum(sum(t_int[k].values()) for k in mult) / len(train):.1f}% of train")
print("rows in minority intent of a template:", sum(sum(c.values()) - max(c.values()) for c in t_int.values()),
      "| train purity of template majority:", round(100 * sum(max(c.values()) for c in t_int.values()) / len(train), 2))
for flag, name in ((True, "with slot"), (False, "no slot")):
    ks = [k for k in t_int if has_slot[k] == flag]
    ge2 = [k for k in ks if sum(t_int[k].values()) >= 2]
    print(f"  {name}: templates {len(ks)}, rows {sum(sum(t_int[k].values()) for k in ks)}, n>=2 {len(ge2)}, "
          f"multi-intent {sum(len(t_int[k]) > 1 for k in ge2)}, "
          f"rows in multi-intent {sum(sum(t_int[k].values()) for k in ge2 if len(t_int[k]) > 1)}")
top = sorted(mult, key=lambda k: -sum(t_int[k].values()))[:20]
show(pd.DataFrame([{"template": k, "n": sum(t_int[k].values()), "intents": dict(t_int[k].most_common())} for k in top]))
save(pd.DataFrame([{"template": k, "n": sum(t_int[k].values()), "intents": dict(t_int[k].most_common())}
                   for k in sorted(mult, key=lambda k: -sum(t_int[k].values()))]), "q1_train_template_conflicts.csv")
pair_t = Counter()
for k in mult:
    for a, b in combinations(sorted(t_int[k]), 2):
        pair_t[(a, b)] += 1
print("top intent pairs sharing a template in train:", pair_t.most_common(12))
res_t = {"train templates": stats_for_groups([c for c in t_int.values() if sum(c.values()) >= 2])}
print({a: (round(b, 2) if isinstance(b, float) else b) for a, b in res_t["train templates"].items()})

m["t_seen"] = m["tmpl"].isin(t_int)
m["t_n"] = [sum(t_int[k].values()) if k in t_int else 0 for k in m["tmpl"]]
m["t_pure"] = [len(t_int[k]) == 1 if k in t_int else False for k in m["tmpl"]]
m["t_has_slot"] = m["bio"].map(lambda b: any(t != "O" for t in b))
m["t_maj"] = [maj(t_int[k])[0] if k in t_int else None for k in m["tmpl"]]
m["t_ok"] = m["t_maj"] == m["gold_intent"]


def row(name, s):
    k = int(s["t_ok"].sum())
    n_ = len(s)
    return {"subset": name, "n": n_, "share_dev_pct": round(100 * n_ / len(m), 2),
            "tmpl_major_acc": round(100 * k / max(n_, 1), 2), "ci95": wilson(k, n_),
            "S1_acc": round(100 * s["int_ok"].mean(), 2) if n_ else None}


rows = [row("template seen in train", m[m["t_seen"]]),
        row(" seen, twin too", m[m["t_seen"] & m["twin"]]),
        row(" seen, no twin", m[m["t_seen"] & ~m["twin"]]),
        row(" seen, with slot", m[m["t_seen"] & m["t_has_slot"]]),
        row(" seen, no slot", m[m["t_seen"] & ~m["t_has_slot"]]),
        row(" seen n=1", m[m["t_n"] == 1]),
        row(" seen n=2-4", m[(m["t_n"] >= 2) & (m["t_n"] <= 4)]),
        row(" seen n>=5", m[m["t_n"] >= 5]),
        row(" seen, pure in train", m[m["t_seen"] & m["t_pure"]]),
        row(" seen, multi-intent in train", m[m["t_seen"] & ~m["t_pure"]]),
        row("template unseen", m[~m["t_seen"]]),
        row("all dev", m)]
tb = pd.DataFrame(rows)
show(tb)
save(tb, "q1_template_majority_on_dev.csv")
print("template unseen: mean tokens", round(m[~m["t_seen"]]["n_tok"].mean(), 2), "| seen: mean tokens",
      round(m[m["t_seen"]]["n_tok"].mean(), 2))
print("OOV group x template seen:\n", pd.crosstab(m["train_oov_group"], m["t_seen"]).to_string())

# seen + S1 wrong while template right, and vice versa
sn = m[m["t_seen"]]
print("on seen templates: template right & S1 wrong", int((sn["t_ok"] & ~sn["int_ok"]).sum()),
      "| template wrong & S1 right", int((~sn["t_ok"] & sn["int_ok"]).sum()),
      "| both wrong", int((~sn["t_ok"] & ~sn["int_ok"]).sum()),
      "| both wrong same label", int(((~sn["t_ok"]) & (~sn["int_ok"]) & (sn["t_maj"] == sn["pred_intent"])).sum()))

# ------------------------------------------------------------------ what the numbers imply overall
print("\n=== simple dev-wide composite ===")
frac_t = m["twin"].mean()
frac_s = (m["t_seen"] & ~m["twin"]).mean()
frac_u = (~m["t_seen"]).mean()
print(f"dev mix: twin {100 * frac_t:.1f}% | seen template no twin {100 * frac_s:.1f}% | unseen template {100 * frac_u:.1f}%")
print("S1 acc per bucket:", round(100 * m[m["twin"]]["int_ok"].mean(), 2),
      round(100 * m[m["t_seen"] & ~m["twin"]]["int_ok"].mean(), 2), round(100 * m[~m["t_seen"]]["int_ok"].mean(), 2))
out = {"dup": res_dup, "templates": res_t}
save_json(out, "q1_bayes_brackets.json")
m[["id", "twin", "t_seen", "t_n", "t_pure", "t_maj", "t_ok"]].to_pickle(OUT / "q1_dev_flags.pkl")
mem("end")

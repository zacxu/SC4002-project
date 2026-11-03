"""Q1 supplement. Exact twins are too few for a ceiling, so label agreement is also measured on
(a) keys with the wake word and punctuation removed, (b) order-free content keys, (c) templates keeping the device_type value,
and (d) nearest train neighbour by word TF-IDF cosine, as a function of similarity."""
from common import *
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

train, m, intents = load_all()
m = m.merge(pd.read_pickle(OUT / "q4_linear_dev_pred.pkl"), on="id")
_, a1, _ = load_all(A1_FRAME)
m = m.merge(a1[["id", "int_ok"]].rename(columns={"int_ok": "a1_ok"}), on="id")
rng = np.random.default_rng(42)


def maj(c):
    return sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0])))[0]


def pair_stats(groups):
    """Instance weighted pairwise disagreement over groups with n>=2, plus a cluster bootstrap CI."""
    arr = []
    for c in groups:
        n = sum(c.values())
        if n >= 2:
            arr.append((n, sum(v * (v - 1) for v in c.values()) / (n * (n - 1))))
    a = np.array(arr)
    r = 1 - (a[:, 0] * a[:, 1]).sum() / a[:, 0].sum()
    boot = []
    for _ in range(1000):
        s = a[rng.integers(0, len(a), len(a))]
        boot.append(1 - (s[:, 0] * s[:, 1]).sum() / s[:, 0].sum())
    return len(a), int(a[:, 0].sum()), 100 * r, 100 * np.percentile(boot, 2.5), 100 * np.percentile(boot, 97.5)


STOP = {"olly", "please", "can", "could", "would", "will", "you", "u", "i", "i'd", "i'm", "want", "wanna", "need", "like", "to", "the", "a", "an", "my", "me", "for",
        "of", "is", "are", "do", "does", "it", "that", "this", "tell", "let", "know", "and", "hey", "hi", "ok", "okay"}


def k_wake(toks):
    return " ".join(t.casefold() for t in toks if is_lexical(t) and t.casefold() != "olly")


def k_content(toks):
    return " ".join(sorted({t.casefold() for t in toks if is_lexical(t) and t.casefold() not in STOP}))


def t_keep_device(toks, bio):
    """Template that keeps the value of device_type, which often decides the iot intent."""
    out, j = [], 0
    spans = {s: (t, e) for t, s, e in bio_to_spans(bio)}
    while j < len(toks):
        if j in spans:
            t, e = spans[j]
            if t == "device_type":
                out.extend(x.casefold() for x in toks[j:e])
            else:
                out.append(f"<{t}>")
            j = e
        else:
            out.append(toks[j].casefold())
            j += 1
    return " ".join(out)


KEYS_ = {"exact casefold": lambda r: r.key,
         "no wake word, no punctuation": lambda r: k_wake(r.toks),
         "content words, order free": lambda r: k_content(r.toks),
         "template": lambda r: r.tmpl,
         "template keeping device_type value": lambda r: t_keep_device(r.toks, r.bio)}
rows = []
for name, fn in KEYS_.items():
    g = {}
    for r in train.itertuples(index=False):
        k = fn(r)
        if k:
            g.setdefault(k, Counter())[r.intent] += 1
    dev_keys = [fn(r) for r in m.itertuples(index=False)]
    seen = np.array([k in g for k in dev_keys])
    maj_lab = [maj(g[k])[0] if k in g else None for k in dev_keys]
    ok = np.array([a == b for a, b in zip(maj_lab, m["gold_intent"])])
    n_g, n_i, r_, lo, hi = pair_stats(g.values())
    sub = m[seen]
    rows.append({"key": name, "dev_seen": int(seen.sum()), "dev_seen_pct": round(100 * seen.mean(), 1),
                 "majority_acc_on_seen": round(100 * ok[seen].mean(), 2),
                 "S1_on_seen": round(100 * sub["int_ok"].mean(), 2), "A1_on_seen": round(100 * sub["a1_ok"].mean(), 2),
                 "linear_on_seen": round(100 * sub["lin_ok"].mean(), 2), "S1_on_unseen": round(100 * m[~seen]["int_ok"].mean(), 2),
                 "train_groups_n>=2": n_g, "train_rows_in_them": n_i, "pair_disagree_pct": round(r_, 2), "ci_low": round(lo, 2), "ci_high": round(hi, 2)})
R = pd.DataFrame(rows)
show(R)
save(R, "q1b_key_levels.csv")

# conflicts under the key keeping device_type
g = {}
for r in train.itertuples(index=False):
    g.setdefault(t_keep_device(r.toks, r.bio), Counter())[r.intent] += 1
mult = sorted([k for k, c in g.items() if len(c) > 1], key=lambda k: -sum(g[k].values()))
print("templates keeping device_type with >1 intent:", len(mult), "| rows:", sum(sum(g[k].values()) for k in mult))
show(pd.DataFrame([{"template": k, "n": sum(g[k].values()), "intents": dict(g[k].most_common())} for k in mult[:15]]))

# ---- nearest train neighbour by word 1-2 gram TF-IDF
vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=True, ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32)
Xt = vec.fit_transform(train["utt"])
Xd = vec.transform(m["utt"])
best_sim = np.zeros(len(m), dtype=np.float32)
best_idx = np.zeros(len(m), dtype=np.int64)
Tt = Xt.T.tocsr()
for i in range(0, len(m), 100):
    S = (Xd[i:i + 100] @ Tt).tocsr()
    for j in range(S.shape[0]):
        d, c = S.data[S.indptr[j]:S.indptr[j + 1]], S.indices[S.indptr[j]:S.indptr[j + 1]]
        if len(d):
            k = d.argmax()
            best_sim[i + j], best_idx[i + j] = d[k], c[k]
    del S
mem("neighbours")
nn_int = train["intent"].to_numpy()[best_idx]
m["nn_sim"] = best_sim
m["nn_ok"] = nn_int == m["gold_intent"].to_numpy()
m["nn_utt"] = train["utt"].to_numpy()[best_idx]
m["nn_intent"] = nn_int
print("\n1-NN (word 1-2 gram TF-IDF, train only) accuracy overall:", round(100 * m["nn_ok"].mean(), 2))
bins = [-.01, .2, .4, .5, .6, .7, .8, .9, 1.01]
labels = ["<=.2", ".2-.4", ".4-.5", ".5-.6", ".6-.7", ".7-.8", ".8-.9", ">.9"]
m["sim_bin"] = pd.cut(m["nn_sim"], bins, labels=labels)
rows = []
for l in labels:
    s = m[m["sim_bin"] == l]
    if len(s):
        rows.append({"cosine": l, "n": len(s), "share_pct": round(100 * len(s) / len(m), 1), "nn_label_agrees": round(100 * s["nn_ok"].mean(), 1),
                     "S1": round(100 * s["int_ok"].mean(), 1), "A1": round(100 * s["a1_ok"].mean(), 1), "linear": round(100 * s["lin_ok"].mean(), 1),
                     "mean_tokens": round(s["n_tok"].mean(), 1)})
NN = pd.DataFrame(rows)
show(NN)
save(NN, "q1b_nn_by_similarity.csv")
hi = m[(m["nn_sim"] > .8) & (~m["nn_ok"])]
print("\nhigh-similarity dev utterances whose nearest train neighbour has another label:", len(hi))
show(hi[["utt", "gold_intent", "nn_utt", "nn_intent", "nn_sim"]].round(2).head(30))
save(hi[["utt", "gold_intent", "nn_utt", "nn_intent", "nn_sim"]], "q1b_high_similarity_conflicts.csv")

# ---- consensus errors and a fixed random sample of Stage-1 errors for reading
both = m[(~m["int_ok"]) & (~m["lin_ok"]) & (~m["a1_ok"])]
same = m[(~m["int_ok"]) & (m["pred_intent"] == m["lin_pred"])]
print("\nStage 1, A1 and linear all wrong:", len(both), f"= {100 * len(both) / len(m):.1f}% of dev | S1 and linear agree on a wrong label:", len(same))
sample = m[~m["int_ok"]].sample(60, random_state=20261101)
print("\nfixed random sample of 60 Stage-1 intent errors:")
for r in sample.itertuples(index=False):
    print(f"gold={r.gold_intent:26s} s1={r.pred_intent:26s} lin={r.lin_pred:26s} | {r.annot_utt}")
save(sample[["id", "gold_intent", "pred_intent", "lin_pred", "annot_utt"]], "q1b_error_audit_sample.csv")
mem("end")

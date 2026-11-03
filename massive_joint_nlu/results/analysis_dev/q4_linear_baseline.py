"""Q4: TF-IDF word 1-2 grams + char_wb 3-5 grams with multinomial logistic regression.
Vocabulary, idf and weights come from train only; C is chosen on dev. saga is used because L-BFGS history
for 60 x ~100k weights would exceed the 400 MB process limit."""
import time
import warnings

from common import *
from scipy import sparse
from scipy.stats import binomtest
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=ConvergenceWarning)
train, m, intents = load_all()
_, a1, _ = load_all(A1_FRAME)
m = m.merge(a1[["id", "pred_intent", "int_ok"]].rename(columns={"pred_intent": "a1_pred", "int_ok": "a1_ok"}), on="id")
y_tr = train["intent"].map({k: i for i, k in enumerate(intents)}).to_numpy()
y_dv = m["gold_intent_id"].to_numpy()
mem("loaded")


def vecs(word_cap=None, char_cap=None):
    w = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=True, ngram_range=(1, 2), sublinear_tf=True,
                        dtype=np.float32, max_features=word_cap)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True,
                        dtype=np.float32, max_features=char_cap)
    return w, c


def features(kind="both", word_cap=None, char_cap=None):
    w, c = vecs(word_cap, char_cap)
    parts_tr, parts_dv = [], []
    if kind in ("both", "word"):
        parts_tr.append(w.fit_transform(train["utt"]))
        parts_dv.append(w.transform(m["utt"]))
    if kind in ("both", "char"):
        parts_tr.append(c.fit_transform(train["utt"]))
        parts_dv.append(c.transform(m["utt"]))
    Xt = sparse.hstack(parts_tr, format="csr", dtype=np.float32)
    Xd = sparse.hstack(parts_dv, format="csr", dtype=np.float32)
    Xt.indices = Xt.indices.astype(np.int32)
    Xt.indptr = Xt.indptr.astype(np.int32)
    return Xt, Xd, [p.shape[1] for p in parts_tr]


def fit(Xt, C, max_iter=200, tol=1e-3):
    t0 = time.time()
    clf = LogisticRegression(solver="saga", C=C, max_iter=max_iter, tol=tol, random_state=42)
    clf.fit(Xt, y_tr)
    return clf, time.time() - t0


Xt, Xd, sizes = features()
print("features word, char:", sizes, "| total", Xt.shape[1], "| nnz train", Xt.nnz, "| params if used as a wide head:",
      Xt.shape[1] * 60 + 60, flush=True)
mem("features")

rows, best = [], None
for C in (1.0, 10.0, 100.0):
    clf, sec = fit(Xt, C)
    pred = clf.predict(Xd)
    acc = 100 * (pred == y_dv).mean()
    tr_acc = 100 * (clf.predict(Xt) == y_tr).mean()
    rows.append({"C": C, "dev_acc": round(acc, 2), "train_acc": round(tr_acc, 2), "iters": int(clf.n_iter_[0]), "sec": round(sec)})
    print(rows[-1], flush=True)
    mem(f"C={C}")
    if best is None or acc > best[0]:
        best = (acc, C, clf)
    else:
        del clf
grid = pd.DataFrame(rows)
save(grid, "q4_c_grid.csv")
acc, C, clf = best
print(f"selected C={C}: dev accuracy {acc:.2f}", flush=True)

proba = clf.predict_proba(Xd).astype(np.float32)
pred = proba.argmax(1)
top3 = np.argsort(-proba, axis=1)[:, :3]
m["lin_id"] = pred
m["lin_pred"] = [intents[i] for i in pred]
m["lin_ok"] = pred == y_dv
m["lin_p"] = proba.max(1)
m["lin_top3_ok"] = [(y in t) for y, t in zip(y_dv, top3)]
m[["id", "lin_pred", "lin_ok", "lin_p", "lin_top3_ok"]].to_pickle(OUT / "q4_linear_dev_pred.pkl")
print("top-3 accuracy:", round(100 * m["lin_top3_ok"].mean(), 2), "| gold in top 3 among linear errors:",
      int(m[~m["lin_ok"]]["lin_top3_ok"].sum()), "of", int((~m["lin_ok"]).sum()))


def compare(ok_name, pred_name, label):
    ok = m[ok_name].to_numpy()
    lin = m["lin_ok"].to_numpy()
    b, c = int((~ok & lin).sum()), int((ok & ~lin).sum())
    both = int((~ok & ~lin).sum())
    same_wrong = int(((~ok) & (~lin) & (m[pred_name] == m["lin_pred"]).to_numpy()).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    agree = 100 * (m[pred_name] == m["lin_pred"]).mean()
    acc_agree = 100 * ok[(m[pred_name] == m["lin_pred"]).to_numpy()].mean()
    out = {"model": label, "acc": round(100 * ok.mean(), 2), "linear_acc": round(100 * lin.mean(), 2),
           "model_errors": int((~ok).sum()), "linear_fixes": b, "linear_breaks": c, "both_wrong": both,
           "both_wrong_same_label": same_wrong, "both_wrong_diff_label": both - same_wrong,
           "oracle_union_acc": round(100 * (1 - both / len(m)), 2), "mcnemar_p": round(p, 4),
           "agreement_pct": round(agree, 1), "acc_when_agree": round(acc_agree, 2)}
    return out


cmp_tbl = pd.DataFrame([compare("int_ok", "pred_intent", "Stage 1"), compare("a1_ok", "a1_pred", "A1_ft_recipe")])
show(cmp_tbl)
save(cmp_tbl, "q4_complementarity.csv")

print("\nper OOV group:")
rows = []
for g in GROUPS + ["Overall"]:
    s = m if g == "Overall" else m[m["train_oov_group"] == g]
    rows.append({"group": g, "n": len(s), "S1": round(100 * s["int_ok"].mean(), 2), "A1": round(100 * s["a1_ok"].mean(), 2),
                 "linear": round(100 * s["lin_ok"].mean(), 2),
                 "lin_fixes_S1": int((~s["int_ok"] & s["lin_ok"]).sum()), "lin_breaks_S1": int((s["int_ok"] & ~s["lin_ok"]).sum()),
                 "lin_fixes_A1": int((~s["a1_ok"] & s["lin_ok"]).sum()), "lin_breaks_A1": int((s["a1_ok"] & ~s["lin_ok"]).sum())})
gt = pd.DataFrame(rows)
show(gt)
save(gt, "q4_linear_by_group.csv")

# ---- confidence-gated override, threshold cross-validated on dev (no Stage-1 probabilities are available here)
print("\nconfidence-gated ensemble, 5-fold CV of the threshold on dev:")
rs = np.random.RandomState(42)
fold = rs.randint(0, 5, len(m))
rows = []
for name, pn in (("Stage 1", "pred_intent"), ("A1_ft_recipe", "a1_pred")):
    base_pred = m[pn].to_numpy()
    lin_pred = m["lin_pred"].to_numpy()
    gold = m["gold_intent"].to_numpy()
    ps = m["lin_p"].to_numpy()
    cand = np.unique(np.quantile(ps, np.linspace(0, 1, 101)))
    res = np.zeros(len(m), dtype=bool)
    chosen = []
    for f in range(5):
        tr = fold != f
        te = fold == f
        best_t, best_a = None, -1
        for t in cand:
            a = ((np.where(ps[tr] >= t, lin_pred[tr], base_pred[tr])) == gold[tr]).mean()
            if a > best_a:
                best_a, best_t = a, t
        chosen.append(round(float(best_t), 3))
        res[te] = np.where(ps[te] >= best_t, lin_pred[te], base_pred[te]) == gold[te]
    rows.append({"base": name, "base_acc": round(100 * (base_pred == gold).mean(), 2), "linear_acc": round(100 * (lin_pred == gold).mean(), 2),
                 "gated_cv_acc": round(100 * res.mean(), 2), "thresholds": chosen})
ens = pd.DataFrame(rows)
show(ens)
save(ens, "q4_gated_ensemble.csv")

# ---- per intent gains
rows = []
for it in intents:
    s = m[m["gold_intent"] == it]
    if len(s):
        rows.append({"intent": it, "dev": len(s), "S1_recall": round(100 * s["int_ok"].mean(), 1), "A1_recall": round(100 * s["a1_ok"].mean(), 1),
                     "lin_recall": round(100 * s["lin_ok"].mean(), 1), "lin_minus_S1": round(100 * (s["lin_ok"].mean() - s["int_ok"].mean()), 1),
                     "lin_minus_S1_n": int(s["lin_ok"].sum() - s["int_ok"].sum())})
pi = pd.DataFrame(rows).sort_values("lin_minus_S1_n")
save(pi, "q4_per_intent.csv")
print("\nintents where linear loses most to Stage 1:")
show(pi.head(8))
print("intents where linear gains most on Stage 1:")
show(pi.tail(8))

# ---- examples
print("\nS1 wrong, linear right (sample):")
show(m[~m["int_ok"] & m["lin_ok"]].sample(12, random_state=1)[["utt", "gold_intent", "pred_intent", "lin_p", "train_oov_group"]])
print("\nS1 right, linear wrong (sample):")
show(m[m["int_ok"] & ~m["lin_ok"]].sample(12, random_state=1)[["utt", "gold_intent", "lin_pred", "lin_p", "train_oov_group"]])
mem("after compare")
del clf, proba, Xt, Xd

# ---- feature families at the selected C
print("\nfeature families and caps at the selected C:", flush=True)
rows = []
for name, kw in (("word 1-2 only", dict(kind="word")), ("char_wb 3-5 only", dict(kind="char")),
                 ("both, cap 5k+5k", dict(word_cap=5000, char_cap=5000)), ("both, cap 10k+10k", dict(word_cap=10000, char_cap=10000)),
                 ("both, cap 20k+20k", dict(word_cap=20000, char_cap=20000))):
    Xt2, Xd2, sz = features(**kw)
    clf2, sec = fit(Xt2, C)
    p2 = clf2.predict(Xd2)
    ok2 = p2 == y_dv
    gr = {g[:7]: round(100 * ok2[(m["train_oov_group"] == g).to_numpy()].mean(), 2) for g in GROUPS}
    rows.append({"features": name, "n_features": int(sum(sz)), "wide_params": int(sum(sz)) * 60 + 60, "dev_acc": round(100 * ok2.mean(), 2),
                 **gr, "fixes_S1": int((~m["int_ok"].to_numpy() & ok2).sum()), "breaks_S1": int((m["int_ok"].to_numpy() & ~ok2).sum()), "sec": round(sec)})
    print(rows[-1], flush=True)
    del Xt2, Xd2, clf2
save(pd.DataFrame(rows), "q4_feature_families.csv")
mem("end")

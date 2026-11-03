"""Q4 supplement: C=100 stopped at max_iter=200, at the edge of the requested grid. This checks that neither convergence
nor a wider C range changes the conclusion. Reported as a sensitivity check; the selected model stays the 3-value grid winner."""
import time
import warnings

from common import *
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=ConvergenceWarning)
train, m, intents = load_all()
y_tr = train["intent"].map({k: i for i, k in enumerate(intents)}).to_numpy()
y_dv = m["gold_intent_id"].to_numpy()
w = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=True, ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32)
c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True, dtype=np.float32)
Xt = sparse.hstack([w.fit_transform(train["utt"]), c.fit_transform(train["utt"])], format="csr", dtype=np.float32)
Xd = sparse.hstack([w.transform(m["utt"]), c.transform(m["utt"])], format="csr", dtype=np.float32)
Xt.indices = Xt.indices.astype(np.int32)
Xt.indptr = Xt.indptr.astype(np.int32)
rows = []
for C, it, tol in ((3.0, 300, 1e-3), (30.0, 300, 1e-3), (100.0, 1000, 1e-4), (300.0, 1000, 1e-3)):
    t0 = time.time()
    clf = LogisticRegression(solver="saga", C=C, max_iter=it, tol=tol, random_state=42).fit(Xt, y_tr)
    ok = clf.predict(Xd) == y_dv
    gr = {g[:7]: round(100 * ok[(m["train_oov_group"] == g).to_numpy()].mean(), 2) for g in GROUPS}
    rows.append({"C": C, "max_iter": it, "tol": tol, "iters": int(clf.n_iter_[0]), "dev_acc": round(100 * ok.mean(), 2), **gr,
                 "fixes_S1": int((~m["int_ok"].to_numpy() & ok).sum()), "breaks_S1": int((m["int_ok"].to_numpy() & ~ok).sum()), "sec": round(time.time() - t0)})
    print(rows[-1], flush=True)
    del clf
save(pd.DataFrame(rows), "q4b_sensitivity.csv")
mem("end")

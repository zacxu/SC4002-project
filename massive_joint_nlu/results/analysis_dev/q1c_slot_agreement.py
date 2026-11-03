"""Q1 supplement: slot annotation agreement on near-identical utterances, a rough slot-side counterpart of the intent ceiling.
Pairs inside a group are compared on the multiset of (type, value), so word order and the wake word do not matter."""
from common import *

train, m, intents = load_all()
rng = np.random.default_rng(42)
STOP = {"olly", "please", "can", "could", "would", "will", "you", "u", "i", "i'd", "i'm", "want", "wanna", "need", "like", "to", "the", "a", "an", "my", "me", "for",
        "of", "is", "are", "do", "does", "it", "that", "this", "tell", "let", "know", "and", "hey", "hi", "ok", "okay"}


def k_wake(toks):
    return " ".join(t.casefold() for t in toks if is_lexical(t) and t.casefold() != "olly")


def k_content(toks):
    return " ".join(sorted({t.casefold() for t in toks if is_lexical(t) and t.casefold() not in STOP}))


def spans_of(r):
    cf = [t.casefold() for t in r.toks]
    return Counter((t, " ".join(cf[s:e])) for t, s, e in bio_to_spans(r.bio))


df = pd.concat([train, m[train.columns]], ignore_index=True)
rows = []
for name, fn in (("no wake word, no punctuation", k_wake), ("content words, order free", k_content)):
    groups = {}
    for r in df.itertuples(index=False):
        k = fn(r.toks)
        if k:
            groups.setdefault(k, []).append(spans_of(r))
    tp = fp = fn_ = 0
    per = []
    same = tot = 0
    for g in groups.values():
        if len(g) < 2:
            continue
        gt = gf = 0
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g[i], g[j]
                t = sum((a & b).values())
                gt += t
                gf += sum((a - b).values()) + sum((b - a).values())
                same += a == b
                tot += 1
        per.append((gt, gf))
    per = np.array(per)
    f1 = 100 * 2 * per[:, 0].sum() / (2 * per[:, 0].sum() + per[:, 1].sum())
    boot = []
    for _ in range(1000):
        s = per[rng.integers(0, len(per), len(per))]
        boot.append(100 * 2 * s[:, 0].sum() / (2 * s[:, 0].sum() + s[:, 1].sum()))
    rows.append({"key": name, "groups": len(per), "pairs": tot, "identical_annotation_pct": round(100 * same / tot, 1), "pairwise_span_f1": round(f1, 2),
                 "ci_low": round(np.percentile(boot, 2.5), 2), "ci_high": round(np.percentile(boot, 97.5), 2)})
R = pd.DataFrame(rows)
show(R)
save(R, "q1c_slot_agreement.csv")
mem("end")

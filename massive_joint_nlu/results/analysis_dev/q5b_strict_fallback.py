"""Q5 supplement: how many dev Train-OOV tokens could reach an existing train word by a parameter-free lookup fallback.
Spans stay on the original tokens; only the word-id lookup of an unseen token would change. Train vocabulary only, no Word2Vec."""
from common import *

train, m, intents = load_all()
train_lex = {t.casefold() for toks in train["toks"] for t in toks if is_lexical(t)}
rows = []
for r in m.itertuples(index=False):
    for t, s in zip(r.toks, r.smask):
        cf = t.casefold()
        if is_lexical(t) and cf not in train_lex:
            rows.append({"tok": cf, "in_slot": s})
O = pd.DataFrame(rows)


def dl1(a, b):
    """True if Damerau-Levenshtein distance is exactly at most 1."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        if len(diff) == 1:
            return True
        return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]
    if la > lb:
        a, b = b, a
    return any(b[:i] + b[i + 1:] == a for i in range(len(b)))


by_len = {}
for w in train_lex:
    by_len.setdefault(len(w), []).append(w)


def rule(tok):
    m_ = re.fullmatch(r"(.+?)['’](s|d|ll|re|ve|m|t)", tok)
    if m_ and m_.group(1) in train_lex:
        return "possessive or clitic -> base", m_.group(1)
    if "-" in tok:
        p = tok.split("-")
        if all(x in train_lex for x in p):
            return "hyphen parts all known", " ".join(p)
    for suf, rep in (("ies", "y"), ("es", ""), ("s", ""), ("ing", ""), ("ing", "e"), ("ed", ""), ("ed", "e"), ("d", ""), ("er", ""), ("ers", ""), ("ly", ""), ("'", "")):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            b = tok[: -len(suf)] + rep
            if b in train_lex:
                return "suffix strip", b
    if len(tok) >= 5:
        cands = [w for L in (len(tok) - 1, len(tok), len(tok) + 1) for w in by_len.get(L, []) if dl1(tok, w)]
        if cands:
            return "edit distance 1", sorted(cands)[0]
    return "none", ""


info = {t: rule(t) for t in O["tok"].unique()}
O["rule"] = O["tok"].map(lambda t: info[t][0])
O["target"] = O["tok"].map(lambda t: info[t][1])
tab = O.groupby("rule").agg(tokens=("tok", "size"), types=("tok", "nunique"), in_slot=("in_slot", "sum")).reset_index().sort_values("tokens", ascending=False)
tab["token_pct"] = (100 * tab["tokens"] / len(O)).round(1)
tab["slot_token_pct"] = (100 * tab["in_slot"] / O["in_slot"].sum()).round(1)
show(tab)
save(tab, "q5b_strict_fallback.csv")
for k in ("possessive or clitic -> base", "hyphen parts all known", "suffix strip", "edit distance 1"):
    s = O[O["rule"] == k].drop_duplicates("tok")
    print(f"\n{k}: ", [f"{a}->{b}" for a, b in zip(s["tok"], s["target"])])
save(O.drop_duplicates("tok")[["tok", "rule", "target"]], "q5b_fallback_map_dev_only_inspection.csv")
mem("end")

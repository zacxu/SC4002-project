"""Q5: how much of the dev Train-OOV vocabulary a frequency-ordered Word2Vec key list covers, and the legal K.
The filtered list mirrors ExtVocab: keys in file order, skipping train-vocabulary words and blank keys.
The w2v matrix is not loaded; only the key list is read."""
from common import *

train, m, intents = load_all()
train_lex = {t.casefold() for toks in train["toks"] for t in toks if is_lexical(t)}
word_vocab = {t.casefold() for toks in train["toks"] for t in toks}
meta = json.loads(KEYS.read_text(encoding="utf-8"))
keys = meta["keys"]
del meta
mem("keys")

rank = {}
r_ = 0
for w in keys:
    if w in word_vocab or not w.strip():
        continue
    r_ += 1
    rank[w] = r_
N_EXTRA = r_
print("keys:", len(keys), "| filtered non-train keys:", N_EXTRA)
top_words = [w for w in keys[:30000]]
key_set_all_lower = None
del keys

rows = []
for r in m.itertuples(index=False):
    spans = bio_to_spans(r.bio)
    for i, t in enumerate(r.toks):
        cf = t.casefold()
        if is_lexical(t) and cf not in train_lex:
            ty = next((s[0] for s in spans if s[1] <= i < s[2]), None)
            rows.append({"id": r.id, "group": r.train_oov_group, "tok": cf, "in_slot": r.smask[i], "slot_type": ty,
                         "rank": rank.get(cf, 10 ** 9)})
O = pd.DataFrame(rows)
print("dev Train-OOV lexical tokens:", len(O), "| types:", O["tok"].nunique(), "| in gold slots:", int(O["in_slot"].sum()),
      "| types in slots:", O[O["in_slot"]]["tok"].nunique(), "| outside:", int((~O["in_slot"]).sum()))
lex_total = sum(is_lexical(t) for toks in m["toks"] for t in toks)
print("share of all dev lexical tokens:", round(100 * len(O) / lex_total, 2), "%  of", lex_total)

BASE = 4_344_472
CAP = 20_000_000
TRAIN_ROWS = 4731
MAP300, MAP100 = 300 * 300 + 300, 100 * 300 + 300
Ks = [5000, 10000, 20000, 30000, 46000, 100000, 150000, N_EXTRA]
# utterance and span level helpers
slot_ids = O[O["in_slot"]].groupby("id")["rank"].max()       # worst rank of an OOV slot token per utterance
all_ids = O.groupby("id")["rank"].max()
slot_utts = m[m["train_oov_group"] == "Slot-Train-OOV"]["id"]
nonslot_utts = m[m["train_oov_group"] == "NonSlot-Train-OOV"]["id"]
rows = []
for K in Ks:
    c = O["rank"] <= K
    types = O.drop_duplicates("tok")
    types_s = O[O["in_slot"]].drop_duplicates("tok")
    types_n = O[~O["in_slot"]].drop_duplicates("tok")
    p300 = BASE + (TRAIN_ROWS + K) * 300 + MAP300
    p100 = BASE + (TRAIN_ROWS + K) * 100 + MAP100
    rows.append({"K": K, "tok_all_%": round(100 * c.mean(), 1), "tok_slot_%": round(100 * c[O["in_slot"]].mean(), 1),
                 "tok_nonslot_%": round(100 * c[~O["in_slot"]].mean(), 1),
                 "type_all_%": round(100 * (types["rank"] <= K).mean(), 1), "type_slot_%": round(100 * (types_s["rank"] <= K).mean(), 1),
                 "type_nonslot_%": round(100 * (types_n["rank"] <= K).mean(), 1),
                 "slotoov_utts_fully_covered_%": round(100 * (slot_ids.reindex(slot_utts) <= K).mean(), 1),
                 "nonslot_utts_fully_covered_%": round(100 * (all_ids.reindex(nonslot_utts) <= K).mean(), 1),
                 "params_300d_M": round(p300 / 1e6, 3), "legal_300d": p300 <= CAP, "params_100d_M": round(p100 / 1e6, 3), "legal_100d": p100 <= CAP})
C = pd.DataFrame(rows)
show(C)
save(C, "q5_coverage_curve.csv")

# ---- largest legal K
budget = CAP - BASE
print("\nbudget above the base model:", budget)
print("largest K ignoring the map: 300d", budget // 300 - TRAIN_ROWS, "| 100d", budget // 100 - TRAIN_ROWS)
ADAPT = {"none": 0, "bpe": 148_800, "char": 36_464, "ngram": 978_816, "bpe+char": 148_800 + 36_464,
         "bpe+ngram": 148_800 + 978_816, "all3": 148_800 + 978_816 + 36_464}
rows = []
for name, a in ADAPT.items():
    k3 = (CAP - BASE - a - MAP300) // 300 - TRAIN_ROWS
    k1 = (CAP - BASE - a - MAP100) // 100 - TRAIN_ROWS
    rows.append({"adapters": name, "adapter_params": a, "max_K_300d_with_map": k3, "max_K_100d_with_map": k1,
                 "ext46k_300d_total": BASE + a + MAP300 + (TRAIN_ROWS + 46000) * 300,
                 "ext150k_100d_total": BASE + a + MAP100 + (TRAIN_ROWS + 150000) * 100,
                 "ext46k_legal": k3 >= 46000, "ext150k_100d_legal": k1 >= 150000})
L = pd.DataFrame(rows)
show(L)
save(L, "q5_legal_K.csv")
for K, d, mp in ((47153, 300, MAP300), (47154, 300, MAP300), (151521, 100, MAP100), (151522, 100, MAP100)):
    print(f"  check K={K} d={d}: total {BASE + (TRAIN_ROWS + K) * d + mp:,}")

# ---- what stays uncovered at K=46k
print("\n=== uncovered at K=46,000 ===")
U = O[O["rank"] > 46000].copy()
print("uncovered tokens:", len(U), f"= {100 * len(U) / len(O):.1f}% of OOV tokens | in slots:", int(U["in_slot"].sum()),
      "| types:", U["tok"].nunique())
print("of those, inside key list beyond 46k:", int((U["rank"] < 10 ** 9).sum()), "tokens,", U[U["rank"] < 10 ** 9]["tok"].nunique(),
      "types | absent from all keys:", int((U["rank"] == 10 ** 9).sum()), "tokens,", U[U["rank"] == 10 ** 9]["tok"].nunique(), "types")

# dictionary for typo candidates: train words and the 30k most frequent keys
dict_words = sorted(set(w for w in word_vocab if is_lexical(w)) | set(top_words))
ALPHA = "abcdefghijklmnopqrstuvwxyz0123456789'-"
ci = {c: i for i, c in enumerate(ALPHA)}


def bag(w):
    v = np.zeros(len(ALPHA) + 1, dtype=np.int16)
    for ch in w:
        v[ci.get(ch, len(ALPHA))] += 1
    return v


B = np.stack([bag(w) for w in dict_words])
lens = np.array([len(w) for w in dict_words])
in_train = np.array([w in train_lex for w in dict_words])


def dl(a, b, maxd=2):
    """Damerau-Levenshtein (optimal string alignment) with early exit."""
    if abs(len(a) - len(b)) > maxd:
        return maxd + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > maxd:
            return maxd + 1
        prev2, prev = prev, cur
    return prev[-1]


def nearest(w):
    maxd = 1 if len(w) < 7 else 2
    if len(w) < 4:
        return None, None, None
    d = np.abs(B - bag(w)).sum(1)
    cand = np.where((d <= 2 * maxd) & (np.abs(lens - len(w)) <= maxd))[0]
    best = (None, 9, False)
    for i in cand:
        x = dl(w, dict_words[i], maxd)
        if x <= maxd and (x < best[1] or (x == best[1] and in_train[i] and not best[2])):
            best = (dict_words[i], x, bool(in_train[i]))
    return best if best[0] is not None else (None, None, None)


def shape(w):
    if re.fullmatch(r"[0-9]+", w):
        return "digits only"
    if re.search(r"[0-9]", w):
        return "has digit"
    if re.search(r"['’]", w):
        return "apostrophe"
    if "-" in w:
        return "hyphen"
    return "alpha"


kset_base = None
tinfo = []
for tok, g in U.groupby("tok"):
    sh = shape(tok)
    base = None
    if sh == "apostrophe":
        base = re.split(r"['’]", tok)[0]
    elif sh == "hyphen":
        base = tok.split("-")
    ok_base = None
    if isinstance(base, str):
        ok_base = base in word_vocab or rank.get(base, 10 ** 9) <= 46000
    elif isinstance(base, list):
        ok_base = all(p in word_vocab or rank.get(p, 10 ** 9) <= 46000 for p in base)
    near, dist, near_train = nearest(tok)
    tinfo.append({"tok": tok, "n": len(g), "n_slot": int(g["in_slot"].sum()), "rank": int(g["rank"].iloc[0]),
                  "in_keys": g["rank"].iloc[0] < 10 ** 9, "shape": sh, "base_known": ok_base, "near": near, "dist": dist,
                  "near_in_train": near_train, "slot_types": dict(Counter(g["slot_type"].dropna()).most_common(3))})
TI = pd.DataFrame(tinfo).sort_values(["n", "tok"], ascending=[False, True])
save(TI, "q5_uncovered_at_46k_types.csv")
print("\nshape x in-keys, tokens (types):")
print(TI.groupby(["shape", "in_keys"]).agg(types=("tok", "size"), tokens=("n", "sum"), slot_tokens=("n_slot", "sum")).to_string())
print("\nslot type of uncovered in-slot tokens:", U[U["in_slot"]]["slot_type"].value_counts().head(15).to_dict())
print("\nall uncovered types (token, n, n_slot, in_keys, shape, near, slot types):")
for r in TI.itertuples(index=False):
    print(f"{r.tok}\t{r.n}\t{r.n_slot}\t{'K' if r.in_keys else '-'}\t{r.shape}\t{r.near if r.near else ''}\t{r.slot_types if r.slot_types else ''}")

# ---- tokens covered by the list, what they look like at 46k
Cov = O[O["rank"] <= 46000]
print("\nmost frequent covered OOV types at 46k:", Counter(Cov["tok"]).most_common(40))
# ---- normalisation fallback: strip possessive, split hyphen; how much would that add at K=46k
fall = 0
for t, n_ in zip(TI["tok"], TI["n"]):
    if re.search(r"['’]", t) or "-" in t:
        parts = re.split(r"['’\-]", t)
        if parts[0] in word_vocab or rank.get(parts[0], 10 ** 9) <= 46000:
            fall += n_
print("uncovered tokens with apostrophe or hyphen whose first part is known at 46k:", fall)
mem("end")

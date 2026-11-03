"""Verifies inputs: frame vs CSV, recomputed metrics, OOV groups, vocabulary numbers the other scripts rely on."""
from common import *

train, m, intents = load_all()
mem("loaded")

# ---- no test row anywhere
assert set(m["partition"]) == {"dev"} and set(train["partition"]) == {"train"}

# ---- recompute Stage-1 metrics
acc = 100 * m["int_ok"].mean()
tp, fp, fn = slot_counts(m["gold_tags"], m["pred_tags"])
print(f"Stage 1 recomputed: intent {acc:.2f} | slot F1 {f1_from(tp, fp, fn):.2f} | TP {tp} FP {fp} FN {fn} | "
      f"joint {(acc + f1_from(tp, fp, fn)) / 2:.2f}")

# ---- recompute OOV groups exactly as build_oov_metadata
train_lex = {t.casefold() for toks in train["toks"] for t in toks if is_lexical(t)}
word_counter = Counter(t.casefold() for toks in train["toks"] for t in toks)
print("train lexical casefold types:", len(train_lex), "| all train casefold types:", len(word_counter),
      "| word_vocab with PAD/UNK:", len(word_counter) + 2)
bad = 0
rows = []
for r in m.itertuples(index=False):
    t_oov = [is_lexical(t) and t.casefold() not in train_lex for t in r.toks]
    n_s = sum(o and s for o, s in zip(t_oov, r.smask))
    g = "No-Train-OOV" if not any(t_oov) else ("Slot-Train-OOV" if n_s else "NonSlot-Train-OOV")
    if g != r.train_oov_group or sum(t_oov) != r.n_train_oov or n_s != r.n_slot_train_oov or len(r.toks) != r.n_tokens:
        bad += 1
print("OOV metadata mismatches vs frame:", bad)
print(m["train_oov_group"].value_counts().to_dict())

# ---- groups table vs dev_oov_table.csv
rows = []
for g in GROUPS + ["Overall"]:
    s = m if g == "Overall" else m[m["train_oov_group"] == g]
    tp, fp, fn = slot_counts(s["gold_tags"], s["pred_tags"])
    rows.append({"group": g, "n": len(s), "intent_acc": round(100 * s["int_ok"].mean(), 2),
                 "slot_f1": round(f1_from(tp, fp, fn), 2), "tp": tp, "fp": fp, "fn": fn})
show(pd.DataFrame(rows))
print(pd.read_csv(ROOT / "results/stage1_baseline/dev_oov_table.csv").round(2).to_string(index=False))

# ---- Word2Vec keys
meta = json.loads(KEYS.read_text(encoding="utf-8"))
keys = meta["keys"]
print("keys:", len(keys), "| n_raw:", meta.get("n_raw"), "| pos_ratio:", meta.get("pos_ratio"), "| source:", meta.get("source"))
kset = set(keys)
assert len(kset) == len(keys)
cov = sum(w in kset for w in word_counter)
print("train casefold types with a Word2Vec row:", cov, "| without:", len(word_counter) - cov)
print("non-train keys:", sum(w not in word_counter and bool(w.strip()) for w in keys))
print("first 25 keys:", keys[:25])
print("keys at ranks 46k, 100k, 150k, 250k:", [keys[i:i + 4] for i in (46000, 100000, 150000, 250000)])
del keys, kset, meta

# ---- label inventory (train and dev only)
tr_lab = {t for b in train["bio"] for t in b}
dv_lab = {t for b in m["bio"] for t in b}
print("BIO labels in train:", len(tr_lab), "| in dev:", len(dv_lab), "| train+dev:", len(tr_lab | dv_lab),
      "| dev-only labels:", sorted(dv_lab - tr_lab))
tr_t = {t[2:] for t in tr_lab if t != "O"}
dv_t = {t[2:] for t in dv_lab if t != "O"}
print("slot types in train:", len(tr_t), "| in dev:", len(dv_t), "| dev-only types:", sorted(dv_t - tr_t))
print("intents dev not in train:", sorted(set(m["intent"]) - set(train["intent"])),
      "| train intents absent from dev:", sorted(set(train["intent"]) - set(m["intent"])))
mem("end")

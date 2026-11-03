"""Builds report.md from the saved tables, so report numbers cannot drift from the scripts' outputs.
The manual audit labels below are one reader's judgement: L = another label is at least as defensible or the gold label looks wrong,
U = nonsense utterance with the catch-all gold label, M = gold is clear and the model is wrong."""
from common import *

R = lambda n: pd.read_csv(OUT / n)


def md(df, cols=None, rename=None):
    d = df[cols] if cols else df
    if rename:
        d = d.rename(columns=rename)
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" if d[c].dtype == object else "---:" for c in d.columns) + "|"
    rows = []
    for row in d.itertuples(index=False):
        rows.append("| " + " | ".join("" if (isinstance(x, float) and np.isnan(x)) else str(x).replace(" | ", " / ") for x in row) + " |")
    return "\n".join([head, sep] + rows)


# ---- manual audit files
s = R("q1b_error_audit_sample.csv")
lab = list("MLLMMMLLLMMLLMLLMLLLMMLLLMMMMLMLLMLMMULMLMLMLLLMMMULMLLLLMLM")
assert len(lab) == 60 and lab.count("L") == 31 and lab.count("U") == 2
s["audit"] = lab
save(s, "q1b_error_audit_sample.csv")
t = R("q2_top5_pair_errors.csv")
lab2 = list("MLLMLLLLMLLMLLMMLLLLLLL") + list("LLLLMLMLLL") + list("LMLLLMLL") + list("MMLLUML") + list("LMLMLLM")
assert len(lab2) == len(t) == 55
t["audit"] = lab2
save(t, "q2_top5_pair_errors.csv")
aud = t.groupby("pair", sort=False)["audit"].value_counts().unstack(fill_value=0).reset_index()

ex = []
for pair, g in t.groupby("pair", sort=False):
    pick = g.sample(3, random_state=7)
    for r in pick.itertuples(index=False):
        ex.append({"pair": pair, "gold": r.gold, "pred": r.pred, "annotated utterance": r.annot_utt.replace("|", "/"), "audit": r.audit})
EX = pd.DataFrame(ex)
save(EX, "q2_examples.csv")

P = pd.read_pickle(OUT / "q3_pred_spans.pkl")
G = pd.read_pickle(OUT / "q3_gold_spans.pkl")
noslot_ids = set(P["id"]) - set(G["id"])
print("spurious spans in utterances with no gold slot:", int(((P["cat"] == "spurious") & P["id"].isin(noslot_ids)).sum()))

tw = R("q1_twin_vs_rest.csv")
kl = R("q1b_key_levels.csv")
tm = R("q1_template_majority_on_dev.csv")
nn = R("q1b_nn_by_similarity.csv")
sa = R("q1c_slot_agreement.csv")
tp20 = R("q2_top20_confused_pairs.csv")
ig = R("q2_intent_by_group.csv")
tx = R("q3_taxonomy_by_group.csv")
wi = R("q3_what_if_by_category.csv")
os_ = R("q3_oov_span_rates.csv")
pt = R("q3_per_type.csv")
wt = R("q3_worst_types.csv")
ba = R("q3b_stage1_vs_A1.csv")
cg = R("q4_c_grid.csv")
cp = R("q4_complementarity.csv")
lg = R("q4_linear_by_group.csv")
ge = R("q4_gated_ensemble.csv")
ff = R("q4_feature_families.csv")
sn = R("q4b_sensitivity.csv")
cv = R("q5_coverage_curve.csv")
lk = R("q5_legal_K.csv")
gb = R("q6_gold_blocked.csv")
ms = R("q6_mask_simulation.csv")
lt = R("q7_by_length.csv")
hv = R("q7_gold_spans_by_value_history.csv")
sb = R("q5b_strict_fallback.csv")
tm_show = tm.copy()
tm_show["tmpl_major_acc"] = tm_show["tmpl_major_acc"].astype(object)
mask = tm_show["subset"].isin(["template unseen", "all dev"])
tm_show.loc[mask, "tmpl_major_acc"] = "n/a"
tm_show.loc[mask, "ci95"] = ""
cp["mcnemar_p"] = ["<0.0001" if x < 0.0001 else str(x) for x in cp["mcnemar_p"]]
bc = R("q7_boundary_convention.csv")
ri = R("q7_rare_intents.csv")
ns = R("q7_bootstrap_noise.csv")
pn = R("q7b_paired_noise.csv")

tcols = ["group", "gold", "F1", "g%_exact", "g%_wrong_type_same_span", "g%_boundary_same_type", "g%_overlap_other", "g%_missed"]
trn = {"g%_exact": "exact %", "g%_wrong_type_same_span": "right span wrong type %", "g%_boundary_same_type": "boundary same type %",
       "g%_overlap_other": "overlap other type %", "g%_missed": "missed %"}
pcols = ["group", "pred", "p%_exact", "p%_wrong_type_same_span", "p%_boundary_same_type", "p%_overlap_other", "p%_spurious"]
prn = {"p%_exact": "exact %", "p%_wrong_type_same_span": "right span wrong type %", "p%_boundary_same_type": "boundary same type %",
       "p%_overlap_other": "overlap other type %", "p%_spurious": "spurious %"}
tc = ["type", "train_spans", "support", "pred", "tp", "fp", "fn", "precision", "recall", "f1", "oov_span_pct"]

doc = f"""# Dev analysis of Stage 1 (train and dev only)

Test rows were dropped on read. Recomputed from `dev_pred_frame.pkl` and the CSV: intent 82.83, slot F1 66.09, joint 74.46, TP 1334, FP 691, FN 678; all OOV group labels match. Scripts and full tables are in this folder. "A1" is `A1_ft_recipe` from the running ladder, added where it changes a conclusion.

## 1. Label noise and intent ceiling

**Exact twins.** 11 dev utterances (0.54%) have a casefold twin in train: intent agrees 10/11, BIO 11/11. Train has 46 duplicate groups (92 rows): 4 conflict on intent, 1 on annotation, 5 on either. Exact twins are too few to carry a ceiling, so near-duplicate keys were added.

{md(kl, ["key", "dev_seen", "dev_seen_pct", "majority_acc_on_seen", "S1_on_seen", "A1_on_seen", "linear_on_seen", "S1_on_unseen", "train_groups_n>=2", "train_rows_in_them", "pair_disagree_pct", "ci_low", "ci_high"])}

**Templates.** 10,521 unique templates in 11,514 train rows; 517 occur at least twice (1,510 rows); 51 of those (9.9%) map to more than one intent (154 rows, 1.3% of train). Many conflicts are an artefact of abstraction: `turn on the <device_type>` is decided by the device value.

{md(tm_show, ["subset", "n", "share_dev_pct", "tmpl_major_acc", "ci95", "S1_acc"])}

**Nearest train neighbour by word TF-IDF cosine.**

{md(nn)}

**Slot annotation on near-duplicates** (pairs compared on (type, value) sets):

{md(sa)}

**Audit of a fixed random sample of 60 Stage-1 intent errors** (one reader, labels in `q1b_error_audit_sample.csv`): gold doubtful or defensibly another label 31, nonsense with catch-all gold 2, clear model error 27. That puts 52-55% of errors, about 9% of dev (CI about 7-11%), out of reach from the text alone.

| Evidence | Coverage of dev | Implied intent ceiling on dev | Caveat |
|---|---:|---:|---|
| Exact duplicates, Cover-Hart bracket | 0.5% | 91.3-95.6 | 46 groups only, CI of disagreement 2.2-17.4% |
| Template majority on seen templates | 12.7% | 93.0 (CI 89.2-95.5) | uses gold slots; short forms; conflict inflated by abstraction |
| Template bracket keeping `device_type` | 12.1% | 95.1-97.5 from 4.92% disagreement | short forms only |
| Stage 1, A1 and linear all wrong | 100% | 90.8 (187 utterances) | not a bound; shared hard core |
| Error audit, 60 sampled errors | 100% | about 91 (CI about 89-93) | single reader, small sample |

**Estimate: about 90-94% is a defensible upper range, about 91-92% the central guess. Evidence strength: weak to moderate.** 87% of dev has an unseen template, so nothing measures label noise there directly. Near-duplicates show low noise (2-5% disagreement) but are short, easy forms; the fuzzy classes live in the low-similarity region. Re-annotating a sample would be stronger evidence.

## 2. Intent confusions (349 errors)

{md(tp20, ["pair", "n", "share_pct", "cum_pct", "a_to_b", "b_to_a", "same_scenario", "tmpl_shared_train", "a1_also_wrong", "no_oov", "nonslot_oov", "slot_oov"])}

`a_to_b` counts gold a predicted as b. Top 20 pairs hold 36.7% of errors; 167 distinct pairs exist, 93 of them once. 89 errors (25.5%) stay inside a scenario. Scenario accuracy is 87.21. A1 repeats 268 of the 349 errors (76.8%), 183 with the same wrong label.

Examples (seeded random pick; all 55 errors of these pairs are in `q2_top5_pair_errors.csv`):

{md(EX)}

Audit per pair (L doubtful, U catch-all nonsense, M model error):

{md(aud)}

Top-5 pairs: 38 of 55 doubtful (69%). Random sample of all errors: 31-33 of 60. Most top-pair errors are plausibly label ambiguity, about half of all errors are, and the rest are real model errors such as `switch on the lights` -> `iot_hue_lightup`.

{md(ig)}

Lifting both OOV groups to the No-OOV accuracy would add at most 3.27 points; 2.67 after standardising the intent mix. The NonSlot group is heavy in `general_quirky` (34) and `qa_factoid` (22); even the char-gram linear model only reaches 57.05 there.

## 3. Slot error taxonomy

Gold view (every FN assigned once; the same span may have a counterpart in the prediction view):

{md(tx, tcols, trn)}

Prediction view (every FP assigned once):

{md(tx, pcols, prn)}

What-if overall F1 if a category disappeared (base 66.09):

{md(wi)}

Lifting the Slot-Train-OOV group to F1 71.51 / 77.78 gives overall 71.04 / 72.01 (joint +2.48 / +2.96). That group holds 16.0% of gold spans.

{md(os_)}

Boundary errors: 200 of 232 have the prediction inside the gold span (`new york city` -> `new york`, `hurts like heaven` -> `heaven`). Only 30 of 232 differ by function words or dots (F1 67.57 if repaired). Intent right: slot F1 68.46; intent wrong: 51.09.

Illegal BIO in predictions and the effect of dropping stray I- runs at decode time:

{md(ba, ["model", "F1", "tp", "fp", "fn", "g_boundary", "p_boundary", "g_missed", "p_spurious", "I_tags", "stray_after_O", "stray_after_other_type", "F1_drop_stray", "gain_drop_stray"])}

Top 15 types by support (75.6% of gold spans):

{md(pt.head(15), tc)}

Worst 10 with support at least 10:

{md(wt, tc)}

## 4. Linear baseline

Word 1-2 grams plus char_wb 3-5 grams, TF-IDF fit on train, multinomial logistic regression (saga; L-BFGS history would break the memory limit), {59947:,} features.

{md(cg)}

C=100 stopped at max_iter and sits at the grid edge, so a sensitivity run followed; the selected model stays the grid winner.

{md(sn, ["C", "max_iter", "iters", "dev_acc", "fixes_S1", "breaks_S1"])}

{md(cp)}

{md(lg)}

Override with the linear label above a confidence threshold, threshold cross-validated on dev (no deep probabilities were available here):

{md(ge)}

{md(ff, ["features", "n_features", "wide_params", "dev_acc", "No-Trai", "NonSlot", "Slot-Tr", "fixes_S1", "breaks_S1"])}

## 5. Word2Vec coverage

464 dev Train-OOV lexical tokens (3.32% of lexical tokens), 439 types; 254 inside gold slots, 210 outside. Filtered key list: 256,856 words.

{md(cv)}

Largest legal K, base 4,344,472: ignoring the map 47,454 at 300d and 151,824 at 100d. With the `frozen_map` the code builds (90,300 or 30,300 parameters): 47,153 and 151,521.

{md(lk, ["adapters", "adapter_params", "max_K_300d_with_map", "max_K_100d_with_map", "ext46k_300d_total", "ext150k_100d_total", "ext46k_legal", "ext150k_100d_legal"])}

Uncovered at K=46k: 109 tokens (23.5%), 106 types, 77 in slots, by reading all of them:

| Kind | Tokens | Share | Examples |
|---|---:|---:|---|
| Ordinary words rare in Wikipedia or missing | 36 | 33.0% | bedtime, crispy, toggle, thesaurus, marinara, both |
| Names and brands | 34 | 31.2% | cheteshwar, zaalima, beyonce, iheartradio, tinder |
| Possessive forms (18 have a known base word) | 20 | 18.3% | romero's, thursday's, queen's |
| Typos and concatenations | 19 | 17.4% | washingdon, fransisco, metion, tiangle, harrypotter |
| Numbers | 0 | 0% | the data has no digit at all |

50 of the 109 tokens are in the key list beyond rank 46k; 59 are absent from it. Names at the boundary are judged by reading, so treat the split as approximate. Per-type list: `q5_uncovered_at_46k_types.csv`.

A parameter-free lookup fallback to train words needs no Word2Vec row and no flag; edit distance 1 is too noisy to use (`cover` -> `over`, `poland` -> `boland`):

{md(sb)}

## 6. Intent to slot compatibility

432 of 3,300 intent-type cells occur in train (13.1%); median 8 types per intent; 126 cells seen once.

{md(gb)}

Stage-1 FP spans 691: 33 (4.8%) have a type never seen with the predicted intent, 70 (10.1%) never seen with the gold intent. TP spans blocked by a predicted-intent mask: 9 of 1,334.

Post-hoc span deletion (`k` = minimum train count for a cell to be allowed), a lower bound since real masked decoding could also re-label a span:

{md(ms)}

Reverse direction: a predicted slot incompatible with the predicted intent flags 41 utterances (intent wrong in 43.9% vs 16.6%), only 5.2% of intent errors.

## 7. Other dev facts

- Casing and characters: no uppercase, no digit, no non-ASCII in train or dev. Numbers are spelled out. Casefolding is a no-op.
- Punctuation: `.` is 208 of 212 non-lexical dev tokens and 199 of the 212 sit inside gold slots, almost all abbreviations (`p. m.`, `u. s. a.`). 44 dev utterances end in `.`. "am" is written `am` (train 303) but "pm" is written `p. m.` (train 266, `pm` 0). Spans with a dot: boundary errors 29.2% vs 10.7%.
- Same value inside and outside slots: closed-class values are consistent (today 524 slot / 21 plain, tomorrow 220 / 2, this week 93 / 0, five am 14 / 0, eight am 35 / 0). Mixed cases: `meeting` 279 / 26 (dev 48 / 8, and Stage 1 tags 7 of those as FP), `to do` 32 / 38, `home` 26 / 31. Type conflicts: weekend time 8, date 6; trump person 13, news_topic 8; us place_name 7, currency_name 6.
- 41.7% of spurious spans (80 of 192) have a (value, type) that is a gold slot in train; 67 of the 192 fall in utterances with no gold slot (31.8% of dev has none).

Boundary convention, train:

{md(bc.head(8))}

Dev gold spans by value history in train:

{md(hv)}

By length:

{md(lt)}

Rare intents:

{md(ri)}

Intents with fewer than 100 train rows cover 155 dev utterances at 84.52%, so they are not the weak spot. `audio_volume_other` (18 train) has no dev row and `music_album`, `transport_descriptor` no dev span. Train has 107 BIO labels and dev adds none, so the 108th label of the model comes from test only (the inventory in `Assets` reads all splits); I did not look at which one.

Sampling noise on dev:

{md(ns)}

Paired differences between finished ladder runs (snapshot during this analysis):

{md(pn)}

Differences below about 0.8 joint are not distinguishable from utterance sampling alone, before seed noise.

## Recommendations

Favoured: constrained decoding or CRF first (zero-parameter stray-I repair already +2.05 / +2.10 slot F1 on Stage 1 / A1), then span-aware heads for multi-token names; slot-weighted loss; a wide linear path inside the joint model (0.6-1.2M parameters for 85.4-85.9 alone), regularised heavily because the linear part reaches 99.9% on train; BPE before a large frozen table; a suffix and possessive lookup fallback (16.8% of OOV tokens, no parameters).

Disfavoured: a hard intent-to-slot mask (+0.24 / +0.06 F1 with predicted intent); a separately trained linear ensemble (rule risk, hard gating gave no gain); K beyond 46k with PCA (+8.8 points of token coverage, 41 tokens); hierarchical intent as a priority (74.5% of errors cross scenarios and those scenarios are fuzzy); rare-intent rebalancing. The soft mask and top-k options of rung E are cheap to keep but this evidence expects little from them. Deep runs disagree with each other on 6-8% of intents, A1 and linear on 12.5%, so deep-only ensembles add less intent diversity; whether any ensemble counts as one joint model needs confirming.

Process: treat dev joint differences below about 0.8 as ties, since `greedy` in `run_ladder.py` accepts any positive difference on one seed. Rung C compares the CRF, which has hard constraints, with an unconstrained argmax on the linear head, so about +1 joint of any CRF win may be the constraints alone; rung E gives that to linear heads only later. By the table in section 5, `B3_*_subword` exceeds the 20M assertion if the strict best uses the n-gram adapter, or BPE+char with `ext150k_pca100`; only `ResourceAbort` is caught, so rung B would stop there. This comes from parameter arithmetic, not from running it.
"""
(OUT / "report.md").write_text(doc, encoding="utf-8")
print("report chars:", len(doc), "| lines:", doc.count("\n"))
show(EX)
show(aud)
mem("end")

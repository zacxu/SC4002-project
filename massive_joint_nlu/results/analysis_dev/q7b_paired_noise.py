"""Q7 supplement: how large a dev joint difference between two ladder runs is beyond utterance-sampling noise.
Reads only finished runs' prediction frames; nothing in the ladder folders is changed."""
from common import *

train, m, intents = load_all()
ids = m["id"].to_numpy()
runs = {}
for p in sorted((ROOT / "results").glob("*/dev_pred_frame.pkl")):
    if p.parent.name == "analysis_dev":
        continue
    if not (p.parent / "dev_metrics.json").exists():
        continue
    fr = pd.read_pickle(p).set_index("id").loc[ids]
    per = np.array([slot_counts([g], [q]) for g, q in zip(fr["gold_tags"], fr["pred_tags"])])
    ok = (fr["gold_intent_id"].to_numpy() == fr["pred_intent_id"].to_numpy()).astype(float)
    runs[p.parent.name] = (per, ok, fr["pred_intent_id"].to_numpy())
    del fr
print("finished runs:", list(runs))


def joint(per, ok, idx=None):
    c = per if idx is None else per[idx]
    o = ok if idx is None else ok[idx]
    tp, fp, fn = c.sum(0)
    return (100 * o.mean() + 100 * 2 * tp / max(2 * tp + fp + fn, 1)) / 2


base_name = max((k for k in runs if k != "stage1_baseline"), key=lambda k: joint(*runs[k][:2]))
print("reference run (highest joint so far):", base_name, round(joint(*runs[base_name][:2]), 2))
rng = np.random.default_rng(42)
n = len(ids)
idxs = [rng.integers(0, n, n) for _ in range(1000)]
rows = []
for k, (per, ok, pred) in runs.items():
    tp, fp, fn = per.sum(0)
    d = [joint(per, ok, i) - joint(runs[base_name][0], runs[base_name][1], i) for i in idxs]
    rows.append({"run": k, "intent": round(100 * ok.mean(), 2), "slot_f1": round(f1_from(tp, fp, fn), 2), "joint": round(joint(per, ok), 2),
                 "diff_vs_ref": round(joint(per, ok) - joint(runs[base_name][0], runs[base_name][1]), 2), "paired_se": round(float(np.std(d)), 2),
                 "ci_low": round(float(np.percentile(d, 2.5)), 2), "ci_high": round(float(np.percentile(d, 97.5)), 2),
                 "intent_disagreement_pct": round(100 * (pred != runs[base_name][2]).mean(), 1)})
T = pd.DataFrame(rows).sort_values("joint", ascending=False)
show(T)
save(T, "q7b_paired_noise.csv")
mem("end")

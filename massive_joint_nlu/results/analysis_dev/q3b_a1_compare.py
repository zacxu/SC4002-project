"""Q3 supplement: the same taxonomy and illegal-BIO repairs for the current ladder base A1_ft_recipe, next to Stage 1."""
from common import *

train, m, intents = load_all()
_, a1, _ = load_all(A1_FRAME)


def overlap(a, b):
    return a[1] < b[2] and b[1] < a[2]


def tax(frame):
    g_c, p_c = Counter(), Counter()
    for gold, pred in zip(frame["gold_tags"], frame["pred_tags"]):
        gs, ps = set(bio_to_spans(gold)), set(bio_to_spans(pred))
        ex = gs & ps
        gl, pl = gs - ex, ps - ex
        g_c["exact"] += len(ex)
        p_c["exact"] += len(ex)
        for p in pl:
            ov = [g for g in gl if overlap(p, g)]
            if any(g[1] == p[1] and g[2] == p[2] for g in ov):
                p_c["wrong_type"] += 1
            elif any(g[0] == p[0] for g in ov):
                p_c["boundary"] += 1
            elif ov:
                p_c["overlap_other"] += 1
            else:
                p_c["spurious"] += 1
        for g in gl:
            ov = [p for p in pl if overlap(p, g)]
            if any(p[1] == g[1] and p[2] == g[2] for p in ov):
                g_c["wrong_type"] += 1
            elif any(p[0] == g[0] for p in ov):
                g_c["boundary"] += 1
            elif ov:
                g_c["overlap_other"] += 1
            else:
                g_c["missed"] += 1
    return g_c, p_c


def repairs(frame):
    stray_o = stray_t = n_i = 0
    drop = []
    for tags in frame["pred_tags"]:
        prev, open_type, out = "O", None, []
        for t in tags:
            if t.startswith("I-"):
                n_i += 1
                if prev == "O":
                    stray_o += 1
                elif prev[2:] != t[2:]:
                    stray_t += 1
            prev = t
            if t == "O":
                out.append("O"); open_type = None
            elif t.startswith("B-"):
                out.append(t); open_type = t[2:]
            elif open_type == t[2:]:
                out.append(t)
            else:
                out.append("O"); open_type = None
        drop.append(out)
    c = slot_counts(frame["gold_tags"], drop)
    return n_i, stray_o, stray_t, c, f1_from(*c)


rows = []
for name, fr in (("Stage 1", m), ("A1_ft_recipe", a1)):
    g_c, p_c = tax(fr)
    c = slot_counts(fr["gold_tags"], fr["pred_tags"])
    n_i, so, st, c2, f2 = repairs(fr)
    rows.append({"model": name, "F1": round(f1_from(*c), 2), "tp": c[0], "fp": c[1], "fn": c[2],
                 "g_wrong_type": g_c["wrong_type"], "g_boundary": g_c["boundary"], "g_overlap": g_c["overlap_other"], "g_missed": g_c["missed"],
                 "p_wrong_type": p_c["wrong_type"], "p_boundary": p_c["boundary"], "p_overlap": p_c["overlap_other"], "p_spurious": p_c["spurious"],
                 "I_tags": n_i, "stray_after_O": so, "stray_after_other_type": st, "F1_drop_stray": round(f2, 2), "gain_drop_stray": round(f2 - f1_from(*c), 2)})
R = pd.DataFrame(rows)
show(R)
save(R, "q3b_stage1_vs_A1.csv")
c = slot_counts(a1["gold_tags"], a1["pred_tags"])
g_c, p_c = tax(a1)
print("A1 what-if F1:", {"boundary": round(f1_from(c[0] + g_c["boundary"], c[1] - p_c["boundary"], c[2] - g_c["boundary"]), 2),
                         "missed": round(f1_from(c[0] + g_c["missed"], c[1], c[2] - g_c["missed"]), 2),
                         "spurious": round(f1_from(c[0], c[1] - p_c["spurious"], c[2]), 2),
                         "wrong_type": round(f1_from(c[0] + g_c["wrong_type"], c[1] - p_c["wrong_type"], c[2] - g_c["wrong_type"]), 2)})
mem("end")

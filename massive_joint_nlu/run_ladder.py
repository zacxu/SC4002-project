"""Stage-2 method ladder. Usage: python run_ladder.py A|V|B|C|D|R|H|F|E

Protocol for every rung (dev-only selection, test is never touched here):
  screen   each variant once with seed 42 at 40 epochs, on top of the current best configuration
  promote  the most promising variants get two more seeds; they are judged on the 3-seed mean, because
           differences below about 0.8 joint on one seed are ties (paired dev SE 0.35-0.44, plus seed noise)
  compose  helpful variants are added one by one, and each is kept only if the 3-seed mean improves

Two tracks are reported side by side:
  strict   nothing beyond the assignment's plain reading
  flagged  extended Word2Vec vocabulary or token recombination: needs the lecturer's OK
Finished runs are never repeated or overwritten, so a crashed rung can simply be started again.
"""
import json
import random as pyrandom
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

from src.config import ModelConfig, TrainConfig, PARAM_CAP, RESULTS_DIR
from src.experiment import Context, run, rebuild_leaderboard, write_json_atomic
from src.guard import ResourceAbort
from src.models import JointModel, count_params

RECIPE = dict(epochs=40, patience=8, schedule="cosine", warmup_epochs=1.0, label_smoothing=0.1, word_dropout=0.1)
EXTRA_SEEDS = (7, 1234)
ALL_PRIOR = "ABCDRHF"


@dataclass
class Cand:
    name: str
    m: ModelConfig
    t: TrainConfig
    init: Optional[str] = None
    k: Optional[int] = None
    dim: int = 300


def derive(c, name, m=None, t=None, init="keep", k="keep", dim=None):
    return Cand(name, replace(c.m, **(m or {})), replace(c.t, **(t or {})),
                c.init if init == "keep" else init, c.k if k == "keep" else k, dim or c.dim)


def is_flagged(c):
    return c.k is not None or c.t.mention_replace > 0 or c.t.lwtr > 0 or c.t.sis > 0


def execute(ctx, c, rung, note=""):
    return run(ctx, c.name, c.m, c.t, init=c.init, k_extra=c.k, frozen_dim=c.dim, rung=rung, note=note, flagged=is_flagged(c))


def row_to_cand(r):
    k = int(r["k_extra"]) if pd.notna(r["k_extra"]) and r["k_extra"] > 0 else None
    return Cand(r["run"], ModelConfig.from_dict(json.loads(r["model_config"])),
                TrainConfig.from_dict(json.loads(r["train_config"])),
                None if r["init"] == "fresh" else r["init"], k,
                int(r["frozen_dim"]) if "frozen_dim" in r and pd.notna(r["frozen_dim"]) else 300)


def conflict(ctx, c):
    """Why a configuration cannot run, or None. Mirrors the asserts in the model and the training loop."""
    m, t = c.m, c.t
    if m.slot_head == "biaffine" and (m.tag_scheme != "bio" or m.factorized or m.slot_to_intent or m.intent_to_slot
                                      or m.refine or m.coint or m.wide_slot_bits):
        return "the biaffine head takes none of BIOES, factorized, i2s, s2i, refine, coint or the sparse tag path"
    if m.encoder == "bilstm" and (m.pre_lstm or m.rel_bias or m.task_top or m.refine):
        return "Transformer-only switch on a BiLSTM encoder"
    if t.adv_eps > 0 and t.sam_rho > 0:
        return "adversarial noise and SAM cannot be combined"
    if t.o_weight != 1.0 and m.slot_head != "linear":
        return "o_weight only acts on the linear head"
    probe = m
    if c.k is not None:
        probe = replace(m, n_frozen=ctx.ext(c.k, c.dim).n_frozen, frozen_dim=c.dim)
    total = count_params(JointModel(probe, ctx.assets))[0]
    if total > PARAM_CAP:
        return f"{total:,} parameters exceed the 20M cap"
    return None


def family_table(rungs, strict=True, only_flagged=False, max_epochs=None):
    lb = rebuild_leaderboard()
    lb = lb[lb["rung"].astype(str).str[0].isin(list(rungs))]
    flagged = lb["flagged"].astype(bool)
    if strict:
        lb = lb[~flagged]
    elif only_flagged:
        lb = lb[flagged]
    if max_epochs is not None:
        lb = lb[lb["train_config"].map(lambda s: json.loads(s)["epochs"]) <= max_epochs]
    return (lb.groupby("family")
            .agg(joint=("dev_joint", "mean"), std=("dev_joint", "std"), n=("dev_joint", "size"),
                 intent=("dev_intent", "mean"), slot=("dev_slot_f1", "mean"), slot_oov=("f1_slot_oov", "mean"),
                 params=("params_total", "first"), minutes=("train_time_s", lambda s: s.mean() / 60))
            .sort_values("joint", ascending=False))


def best(rungs, strict=True, only_flagged=False, min_seeds=3, max_epochs=None):
    t = family_table(rungs, strict, only_flagged, max_epochs)
    t = t[t["n"] >= min_seeds]
    fam = t.index[0]
    lb = rebuild_leaderboard()
    r = lb[(lb["family"] == fam) & (lb["seed"] == 42)].iloc[0]
    print(f"BASE <- {fam}: joint {t.loc[fam, 'joint']:.2f} over {int(t.loc[fam, 'n'])} seed(s), "
          f"intent {t.loc[fam, 'intent']:.2f}, slot {t.loc[fam, 'slot']:.2f}", flush=True)
    return row_to_cand(r), float(t.loc[fam, "joint"])


def three_seeds(ctx, c, rung, note=""):
    rows = [execute(ctx, c, rung, note=note)]
    for s in EXTRA_SEEDS:
        rows.append(execute(ctx, derive(c, f"{c.name}__s{s}", t={"seed": s}), rung, note=f"{c.name} seed {s}"))
    return sum(r["dev_joint"] for r in rows) / len(rows)


def screen(ctx, rung, base, variants):
    """Runs each legal variant once. Returns [(key, overrides, joint)], best first."""
    out = []
    for key, ov in variants.items():
        c = derive(base, f"{rung}_{key}", **ov)
        why = conflict(ctx, c)
        if why:
            print(f"screen: skipped {key}: {why}", flush=True)
            continue
        out.append((key, ov, execute(ctx, c, rung, note=f"{base.name} + {key}")["dev_joint"]))
    out.sort(key=lambda x: -x[2])
    print(pd.DataFrame(out, columns=["variant", "ov", "joint"]).drop(columns="ov").to_string(index=False), flush=True)
    return out


def fields_of(ov):
    f = set((ov.get("m") or {}).keys()) | set((ov.get("t") or {}).keys())
    if "o_weight" in f:
        f.add("slot_head")      # only the linear head reads it
    if f & {"adv_eps", "sam_rho"}:
        f |= {"adv_eps", "sam_rho"}
    return f


def protocol(ctx, rung, base, variants, top=4, min_gain=0.15, limit=3):
    """screen, promote, compose. Returns the winning candidate and its 3-seed mean."""
    lb = rebuild_leaderboard()
    rows = lb[lb["family"] == base.name]
    base_42 = float(rows[rows["seed"] == 42].iloc[0]["dev_joint"])
    base_mean = float(rows["dev_joint"].mean())
    results = screen(ctx, rung, base, variants)
    promoted = []
    for key, ov, joint in results:
        if joint < base_42 + min_gain or len(promoted) >= top:
            continue
        c = derive(base, f"{rung}_{key}", **ov)
        promoted.append((key, ov, three_seeds(ctx, c, rung, note=f"{base.name} + {key}")))
    promoted.sort(key=lambda x: -x[2])
    print("promoted, 3-seed means, base", round(base_mean, 2), [(k, round(j, 2)) for k, _, j in promoted], flush=True)

    cur, cur_mean, used, step = base, base_mean, set(), 0
    for key, ov, mean in promoted:
        if mean <= base_mean or step >= limit:
            continue
        fields = fields_of(ov)
        if fields & used:
            print(f"compose: skipped {key}: it changes a field already decided", flush=True)
            continue
        if step == 0:           # already evaluated on three seeds
            cur, cur_mean, used = derive(base, f"{rung}_{key}", **ov), mean, used | fields
            step = 1
            print(f"compose: kept {key} ({base_mean:.2f} -> {mean:.2f})", flush=True)
            continue
        c = derive(cur, f"{rung}g{step + 1}_{key}", **ov)
        why = conflict(ctx, c)
        if why:
            print(f"compose: skipped {key}: {why}", flush=True)
            continue
        step += 1
        m = three_seeds(ctx, c, rung, note=f"{cur.name} + {key}")
        if m > cur_mean:
            print(f"compose: kept {key} ({cur_mean:.2f} -> {m:.2f})", flush=True)
            cur, cur_mean, used = c, m, used | fields
        else:
            print(f"compose: dropped {key} ({m:.2f} <= {cur_mean:.2f})", flush=True)
    print(f"RUNG {rung} WINNER {cur.name}: 3-seed joint {cur_mean:.2f} (base {base.name} {base_mean:.2f})", flush=True)
    return cur, cur_mean


def carry_to_flagged(ctx, rung, prev_strict, new_strict, prior_rungs):
    """Applies this rung's strict gains to the best flagged configuration, on three seeds."""
    t = family_table(prior_rungs, strict=False, only_flagged=True, max_epochs=40)
    t = t[t["n"] >= 3]
    if t.empty or new_strict.name == prev_strict.name:
        return
    m = {k: v for k, v in new_strict.m.__dict__.items()
         if prev_strict.m.__dict__[k] != v and k not in ("n_frozen", "frozen_dim")}
    tr = {k: v for k, v in new_strict.t.__dict__.items() if prev_strict.t.__dict__[k] != v and k != "seed"}
    if not m and not tr:
        return
    base, _ = best(prior_rungs, strict=False, only_flagged=True, max_epochs=40)
    c = derive(base, f"{rung}f_flagged", m=m, t=tr)
    why = conflict(ctx, c)
    if why:
        print(f"flagged carry skipped: {why}", flush=True)
        return
    three_seeds(ctx, c, rung, note=f"{base.name} + strict gains of rung {rung}")


# ------------------------------------------------------------------ rungs
def rung_A(ctx):
    s1 = ModelConfig()
    rows = [execute(ctx, c, "A", note="recipe: 40 epochs, cosine, warm-up, label smoothing .1, word dropout .1")
            for c in (Cand("A1_ft_recipe", s1, TrainConfig(lr=1e-4, **RECIPE), "stage1"),
                      Cand("A1_fresh_recipe", s1, TrainConfig(lr=3e-4, **RECIPE)),
                      Cand("A1_fresh_lr5e-4", s1, TrainConfig(lr=5e-4, **RECIPE)))]
    top = max(rows, key=lambda r: r["dev_joint"])    # seed-42 A1 rows only: the same base after any restart
    base, score = row_to_cand(top), float(top["dev_joint"])
    print("A1 winner", base.name, score, flush=True)
    variants = {
        "slot2": {"t": {"lambda_slot": 2.0}},
        "drop.10": {"m": {"dropout": 0.10}},
        "drop.25": {"m": {"dropout": 0.25}},
        "wd.01": {"t": {"weight_decay": 0.01}},
        "ls0": {"t": {"label_smoothing": 0.0}},
        "ls.2": {"t": {"label_smoothing": 0.2}},
        "wdrop.2": {"t": {"word_dropout": 0.2}},
        "slotdrop.2": {"t": {"slot_dropout": 0.2}},
        "rare1": {"t": {"rare_alpha": 1.0}},
        "rare.25": {"t": {"rare_alpha": 0.25}},
        "bs32": {"t": {"batch_size": 32}},
        "ep80": {"t": {"epochs": 80, "patience": 12}},
        "emb.1": {"t": {"emb_lr_mult": 0.1}},
        "emb0": {"t": {"emb_lr_mult": 0.0}},
    }
    results = screen(ctx, "A2", base, variants)
    # greedy on one seed, as first run; rung A set the working recipe before the stricter protocol existed
    cur, cur_score, used, step = base, score, set(), 0
    for key, ov, joint in results:
        if joint <= score or step >= 4:
            continue
        fields = fields_of(ov)
        if fields & used:
            continue
        if step == 0:
            cur, cur_score, used, step = derive(base, f"A2_{key}", **ov), joint, used | fields, 1
            continue
        step += 1
        c = derive(cur, f"A3g{step}_{key}", **ov)
        r = execute(ctx, c, "A3", note=f"{cur.name} + {key}")
        if r["dev_joint"] > cur_score:
            cur, cur_score, used = c, r["dev_joint"], used | fields
    lb = rebuild_leaderboard()
    a = lb[(lb["rung"].astype(str).str[0] == "A") & (lb["seed"] == 42)].sort_values("dev_joint", ascending=False)
    for name in a["run"].tolist()[:3]:
        three_seeds(ctx, row_to_cand(a[a["run"] == name].iloc[0]), "A", note=f"{name}")
    print(family_table("A").round(2).to_string(), flush=True)


def rung_V(ctx):
    """Hard-constrained Viterbi on the linear head becomes the standard. Training is unchanged, except that the
    best epoch is now chosen under the decoding that is actually used."""
    base, _ = best("A", max_epochs=40)
    three_seeds(ctx, derive(base, "A4_vit", m={"decode": "viterbi"}), "A4", note=f"{base.name} + Viterbi decoding")
    print(family_table("A").head(6).round(2).to_string(), flush=True)


def rung_B(ctx):
    base, _ = best("A", max_epochs=40)
    strict, _ = protocol(ctx, "B1", base, {
        "bpe": {"m": {"adapters": ("bpe",)}},
        "ngram": {"m": {"adapters": ("ngram",)}},
        "char": {"m": {"adapters": ("char",)}},
        "bpe_char": {"m": {"adapters": ("bpe", "char")}},
        "bpe_ngram": {"m": {"adapters": ("bpe", "ngram")}},
        "all3": {"m": {"adapters": ("bpe", "ngram", "char")}},
        "wide10k": {"m": {"wide_intent": 10000}},
        "wide20k": {"m": {"wide_intent": 20000}},
        "wslot14": {"m": {"wide_slot_bits": 14}},
        "wslot15": {"m": {"wide_slot_bits": 15}},
        "gaz": {"m": {"gaz": True}},
    })
    print(family_table("AB").head(8).round(2).to_string(), flush=True)

    # Flagged track: frozen Word2Vec rows from the same NLPL model 6, chosen by file order only.
    strict_best, _ = best("AB", max_epochs=40)
    sim = {"swap_frac": 0.7, "swap_prob": 0.1}
    res = screen(ctx, "B2", strict_best, {
        "ext10k": dict(k=10000, t=sim), "ext20k": dict(k=20000, t=sim), "ext40k": dict(k=40000, t=sim),
        "ext20k_noswap": dict(k=20000),
        "ext100k_pca100": dict(k=100000, dim=100, t=sim),
    })
    if res:
        key, ov, _ = res[0]
        three_seeds(ctx, derive(strict_best, f"B2_{key}", **ov), "B2", note=f"{strict_best.name} + {key}")
    print(family_table("AB", strict=False, only_flagged=True).head(6).round(2).to_string(), flush=True)


def rung_C(ctx):
    base, _ = best("AB", max_epochs=40)
    strict, _ = protocol(ctx, "C1", base, {
        "crf": {"m": {"slot_head": "crf"}},
        "bioes_crf": {"m": {"slot_head": "crf", "tag_scheme": "bioes"}},
        "biaffine": {"m": {"slot_head": "biaffine"}},
        "factorized": {"m": {"factorized": True}},
        "interact": {"m": {"interaction": True}},
        "i2s": {"m": {"intent_to_slot": True}},
        "s2i": {"m": {"slot_to_intent": True}},
        "attn": {"m": {"pooling": "attn"}},
        "meanmax": {"m": {"pooling": "meanmax"}},
        "labelattn": {"m": {"label_attn": True}},
        "hier": {"m": {"hier_intent": True}},
        "tokintent": {"t": {"lambda_token_intent": 0.2}},
        "auxlm": {"m": {"aux_lm": True}, "t": {"lambda_aux": 0.2}},
        "prelstm": {"m": {"pre_lstm": True}},
        "relbias": {"m": {"rel_bias": True}},
        "tasktop": {"m": {"task_top": True}},
        "refine": {"m": {"refine": True}},
        "coint": {"m": {"coint": True}},
        "presence": {"m": {"presence": True}},
        "oweight.5": {"t": {"o_weight": 0.5}},
    })
    print(family_table("ABC").head(8).round(2).to_string(), flush=True)
    carry_to_flagged(ctx, "C2", base, strict, "ABC")


def rung_D(ctx):
    base, _ = best("ABC", max_epochs=40)
    fresh = {"init": None}
    lstm_off = {"pre_lstm": False, "rel_bias": False, "task_top": False, "refine": False}
    t3 = {"lr": max(base.t.lr, 3e-4)}
    strict, _ = protocol(ctx, "D1", base, {
        "bilstm256": dict(fresh, m={"encoder": "bilstm", "lstm_hidden": 256, **lstm_off}, t={"lr": 1e-3}),
        "bilstm256_lr3": dict(fresh, m={"encoder": "bilstm", "lstm_hidden": 256, **lstm_off}, t=t3),
        "bilstm384": dict(fresh, m={"encoder": "bilstm", "lstm_hidden": 384, **lstm_off}, t={"lr": 1e-3}),
        "tfm128x2": dict(fresh, m={"d_model": 128, "n_layers": 2, "d_ff": 384}, t=t3),
        "tfm256x2": dict(fresh, m={"n_layers": 2}, t=t3),
        "tfm256x6": dict(fresh, m={"n_layers": 6}, t=t3),
        "tfm384x6": dict(fresh, m={"d_model": 384, "n_heads": 6, "n_layers": 6, "d_ff": 1536}, t=t3),
    }, limit=1)
    print(family_table("ABCD").head(8).round(2).to_string(), flush=True)
    carry_to_flagged(ctx, "D2", base, strict, "ABCD")


def rung_R(ctx):
    base, _ = best("ABCD", max_epochs=40)
    strict, _ = protocol(ctx, "R1", base, {
        "adv.05": {"t": {"adv_eps": 0.05}},
        "adv.2": {"t": {"adv_eps": 0.2}},
        "rdrop1": {"t": {"rdrop_alpha": 1.0}},
        "rdrop.25": {"t": {"rdrop_alpha": 0.25}},
        "sam.05": {"t": {"sam_rho": 0.05}},
        "sam.2": {"t": {"sam_rho": 0.2}},
        "ema": {"t": {"ema_decay": 0.999}},
        "mdrop.1": {"t": {"mention_dropout": 0.1}},
        "mdrop.3": {"t": {"mention_dropout": 0.3}},
        "cons.25": {"t": {"cons_alpha": 0.25}},
        "cons1": {"t": {"cons_alpha": 1.0}},
        "mixup.2": {"t": {"mixup_alpha": 0.2}},
        "supcon.1": {"m": {"supcon_dim": 128}, "t": {"supcon_lambda": 0.1}},
        "anchor1e-3": {"m": {"anchor": True}, "t": {"anchor_lambda": 1e-3}},
        "anchor1e-2": {"m": {"anchor": True}, "t": {"anchor_lambda": 1e-2}},
        "dualemb": {"m": {"dual_emb": True}},
    })
    print(family_table("ABCDR").head(8).round(2).to_string(), flush=True)
    carry_to_flagged(ctx, "R2", base, strict, "ABCDR")
    # Flagged: new token sequences recombined from train only.
    strict_best, _ = best("ABCDR", max_epochs=40)
    aug = screen(ctx, "R3", strict_best, {"mention.3": {"t": {"mention_replace": 0.3}},
                                          "mention.6": {"t": {"mention_replace": 0.6}},
                                          "lwtr.3": {"t": {"lwtr": 0.3}},
                                          "sis.3": {"t": {"sis": 0.3}}})
    if aug:
        key, ov, _ = aug[0]
        three_seeds(ctx, derive(strict_best, f"R3_{key}", **ov), "R3", note=f"{strict_best.name} + {key}")


def rung_H(ctx, n=12):
    base, _ = best("ABCDR", max_epochs=40)
    rng = pyrandom.Random(2026)
    grid = {"dropout": (0.1, 0.15, 0.2, 0.25), "lr": (3e-4, 5e-4, 7e-4, 1e-3), "weight_decay": (1e-4, 1e-3, 1e-2),
            "label_smoothing": (0.05, 0.1, 0.15, 0.2), "word_dropout": (0.05, 0.1, 0.15, 0.2),
            "rare_alpha": (0.25, 0.5, 1.0, 2.0), "slot_dropout": (0.0, 0.1, 0.2), "lambda_slot": (1.0, 1.5, 2.0, 3.0)}
    variants = {}
    for i in range(n):
        s = {k: rng.choice(v) for k, v in grid.items()}
        variants[f"{i:02d}"] = {"m": {"dropout": s.pop("dropout")}, "t": s}
    protocol(ctx, "H", base, variants, top=2, limit=1)
    print(family_table("ABCDRH").head(8).round(2).to_string(), flush=True)


def rung_F(ctx):
    """The winners get the longer schedule, on three seeds."""
    long = {"epochs": 80, "patience": 12}
    base, mean = best("ABCDRH", max_epochs=40)
    m80 = three_seeds(ctx, derive(base, "F_strict_ep80", t=long), "F", note=f"{base.name} at 80 epochs")
    print(f"strict: 40 epochs {mean:.2f}, 80 epochs {m80:.2f}", flush=True)
    t = family_table("ABCDRH", strict=False, only_flagged=True, max_epochs=40)
    if not t[t["n"] >= 3].empty:
        fb, fmean = best("ABCDRH", strict=False, only_flagged=True, max_epochs=40)
        f80 = three_seeds(ctx, derive(fb, "F_flagged_ep80", t=long), "F", note=f"{fb.name} at 80 epochs")
        print(f"flagged: 40 epochs {fmean:.2f}, 80 epochs {f80:.2f}", flush=True)
    print(family_table(ALL_PRIOR).head(8).round(2).to_string(), flush=True)


def rung_E(ctx):
    """Dev-selected final systems from saved checkpoints: single model, decoding, MC dropout, ensemble."""
    from src.posthoc import (Decoder, System, Decode, gold_frame, greedy_ensemble, tuned_decode, score_system,
                             system_params, stored_values, TIE_MARGIN)
    from src.evaluate import oov_table
    lb = rebuild_leaderboard()
    gold = gold_frame(ctx, "dev")
    decoder = Decoder(ctx.assets)
    cache, systems, rows = {}, {}, []

    def save():
        """Written after each step, so a later failure cannot discard finished results."""
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "posthoc_leaderboard.csv", index=False)
        write_json_atomic(RESULTS_DIR / "final_systems.json", {k: v.to_dict() for k, v in systems.items()})

    def record(label, system):
        met, frame = score_system(ctx, system, "dev", gold, decoder, cache)
        table, _ = oov_table(frame, ctx.oov["dev"])
        g = table.set_index("group")["slot_micro_f1"]
        stored = stored_values(system.members, system.decode, decoder)
        assert stored <= PARAM_CAP, f"{label} stores {stored:,} values, over the cap"
        rows.append({"system": label, "members": len(system.members), "stored": stored,
                     "params": system_params(lb, system.members, system.decode, decoder),
                     "dev_intent": round(met["intent_accuracy"], 2), "dev_slot_f1": round(met["slot_micro_f1"], 2),
                     "dev_joint": round(met["joint_score"], 2), "f1_slot_oov": round(g["Slot-Train-OOV"], 2),
                     "decode": str(system.decode), "mc": system.mc, "flagged": system.flagged,
                     "run_list": "+".join(system.members)})
        print(rows[-1], flush=True)
        return met["joint_score"]

    for track, strict in (("strict", True), ("flagged", False)):
        pool = lb[lb["rung"].astype(str).str[0].isin(list(ALL_PRIOR))]
        flagged = pool["flagged"].astype(bool)
        if strict:
            pool = pool[~flagged]
        elif not flagged.any():
            continue
        fams = family_table(ALL_PRIOR, strict=strict, only_flagged=not strict)
        fams = fams[fams["n"] >= 3]
        if fams.empty:
            print(f"{track}: no family has three seeds, track skipped", flush=True)
            continue
        family = fams.index[0]
        seeds = pool[pool["family"] == family].sort_values("dev_joint", ascending=False)
        single = seeds.iloc[0]["run"]
        flag = bool(seeds.iloc[0]["flagged"])
        print(f"{track}: best family {family} over seeds {seeds['dev_joint'].tolist()}, single {single}", flush=True)
        score_system(ctx, System("t", [single], Decode()), "dev", gold, decoder, cache)
        dec, table = tuned_decode(ctx, [single], gold, decoder, cache)
        table.to_csv(RESULTS_DIR / f"decode_grid_{track}_single.csv", index=False)
        s = System(f"{track}_single", [single], dec, flag)
        chosen, score = s, record(s.name, s)
        for passes in (8, 32):
            m = System(f"{track}_single_mc{passes}", [single], dec, flag, mc=passes)
            joint = record(m.name, m)
            if joint > score + TIE_MARGIN:
                chosen, score = m, joint
        systems[chosen.name] = chosen
        save()
        # All seeds of the best family, the 3-seed winners of each rung and the best of each head and encoder.
        cands = pool.sort_values("dev_joint", ascending=False)["run"].tolist()[:24]
        for key in ("encoder", "slot_head", "tag_scheme"):
            vals = pool["model_config"].map(lambda c: json.loads(c).get(key))
            for v in vals.unique():
                cands.append(pool[vals == v].sort_values("dev_joint", ascending=False).iloc[0]["run"])
        cands = list(dict.fromkeys(cands))
        members, best_joint, trail = greedy_ensemble(ctx, cands, pool, gold, decoder, cache)
        (RESULTS_DIR / f"ensemble_trail_{track}.json").write_text(json.dumps(trail, indent=2))
        if len(members) > 1:
            dec2, table = tuned_decode(ctx, members, gold, decoder, cache)
            table.to_csv(RESULTS_DIR / f"decode_grid_{track}_ensemble.csv", index=False)
            e = System(f"{track}_ensemble", members, dec2,
                       flag or bool(pool[pool["run"].isin(members)]["flagged"].astype(bool).any()))
            total = stored_values(members, dec2, decoder)
            if total > PARAM_CAP:
                print(f"{track} ensemble rejected: {total:,} stored values exceed the cap", flush=True)
            elif record(e.name, e) > score + TIE_MARGIN:
                systems[e.name] = e
        save()
    print("FINAL SYSTEMS:", list(systems), flush=True)


RUNGS = {"A": rung_A, "V": rung_V, "B": rung_B, "C": rung_C, "D": rung_D, "R": rung_R, "H": rung_H, "F": rung_F, "E": rung_E}

if __name__ == "__main__":
    ctx = Context()
    try:
        RUNGS[sys.argv[1]](ctx)
    except ResourceAbort as e:
        print("STOPPED SAFELY:", e, flush=True)
        sys.exit(3)
    print("RUNG DONE", sys.argv[1], flush=True)

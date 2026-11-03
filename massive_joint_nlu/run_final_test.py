"""Final evaluation: reloads saved checkpoints, no training, and evaluates dev and test.

The systems come from results/final_systems.json, which rung E wrote using dev only. Before test is touched, every
system must fit the cap and reproduce the dev numbers that rung E recorded. Test is then evaluated once. A lock file
records when, and a repeat needs an explicit written reason:
    python run_final_test.py --rerun "reason"
--dev-only runs the dev checks and nothing else: no lock, no test data.
    python run_final_test.py --dev-only
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import torch

from src.config import RESULTS_DIR, PARAM_CAP
from src.data import make_collate
from src.experiment import Context, rebuild_leaderboard, STAGE1_NAME
from src.evaluate import predict, oov_table, paired_bootstrap, matched_control, error_cases, GROUPS, metrics_from_frame
from src.posthoc import System, Decode, Decoder, gold_frame, score_system, system_params, stored_values, load_member

OUT = RESULTS_DIR / "final"
LOCK = OUT / "TEST_EVALUATED.json"


def native(ctx, run_name, split, allow_test=False):
    """The model's own decoding, exactly as during training."""
    if split == "test" and not allow_test:
        raise RuntimeError("test is evaluated only after the lock is written")
    model, ck = load_member(ctx, run_name)
    ds = ctx.datasets(model.cfg, ck["meta"].get("k_extra"), splits=(split,))[split]
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, collate_fn=make_collate(ctx.assets))
    r = predict(model, loader, ctx.assets)
    del model
    torch.cuda.empty_cache()
    return r, ck


if __name__ == "__main__":
    dev_only = "--dev-only" in sys.argv
    OUT.mkdir(exist_ok=True)
    log = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "reason": "first evaluation"}
    if not dev_only and LOCK.exists():
        at = sys.argv.index("--rerun") + 1 if "--rerun" in sys.argv else None
        if at is None or at >= len(sys.argv):
            sys.exit(f"Test was already evaluated: {LOCK.read_text()}\nUse --rerun \"reason\" to repeat it.")
        log["reason"] = sys.argv[at]
        log["previous"] = json.loads(LOCK.read_text())

    ctx = Context(verbose=False)
    lb = rebuild_leaderboard()
    systems = {k: System.from_dict(v) for k, v in json.loads((RESULTS_DIR / "final_systems.json").read_text()).items()}
    decoder = Decoder(ctx.assets)
    golds = {s: gold_frame(ctx, s) for s in ("dev", "test")}
    logged = pd.read_csv(RESULTS_DIR / "posthoc_leaderboard.csv").set_index("system")
    cache, frames, rows, oov_rows = {}, {}, [], []

    def describe(label, split, frame, total, stored, trainable, epochs, members):
        table, merged = oov_table(frame, ctx.oov[split])
        frames[(label, split)] = merged
        t = table.copy()
        t.insert(0, "split", split)
        t.insert(0, "system", label)
        oov_rows.append(t)
        m = metrics_from_frame(frame)
        g = table.set_index("group")["slot_micro_f1"]
        rows.append({"system": label, "split": split, "total_params": total, "stored_values": stored,
                     "trainable_params": trainable, "intent_accuracy": round(m["intent_accuracy"], 2),
                     "slot_micro_f1": round(m["slot_micro_f1"], 2), "exact_match": round(m["exact_match"], 2),
                     "best_epoch_of_epochs_run": epochs, "slot_f1_no_oov": round(g["No-Train-OOV"], 2),
                     "slot_f1_slot_oov": round(g["Slot-Train-OOV"], 2), "members": members})
        print(f"{label:24s} {split:4s} intent {m['intent_accuracy']:.2f} slot {m['slot_micro_f1']:.2f} "
              f"exact {m['exact_match']:.2f} slot-OOV F1 {g['Slot-Train-OOV']:.2f}", flush=True)

    # Dev first. Stage 1 is the baseline; every system must fit the cap and reproduce the dev numbers rung E recorded.
    stage1_stored = stored_values([STAGE1_NAME], Decode(), decoder)
    r, ck = native(ctx, STAGE1_NAME, "dev")
    describe("Stage1", "dev", r["pred_frame"], ck["params_total"], stage1_stored, ck["params_trainable"],
             f"{ck['best_epoch']}/{ck['epochs_run']}", STAGE1_NAME)
    system_info = {}
    for label, system in systems.items():
        stored = stored_values(system.members, system.decode, decoder)
        assert stored <= PARAM_CAP, f"{label} stores {stored:,} values, over the cap"
        rows_m = lb[lb["run"].isin(system.members)]
        epochs = ", ".join(f"{int(lb[lb['run'] == m].iloc[0]['best_epoch'])}/{int(lb[lb['run'] == m].iloc[0]['epochs_run'])}"
                           for m in system.members)
        total = system_params(lb, system.members, system.decode, decoder)
        trainable = int(rows_m["params_trainable"].sum())
        system_info[label] = (total, stored, trainable, epochs)
        _, frame = score_system(ctx, system, "dev", golds["dev"], decoder, cache)
        m = metrics_from_frame(frame)
        ref = logged.loc[label]
        assert abs(m["intent_accuracy"] - ref["dev_intent"]) < 0.01 and abs(m["slot_micro_f1"] - ref["dev_slot_f1"]) < 0.01, (
            f"{label}: dev {m['intent_accuracy']:.2f} / {m['slot_micro_f1']:.2f} differs from rung E's "
            f"{ref['dev_intent']} / {ref['dev_slot_f1']}; the checkpoints or the decode changed")
        describe(label, "dev", frame, total, stored, trainable, epochs, "+".join(system.members))

    # Across-seed spread of each single-model family, on dev only: the other seeds are not final systems.
    seed_rows = []
    for label, system in systems.items():
        if len(system.members) != 1:
            continue
        fam = lb[lb["run"] == system.members[0]].iloc[0]["family"]
        for run_name in lb[lb["family"] == fam]["run"]:
            r, _ = native(ctx, run_name, "dev")
            seed_rows.append({"system": label, "run": run_name, "intent": r["intent_accuracy"], "slot": r["slot_micro_f1"]})
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(OUT / "seed_spread_dev.csv", index=False)
    print("\nAcross seeds on dev, the model's own decoding:")
    print(seeds.groupby("system")[["intent", "slot"]].agg(["mean", "std"]).round(2).to_string())

    # Dry run of the post-test steps on dev, so that a bug there cannot surface after the lock is written.
    for label in systems:
        for grp in [None] + GROUPS:
            paired_bootstrap(frames[("Stage1", "dev")], frames[(label, "dev")], group=grp)
        error_cases(frames[(label, "dev")], ctx.assets, ctx.assets.dev_df)
    for label in ["Stage1"] + list(systems):
        matched_control(frames[(label, "dev")])

    if dev_only:
        print("\nDev checks passed: every system fits the cap and reproduces rung E. Test was not touched.")
        sys.exit(0)

    # Test: written once, after the lock. Only Stage 1 and the final systems are scored on it.
    log["systems"] = {k: v.to_dict() for k, v in systems.items()}
    LOCK.write_text(json.dumps(log, indent=2))
    r, ck = native(ctx, STAGE1_NAME, "test", allow_test=True)
    describe("Stage1", "test", r["pred_frame"], ck["params_total"], stage1_stored, ck["params_trainable"],
             f"{ck['best_epoch']}/{ck['epochs_run']}", STAGE1_NAME)
    for label, system in systems.items():
        total, stored, trainable, epochs = system_info[label]
        _, frame = score_system(ctx, system, "test", golds["test"], decoder, cache, allow_test=True)
        describe(label, "test", frame, total, stored, trainable, epochs, "+".join(system.members))

    comp = pd.DataFrame(rows)
    comp.to_csv(OUT / "final_comparison.csv", index=False)
    pd.concat(oov_rows).round(2).to_csv(OUT / "final_oov_tables.csv", index=False)

    # Paired bootstrap against Stage 1 and the difficulty-matched OOV gap.
    boot = []
    for label in systems:
        for split in ("dev", "test"):
            for grp in [None] + GROUPS:
                boot.append({"system": label, "split": split,
                             **paired_bootstrap(frames[("Stage1", split)], frames[(label, split)], group=grp)})
    boot = pd.DataFrame(boot)
    boot.round(2).to_csv(OUT / "bootstrap_vs_stage1.csv", index=False)
    mc = pd.DataFrame([{"system": label, "split": "test", **matched_control(frames[(label, "test")])}
                       for label in ["Stage1"] + list(systems)])
    mc.round(2).to_csv(OUT / "matched_control_test.csv", index=False)

    # Error cases of every final system on test.
    for label in systems:
        errs = error_cases(frames[(label, "test")], ctx.assets, ctx.assets.test_df)
        errs.to_csv(OUT / f"error_cases_{label}.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\nBootstrap, system minus Stage 1:")
    print(boot.round(2).to_string(index=False))
    print("\nMatched control on test:")
    print(mc.round(2).to_string(index=False))
    print("\nFINAL TABLE")
    print(comp.drop(columns="members").to_string(index=False))

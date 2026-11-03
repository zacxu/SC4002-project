"""Shared context + one-call experiment runner. Results live in results/<run>/dev_metrics.json and
results/leaderboard.csv is rebuilt from those files, so a crash never loses a finished run."""
import gc
import json
import os
import re
import shutil
import time
from collections import OrderedDict
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from .config import ModelConfig, TrainConfig, CKPT_DIR, RESULTS_DIR, CACHE_DIR, DEVICE, seed_everything
from .data import load_massive, Assets, MassiveDataset, ExtVocab, build_oov_metadata
from .embeddings import load_word2vec, build_embedding_matrix
from .evaluate import predict, oov_table, matched_control
from .models import build_model, init_from
from .train import train_model, make_loaders
from . import guard

LEADERBOARD = RESULTS_DIR / "leaderboard.csv"
STAGE1_NAME = "stage1_baseline"


def write_json_atomic(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=2))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def rebuild_leaderboard():
    rows = []
    for p in sorted(RESULTS_DIR.glob("*/dev_metrics.json")):
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
        except ValueError as e:     # damaged by an unclean shutdown: that run is treated as not finished
            print(f"[leaderboard] skipping unreadable {p}: {e}", flush=True)
            continue
        row.setdefault("family", re.sub(r"__s\d+$", "", row["run"]))
        row.setdefault("seed", json.loads(row["train_config"]).get("seed"))
        row.setdefault("flagged", bool(row.get("k_extra")))
        rows.append(row)
    lb = pd.DataFrame(rows)
    try:
        lb.to_csv(LEADERBOARD, index=False)
    except PermissionError as e:    # for example open in Excel; the next run rewrites it
        print(f"[leaderboard] not updated, file is locked: {e}", flush=True)
    return lb


class Context:
    """Data and lookup tables. Word2Vec is loaded only when a run needs it, to keep RAM use low."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.df = load_massive(verbose=verbose)
        self.assets = Assets(self.df)
        if verbose:
            print("Assets:", self.assets.summary())
        self._w2v = None
        self._ext = OrderedDict()
        self._ds = OrderedDict()

        mat, stats, pre = CACHE_DIR / "emb_matrix.npy", CACHE_DIR / "emb_stats.json", CACHE_DIR / "oov_{}.pkl"
        if mat.exists() and stats.exists() and all(pre.with_name(pre.name.format(s)).exists() for s in ("dev", "test")):
            self.emb_matrix = np.load(mat)
            self.emb_stats = json.loads(stats.read_text())
            self.oov = {s: pd.read_pickle(pre.with_name(pre.name.format(s))) for s in ("dev", "test")}
        else:
            self.emb_matrix, self.emb_stats = build_embedding_matrix(self.assets, self.w2v, verbose=verbose)
            np.save(mat, self.emb_matrix)
            stats.write_text(json.dumps(self.emb_stats))
            pre_vocab = set(self.w2v.keys)
            self.oov = {s: build_oov_metadata(self.assets, self.assets.split(s), pre_vocab) for s in ("dev", "test")}
            for s, f in self.oov.items():
                f.to_pickle(pre.with_name(pre.name.format(s)))
        if verbose:
            print("Embedding init:", self.emb_stats)

    @property
    def w2v(self):
        if self._w2v is None:
            self._w2v = load_word2vec(verbose=self.verbose)
        return self._w2v

    def ext(self, k_extra, dim=300):
        key = (k_extra, dim)
        if key not in self._ext:
            self._ext[key] = ExtVocab(self.assets, self.w2v, k_extra, dim)
            while len(self._ext) > 1:
                self._ext.popitem(last=False)
        return self._ext[key]

    def datasets(self, cfg, k_extra=None, splits=("train", "dev")):
        """Datasets holding exactly the features of a model configuration. The last one stays cached."""
        key = (k_extra, tuple(sorted(cfg.adapters)), cfg.tag_scheme, cfg.frozen_dim, cfg.wide_intent,
               cfg.wide_slot_bits, cfg.gaz, tuple(splits))
        if key not in self._ds:
            self._ds.clear()
            ext = self.ext(k_extra, cfg.frozen_dim) if k_extra is not None else None
            self._ds[key] = {s: MassiveDataset(self.assets, self.assets.split(s), ext, cfg.adapters, cfg.tag_scheme,
                                               wide=cfg.wide_intent, wslot_bits=cfg.wide_slot_bits, gaz=cfg.gaz,
                                               loo=s == "train")
                             for s in splits}
        return self._ds[key]


def _same_config(row, mcfg, tcfg):
    a = ModelConfig.from_dict(json.loads(row["model_config"])).to_dict()
    b = TrainConfig.from_dict(json.loads(row["train_config"])).to_dict()
    return a == mcfg.to_dict() and b == tcfg.to_dict()


def _archive(name):
    """A finished run is never overwritten: if its name comes back with another configuration, it moves aside."""
    dest = RESULTS_DIR / "_archive" / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(str(RESULTS_DIR / name), str(dest / "results"))
    ckpt = CKPT_DIR / f"{name}.pt"
    if ckpt.exists():
        shutil.move(str(ckpt), str(dest / f"{name}.pt"))
    print(f"[{name}] configuration changed: earlier result archived to {dest}", flush=True)


def run(ctx, name, mcfg=None, tcfg=None, init=None, k_extra=None, frozen_dim=300, rung="", note="", flagged=None,
        verbose=True):
    """init: None (fresh, Word2Vec-initialised) | 'stage1' | a checkpoint run name.
    A run whose results already exist with the same configuration is not repeated."""
    mcfg = mcfg or ModelConfig()
    tcfg = tcfg or TrainConfig()
    ext = None
    if k_extra is not None:
        ext = ctx.ext(k_extra, frozen_dim)
        mcfg = replace(mcfg, n_frozen=ext.n_frozen, frozen_dim=frozen_dim)
    if flagged is None:
        flagged = k_extra is not None or tcfg.mention_replace > 0 or tcfg.lwtr > 0 or tcfg.sis > 0

    metrics_path = RESULTS_DIR / name / "dev_metrics.json"
    if metrics_path.exists() and (CKPT_DIR / f"{name}.pt").exists():
        row = None
        try:
            row = json.loads(metrics_path.read_text(encoding="utf-8"))
        except ValueError:
            pass
        if row is not None and _same_config(row, mcfg, tcfg):
            if verbose:
                print(f"[{name}] already done: dev joint {row['dev_joint']} (intent {row['dev_intent']}, "
                      f"slot {row['dev_slot_f1']})", flush=True)
            return row
        if row is not None:
            _archive(name)

    ds = ctx.datasets(mcfg, k_extra)
    init_name = "fresh" if init is None else str(init)
    meta = {"rung": rung, "note": note, "init": init_name, "k_extra": k_extra, "frozen_dim": frozen_dim,
            "flagged": flagged, "embedding_stats": ctx.emb_stats}
    if ext is not None:
        meta["ext"] = {"n_frozen": ext.n_frozen, "k_extra": ext.k_extra}

    model = ckpt = None
    for attempt in range(4):
        seed_everything(tcfg.seed)
        model = build_model(mcfg, ctx.assets, ctx.emb_matrix, ext)
        if init is not None:
            src = CKPT_DIR / f"{STAGE1_NAME if init == 'stage1' else init}.pt"
            state = torch.load(src, map_location="cpu")["model_state_dict"]
            copied = init_from(model, state)
            if verbose and attempt == 0:
                print(f"[{name}] warm start from {src.name}: {len(copied)} tensors copied", flush=True)
        oom = False
        try:
            model, ckpt = train_model(model, ctx.assets, ds["train"], ds["dev"], tcfg, name, meta=meta, verbose=verbose)
        except RuntimeError as e:
            if isinstance(e, guard.ResourceAbort) or "out of memory" not in str(e).lower() or attempt == 3:
                raise
            print(f"[{name}] GPU out of memory, waiting 5 minutes before retry {attempt + 1}", flush=True)
            oom = True
        if not oom:
            break
        # outside the except block: the traceback no longer pins the tensors, so they can really be freed
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(300)

    _, dev_loader = make_loaders(ctx.assets, ds["train"], ds["dev"], tcfg.batch_size, tcfg.seed)
    dev = predict(model, dev_loader, ctx.assets)
    table, merged = oov_table(dev["pred_frame"], ctx.oov["dev"])
    mc = matched_control(merged)
    run_dir = RESULTS_DIR / name
    table.to_csv(run_dir / "dev_oov_table.csv", index=False)
    merged.to_pickle(run_dir / "dev_pred_frame.pkl")
    g = table.set_index("group")["slot_micro_f1"]
    gi = table.set_index("group")["intent_accuracy"]
    row = {
        "run": name, "family": re.sub(r"__s\d+$", "", name), "seed": tcfg.seed, "rung": rung, "note": note,
        "init": init_name, "flagged": bool(flagged),
        "params_total": ckpt["params_total"], "params_trainable": ckpt["params_trainable"],
        "best_epoch": ckpt["best_epoch"], "epochs_run": ckpt["epochs_run"], "train_time_s": round(ckpt["elapsed_s"], 1),
        "dev_intent": round(dev["intent_accuracy"], 2), "dev_slot_f1": round(dev["slot_micro_f1"], 2),
        "dev_joint": round(dev["joint_score"], 2), "dev_exact": round(dev["exact_match"], 2),
        "f1_no_oov": round(g["No-Train-OOV"], 2), "f1_nonslot_oov": round(g["NonSlot-Train-OOV"], 2),
        "f1_slot_oov": round(g["Slot-Train-OOV"], 2), "intent_nonslot_oov": round(gi["NonSlot-Train-OOV"], 2),
        "matched_gap": round(mc["gap"], 2),
        "model_config": json.dumps(model.cfg.to_dict()), "train_config": json.dumps(tcfg.to_dict()),
        "k_extra": k_extra if k_extra is not None else 0, "frozen_dim": frozen_dim,
    }
    # Consistency check: the re-evaluated best checkpoint equals the logged best epoch.
    assert abs(dev["joint_score"] - ckpt["dev"]["joint_score"]) < 1e-6, (dev["joint_score"], ckpt["dev"])
    # A Stage-1 warm start whose new parts all start at zero must begin at Stage 1. Tolerance: the fast and
    # slow attention paths differ by float rounding, which can flip a single prediction.
    if (init == "stage1" and k_extra is None and mcfg.slot_head == "linear" and mcfg.tag_scheme == "bio"
            and mcfg.decode == "greedy" and not mcfg.wide_intent and not mcfg.wide_slot_bits):
        s1_dev = torch.load(CKPT_DIR / f"{STAGE1_NAME}.pt", map_location="cpu")["dev"]
        e0 = ckpt["history"][0]
        delta = abs(e0["dev_slot_f1"] - s1_dev["slot_micro_f1"]) + abs(e0["dev_intent_acc"] - s1_dev["intent_accuracy"])
        assert delta < 0.2, f"{name}: epoch 0 differs from Stage 1 by {delta}"

    s1_path = RESULTS_DIR / STAGE1_NAME / "dev_metrics.json"
    if s1_path.exists() and name != STAGE1_NAME:
        s1 = json.loads(s1_path.read_text(encoding="utf-8"))
        row["gain_slot_f1"] = round(row["dev_slot_f1"] - s1["dev_slot_f1"], 2)
        row["gain_intent"] = round(row["dev_intent"] - s1["dev_intent"], 2)
        row["gain_slot_oov"] = round(row["f1_slot_oov"] - s1["f1_slot_oov"], 2)
        row["targeted_adv"] = round(row["gain_slot_oov"] - (row["f1_no_oov"] - s1["f1_no_oov"]), 2)
    write_json_atomic(metrics_path, row)
    rebuild_leaderboard()
    if verbose:
        print(table.round(2).to_string(index=False))
        print(f"[{name}] matched control: { {k: round(v, 2) for k, v in mc.items()} }", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return row

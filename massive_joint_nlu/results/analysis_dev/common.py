"""Shared helpers for the dev-only analysis.

No torch, no sentencepiece, no GPU, no w2v matrix. Test rows are dropped the moment the CSV is read.
Tokeniser and BIO parser are copied verbatim from .review_snapshot/src/data.py (that module imports torch);
bio_to_spans is copied verbatim from .review_snapshot/src/evaluate.py.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import ctypes
import json
import re
import sys
from collections import Counter
from ctypes import wintypes
from pathlib import Path

MEM_CAP_MB = 400


def apply_guards(limit_mb=MEM_CAP_MB):
    """Self-imposed commit limit and below-normal priority, so this process cannot starve the GPU job."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("Basic", BASIC), ("IoInfo", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    k32.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)

    job = k32.CreateJobObjectW(None, None)
    info = EXTENDED()
    info.Basic.LimitFlags = 0x00000100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    info.ProcessMemoryLimit = limit_mb * 1024 * 1024
    ok1 = k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
    ok2 = k32.AssignProcessToJobObject(job, k32.GetCurrentProcess())
    k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
    if not (ok1 and ok2):
        raise RuntimeError(f"memory guard not applied: {ctypes.get_last_error()}")
    return job  # keep the handle alive


_JOB = apply_guards()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import psutil

ROOT = Path("C:/Users/user/massive_joint_nlu")
OUT = ROOT / "results" / "analysis_dev"
CSV = ROOT / "data" / "massive_enUS_full.csv"
KEYS = ROOT / "cache" / "w2v_casefold_keys.json"
FRAME = ROOT / "results" / "stage1_baseline" / "dev_pred_frame.pkl"
A1_FRAME = ROOT / "results" / "A1_ft_recipe" / "dev_pred_frame.pkl"
GROUPS = ["No-Train-OOV", "NonSlot-Train-OOV", "Slot-Train-OOV"]

# ---- verbatim from .review_snapshot/src/data.py
TOKEN_RE = re.compile(r"\w+(?:['’\-]\w+)*|[^\w\s]", re.UNICODE)
SLOT_RE = re.compile(r"\[([^\[\]:]+?)\s*:\s*([^\[\]]*?)\]")


def tokenize(text):
    return [m.group(0) for m in TOKEN_RE.finditer(str(text))]


def is_lexical(token):
    return bool(re.search(r"[A-Za-z0-9]", token))


def parse_annotated_utterance(annot_utt):
    tokens, bio, slot_mask = [], [], []
    pos = 0
    for m in SLOT_RE.finditer(str(annot_utt)):
        before = tokenize(annot_utt[pos:m.start()])
        tokens += before
        bio += ["O"] * len(before)
        slot_mask += [False] * len(before)
        slot_type = m.group(1).strip()
        span = tokenize(m.group(2).strip())
        for j, tok in enumerate(span):
            tokens.append(tok)
            bio.append(("B-" if j == 0 else "I-") + slot_type)
            slot_mask.append(True)
        pos = m.end()
    after = tokenize(annot_utt[pos:])
    tokens += after
    bio += ["O"] * len(after)
    slot_mask += [False] * len(after)
    return {"tokens": tokens, "bio": bio, "slot_mask": slot_mask}


# ---- verbatim from .review_snapshot/src/evaluate.py
def bio_to_spans(tags):
    """(slot_type, start, end_exclusive); a stray I-X opens a new span (hand-off rule)."""
    spans, cur, start = [], None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        if tag == "O":
            if cur is not None:
                spans.append((cur, start, i))
            cur = start = None
            continue
        prefix, typ = tag.split("-", 1)
        if prefix == "B" or cur != typ:
            if cur is not None:
                spans.append((cur, start, i))
            cur, start = typ, i
    return spans


def slot_counts(gold_lists, pred_lists):
    TP = FP = FN = 0
    for g, p in zip(gold_lists, pred_lists):
        g, p = Counter(bio_to_spans(g)), Counter(bio_to_spans(p))
        TP += sum((g & p).values())
        FP += sum((p - g).values())
        FN += sum((g - p).values())
    return TP, FP, FN


def f1_from(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 0.0 if d == 0 else 100.0 * 2 * tp / d


# ---- loading
def load_train_dev():
    """Reads the CSV, drops test rows immediately, parses annotations of train and dev only."""
    df = pd.read_csv(CSV)
    df = df[df["partition"] != "test"].reset_index(drop=True)
    assert set(df["partition"]) == {"train", "dev"}
    assert (df["partition"] == "train").sum() == 11514 and (df["partition"] == "dev").sum() == 2033
    assert df[["id", "partition", "utt", "intent", "annot_utt"]].isna().sum().sum() == 0
    parsed = [parse_annotated_utterance(a) for a in df["annot_utt"]]
    df["toks"] = [p["tokens"] for p in parsed]
    df["bio"] = [p["bio"] for p in parsed]
    df["smask"] = [p["slot_mask"] for p in parsed]
    df["n_tok"] = df["toks"].map(len)
    assert all(tokenize(u) == t for u, t in zip(df["utt"], df["toks"])), "alignment error"
    df["key"] = [" ".join(t.casefold() for t in toks) for toks in df["toks"]]
    df["tmpl"] = [template(t, b) for t, b in zip(df["toks"], df["bio"])]
    df["scenario"] = df["intent"].str.split("_", n=1).str[0]
    train = df[df["partition"] == "train"].reset_index(drop=True)
    dev = df[df["partition"] == "dev"].reset_index(drop=True)
    return train, dev


def template(toks, bio):
    """Casefolded tokens with each gold slot span replaced by <type>."""
    out, j = [], 0
    spans = {s: (t, e) for t, s, e in bio_to_spans(bio)}
    while j < len(toks):
        if j in spans:
            t, e = spans[j]
            out.append(f"<{t}>")
            j = e
        else:
            out.append(toks[j].casefold())
            j += 1
    return " ".join(out)


def load_frame(path=FRAME):
    """Stage-1 dev predictions, joined to dev rows and checked against the CSV."""
    fr = pd.read_pickle(path)
    assert "torch" not in sys.modules
    return fr


def load_all(path=FRAME):
    train, dev = load_train_dev()
    intents = sorted(train["intent"].unique())
    assert len(intents) == 60
    fr = load_frame(path)
    assert set(fr["id"]) == set(dev["id"]) and len(fr) == len(dev)
    keep = ["id", "gold_intent_id", "pred_intent_id", "gold_tags", "pred_tags"]
    extra = [c for c in ("n_tokens", "n_gold_slots", "n_train_oov", "n_slot_train_oov",
                         "train_oov_group", "pretrained_oov_group") if c in fr.columns]
    m = dev.merge(fr[keep + extra], on="id", how="left", validate="one_to_one")
    m["gold_intent"] = [intents[i] for i in m["gold_intent_id"]]
    m["pred_intent"] = [intents[i] for i in m["pred_intent_id"]]
    assert (m["gold_intent"] == m["intent"]).all(), "gold intent ids do not match the CSV"
    assert all(list(a) == list(b) for a, b in zip(m["gold_tags"], m["bio"])), "gold tags do not match the CSV"
    m["int_ok"] = m["gold_intent"] == m["pred_intent"]
    return train, m, intents


def mem(tag=""):
    mi = psutil.Process().memory_info()
    print(f"[mem {tag}] wset {mi.rss/2**20:.0f} MB | peak wset {mi.peak_wset/2**20:.0f} MB | "
          f"private {mi.private/2**20:.0f} MB | peak private {mi.peak_pagefile/2**20:.0f} MB", flush=True)


def save(df, name):
    df.to_csv(OUT / name, index=False, encoding="utf-8", lineterminator="\n")


def save_json(obj, name):
    (OUT / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float), encoding="utf-8")


def show(df, n=None):
    with pd.option_context("display.width", 220, "display.max_columns", 50, "display.max_rows", 500,
                           "display.max_colwidth", 90):
        print((df if n is None else df.head(n)).to_string(index=False), flush=True)

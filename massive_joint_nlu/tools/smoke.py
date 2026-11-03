"""Smoke test (runs on CPU so it can never compete for the GPU): data, Stage-1 param count, every
switch, tag schemes, CRF against brute force, biaffine round trip, frozen rows, OOV simulation,
regularisers, sparse paths, post-hoc decoding. Reads train and dev annotations only."""
import os
os.environ["MASSIVE_CPU"] = "1"
import itertools
import random
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import torch
from src.config import ModelConfig, TrainConfig, DEVICE, seed_everything
from src.data import (load_massive, Assets, MassiveDataset, ExtVocab, make_collate, to_bio, gold_spans,
                      parse_annotated_utterance, ALL_FEATURES)
from src.embeddings import CasefoldVectors
from src.evaluate import to_device, bio_to_spans, micro_f1, matched_control, metrics_from_frame
from src.models import build_model, count_params, init_from, CRF, viterbi, JointModel
from src.posthoc import (crf_log_marginals, Decoder, Decode, System, collect, system_params, frame_from_decode,
                         _biaffine_tag_probs)
from src.train import corrupt_batch, _train_step, EMA

assert DEVICE.type == "cpu"
torch.set_num_threads(2)
df = load_massive()
A = Assets(df)
print(A.summary())
assert len(A.word_vocab) == 5300 and len(A.slot_labels) == 108 and len(A.char_vocab) == 34 and len(A.slot_types) == 55
print("types with touching spans in train:", sorted(A.adjacent_types))

seed_everything(0)
s1 = build_model(ModelConfig(), A).eval()
assert count_params(s1)[0] == 4_344_472, count_params(s1)

small = A.dev_df.sort_values("utt", key=lambda c: c.str.len()).tail(64)
extras = dict(wide=2000, wslot_bits=10, gaz=True)
ds = MassiveDataset(A, small, features=ALL_FEATURES, **extras)
collate = make_collate(A)
batch = to_device(collate([ds[i] for i in range(32)]))
with torch.no_grad():
    ref_i, ref_s = s1(batch)

# Zero-initialised switches must reproduce the Stage-1 outputs exactly.
equiv = {
    "bpe": ModelConfig(adapters=("bpe",)), "ngram": ModelConfig(adapters=("ngram",)), "char": ModelConfig(adapters=("char",)),
    "meanmax": ModelConfig(pooling="meanmax"), "attn": ModelConfig(pooling="attn"), "labelattn": ModelConfig(label_attn=True),
    "hier": ModelConfig(hier_intent=True), "interact": ModelConfig(interaction=True), "i2s": ModelConfig(intent_to_slot=True),
    "s2i": ModelConfig(slot_to_intent=True), "factorized": ModelConfig(factorized=True), "prelstm": ModelConfig(pre_lstm=True),
    "auxlm": ModelConfig(aux_lm=True), "relbias": ModelConfig(rel_bias=True), "tasktop": ModelConfig(task_top=True),
    "refine": ModelConfig(refine=True), "coint": ModelConfig(coint=True), "gaz": ModelConfig(gaz=True),
    "presence": ModelConfig(presence=True), "supcon": ModelConfig(supcon_dim=64), "anchor": ModelConfig(anchor=True),
    "many": ModelConfig(adapters=("bpe", "char"), pooling="attn", label_attn=True, hier_intent=True, interaction=True,
                        intent_to_slot=True, slot_to_intent=True, factorized=True, pre_lstm=True, rel_bias=True,
                        task_top=True, refine=True, coint=True, gaz=True, presence=True),
}
expected_counts = {"bpe": 4_493_272, "ngram": 5_323_288, "char": 4_380_936}
for name, cfg in equiv.items():
    v = build_model(cfg, A)
    init_from(v, s1.state_dict())
    v.eval()
    with torch.no_grad():
        i2, s2 = v(batch)
    di, ds_ = (i2 - ref_i).abs().max().item(), (s2 - ref_s).abs().max().item()
    print(f"{name:10s} params {count_params(v)[0]:>10,}  max|diff| intent {di:.1e} slot {ds_:.1e}")
    assert di < 1e-3 and ds_ < 1e-3, name
    if name in expected_counts:
        assert count_params(v)[0] == expected_counts[name], name
    v.train()
    il, sl = v(batch)
    tc = TrainConfig(lambda_aux=0.2, supcon_lambda=0.1, lambda_presence=0.2, label_smoothing=0.1)
    b_ = dict(batch, sim_mask=batch["attention_mask"] & batch["lexical_mask"], orig_word_ids=batch["word_ids"])
    loss = v.loss(il, sl, b_, tc)
    loss.backward()
    assert torch.isfinite(loss), name
assert count_params(build_model(ModelConfig(anchor=True), A))[0] == 4_344_472 + 1_590_000   # the frozen copy counts

# Tag schemes: conversion round trip and legality of every gold sequence on train and dev.
for scheme in ("bio", "bioes"):
    ts = A.tagset(scheme)
    for r in df[df["partition"] != "test"].itertuples(index=False):
        tags = parse_annotated_utterance(r.annot_utt)["bio"]
        ids = ts.encode(tags)
        assert [to_bio(ts.labels[i]) for i in ids] == tags
        assert ts.allowed_start[ids[0]] and ts.allowed_end[ids[-1]]
        assert all(ts.allowed[a, b] for a, b in zip(ids, ids[1:])), (r.id, tags)
    print(scheme, "labels", len(ts.labels), "all gold paths legal on train and dev")

# Heads and encoders that do not start at Stage 1: forward, backward, valid decode.
for name, cfg in {"crf": ModelConfig(slot_head="crf"),
                  "bioes_crf": ModelConfig(slot_head="crf", tag_scheme="bioes"),
                  "biaffine": ModelConfig(slot_head="biaffine"),
                  "viterbi": ModelConfig(decode="viterbi"),
                  "bilstm": ModelConfig(encoder="bilstm", lstm_hidden=256, coint=True),
                  "big": ModelConfig(d_model=384, n_heads=6, n_layers=6, d_ff=1536),
                  "tiny": ModelConfig(d_model=128, n_layers=2, d_ff=384),
                  "dual": ModelConfig(dual_emb=True),
                  "wide": ModelConfig(wide_intent=2000, decode="viterbi"),
                  "wslot": ModelConfig(wide_slot_bits=10, decode="viterbi"),
                  "wslotcrf": ModelConfig(wide_slot_bits=10, slot_head="crf")}.items():
    v = build_model(cfg, A)
    d2 = MassiveDataset(A, small, features=(), scheme=cfg.tag_scheme, wide=cfg.wide_intent, wslot_bits=cfg.wide_slot_bits)
    b2 = to_device(collate([d2[i] for i in range(16)]))
    v.train()
    il, sl = v(b2)
    loss = v.loss(il, sl, b2, TrainConfig(boundary_eps=0.2 if cfg.slot_head == "biaffine" else 0.0))
    loss.backward()
    v.eval()
    with torch.no_grad():
        il, sl = v(b2)
    tags = v.decode_tags(sl, b2["attention_mask"])
    assert [len(t) for t in tags] == b2["attention_mask"].sum(1).tolist()
    if cfg.slot_head == "crf" or cfg.decode == "viterbi":
        ts = A.tagset(cfg.tag_scheme)
        if cfg.slot_head == "crf":
            paths = v.crf.decode(sl, b2["attention_mask"])
        else:
            paths = viterbi(torch.log_softmax(sl, -1), b2["attention_mask"], v.v_trans, v.v_start, v.v_end)
        for p in paths:
            assert ts.allowed_start[p[0]] and ts.allowed_end[p[-1]]
            assert all(ts.allowed[a, c] for a, c in zip(p, p[1:]))
    if cfg.slot_head == "biaffine":
        gold = v.biaffine.gold(b2["slot_ids"], b2["attention_mask"], v.id2tag)
        for b in range(b2["slot_ids"].size(0)):
            n = int(b2["attention_mask"][b].sum())
            spans = bio_to_spans([v.id2tag[x] for x in b2["slot_ids"][b, :n].tolist()])
            got = {(v.biaffine.types[int(gold[b, i, j])], i, j + 1) for i in range(n) for j in range(n)
                   if int(gold[b, i, j]) != v.biaffine.none}
            assert got == set(spans), (got, spans)
        for t in tags:   # spans never overlap and tags are well formed
            for i, x in enumerate(t):
                if x.startswith("I-"):
                    assert i > 0 and t[i - 1][2:] == x[2:] and t[i - 1] != "O"
    print(f"{name:10s} params {count_params(v)[0]:>10,}  loss {loss.item():.3f}  decode ok")

# The sparse paths are mixed at evaluation and receive their own loss in training.
v = build_model(ModelConfig(wide_intent=2000), A)
d3 = MassiveDataset(A, small, features=(), wide=2000)
b3 = to_device(collate([d3[i] for i in range(8)]))
v.eval()
with torch.no_grad():
    il, _ = v(b3)
assert v._deep_logp is not None and torch.allclose(il, 0.5 * v._deep_logp + 0.5 * v._wide_logp)
v.train()
il, sl = v(b3)
assert v._wide_logits is not None and v.loss(il, sl, b3, TrainConfig()) > v.loss(il, sl, b3, None) - 1e-6
assert (b3["wide_vals"] >= 0).all() and b3["wide_ids"].max() < A.wide_size(2000)

# Gazetteer features are leave-one-out on train: a mention seen once in train does not fire on itself.
ng_all, ng_type, best = A.mention_tables()
once = next(g for (t, g), c in ng_type.items() if c == 1 and ng_all[g] == 1)
row = next(r for r in A.train_df.itertuples(index=False)
           if tuple(g.casefold() for g in once) in
           {tuple(x.casefold() for x in parse_annotated_utterance(r.annot_utt)["tokens"][a:b])
            for _, a, b in gold_spans(parse_annotated_utterance(r.annot_utt)["bio"])})
parsed = parse_annotated_utterance(row.annot_utt)
low = [t.casefold() for t in parsed["tokens"]]
pos = next(i for i in range(len(low)) if tuple(low[i:i + len(once)]) == once)
frame1 = A.train_df[A.train_df["id"] == row.id]
loo = MassiveDataset(A, frame1, features=(), gaz=True, loo=True)[0]["gaz_ids"].tolist()
noloo = MassiveDataset(A, frame1, features=(), gaz=True, loo=False)[0]["gaz_ids"].tolist()
assert noloo[pos] != 0, (once, noloo)
assert loo[pos] == 0 or loo[pos] != noloo[pos], (once, loo, noloo)
print("gazetteer leave-one-out ok")

# CRF against brute force, including padding.
K = 3
crf = CRF(K, np.ones((K, K), bool), np.ones(K, bool))
for p in (crf.transitions, crf.start, crf.end):
    torch.nn.init.normal_(p)
em = torch.randn(2, 4, K)
tags = torch.tensor([[0, 2, 1, 1], [1, 0, -100, -100]])
mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)


def path_score(b, p):
    s = crf.start[p[0]] + em[b, 0, p[0]]
    for t in range(1, len(p)):
        s = s + crf.transitions[p[t - 1], p[t]] + em[b, t, p[t]]
    return s + crf.end[p[-1]]


exp = 0.0
dec = []
for b, n in enumerate((4, 2)):
    paths = list(itertools.product(range(K), repeat=n))
    scores = torch.stack([path_score(b, p) for p in paths])
    gold = tags[b, :n].tolist()
    exp = exp + torch.logsumexp(scores, 0) - path_score(b, gold)
    dec.append(list(paths[int(scores.argmax())]))
got = crf.nll(em, tags, mask)
assert abs(got.item() - exp.item()) < 1e-3, (got.item(), exp.item())
assert crf.decode(em, mask) == dec
trans, start, end = crf.transitions.detach(), crf.start.detach(), crf.end.detach()
lm = crf_log_marginals(em, mask, trans, start, end)
for b, n in enumerate((4, 2)):
    paths = list(itertools.product(range(K), repeat=n))
    probs = torch.softmax(torch.stack([path_score(b, p) for p in paths]), 0).detach()
    for t in range(n):
        want = torch.zeros(K)
        for pth, pr in zip(paths, probs):
            want[pth[t]] += pr
        assert torch.allclose(lm[b, t].exp(), want, atol=1e-4), (b, t)
print("CRF nll, Viterbi and marginals match brute force, with padding")

# Metric on hand-made cases.
assert bio_to_spans(["O", "B-a", "I-a", "B-a", "I-b", "O"]) == [("a", 1, 3), ("a", 3, 4), ("b", 4, 5)]
assert micro_f1([["B-a", "I-a", "O"]], [["B-a", "O", "O"]]) == 0.0
assert micro_f1([["B-a", "I-a", "O"]], [["B-a", "I-a", "O"]]) == 100.0
assert abs(micro_f1([["B-a", "O", "B-b"]], [["B-a", "O", "O"]]) - 100 * 2 / 3) < 1e-9

# Matched control averages over all tied rows and is balanced on exact matches.
fr = pd.DataFrame([
    {"train_oov_group": "Slot-Train-OOV", "n_tokens": 3, "n_gold_slots": 1, "gold_tags": ["B-a", "O", "O"], "pred_tags": ["O", "O", "O"],
     "gold_intent_id": 0, "pred_intent_id": 0},
    {"train_oov_group": "No-Train-OOV", "n_tokens": 3, "n_gold_slots": 1, "gold_tags": ["B-a", "O", "O"], "pred_tags": ["B-a", "O", "O"],
     "gold_intent_id": 0, "pred_intent_id": 0},
    {"train_oov_group": "No-Train-OOV", "n_tokens": 3, "n_gold_slots": 1, "gold_tags": ["B-a", "O", "O"], "pred_tags": ["O", "O", "O"],
     "gold_intent_id": 0, "pred_intent_id": 0},
    {"train_oov_group": "No-Train-OOV", "n_tokens": 9, "n_gold_slots": 3, "gold_tags": ["B-a"] + ["O"] * 8, "pred_tags": ["O"] * 9,
     "gold_intent_id": 0, "pred_intent_id": 0},
])
mc = matched_control(fr)
# weighted pool: half a true positive and half a false negative, so micro-F1 = 2*.5 / (2*.5 + .5) = 66.67
assert abs(mc["matched_no_oov_f1"] - 200 / 3) < 1e-6 and mc["slot_oov_f1"] == 0.0 and mc["matched_avg_tokens"] == 3.0, mc
print("matched control ok")

# Extended vocabulary with fake vectors of low rank, so that PCA must reconstruct them exactly.
rng = np.random.default_rng(0)
train_words = [w for w in A.word_vocab[2:] if w.isalpha()][:3000]
fake_keys = train_words + [f"zzword{i}" for i in range(2000)]
basis = rng.normal(size=(50, 300)).astype(np.float32)
fake_mat = rng.normal(size=(len(fake_keys), 50)).astype(np.float32) @ basis
fake = CasefoldVectors(fake_keys, fake_mat, len(fake_keys))
for dim in (300, 100):
    ext = ExtVocab(A, fake, 500, dim)
    assert ext.n_frozen == 3500 and ext.k_extra == 500 and ext.matrix.shape == (3500, dim)
    cfg = ModelConfig(n_frozen=ext.n_frozen, frozen_dim=dim)
    m = build_model(cfg, A, None, ext)
    init_from(m, s1.state_dict())
    m.eval()
    assert not m.frozen_emb.weight.requires_grad
    gid = ext.word2gid["zzword7"]
    assert gid >= len(A.word_vocab)
    with torch.no_grad():
        got = m.embed(torch.tensor([[gid, ext.swap_of[A.word2id[train_words[3]]], A.word2id[train_words[3]]]]))[0]
    assert np.allclose(got[0].numpy(), fake_mat[fake.index["zzword7"]], atol=2e-2), dim
    assert np.allclose(got[1].numpy(), fake_mat[fake.index[train_words[3]]], atol=2e-2), dim
    assert np.allclose(got[2].numpy(), s1.embedding.weight[A.word2id[train_words[3]]].detach().numpy())
    frame = small.assign(annot_utt="play zzword7 now", utt="play zzword7 now")
    dsx = MassiveDataset(A, frame.head(2), ext, features=())
    assert dsx[0]["word_ids"][1] == gid
    print(f"ext dim {dim}: params {count_params(m)[0]:,}, trainable {count_params(m)[1]:,}, frozen rows read back correctly")

# OOV simulation.
ds3 = MassiveDataset(A, A.train_df.head(256), features=())
b3 = to_device(collate([ds3[i] for i in range(256)]))
counts = torch.as_tensor(A.word_counts)
starts = torch.tensor([t[0] in "BS" for t in ds3.tagset.labels])
orig = b3["word_ids"].clone()
out = corrupt_batch(dict(b3), TrainConfig(slot_dropout=1.0), A.UNK_ID, counts, starts)
in_slot = (b3["slot_ids"] > 0) & b3["lexical_mask"]
assert (out["word_ids"][in_slot] == A.UNK_ID).all()
assert (out["word_ids"][~in_slot] == orig[~in_slot]).all()
out = corrupt_batch(dict(b3), TrainConfig(rare_alpha=1e-3), A.UNK_ID, counts, starts)
assert (out["word_ids"] != orig).float().mean() < 0.01
out = corrupt_batch(dict(b3), TrainConfig(), A.UNK_ID, counts, starts)
assert (out["word_ids"] == orig).all() and "sim_mask" not in out
out = corrupt_batch(dict(b3), TrainConfig(mention_dropout=1.0), A.UNK_ID, counts, starts)
assert (out["word_ids"][in_slot] == A.UNK_ID).all() and (out["word_ids"][~in_slot] == orig[~in_slot]).all()
torch.manual_seed(0)
out = corrupt_batch(dict(b3), TrainConfig(mention_dropout=0.5), A.UNK_ID, counts, starts)
whole = 0
for i in range(256):
    n = int(b3["attention_mask"][i].sum())
    tags_i = [ds3.tagset.labels[x] for x in b3["slot_ids"][i, :n].tolist()]
    for _, a, c in gold_spans(tags_i):
        lex = [j for j in range(a, c) if b3["lexical_mask"][i, j]]
        dropped = [bool(out["word_ids"][i, j] == A.UNK_ID) for j in lex]
        assert all(dropped) or not any(dropped), "a span is dropped as a whole"
        whole += int(any(dropped))
assert whole > 20
print("OOV simulation targets the intended tokens")

# Recombination keeps tokens and tags aligned, and only uses train values.
ds4 = MassiveDataset(A, A.train_df.head(512), features=())
random.seed(1)
for fn, min_changed in ((ds4._replaced, 50), (ds4._lwtr, 50), (ds4._sis, 5)):
    changed = 0
    for i in range(512):
        row = fn(i)
        assert len(row["word_ids"]) == len(row["slot_ids"]) == len(row["lexical_mask"])
        changed += int(len(row["word_ids"]) != len(ds4.rows[i]["word_ids"]) or (row["word_ids"] != ds4.rows[i]["word_ids"]).any())
        assert (row["word_ids"] != A.UNK_ID).all()
    assert changed > min_changed, (fn.__name__, changed)
print("recombination ok")

# One training step with each regulariser, and EMA.
ds5 = MassiveDataset(A, small, features=())
b5 = to_device(collate([ds5[i] for i in range(16)]))
for name, tc, cfg in [("adv", TrainConfig(adv_eps=0.1), ModelConfig()),
                      ("adv_frozen", TrainConfig(adv_eps=0.1, emb_lr_mult=0.0), ModelConfig()),
                      ("rdrop", TrainConfig(rdrop_alpha=1.0), ModelConfig(hier_intent=True)),
                      ("sam", TrainConfig(sam_rho=0.1), ModelConfig(slot_head="crf")),
                      ("tok+aux", TrainConfig(lambda_token_intent=0.2, lambda_aux=0.2, word_dropout=0.3), ModelConfig(aux_lm=True)),
                      ("oweight", TrainConfig(o_weight=0.5, label_smoothing=0.1), ModelConfig()),
                      ("cons", TrainConfig(cons_alpha=1.0, word_dropout=0.3), ModelConfig()),
                      ("mixup", TrainConfig(mixup_alpha=0.2), ModelConfig()),
                      ("supcon", TrainConfig(supcon_lambda=0.1), ModelConfig(supcon_dim=64)),
                      ("anchor", TrainConfig(anchor_lambda=1e-2), ModelConfig(anchor=True))]:
    v = build_model(cfg, A)
    if tc.emb_lr_mult == 0:
        v.embedding.weight.requires_grad_(False)
    v._track_emb = tc.adv_eps > 0
    v.train()
    params = [p for p in v.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in params]
    opt = torch.optim.AdamW(params, lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    ema = EMA(v, 0.9)
    b = corrupt_batch(dict(b5), tc, A.UNK_ID, counts, starts)
    loss = _train_step(v, b, tc, opt, sched, ema, params)
    moved = sum((a != p.detach()).any().item() for a, p in zip(before, params))
    assert np.isfinite(loss) and moved > 0.5 * len(params), name
    print(f"{name:10s} one step ok, loss {loss:.3f}, {moved}/{len(params)} tensors moved")

# Post-hoc decoding rules, on hand-made outputs.
dec = Decoder(A)
bio = A.slot_labels
L = len(bio)


def fake_tags(seq, strong=8.0):
    x = torch.full((len(seq), L), -strong)
    for t, tag in enumerate(seq):
        x[t, bio.index(tag)] = 0.0
    return torch.log_softmax(x, -1)


def out_of(intent, tl, presence=None):
    return {"intent": intent, "deep": None, "wide": None, "tags": tl, "presence": presence}


typ = next(t for t in A.slot_types if "I-" + t in bio and t in A.slot_values)
intent = torch.log_softmax(torch.randn(60), 0)
# a stray I- start is repaired by the BIO rules
got = dec.decode([{7: out_of(intent, fake_tags(["I-" + typ, "O"]))}], Decode())
assert not got[7][1][0].startswith("I-"), got
assert got[7][0] == int(intent.argmax())
# the same rules hold under span decoding, with and without the mention table
lows = {7: ["alpha", "beta"]}
got = dec.decode([{7: out_of(intent, fake_tags(["B-" + typ, "I-" + typ], strong=6.0))}], Decode(method="span", tau=0.4), lows)
assert got[7][1] == ["B-" + typ, "I-" + typ], got
# two touching spans of one type are illegal unless train has them
if typ not in A.adjacent_types:
    got = dec.decode([{7: out_of(intent, fake_tags(["B-" + typ, "B-" + typ]))}], Decode())
    assert got[7][1] != ["B-" + typ, "B-" + typ], got
# a high threshold removes a weak span, a low one keeps it
weak = torch.log_softmax(torch.tensor([[1.0 if i == 0 else (1.2 if bio[i] == "B-" + typ else -8.0) for i in range(L)]]), -1)
assert dec.decode([{1: out_of(intent, weak)}], Decode(method="span", tau=0.9), {1: ["alpha"]})[1][1] == ["O"]
assert dec.decode([{1: out_of(intent, weak)}], Decode(method="span", tau=0.3), {1: ["alpha"]})[1][1] == ["B-" + typ]
# span marginals agree with brute force over legal paths
em = torch.log_softmax(torch.randn(3, L) * 2, -1).numpy().astype(np.float64)
marg = dec.span_marginals(em)
legal = []
sub = [0] + [bio.index("B-" + typ), bio.index("I-" + typ)]
for t in A.slot_types:
    if t != typ and "B-" + t in bio:
        sub.append(bio.index("B-" + t))
        break
em2 = np.full_like(em, -50.0)
em2[:, sub] = em[:, sub]
em2 = em2 - np.logaddexp.reduce(em2, axis=1, keepdims=True)
marg2 = dec.span_marginals(em2)
scores = {}
for p in itertools.product(range(L), repeat=1):
    pass
tot = 0.0
acc = {}
ts = A.tagset("bio")
for p in itertools.product(sub, repeat=3):
    if not ts.allowed_start[p[0]] or not all(ts.allowed[a, b] for a, b in zip(p, p[1:])):
        continue
    v = np.exp(sum(em2[i, p[i]] for i in range(3)))
    tot += v
    for sp in bio_to_spans([bio[x] for x in p]):
        acc[sp] = acc.get(sp, 0.0) + v
for (name, a, b), pv in acc.items():
    exp_p = pv / tot
    got_p = marg2.get((A.slot_types.index(name), a, b), 0.0)
    if exp_p > 0.02:
        assert abs(exp_p - got_p) < 1e-3, (name, a, b, exp_p, got_p)
print("span posteriors match brute force over legal paths")
# the mask removes a type never seen with the intent; joint top-k can switch intent
name = next(i for i, n in enumerate(A.intent_labels) if A.intent_slot_types.get(n))
seen = A.intent_slot_types[A.intent_labels[name]]
bad = next(t for t in A.slot_types if t not in seen and "B-" + t in bio)
good = sorted(seen)[0]
il = torch.full((60,), -20.0)
il[name] = 0.0
d1 = dec.decode([{1: out_of(il, fake_tags(["B-" + bad, "O"], strong=2.0))}], Decode(mask_beta=50.0))
assert d1[1][1][0] == "O" or d1[1][1][0][2:] in seen, d1
d2 = dec.decode([{1: out_of(il, fake_tags(["B-" + good, "O"], strong=2.0))}], Decode(mask_beta=50.0))
assert d2[1][1][0] == "B-" + good, d2
other = next(i for i, n in enumerate(A.intent_labels) if bad in A.intent_slot_types.get(n, set()))
il2 = torch.full((60,), -20.0)
il2[name], il2[other] = -0.5, -1.0
d3 = dec.decode([{1: out_of(il2, fake_tags(["B-" + bad], strong=4.0))}], Decode(mask_beta=50.0, top_k=3, lam=1.0))
assert d3[1][0] == other, d3
# averaging members: two votes against one, in probability space
a, b_ = fake_tags(["B-" + good]), fake_tags(["O"])
d4 = dec.decode([{1: out_of(il, a)}, {1: out_of(il, a)}, {1: out_of(il, b_)}], Decode())
assert d4[1][1] == ["B-" + good], d4
# the deep and sparse intent parts are remixed on request
deep, wide = torch.log_softmax(torch.randn(60), 0), torch.log_softmax(torch.randn(60), 0)
m = {"intent": deep, "deep": deep, "wide": wide, "tags": fake_tags(["O"]), "presence": None}
assert dec.decode([{1: m}], Decode(wide_mix=1.0))[1][0] == int(deep.argmax())
assert dec.decode([{1: m}], Decode(wide_mix=0.0))[1][0] == int(wide.argmax())
# every ensemble member is stored in full, so its frozen table counts again
lb_fake = pd.DataFrame([{"run": "x", "params_total": 100, "params_trainable": 60, "k_extra": 5, "frozen_dim": 300},
                        {"run": "y", "params_total": 100, "params_trainable": 60, "k_extra": 5, "frozen_dim": 300},
                        {"run": "z", "params_total": 50, "params_trainable": 50, "k_extra": 0, "frozen_dim": 300}])
assert system_params(lb_fake, ["x", "y"]) == 200 and system_params(lb_fake, ["x", "y", "z"]) == 250
assert system_params(lb_fake, ["z"]) == 50
try:
    collect(None, "x", "test")
    raise AssertionError("test must be locked")
except RuntimeError:
    pass
# biaffine span probabilities become a normalised token distribution
bv = build_model(ModelConfig(slot_head="biaffine"), A).eval()
b2b = to_device(collate([ds5[i] for i in range(4)]))
with torch.no_grad():
    _, sb = bv(b2b)
per = _biaffine_tag_probs(bv.biaffine, sb, b2b["attention_mask"], bio)
for p, n in zip(per, b2b["attention_mask"].sum(1).tolist()):
    assert p.shape == (n, L) and torch.allclose(p.sum(1), torch.ones(n), atol=1e-2)
print("post-hoc decoding rules ok")

print("SMOKE OK")

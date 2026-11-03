"""Systems built from saved checkpoints without further training.

Every model, whatever its slot head or tag scheme, is converted into per-utterance outputs:
  intent    log-probabilities over the 60 intents (deep and sparse parts kept apart when present)
  tags      per-token log-probabilities over the BIO labels
  presence  optional probabilities that each slot type occurs in the utterance
Those can be averaged across models and decoded in two ways:
  viterbi   best legal tag path under hard BIO rules
  span      span posteriors, thresholded and chosen as a non-overlapping set (the metric counts exact
            spans, and the best threshold for F1 is about half the attainable F1; Lipton et al. 2014),
            optionally blended with a train-only mention table and a soft slot-type presence filter
All tables come from train. All tuning uses dev. Test outputs need allow_test=True (final script only).
"""
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import torch

from .config import ModelConfig, CKPT_DIR, RESULTS_DIR, DEVICE, PARAM_CAP
from .data import make_collate, to_bio, parse_annotated_utterance
from .evaluate import to_device, metrics_from_frame
from .models import build_model, viterbi


def load_member(ctx, run_name):
    ck = torch.load(CKPT_DIR / f"{run_name}.pt", map_location="cpu")
    assert ck["vocab"]["word_vocab"] == ctx.assets.word_vocab
    assert ck["vocab"]["slot_labels"] == ctx.assets.slot_labels
    assert ck["vocab"]["intent_labels"] == ctx.assets.intent_labels
    cfg = ModelConfig.from_dict(ck["model_config"])
    model = build_model(cfg, ctx.assets)
    model.load_state_dict(ck["model_state_dict"])
    return model.to(DEVICE).eval(), ck


def crf_log_marginals(em, mask, trans, start, end):
    """Forward-backward. em B,T,K; returns per-token log-marginals B,T,K."""
    B, T, K = em.shape
    alphas = [start.unsqueeze(0) + em[:, 0]]
    for t in range(1, T):
        nxt = torch.logsumexp(alphas[-1].unsqueeze(2) + trans.unsqueeze(0), dim=1) + em[:, t]
        alphas.append(torch.where(mask[:, t].unsqueeze(1), nxt, alphas[-1]))
    betas = [None] * T
    betas[T - 1] = end.unsqueeze(0).expand(B, K)
    for t in range(T - 2, -1, -1):
        step = torch.logsumexp(trans.unsqueeze(0) + (em[:, t + 1] + betas[t + 1]).unsqueeze(1), dim=2)
        betas[t] = torch.where(mask[:, t + 1].unsqueeze(1), step, end.unsqueeze(0).expand(B, K))
    return torch.log_softmax(torch.stack(alphas, 1) + torch.stack(betas, 1), -1)


def _biaffine_tag_probs(head, logits, mask, bio_labels):
    """Approximate token-level BIO probabilities from span probabilities."""
    index = {t: i for i, t in enumerate(bio_labels)}
    probs = torch.softmax(logits, -1).cpu()
    out = []
    for b, n in enumerate(mask.long().sum(1).tolist()):
        p = torch.triu(probs[b, :n, :n, :head.none].permute(2, 0, 1)).permute(1, 2, 0)  # n,n,types
        tags = torch.zeros(n, len(bio_labels))
        for ty, name in enumerate(head.types):
            bi, ii = index.get("B-" + name), index.get("I-" + name)
            for t in range(n):
                if bi is not None:
                    tags[t, bi] += p[t, t:, ty].sum()
                if t > 0:
                    tags[t, ii if ii is not None else bi] += p[:t, t:, ty].sum()
        tags[:, 0] = (1 - tags.sum(1)).clamp_min(1e-4)
        out.append(tags / tags.sum(1, keepdim=True))
    return out


@torch.no_grad()
def collect(ctx, run_name, split, allow_test=False, mc_passes=0, seed=0):
    """Returns {id: outputs} in BIO label space. Plain forward passes are cached on disk."""
    if split == "test" and not allow_test:
        raise RuntimeError("test outputs are only available to the final evaluation")
    cache = RESULTS_DIR / run_name / f"outputs_{split}_v2.pt"    # v2: both intent paths kept per pass
    if mc_passes == 0 and cache.exists():
        return torch.load(cache, map_location="cpu")
    model, ck = load_member(ctx, run_name)
    cfg = model.cfg
    ds = ctx.datasets(cfg, ck["meta"].get("k_extra"), splits=(split,))[split]
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, collate_fn=make_collate(ctx.assets))

    bio = ctx.assets.slot_labels
    bio_index = {t: i for i, t in enumerate(bio)}
    src2bio = []    # collapse map from the head's own tag space into BIO labels
    for t in model.id2tag:
        tb = to_bio(t)
        src2bio.append(bio_index.get(tb, bio_index.get("B-" + tb[2:], 0)))
    src2bio = torch.tensor(src2bio)

    if mc_passes:
        torch.manual_seed(seed)
        model.train()           # dropout on in every sub-module; there is no batch norm in these models
        model.training = False  # the model itself keeps taking its evaluation path
    out = {}
    for raw in loader:
        b = to_device(raw)
        mask = b["attention_mask"]
        passes = max(1, mc_passes)
        intent_acc, tag_acc, pres_acc = 0.0, None, 0.0
        deep, wide = [], []     # both intent paths per pass, so that any intent mix can be applied afterwards
        for _ in range(passes):
            il, sl = model(b)
            intent_acc = intent_acc + torch.softmax(il, -1).cpu()
            if model._wide_logp is not None:
                deep.append(model._deep_logp.cpu())
                wide.append(model._wide_logp.cpu())
            if model._presence is not None:
                pres_acc = pres_acc + torch.sigmoid(model._presence).cpu()
            if cfg.slot_head == "biaffine":
                per = _biaffine_tag_probs(model.biaffine, sl, mask, bio)
                tags = torch.zeros(mask.size(0), mask.size(1), len(bio))
                for i, p in enumerate(per):
                    tags[i, :p.size(0)] = p
            else:
                if cfg.slot_head == "crf":
                    trans, start, end = model.crf.scores()
                    logm = crf_log_marginals(sl, mask, trans, start, end)
                else:
                    logm = torch.log_softmax(sl, -1)
                tags = torch.zeros(*logm.shape[:2], len(bio)).index_add_(2, src2bio, logm.exp().cpu())
            tag_acc = tags if tag_acc is None else tag_acc + tags
        deep = torch.stack(deep, 1) if deep else None     # B, passes, intents
        wide = torch.stack(wide, 1) if wide else None
        intent_lp = torch.log((intent_acc / passes).clamp_min(1e-8)).float()
        tag_lp = torch.log((tag_acc / passes).clamp_min(1e-8)).float()
        for i, n in enumerate(mask.long().sum(1).tolist()):
            out[raw["ids"][i]] = {
                "intent": intent_lp[i],
                "deep": deep[i].float() if deep is not None else None,
                "wide": wide[i].float() if wide is not None else None,
                "tags": tag_lp[i, :n].clone(),
                "presence": (pres_acc[i] / passes).float() if torch.is_tensor(pres_acc) else None,
            }
    del model
    torch.cuda.empty_cache()
    if mc_passes == 0:
        torch.save(out, cache)
    return out


@dataclass
class Decode:
    method: str = "viterbi"     # "viterbi" | "span"
    tau: float = 0.4            # span threshold
    gm: float = 0.0             # weight of the train-only mention table
    eta: float = 0.0            # exponent of the soft slot-type presence filter
    gamma: float = 0.0          # weight of the intent-to-tag PMI prior
    wide_mix: float = -1.0      # deep share of the intent mix; -1 keeps the trained value
    avg: str = "prob"           # how members are averaged: "prob" or "log"
    mask_beta: float = 0.0      # viterbi only: penalty on slot types never seen with the intent
    top_k: int = 1              # viterbi only: number of intents considered jointly
    lam: float = 0.0            # viterbi only: weight of the slot path score against intent


@dataclass
class System:
    name: str
    members: list
    decode: Decode = field(default_factory=Decode)
    flagged: bool = False
    mc: int = 0                 # MC-dropout passes per member; 0 = plain forward pass

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], list(d["members"]), Decode(**d["decode"]), d.get("flagged", False), d.get("mc", 0))


def _lse(x, axis):
    m = x.max(axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True)), axis=axis)


class Decoder:
    """Averaging and decoding over BIO log-probabilities. Tables are built from train only."""

    def __init__(self, assets):
        ts = assets.tagset("bio")
        self.assets = assets
        self.labels = ts.labels
        K = len(ts.labels)
        self.trans = (~torch.as_tensor(ts.allowed)).float() * -1e4
        self.start = (~torch.as_tensor(ts.allowed_start)).float() * -1e4
        self.end = torch.zeros(K)
        self.M, self.S = self.trans.numpy().astype(np.float64), self.start.numpy().astype(np.float64)
        self.types = assets.slot_types
        idx = {t: i for i, t in enumerate(ts.labels)}
        self.B = [idx["B-" + t] if "B-" + t in idx else -1 for t in self.types]
        self.I = [idx.get("I-" + t, -1) for t in self.types]
        self.cont = np.full(K, -1)
        for j, t in enumerate(ts.labels):
            if t != "O":
                self.cont[j] = idx.get("I-" + t[2:], -1)
        self.cont_rows = np.where(self.cont >= 0)[0]
        self.intent_ok = torch.zeros(len(assets.intent_labels), K, dtype=torch.bool)
        self.intent_ok[:, 0] = True
        for i, name in enumerate(assets.intent_labels):
            seen = assets.intent_slot_types.get(name, set())
            for j, t in enumerate(ts.labels):
                if t != "O" and t[2:] in seen:
                    self.intent_ok[i, j] = True
        self.pmi = torch.tensor(assets.intent_tag_pmi(), dtype=torch.float32)
        self._marginals = {}

    def table_entries(self):
        ng_all, ng_type, _ = self.assets.mention_tables()
        return len(ng_all) + len(ng_type)

    # ------------------------------------------------------------ averaging
    def combine(self, outputs, spec):
        ids = list(outputs[0].keys())
        intents, tags, pres = [], [], []
        for i in ids:
            members = [o[i] for o in outputs]
            lps = []
            for m in members:
                if m["deep"] is not None and spec.wide_mix >= 0:
                    # the mix is applied to every pass and renormalised, as the model does when it predicts
                    mixed = spec.wide_mix * m["deep"] + (1 - spec.wide_mix) * m["wide"]
                    lps.append(torch.log(torch.softmax(mixed, -1).mean(0).clamp_min(1e-8)))
                else:
                    lps.append(m["intent"])
            if spec.avg == "log":
                il = torch.stack(lps).mean(0)
                tl = torch.stack([m["tags"] for m in members]).mean(0)
            else:
                il = torch.log(torch.stack([x.exp() for x in lps]).mean(0).clamp_min(1e-8))
                tl = torch.log(torch.stack([m["tags"].exp() for m in members]).mean(0).clamp_min(1e-8))
            intents.append(il)
            tags.append(tl)
            ps = [m["presence"] for m in members if m["presence"] is not None]
            pres.append(torch.stack(ps).mean(0) if ps else None)
        return ids, intents, tags, pres

    # ------------------------------------------------------------ viterbi
    def _decode_batch(self, ems):
        """ems: list of T_i,K tensors. Returns (paths, path scores)."""
        order = sorted(range(len(ems)), key=lambda i: ems[i].size(0))
        paths, scores = [None] * len(ems), [0.0] * len(ems)
        for s in range(0, len(order), 256):
            idx = order[s:s + 256]
            T = max(ems[i].size(0) for i in idx)
            em = torch.zeros(len(idx), T, len(self.labels))
            mask = torch.zeros(len(idx), T, dtype=torch.bool)
            for j, i in enumerate(idx):
                n = ems[i].size(0)
                em[j, :n] = ems[i]
                mask[j, :n] = True
            for j, p in zip(idx, viterbi(em, mask, self.trans, self.start, self.end)):
                paths[j] = p
                scores[j] = float(ems[j][torch.arange(len(p)), torch.tensor(p)].sum())
        return paths, scores

    def _viterbi(self, ids, intents, tags, spec):
        cand_i = [torch.topk(il, spec.top_k).indices.tolist() for il in intents]
        ems, owner = [], []
        for u, (tl, cands) in enumerate(zip(tags, cand_i)):
            for c in cands:
                em = tl
                if spec.gamma:
                    em = em + spec.gamma * self.pmi[c]
                if spec.mask_beta > 0:
                    em = em - spec.mask_beta * (~self.intent_ok[c]).float()
                ems.append(em)
                owner.append((u, c))
        paths, scores = self._decode_batch(ems)
        best = {}
        for (u, c), p, sc in zip(owner, paths, scores):
            total = float(intents[u][c]) + spec.lam * sc / max(len(p), 1)
            if u not in best or total > best[u][0]:
                best[u] = (total, c, p)
        return {ids[u]: (best[u][1], [self.labels[x] for x in best[u][2]]) for u in range(len(ids))}

    # ------------------------------------------------------------ span posteriors
    def span_marginals(self, em):
        """Posterior of every typed span with non-trivial mass. em: T,K log-probabilities."""
        M, K, n = self.M, em.shape[1], em.shape[0]
        a = np.zeros((n, K))
        a[0] = self.S + em[0]
        for t in range(1, n):
            a[t] = _lse(a[t - 1][:, None] + M, 0) + em[t]
        b = np.zeros((n, K))
        for t in range(n - 2, -1, -1):
            b[t] = _lse(M + (em[t + 1] + b[t + 1])[None, :], 1)
        z = _lse(a[-1], 0)
        close = np.zeros((n, K))    # log-mass of ending the span right after this token
        for j in range(n - 1):
            c = M + (em[j + 1] + b[j + 1])[None, :]
            c[self.cont_rows, self.cont[self.cont_rows]] = -np.inf
            close[j] = _lse(c, 1)
        gam = a + b - z
        out = {}
        lim, stop = np.log(0.02), np.log(1e-3)
        for ty, (bi, ii) in enumerate(zip(self.B, self.I)):
            if bi < 0:
                continue
            for i in np.where(gam[:, bi] > lim)[0]:
                cur, tag = a[i, bi], bi
                for j in range(int(i), n):
                    lp = cur + close[j, tag] - z
                    if lp > lim:
                        out[(ty, int(i), j + 1)] = float(np.exp(lp))
                    if ii < 0 or j + 1 >= n:
                        break
                    cur = cur + M[tag, ii] + em[j + 1, ii]
                    tag = ii
                    if cur - z < stop:
                        break
        return out

    def best_set(self, marg, n, tau, low, gm, q, eta, tables):
        """Non-overlapping set of spans maximising the summed (score - tau), by dynamic programming."""
        by_end = {}
        for (ty, s, e), pv in marg.items():
            if gm:
                g = tuple(low[s:e])
                c = tables[0].get(g, 0)
                if c:
                    pv = (1 - gm) * pv + gm * tables[1].get((self.types[ty], g), 0) / c
            if eta and q is not None:
                pv = pv * float(q[ty]) ** eta
            if pv > tau:
                by_end.setdefault(e, []).append((s, ty, pv - tau))
        dp, choice = [0.0] * (n + 1), [None] * (n + 1)
        for e in range(1, n + 1):
            dp[e] = dp[e - 1]
            for s, ty, w in by_end.get(e, ()):
                if dp[s] + w > dp[e]:
                    dp[e], choice[e] = dp[s] + w, (s, ty)
        tags, e = ["O"] * n, n
        while e > 0:
            c = choice[e]
            if c is None:
                e -= 1
            else:
                s, ty = c
                tags[s] = "B-" + self.types[ty]
                for x in range(s + 1, e):
                    tags[x] = "I-" + self.types[ty]
                e = s
        return tags

    def _span(self, ids, intents, tags, pres, spec, lows, key):
        tables = self.assets.mention_tables() if spec.gm else (None, None, None)
        # the PMI prior depends on the predicted intent, so a different intent mix needs new marginals
        mkey = (key, spec.gamma, spec.wide_mix if spec.gamma else None)
        if key is None or mkey not in self._marginals:
            margs = []
            for il, tl in zip(intents, tags):
                em = tl + spec.gamma * self.pmi[int(il.argmax())] if spec.gamma else tl
                margs.append(self.span_marginals(em.numpy().astype(np.float64)))
            if key is not None:
                self._marginals[mkey] = margs
        margs = margs if key is None or mkey not in self._marginals else self._marginals[mkey]
        out = {}
        for u, i in enumerate(ids):
            low = lows[i]
            out[i] = (int(intents[u].argmax()),
                      self.best_set(margs[u], len(low), spec.tau, low, spec.gm, pres[u], spec.eta, tables))
        return out

    def decode(self, outputs, spec, lows=None, key=None):
        """outputs: list of member dicts. Returns id -> (intent id, BIO strings)."""
        ids, intents, tags, pres = self.combine(outputs, spec)
        if spec.method == "span":
            return self._span(ids, intents, tags, pres, spec, lows, key)
        return self._viterbi(ids, intents, tags, spec)


def gold_frame(ctx, split):
    rows = []
    for r in ctx.assets.split(split).itertuples(index=False):
        parsed = parse_annotated_utterance(r.annot_utt)
        rows.append({"id": r.id, "gold_intent_id": ctx.assets.intent2id[r.intent], "gold_tags": parsed["bio"],
                     "low": [t.casefold() for t in parsed["tokens"]]})
    return pd.DataFrame(rows)


def frame_from_decode(decoded, gold):
    rows = []
    for r in gold.itertuples(index=False):
        ci, tg = decoded[r.id]
        rows.append({"id": r.id, "gold_intent_id": r.gold_intent_id, "pred_intent_id": ci,
                     "gold_tags": r.gold_tags, "pred_tags": tg})
    return pd.DataFrame(rows)


def score_system(ctx, system, split, gold, decoder, cache, allow_test=False):
    outs = []
    for m in system.members:
        key = (m, split, system.mc)
        if key not in cache:
            cache[key] = collect(ctx, m, split, allow_test=allow_test, mc_passes=system.mc)
        outs.append(cache[key])
    lows = dict(zip(gold["id"], gold["low"]))
    mkey = (tuple(system.members), split, system.mc, system.decode.avg)
    frame = frame_from_decode(decoder.decode(outs, system.decode, lows, key=mkey), gold)
    return metrics_from_frame(frame), frame


def system_params(lb, members, decode=None, decoder=None):
    """Stored values of a system. Every member is its own checkpoint, so each one counts in full: its trainable
    weights, its frozen Word2Vec table and any anchor copy. Decode-time tables count once."""
    rows = [lb[lb["run"] == m].iloc[0] for m in members]
    total = int(sum(r["params_total"] for r in rows))
    if decode is not None and decoder is not None:
        if decode.gm:
            total += decoder.table_entries()
        if decode.gamma:
            total += int(decoder.pmi.numel())
    return total


def stored_values(members, decode, decoder):
    """The cap's basis: every numeric value a saved system holds at inference. Each member's checkpoint counts in full,
    buffers included (positional tables, CRF masks). TF-IDF idf vectors count once per feature size, the BIO transition
    tables once unless CRF masks already hold them, and the decode tables that the decode uses. Vocabularies and lexicon
    keys are dictionaries, not counted."""
    total, sizes, gaz, bio_needed = 0, set(), False, False
    for m in members:
        ck = torch.load(CKPT_DIR / f"{m}.pt", map_location="cpu")
        cfg = ck["model_config"]
        total += sum(v.numel() for v in ck["model_state_dict"].values())
        # a CRF checkpoint already stores the BIO constraints as masks, so they are counted with it
        bio_needed = bio_needed or not any(k.endswith("trans_mask") for k in ck["model_state_dict"])
        if cfg.get("wide_intent"):
            sizes.add(cfg["wide_intent"])
        gaz = gaz or bool(cfg.get("gaz"))
    for k in sizes:
        word, char = decoder.assets.wide_vectorizers(k)
        total += len(word.idf_) + len(char.idf_)
    if bio_needed:
        total += decoder.trans.numel() + decoder.start.numel() + decoder.end.numel()
    if decode.gm or gaz:
        total += decoder.table_entries()
    if decode.gamma:
        total += int(decoder.pmi.numel())
    return total


def greedy_ensemble(ctx, candidates, lb, gold, decoder, cache, base_decode=None, budget=PARAM_CAP, max_members=8, log=print):
    """Forward selection on dev: add the member that most improves the joint score, within the cap."""
    decode = base_decode or Decode()
    scored = []
    for m in candidates:
        met, _ = score_system(ctx, System(m, [m], decode), "dev", gold, decoder, cache)
        scored.append((met["joint_score"], m))
    scored.sort(reverse=True)
    members = [scored[0][1]]
    best = scored[0][0]
    log(f"ensemble start {members[0]}: dev joint {best:.2f}")
    trail = [{"members": list(members), "joint": best}]
    while len(members) < max_members:
        gain, pick = TIE_MARGIN, None      # a member must add more than a tie on dev
        for m in candidates:
            if m in members or system_params(lb, members + [m]) > budget:
                continue
            met, _ = score_system(ctx, System("t", members + [m], decode), "dev", gold, decoder, cache)
            if met["joint_score"] - best > gain:
                gain, pick = met["joint_score"] - best, m
        if pick is None:
            break
        members.append(pick)
        best += gain
        trail.append({"members": list(members), "joint": best, "params": system_params(lb, members)})
        log(f"ensemble + {pick}: dev joint {best:.2f}, params {system_params(lb, members):,}")
    return members, best, trail


def tune_decode(ctx, members, gold, decoder, cache, mc=0, log=print, tables=True):
    """Stepwise dev tuning of a few scalars. Returns the best Decode and the whole table."""
    rows = []

    def trial(d):
        met, _ = score_system(ctx, System("t", list(members), d, mc=mc), "dev", gold, decoder, cache)
        rows.append({**asdict(d), "intent": met["intent_accuracy"], "slot": met["slot_micro_f1"], "joint": met["joint_score"]})
        return met

    def replace_d(d, **kw):
        return Decode(**{**asdict(d), **kw})

    cur = Decode()
    trial(cur)
    has_wide = any(v["deep"] is not None for v in list(cache[(members[0], "dev", mc)].values())[:1])
    if has_wide:    # keep at least 35% on the deep path, so that one joint model stays the decision maker
        cur = max((replace_d(cur, wide_mix=a) for a in (0.35, 0.5, 0.65, 0.8)),
                  key=lambda d: trial(d)["intent_accuracy"])
    if len(members) > 1:
        log_avg = replace_d(cur, avg="log")
        if trial(log_avg)["joint_score"] > trial(cur)["joint_score"]:
            cur = log_avg
    for g in (0.1, 0.3):
        cand = replace_d(cur, gamma=g)
        if trial(cand)["slot_micro_f1"] > trial(cur)["slot_micro_f1"]:
            cur = cand
    for b in (1.0, 3.0, 10.0):      # hard mask, kept for the record
        trial(replace_d(cur, mask_beta=b))
    for b, l in ((0.0, 1.0), (3.0, 1.0), (3.0, 3.0)):
        trial(replace_d(cur, mask_beta=b, top_k=3, lam=l))
    viterbi_slot = trial(cur)["slot_micro_f1"]

    taus = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)  # 0.5 was the old edge
    span = max((replace_d(cur, method="span", tau=t) for t in taus),
               key=lambda d: trial(d)["slot_micro_f1"])
    best_gm = span
    for gm in ((0.2, 0.4, 0.6) if tables else ()):
        for t in taus:
            cand = replace_d(span, gm=gm, tau=t)
            if trial(cand)["slot_micro_f1"] > trial(best_gm)["slot_micro_f1"]:
                best_gm = cand
    has_pres = any(v["presence"] is not None for v in list(cache[(members[0], "dev", mc)].values())[:1])
    best_all = best_gm
    if has_pres:
        for eta in (0.3, 0.5, 0.8):
            for t in taus:
                cand = replace_d(best_gm, eta=eta, tau=t)
                if trial(cand)["slot_micro_f1"] > trial(best_all)["slot_micro_f1"]:
                    best_all = cand
    if trial(cur)["slot_micro_f1"] > trial(best_all)["slot_micro_f1"]:
        best_all = cur      # plain Viterbi scored higher: keep it, the span decoder is not always the better one
    table = pd.DataFrame(rows).drop_duplicates().sort_values("joint", ascending=False)
    log(table.round(2).head(10).to_string(index=False))
    log(f"slot F1, viterbi {viterbi_slot:.2f}, best decode {trial(best_all)['slot_micro_f1']:.2f}")
    return best_all, table


TIE_MARGIN = 0.8    # dev joint points: the one-seed tie margin of the protocol


def tuned_decode(ctx, members, gold, decoder, cache, mc=0):
    """tune_decode, except that a mention table stays in the decode only if it beats the best table-free decode by TIE_MARGIN."""
    dec, table = tune_decode(ctx, members, gold, decoder, cache, mc=mc)
    if not dec.gm:
        return dec, table
    free, free_table = tune_decode(ctx, members, gold, decoder, cache, mc=mc, tables=False)

    def joint(d):
        return score_system(ctx, System("t", list(members), d, mc=mc), "dev", gold, decoder, cache)[0]["joint_score"]

    if joint(dec) - joint(free) < TIE_MARGIN:
        return free, free_table
    return dec, table

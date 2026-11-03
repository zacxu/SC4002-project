"""Joint intent + slot models.

`JointModel(ModelConfig())` is the hand-off Stage-1 Transformer, parameter-for-parameter
(module names embedding / proj / pos / encoder / intent_head / slot_head).
Every optional output module starts at zero, so a model warm-started from the Stage-1 checkpoint
reproduces Stage 1 exactly before training. The exceptions change decoding or lookups by design:
Viterbi decoding, the CRF (hard tag constraints), BIOES and the biaffine head (different outputs),
the extended vocabulary (unseen words stop being <UNK>) and the sparse paths (mixed at evaluation).
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MAX_LEN, W2V_DIM, PARAM_CAP
from .data import to_bio, N_LEX_FEATURES, gold_spans


def count_params(model):
    """Total counts every stored model value: a frozen Word2Vec copy kept as a buffer is included."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if hasattr(model, "emb0"):
        total += model.emb0.numel()
    return total, trainable


def zero_last_linear(seq):
    nn.init.zeros_(seq[-1].weight)
    if seq[-1].bias is not None:
        nn.init.zeros_(seq[-1].bias)


def zero_linear(m):
    nn.init.zeros_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ---------------------------------------------------------------- subword adapters
ADAPTER_HIDDEN = 64


class BPEAdapter(nn.Module):
    """Train-only BPE pieces (Sennrich et al. 2016), mean-pooled per token, added as a residual."""

    def __init__(self, assets, d_model, emb_dim=64):
        super().__init__()
        self.pad = assets.BPE_PAD_ID
        self.embedding = nn.Embedding(assets.sp.get_piece_size(), emb_dim, padding_idx=self.pad)
        self.proj = nn.Sequential(nn.Linear(emb_dim, ADAPTER_HIDDEN), nn.GELU(), nn.Linear(ADAPTER_HIDDEN, d_model))
        zero_last_linear(self.proj)

    def forward(self, batch):
        ids = batch["bpe_ids"]
        m = (ids != self.pad).unsqueeze(-1).float()
        return self.proj((self.embedding(ids) * m).sum(2) / m.sum(2).clamp_min(1.0))


class NGramAdapter(nn.Module):
    """Character 3-5-grams (Bojanowski et al. 2017), mean-pooled per token, added as a residual."""

    def __init__(self, assets, d_model, emb_dim=32):
        super().__init__()
        self.pad = assets.NGRAM_PAD_ID
        self.embedding = nn.Embedding(len(assets.ngram_vocab), emb_dim, padding_idx=self.pad)
        self.proj = nn.Sequential(nn.Linear(emb_dim, ADAPTER_HIDDEN), nn.GELU(), nn.Linear(ADAPTER_HIDDEN, d_model))
        zero_last_linear(self.proj)

    def forward(self, batch):
        ids = batch["ngram_ids"]
        m = (ids != self.pad).unsqueeze(-1).float()
        return self.proj((self.embedding(ids) * m).sum(2) / m.sum(2).clamp_min(1.0))


class CharBiLSTMAdapter(nn.Module):
    """Character BiLSTM word composition (Ling et al. 2015), added as a residual."""

    def __init__(self, assets, d_model, emb_dim=24, hidden=32):
        super().__init__()
        self.embedding = nn.Embedding(len(assets.char_vocab), emb_dim, padding_idx=assets.CHAR_PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(2 * hidden, ADAPTER_HIDDEN), nn.GELU(), nn.Linear(ADAPTER_HIDDEN, d_model))
        zero_last_linear(self.proj)

    def forward(self, batch):
        ids, lengths = batch["char_ids"], batch["char_lengths"]
        B, T, C = ids.shape
        x = self.embedding(ids.reshape(B * T, C))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.reshape(-1).clamp_min(1).cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.proj(torch.cat([h[-2], h[-1]], -1).reshape(B, T, -1))


ADAPTERS = {"bpe": BPEAdapter, "ngram": NGramAdapter, "char": CharBiLSTMAdapter}


# ---------------------------------------------------------------- CRF
class CRF(nn.Module):
    """Linear-chain CRF with hard tag constraints (Lample et al. 2016)."""

    NEG = -1e4

    def __init__(self, num_tags, allowed, allowed_start, allowed_end=None):
        super().__init__()
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))  # [from, to]
        self.start = nn.Parameter(torch.zeros(num_tags))
        self.end = nn.Parameter(torch.zeros(num_tags))
        if allowed_end is None:
            allowed_end = [True] * num_tags
        self.register_buffer("trans_mask", (~torch.as_tensor(allowed)).float() * self.NEG)
        self.register_buffer("start_mask", (~torch.as_tensor(allowed_start)).float() * self.NEG)
        self.register_buffer("end_mask", (~torch.as_tensor(allowed_end)).float() * self.NEG)

    def scores(self):
        return self.transitions + self.trans_mask, self.start + self.start_mask, self.end + self.end_mask

    def nll(self, emissions, tags, mask):
        """Summed negative log-likelihood. emissions B,T,K; tags B,T (-100 pad); mask B,T bool."""
        trans, start, end = self.scores()
        tags = tags.clamp_min(0)
        B, T, K = emissions.shape
        maskf = mask.float()
        # gold path score
        score = start[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for t in range(1, T):
            s = trans[tags[:, t - 1], tags[:, t]] + emissions[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            score = score + s * maskf[:, t]
        lengths = mask.long().sum(1)
        last = tags.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
        score = score + end[last]
        # partition function
        alpha = start.unsqueeze(0) + emissions[:, 0]
        for t in range(1, T):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + trans.unsqueeze(0), dim=1) + emissions[:, t]
            alpha = torch.where(mask[:, t].unsqueeze(1), nxt, alpha)
        logz = torch.logsumexp(alpha + end.unsqueeze(0), dim=1)
        return (logz - score).sum()

    @torch.no_grad()
    def decode(self, emissions, mask):
        trans, start, end = self.scores()
        return viterbi(emissions, mask, trans, start, end)


@torch.no_grad()
def viterbi(emissions, mask, trans, start, end):
    """Best tag path per utterance. Returns python lists of tag ids."""
    B, T, K = emissions.shape
    score = start.unsqueeze(0) + emissions[:, 0]
    history = []
    for t in range(1, T):
        best, idx = (score.unsqueeze(2) + trans.unsqueeze(0)).max(1)  # B, to
        m = mask[:, t].unsqueeze(1)
        score = torch.where(m, best + emissions[:, t], score)
        history.append(torch.where(m, idx, torch.arange(K, device=idx.device).expand(B, K)))
    best_last = (score + end.unsqueeze(0)).argmax(1).cpu()
    lengths = mask.long().sum(1).tolist()
    hist = torch.stack(history, 1).cpu() if history else None
    paths = []
    for b in range(B):
        tag = int(best_last[b])
        path = [tag]
        for t in range(lengths[b] - 1, 0, -1):
            tag = int(hist[b, t - 1, tag])
            path.append(tag)
        paths.append(path[::-1])
    return paths


# ---------------------------------------------------------------- biaffine span head
class BiaffineSpanHead(nn.Module):
    """Scores every (start, end) token pair with a slot type or 'none' (Yu et al. 2020)."""

    def __init__(self, h, types, mlp=128, dropout=0.2):
        super().__init__()
        self.types = list(types)
        self.type2id = {t: i for i, t in enumerate(self.types)}
        self.none = len(self.types)
        k = self.none + 1
        self.start_mlp = nn.Sequential(nn.Linear(h, mlp), nn.GELU(), nn.Dropout(dropout))
        self.end_mlp = nn.Sequential(nn.Linear(h, mlp), nn.GELU(), nn.Dropout(dropout))
        self.U = nn.Parameter(torch.empty(mlp, k, mlp))
        nn.init.xavier_uniform_(self.U)
        self.W = nn.Linear(2 * mlp, k)

    def forward(self, x):
        s, e = self.start_mlp(x), self.end_mlp(x)
        T = x.size(1)
        bil = torch.einsum("bim,mkn,bjn->bijk", s, self.U, e)
        pair = torch.cat([s.unsqueeze(2).expand(-1, -1, T, -1), e.unsqueeze(1).expand(-1, T, -1, -1)], -1)
        return bil + self.W(pair)

    def gold(self, slot_ids, mask, labels):
        """Gold span matrix B,T,T (none everywhere, type id at each gold (start, end))."""
        B, T = slot_ids.shape
        gold = torch.full((B, T, T), self.none, dtype=torch.long)
        rows = slot_ids.cpu().tolist()
        for b, n in enumerate(mask.long().sum(1).tolist()):
            for typ, s, e in gold_spans([labels[i] for i in rows[b][:n]]):
                gold[b, s, e - 1] = self.type2id[typ]
        return gold.to(slot_ids.device)

    def loss(self, logits, slot_ids, mask, labels, boundary_eps=0.0):
        """Span cross-entropy without label smoothing, which hurts span heads (Zhu & Li 2022).
        boundary_eps moves that share of each gold span's mass to spans one boundary step away."""
        gold = self.gold(slot_ids, mask, labels)
        valid = torch.triu((mask.unsqueeze(1) & mask.unsqueeze(2)).long()).bool()   # start <= end
        if boundary_eps <= 0:
            return F.cross_entropy(logits[valid], gold[valid], reduction="sum") / mask.sum().clamp_min(1)
        soft = F.one_hot(gold, self.none + 1).float()
        lengths = mask.long().sum(1).tolist()
        for b, i, j in (gold != self.none).nonzero().tolist():
            ty = int(gold[b, i, j])
            near = [(a, c) for a, c in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
                    if 0 <= a <= c < lengths[b] and gold[b, a, c] == self.none]
            if not near:
                continue
            soft[b, i, j, ty] = 1.0 - boundary_eps
            for a, c in near:
                soft[b, a, c, ty] += boundary_eps / len(near)
                soft[b, a, c, self.none] -= boundary_eps / len(near)
        return -(soft[valid] * F.log_softmax(logits[valid], -1)).sum() / mask.sum().clamp_min(1)

    @torch.no_grad()
    def decode(self, logits, mask):
        """Greedy highest-score-first selection of non-overlapping spans, as BIO strings."""
        probs = torch.softmax(logits, -1)
        conf, pred = probs.max(-1)
        conf, pred = conf.cpu(), pred.cpu()
        out = []
        for b, n in enumerate(mask.long().sum(1).tolist()):
            hits = (pred[b, :n, :n] != self.none).nonzero().tolist()
            cands = sorted(((float(conf[b, i, j]), i, j, int(pred[b, i, j])) for i, j in hits if j >= i), reverse=True)
            tags, used = ["O"] * n, [False] * n
            for _, i, j, ty in cands:
                if any(used[i:j + 1]):
                    continue
                used[i:j + 1] = [True] * (j - i + 1)
                tags[i] = "B-" + self.types[ty]
                for x in range(i + 1, j + 1):
                    tags[x] = "I-" + self.types[ty]
            out.append(tags)
        return out


# ---------------------------------------------------------------- joint model
class JointModel(nn.Module):
    REL_RANGE = 8

    def __init__(self, cfg, assets, embedding_matrix=None, frozen_matrix=None, frozen_map_init=None):
        super().__init__()
        self.cfg = cfg
        V = len(assets.word_vocab)
        self.base_vocab = V
        self.n_intents = len(assets.intent_labels)
        tagset = assets.tagset(cfg.tag_scheme)
        self.id2tag = list(tagset.labels)
        self.n_slots = len(self.id2tag)
        d = cfg.d_model
        self._track_emb = False
        self._emb = None
        self._wide_dropout = 0.0

        self.embedding = nn.Embedding(V, W2V_DIM, padding_idx=assets.PAD_ID)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.as_tensor(embedding_matrix))
        if cfg.anchor or cfg.dual_emb:
            self.register_buffer("emb0", self.embedding.weight.detach().clone())

        if cfg.n_frozen > 0:
            self.frozen_emb = nn.Embedding(cfg.n_frozen, cfg.frozen_dim)
            if frozen_matrix is not None:
                self.frozen_emb.weight.data.copy_(torch.as_tensor(frozen_matrix))
            self.frozen_emb.weight.requires_grad_(False)
            if cfg.frozen_map or cfg.frozen_dim != W2V_DIM:
                self.frozen_map = nn.Linear(cfg.frozen_dim, W2V_DIM)
                if frozen_map_init is not None:     # PCA inverse transform
                    self.frozen_map.weight.data.copy_(torch.as_tensor(frozen_map_init[0]))  # 300 x dim
                    self.frozen_map.bias.data.copy_(torch.as_tensor(frozen_map_init[1]))
                else:
                    nn.init.eye_(self.frozen_map.weight)
                    nn.init.zeros_(self.frozen_map.bias)

        self.proj = nn.Linear(W2V_DIM, d)
        self.adapters = nn.ModuleDict({a: ADAPTERS[a](assets, d) for a in cfg.adapters})
        if cfg.gaz:
            self.gaz_type = nn.Embedding(1 + 4 * len(assets.slot_types), 32, padding_idx=0)
            self.gaz_bucket = nn.Embedding(6, 8, padding_idx=0)
            self.gaz_out = nn.Linear(40, d)
            zero_linear(self.gaz_out)

        if cfg.encoder == "transformer":
            self.pos = PositionalEncoding(d)
            if cfg.pre_lstm:
                self.pre_rnn = nn.LSTM(d, d // 2, batch_first=True, bidirectional=True)
                self.pre_out = nn.Linear(d // 2 * 2, d)
                zero_linear(self.pre_out)
            if cfg.rel_bias:
                self.rel_bias = nn.Parameter(torch.zeros(cfg.n_heads, 2 * self.REL_RANGE + 1))
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff, dropout=cfg.dropout,
                activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers, norm=nn.LayerNorm(d),
                                                 enable_nested_tensor=False)
            h = d
            if cfg.task_top:
                self.top_int = copy.deepcopy(self.encoder.layers[-1])
                self.norm_int = nn.LayerNorm(d)
            if cfg.refine:
                self.refine_layer = copy.deepcopy(layer)
        elif cfg.encoder == "bilstm":
            assert not (cfg.pre_lstm or cfg.rel_bias or cfg.task_top or cfg.refine), "Transformer-only switches"
            self.in_drop = nn.Dropout(cfg.dropout)
            self.lstm = nn.LSTM(d, cfg.lstm_hidden, num_layers=cfg.lstm_layers, batch_first=True,
                                bidirectional=True, dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0)
            self.out_drop = nn.Dropout(cfg.dropout)
            h = 2 * cfg.lstm_hidden
        else:
            raise ValueError(cfg.encoder)
        self.hidden = h

        # ---- intent head
        self.intent_head = nn.Linear(h, self.n_intents)
        if cfg.pooling == "meanmax":
            self.intent_max = nn.Linear(h, self.n_intents, bias=False)
            nn.init.zeros_(self.intent_max.weight)
        elif cfg.pooling == "attn":
            self.attn_query = nn.Parameter(torch.zeros(h))   # zero query == mean pooling
            self.attn_key = nn.Linear(h, h, bias=False)
        elif cfg.pooling != "mean":
            raise ValueError(cfg.pooling)
        if cfg.label_attn:
            self.la_key = nn.Linear(h, h, bias=False)
            self.la_query = nn.Parameter(0.02 * torch.randn(self.n_intents, h))
            self.la_out = nn.Parameter(torch.zeros(self.n_intents, h))
        if cfg.hier_intent:
            scen = sorted({x.split("_", 1)[0] for x in assets.intent_labels})
            self.n_scen = len(scen)
            self.scenario_head = nn.Linear(h, self.n_scen)
            zero_linear(self.scenario_head)
            self.register_buffer("intent2scen", torch.tensor(
                [scen.index(x.split("_", 1)[0]) for x in assets.intent_labels]))
        if cfg.wide_intent:
            self.wide_emb = nn.Embedding(assets.wide_size(cfg.wide_intent), self.n_intents, padding_idx=0)
            nn.init.zeros_(self.wide_emb.weight)
            self.wide_bias = nn.Parameter(torch.zeros(self.n_intents))
        if cfg.supcon_dim:
            self.supcon_proj = nn.Linear(h, cfg.supcon_dim)

        # ---- slot head
        if cfg.slot_head == "biaffine":
            assert cfg.tag_scheme == "bio" and not (cfg.factorized or cfg.slot_to_intent or cfg.intent_to_slot
                                                    or cfg.refine or cfg.coint or cfg.wide_slot_bits)
            self.biaffine = BiaffineSpanHead(h, assets.slot_types)
        else:
            self.slot_head = nn.Linear(h, self.n_slots)
        if cfg.interaction:
            self.slot_ctx = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, h))
            zero_last_linear(self.slot_ctx)
        if cfg.factorized:
            prefixes = sorted({t[0] for t in self.id2tag})     # O, B, I (and E, S under BIOES)
            types = sorted({t[2:] for t in self.id2tag if t != "O"})
            self.prefix_head = nn.Linear(h, len(prefixes))
            self.type_head = nn.Linear(h, len(types) + 1)      # last entry: no type (O)
            zero_linear(self.prefix_head)
            zero_linear(self.type_head)
            self.register_buffer("tag_prefix", torch.tensor([prefixes.index(t[0]) for t in self.id2tag]))
            self.register_buffer("tag_type", torch.tensor(
                [len(types) if t == "O" else types.index(t[2:]) for t in self.id2tag]))
        if cfg.intent_to_slot:
            self.i2s = nn.Linear(self.n_intents, self.n_slots, bias=False)
            nn.init.zeros_(self.i2s.weight)
        if cfg.slot_to_intent:
            self.s2i = nn.Linear(self.n_slots, self.n_intents, bias=False)
            nn.init.zeros_(self.s2i.weight)
        if cfg.refine:
            self.tag_emb = nn.Embedding(self.n_slots, h)
            nn.init.zeros_(self.tag_emb.weight)
            self.refine_out = nn.Linear(h, h)
            zero_linear(self.refine_out)
            self.register_buffer("tag_is_start", torch.tensor([t[0] in "BS" for t in self.id2tag]), persistent=False)
        if cfg.coint:
            self.ci_s = nn.MultiheadAttention(h, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
            self.ci_i = nn.MultiheadAttention(h, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
            self.ci_norm_s, self.ci_norm_i = nn.LayerNorm(h), nn.LayerNorm(h)
            self.ci_out_s, self.ci_out_i = nn.Linear(h, h), nn.Linear(h, h)
            zero_linear(self.ci_out_s)
            zero_linear(self.ci_out_i)
        if cfg.presence:
            self.presence_head = nn.Linear(h, len(assets.slot_types))
            self.register_buffer("tag_type_idx", torch.tensor(
                [-1 if t == "O" else assets.type_index[t[2:]] for t in self.id2tag]), persistent=False)
        if cfg.wide_slot_bits:
            self.wslot_emb = nn.Embedding(2 ** cfg.wide_slot_bits, self.n_slots)
            nn.init.zeros_(self.wslot_emb.weight)
        if cfg.slot_head == "crf":
            self.crf = CRF(self.n_slots, tagset.allowed, tagset.allowed_start, tagset.allowed_end)
        elif cfg.slot_head == "linear" and cfg.decode == "viterbi":
            neg = -1e4
            self.register_buffer("v_trans", (~torch.as_tensor(tagset.allowed)).float() * neg, persistent=False)
            self.register_buffer("v_start", (~torch.as_tensor(tagset.allowed_start)).float() * neg, persistent=False)
            self.register_buffer("v_end", (~torch.as_tensor(tagset.allowed_end)).float() * neg, persistent=False)
        if cfg.aux_lm:
            self.aux_head = nn.Linear(h, W2V_DIM)

    def sync_task_top(self):
        """After a warm start, both private top layers begin as the shared top layer."""
        if self.cfg.task_top:
            self.top_int.load_state_dict(self.encoder.layers[-1].state_dict())
            self.norm_int.load_state_dict(self.encoder.norm.state_dict())

    # ---------------------------------------------------------------- forward
    def embed(self, word_ids, use_emb0=False):
        V = self.base_vocab
        table = self.emb0 if use_emb0 else self.embedding.weight
        if self.cfg.n_frozen == 0:
            return table[word_ids] if use_emb0 else self.embedding(word_ids)
        is_frozen = word_ids >= V
        clamped = word_ids.clamp(max=V - 1)
        base = table[clamped] if use_emb0 else self.embedding(clamped)
        froz = self.frozen_emb((word_ids - V).clamp(min=0))
        if hasattr(self, "frozen_map"):
            froz = self.frozen_map(froz)
        return torch.where(is_frozen.unsqueeze(-1), froz, base)

    @staticmethod
    def _rnn(rnn, x, mask):
        packed = nn.utils.rnn.pack_padded_sequence(x, mask.long().sum(1).cpu(), batch_first=True,
                                                   enforce_sorted=False)
        out, _ = rnn(packed)
        return nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=mask.size(1))[0]

    def _encode(self, x, mask):
        """Returns token states for the slot task and for the intent task."""
        if self.cfg.encoder != "transformer":
            h = self.out_drop(self._rnn(self.lstm, self.in_drop(x), mask))
            return h, h
        kpm = ~mask
        if self.cfg.pre_lstm:
            x = x + self.pre_out(self._rnn(self.pre_rnn, x, mask))
        attn = None
        if self.cfg.rel_bias:
            B, T = x.size(0), x.size(1)
            idx = torch.arange(T, device=x.device)
            off = (idx[:, None] - idx[None, :]).clamp(-self.REL_RANGE, self.REL_RANGE) + self.REL_RANGE
            attn = self.rel_bias[:, off].unsqueeze(0).expand(B, -1, -1, -1).reshape(B * self.cfg.n_heads, T, T)
        h = self.pos(x)
        if self.cfg.task_top:
            for layer in self.encoder.layers[:-1]:
                h = layer(h, src_mask=attn, src_key_padding_mask=kpm)
            h_int = self.norm_int(self.top_int(h, src_mask=attn, src_key_padding_mask=kpm))
            h_slot = self.encoder.norm(self.encoder.layers[-1](h, src_mask=attn, src_key_padding_mask=kpm))
            return h_slot, h_int
        h = self.encoder(h, mask=attn, src_key_padding_mask=kpm)
        return h, h

    def _co_interact(self, h_s, h_i, mask):
        """Co-Interactive attention (Qin et al. 2021), lighter form; output layers start at zero."""
        wi, ws = self.intent_head.weight, self.slot_head.weight
        hi = h_i + torch.softmax(h_i @ wi.T, -1) @ wi
        hs = h_s + torch.softmax(h_s @ ws.T, -1) @ ws
        a_s, _ = self.ci_s(hs, hi, hi, key_padding_mask=~mask, need_weights=False)
        a_i, _ = self.ci_i(hi, hs, hs, key_padding_mask=~mask, need_weights=False)
        return h_s + self.ci_out_s(self.ci_norm_s(hs + a_s)), h_i + self.ci_out_i(self.ci_norm_i(hi + a_i))

    def forward(self, batch, emb_delta=None):
        cfg = self.cfg
        mask = batch["attention_mask"]
        emb = self.embed(batch["word_ids"])
        if self._track_emb and self.training and emb_delta is None:
            if emb.requires_grad:
                emb.retain_grad()
            else:
                emb.requires_grad_(True)
            self._emb = emb
        if emb_delta is not None:
            emb = emb + emb_delta
        residual = 0.0
        for a in self.adapters.values():
            residual = residual + a(batch)
        if cfg.gaz:
            residual = residual + self.gaz_out(torch.cat(
                [self.gaz_type(batch["gaz_ids"]), self.gaz_bucket(batch["gaz_buckets"])], -1))
        h_slot, h_int = self._encode(self.proj(emb) + residual, mask)
        if cfg.dual_emb:   # intent reads a separate pass over the untouched Word2Vec rows
            _, h_int = self._encode(self.proj(self.embed(batch["word_ids"], use_emb0=True)) + residual, mask)
        if cfg.coint:
            h_slot, h_int = self._co_interact(h_slot, h_int, mask)
        self._h_tok, self._h_slot = h_int, h_slot

        # ---- intent
        mf = mask.unsqueeze(-1).float()
        if cfg.pooling == "attn":
            scores = (self.attn_key(h_int) * self.attn_query).sum(-1).masked_fill(~mask, -1e9)
            pooled = (h_int * torch.softmax(scores, -1).unsqueeze(-1)).sum(1)
        else:
            pooled = (h_int * mf).sum(1) / mf.sum(1).clamp_min(1.0)
        self._pooled = pooled
        intent_logits = self.intent_head(pooled)
        self._scen_logits = None
        if cfg.hier_intent:
            self._scen_logits = self.scenario_head(pooled)
            intent_logits = intent_logits + self._scen_logits[:, self.intent2scen]
        if cfg.pooling == "meanmax":
            mx = h_int.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
            intent_logits = intent_logits + self.intent_max(mx)
        if cfg.label_attn:
            scores = torch.einsum("bth,kh->btk", self.la_key(h_int), self.la_query)
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
            rep = torch.einsum("btk,bth->bkh", torch.softmax(scores, 1), h_int)
            intent_logits = intent_logits + (rep * self.la_out).sum(-1)
        self._presence = self.presence_head(pooled) if cfg.presence else None

        # ---- slots
        h = h_slot
        if cfg.interaction:
            h = h + self.slot_ctx(torch.cat([h, pooled.unsqueeze(1).expand_as(h)], -1))
        self._first_pass = None
        if cfg.slot_head == "biaffine":
            slot_logits = self.biaffine(h)
        else:
            if cfg.refine:     # SlotRefine (Wu et al. 2020), lighter form
                first = self.slot_head(h)
                self._first_pass = first
                with torch.no_grad():
                    prev = first.argmax(-1)
                    prev = torch.where(self.tag_is_start[prev], prev, torch.zeros_like(prev))
                h = h + self.refine_out(self.refine_layer(h + self.tag_emb(prev), src_key_padding_mask=~mask))
            slot_logits = self.slot_head(h)
            if cfg.factorized:
                slot_logits = (slot_logits + self.prefix_head(h)[..., self.tag_prefix]
                               + self.type_head(h)[..., self.tag_type])
            if cfg.intent_to_slot:
                slot_logits = slot_logits + self.i2s(torch.softmax(intent_logits, -1)).unsqueeze(1)
            if cfg.slot_to_intent:
                probs = torch.softmax(slot_logits, -1) * mf
                # drop the O column: the signal is "which slot labels are present"
                probs = torch.cat([torch.zeros_like(probs[..., :1]), probs[..., 1:]], -1)
                intent_logits = intent_logits + self.s2i(probs.sum(1) / mf.sum(1).clamp_min(1.0))

        # ---- sparse linear paths: own loss while training, mixed with the deep path at evaluation
        self._wide_logits = self._deep_logp = self._wide_logp = None
        if cfg.wide_intent:
            vals = batch["wide_vals"]
            if self.training and self._wide_dropout > 0:
                vals = vals * (torch.rand(vals.shape, device=vals.device) >= self._wide_dropout).float() / (1 - self._wide_dropout)
            wide = (self.wide_emb(batch["wide_ids"]) * vals.unsqueeze(-1)).sum(1) + self.wide_bias
            self._wide_logits = wide
            if not self.training:
                self._deep_logp = F.log_softmax(intent_logits, -1)
                self._wide_logp = F.log_softmax(wide, -1)
                intent_logits = cfg.wide_mix * self._deep_logp + (1 - cfg.wide_mix) * self._wide_logp
        self._wslot_logits = None
        if cfg.wide_slot_bits:
            ids = batch["wslot_ids"]
            feats = self.wslot_emb(ids)         # B,T,F,K
            if self.training and self._wide_dropout > 0:
                keep = (torch.rand(ids.shape, device=ids.device) >= self._wide_dropout).float().unsqueeze(-1)
                feats = feats * keep / (1 - self._wide_dropout)
            wide = feats.sum(2)
            self._wslot_logits = wide
            if not self.training:
                deep = F.log_softmax(slot_logits, -1) if cfg.slot_head == "linear" else slot_logits
                slot_logits = deep + cfg.wide_slot_mix * F.log_softmax(wide, -1)
        return intent_logits, slot_logits

    # ---------------------------------------------------------------- loss / decode
    def loss(self, intent_logits, slot_logits, batch, tc=None):
        """tc=None gives the plain joint loss used for dev reporting."""
        smoothing = tc.label_smoothing if tc is not None else 0.0
        gold_i = batch["intent_ids"]
        li = F.cross_entropy(intent_logits, gold_i, label_smoothing=smoothing)
        if self._scen_logits is not None:
            li = li + 0.5 * F.cross_entropy(self._scen_logits, self.intent2scen[gold_i], label_smoothing=smoothing)
        mask = batch["attention_mask"]
        slot_ids = batch["slot_ids"]
        if self.cfg.slot_head == "crf":
            ls = self.crf.nll(slot_logits, slot_ids, mask) / mask.sum().clamp_min(1)
        elif self.cfg.slot_head == "biaffine":
            ls = self.biaffine.loss(slot_logits, slot_ids, mask, self.id2tag, tc.boundary_eps if tc is not None else 0.0)
        else:
            weight = None
            if tc is not None and tc.o_weight != 1.0:
                weight = torch.ones(self.n_slots, device=slot_logits.device)
                weight[0] = tc.o_weight
            ls = F.cross_entropy(slot_logits.reshape(-1, self.n_slots), slot_ids.reshape(-1),
                                 ignore_index=-100, label_smoothing=smoothing, weight=weight)
        if tc is None:
            return li + ls

        if tc.mixup_alpha > 0:     # sentence mixup on the plain intent head (Guo et al. 2019)
            lam = torch.distributions.Beta(tc.mixup_alpha, tc.mixup_alpha).sample().item()
            perm = torch.randperm(gold_i.size(0), device=gold_i.device)
            mixed = self.intent_head(lam * self._pooled + (1 - lam) * self._pooled[perm])
            li_mix = (lam * F.cross_entropy(mixed, gold_i, label_smoothing=smoothing)
                      + (1 - lam) * F.cross_entropy(mixed, gold_i[perm], label_smoothing=smoothing))
            li = 0.5 * li + 0.5 * li_mix
        total = tc.lambda_intent * li + tc.lambda_slot * ls

        if self._first_pass is not None:
            total = total + tc.lambda_slot * F.cross_entropy(
                self._first_pass.reshape(-1, self.n_slots), slot_ids.reshape(-1),
                ignore_index=-100, label_smoothing=smoothing)
        if self._wide_logits is not None:
            total = total + F.cross_entropy(self._wide_logits, gold_i, label_smoothing=smoothing)
        if self._wslot_logits is not None:
            total = total + F.cross_entropy(self._wslot_logits.reshape(-1, self.n_slots), slot_ids.reshape(-1),
                                            ignore_index=-100, label_smoothing=smoothing)
        if self._presence is not None and tc.lambda_presence > 0:
            types = self.tag_type_idx[slot_ids.clamp_min(0)]        # -1 for O and padding
            target = torch.zeros(types.size(0), self._presence.size(1) + 1, device=types.device)
            target.scatter_(1, types + 1, 1.0)
            total = total + tc.lambda_presence * F.binary_cross_entropy_with_logits(self._presence, target[:, 1:])
        if tc.lambda_token_intent > 0:
            n = mask.sum(1)
            total = total + tc.lambda_token_intent * F.cross_entropy(
                self.intent_head(self._h_tok[mask]), gold_i.repeat_interleave(n), label_smoothing=smoothing)
        if self.cfg.aux_lm and tc.lambda_aux > 0 and "sim_mask" in batch and batch["sim_mask"].any():
            sim = batch["sim_mask"]
            pred = self.aux_head(self._h_slot[sim]) @ self.embedding.weight.detach().T
            total = total + tc.lambda_aux * F.cross_entropy(pred, batch["orig_word_ids"][sim])
        if self.cfg.supcon_dim and tc.supcon_lambda > 0:
            total = total + tc.supcon_lambda * self._supcon(gold_i)
        if self.cfg.anchor and tc.anchor_lambda > 0:   # L2-SP (Li et al. 2018): used rows stay near Word2Vec
            ids = batch.get("orig_word_ids", batch["word_ids"])
            active = torch.unique(ids[(ids < self.base_vocab) & mask])
            diff = self.embedding.weight[active] - self.emb0[active]
            total = total + tc.anchor_lambda * (diff ** 2).sum() / max(active.numel(), 1)
        return total

    def _supcon(self, gold_i, tau=0.3):
        """Supervised contrastive loss on the utterance vector (Gunel et al. 2021)."""
        z = F.normalize(self.supcon_proj(self._pooled), dim=-1)
        sim = z @ z.T / tau
        n = gold_i.size(0)
        eye = torch.eye(n, dtype=torch.bool, device=z.device)
        logp = sim - torch.logsumexp(sim.masked_fill(eye, -1e9), dim=1, keepdim=True)
        pos = (gold_i.unsqueeze(0) == gold_i.unsqueeze(1)) & ~eye
        have = pos.any(1)
        if not have.any():
            return sim.sum() * 0.0
        return -((logp * pos.float()).sum(1)[have] / pos.float().sum(1)[have]).mean()

    @torch.no_grad()
    def decode_tags(self, slot_logits, mask):
        """Predicted tags per utterance, always as BIO strings."""
        if self.cfg.slot_head == "biaffine":
            return self.biaffine.decode(slot_logits, mask)
        if self.cfg.slot_head == "crf":
            ids = self.crf.decode(slot_logits, mask)
        elif self.cfg.decode == "viterbi":
            ids = viterbi(F.log_softmax(slot_logits, -1), mask, self.v_trans, self.v_start, self.v_end)
        else:
            pred = slot_logits.argmax(-1).cpu()
            ids = [pred[i, :n].tolist() for i, n in enumerate(mask.long().sum(1).tolist())]
        return [[to_bio(self.id2tag[x]) for x in seq] for seq in ids]


def build_model(cfg, assets, embedding_matrix=None, ext=None):
    frozen = ext.matrix if ext is not None else None
    map_init = (ext.map_weight, ext.map_bias) if ext is not None and ext.map_weight is not None else None
    model = JointModel(cfg, assets, embedding_matrix, frozen, map_init)
    total, _ = count_params(model)
    assert total <= PARAM_CAP, f"{total:,} params exceeds the 20M cap"
    return model


def init_from(model, source_state):
    """Copy every tensor whose name and shape match (Stage-1 -> Stage-2 warm start)."""
    own = model.state_dict()
    copied = []
    for k, v in source_state.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v)
            copied.append(k)
    model.load_state_dict(own)
    model.sync_task_top()
    return copied

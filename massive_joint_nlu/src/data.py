"""MASSIVE loading, tokenisation, BIO parsing, vocabularies, OOV metadata and datasets.

Tokeniser, BIO parser, vocabulary construction and OOV grouping are ported unchanged
from the hand-off notebook so that numbers stay comparable.
"""
import random
import re
import zlib
from collections import Counter

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from torch.utils.data import Dataset

from .config import CSV_PATH, CACHE_DIR, EXPECTED_SPLITS, MAX_LEN, MAX_CHARS

TOKEN_RE = re.compile(r"\w+(?:['’\-]\w+)*|[^\w\s]", re.UNICODE)
SLOT_RE = re.compile(r"\[([^\[\]:]+?)\s*:\s*([^\[\]]*?)\]")

PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"
CHAR_PAD, CHAR_UNK = "<CPAD>", "<CUNK>"
NGRAM_PAD, NGRAM_UNK = "<NGRAM_PAD>", "<NGRAM_UNK>"

NGRAM_MIN, NGRAM_MAX = 3, 5
MAX_NGRAM_VOCAB = 30000
MAX_NGRAMS_PER_TOKEN = 48
BPE_VOCAB_SIZE = 2000
MAX_BPE_PIECES = 12
N_LEX_FEATURES = 16

ALL_FEATURES = ("bpe", "ngram", "char")


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


def load_massive(csv_path=CSV_PATH, verbose=True):
    """Load the verified CSV and assert the official split and label inventory."""
    df = pd.read_csv(csv_path)
    required = {"id", "partition", "utt", "intent", "annot_utt"}
    assert not (required - set(df.columns)), "missing columns"
    assert df["id"].duplicated().sum() == 0
    assert df[list(required)].isna().sum().sum() == 0
    counts = df["partition"].value_counts().to_dict()
    for split, expected in EXPECTED_SPLITS.items():
        assert counts.get(split, 0) == expected, (split, counts.get(split))
    assert df["intent"].nunique() == 60
    errors = [r.id for r in df.itertuples(index=False)
              if tokenize(r.utt) != parse_annotated_utterance(r.annot_utt)["tokens"]]
    assert not errors, f"{len(errors)} alignment errors"
    max_len = max(len(tokenize(x)) for x in df["utt"])
    assert max_len <= MAX_LEN
    if verbose:
        print("Split sizes:", {k: counts[k] for k in EXPECTED_SPLITS},
              "| intents:", df["intent"].nunique(),
              "| alignment errors: 0 | max tokens:", max_len)
    return df


def normalize_word(token):
    return str(token).casefold()


def gold_spans(tags):
    """(type, start, end_exclusive) of well-formed gold BIO tags."""
    out, j = [], 0
    while j < len(tags):
        if tags[j].startswith("B-"):
            typ = tags[j][2:]
            e = j + 1
            while e < len(tags) and tags[e] == "I-" + typ:
                e += 1
            out.append((typ, j, e))
            j = e
        else:
            j += 1
    return out


def character_ngrams(token, n_min=NGRAM_MIN, n_max=NGRAM_MAX):
    word = "<" + normalize_word(token) + ">"
    grams = []
    for n in range(n_min, n_max + 1):
        if len(word) >= n:
            grams += [word[i:i + n] for i in range(len(word) - n + 1)]
    return grams


def lexical_features(low, i, word_counter):
    """Hashable local features of token i, for the sparse tag path."""
    w, n = low[i], len(low)
    p1 = low[i - 1] if i else "<s>"
    p2 = low[i - 2] if i > 1 else "<s>"
    n1 = low[i + 1] if i + 1 < n else "</s>"
    n2 = low[i + 2] if i + 2 < n else "</s>"
    c = word_counter.get(w, 0)
    bucket = 0
    while c > 0 and bucket < 8:
        c >>= 1
        bucket += 1
    return ("w=" + w, "p1=" + p1, "n1=" + n1, "p2=" + p2, "n2=" + n2, "p1w=" + p1 + "_" + w, "wn1=" + w + "_" + n1,
            "p1n1=" + p1 + "_" + n1, "pre3=" + w[:3], "suf3=" + w[-3:], "suf2=" + w[-2:], "pos=%d" % min(i, 5),
            "rpos=%d" % min(n - 1 - i, 5), "len=%d" % min(len(w), 12), "cnt=%d" % bucket, "bias")


# ------------------------------------------------------------------ tag schemes
def bio_to_bioes(tags):
    out = []
    for i, t in enumerate(tags):
        if t == "O":
            out.append("O")
            continue
        prefix, typ = t.split("-", 1)
        continues = i + 1 < len(tags) and tags[i + 1] == "I-" + typ
        if prefix == "B":
            out.append(("B-" if continues else "S-") + typ)
        else:
            out.append(("I-" if continues else "E-") + typ)
    return out


def to_bio(tag):
    if tag == "O":
        return tag
    prefix, typ = tag.split("-", 1)
    return {"B": "B-", "I": "I-", "E": "I-", "S": "B-"}[prefix] + typ


class TagSet:
    """Output labels of a slot head and its hard transition constraints. Label 0 is always O."""

    def __init__(self, scheme, labels, allowed, allowed_start, allowed_end):
        self.scheme = scheme
        self.labels = labels
        self.label2id = {x: i for i, x in enumerate(labels)}
        self.allowed = allowed
        self.allowed_start = allowed_start
        self.allowed_end = allowed_end
        assert labels[0] == "O"

    def encode(self, bio_tags):
        tags = bio_to_bioes(bio_tags) if self.scheme == "bioes" else bio_tags
        return [self.label2id[t] for t in tags]


def _bio_tagset(labels, adjacent):
    """I-X only after B-X or I-X. B-X may follow B-X or I-X only if train has touching spans of X."""
    n = len(labels)
    allowed = np.ones((n, n), dtype=bool)
    start = np.ones(n, dtype=bool)
    for j, tag in enumerate(labels):
        if tag.startswith("I-"):
            typ = tag[2:]
            start[j] = False
            for i, prev in enumerate(labels):
                allowed[i, j] = prev in ("B-" + typ, "I-" + typ)
        elif tag.startswith("B-") and tag[2:] not in adjacent:
            for i, prev in enumerate(labels):
                if prev in (tag, "I-" + tag[2:]):
                    allowed[i, j] = False
    return TagSet("bio", labels, allowed, start, np.ones(n, dtype=bool))


def _bioes_tagset(types, adjacent):
    labels = ["O"] + [p + t for t in types for p in ("B-", "I-", "E-", "S-")]
    n = len(labels)
    allowed = np.zeros((n, n), dtype=bool)
    start = np.zeros(n, dtype=bool)
    end = np.zeros(n, dtype=bool)
    for i, a in enumerate(labels):
        start[i] = a == "O" or a[0] in "BS"
        end[i] = a == "O" or a[0] in "ES"
        for j, b in enumerate(labels):
            if a == "O" or a[0] in "ES":
                allowed[i, j] = b == "O" or b[0] in "BS"
                if a != "O" and b != "O" and a[2:] == b[2:] and a[2:] not in adjacent:
                    allowed[i, j] = False
            else:  # inside a span: only the same type may continue or close it
                allowed[i, j] = b != "O" and b[0] in "IE" and b[2:] == a[2:]
    return TagSet("bioes", labels, allowed, start, end)


class Assets:
    """Every train-derived lookup table, built from the train split. The BIO label inventory is the one exception."""

    def __init__(self, df):
        self.train_df = df[df["partition"] == "train"].copy()
        self.dev_df = df[df["partition"] == "dev"].copy()
        self.test_df = df[df["partition"] == "test"].copy()

        self.intent_labels = sorted(self.train_df["intent"].unique())
        self.intent2id = {x: i for i, x in enumerate(self.intent_labels)}

        train_tokens, char_counter = [], Counter()
        self.train_lexical_casefold = set()
        self.slot_values = {}               # type -> token lists seen in train
        self.intent_slot_types = {}         # intent -> slot types seen with it in train
        self.adjacent_types = set()         # types with two touching spans somewhere in train
        self.label_tokens = {}              # BIO tag -> tokens seen with it in train
        for r in self.train_df.itertuples(index=False):
            parsed = parse_annotated_utterance(r.annot_utt)
            toks, tags = parsed["tokens"], parsed["bio"]
            for tok, tag in zip(toks, tags):
                train_tokens.append(tok)
                if is_lexical(tok):
                    self.train_lexical_casefold.add(tok.casefold())
                char_counter.update(tok.casefold())
                self.label_tokens.setdefault(tag, []).append(tok)
            seen = self.intent_slot_types.setdefault(r.intent, set())
            prev = None
            for typ, a, b in gold_spans(tags):
                self.slot_values.setdefault(typ, []).append(toks[a:b])
                seen.add(typ)
                if prev == (typ, a):
                    self.adjacent_types.add(typ)
                prev = (typ, b)

        # Label inventory taken from all splits exactly as in the hand-off notebook. It only defines the
        # output index space: one label (I-transport_type) occurs only in test and is never predicted.
        label_set = {"O"}
        for split_df in (self.train_df, self.dev_df, self.test_df):
            for annot in split_df["annot_utt"]:
                label_set.update(parse_annotated_utterance(annot)["bio"])
        self.slot_labels = ["O"] + sorted(x for x in label_set if x != "O")
        self.slot2id = {x: i for i, x in enumerate(self.slot_labels)}
        self.id2slot = {i: x for x, i in self.slot2id.items()}
        self.slot_types = sorted({x[2:] for x in self.slot_labels if x != "O"})
        self.type_index = {t: i for i, t in enumerate(self.slot_types)}
        self._tagsets = {}
        self._wide = {}
        self._mention = None

        self.word_counter = Counter(t.casefold() for t in train_tokens)
        self.word_vocab = [PAD_TOKEN, UNK_TOKEN] + sorted(self.word_counter)
        self.word2id = {w: i for i, w in enumerate(self.word_vocab)}
        self.PAD_ID, self.UNK_ID = 0, 1
        self.word_counts = np.array([1e9, 1e9] + [self.word_counter[w] for w in self.word_vocab[2:]], dtype=np.float32)

        self.char_vocab = [CHAR_PAD, CHAR_UNK] + sorted(char_counter)
        self.char2id = {c: i for i, c in enumerate(self.char_vocab)}
        self.CHAR_PAD_ID, self.CHAR_UNK_ID = 0, 1

        ngram_counter = Counter()
        for tok, freq in self.word_counter.items():
            for g in character_ngrams(tok):
                ngram_counter[g] += freq
        self.ngram_vocab = [NGRAM_PAD, NGRAM_UNK] + [g for g, _ in ngram_counter.most_common(MAX_NGRAM_VOCAB)]
        self.ngram2id = {g: i for i, g in enumerate(self.ngram_vocab)}
        self.NGRAM_PAD_ID, self.NGRAM_UNK_ID = 0, 1

        self.sp = self._train_bpe()
        self.BPE_UNK_ID, self.BPE_PAD_ID = int(self.sp.unk_id()), int(self.sp.pad_id())

    def tagset(self, scheme="bio"):
        if scheme not in self._tagsets:
            self._tagsets[scheme] = (_bio_tagset(self.slot_labels, self.adjacent_types) if scheme == "bio"
                                     else _bioes_tagset(self.slot_types, self.adjacent_types))
        return self._tagsets[scheme]

    def wide_vectorizers(self, k):
        """TF-IDF word 1-2 grams and char_wb 2-5 grams, fitted on train only, k features each."""
        if k not in self._wide:
            from sklearn.feature_extraction.text import TfidfVectorizer
            texts = [" ".join(t.casefold() for t in tokenize(u)) for u in self.train_df["utt"]]
            word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"\S+", max_features=k).fit(texts)
            char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=k).fit(texts)
            self._wide[k] = (word, char)
        return self._wide[k]

    def wide_size(self, k):
        word, char = self.wide_vectorizers(k)
        return 1 + len(word.vocabulary_) + len(char.vocabulary_)    # id 0 is padding

    def mention_tables(self):
        """Train only. ng_all[g]: occurrences of token n-gram g (length <= 8). ng_type[(t, g)]: times g was a t mention.
        best[g]: the type most often given to g."""
        if self._mention is None:
            ng_all, ng_type = Counter(), Counter()
            for annot in self.train_df["annot_utt"]:
                parsed = parse_annotated_utterance(annot)
                low = [t.casefold() for t in parsed["tokens"]]
                for i in range(len(low)):
                    for L in range(1, min(8, len(low) - i) + 1):
                        ng_all[tuple(low[i:i + L])] += 1
                for typ, a, b in gold_spans(parsed["bio"]):
                    ng_type[(typ, tuple(low[a:b]))] += 1
            best = {}
            for (typ, g), c in ng_type.items():
                if g not in best or c > best[g][1]:
                    best[g] = (typ, c)
            self._mention = (ng_all, ng_type, {g: v[0] for g, v in best.items()})
        return self._mention

    def intent_tag_pmi(self):
        """Train only: log P(tag | intent) - log P(tag), with add-0.5 smoothing. intents x BIO labels."""
        counts = np.full((len(self.intent_labels), len(self.slot_labels)), 0.5)
        for r in self.train_df.itertuples(index=False):
            for tag in parse_annotated_utterance(r.annot_utt)["bio"]:
                counts[self.intent2id[r.intent], self.slot2id[tag]] += 1
        cond = counts / counts.sum(1, keepdims=True)
        prior = counts.sum(0) / counts.sum()
        return np.log(cond) - np.log(prior)

    def _train_bpe(self):
        prefix = CACHE_DIR / "massive_train_bpe"
        model = CACHE_DIR / "massive_train_bpe.model"
        if not model.exists():
            corpus = CACHE_DIR / "massive_train_bpe.txt"
            with open(corpus, "w", encoding="utf-8") as f:
                for utt in self.train_df["utt"]:
                    f.write(str(utt).casefold() + "\n")
            spm.SentencePieceTrainer.train(
                input=str(corpus), model_prefix=str(prefix), vocab_size=BPE_VOCAB_SIZE,
                model_type="bpe", character_coverage=1.0, bos_id=-1, eos_id=-1,
                unk_id=0, pad_id=1, hard_vocab_limit=False, minloglevel=2)
        sp = spm.SentencePieceProcessor()
        sp.load(str(model))
        return sp

    def summary(self):
        return {"intents": len(self.intent_labels), "bio_labels": len(self.slot_labels),
                "slot_types": len(self.slot_types), "word_vocab": len(self.word_vocab),
                "char_vocab": len(self.char_vocab), "ngram_vocab": len(self.ngram_vocab),
                "bpe_vocab": self.sp.get_piece_size()}

    def split(self, name):
        return {"train": self.train_df, "dev": self.dev_df, "test": self.test_df}[name]


def build_oov_metadata(assets, split_df, pretrained_vocab):
    rows = []
    for r in split_df.itertuples(index=False):
        parsed = parse_annotated_utterance(r.annot_utt)
        toks, smask = parsed["tokens"], parsed["slot_mask"]
        t_oov = [is_lexical(t) and t.casefold() not in assets.train_lexical_casefold for t in toks]
        p_oov = [is_lexical(t) and t.casefold() not in pretrained_vocab for t in toks]

        def group(flags, prefix):
            if not any(flags):
                return f"No-{prefix}-OOV"
            if any(o and s for o, s in zip(flags, smask)):
                return f"Slot-{prefix}-OOV"
            return f"NonSlot-{prefix}-OOV"

        rows.append({
            "id": r.id, "n_tokens": len(toks),
            "n_gold_slots": sum(t.startswith("B-") for t in parsed["bio"]),
            "n_train_oov": sum(t_oov),
            "n_slot_train_oov": sum(o and s for o, s in zip(t_oov, smask)),
            "train_oov_group": group(t_oov, "Train"),
            "pretrained_oov_group": group(p_oov, "Pretrained"),
        })
    return pd.DataFrame(rows)


class ExtVocab:
    """Frozen Word2Vec rows appended after the trainable train vocabulary.

    rows = [original vectors of pretrained-covered train words] + [top-K extra words].
    Train words keep their trainable row; the frozen originals exist so that training can move a
    train word onto the frozen space and learn to read it. The K extra words are chosen by Word2Vec
    file order only, never by looking at dev or test. dim < 300 stores the rows PCA-compressed
    (fit on these rows only) so that more words fit under the parameter cap.
    """

    def __init__(self, assets, w2v, k_extra, dim=300):
        V = len(assets.word_vocab)
        self.base_size = V
        self.words, rows = [], []
        self.swap_of = {}  # train-vocab id -> frozen global id
        for w, idx in assets.word2id.items():
            if idx < 2:
                continue
            j = w2v.index.get(w)
            if j is not None:
                self.swap_of[idx] = V + len(self.words)
                self.words.append(w)
                rows.append(j)
        self.extra_words = []
        for w in w2v.keys:
            if len(self.extra_words) >= k_extra:
                break
            if w in assets.word2id or not w.strip():
                continue
            self.words.append(w)
            rows.append(w2v.index[w])
            self.extra_words.append(w)
        self.word2gid = {w: V + i for i, w in enumerate(self.words)}
        matrix = np.asarray(w2v.matrix[np.sort(rows)], dtype=np.float32)[np.argsort(np.argsort(rows))]
        self.k_extra = len(self.extra_words)
        self.dim = dim
        self.map_weight, self.map_bias = None, None
        if dim < matrix.shape[1]:
            mean = matrix.mean(0)
            centered = matrix - mean
            _, eigvecs = np.linalg.eigh(centered.T @ centered)
            comps = eigvecs[:, ::-1][:, :dim].astype(np.float32)   # 300 x dim
            matrix = centered @ comps
            self.map_weight, self.map_bias = comps, mean.astype(np.float32)
        self.matrix = matrix.astype(np.float32)

    @property
    def n_frozen(self):
        return len(self.words)


class MassiveDataset(Dataset):
    """Encodes each utterance once. Only the feature families a model needs are kept in memory.
    wide: k of the sparse intent path. wslot_bits: hash size of the sparse tag path. gaz: mention-table
    input; loo=True removes each utterance's own counts first and must be set for the train split."""

    def __init__(self, assets, split_df, ext=None, features=ALL_FEATURES, scheme="bio", wide=0, wslot_bits=0,
                 gaz=False, loo=False):
        self.A = assets
        self.ext = ext
        self.features = tuple(features)
        self.tagset = assets.tagset(scheme)
        self.wide, self.wslot_bits, self.gaz, self.loo = wide, wslot_bits, gaz, loo
        self.mention_replace = self.lwtr = self.sis = 0.0
        self._tok = {}
        self.raw, self.rows = [], []
        for r in split_df.itertuples(index=False):
            parsed = parse_annotated_utterance(r.annot_utt)
            iid = assets.intent2id[r.intent]
            self.raw.append((r.id, parsed["tokens"], parsed["bio"], iid))
            self.rows.append(self._encode(r.id, parsed["tokens"], parsed["bio"], iid, sparse=False))
        if wide:
            texts = [" ".join(t.casefold() for t in x[1]) for x in self.raw]
            for row, (ids, vals) in zip(self.rows, self._wide_rows(texts)):
                row["wide_ids"], row["wide_vals"] = ids, vals

    def _wide_rows(self, texts):
        word, char = self.A.wide_vectorizers(self.wide)
        xw, xc = word.transform(texts).tocsr(), char.transform(texts).tocsr()
        off = 1 + len(word.vocabulary_)
        out = []
        for i in range(len(texts)):
            ids = np.concatenate([xw[i].indices + 1, xc[i].indices + off])
            vals = np.concatenate([xw[i].data, xc[i].data]).astype(np.float32)
            if len(ids) == 0:
                ids, vals = np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.float32)
            out.append((torch.from_numpy(ids.astype(np.int64)), torch.from_numpy(vals)))
        return out

    def _token(self, t):
        got = self._tok.get(t)
        if got is not None:
            return got
        A, ext = self.A, self.ext
        cf = normalize_word(t)
        wid = A.word2id.get(cf)
        if wid is None:
            word, swap = (ext.word2gid.get(cf, A.UNK_ID) if ext is not None else A.UNK_ID), -1
        else:
            word, swap = wid, (ext.swap_of.get(wid, -1) if ext is not None else -1)
        out = {"word": word, "swap": swap, "lex": is_lexical(t)}
        if "bpe" in self.features:
            b = A.sp.encode(cf, out_type=int)[:MAX_BPE_PIECES] or [A.BPE_UNK_ID]
            out["bpe"] = b + [A.BPE_PAD_ID] * (MAX_BPE_PIECES - len(b))
        if "ngram" in self.features:
            g = [A.ngram2id.get(x, A.NGRAM_UNK_ID) for x in character_ngrams(t)[:MAX_NGRAMS_PER_TOKEN]] or [A.NGRAM_UNK_ID]
            out["ngram"] = g + [A.NGRAM_PAD_ID] * (MAX_NGRAMS_PER_TOKEN - len(g))
        if "char" in self.features:
            c = [A.char2id.get(ch, A.CHAR_UNK_ID) for ch in cf[:MAX_CHARS]] or [A.CHAR_UNK_ID]
            out["clen"] = len(c)
            out["char"] = c + [A.CHAR_PAD_ID] * (MAX_CHARS - len(c))
        self._tok[t] = out
        return out

    def _gaz(self, low, tags):
        """Longest train mention covering each token: 1 + 4*type + position (S,B,I,E), and a reliability bucket.
        On train the utterance's own counts are removed first, otherwise the feature fires on every mention."""
        A = self.A
        ng_all, ng_type, best = A.mention_tables()
        own_type, own_all = Counter(), Counter()
        if self.loo:
            for typ, a, b in gold_spans(tags):
                own_type[(typ, tuple(low[a:b]))] += 1
            for i in range(len(low)):
                for L in range(1, min(8, len(low) - i) + 1):
                    own_all[tuple(low[i:i + L])] += 1
        n = len(low)
        ids, buckets, used = [0] * n, [0] * n, [False] * n
        for L in range(min(8, n), 0, -1):
            for i in range(n - L + 1):
                g = tuple(low[i:i + L])
                typ = best.get(g)
                if typ is None or any(used[i:i + L]):
                    continue
                c_type = ng_type[(typ, g)] - own_type[(typ, g)]
                c_all = ng_all[g] - own_all[g]
                if c_type <= 0 or c_all <= 0:
                    continue
                bucket = min(int(5 * c_type / c_all), 4) + 1
                for j in range(i, i + L):
                    pos = 0 if L == 1 else (1 if j == i else (3 if j == i + L - 1 else 2))
                    ids[j], buckets[j], used[j] = 1 + 4 * A.type_index[typ] + pos, bucket, True
        return ids, buckets

    def _encode(self, rid, toks, tags, iid, sparse=True):
        enc = [self._token(t) for t in toks]
        row = {
            "id": rid,
            "word_ids": torch.tensor([e["word"] for e in enc]),
            "swap_ids": torch.tensor([e["swap"] for e in enc]),
            "slot_ids": torch.tensor(self.tagset.encode(tags)),
            "lexical_mask": torch.tensor([e["lex"] for e in enc]),
            "intent_id": iid,
        }
        if "bpe" in self.features:
            row["bpe_ids"] = torch.tensor([e["bpe"] for e in enc])
        if "ngram" in self.features:
            row["ngram_ids"] = torch.tensor([e["ngram"] for e in enc])
        if "char" in self.features:
            row["char_ids"] = torch.tensor([e["char"] for e in enc])
            row["char_lengths"] = torch.tensor([e["clen"] for e in enc])
        low = [normalize_word(t) for t in toks]
        if self.wslot_bits:
            size = 2 ** self.wslot_bits
            row["wslot_ids"] = torch.tensor(
                [[zlib.crc32(f.encode("utf-8")) % size for f in lexical_features(low, i, self.A.word_counter)]
                 for i in range(len(low))])
        if self.gaz:
            ids, buckets = self._gaz(low, tags)
            row["gaz_ids"], row["gaz_buckets"] = torch.tensor(ids), torch.tensor(buckets)
        if sparse and self.wide:
            row["wide_ids"], row["wide_vals"] = self._wide_rows([" ".join(low)])[0]
        return row

    def _replaced(self, i):
        """Mention replacement (Dai & Adel 2020): swap slot values within a type, train values only."""
        rid, toks, tags, iid = self.raw[i]
        spans = {a: (typ, b) for typ, a, b in gold_spans(tags)}
        new_t, new_b, j = [], [], 0
        while j < len(toks):
            if j in spans:
                typ, e = spans[j]
                pool = self.A.slot_values.get(typ)
                if pool and random.random() < 0.5:
                    val = random.choice(pool)
                    new_t += val
                    new_b += ["B-" + typ] + ["I-" + typ] * (len(val) - 1)
                else:
                    new_t += toks[j:e]
                    new_b += tags[j:e]
                j = e
            else:
                new_t.append(toks[j])
                new_b.append(tags[j])
                j += 1
        if len(new_t) > MAX_LEN:
            return self.rows[i]
        return self._encode(rid, new_t, new_b, iid)

    def _lwtr(self, i):
        """Label-wise token replacement: tokens swapped for train tokens seen with the same BIO tag."""
        rid, toks, tags, iid = self.raw[i]
        new_t = [random.choice(self.A.label_tokens[g]) if g in self.A.label_tokens and random.random() < 0.2 else t
                 for t, g in zip(toks, tags)]
        return self._encode(rid, new_t, tags, iid)

    def _sis(self, i):
        """Shuffle within segments: tokens of each slot span are permuted, tags stay in place."""
        rid, toks, tags, iid = self.raw[i]
        new_t = list(toks)
        for _, a, b in gold_spans(tags):
            seg = new_t[a:b]
            random.shuffle(seg)
            new_t[a:b] = seg
        return self._encode(rid, new_t, tags, iid)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.mention_replace > 0 and random.random() < self.mention_replace:
            return self._replaced(i)
        if self.lwtr > 0 and random.random() < self.lwtr:
            return self._lwtr(i)
        if self.sis > 0 and random.random() < self.sis:
            return self._sis(i)
        return self.rows[i]


def make_collate(assets):
    A = assets

    def collate(batch):
        B, T = len(batch), max(len(x["word_ids"]) for x in batch)
        out = {
            "ids": [x["id"] for x in batch],
            "word_ids": torch.full((B, T), A.PAD_ID, dtype=torch.long),
            "swap_ids": torch.full((B, T), -1, dtype=torch.long),
            "slot_ids": torch.full((B, T), -100, dtype=torch.long),
            "attention_mask": torch.zeros((B, T), dtype=torch.bool),
            "lexical_mask": torch.zeros((B, T), dtype=torch.bool),
            "intent_ids": torch.tensor([x["intent_id"] for x in batch]),
        }
        first = batch[0]
        if "bpe_ids" in first:
            out["bpe_ids"] = torch.full((B, T, MAX_BPE_PIECES), A.BPE_PAD_ID, dtype=torch.long)
        if "ngram_ids" in first:
            out["ngram_ids"] = torch.full((B, T, MAX_NGRAMS_PER_TOKEN), A.NGRAM_PAD_ID, dtype=torch.long)
        if "char_ids" in first:
            out["char_ids"] = torch.full((B, T, MAX_CHARS), A.CHAR_PAD_ID, dtype=torch.long)
            out["char_lengths"] = torch.ones((B, T), dtype=torch.long)
        if "wslot_ids" in first:
            out["wslot_ids"] = torch.zeros((B, T, N_LEX_FEATURES), dtype=torch.long)
        if "gaz_ids" in first:
            out["gaz_ids"] = torch.zeros((B, T), dtype=torch.long)
            out["gaz_buckets"] = torch.zeros((B, T), dtype=torch.long)
        keys = [k for k in ("word_ids", "swap_ids", "slot_ids", "lexical_mask", "bpe_ids", "ngram_ids", "char_ids",
                            "char_lengths", "wslot_ids", "gaz_ids", "gaz_buckets") if k in first]
        for i, x in enumerate(batch):
            n = len(x["word_ids"])
            out["attention_mask"][i, :n] = True
            for k in keys:
                out[k][i, :n] = x[k]
        if "wide_ids" in first:
            L = max(len(x["wide_ids"]) for x in batch)
            out["wide_ids"] = torch.zeros((B, L), dtype=torch.long)
            out["wide_vals"] = torch.zeros((B, L))
            for i, x in enumerate(batch):
                n = len(x["wide_ids"])
                out["wide_ids"][i, :n] = x["wide_ids"]
                out["wide_vals"][i, :n] = x["wide_vals"]
        return out

    return collate

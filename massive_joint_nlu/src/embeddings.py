"""NLPL Wikipedia Word2Vec 2017 (model 6) loading without gensim, plus casefold map."""
import json
import re
from collections import OrderedDict

import numpy as np

from .config import W2V_DIR, CACHE_DIR, W2V_DIM, SEED

POS_SUFFIX_RE = re.compile(
    r"_(ADJ|ADP|ADV|AUX|CCONJ|DET|INTJ|NOUN|NUM|PART|PRON|PROPN|PUNCT|SCONJ|SYM|VERB|X)$")


class CasefoldVectors:
    """Casefold-averaged vectors; `keys` keep the file's (frequency) order."""

    def __init__(self, keys, matrix, n_raw):
        self.keys = keys
        self.matrix = matrix
        self.index = {k: i for i, k in enumerate(keys)}
        self.n_raw = n_raw

    def __contains__(self, w):
        return w in self.index

    def __len__(self):
        return len(self.keys)


def _find_vector_file():
    cands = [p for p in W2V_DIR.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".vec", ".bin"}]
    assert cands, f"No vector file under {W2V_DIR}"
    return max(cands, key=lambda p: p.stat().st_size)


def load_word2vec(verbose=True):
    """Returns the casefold map. The matrix is memory-mapped once cached, to keep RAM use low."""
    keys_path, mat_path = CACHE_DIR / "w2v_casefold_keys.json", CACHE_DIR / "w2v_casefold.npy"
    if keys_path.exists() and mat_path.exists():
        meta = json.loads(keys_path.read_text(encoding="utf-8"))
        cv = CasefoldVectors(meta["keys"], np.load(mat_path, mmap_mode="r"), meta["n_raw"])
        if verbose:
            print(f"Word2Vec (cached): raw {cv.n_raw:,} -> casefold {len(cv):,} x {cv.matrix.shape[1]}")
        return cv

    path = _find_vector_file()
    assert path.suffix.lower() != ".bin", "binary format not handled; expected model.txt"
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        n_raw, dim = map(int, f.readline().split())
        assert dim == W2V_DIM
        raw_keys = []
        raw = np.empty((n_raw, dim), dtype=np.float32)
        i = 0
        for line in f:
            parts = line.rstrip("\n").rstrip(" ").split(" ")
            if len(parts) != dim + 1:
                continue
            raw_keys.append(parts[0])
            raw[i] = np.asarray(parts[1:], dtype=np.float32)
            i += 1
        raw = raw[:i]
    assert i == n_raw, (i, n_raw)

    probe = raw_keys[:50_000]
    pos_ratio = float(np.mean([bool(POS_SUFFIX_RE.search(k)) for k in probe]))
    sums, counts = OrderedDict(), {}
    for k, vec in zip(raw_keys, raw):
        surface = POS_SUFFIX_RE.sub("", k) if pos_ratio > 0.20 else k
        cf = surface.casefold()
        if cf in sums:
            sums[cf] += vec
            counts[cf] += 1
        else:
            sums[cf] = vec.copy()
            counts[cf] = 1
    keys = list(sums)
    matrix = np.stack([sums[k] / counts[k] for k in keys]).astype(np.float32)
    np.save(mat_path, matrix)
    keys_path.write_text(json.dumps({"keys": keys, "n_raw": n_raw, "pos_ratio": pos_ratio,
                                     "source": str(path.name)}), encoding="utf-8")
    if verbose:
        print(f"Word2Vec loaded from {path.name}: raw {n_raw:,} (POS-suffix ratio {pos_ratio:.2%})"
              f" -> casefold {len(keys):,} x {dim}")
    return CasefoldVectors(keys, matrix, n_raw)


def build_embedding_matrix(assets, w2v, seed=SEED, verbose=True):
    """Hand-off 'fair' init: Word2Vec where available, own random row otherwise."""
    rng = np.random.default_rng(seed)
    V = len(assets.word_vocab)
    emb = rng.normal(0.0, 0.05, size=(V, W2V_DIM)).astype(np.float32)
    emb[assets.PAD_ID] = 0.0
    emb[assets.UNK_ID] = rng.normal(0.0, 0.05, size=(W2V_DIM,)).astype(np.float32)
    pre = rnd = 0
    for w, idx in assets.word2id.items():
        if idx < 2:
            continue
        j = w2v.index.get(w)
        if j is not None:
            emb[idx] = w2v.matrix[j]
            pre += 1
        else:
            rnd += 1
    stats = {"vocab": V, "embedding_params": V * W2V_DIM, "pretrained_init": pre,
             "random_init": rnd, "coverage_pct": round(100.0 * pre / max(pre + rnd, 1), 2)}
    if verbose:
        print("Embedding init:", stats)
    return emb, stats

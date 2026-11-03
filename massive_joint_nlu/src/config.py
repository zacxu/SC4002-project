"""Paths, constants and reproducibility helpers shared by every experiment."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import random
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "massive_enUS_full.csv"
W2V_DIR = DATA_DIR / "nlpl_model_6"
CACHE_DIR = ROOT / "cache"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
for _d in (CACHE_DIR, CKPT_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

EXPECTED_SPLITS = {"train": 11514, "dev": 2033, "test": 2974}
PARAM_CAP = 20_000_000
SEED = 42
MAX_LEN = 64
MAX_CHARS = 24
W2V_DIM = 300

_force_cpu = bool(os.environ.get("MASSIVE_CPU"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not _force_cpu else "cpu")


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _known_fields(cls, d):
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in names}


@dataclass
class ModelConfig:
    """Architecture switches. Defaults reproduce the hand-off Stage-1 model exactly."""
    # encoder
    encoder: str = "transformer"        # "transformer" | "bilstm"
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 768
    dropout: float = 0.15
    lstm_hidden: int = 128              # per direction, bilstm only
    lstm_layers: int = 2
    pre_lstm: bool = False              # zero-init residual BiLSTM before the Transformer
    rel_bias: bool = False              # learned relative-position attention bias (TENER-style)
    task_top: bool = False              # one private top Transformer layer per task
    coint: bool = False                 # co-interactive attention between intent and slot streams
    # word representation
    adapters: tuple = ()                # subset of ("bpe", "ngram", "char")
    n_frozen: int = 0                   # frozen Word2Vec rows (extended vocabulary)
    frozen_dim: int = 300               # < 300: rows stored PCA-compressed
    frozen_map: bool = True             # trainable map from frozen rows into embedding space
    gaz: bool = False                   # train-only mention table as input, leave-one-out on train
    anchor: bool = False                # keep a frozen Word2Vec copy for the L2-SP penalty
    dual_emb: bool = False              # intent reads a second pass over the frozen Word2Vec copy
    # intent head
    pooling: str = "mean"               # "mean" | "meanmax" | "attn"
    label_attn: bool = False            # per-intent attention over tokens
    hier_intent: bool = False           # add a scenario score (intent = scenario_action) + aux loss
    slot_to_intent: bool = False        # intent head also sees the mean slot-label distribution
    wide_intent: int = 0                # sparse linear intent path: k word + k char TF-IDF features
    wide_mix: float = 0.5               # weight of the deep path when the two are mixed at evaluation
    supcon_dim: int = 0                 # projection size of a supervised contrastive term
    # slot head
    slot_head: str = "linear"           # "linear" | "crf" | "biaffine"
    tag_scheme: str = "bio"             # "bio" | "bioes"
    decode: str = "greedy"              # linear head only: "greedy" | "viterbi" with hard tag rules
    factorized: bool = False            # tag score = full + boundary + type
    interaction: bool = False           # slot head also sees the pooled utterance vector
    intent_to_slot: bool = False        # slot scores get a bias from the intent distribution
    refine: bool = False                # SlotRefine-style second pass fed with first-pass B- tags
    presence: bool = False              # head predicting which slot types occur in the utterance
    wide_slot_bits: int = 0             # sparse linear tag path over hashed lexical features, 2**bits
    wide_slot_mix: float = 0.5          # weight of the sparse tag path at evaluation
    # auxiliary training-only heads
    aux_lm: bool = False                # predict the original word at simulated-OOV positions

    def to_dict(self):
        d = asdict(self)
        d["adapters"] = list(self.adapters)
        return d

    @classmethod
    def from_dict(cls, d):
        d = _known_fields(cls, d)
        d["adapters"] = tuple(d.get("adapters", ()))
        return cls(**d)


@dataclass
class TrainConfig:
    """Optimisation switches. Defaults reproduce the hand-off Stage-1 recipe exactly."""
    epochs: int = 12
    patience: int = 3
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    clip: float = 1.0
    schedule: str = "constant"          # "constant" | "cosine"
    warmup_epochs: float = 0.0
    label_smoothing: float = 0.0
    # simulated OOV (training only). Per-token rate = word + slot*in_gold_slot + a/(a+train_count)
    word_dropout: float = 0.0
    slot_dropout: float = 0.0
    rare_alpha: float = 0.0
    mention_dropout: float = 0.0        # P(a whole gold span becomes <UNK>)
    swap_frac: float = 0.0              # share of simulated tokens sent to their frozen row, not <UNK>
    swap_prob: float = 0.0              # extra uniform rate of train word -> its frozen row
    # utterance recombination from train tokens; flagged because it creates new token sequences
    mention_replace: float = 0.0        # swap slot values within a type
    lwtr: float = 0.0                   # label-wise token replacement
    sis: float = 0.0                    # shuffle tokens within each slot span
    # loss weights
    lambda_intent: float = 1.0
    lambda_slot: float = 1.0
    lambda_token_intent: float = 0.0    # stack-propagation style per-token intent loss
    lambda_aux: float = 0.0             # weight of the aux_lm head
    lambda_presence: float = 0.2        # weight of the slot-type presence head
    o_weight: float = 1.0               # CE weight of the O tag (linear head only)
    boundary_eps: float = 0.0           # biaffine head: boundary smoothing (Zhu & Li 2022)
    # sparse linear paths
    wide_lr: float = 0.05
    wide_wd: float = 0.01
    wide_dropout: float = 0.3
    # regularisers
    adv_eps: float = 0.0                # adversarial embedding noise, relative to embedding norm
    rdrop_alpha: float = 0.0            # consistency between two dropout passes
    cons_alpha: float = 0.0             # consistency between word-dropped and clean views
    sam_rho: float = 0.0                # sharpness-aware minimisation radius
    mixup_alpha: float = 0.0            # sentence mixup on the intent head
    supcon_lambda: float = 0.0          # supervised contrastive term (needs supcon_dim)
    anchor_lambda: float = 0.0          # L2-SP pull of used word rows towards Word2Vec (needs anchor)
    # other
    emb_lr_mult: float = 1.0            # LR multiplier for the trainable word table (0 = frozen)
    ema_decay: float = 0.0              # 0 disables EMA weight averaging
    seed: int = SEED

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**_known_fields(cls, d))

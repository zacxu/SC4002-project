"""One joint Transformer, with independently ablatable spelling/slot improvements.

Only utterance-derived IDs and masks enter forward. Pretrained *word vectors*
may initialize the fitted task vocabulary; no pretrained encoder is loaded.
This file's definitions can be inlined with BIOCRF into a standalone notebook.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from src.stage2_model import BIOCRF


@dataclass(frozen=True)
class RevisedTransformerConfig:
    word_dim: int = 300
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1792
    dropout: float = 0.25
    char_dim: int = 32
    char_channels: int = 64
    char_kernels: tuple[int, ...] = (3, 4, 5)
    intent_context_dim: int = 64
    attention_pool_dim: int = 128
    head_hidden_dim: int = 256
    use_char: bool = True
    use_crf: bool = True
    use_intent_context: bool = True
    use_attention_pool: bool = True
    use_input_norm: bool = True
    independent_init: bool = True

    def __post_init__(self):
        for name in ('word_dim', 'd_model', 'n_heads', 'n_layers', 'd_ff', 'char_dim',
                     'char_channels', 'intent_context_dim', 'attention_pool_dim', 'head_hidden_dim'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + ' must be a positive integer')
        if self.d_model % self.n_heads:
            raise ValueError('d_model must be divisible by n_heads')
        if self.d_model % 2:
            raise ValueError('d_model must be even for sinusoidal positions')
        if (isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float))
                or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1):
            raise ValueError('dropout must be finite and in [0, 1)')
        kernels = tuple(self.char_kernels)
        if not kernels or len(set(kernels)) != len(kernels) or any(type(k) is not int or k <= 0 for k in kernels):
            raise ValueError('Character kernels must be distinct positive integers')
        object.__setattr__(self, 'char_kernels', kernels)
        for name in ('use_char', 'use_crf', 'use_intent_context', 'use_attention_pool',
                     'use_input_norm', 'independent_init'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(name + ' must be Boolean')

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict) or set(values) != {field.name for field in fields(cls)}:
            raise ValueError('Configuration must contain exactly the Transformer fields')
        return cls(**values)

    @classmethod
    def original_baseline(cls):
        return cls(d_model=256, n_heads=4, n_layers=4, d_ff=768, dropout=.15,
                   use_char=False, use_crf=False, use_intent_context=False,
                   use_attention_pool=False, use_input_norm=False, independent_init=False)


class DynamicCharacterCNN(nn.Module):
    """Convolve full character strings; pool only centers at actual characters."""
    def __init__(self, vocabulary_size, embedding_dim, channels, kernels):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, channels, kernel_size=k) for k in kernels])
        self.kernels = tuple(kernels)
        self.output_dim = channels * len(kernels)

    def forward(self, char_ids, token_mask):
        if (char_ids.ndim != 3 or char_ids.dtype != torch.long
                or char_ids.shape[:2] != token_mask.shape or char_ids.shape[2] == 0):
            raise ValueError('Characters require [batch,tokens,characters] long IDs')
        batch, tokens, width = char_ids.shape
        flat = char_ids.reshape(batch * tokens, width)
        mask = flat.ne(0)
        embedded = self.embedding(flat).transpose(1, 2)
        pooled = []
        for kernel, convolution in zip(self.kernels, self.convs):
            left = (kernel - 1) // 2
            values = F.relu(convolution(F.pad(embedded, (left, kernel - 1 - left))))
            values = values.masked_fill(~mask.unsqueeze(1), -torch.inf).amax(dim=-1)
            values = torch.where(mask.any(1, keepdim=True), values, torch.zeros_like(values))
            pooled.append(values)
        features = torch.cat(pooled, dim=1).reshape(batch, tokens, self.output_dim)
        return features.masked_fill(~token_mask.unsqueeze(-1), 0)


class SinusoidalPositions(nn.Module):
    """Fixed, dynamically sized positions; no sequence or word is truncated."""
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.register_buffer('_cache', torch.empty(0, width), persistent=False)

    def forward(self, values):
        if self._cache.shape[0] < values.shape[1]:
            # The supplied baseline computed trigonometric constants on CPU.
            # Keep those values while extending positions without truncation.
            position = torch.arange(values.shape[1], dtype=torch.float32).unsqueeze(1)
            frequencies = torch.exp(torch.arange(0, self.width, 2, dtype=torch.float32)
                                    * (-math.log(10000.0) / self.width))
            positions = torch.zeros(values.shape[1], self.width, dtype=torch.float32)
            positions[:, 0::2] = torch.sin(position * frequencies)
            positions[:, 1::2] = torch.cos(position * frequencies)
            self._cache = positions.to(device=values.device)
        return values + self._cache[:values.shape[1]].to(dtype=values.dtype).unsqueeze(0)


class IndependentlyInitializedEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.n_heads, dim_feedforward=config.d_ff,
            dropout=config.dropout, activation='gelu', batch_first=True, norm_first=True)
            for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, values, src_key_padding_mask):
        for layer in self.layers:
            values = layer(values, src_key_padding_mask=src_key_padding_mask)
        return self.norm(values)


class JointTransformer(nn.Module):
    """One shared encoder jointly optimized for intent and exact slot spans."""
    def __init__(self, vocab_size: int, num_intents: int, tag_names: Sequence[str],
                 char_vocab_size: int, config: RevisedTransformerConfig | None = None,
                 embedding_matrix=None):
        super().__init__()
        for name, value in (('vocab_size', vocab_size), ('num_intents', num_intents),
                            ('char_vocab_size', char_vocab_size)):
            if type(value) is not int or value <= 0:
                raise ValueError(name + ' must be a positive integer')
        if vocab_size < 2 or char_vocab_size < 2:
            raise ValueError('Vocabularies require PAD=0 and UNK=1')
        self.config = RevisedTransformerConfig() if config is None else config
        if not isinstance(self.config, RevisedTransformerConfig):
            raise TypeError('Expected RevisedTransformerConfig')
        cfg = self.config
        # Validate the same systematic BIO inventory even if the CRF is ablated.
        candidate_crf = BIOCRF(tag_names)
        self.num_intents, self.num_tags = num_intents, len(tag_names)
        self.tag_names = tuple(tag_names)
        self.embedding = nn.Embedding(vocab_size, cfg.word_dim, padding_idx=0)
        if embedding_matrix is not None:
            matrix = torch.as_tensor(embedding_matrix, dtype=self.embedding.weight.dtype)
            if matrix.shape != self.embedding.weight.shape or not bool(torch.isfinite(matrix).all()):
                raise ValueError('Task embedding matrix has wrong shape or nonfinite values')
            with torch.no_grad():
                self.embedding.weight.copy_(matrix)
                self.embedding.weight[0].zero_()
        self.char_encoder = DynamicCharacterCNN(char_vocab_size, cfg.char_dim, cfg.char_channels,
                                                cfg.char_kernels) if cfg.use_char else None
        # Observed pretrained vectors have much smaller feature RMS than max-
        # pooled character CNN outputs. Normalize each branch before mixing;
        # a post-projection norm alone cannot undo unequal input dominance.
        self.word_norm = nn.LayerNorm(cfg.word_dim) if cfg.use_input_norm else nn.Identity()
        self.char_norm = (nn.LayerNorm(self.char_encoder.output_dim)
                          if cfg.use_input_norm and self.char_encoder is not None else nn.Identity())
        input_width = cfg.word_dim + (self.char_encoder.output_dim if cfg.use_char else 0)
        self.proj = nn.Linear(input_width, cfg.d_model)
        self.pos = SinusoidalPositions(cfg.d_model)
        self.input_norm = nn.LayerNorm(cfg.d_model) if cfg.use_input_norm else nn.Identity()
        self.input_dropout = nn.Dropout(cfg.dropout) if cfg.use_input_norm else nn.Identity()
        if cfg.independent_init:
            self.encoder = IndependentlyInitializedEncoder(cfg)
        else:
            layer = nn.TransformerEncoderLayer(d_model=cfg.d_model, nhead=cfg.n_heads,
                dim_feedforward=cfg.d_ff, dropout=cfg.dropout, activation='gelu',
                batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers,
                norm=nn.LayerNorm(cfg.d_model), enable_nested_tensor=False)
        if cfg.use_attention_pool:
            self.attention_pool = nn.Sequential(nn.Linear(cfg.d_model, cfg.attention_pool_dim),
                                               nn.Tanh(), nn.Linear(cfg.attention_pool_dim, 1))
            self.intent_head = nn.Sequential(nn.Linear(2 * cfg.d_model, cfg.head_hidden_dim),
                nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.head_hidden_dim, num_intents))
        else:
            self.attention_pool = None
            self.intent_head = nn.Linear(cfg.d_model, num_intents)
        self.intent_embeddings = nn.Embedding(num_intents, cfg.intent_context_dim) if cfg.use_intent_context else None
        slot_width = cfg.d_model + (cfg.intent_context_dim if cfg.use_intent_context else 0)
        self.slot_norm = nn.LayerNorm(slot_width) if cfg.use_attention_pool else nn.Identity()
        self.slot_head = (nn.Sequential(nn.Linear(slot_width, cfg.head_hidden_dim), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.head_hidden_dim, self.num_tags))
            if cfg.use_attention_pool else nn.Linear(slot_width, self.num_tags))
        self.crf = candidate_crf if cfg.use_crf else None
        enforce_parameter_cap(self)

    @staticmethod
    def _validate_mask(mask, shape):
        if (mask.dtype != torch.bool or tuple(mask.shape) != tuple(shape) or len(shape) != 2
                or shape[0] == 0 or shape[1] == 0 or not bool(mask[:, 0].all())
                or bool((mask[:, 1:] & ~mask[:, :-1]).any())):
            raise ValueError('Attention mask must contain nonempty right-padded prefixes')

    def forward(self, word_ids, attention_mask, char_ids=None):
        if word_ids.ndim != 2 or word_ids.dtype != torch.long:
            raise ValueError('Words require long IDs [batch,tokens]')
        self._validate_mask(attention_mask, word_ids.shape)
        values = self.word_norm(self.embedding(word_ids))
        if self.config.use_input_norm:
            values = values.masked_fill(~attention_mask.unsqueeze(-1), 0)
        if self.char_encoder is not None:
            if char_ids is None:
                raise ValueError('Character-enabled model requires utterance character IDs')
            characters = self.char_norm(self.char_encoder(char_ids, attention_mask))
            if self.config.use_input_norm:
                characters = characters.masked_fill(~attention_mask.unsqueeze(-1), 0)
            values = torch.cat((values, characters), dim=-1)
        encoded = self.encoder(self.input_dropout(self.input_norm(self.pos(self.proj(values)))),
                               src_key_padding_mask=~attention_mask)
        if self.attention_pool is None:
            mask = attention_mask.unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * mask).sum(1) / mask.sum(1)
        else:
            scores = self.attention_pool(encoded).squeeze(-1).masked_fill(~attention_mask, -torch.inf)
            attended = (F.softmax(scores, dim=1).unsqueeze(-1) * encoded).sum(1)
            maximum = encoded.masked_fill(~attention_mask.unsqueeze(-1), -torch.inf).amax(1)
            pooled = torch.cat((attended, maximum), dim=-1)
        intent_logits = self.intent_head(pooled)
        if self.intent_embeddings is not None:
            context = F.softmax(intent_logits, dim=-1) @ self.intent_embeddings.weight
            encoded = torch.cat((encoded, context.unsqueeze(1).expand(-1, encoded.shape[1], -1)), dim=-1)
        return {'intent_logits': intent_logits, 'slot_logits': self.slot_head(self.slot_norm(encoded))}

    def loss(self, outputs, intent_targets, slot_targets, mask, *, intent_label_smoothing=0.0):
        if (isinstance(intent_label_smoothing, bool)
                or not isinstance(intent_label_smoothing, (int, float))
                or not math.isfinite(intent_label_smoothing)
                or not 0 <= intent_label_smoothing < 1):
            raise ValueError('Intent label smoothing must be finite and in [0, 1)')
        logits, emissions = outputs['intent_logits'], outputs['slot_logits']
        self._validate_mask(mask, emissions.shape[:2])
        if (logits.shape != (mask.shape[0], self.num_intents)
                or emissions.shape != (*mask.shape, self.num_tags)
                or intent_targets.shape != (mask.shape[0],) or intent_targets.dtype != torch.long
                or slot_targets.shape != mask.shape or slot_targets.dtype != torch.long):
            raise ValueError('Joint logits and target shapes/dtypes do not match')
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(emissions).all()):
            raise ValueError('Joint logits must be finite')
        if bool(((intent_targets < 0) | (intent_targets >= self.num_intents)).any()):
            raise ValueError('Intent target is outside the fitted label inventory')
        if bool(((slot_targets[mask] < 0) | (slot_targets[mask] >= self.num_tags)).any()):
            raise ValueError('Real slot target is outside the fitted label inventory')
        # Keep the historical call exactly when disabled. Evaluation always
        # reports ordinary CE, even if a training coefficient is supplied.
        if self.training and intent_label_smoothing > 0:
            intent_sum = F.cross_entropy(logits.float(), intent_targets, reduction='sum',
                                         label_smoothing=float(intent_label_smoothing))
        else:
            intent_sum = F.cross_entropy(logits.float(), intent_targets, reduction='sum')
        if self.crf is None:
            safe_targets = slot_targets.masked_fill(~mask, -100)
            slot_sum = F.cross_entropy(emissions.float().reshape(-1, self.num_tags),
                                      safe_targets.reshape(-1), reduction='sum', ignore_index=-100)
        else:
            slot_sum = self.crf.neg_log_likelihood(emissions.float(), slot_targets, mask).sum()
        examples, tokens = mask.shape[0], int(mask.sum().item())
        intent_loss, slot_loss = intent_sum / examples, slot_sum / tokens
        total = intent_loss + slot_loss
        if not bool(torch.isfinite(total)):
            raise ValueError('Joint loss must be finite')
        return {'total': total, 'intent_sum': intent_sum, 'slot_sum': slot_sum,
                'num_examples': examples, 'num_tokens': tokens,
                'intent_loss': intent_loss, 'slot_loss': slot_loss}

    def decode(self, outputs, mask):
        self._validate_mask(mask, outputs['slot_logits'].shape[:2])
        if self.crf is not None:
            return self.crf.decode(outputs['slot_logits'].float(), mask)
        return outputs['slot_logits'].argmax(-1).masked_fill(~mask, 0)


def count_parameters(model):
    return {'total': sum(parameter.numel() for parameter in model.parameters()),
            'trainable': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}


def enforce_parameter_cap(model, cap=20_000_000):
    if type(cap) is not int or cap <= 0:
        raise ValueError('Parameter cap must be a positive integer')
    counts = count_parameters(model)
    if counts['total'] > cap:
        raise ValueError(f"Complete model exceeds {cap:,} total parameters: {counts['total']:,}")
    return counts

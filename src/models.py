"""Train-from-scratch joint intent classification and BIO sequence tagging.

The model accepts only utterance token IDs, lengths, and the corresponding
padding mask. Targets and evaluation sidecars never enter the forward pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass(frozen=True)
class BaselineConfig:
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 1
    dropout: float = 0.2

    def __post_init__(self) -> None:
        for name in ("embedding_dim", "hidden_dim", "num_layers"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (float, int)):
            raise ValueError("dropout must be a finite number in [0, 1)")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be a finite number in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> BaselineConfig:
        if not isinstance(values, dict) or set(values) != {"embedding_dim", "hidden_dim", "num_layers", "dropout"}:
            raise ValueError("Baseline configuration must contain exactly the four architecture fields")
        return cls(**values)


class JointBiLSTM(nn.Module):
    """Shared packed BiLSTM with max-pooled intent and per-token slot heads."""

    def __init__(self, vocab_size: int, num_intents: int, num_tags: int,
                 config: BaselineConfig | None = None) -> None:
        super().__init__()
        for name, value in (("vocab_size", vocab_size), ("num_intents", num_intents), ("num_tags", num_tags)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if vocab_size < 2:
            raise ValueError("vocab_size must include PAD and UNK")
        self.config = BaselineConfig() if config is None else config
        if not isinstance(self.config, BaselineConfig):
            raise TypeError("config must be a BaselineConfig")
        self.vocab_size = vocab_size
        self.num_intents = num_intents
        self.num_tags = num_tags
        self.embedding = nn.Embedding(vocab_size, self.config.embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(self.config.dropout)
        self.encoder = nn.LSTM(
            input_size=self.config.embedding_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
        )
        self.output_dropout = nn.Dropout(self.config.dropout)
        self.intent_head = nn.Linear(2 * self.config.hidden_dim, num_intents)
        self.slot_head = nn.Linear(2 * self.config.hidden_dim, num_tags)

    def _validate_inputs(self, input_ids: torch.Tensor, lengths: torch.Tensor,
                         attention_mask: torch.Tensor | None) -> torch.Tensor:
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("input_ids must be a tensor with shape [batch, tokens]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must contain integer indices (int32 or int64)")
        batch, width = input_ids.shape
        if batch == 0 or width == 0:
            raise ValueError("input_ids must contain a nonempty batch and token dimension")
        if not isinstance(lengths, torch.Tensor) or lengths.ndim != 1 or len(lengths) != batch:
            raise ValueError("lengths must have shape [batch]")
        if lengths.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("lengths must contain integers")
        if torch.any(lengths <= 0).item() or torch.any(lengths > width).item():
            raise ValueError("Every length must be positive and no greater than the padded width")
        if torch.any(input_ids < 0).item() or torch.any(input_ids >= self.vocab_size).item():
            raise ValueError("input_ids contain an index outside the vocabulary")
        mask = torch.arange(width, device=input_ids.device).unsqueeze(0) < lengths.to(input_ids.device).unsqueeze(1)
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            if attention_mask.dtype not in (torch.bool, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise ValueError("attention_mask must contain Boolean or integer 0/1 values")
            if not torch.equal(attention_mask.to(input_ids.device), mask.to(attention_mask.dtype)):
                raise ValueError("attention_mask must be the right-padded prefix defined by lengths")
        if torch.any(input_ids.masked_select(~mask) != 0).item():
            raise ValueError("All padded input positions must use PAD index 0")
        if torch.any(input_ids.masked_select(mask) == 0).item():
            raise ValueError("Real token positions must not use PAD index 0")
        return mask

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        mask = self._validate_inputs(input_ids, lengths, attention_mask)
        embedded = self.embedding_dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(embedded, lengths.detach().to(device="cpu", dtype=torch.long),
                                      batch_first=True, enforce_sorted=False)
        packed_output, _ = self.encoder(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=input_ids.shape[1])
        output = self.output_dropout(output)
        pooled = output.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(dim=1)
        return {"intent_logits": self.intent_head(pooled), "slot_logits": self.slot_head(output)}


def parameter_counts(model: nn.Module) -> dict[str, int]:
    """Count every unique parameter, including frozen embeddings in the total."""
    return {"total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}


def enforce_parameter_cap(model: nn.Module, limit: int = 20_000_000) -> dict[str, int]:
    if type(limit) is not int or limit <= 0:
        raise ValueError("Parameter limit must be a positive integer")
    counts = parameter_counts(model)
    if counts["total"] > limit:
        raise ValueError(f"Model has {counts['total']} total parameters, exceeding the limit {limit}")
    return counts


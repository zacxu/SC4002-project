"""Optional train-from-scratch additions to the frozen joint BiLSTM baseline.

All features are derived from the unchanged utterance. Gold intent and BIO
labels enter only ``loss``; predicted intent probabilities may condition the
slot head. Character mappings must be fitted on original training records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.data import PreparedExample, SLOT_IGNORE_INDEX, collate_examples
from src.models import JointBiLSTM
from src.training import LossBundle, compute_losses


@dataclass(frozen=True)
class Stage2Config:
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 1
    dropout: float = 0.3
    use_char: bool = False
    char_embedding_dim: int = 32
    char_channels: int = 64
    char_kernel_size: int = 3
    use_crf: bool = False
    intent_conditioning: bool = False
    intent_context_dim: int = 64

    def __post_init__(self) -> None:
        for name in ("embedding_dim", "hidden_dim", "num_layers", "char_embedding_dim",
                     "char_channels", "char_kernel_size", "intent_context_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.char_kernel_size % 2 != 1:
            raise ValueError("char_kernel_size must be odd for centered same-width convolution")
        for name in ("use_char", "use_crf", "intent_conditioning"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be Boolean")
        if (isinstance(self.dropout, bool) or not isinstance(self.dropout, (float, int))
                or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1):
            raise ValueError("dropout must be finite and in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Stage2Config:
        if not isinstance(values, dict) or set(values) != {field.name for field in fields(cls)}:
            raise ValueError("Stage2 configuration must contain exactly the architecture fields")
        return cls(**values)


def _validate_char_vocab(char_to_id: dict[str, int]) -> None:
    if (not isinstance(char_to_id, dict) or char_to_id.get("<PAD>") != 0
            or char_to_id.get("<UNK>") != 1
            or any(type(value) is not int for value in char_to_id.values())
            or sorted(char_to_id.values()) != list(range(len(char_to_id)))
            or any(len(key) != 1 for key in char_to_id if key not in {"<PAD>", "<UNK>"})):
        raise ValueError("Character vocabulary must use PAD=0, UNK=1 and contiguous unique IDs")


def fit_char_vocab(prepared_train: Sequence[PreparedExample]) -> dict[str, int]:
    """Fit case-preserving character IDs on training utterance tokens only."""
    if not prepared_train or any(item.example.partition != "train" for item in prepared_train):
        raise ValueError("Character vocabulary fitting requires nonempty training records only")
    ids = [item.example.record_id for item in prepared_train]
    if len(ids) != len(set(ids)):
        raise ValueError("Character vocabulary fitting cannot repeat training records")
    characters: set[str] = set()
    for item in prepared_train:
        for token in item.tokens:
            if not token.text or item.example.utt[token.start:token.end] != token.text:
                raise ValueError("Character features must be exact utterance token slices")
            characters.update(token.text)
    return {"<PAD>": 0, "<UNK>": 1, **{char: index + 2 for index, char in enumerate(sorted(characters))}}


def collate_stage2(examples: Sequence[PreparedExample], char_to_id: dict[str, int]) -> dict[str, Any]:
    """Preserve the shared batch contract and add untruncated character IDs."""
    _validate_char_vocab(char_to_id)
    batch = collate_examples(examples)
    max_chars = max(len(token.text) for item in examples for token in item.tokens)
    char_ids = torch.zeros((*batch["input_ids"].shape, max_chars), dtype=torch.long)
    for row, item in enumerate(examples):
        for column, token in enumerate(item.tokens):
            if item.example.utt[token.start:token.end] != token.text:
                raise ValueError("Character features must be exact utterance token slices")
            char_ids[row, column, :len(token.text)] = torch.tensor(
                [char_to_id.get(character, 1) for character in token.text], dtype=torch.long)
    batch["char_ids"] = char_ids
    return batch


class CharacterCNN(nn.Module):
    """Centered character convolution and max pooling over real characters."""

    def __init__(self, vocab_size: int, embedding_dim: int, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convolution = nn.Conv1d(embedding_dim, channels, kernel_size, padding=kernel_size // 2)

    def forward(self, char_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        if (not isinstance(char_ids, torch.Tensor) or char_ids.ndim != 3
                or char_ids.shape[:2] != token_mask.shape or char_ids.shape[-1] == 0
                or char_ids.dtype != torch.long):
            raise ValueError("char_ids must be long [batch, tokens, characters] matching the token mask")
        if torch.any(char_ids < 0).item() or torch.any(char_ids >= self.vocab_size).item():
            raise ValueError("Character index outside vocabulary")
        char_mask = char_ids != 0
        lengths = char_mask.sum(-1)
        expected = torch.arange(char_ids.shape[-1], device=char_ids.device) < lengths.unsqueeze(-1)
        if not torch.equal(char_mask, expected) or not torch.equal(lengths > 0, token_mask):
            raise ValueError("Characters must be nonempty right-padded prefixes for precisely the real tokens")
        flat = char_ids.flatten(0, 1)
        features = F.relu(self.convolution(self.embedding(flat).transpose(1, 2)))
        features = features.masked_fill(~char_mask.flatten(0, 1).unsqueeze(1), -torch.inf).amax(-1)
        features = torch.where(lengths.flatten().unsqueeze(1) > 0, features, torch.zeros_like(features))
        return features.reshape(*token_mask.shape, -1)


class BIOCRF(nn.Module):
    """Linear-chain CRF with exact BIO legality in likelihood and decoding.

    Transition indices are [previous, next]. ``I-x`` may only follow ``B-x``
    or ``I-x`` and cannot start a sequence. Every I tag must have a B partner,
    so each destination has a reachable predecessor after the first token;
    exact -infinity constraints therefore avoid all-infinite logsumexp rows.
    """

    def __init__(self, tag_names: Sequence[str]) -> None:
        super().__init__()
        names = list(tag_names)
        if (not names or names[0] != "O" or len(names) != len(set(names))
                or any(name != "O" and (len(name) < 3 or name[:2] not in {"B-", "I-"}) for name in names)):
            raise ValueError("Tags must be unique BIO names with O at index zero")
        for name in names:
            if name.startswith(("B-", "I-")) and ("B-" + name[2:] not in names or "I-" + name[2:] not in names):
                raise ValueError("Each slot type requires both B and I tags")
        self.tag_names = tuple(names)
        self.num_tags = len(names)
        start_allowed = torch.tensor([not name.startswith("I-") for name in names], dtype=torch.bool)
        transition_allowed = torch.tensor([
            [not next_name.startswith("I-") or previous_name in {"B-" + next_name[2:], next_name}
             for next_name in names] for previous_name in names], dtype=torch.bool)
        self.register_buffer("start_allowed", start_allowed)
        self.register_buffer("transition_allowed", transition_allowed)
        self.start_transitions = nn.Parameter(torch.zeros(self.num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(self.num_tags))
        self.transitions = nn.Parameter(torch.zeros(self.num_tags, self.num_tags))

    def _validate(self, emissions: torch.Tensor, mask: torch.Tensor) -> None:
        if (emissions.ndim != 3 or emissions.shape[-1] != self.num_tags
                or emissions.shape[0] == 0 or emissions.shape[1] == 0
                or mask.shape != emissions.shape[:2] or mask.dtype != torch.bool):
            raise ValueError("CRF requires emissions [batch, tokens, tags] and matching Boolean mask")
        if not bool(torch.isfinite(emissions).all()):
            raise ValueError("CRF emissions must be finite")
        if not bool(mask[:, 0].all()) or bool((mask[:, 1:] & ~mask[:, :-1]).any()):
            raise ValueError("CRF mask must be nonempty right-padded prefixes")

    def _scores(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.start_transitions.masked_fill(~self.start_allowed, -torch.inf),
                self.transitions.masked_fill(~self.transition_allowed, -torch.inf))

    def log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._validate(emissions, mask)
        start, transitions = self._scores()
        alpha = start + emissions[:, 0]
        for position in range(1, emissions.shape[1]):
            candidates = alpha.unsqueeze(2) + transitions.unsqueeze(0)
            next_alpha = torch.logsumexp(candidates, dim=1) + emissions[:, position]
            alpha = torch.where(mask[:, position].unsqueeze(1), next_alpha, alpha)
        return torch.logsumexp(alpha + self.end_transitions, dim=1)

    def gold_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._validate(emissions, mask)
        if tags.shape != mask.shape or tags.dtype != torch.long:
            raise ValueError("CRF targets must be matching long tensors")
        if bool(((tags[mask] < 0) | (tags[mask] >= self.num_tags)).any()):
            raise ValueError("CRF real targets contain invalid tag IDs")
        safe_tags = tags.masked_fill(~mask, 0)
        if not bool(self.start_allowed[safe_tags[:, 0]].all()):
            raise ValueError("Gold BIO sequence starts with I")
        if bool((~self.transition_allowed[safe_tags[:, :-1], safe_tags[:, 1:]] & mask[:, 1:]).any()):
            raise ValueError("Gold BIO sequence contains an illegal transition")
        rows = torch.arange(emissions.shape[0], device=emissions.device)
        score = self.start_transitions[safe_tags[:, 0]] + emissions[rows, 0, safe_tags[:, 0]]
        for position in range(1, emissions.shape[1]):
            step = (self.transitions[safe_tags[:, position - 1], safe_tags[:, position]]
                    + emissions[rows, position, safe_tags[:, position]])
            score = score + torch.where(mask[:, position], step, torch.zeros_like(step))
        last_tags = safe_tags.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)
        return score + self.end_transitions[last_tags]

    def neg_log_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
        """Return one summed-over-token negative log likelihood per sequence."""
        return self.log_partition(emissions, mask) - self.gold_score(emissions, tags, mask)

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._validate(emissions, mask)
        start, transitions = self._scores()
        score = start + emissions[:, 0]
        backpointers = []
        for position in range(1, emissions.shape[1]):
            candidate, previous = (score.unsqueeze(2) + transitions.unsqueeze(0)).max(dim=1)
            updated = candidate + emissions[:, position]
            score = torch.where(mask[:, position].unsqueeze(1), updated, score)
            backpointers.append(previous)
        best = (score + self.end_transitions).argmax(dim=1)
        decoded = torch.zeros(mask.shape, dtype=torch.long, device=emissions.device)
        # Batched reverse recurrence skips padded suffixes; the active best tag
        # first appears at each row's actual final token, without CPU roundtrips.
        for position in range(emissions.shape[1] - 1, -1, -1):
            decoded[:, position] = torch.where(mask[:, position], best, torch.zeros_like(best))
            if position:
                previous = backpointers[position - 1].gather(1, best.unsqueeze(1)).squeeze(1)
                best = torch.where(mask[:, position], previous, best)
        return decoded


class JointStage2(JointBiLSTM):
    """One shared utterance encoder, with independently ablatable additions."""

    def __init__(self, vocab_size: int, num_intents: int, tag_names: Sequence[str],
                 char_vocab_size: int, config: Stage2Config | None = None) -> None:
        nn.Module.__init__(self)
        for name, value in (("vocab_size", vocab_size), ("num_intents", num_intents),
                            ("char_vocab_size", char_vocab_size)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if vocab_size < 2 or char_vocab_size < 2:
            raise ValueError("Word and character vocabularies must include PAD and UNK")
        self.config = Stage2Config() if config is None else config
        if not isinstance(self.config, Stage2Config):
            raise TypeError("config must be a Stage2Config")
        # BIOCRF validates the label inventory even when the CRF option is off.
        crf = BIOCRF(tag_names)
        self.tag_names = tuple(tag_names)
        self.vocab_size, self.num_intents, self.num_tags = vocab_size, num_intents, len(tag_names)
        self.char_vocab_size = char_vocab_size
        self.embedding = nn.Embedding(vocab_size, self.config.embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(self.config.dropout)
        self.encoder = nn.LSTM(
            self.config.embedding_dim + (self.config.char_channels if self.config.use_char else 0),
            self.config.hidden_dim, num_layers=self.config.num_layers, batch_first=True,
            bidirectional=True, dropout=self.config.dropout if self.config.num_layers > 1 else 0.0)
        self.output_dropout = nn.Dropout(self.config.dropout)
        self.intent_head = nn.Linear(2 * self.config.hidden_dim, num_intents)
        self.slot_head = nn.Linear(2 * self.config.hidden_dim + (
            self.config.intent_context_dim if self.config.intent_conditioning else 0), self.num_tags)
        self.char_encoder = CharacterCNN(char_vocab_size, self.config.char_embedding_dim,
                                         self.config.char_channels, self.config.char_kernel_size) if self.config.use_char else None
        self.intent_embeddings = nn.Embedding(num_intents, self.config.intent_context_dim) if self.config.intent_conditioning else None
        self.crf = crf if self.config.use_crf else None

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor,
                attention_mask: torch.Tensor | None = None,
                char_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        mask = self._validate_inputs(input_ids, lengths, attention_mask)
        embedded = self.embedding(input_ids)
        if self.char_encoder is not None:
            if char_ids is None:
                raise ValueError("Character-enabled model requires utterance-derived char_ids")
            embedded = torch.cat((embedded, self.char_encoder(char_ids, mask)), dim=-1)
        packed = pack_padded_sequence(self.embedding_dropout(embedded), lengths.detach().cpu().long(),
                                      batch_first=True, enforce_sorted=False)
        encoded, _ = self.encoder(packed)
        output, _ = pad_packed_sequence(encoded, batch_first=True, total_length=input_ids.shape[1])
        output = self.output_dropout(output)
        pooled = output.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(dim=1)
        intent_logits = self.intent_head(pooled)
        if self.intent_embeddings is not None:
            predicted_context = F.softmax(intent_logits, dim=-1) @ self.intent_embeddings.weight
            output = torch.cat((output, predicted_context.unsqueeze(1).expand(-1, output.shape[1], -1)), dim=-1)
        return {"intent_logits": intent_logits, "slot_logits": self.slot_head(output)}

    def loss(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> LossBundle:
        # Reuse all baseline CE target/shape/finite validation. CRF replaces only
        # the slot objective, normalized by real tokens just as baseline slot CE.
        ce = compute_losses(outputs, batch)
        if self.crf is None:
            return ce
        slot_sum = self.crf.neg_log_likelihood(outputs["slot_logits"], batch["slot_labels"],
                                               batch["attention_mask"]).sum()
        slot_loss = slot_sum / ce.num_tokens
        total = ce.intent + slot_loss
        if not bool(torch.isfinite(total)):
            raise ValueError("Nonfinite joint CRF loss")
        return LossBundle(total, ce.intent, slot_loss, ce.intent_sum, slot_sum, ce.num_examples, ce.num_tokens)

    def decode(self, outputs: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        if self.crf is not None:
            return self.crf.decode(outputs["slot_logits"], mask)
        return outputs["slot_logits"].argmax(-1).masked_fill(~mask, 0)

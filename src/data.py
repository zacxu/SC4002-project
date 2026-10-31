"""Original MASSIVE splits, exact annotations, train-fitted mappings, and batches.

Only token IDs, lengths, and masks are model features. Gold labels, raw text,
offsets, and record IDs remain explicit targets or evaluation sidecars.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from src.schema import IntentSlotPrediction, SlotSpan


EXPECTED_COUNTS = {"train": 11514, "dev": 2033, "test": 2974}
TOKEN_PATTERN = r"\w+|[^\w\s]"
ANNOTATION_PATTERN = re.compile(r"\[([A-Za-z_][A-Za-z_0-9]*)[ \t]*:[ \t]*(.*?)\]", re.DOTALL)
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SLOT_IGNORE_INDEX = -100


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class UtteranceExample:
    record_id: str
    partition: str
    utt: str
    intent: str
    annot_utt: str
    slots: tuple[SlotSpan, ...]

    def gold_prediction(self) -> IntentSlotPrediction:
        return IntentSlotPrediction(self.record_id, self.intent, self.slots, utt=self.utt)


@dataclass(frozen=True)
class FittedMappings:
    token_to_id: dict[str, int]
    intent_to_id: dict[str, int]
    tag_to_id: dict[str, int]
    slot_types: tuple[str, ...]
    training_ids_sha256: str
    training_records_sha256: str
    training_count: int

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "fit_partition": "train", "training_count": self.training_count,
                "training_ids_sha256": self.training_ids_sha256, "training_records_sha256": self.training_records_sha256,
                "tokenizer_pattern": TOKEN_PATTERN, "normalization": "none", "truncation": "none",
                "token_to_id": self.token_to_id, "intent_to_id": self.intent_to_id,
                "tag_to_id": self.tag_to_id, "slot_types": list(self.slot_types),
                "pad_token_id": 0, "unknown_token_id": 1, "slot_ignore_index": SLOT_IGNORE_INDEX}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FittedMappings:
        if data["schema_version"] != 1 or data["fit_partition"] != "train":
            raise ValueError("Unsupported mappings or fitting partition")
        if data["tokenizer_pattern"] != TOKEN_PATTERN or data["normalization"] != "none" or data["truncation"] != "none":
            raise ValueError("Saved mappings use a different preprocessing protocol")
        for key in ("token_to_id", "intent_to_id", "tag_to_id"):
            values = list(data[key].values())
            if any(type(value) is not int for value in values) or sorted(values) != list(range(len(values))):
                raise ValueError(f"Mapping indices must be unique and contiguous: {key}")
        if data["token_to_id"].get(PAD_TOKEN) != 0 or data["token_to_id"].get(UNK_TOKEN) != 1:
            raise ValueError("Reserved vocabulary indices do not match")
        if data["tag_to_id"].get("O") != 0 or data["slot_ignore_index"] != SLOT_IGNORE_INDEX:
            raise ValueError("Slot target conventions do not match")
        return cls(dict(data["token_to_id"]), dict(data["intent_to_id"]), dict(data["tag_to_id"]),
                   tuple(data["slot_types"]), data["training_ids_sha256"], data["training_records_sha256"], data["training_count"])


@dataclass(frozen=True)
class PreparedExample:
    example: UtteranceExample
    tokens: tuple[Token, ...]
    token_ids: tuple[int, ...]
    tag_ids: tuple[int, ...]
    intent_id: int


def _json_hash(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def tokenize_with_offsets(utt: str) -> tuple[Token, ...]:
    """Case-preserving Unicode word/punctuation tokens; no gold-dependent cuts."""
    return tuple(Token(match.group(), match.start(), match.end())
                 for match in re.finditer(TOKEN_PATTERN, utt, flags=re.UNICODE))


def parse_annotations(utt: str, annot_utt: str) -> tuple[SlotSpan, ...]:
    """Remove only annotation syntax, then require exact reconstruction of utt."""
    parts: list[str] = []
    slots: list[SlotSpan] = []
    annotation_cursor = 0
    plain_cursor = 0
    for match in ANNOTATION_PATTERN.finditer(annot_utt):
        literal = annot_utt[annotation_cursor:match.start()]
        parts.append(literal)
        plain_cursor += len(literal)
        value = match.group(2)
        slots.append(SlotSpan(match.group(1), value, plain_cursor, plain_cursor + len(value)))
        parts.append(value)
        plain_cursor += len(value)
        annotation_cursor = match.end()
    parts.append(annot_utt[annotation_cursor:])
    if "".join(parts) != utt:
        raise ValueError("Removing annotation syntax does not reproduce utt exactly")
    for slot in slots:
        if utt[slot.start:slot.end] != slot.text:
            raise ValueError("Slot offsets do not reproduce the annotated surface text")
    return tuple(slots)


def load_massive(path: Path, expected_counts: dict[str, int] | None = None) -> dict[str, tuple[UtteranceExample, ...]]:
    """Preserve source order within each original partition; never filter/dedupe."""
    expected = EXPECTED_COUNTS if expected_counts is None else expected_counts
    buckets: dict[str, list[UtteranceExample]] = {key: [] for key in EXPECTED_COUNTS}
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL record at line {line_number}; no records may be silently skipped")
            row = json.loads(line)
            try:
                if row["locale"] != "en-US" or row["partition"] not in buckets:
                    raise ValueError("Unexpected locale or partition")
                for key in ("id", "utt", "intent", "annot_utt"):
                    if not isinstance(row[key], str) or (key != "annot_utt" and not row[key]):
                        raise ValueError(f"Expected a nonempty string field: {key}")
                if row["id"] in seen:
                    raise ValueError("Duplicate record ID; refusing to deduplicate")
                seen.add(row["id"])
                example = UtteranceExample(row["id"], row["partition"], row["utt"], row["intent"], row["annot_utt"],
                                           parse_annotations(row["utt"], row["annot_utt"]))
                buckets[row["partition"]].append(example)
            except (KeyError, ValueError) as error:
                raise ValueError(f"Invalid original record at line {line_number}, ID {row.get('id')!r}: {error}") from error
    counts = {key: len(value) for key, value in buckets.items()}
    if counts != expected:
        raise ValueError(f"Original split count mismatch: observed {counts}, expected {expected}")
    return {key: tuple(value) for key, value in buckets.items()}


def fit_mappings(training: Sequence[UtteranceExample]) -> FittedMappings:
    """Fit every learned vocabulary/label mapping from training records only."""
    if not training or any(example.partition != "train" for example in training):
        raise ValueError("Mappings must be fitted on a nonempty train-only sequence")
    if len({example.record_id for example in training}) != len(training):
        raise ValueError("Duplicate training IDs")
    counts = Counter(token.text for example in training for token in tokenize_with_offsets(example.utt))
    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])):
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
    intents = {name: index for index, name in enumerate(sorted({example.intent for example in training}))}
    slot_types = tuple(sorted({slot.slot_type for example in training for slot in example.slots}))
    tags = {"O": 0}
    for slot_type in slot_types:
        for prefix in ("B", "I"):
            tags[f"{prefix}-{slot_type}"] = len(tags)
    return FittedMappings(vocabulary, intents, tags, slot_types,
                         _json_hash([example.record_id for example in training]),
                         _json_hash([[e.record_id, e.utt, e.intent, e.annot_utt] for e in training]), len(training))


def gold_bio_tags(example: UtteranceExample, tokens: Sequence[Token]) -> tuple[str, ...]:
    tags = ["O"] * len(tokens)
    previous_end = -1
    for slot in example.slots:
        if slot.start < previous_end or example.utt[slot.start:slot.end] != slot.text:
            raise ValueError(f"Overlapping or inconsistent slot: {example.record_id}")
        positions = [index for index, token in enumerate(tokens) if token.end > slot.start and token.start < slot.end]
        if not positions or tokens[positions[0]].start != slot.start or tokens[positions[-1]].end != slot.end:
            raise ValueError(f"Gold boundary cannot be represented by input-only tokenization: {example.record_id}")
        for number, position in enumerate(positions):
            if tags[position] != "O":
                raise ValueError(f"Overlapping token labels: {example.record_id}")
            tags[position] = f"{'B' if number == 0 else 'I'}-{slot.slot_type}"
        previous_end = slot.end
    return tuple(tags)


def decode_bio(utt: str, tokens: Sequence[Token], tags: Sequence[str]) -> tuple[SlotSpan, ...]:
    """Decode exact surface slices; orphan/type-changing I starts a new slot."""
    if len(tokens) != len(tags):
        raise ValueError("BIO tag length must equal the unpadded token length")
    slots: list[SlotSpan] = []
    active_type: str | None = None
    active_start = active_end = 0

    def close() -> None:
        nonlocal active_type
        if active_type is not None:
            slots.append(SlotSpan(active_type, utt[active_start:active_end], active_start, active_end))
        active_type = None

    last_end = -1
    for token, tag in zip(tokens, tags):
        if token.start < last_end or token.end <= token.start or utt[token.start:token.end] != token.text:
            raise ValueError("Token offsets must be ordered exact slices of utt")
        last_end = token.end
        if tag == "O":
            close()
            continue
        if "-" not in tag:
            raise ValueError(f"Invalid BIO tag: {tag!r}")
        prefix, slot_type = tag.split("-", 1)
        if prefix not in {"B", "I"} or not slot_type:
            raise ValueError(f"Invalid BIO tag: {tag!r}")
        if prefix == "B" or active_type != slot_type:
            close()
            active_type, active_start = slot_type, token.start
        active_end = token.end
    close()
    return tuple(slots)


def prepare_example(example: UtteranceExample, mappings: FittedMappings) -> PreparedExample:
    tokens = tokenize_with_offsets(example.utt)
    if not tokens:
        raise ValueError(f"Utterance has no non-whitespace tokens: {example.record_id}")
    tags = gold_bio_tags(example, tokens)
    try:
        return PreparedExample(example, tokens,
                               tuple(mappings.token_to_id.get(token.text, 1) for token in tokens),
                               tuple(mappings.tag_to_id[tag] for tag in tags), mappings.intent_to_id[example.intent])
    except KeyError as error:
        raise ValueError(f"Label not observed in training for record {example.record_id}: {error}") from error


def prepare_split(examples: Sequence[UtteranceExample], mappings: FittedMappings) -> tuple[PreparedExample, ...]:
    return tuple(prepare_example(example, mappings) for example in examples)


def collate_examples(examples: Sequence[PreparedExample]) -> dict[str, Any]:
    """Dynamic right padding; no maximum-length truncation or dropped records."""
    import torch

    if not examples:
        raise ValueError("Cannot collate an empty batch")
    lengths = torch.tensor([len(example.tokens) for example in examples], dtype=torch.long)
    shape = (len(examples), int(lengths.max()))
    input_ids = torch.zeros(shape, dtype=torch.long)
    slot_labels = torch.full(shape, SLOT_IGNORE_INDEX, dtype=torch.long)
    for row, example in enumerate(examples):
        size = len(example.tokens)
        input_ids[row, :size] = torch.tensor(example.token_ids, dtype=torch.long)
        slot_labels[row, :size] = torch.tensor(example.tag_ids, dtype=torch.long)
    mask = torch.arange(shape[1])[None, :] < lengths[:, None]
    return {"input_ids": input_ids, "attention_mask": mask, "lengths": lengths,
            "intent_labels": torch.tensor([example.intent_id for example in examples], dtype=torch.long),
            "slot_labels": slot_labels, "record_ids": [example.example.record_id for example in examples],
            "examples": tuple(example.example for example in examples), "tokens": tuple(example.tokens for example in examples)}


def write_predictions(path: Path, predictions: Iterable[IntentSlotPrediction]) -> None:
    """One ID-associated intent plus slot list per line; preserve all predictions."""
    seen: set[str] = set()
    rows = []
    for prediction in predictions:
        if prediction.record_id in seen:
            raise ValueError("Duplicate prediction record ID")
        seen.add(prediction.record_id)
        rows.append(json.dumps(prediction.to_dict(), ensure_ascii=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def read_predictions(path: Path) -> tuple[IntentSlotPrediction, ...]:
    with path.open(encoding="utf-8") as handle:
        predictions = tuple(IntentSlotPrediction.from_dict(json.loads(line)) for line in handle)
    if len({prediction.record_id for prediction in predictions}) != len(predictions):
        raise ValueError("Duplicate prediction record ID")
    return predictions


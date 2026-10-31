"""Shared typed prediction format; offsets are Python character offsets in utt."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SlotSpan:
    slot_type: str
    text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.slot_type, str) or not self.slot_type:
            raise ValueError("Slot type must be a nonempty string")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("Slot span text must be a nonempty string")
        if type(self.start) is not int or type(self.end) is not int:
            raise ValueError("Slot offsets must be integers")
        if self.start < 0 or self.end <= self.start or self.end - self.start != len(self.text):
            raise ValueError("Slot offsets must describe the exact nonempty surface text")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IntentSlotPrediction:
    record_id: str
    intent: str
    slots: tuple[SlotSpan, ...]
    # Gold references retain raw input for evaluator slice/bounds validation.
    # It is an audit sidecar, not a model feature or prediction-equality key.
    utt: str | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id:
            raise ValueError("Prediction record ID must be a nonempty string")
        if not isinstance(self.intent, str) or not self.intent:
            raise ValueError("Prediction intent must be a nonempty string")
        object.__setattr__(self, "slots", tuple(self.slots))
        if any(not isinstance(slot, SlotSpan) for slot in self.slots):
            raise ValueError("Prediction slots must contain SlotSpan values")
        if self.utt is not None:
            if not isinstance(self.utt, str):
                raise ValueError("Utterance sidecar must be a string")
            if any(self.utt[slot.start:slot.end] != slot.text for slot in self.slots):
                raise ValueError("Slot must match its exact original utterance slice")

    def to_dict(self) -> dict[str, Any]:
        result = {"record_id": self.record_id, "intent": self.intent,
                  "slots": [slot.to_dict() for slot in self.slots]}
        if self.utt is not None:
            result["utt"] = self.utt
        return result

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> IntentSlotPrediction:
        return cls(record_id=record["record_id"], intent=record["intent"],
                   slots=tuple(SlotSpan(**slot) for slot in record["slots"]), utt=record.get("utt"))

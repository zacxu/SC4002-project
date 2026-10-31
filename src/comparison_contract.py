"""Tokenizer-independent scoring and pinned MASSIVE data shared by both notebooks.

This module intentionally uses only the Python standard library so its exact
source can be embedded in a standalone submission notebook. Scores are fractions.
Character offsets refer to the original Python Unicode string, with exclusive
ends. The recorded data digest pins the project's original release bytes; it is
not an independently published authenticity certificate.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile
from urllib.request import Request, urlopen


EVALUATOR_VERSION = "exact_character_span_v2"
DATASET_SHA256 = "c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e"
EXPECTED_COUNTS = {"train": 11514, "dev": 2033, "test": 2974}
SOURCE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz"
ENGLISH_MEMBER = "1.0/data/en-US.jsonl"
_CONTRACT_ANNOTATION = re.compile(r"\[([A-Za-z_][A-Za-z_0-9]*)[ \t]*:[ \t]*(.*?)\]", re.DOTALL)
_CONTRACT_MAX_BYTES = 128 * 1024 * 1024


def _contract_json_hash(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def parse_annotated_spans(utt, annot_utt):
    """Strip annotation syntax only, requiring exact original-text reconstruction."""
    if not isinstance(utt, str) or not isinstance(annot_utt, str):
        raise TypeError("Utterance and annotated utterance must be strings")
    pieces, spans = [], []
    annotation_cursor = plain_cursor = 0
    for match in _CONTRACT_ANNOTATION.finditer(annot_utt):
        before = annot_utt[annotation_cursor:match.start()]
        pieces.append(before)
        plain_cursor += len(before)
        text = match.group(2)
        if not text:
            raise ValueError("An annotated slot cannot have empty text")
        spans.append({"slot_type": match.group(1), "text": text,
                      "start": plain_cursor, "end": plain_cursor + len(text)})
        pieces.append(text)
        plain_cursor += len(text)
        annotation_cursor = match.end()
    pieces.append(annot_utt[annotation_cursor:])
    if "".join(pieces) != utt:
        raise ValueError("Removing annotation syntax does not exactly reconstruct utt")
    return spans


def load_verified_massive(path):
    """Load only the pinned original bytes, preserving every partition and row."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Dataset must be a regular, non-symlink file")
    if not 0 < path.stat().st_size <= _CONTRACT_MAX_BYTES:
        raise ValueError("Dataset size is outside the accepted range")
    raw = path.read_bytes()
    observed_hash = hashlib.sha256(raw).hexdigest()
    if observed_hash != DATASET_SHA256:
        raise ValueError("Dataset SHA256 differs from the pinned original release; refusing reuse")
    splits = {name: [] for name in EXPECTED_COUNTS}
    seen = set()
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"Blank dataset row at line {number}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Dataset row {number} is not an object")
        for field in ("id", "utt", "intent", "annot_utt", "partition", "locale"):
            if not isinstance(row.get(field), str) or (field != "annot_utt" and not row[field]):
                raise ValueError(f"Invalid dataset field {field!r} at line {number}")
        if row["locale"] != "en-US" or row["partition"] not in splits:
            raise ValueError(f"Unexpected locale or partition at line {number}")
        if row["id"] in seen:
            raise ValueError(f"Duplicate dataset ID: {row['id']!r}")
        seen.add(row["id"])
        parse_annotated_spans(row["utt"], row["annot_utt"])
        splits[row["partition"]].append(row)
    counts = {name: len(rows) for name, rows in splits.items()}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Original split counts differ: {counts!r}")
    manifest = {
        "evaluator_version": EVALUATOR_VERSION,
        "dataset_sha256": observed_hash,
        "source_url": SOURCE_URL,
        "archive_member": ENGLISH_MEMBER,
        "counts": counts,
        "split_id_sha256": {name: _contract_json_hash([row["id"] for row in rows])
                            for name, rows in splits.items()},
        "split_content_sha256": {name: _contract_json_hash(rows)
                                 for name, rows in splits.items()},
        "identity_scope": "Pinned original bytes; original row order and every record field retained",
    }
    return splits, manifest


def ensure_verified_massive(path):
    """Reuse verified data or safely stream its exact member from the official URL.

    Existing mismatched files are never overwritten. A verified temporary file
    is published using an exclusive atomic hard link; archive-controlled names
    are never used for filesystem writes and extract/extractall are not used.
    """
    path = Path(path)
    if path.exists() or path.is_symlink():
        return load_verified_massive(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".massive-verified-", suffix=".part", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        found = False
        with os.fdopen(descriptor, "wb") as output:
            request = Request(SOURCE_URL, headers={"User-Agent": "SC4002-verified-comparison/2"})
            with urlopen(request, timeout=60) as response, tarfile.open(fileobj=response, mode="r|gz") as archive:
                for member in archive:
                    if member.name != ENGLISH_MEMBER:
                        continue
                    if not member.isfile() or not 0 < member.size <= _CONTRACT_MAX_BYTES:
                        raise ValueError("English archive member must be a bounded regular file")
                    source = archive.extractfile(member)
                    if source is None:
                        raise ValueError("English archive member could not be read")
                    written = 0
                    with source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            written += len(block)
                            if written > member.size:
                                raise ValueError("English archive member exceeds its declared size")
                            output.write(block)
                    if written != member.size:
                        raise ValueError("English archive member is truncated")
                    found = True
                    break
            output.flush()
            os.fsync(output.fileno())
        if not found:
            raise ValueError("Official archive does not contain its exact English member")
        verified = load_verified_massive(temporary)
        try:
            os.link(temporary, path)
        except FileExistsError:
            return load_verified_massive(path)
        return verified
    finally:
        temporary.unlink(missing_ok=True)


def _contract_field(value, key):
    if isinstance(value, dict):
        if key not in value:
            raise ValueError(f"Missing required field: {key}")
        return value[key]
    try:
        return getattr(value, key)
    except AttributeError as error:
        raise ValueError(f"Missing required field: {key}") from error


def _contract_index(records, source):
    indexed = {}
    for record in records:
        record_id = _contract_field(record, "record_id")
        intent = _contract_field(record, "intent")
        if not isinstance(record_id, str) or not record_id or not isinstance(intent, str) or not intent:
            raise ValueError(f"{source} record ID and intent must be nonempty strings")
        if record_id in indexed:
            raise ValueError(f"Duplicate {source} record ID: {record_id!r}")
        indexed[record_id] = record
    return indexed


def _contract_span_keys(record, utt):
    keys = set()
    for span in _contract_field(record, "slots"):
        slot_type, text, start, end = (_contract_field(span, name)
                                     for name in ("slot_type", "text", "start", "end"))
        if not isinstance(slot_type, str) or not slot_type or not isinstance(text, str) or not text:
            raise ValueError("Slot type and text must be nonempty strings")
        if type(start) is not int or type(end) is not int or not 0 <= start < end or end - start != len(text):
            raise ValueError("Slot character offsets must describe its exact nonempty text")
        if utt is not None and (end > len(utt) or utt[start:end] != text):
            raise ValueError("Slot offsets and text do not match the original utterance")
        key = (slot_type, start, end, text)
        if key in keys:
            raise ValueError("Duplicate identical slot occurrence")
        keys.add(key)
    return keys


def score_exact_spans(gold, predictions):
    """Score typed objects or dictionaries with exact IDs and span occurrences.

    Gold dictionaries may include ``utt``. If supplied, both gold and predicted
    spans are checked against it. Without raw text, schema integrity, exact
    boundaries and literal text equality still apply to every matching key.
    """
    gold_by_id = _contract_index(gold, "gold")
    predicted_by_id = _contract_index(predictions, "prediction")
    if not gold_by_id:
        raise ValueError("Cannot evaluate an empty gold split")
    missing = gold_by_id.keys() - predicted_by_id.keys()
    unexpected = predicted_by_id.keys() - gold_by_id.keys()
    if missing or unexpected:
        raise ValueError(f"Prediction IDs do not match gold IDs: missing={sorted(missing)!r}, unexpected={sorted(unexpected)!r}")
    correct = tp = fp = fn = 0
    for record_id, reference in gold_by_id.items():
        predicted = predicted_by_id[record_id]
        utt = reference.get("utt") if isinstance(reference, dict) else getattr(reference, "utt", None)
        if utt is not None and not isinstance(utt, str):
            raise ValueError("Raw gold utterance must be a string")
        reference_keys = _contract_span_keys(reference, utt)
        predicted_keys = _contract_span_keys(predicted, utt)
        correct += int(_contract_field(reference, "intent") == _contract_field(predicted, "intent"))
        tp += len(reference_keys & predicted_keys)
        fp += len(predicted_keys - reference_keys)
        fn += len(reference_keys - predicted_keys)
    denominator = 2 * tp + fp + fn
    return {"intent_accuracy": correct / len(gold_by_id),
            "slot_micro_f1": 2 * tp / denominator if denominator else 0.0,
            "slot_tp": tp, "slot_fp": fp, "slot_fn": fn,
            "num_examples": len(gold_by_id), "correct_intents": correct}


def bio_to_character_spans(utt, token_offsets, tags):
    """Decode BIO into original character coordinates without tokenization bias.

    Offsets accept ``(start,end)`` pairs, dictionaries, or objects with start/end.
    Orphan/type-changing I starts a new span, consistently for both architectures.
    """
    if not isinstance(utt, str):
        raise TypeError("Utterance must be a string")
    token_offsets, tags = list(token_offsets), list(tags)
    if len(token_offsets) != len(tags):
        raise ValueError("BIO tag count must equal unpadded token count")
    offsets, previous_end = [], 0
    for offset in token_offsets:
        if isinstance(offset, (tuple, list)) and len(offset) == 2:
            start, end = offset
        else:
            start, end = _contract_field(offset, "start"), _contract_field(offset, "end")
        if type(start) is not int or type(end) is not int or not previous_end <= start < end <= len(utt):
            raise ValueError("Token offsets must be ordered, nonoverlapping character slices")
        offsets.append((start, end))
        previous_end = end
    spans = []
    active_type = None
    active_start = active_end = 0

    def close():
        nonlocal active_type
        if active_type is not None:
            spans.append({"slot_type": active_type, "text": utt[active_start:active_end],
                          "start": active_start, "end": active_end})
        active_type = None

    for (start, end), tag in zip(offsets, tags):
        if not isinstance(tag, str):
            raise ValueError("BIO tags must be strings")
        if tag == "O":
            close()
            continue
        if len(tag) < 3 or tag[:2] not in {"B-", "I-"} or not tag[2:]:
            raise ValueError(f"Invalid BIO tag: {tag!r}")
        slot_type = tag[2:]
        if tag.startswith("B-") or active_type != slot_type:
            close()
            active_type, active_start = slot_type, start
        active_end = end
    close()
    return spans

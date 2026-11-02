"""Train-fitted Transformer data and bounded-memory word-vector initialization.

Only raw utterances determine model features. Annotation markup supplies targets
and is checked against exact character slices through the common evaluator.
Pretrained source vectors are initialization assets; only the fitted task table
is retained. This module never silently replaces missing pretrained vectors
with random vectors.
"""
from collections import Counter
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import torch

from src.comparison_contract import bio_to_character_spans, parse_annotated_spans


TRANSFORMER_TOKEN_PATTERN = r"\w+(?:['\u2019\-]\w+)*|[^\w\s]"
_TRANSFORMER_TOKEN_RE = re.compile(TRANSFORMER_TOKEN_PATTERN, re.UNICODE)
_VECTOR_POS_SUFFIX = re.compile(r"_(ADJ|ADP|ADV|AUX|CCONJ|DET|INTJ|NOUN|NUM|PART|PRON|PROPN|PUNCT|SCONJ|SYM|VERB|X)$")
WORD2VEC_SOURCE_URL = "https://vectors.nlpl.eu/repository/20/6.zip"
_MAX_VECTOR_ARCHIVE_BYTES = 2 * 1024**3
_MAX_VECTOR_MEMBER_BYTES = 3 * 1024**3


def transformer_tokens(utt):
    """Return raw-input tokens and their original Unicode character offsets."""
    if not isinstance(utt, str):
        raise TypeError('Utterance must be a string')
    matches = list(_TRANSFORMER_TOKEN_RE.finditer(utt))
    if not matches:
        raise ValueError('An utterance must contain at least one token')
    return [match.group(0) for match in matches], [(match.start(), match.end()) for match in matches]


def _target_tags(utt, offsets, slots):
    tags = ['O'] * len(offsets)
    last_end = -1
    for slot in slots:
        if slot['start'] < last_end:
            raise ValueError('Overlapping original slot targets')
        positions = [i for i, (start, end) in enumerate(offsets)
                     if end > slot['start'] and start < slot['end']]
        if (not positions or offsets[positions[0]][0] != slot['start']
                or offsets[positions[-1]][1] != slot['end']):
            raise ValueError('Gold slot boundary cannot be represented by raw-input tokenization')
        for number, position in enumerate(positions):
            if tags[position] != 'O':
                raise ValueError('Overlapping original slot targets')
            tags[position] = ('B-' if number == 0 else 'I-') + slot['slot_type']
        last_end = slot['end']
    if bio_to_character_spans(utt, offsets, tags) != slots:
        raise ValueError('Gold BIO conversion does not exactly reconstruct original spans')
    return tags


def build_data(rawsplits):
    """Fit mappings on train only and retain every supplied original record.

    Call the shared pinned-data loader first. Small bookkeeping populations are
    also accepted here for unit tests; this function never claims that counts
    alone authenticate an original dataset.
    """
    if not isinstance(rawsplits, dict) or set(rawsplits) != {'train', 'dev', 'test'}:
        raise ValueError('Supply the three original train/dev/test partitions')
    if not rawsplits['train']:
        raise ValueError('Training records must be nonempty')
    seen, parsed = set(), {}
    for partition in ('train', 'dev', 'test'):
        parsed[partition] = []
        for row in rawsplits[partition]:
            if not isinstance(row, dict) or any(not isinstance(row.get(key), str)
                    for key in ('id', 'utt', 'intent', 'annot_utt', 'partition')):
                raise ValueError('Invalid original record fields')
            if not row['id'] or not row['intent'] or row['partition'] != partition:
                raise ValueError('Original record identity or partition is inconsistent')
            if row['id'] in seen:
                raise ValueError('Duplicate original record ID')
            seen.add(row['id'])
            tokens, offsets = transformer_tokens(row['utt'])
            slots = parse_annotated_spans(row['utt'], row['annot_utt'])
            tags = _target_tags(row['utt'], offsets, slots)
            parsed[partition].append((row, tokens, offsets, slots, tags))
    words = sorted({token.casefold() for _, tokens, _, _, _ in parsed['train'] for token in tokens})
    chars = sorted({character for _, tokens, _, _, _ in parsed['train'] for token in tokens for character in token})
    intents = sorted({row['intent'] for row, _, _, _, _ in parsed['train']})
    types = sorted({slot['slot_type'] for _, _, _, slots, _ in parsed['train'] for slot in slots})
    word_to_id = {word: i for i, word in enumerate(['<PAD>', '<UNK>'] + words)}
    char_to_id = {char: i for i, char in enumerate(['<CPAD>', '<CUNK>'] + chars)}
    intent_to_id = {intent: i for i, intent in enumerate(intents)}
    tag_names = ['O'] + [prefix + '-' + slot_type for slot_type in types for prefix in ('B', 'I')]
    tag_to_id = {tag: i for i, tag in enumerate(tag_names)}
    encoded, gold_by_id, stats = {}, {}, {'fit_partition': 'train', 'tokenizer_pattern': TRANSFORMER_TOKEN_PATTERN,
        'word_normalization': 'casefold', 'character_normalization': 'none', 'truncation': 'none',
        'word_vocab_size': len(word_to_id), 'char_vocab_size': len(char_to_id),
        'intent_count': len(intents), 'slot_type_count': len(types), 'bio_tag_count': len(tag_names),
        'counts': {}, 'token_counts': {}, 'maximum_tokens': {}, 'maximum_characters_per_token': {},
        'oov_token_occurrences': {}, 'oov_utterances': {}}
    for partition, rows in parsed.items():
        encoded[partition] = []
        for original, tokens, offsets, slots, tags in rows:
            if original['intent'] not in intent_to_id or any(tag not in tag_to_id for tag in tags):
                raise ValueError('Held-out targets contain labels absent from the training inventory')
            record_id = original['id']
            encoded[partition].append({'record_id': record_id, 'utt': original['utt'],
                'token_offsets': offsets, 'word_ids': [word_to_id.get(token.casefold(), 1) for token in tokens],
                'intent_id': intent_to_id[original['intent']], 'slot_ids': [tag_to_id[tag] for tag in tags],
                'char_ids': [[char_to_id.get(character, 1) for character in token] for token in tokens]})
            gold_by_id[record_id] = {'record_id': record_id, 'utt': original['utt'],
                                     'intent': original['intent'], 'slots': slots}
        stats['counts'][partition] = len(rows)
        stats['token_counts'][partition] = sum(len(tokens) for _, tokens, _, _, _ in rows)
        stats['maximum_tokens'][partition] = max((len(tokens) for _, tokens, _, _, _ in rows), default=0)
        stats['maximum_characters_per_token'][partition] = max((len(token) for _, tokens, _, _, _ in rows for token in tokens), default=0)
        stats['oov_token_occurrences'][partition] = sum(index == 1 for row in encoded[partition] for index in row['word_ids'])
        stats['oov_utterances'][partition] = sum(1 in row['word_ids'] for row in encoded[partition])
    return {'mappings': {'word_to_id': word_to_id, 'char_to_id': char_to_id,
                        'intent_to_id': intent_to_id, 'tag_names': tag_names, 'slot_types': types},
            'splits': encoded, 'gold_by_id': gold_by_id, 'stats': stats}


def collate_transformer(rows):
    """Dynamic right-padding; no token or character truncation."""
    rows = list(rows)
    if not rows:
        raise ValueError('Cannot collate an empty batch')
    for row in rows:
        size = len(row['word_ids'])
        if not size or any(len(row[key]) != size for key in ('token_offsets', 'slot_ids', 'char_ids')):
            raise ValueError('Input, offsets and target token lengths must agree')
        if any(not values for values in row['char_ids']):
            raise ValueError('Every actual token needs its full character sequence')
    batch, tokens = len(rows), max(len(row['word_ids']) for row in rows)
    width = max(5, max(len(chars) for row in rows for chars in row['char_ids']))
    word_ids = torch.zeros((batch, tokens), dtype=torch.long)
    mask = torch.zeros((batch, tokens), dtype=torch.bool)
    char_ids = torch.zeros((batch, tokens, width), dtype=torch.long)
    slot_ids = torch.full((batch, tokens), -100, dtype=torch.long)
    intent_ids = torch.tensor([row['intent_id'] for row in rows], dtype=torch.long)
    for i, row in enumerate(rows):
        size = len(row['word_ids'])
        word_ids[i, :size] = torch.tensor(row['word_ids'], dtype=torch.long)
        mask[i, :size] = True
        slot_ids[i, :size] = torch.tensor(row['slot_ids'], dtype=torch.long)
        for j, chars in enumerate(row['char_ids']):
            char_ids[i, j, :len(chars)] = torch.tensor(chars, dtype=torch.long)
    return {'word_ids': word_ids, 'mask': mask, 'char_ids': char_ids,
            'slot_ids': slot_ids, 'intent_ids': intent_ids, 'rows': rows}


def _vector_archive_member(archive):
    candidates = []
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        mode = member.external_attr >> 16
        if (not member.filename or member.filename.startswith('/') or '\\' in member.filename
                or '..' in path.parts or any(':' in part for part in path.parts)
                or stat.S_ISLNK(mode)):
            raise ValueError('Unsafe word-vector archive member')
        if not member.is_dir() and path.name == 'model.txt':
            if not 0 < member.file_size <= _MAX_VECTOR_MEMBER_BYTES:
                raise ValueError('Word-vector member size is outside accepted bounds')
            candidates.append(member)
    if len(candidates) != 1:
        raise ValueError('Expected exactly one model.txt in the word-vector archive')
    return candidates[0]


def ensure_word2vec_archive(path):
    """Download the original public vector ZIP exclusively, without extraction."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ValueError('Word-vector archive must be a regular non-symlink file')
        with zipfile.ZipFile(path) as archive:
            _vector_archive_member(archive)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.nlpl-vectors-', suffix='.part', dir=path.parent)
    temporary = Path(name)
    try:
        written = 0
        request = Request(WORD2VEC_SOURCE_URL, headers={'User-Agent': 'SC4002-revised-transformer/1'})
        with os.fdopen(descriptor, 'wb') as output, urlopen(request, timeout=60) as response:
            length = response.headers.get('Content-Length')
            if length is not None and int(length) > _MAX_VECTOR_ARCHIVE_BYTES:
                raise ValueError('Word-vector download exceeds accepted size')
            for block in iter(lambda: response.read(1024 * 1024), b''):
                written += len(block)
                if written > _MAX_VECTOR_ARCHIVE_BYTES:
                    raise ValueError('Word-vector download exceeds accepted size')
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if written == 0 or (length is not None and written != int(length)):
            raise ValueError('Word-vector download size differs from HTTP Content-Length')
        with zipfile.ZipFile(temporary) as archive:
            _vector_archive_member(archive)
        try:
            os.link(temporary, path)
        except FileExistsError:
            return ensure_word2vec_archive(path)
        return path
    finally:
        temporary.unlink(missing_ok=True)


def _embedding_vocabulary(word_to_id):
    if (not isinstance(word_to_id, dict) or word_to_id.get('<PAD>') != 0 or word_to_id.get('<UNK>') != 1
            or any(not isinstance(word, str) or not word for word in word_to_id)
            or any(type(index) is not int for index in word_to_id.values())
            or sorted(word_to_id.values()) != list(range(len(word_to_id)))):
        raise ValueError('Task word vocabulary must be contiguous with PAD=0 and UNK=1')


def _vector_header(handle, dimension):
    fields = handle.readline().split()
    if len(fields) != 2:
        raise ValueError('Word-vector model.txt requires a count/dimension header')
    try:
        count, observed_dimension = map(int, fields)
    except ValueError as error:
        raise ValueError('Invalid word-vector count/dimension header') from error
    if not 0 < count <= 5_000_000 or observed_dimension != dimension:
        raise ValueError('Word-vector source count or dimension is unexpected')
    return count


def task_embeddings(word_to_id, source='random', seed=42, dimension=300,
                    archive_path=None, download=False):
    """Return a task-only matrix and observed initialization provenance.

    The original model-6 POS-suffix heuristic and casefold averaging are
    preserved, but source vectors absent from the train-fitted vocabulary are
    never retained. Download must be explicitly enabled; missing vectors do not
    silently change the selected initialization policy.
    """
    _embedding_vocabulary(word_to_id)
    if source not in {'random', 'word2vec'}:
        raise ValueError('Embedding source must be random or word2vec')
    if type(seed) is not int or not 0 <= seed < 2**32 or type(dimension) is not int or dimension <= 0:
        raise ValueError('Use a nonnegative 32-bit seed and positive embedding dimension')
    if len(word_to_id) * dimension > 20_000_000:
        raise ValueError('Task word embedding table alone exceeds the whole-system parameter cap')
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0., .05, (len(word_to_id), dimension)).astype(np.float32)
    matrix[0] = 0.
    # Preserve original notebook's independent second UNK draw.
    matrix[1] = rng.normal(0., .05, dimension).astype(np.float32)
    manifest = {'source': source, 'seed': seed, 'dimension': dimension,
                'task_vocabulary_size': len(word_to_id), 'task_embedding_parameters': int(matrix.size),
                'fit_partition': 'train', 'full_source_table_retained_for_prediction': False,
                'pretrained_initialized': 0, 'random_initialized': len(word_to_id) - 2}
    if source == 'word2vec':
        if archive_path is None:
            raise ValueError('A word-vector archive path is required')
        path = Path(archive_path)
        if download:
            path = ensure_word2vec_archive(path)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError('Word-vector archive is absent; enable download or provide the original cache')
        needed = {word for word, index in word_to_id.items() if index >= 2}
        sums, counts = {}, Counter()
        archive_hash = hashlib.sha256()
        with path.open('rb') as archive_file:
            for block in iter(lambda: archive_file.read(1024 * 1024), b''):
                archive_hash.update(block)
        with zipfile.ZipFile(path) as archive:
            member = _vector_archive_member(archive)
            with archive.open(member) as binary, io.TextIOWrapper(binary, encoding='utf-8') as text:
                count = _vector_header(text, dimension)
                probe = []
                for _ in range(min(count, 50_000)):
                    fields = text.readline().split(maxsplit=1)
                    if not fields:
                        raise ValueError('Word-vector source is truncated before its declared row count')
                    probe.append(fields[0])
            suffix_ratio = sum(bool(_VECTOR_POS_SUFFIX.search(word)) for word in probe) / len(probe)
            strip_suffix = suffix_ratio > .20
            rows = 0
            with archive.open(member) as binary, io.TextIOWrapper(binary, encoding='utf-8') as text:
                if _vector_header(text, dimension) != count:
                    raise ValueError('Word-vector header changed between streamed passes')
                for line in text:
                    fields = line.split()
                    if len(fields) != dimension + 1:
                        raise ValueError('Word-vector row has an invalid dimension')
                    rows += 1
                    word = (_VECTOR_POS_SUFFIX.sub('', fields[0]) if strip_suffix else fields[0]).casefold()
                    if word not in needed:
                        continue
                    try:
                        vector = np.asarray(fields[1:], dtype=np.float32)
                    except ValueError as error:
                        raise ValueError('Invalid numeric word vector') from error
                    if not np.isfinite(vector).all():
                        raise ValueError('Nonfinite numeric word vector')
                    if word in sums:
                        sums[word] += vector
                    else:
                        sums[word] = vector
                    counts[word] += 1
            if rows != count:
                raise ValueError('Word-vector row count differs from its header')
            for word, vector_sum in sums.items():
                matrix[word_to_id[word]] = vector_sum / counts[word]
            initialized = len(sums)
            manifest.update(source_url=WORD2VEC_SOURCE_URL, source_archive_sha256=archive_hash.hexdigest(),
                source_member=member.filename, source_vector_count=count, pos_suffix_probe_ratio=suffix_ratio,
                pos_suffixes_stripped=strip_suffix, matching='casefold_average',
                pretrained_initialized=initialized, random_initialized=len(needed) - initialized,
                pretrained_coverage=initialized / len(needed) if needed else 0.,
                source_identity='Locally observed source bytes; no independently published digest supplied')
        del sums, counts
    manifest['matrix_sha256'] = hashlib.sha256(matrix.tobytes(order='C')).hexdigest()
    return matrix, manifest

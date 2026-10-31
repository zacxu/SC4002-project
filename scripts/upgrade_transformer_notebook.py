"""Rebuild the comparison notebook with the shared verified-data/scoring contract.

The original notebook is retained in the ignored prechange archive. This script
does not train models, download embeddings, or preserve stale numerical outputs.
The generated notebook contains all shared implementation source inline.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build(root: Path = ROOT, output: Path | None = None, archive_executed: Path | None = None) -> dict:
    original = root / "artifacts/evaluator_v2/prechange/NLP_Final_project.ipynb"
    target = output or root / "NLP_Final_project.ipynb"
    if target.exists():
        previous = nbformat.read(target, as_version=4)
        if any(cell.cell_type == "code" and (cell.get("execution_count") is not None or cell.get("outputs"))
               for cell in previous.cells):
            if archive_executed is None:
                raise FileExistsError("Preserve executed outputs: use a fresh --output or explicitly supply --archive-executed.")
            archive_executed = archive_executed.resolve()
            archive_root = (root / "artifacts/evaluator_v2").resolve()
            if not archive_executed.is_relative_to(archive_root) or archive_executed == target.resolve():
                raise ValueError("The executed notebook archive must be a separate file under artifacts/evaluator_v2.")
            archive_executed.parent.mkdir(parents=True, exist_ok=True)
            with archive_executed.open("xb") as destination:
                destination.write(target.read_bytes())
            if archive_executed.read_bytes() != target.read_bytes():
                raise ValueError("Executed notebook archive did not preserve the original bytes")
    notebook = nbformat.read(original, as_version=4)
    source = (root / "src/comparison_contract.py").read_text(encoding="utf-8")
    cells = notebook.cells

    def replace(number: int, text: str) -> None:
        cells[number - 1].source = text.strip() + "\n"

    replace(1, """## Environment

Install the repository's single `requirements.txt` in the local Python 3.12 / CUDA environment before opening this notebook. For a standalone copy, install the same pinned environment; the additional Transformer packages are pandas 2.2.3, gensim 4.3.3, SciPy 1.13.1, scikit-learn 1.6.1 and tqdm 4.67.1. No cell installs arbitrary latest packages.

This notebook preserves the original Transformer architecture and training settings, except for constructing all 111 BIO labels from the 55 training slot types. Stage 1 is executable below. Stage 2 is defined and counted but is not trained in the supplied experiment sequence. All numerical outputs were cleared after the evaluator/data change; fresh training is required before reporting results.
""")
    replace(2, """from importlib.metadata import version
for distribution in ('torch', 'numpy', 'pandas', 'gensim', 'scipy', 'scikit-learn', 'tqdm'):
    print(distribution, version(distribution))
""")

    config = cells[3].source
    config = config.replace("import copy\n", "import copy\nimport json\nimport hashlib\n")
    config = config.replace("import shutil\n", "").replace("import subprocess\n", "")
    config = config.replace('CSV_PATH = Path("/content/massive_enUS_full.csv")', '''WORK_ROOT = Path.cwd()
NOTEBOOK_PATH = WORK_ROOT / "NLP_Final_project.ipynb"
RUN_ROOT = WORK_ROOT / "artifacts" / "transformer_v2"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
DATASET_PATH = WORK_ROOT / "data" / "1.0" / "data" / "en-US.jsonl"
if not NOTEBOOK_PATH.is_file():
    raise FileNotFoundError("Keep NLP_Final_project.ipynb in the kernel working directory.")
notebook_document = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
current_code_hashes = [hashlib.sha256("".join(c["source"]).encode("utf-8")).hexdigest()
                       for c in notebook_document["cells"] if c["cell_type"] == "code"]
if current_code_hashes != notebook_document["metadata"]["evaluator_v2"]["code_cell_sha256"]:
    raise ValueError("Notebook source differs from its reviewed evaluator-v2 manifest")
NOTEBOOK_CODE_SHA256 = hashlib.sha256(json.dumps(current_code_hashes, separators=(",", ":")).encode("ascii")).hexdigest()''')
    config = config.replace('W2V_ZIP = Path("/content/nlpl_model_6.zip")', 'W2V_ZIP = WORK_ROOT / "data" / "pretrained" / "nlpl_model_6.zip"')
    config = config.replace('W2V_DIR = Path("/content/nlpl_model_6")', 'W2V_DIR = WORK_ROOT / "data" / "pretrained" / "nlpl_model_6"')
    replace(4, config)

    replace(5, """## Load and verify the identical original MASSIVE release

Both notebooks use the same inline loader, English JSONL SHA-256 and original split checks. The loader downloads the official archive only if the pinned English file is absent. An existing file with different bytes is rejected. Uploaded CSVs are no longer accepted as an unverified alternate source.
""")
    replace(6, """verified_splits, dataset_manifest = ensure_verified_massive(DATASET_PATH)
assert dataset_manifest['dataset_sha256'] == DATASET_SHA256
df = pd.DataFrame([dict(row, id=str(row['id']))
                   for part in ('train', 'dev', 'test') for row in verified_splits[part]])
required = {'id', 'partition', 'utt', 'intent', 'annot_utt'}
assert required.issubset(df.columns)
assert not df['id'].duplicated().any()
assert {part: int((df['partition'] == part).sum()) for part in EXPECTED_COUNTS} == EXPECTED_COUNTS
GOLD_BY_ID = {str(row.id): {'record_id': str(row.id), 'utt': row.utt,
               'intent': row.intent, 'slots': parse_annotated_spans(row.utt, row.annot_utt)}
              for row in df.itertuples(index=False)}
(RUN_ROOT / 'dataset_manifest.json').write_text(json.dumps(dataset_manifest, indent=2) + '\\n', encoding='utf-8')
print('VERIFIED_DATASET_MANIFEST:', json.dumps(dataset_manifest, indent=2))
print('EVALUATOR_VERSION:', EVALUATOR_VERSION)
print('Dataset identity and every original partition verified.')
""")

    replace(8, r'''TOKEN_RE = re.compile(r"\w+(?:['\u2019\-]\w+)*|[^\w\s]", re.UNICODE)

def tokenize_with_offsets(text):
    return [(match.group(0), match.start(), match.end()) for match in TOKEN_RE.finditer(text)]

def tokenize(text):
    return [token for token, _, _ in tokenize_with_offsets(text)]

def is_lexical(token):
    return bool(re.search(r"[A-Za-z0-9]", token))

def parse_row_targets(utt, annot_utt):
    # Predictive token text and positions come exclusively from raw utt.
    pieces = tokenize_with_offsets(utt)
    tokens = [token for token, _, _ in pieces]
    offsets = [(start, end) for _, start, end in pieces]
    spans = parse_annotated_spans(utt, annot_utt)
    bio = ['O'] * len(tokens)
    slot_mask = [False] * len(tokens)
    for span in spans:
        covered = [i for i, (start, end) in enumerate(offsets)
                   if start < span['end'] and span['start'] < end]
        if not covered or offsets[covered[0]][0] != span['start'] or offsets[covered[-1]][1] != span['end']:
            raise ValueError('A gold boundary is incompatible with the fixed Transformer tokenizer')
        for j, i in enumerate(covered):
            if bio[i] != 'O':
                raise ValueError('Overlapping annotation spans')
            bio[i] = ('B-' if j == 0 else 'I-') + span['slot_type']
            slot_mask[i] = True
    if bio_to_character_spans(utt, offsets, bio) != spans:
        raise ValueError('Gold BIO annotations do not round-trip to exact original character spans')
    return {'tokens': tokens, 'offsets': offsets, 'bio': bio,
            'slot_mask': slot_mask, 'gold_spans': spans}

for row in tqdm(df.itertuples(index=False), total=len(df), desc='Exact annotation alignment'):
    parse_row_targets(row.utt, row.annot_utt)
max_observed = max(len(tokenize(x)) for x in df['utt'])
assert max_observed <= MAX_LEN
print('All original gold boundaries round-trip; maximum token length:', max_observed)
''')

    labels = cells[9].source
    old = '''slot_label_set = {"O"}
# Build the overall slot labels using all partitions
for annot in df["annot_utt"]:
    parsed = parse_annotated_utterance(annot)
    slot_label_set.update(parsed["bio"])

slot_labels = ["O"] + sorted(x for x in slot_label_set if x != "O")'''
    new = '''slot_types = sorted({span['slot_type'] for row in train_df.itertuples(index=False)
                     for span in parse_annotated_spans(row.utt, row.annot_utt)})
assert len(slot_types) == 55
slot_labels = ['O'] + [prefix + '-' + slot_type for slot_type in slot_types for prefix in ('B', 'I')]
assert len(slot_labels) == 111'''
    if old not in labels:
        raise ValueError('Original label-construction cell changed unexpectedly')
    labels = labels.replace(old, new)
    labels = labels.replace('parse_annotated_utterance(row.annot_utt)', 'parse_row_targets(row.utt, row.annot_utt)')
    labels = labels.replace('''    for annot in split_df["annot_utt"]:
        unseen_slot_labels.update(
            set(parse_annotated_utterance(annot)["bio"]) - set(slot2id)
        )''', '''    for row in split_df.itertuples(index=False):
        unseen_slot_labels.update(set(parse_row_targets(row.utt, row.annot_utt)['bio']) - set(slot2id))''')
    replace(10, labels)

    vectors = cells[11].source
    old_downloader = '''def download_file(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["wget", "-c", "-O", str(destination), url]
    return subprocess.run(cmd, check=False).returncode'''
    new_downloader = '''def download_file(url, destination):
    """Stream to a temporary file; publish only a complete new cache entry."""
    import os
    import tempfile
    from urllib.request import Request, urlopen

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise ValueError('Expected a regular pretrained-vector cache file: ' + str(destination))
        return 0
    descriptor, temporary_name = tempfile.mkstemp(
        prefix='.' + destination.name + '-', suffix='.part', dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        request = Request(url, headers={'User-Agent': 'SC4002-Transformer-vector-cache/2'})
        with os.fdopen(descriptor, 'wb') as output:
            with urlopen(request, timeout=60) as response:
                expected = response.headers.get('Content-Length')
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        observed = temporary.stat().st_size
        if not observed or (expected is not None and observed != int(expected)):
            raise ValueError('Incomplete pretrained-vector download')
        # Both files are in the same directory. Hard-link creation is atomic and
        # refuses an existing destination, including one created concurrently.
        os.link(temporary, destination)
        return 0
    except Exception as error:
        print('Pretrained-vector download failed:', str(error))
        print('Download the original NLPL Model 6 ZIP manually and place it at:', destination)
        return 1
    finally:
        temporary.unlink(missing_ok=True)'''
    if old_downloader not in vectors:
        raise ValueError('Original pretrained-vector download function changed unexpectedly')
    vectors = vectors.replace(old_downloader, new_downloader)
    vectors = vectors.replace('''        from google.colab import files
        print("Upload NLPL Model 6 ZIP")
        uploaded = files.upload()
        if not uploaded:
            raise FileNotFoundError("Word2Vec ZIP not supplied.")
        uploaded_name = next(iter(uploaded))
        shutil.move(uploaded_name, W2V_ZIP)''', '''        raise FileNotFoundError('Place the original NLPL Model 6 ZIP at ' + str(W2V_ZIP)
                                + '; automatic download failed.')''')
    replace(12, vectors)

    metadata = cells[17].source.replace('parse_annotated_utterance(row.annot_utt)', 'parse_row_targets(row.utt, row.annot_utt)')
    replace(18, metadata)

    dataset = cells[19].source.replace('parse_annotated_utterance(row.annot_utt)', 'parse_row_targets(row.utt, row.annot_utt)')
    dataset = dataset.replace('''                "id": row.id,
                "word_ids": word_ids,''', '''                "id": str(row.id),
                "utt": row.utt,
                "offsets": parsed['offsets'],
                "word_ids": word_ids,''')
    dataset = dataset.replace('''            "ids": ids,
            "word_ids": word_ids,''', '''            "ids": ids,
            "utts": [x['utt'] for x in batch],
            "token_offsets": [x['offsets'] for x in batch],
            "word_ids": word_ids,''')
    replace(20, dataset)

    counts = cells[23].source
    counts = counts.replace('''    assert total <= 20_000_000''', '''    assert total <= 20_000_000
    assert total == {'Stage 1': 4_345_243, 'Stage 2': 4_379_947}[name]''')
    replace(24, counts)

    replace(25, """## Shared exact-character-span evaluation

The same source used by the BiLSTM notebook is inlined at the start. A correct slot requires the exact slot type and original utterance character start/end offsets; repeated text at different positions is scored separately. Character positions avoid dependence on either model's tokenizer. Scores pool TP/FP/FN over the complete ID-checked split. Invalid source slices, missing/extra/duplicate IDs, malformed BIO and unequal sequence lengths fail rather than silently changing the denominator.
""")
    replace(26, """def score_prediction_frame(frame, repeated_diagnostic_draws=False):
    gold, predicted = [], []
    for draw, row in enumerate(frame.itertuples(index=False)):
        expected = GOLD_BY_ID[str(row.id)]
        observed = dict(row.prediction)
        target = dict(expected)
        if repeated_diagnostic_draws:
            # Nearest-neighbor matching samples with replacement. Each diagnostic
            # draw is explicit; full-split evaluation never relaxes duplicate IDs.
            draw_id = str(row.id) + '#diagnostic_draw:' + str(draw)
            target['record_id'] = observed['record_id'] = draw_id
        gold.append(target)
        predicted.append(observed)
    return score_exact_spans(gold, predicted)

identity = score_exact_spans(list(GOLD_BY_ID.values()), list(GOLD_BY_ID.values()))
assert identity['intent_accuracy'] == 1.0 and identity['slot_micro_f1'] == 1.0
print('Shared evaluator exact-span self-check passed:', EVALUATOR_VERSION)
""")

    training = cells[29].source
    start = training.index('@torch.no_grad()\ndef evaluate_model')
    end = training.index('\ndef train_model(', start)
    evaluation = '''@torch.no_grad()
def evaluate_model(model, loader, include_chars):
    model.eval()
    losses, rows = [], []
    for raw_batch in loader:
        batch = move_batch(raw_batch, include_chars)
        loss, intent_logits, slot_logits = forward_loss(model, batch, include_chars)
        losses.append(loss.item())
        pred_int = intent_logits.argmax(-1).cpu().tolist()
        pred_slot = slot_logits.argmax(-1).cpu().tolist()
        for i, record_id in enumerate(raw_batch['ids']):
            n = int(raw_batch['attention_mask'][i].sum().item())
            tags = [id2slot[index] for index in pred_slot[i][:n]]
            utterance = raw_batch['utts'][i]
            prediction = {'record_id': str(record_id), 'intent': id2intent[pred_int[i]],
                          'utt': utterance,
                          'slots': bio_to_character_spans(utterance, raw_batch['token_offsets'][i], tags)}
            expected = GOLD_BY_ID[str(record_id)]
            rows.append({'id': str(record_id), 'gold_intent_id': intent2id[expected['intent']],
                         'pred_intent_id': pred_int[i], 'prediction': prediction})
    frame = pd.DataFrame(rows)
    metrics = score_prediction_frame(frame)
    expected_ids = [row['id'] for row in loader.dataset.rows]
    if frame['id'].tolist() != expected_ids:
        raise ValueError('Evaluation must cover the complete original split in order')
    return {'loss': float(np.mean(losses)), 'metrics': metrics,
            'intent_accuracy': 100.0 * metrics['intent_accuracy'],
            'slot_micro_f1': 100.0 * metrics['slot_micro_f1'],
            'joint_score': 50.0 * (metrics['intent_accuracy'] + metrics['slot_micro_f1']),
            'pred_frame': frame, 'evaluator_version': EVALUATOR_VERSION}

'''
    training = training[:start] + evaluation + training[end:]
    training = training.replace('''    seed_everything()

    model = model.to(DEVICE)''', '''    seed_everything()
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        raise FileExistsError('Use a fresh run directory; existing checkpoint is preserved: ' + str(checkpoint_path))
    model = model.to(DEVICE)''')
    training = training.replace('''                    "dev_joint": best_score,''', '''                    "dev_joint": best_score,
                    "evaluator_version": EVALUATOR_VERSION,
                    "dataset_sha256": DATASET_SHA256,
                    "dataset_manifest": dataset_manifest,
                    "notebook_code_sha256": NOTEBOOK_CODE_SHA256,
                    "dev_metrics": dev_result['metrics'],''')
    training = training.replace('''        weights_only=False,''', '''        weights_only=True,''')
    training = training.replace('''    model.load_state_dict(
        ckpt["model_state_dict"]
    )''', '''    if (ckpt['evaluator_version'] != EVALUATOR_VERSION
            or ckpt['dataset_sha256'] != DATASET_SHA256
            or ckpt['notebook_code_sha256'] != NOTEBOOK_CODE_SHA256):
        raise ValueError('Checkpoint belongs to a different scoring/data/code contract')
    model.load_state_dict(ckpt['model_state_dict'])
    reloaded = evaluate_model(model, dev_loader, include_chars)
    if reloaded['metrics'] != ckpt['dev_metrics']:
        raise ValueError('Reloaded checkpoint does not reproduce selected development metrics')''')
    replace(30, training)
    stage1 = cells[31].source.replace('STAGE1_CKPT = "/content/stage1_oov_proof.pt"', 'STAGE1_CKPT = RUN_ROOT / "stage1_best.pt"\n\nseed_everything(SEED)')
    replace(32, stage1)

    subset = cells[35].source
    start = subset.index('def subset_metrics(frame):')
    end = subset.index('\ns1_dev =', start)
    subset = subset[:start] + '''def subset_metrics(frame, repeated_diagnostic_draws=False):
    metrics = score_prediction_frame(frame, repeated_diagnostic_draws=repeated_diagnostic_draws)
    return {'n_examples': metrics['num_examples'],
            'intent_accuracy': 100.0 * metrics['intent_accuracy'],
            'slot_micro_f1': 100.0 * metrics['slot_micro_f1']}

''' + subset[end:]
    replace(36, subset)
    matched = cells[41].source.replace('**subset_metrics(matched_no),', '**subset_metrics(matched_no, repeated_diagnostic_draws=True),')
    replace(42, matched)

    notice = nbformat.v4.new_markdown_cell("""# Joint Transformer comparison — evaluator v2

**Fresh training is pending.** This notebook has been upgraded to the identical verified MASSIVE dataset and exact-character-span evaluator used by `main.ipynb`. All old outputs were archived and cleared because old scores and checkpoint selection used a different evaluator. No new performance result is claimed here.

The supplied experiment implements Stage 1 training plus development/OOV diagnostics. Stage 2 is defined and parameter-counted only; neither stage originally included final test evaluation. These limitations remain explicit. The model architectures and optimizer settings are preserved except the train-derived complete 111-tag BIO inventory.

Run from a fresh kernel with this filename in its working directory. Data are under `data/`; pretrained-vector cache is under `data/pretrained/`; new checkpoints are under ignored `artifacts/transformer_v2/`. Existing checkpoints are protected. The model receives only utterance-derived tokens/characters. Shared source is visible below; no repository module is imported at runtime.
""")
    contract_cell = nbformat.v4.new_code_cell(source, metadata={"tags": ["definitions", "comparison_contract"]})
    contract_cell.id = "shared-comparison-contract-v2"
    notebook.cells = [notice, contract_cell] + cells
    notebook.metadata.pop("widgets", None)
    notebook.metadata.pop("colab", None)
    notebook.metadata["kernelspec"] = {"display_name": "SC4002 (.venv)", "language": "python", "name": "sc4002"}
    notebook.metadata["language_info"] = {"name": "python", "version": "3.12.0"}
    for i, cell in enumerate(notebook.cells):
        tags = cell.metadata.get("tags", [])
        cell.metadata = {"tags": tags} if tags else {}
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
            ast.parse(cell.source)
        if cell is not contract_cell:
            cell.id = "transformer-v2-" + str(i + 1)
    notebook.metadata["comparison_contract"] = {
        "evaluator_version": "exact_character_span_v2",
        "dataset_sha256": "c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e",
        "source_sha256": _hash(source), "source_file": "src/comparison_contract.py",
        "inline_cell_id": contract_cell.id,
    }
    notebook.metadata["evaluator_v2"] = {
        "schema_version": 2, "status": "pending_fresh_training",
        "original_notebook_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "code_cell_sha256": [_hash(cell.source) for cell in notebook.cells if cell.cell_type == "code"],
        "architecture_changes": "Train-only complete B/I label inventory: 111 output tags",
        "training_performed_by_upgrade": False,
    }
    nbformat.validate(notebook)
    # Explicit bytes prevent Windows newline translation from changing reviewed
    # notebook hashes after checkout with the repository's LF attributes.
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(nbformat.writes(notebook).replace("\r\n", "\n").encode("utf-8"))
    return {"notebook": str(target), "cells": len(notebook.cells),
            "code_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
            "all_code_cells_compile": True, "stale_outputs_cleared": True,
            "contract_source_sha256": _hash(source), "training_performed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive-executed", type=Path,
                        help="Explicitly preserve an executed target under artifacts/evaluator_v2 before replacing it")
    args = parser.parse_args()
    print(json.dumps(build(ROOT, args.output, args.archive_executed), indent=2))


if __name__ == "__main__":
    main()

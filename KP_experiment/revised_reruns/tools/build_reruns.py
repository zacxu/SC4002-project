"""Build one patched copy of NLP_Final_project_revised.ipynb per rerun.

The source notebook in source/ is never edited. Each copy lives in its own runs/<name>/ folder,
because the notebook insists on its file name and creates transformer_run_artifacts/ exclusively.

Usage: python build_reruns.py            builds the six full runs and the three one-epoch smokes
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "NLP_Final_project_revised.ipynb"
RUNS = ROOT / "runs"
DATASET = ROOT / "cache" / "massive" / "en-US.jsonl"
ARCHIVES = {
    "word2vec": ROOT.parent / "data" / "nlpl_model_6" / "6.zip",
    "fasttext": ROOT / "cache" / "fasttext" / "crawl-300d-2M.vec.zip",
    "glove": ROOT / "cache" / "glove" / "glove.2024.dolma.300d.zip",
}
SAM_RHO = 0.2   # strict winner of rung R in the BiLSTM ladder (family R1_sam.2)

# name: (embedding source, SAM rho, epochs, apply the code patches)
FULL_RUNS = {
    "r0_word2vec_base": ("word2vec", 0.0, 20, True),    # same-machine control for step 1
    "r1_word2vec_sam": ("word2vec", SAM_RHO, 20, True),
    "r2_fasttext_base": ("fasttext", 0.0, 20, True),
    "r3_glove_base": ("glove", 0.0, 20, True),
    "r4_fasttext_sam": ("fasttext", SAM_RHO, 20, True),
    "r5_glove_sam": ("glove", SAM_RHO, 20, True),
}
SMOKE_RUNS = {
    "smoke_original": ("word2vec", 0.0, 1, False),      # unpatched code, config edits only
    "smoke_patched_rho0": ("word2vec", 0.0, 1, True),   # must match smoke_original exactly
    "smoke_patched_sam": ("word2vec", SAM_RHO, 1, True),
}

VECTOR_SOURCES = """VECTOR_SOURCES = {
    'fasttext': {'url': 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip',
                 'member': 'crawl-300d-2M.vec', 'header': True},
    'glove': {'url': 'https://downloads.cs.stanford.edu/nlp/data/wordvecs/glove.2024.dolma.300d.zip',
              'member': 'dolma_300_2024_1.2M.100_combined.txt', 'header': False},
}
"""

READ_EXTERNAL = """def _read_external_vectors(path, source, needed, dimension):
    \"\"\"Stream a fastText or GloVe text member, keeping only train-vocabulary rows.

    Rows are split from the right, because these sources hold raw tokens that can contain
    spaces. GloVe has no count/dimension header.
    \"\"\"
    settings = VECTOR_SOURCES[source]
    sums, counts, rows = ({}, Counter(), 0)
    with zipfile.ZipFile(path) as archive:
        member = _vector_archive_member(archive, settings['member'])
        with archive.open(member) as binary, io.TextIOWrapper(binary, encoding='utf-8', newline='\\n') as text:
            count = _vector_header(text, dimension) if settings['header'] else None
            for line in text:
                fields = line.rstrip().rsplit(' ', dimension)
                if len(fields) != dimension + 1:
                    raise ValueError('Word-vector row has an invalid dimension')
                rows += 1
                word = fields[0].casefold()
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
    if count is not None and rows != count:
        raise ValueError('Word-vector row count differs from its header')
    return sums, counts, rows, member.filename

"""

EXTERNAL_BRANCH = """    if source in {'fasttext', 'glove'}:
        if archive_path is None:
            raise ValueError('A word-vector archive path is required')
        path = Path(archive_path)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError('Word-vector archive is absent; fetch it before the run')
        needed = {word for word, index in word_to_id.items() if index >= 2}
        archive_hash = hashlib.sha256()
        with path.open('rb') as archive_file:
            for block in iter(lambda: archive_file.read(1024 * 1024), b''):
                archive_hash.update(block)
        sums, counts, rows, member_name = _read_external_vectors(path, source, needed, dimension)
        for word, vector_sum in sums.items():
            matrix[word_to_id[word]] = vector_sum / counts[word]
        initialized = len(sums)
        manifest.update(source_url=VECTOR_SOURCES[source]['url'], source_archive_sha256=archive_hash.hexdigest(), source_member=member_name, source_vector_count=rows, pos_suffixes_stripped=False, matching='casefold_average', pretrained_initialized=initialized, random_initialized=len(needed) - initialized, pretrained_coverage=initialized / len(needed) if needed else 0.0, source_identity='Locally observed source bytes; no independently published digest supplied')
        del sums, counts
    manifest['mean_vector_norm'] = float(np.linalg.norm(matrix[2:], axis=1).mean())
"""

GUARD_LOADER = """_RT_GUARD = None
if os.environ.get('SC4002_GUARD_PATH'):  # laptop safety: waits on heat or low RAM, never touches RNG or model state
    import importlib.util
    _guard_spec = importlib.util.spec_from_file_location('sc4002_laptop_guard', os.environ['SC4002_GUARD_PATH'])
    _RT_GUARD = importlib.util.module_from_spec(_guard_spec)
    _guard_spec.loader.exec_module(_RT_GUARD)
"""

SAM_PASS = """def _rt_sam_second_pass(model, batch, rho):
    \"\"\"Sharpness-aware minimisation (Foret et al. 2021), as in the BiLSTM ladder's train.py.

    Replaces the gradient from .backward() with the one taken at the worst-case nearby weights,
    then restores the weights. The caller clips and steps as usual.
    \"\"\"
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    norm = torch.norm(torch.stack([parameter.grad.norm() for parameter in parameters])) + 1e-12
    saved = []
    with torch.no_grad():
        for parameter in parameters:
            step = rho * parameter.grad / norm
            parameter.add_(step)
            saved.append((parameter, step))
    model.zero_grad(set_to_none=True)
    outputs = model(batch['word_ids'], batch['mask'], batch['char_ids'])
    model.loss(outputs, batch['intent_ids'], batch['slot_ids'], batch['mask'])['total'].backward()
    with torch.no_grad():
        for parameter, step in saved:
            parameter.sub_(step)

"""

# cell index -> [(exact old text, new text)]; every old text must occur exactly once
CODE_PATCHES = {
    5: [
        ("_MAX_VECTOR_MEMBER_BYTES = 3 * 1024 ** 3\n", "_MAX_VECTOR_MEMBER_BYTES = 8 * 1024 ** 3\n" + VECTOR_SOURCES),
        ("def _vector_archive_member(archive):\n", "def _vector_archive_member(archive, member_name='model.txt'):\n"),
        ("if not member.is_dir() and path.name == 'model.txt':", "if not member.is_dir() and path.name == member_name:"),
        ("raise ValueError('Expected exactly one model.txt in the word-vector archive')",
         "raise ValueError('Expected exactly one ' + member_name + ' in the word-vector archive')"),
        ("def task_embeddings(word_to_id, source='random'", READ_EXTERNAL + "def task_embeddings(word_to_id, source='random'"),
        ("if source not in {'random', 'word2vec'}:\n        raise ValueError('Embedding source must be random or word2vec')",
         "if source not in {'random', 'word2vec', 'fasttext', 'glove'}:\n        raise ValueError('Embedding source must be random, word2vec, fasttext or glove')"),
        ("    manifest['matrix_sha256'] = hashlib.sha256(matrix.tobytes(order='C')).hexdigest()\n",
         EXTERNAL_BRANCH + "    manifest['matrix_sha256'] = hashlib.sha256(matrix.tobytes(order='C')).hexdigest()\n"),
    ],
    8: [
        ("from torch.utils.data import DataLoader\n", "from torch.utils.data import DataLoader\n\n" + GUARD_LOADER),
        ("'warmup_fraction', 'min_lr_ratio'}\n    if set(config) != required:",
         "'warmup_fraction', 'min_lr_ratio', 'sam_rho'}\n    if set(config) != required:"),
        ("if config['weight_decay'] < 0 or not 0 <= config['warmup_fraction'] < 1 or (not 0 <= config['min_lr_ratio'] <= 1):",
         "if config['weight_decay'] < 0 or config['sam_rho'] < 0 or not 0 <= config['warmup_fraction'] < 1 or (not 0 <= config['min_lr_ratio'] <= 1):"),
        ("def _rt_epoch(model, loader, bundle, device, *, optimizer=None, schedule=None):\n",
         SAM_PASS + "def _rt_epoch(model, loader, bundle, device, *, optimizer=None, schedule=None):\n"),
        ("            if training:\n                optimizer.zero_grad(set_to_none=True)\n",
         "            if training:\n                if _RT_GUARD is not None:\n                    _RT_GUARD.check('train')\n                optimizer.zero_grad(set_to_none=True)\n"),
        ("                losses['total'].backward()\n",
         "                losses['total'].backward()\n                if optimizer._rt_sam_rho > 0:\n                    _rt_sam_second_pass(model, batch, optimizer._rt_sam_rho)\n"),
        ("    optimizer._rt_clip = training['gradient_clip_norm']\n",
         "    optimizer._rt_clip = training['gradient_clip_norm']\n    optimizer._rt_sam_rho = training['sam_rho']\n"),
        ("    output.mkdir(parents=True, exist_ok=False)\n    rt_seed(runtime)\n",
         "    output.mkdir(parents=True, exist_ok=False)\n    if _RT_GUARD is not None:\n        _RT_GUARD.check(stage, need_vram=True)\n    rt_seed(runtime)\n"),
    ],
}


def replace_once(text, old, new, label):
    found = text.count(old)
    if found != 1:
        raise ValueError(f"{label}: expected one occurrence, found {found}: {old[:70]!r}")
    return text.replace(old, new)


def cell_text(cell):
    return cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])


def set_cell_text(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def configure(text, source, rho, epochs, patched):
    """Edit the notebook's one editable cell; both stages always share these settings."""
    if text.count("'max_epochs': 20,") != 2:
        raise ValueError("Configuration cell does not hold two 20-epoch stages")
    text = text.replace("'max_epochs': 20,", f"'max_epochs': {epochs},")
    if patched:
        if text.count("'min_lr_ratio': 0.1}}") != 2:
            raise ValueError("Configuration cell does not hold two training blocks")
        text = text.replace("'min_lr_ratio': 0.1}}", f"'min_lr_ratio': 0.1,\n              'sam_rho': {rho}}}}}")
    text = replace_once(text, "'dataset_path': None", f"'dataset_path': {str(DATASET)!r}", "config")
    text = replace_once(text, "'embedding_source': 'word2vec'", f"'embedding_source': {source!r}", "config")
    text = replace_once(text, "'embedding_archive': None", f"'embedding_archive': {str(ARCHIVES[source])!r}", "config")
    return text


def build(name, source, rho, epochs, patched):
    document = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = document["cells"]
    set_cell_text(cells[3], configure(cell_text(cells[3]), source, rho, epochs, patched))
    if patched:
        for index, patches in CODE_PATCHES.items():
            text = cell_text(cells[index])
            for old, new in patches:
                text = replace_once(text, old, new, f"cell {index}")
            set_cell_text(cells[index], text)
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    for cell in code_cells:
        cell["outputs"], cell["execution_count"] = [], None
    metadata = document["metadata"]["revised_transformer"]
    metadata["code_cell_sha256"] = [hashlib.sha256(cell_text(cell).encode("utf-8")).hexdigest() for cell in code_cells]
    metadata["status"] = "rerun_copy_not_executed"
    metadata["rerun"] = {"name": name, "embedding_source": source, "sam_rho": rho, "max_epochs": epochs,
                         "code_patched": patched, "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
    note = (f"# Rerun copy: {name}\n\nPatched copy of `NLP_Final_project_revised.ipynb`. Embedding: **{source}**. "
            f"SAM rho: **{rho}**. Epochs: **{epochs}**. Development only.\n\nText cells further down, including "
            "\"Measured development qualification\" and \"Remaining weaknesses\", describe the original run, not this one. "
            "Use this run's outputs and `transformer_run_artifacts/`.")
    cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source": note.splitlines(keepends=True)})
    folder = RUNS / name
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "NLP_Final_project_revised.ipynb"
    target.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return target


if __name__ == "__main__":
    for name, settings in {**SMOKE_RUNS, **FULL_RUNS}.items():
        print("built", build(name, *settings))

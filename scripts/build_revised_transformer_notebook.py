"""Build a separate, standalone improved Transformer notebook; never train."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import pprint

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def inline_module(path, names=None):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    body = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module == '__future__' or (node.module or '').startswith('src')):
            continue
        if names is None or isinstance(node, (ast.Import, ast.ImportFrom)) or isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            body.append(node)
    tree.body = body
    return 'from __future__ import annotations\n\n' + ast.unparse(ast.fix_missing_locations(tree)) + '\n'


def build(root=ROOT, output=None, *, word_dropout_alpha=0., intent_label_smoothing=0., early_stopping=False):
    from src.revised_transformer_model import RevisedTransformerConfig
    for name, value in (('word_dropout_alpha', word_dropout_alpha),
                        ('intent_label_smoothing', intent_label_smoothing)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(name + ' must be a finite nonnegative number')
    if intent_label_smoothing >= 1:
        raise ValueError('intent_label_smoothing must be less than 1')
    if type(early_stopping) is not bool:
        raise ValueError('early_stopping must be a boolean')
    regularized = bool(word_dropout_alpha or intent_label_smoothing)
    output = Path(output or root / 'NLP_Final_project_revised.ipynb')
    if output.exists():
        previous = nbformat.read(output, as_version=4)
        if any(c.cell_type == 'code' and (c.get('execution_count') is not None or c.get('outputs')) for c in previous.cells):
            raise FileExistsError('Preserve the executed notebook; build to a new review path')
    cells, provenance = [], {}
    def md(text): cells.append(nbformat.v4.new_markdown_cell(text.strip()))
    def code(text, tags=()): cells.append(nbformat.v4.new_code_cell(text.strip() + '\n', metadata={'tags': list(tags)}))
    def definitions(name, names=None, exact=False):
        path = root / name
        text = path.read_text(encoding='utf-8') if exact else inline_module(path, names)
        cells.append(nbformat.v4.new_code_cell(text, metadata={'tags': ['definitions'] + (['comparison_contract'] if exact else [])}))
        provenance[name] = {'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'inline_sha256': hashlib.sha256(text.encode()).hexdigest()}

    md('''# Revised joint Transformer: diagnosis, implementation and measured evaluation

This is a **separate revision** of `NLP_Final_project.ipynb`. The original notebook and the completed BiLSTM evidence are preserved. The baseline reproduces the supplied 4,345,243-parameter architecture; the revised default has **19,748,595 total parameters**, including its task embeddings, character features, both heads, intent context and CRF. Both are single jointly trained systems with randomly initialized encoders. Only word embeddings may use permitted pretrained vectors.

Run status and actual metrics come from the genuine outputs and final completion ledger. Larger capacity is a hypothesis, not evidence of improved accuracy. Historical evaluator-v1 scores are not comparison targets. All decisions use the original development split and the shared `exact_character_span_v2` evaluator.

## Run instructions

Use the repository's pinned Python 3.12/CUDA environment and **SC4002 (.venv)** kernel. Launch Jupyter with `PYTHONHASHSEED=42` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Copy this file into a fresh folder, keep its name **__NOTEBOOK_NAME__**, save edits to the labelled configuration cell, and Restart Kernel and Run All. The notebook embeds all implementation code and needs no repository modules at runtime. It downloads original English data and allowed Word2Vec initialization if verified caches are not supplied.

The default is **development-only**: no learned model is evaluated on test. After choosing settings, set `evaluate_test=True` and perform a new full run for the final four-row table. Never select settings using test scores. Each run exclusively creates `transformer_run_artifacts/` and never resumes/overwrites a completed run.

**AI history:** the brief requires Stage 1 without generative AI; assistance occurred at the user's request and no lecturer exception is claimed. This revision's analysis, implementation and verification used Codex. Group information and verified human contributions remain pending; do not invent them.
'''.replace('__NOTEBOOK_NAME__', output.name))
    code('''import os
from pathlib import Path
if os.environ.get('PYTHONHASHSEED') != '42':
    raise RuntimeError('Launch a fresh Jupyter kernel with PYTHONHASHSEED=42 before Run All')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if os.environ['CUBLAS_WORKSPACE_CONFIG'] != ':4096:8':
    raise RuntimeError('Restart with CUBLAS_WORKSPACE_CONFIG=:4096:8')
NOTEBOOK_PATH = Path.cwd() / __NOTEBOOK_FILENAME_LITERAL__
if not NOTEBOOK_PATH.is_file():
    raise FileNotFoundError('Keep the revised filename and kernel working directory together')
RUN_ROOT = Path.cwd() / 'transformer_run_artifacts'
RUN_ROOT.mkdir(exist_ok=False)
os.environ['MPLCONFIGDIR'] = str(RUN_ROOT / 'matplotlib_cache')
'''.replace('__NOTEBOOK_FILENAME_LITERAL__', repr(output.name)), ('bootstrap',))
    md('''## Editable configuration

Only this cell accepts ordinary literal dictionary edits without rebuilding or changing hashes. Save before running. Both models use the same initialization matrix, data, scorer and development selection rule. The training protocol uses declared completion rules, two AdamW learning rates, warmup/cosine decay, gradient clipping and globally normalized losses. It is not an exact reproduction of the original 12-epoch training procedure.

Use `embedding_source='word2vec'` to retain the original allowed initialization; missing assets raise rather than silently switching to random. `random` is an explicit alternative experiment. No full pretrained vocabulary enters prediction: only the train-fitted 5,300-by-300 table is retained and counted. Set cache paths if desired, or use the terminal executor's cache flags. Each run archives an exact copy of original English data.
''')
    training = dict(max_epochs=20, batch_size=32, learning_rate=.0005, embedding_learning_rate=.0001,
                    weight_decay=.01, gradient_clip_norm=1., warmup_fraction=.1, min_lr_ratio=.1)
    config = {'RUNTIME': {'seed': 42, 'deterministic_algorithms': True, 'precision': 'float32'},
        'STAGE1_CONFIG': {'model': RevisedTransformerConfig.original_baseline().to_dict(), 'training': dict(training)},
        'STAGE2_CONFIG': {'model': RevisedTransformerConfig().to_dict(), 'training': dict(training)},
        'RUN_OPTIONS': {'evaluate_test': False, 'device': 'cuda', 'dataset_path': None,
            'embedding_source': 'word2vec', 'embedding_archive': None, 'verification_timeout_seconds': 600}}
    if regularized:
        config['STAGE2_CONFIG']['training'].update(word_dropout_alpha=float(word_dropout_alpha),
            intent_label_smoothing=float(intent_label_smoothing))
        cells[-1].source += (
            '\n\nThis separate regularized revision preserves the previous revised notebook and its results. '
            'Only Stage 2 enables the two training regularizers: rare-word IDs are replaced by UNK with '
            '`p = alpha / (training_count + alpha)` while original character IDs, masks and targets remain intact; '
            'intent cross-entropy uses label smoothing during training only. The defaults here are '
            f'`word_dropout_alpha={word_dropout_alpha:g}` and `intent_label_smoothing={intent_label_smoothing:g}`. '
            'Development/test inputs are never masked and their intent CE remains unsmoothed. '
            'Frequency counts fit the original training split only; zero disables either option. '
            'The original-size Stage 1 configuration, model architecture, parameter count, CRF objective and '
            'development selection rule remain unchanged. These changes are hypotheses until measured.')
    if early_stopping:
        config['STAGE2_CONFIG']['training'].update(max_epochs=60, schedule_epochs=20,
            early_stopping_min_epochs=20, early_stopping_patience=8, early_stopping_min_delta=.0005)
        cells[-1].source += (
            '\n\nStage 2 now allows up to 60 epochs and stops after eight consecutive development '
            'evaluations without a joint-score gain of at least 0.0005 (0.05 percentage points), '
            'with stopping permitted from epoch 20. The unrounded joint score is '
            '`(intent_accuracy + slot_micro_f1) / 2`. Its patience anchor is separate from '
            'checkpoint selection: every actual new maximum is saved, including smaller gains. '
            'The first 20 epochs retain the historical two-epoch warmup and cosine decay; '
            'later epochs hold the final low learning rates. The best checkpoint is restored '
            'and independently verified after stopping. Stage 1 retains its complete 20-epoch budget. '
            'This is fresh training, with termination and effective policy recorded in the outputs.')
    code('\n\n'.join(name + ' = ' + pprint.pformat(value, sort_dicts=False) for name, value in config.items()), ('configuration',))
    definitions('src/comparison_contract.py', exact=True)
    definitions('src/revised_transformer_data.py')
    definitions('src/stage2_model.py', {'BIOCRF'})
    definitions('src/revised_transformer_model.py')
    definitions('src/revised_transformer_training.py')
    code('''import shutil
import platform
from IPython.display import Image, display
configuration = capture_configuration(NOTEBOOK_PATH, RUNTIME, STAGE1_CONFIG, STAGE2_CONFIG,
    RUN_OPTIONS, RUN_ROOT / 'run_configuration.json')
CODE_SHA256 = configuration['code_sha256']
rt_seed(RUNTIME)
DEVICE = torch.device(RUN_OPTIONS['device'])
if DEVICE.type == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA requested but unavailable; select the installed GPU kernel or explicitly choose cpu')
rt_write_json(RUN_ROOT / 'runtime.json', {'python': platform.python_version(), 'torch': str(torch.__version__),
    'numpy': np.__version__, 'device': str(DEVICE),
    'gpu': torch.cuda.get_device_name(DEVICE) if DEVICE.type == 'cuda' else None,
    'runtime': RUNTIME, 'code_sha256': CODE_SHA256})
print('Fresh run:', RUN_ROOT, 'device:', DEVICE, 'evaluator:', EVALUATOR_VERSION)
''')
    md('''## Verified original data and train-only features

Both implementations use the same original English-file SHA, ordered split IDs and complete contents. Every record is retained. Raw `utt` alone supplies tokens/characters; annotations supply targets. Word IDs are casefolded, character IDs preserve the original string, and dynamic token/character padding never truncates. Vocabulary and the complete 111-tag schema fit train only.
''')
    code('''DATASET_PATH = RUN_ROOT / 'data/1.0/data/en-US.jsonl'
cached = RUN_OPTIONS['dataset_path'] or os.environ.get('SC4002_DATASET_CACHE')
if cached:
    cached = Path(cached).expanduser().resolve()
    load_verified_massive(cached)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with cached.open('rb') as source, DATASET_PATH.open('xb') as target:
        shutil.copyfileobj(source, target)
raw_splits, dataset_manifest = ensure_verified_massive(DATASET_PATH)
bundle = build_data(raw_splits)
assert dataset_manifest['counts'] == {'train': 11514, 'dev': 2033, 'test': 2974}
rt_write_json(RUN_ROOT / 'dataset_identity.json', dataset_manifest)
rt_write_json(RUN_ROOT / 'mappings.json', bundle['mappings'])
rt_write_json(RUN_ROOT / 'data_statistics.json', bundle['stats'])
print('VERIFIED_DATASET:', json.dumps(dataset_manifest, indent=2))
print('TRAIN_FITTED_DATA_STATISTICS:', json.dumps(bundle['stats'], indent=2))
''')
    md('''## Diagnose the original implementation

1. **Unseen words collapse to UNK.** Train-only word lookup gives one shared word vector to every unseen word, even when the external Word2Vec source contains it. Development has 464 such token occurrences across 364 utterances; 215 utterances have slot-related OOV words. Character spelling is a reasonable targeted addition, not a guaranteed semantic fix.
2. **Slots are independent argmax decisions.** There is no learned BIO sequence decoder or intent-to-slot conditioning in the original model. Structured decoding and soft predicted-intent context address these limitations without using gold intent as a predictive feature.
3. **Mean-only intent pooling and copied initialization.** The original encoder layers are distinct parameters but begin with identical values. The revision constructs each block independently and learns attention pooling alongside max pooling. These are hypotheses to measure rather than proven causes of an accuracy gap.
4. **Optimization/evidence gaps.** The supplied sequence stops before training Stage 2 and does not produce test evidence. The revision includes both complete training loops, measured times, best-checkpoint snapshots, fresh-process reload checks and guarded final evaluation.
5. **Character-padding correctness.** The original optional CNN can pool bias-only padding windows. The revision masks character positions and makes padded tokens exactly zero. No original train/dev token exceeds 24 characters, so the old cutoff did not cause measured truncation here; full-character support is a robustness improvement.
6. **Balance feature branches before fusion.** A train-only initialization check on 128 original examples found character-feature RMS 0.6233 versus word-feature RMS 0.0500 in the first near-cap prototype. The current design normalizes both branches separately before concatenation, preventing that scale mismatch from immediately overwhelming pretrained word information. This was a controlled development-driven correction; it is not proof of a causal performance mechanism.

Current Transformer-v2 baseline performance is not available in the input notebook. Its older 83.47/66.43 figures used a different scorer and cannot demonstrate a v2 gain. The following fresh baseline/revised development results are the comparison evidence.
''')
    code('''archive = RUN_OPTIONS['embedding_archive'] or os.environ.get('SC4002_WORD2VEC_CACHE')
if archive is None:
    archive = RUN_ROOT / 'cache/nlpl_model_6.zip'
if STAGE1_CONFIG['model']['word_dim'] != STAGE2_CONFIG['model']['word_dim']:
    raise ValueError('Use one common word-embedding dimension for this comparison')
embedding_matrix, embedding_manifest = task_embeddings(bundle['mappings']['word_to_id'],
    source=RUN_OPTIONS['embedding_source'], seed=RUNTIME['seed'],
    dimension=STAGE1_CONFIG['model']['word_dim'], archive_path=archive, download=True)
rt_write_json(RUN_ROOT / 'embedding_initialization.json', embedding_manifest)
print('COMMON_WORD_INITIALIZATION:', json.dumps(embedding_manifest, indent=2))
for name, settings in [('stage1', STAGE1_CONFIG), ('stage2', STAGE2_CONFIG)]:
    rt_seed(RUNTIME)
    cfg = RevisedTransformerConfig.from_dict(settings['model'])
    probe = JointTransformer(len(bundle['mappings']['word_to_id']), len(bundle['mappings']['intent_to_id']),
        bundle['mappings']['tag_names'], len(bundle['mappings']['char_to_id']), cfg, embedding_matrix)
    print('ALL_INCLUSIVE_PARAMETERS', name, enforce_parameter_cap(probe))
    del probe
''')
    md('''## Revised architecture (default)

```text
Utterance → train-fitted Word2Vec table (300) ───────────────┐
          → masked character CNN (3/4/5 × 64 = 192) ───────┤
          separate branch norms → concatenate             │
          → projection/normalization (512)                 │
          → sinusoidal positions → 6 independent blocks ─┘
             8 heads; feedforward 1792; dropout 0.25
          ├─ learned attention + masked max → MLP → 60 intents
          └─ token states + soft predicted-intent context (64)
             → MLP → 111 BIO emissions → constrained CRF → spans
```

All encoder weights are trained from scratch. Both tasks update the shared encoder. The sum of per-utterance intent CE and per-real-token slot NLL is optimized jointly. CRF and CE loss heights have different meanings. Capacity alone can worsen overfitting; dropout, weight decay, scheduling and development checkpoint selection remain essential. Character/CRF and intent-slot interaction are motivated by [Ma and Hovy](https://aclanthology.org/P16-1101/) and [Goo et al.](https://aclanthology.org/N18-2118/), without inheriting those papers' benchmark gains.
''')
    md('## Train the original-size Transformer baseline')
    code('''stage1 = run_stage('stage1', STAGE1_CONFIG, RUNTIME, RUN_OPTIONS, bundle, embedding_matrix,
    DEVICE, RUN_ROOT / 'stage1', DATASET_PATH, CODE_SHA256, NOTEBOOK_PATH)
display(Image(filename=str(RUN_ROOT / 'stage1/learning_curves.png')))
''', ('training', 'stage1'))
    md('## Train the revised Transformer from fresh initialization')
    code('''stage2 = run_stage('stage2', STAGE2_CONFIG, RUNTIME, RUN_OPTIONS, bundle, embedding_matrix,
    DEVICE, RUN_ROOT / 'stage2', DATASET_PATH, CODE_SHA256, NOTEBOOK_PATH)
display(Image(filename=str(RUN_ROOT / 'stage2/learning_curves.png')))
''', ('training', 'stage2'))
    if early_stopping:
        cells[-1].source += "\nprint('STAGE2_TRAINING_TERMINATION:', json.dumps(stage2['termination'], indent=2))\n"
    md('''## Genuine comparison, errors and completion

Results use newly trained, development-selected and independently reloaded checkpoints. Preserve improvements and regressions. Two rows mean development-only; four rows include final test after each model was frozen. Error cases come from saved revised development predictions; proposed causes are inferences. This is a single-seed comparison of whole systems, not a controlled causal claim for any one addition.
''')
    code('''results = report_results({'stage1': stage1, 'stage2': stage2}, RUN_ROOT, bundle)
print('GENERATED_RESULTS:', json.dumps(results, indent=2))
print('GENUINE_DEVELOPMENT_ERRORS:', (RUN_ROOT / 'error_cases.json').read_text(encoding='utf-8'))
completion = {'status': 'complete', 'mode': 'final' if RUN_OPTIONS['evaluate_test'] else 'development_only',
    'evaluator_version': EVALUATOR_VERSION, 'dataset_sha256': DATASET_SHA256,
    'code_sha256': CODE_SHA256, 'configuration_sha256': configuration['configuration_sha256'],
    'dataset_identity': dataset_manifest, 'stages': {'stage1': stage1, 'stage2': stage2},
    'results': 'results.json', 'new_model_test_evaluations': {
        'stage1': stage1['test_model_evaluations'], 'stage2': stage2['test_model_evaluations']}}
assert len(results['rows']) == (4 if RUN_OPTIONS['evaluate_test'] else 2)
assert all(value['status'] == 'complete' for value in (stage1, stage2))
rt_write_json(RUN_ROOT / 'completion.json', completion)
print('REVISED_TRANSFORMER_COMPLETION:', json.dumps({key: completion[key] for key in ['status','mode','evaluator_version','new_model_test_evaluations']}))
''', ('completion',))
    md('''## What to report and what remains uncertain

Use this run's actual CSV/JSON, selected/trained epochs, complete parameter counts, measured memory/time, plots and genuine errors. Do not copy old scores or invent logs. If revised development performance does not improve, report the failed hypothesis and revisit training/development evidence in a separately registered run. Test remains reporting-only. Full-vocabulary Word2Vec lookup and pretrained encoders are deliberately excluded; token duplication/augmentation and external task examples are not used.

All Python source is inline. The builder's provenance is informational at runtime; no development source file, historical checkpoint or saved score is loaded. Only the verified original dataset and optional allowed word-vector asset are external inputs.
''')
    code_cells = [c for c in cells if c.cell_type == 'code']
    document = nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'SC4002 (.venv)', 'language': 'python', 'name': 'sc4002'},
        'language_info': {'name': 'python', 'version': '3.12.0'},
        'revised_transformer': {'schema_version': 1, 'status': 'pending_execution',
            'original_notebook': 'NLP_Final_project.ipynb',
            'original_notebook_sha256': hashlib.sha256((root / 'NLP_Final_project.ipynb').read_bytes()).hexdigest(),
            'code_cell_sha256': [hashlib.sha256(c.source.encode()).hexdigest() for c in code_cells],
            'configuration_cell_indices': [i for i,c in enumerate(code_cells) if 'configuration' in c.metadata.tags],
            'source_provenance': provenance},
        'comparison_contract': {'evaluator_version': 'exact_character_span_v2',
            'source_sha256': hashlib.sha256((root / 'src/comparison_contract.py').read_bytes()).hexdigest()}})
    nbformat.validate(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(nbformat.writes(document), encoding='utf-8', newline='\n')
    return {'notebook': str(output), 'cells': len(cells), 'code_cells': len(code_cells)}


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--regularized', action='store_true',
                        help='Build a separate notebook with Stage 2 word dropout .25 and intent smoothing .05')
    parser.add_argument('--early-stopping', action='store_true',
                        help='Build regularized Stage 2 with max60/min20/patience8 and the original 20-epoch LR schedule')
    args = parser.parse_args()
    regularized = args.regularized or args.early_stopping
    output = args.output or (ROOT / 'NLP_Final_project_revised_early_stopping.ipynb' if args.early_stopping
                            else ROOT / 'NLP_Final_project_revised_regularized.ipynb' if regularized else None)
    print(json.dumps(build(output=output, word_dropout_alpha=.25 if regularized else 0.,
                          intent_label_smoothing=.05 if regularized else 0., early_stopping=args.early_stopping)))

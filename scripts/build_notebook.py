"""Assemble a self-contained notebook from the verified source definitions.

This development-time builder is not needed by the resulting notebook. It
preserves core algorithms through AST extraction and records their provenance.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pprint

import nbformat


class RenameGlobals(ast.NodeTransformer):
    def __init__(self, names):
        self.names = names

    def visit_Name(self, node):
        node.id = self.names.get(node.id, node.id)
        return node

    def visit_FunctionDef(self, node):
        node.name = self.names.get(node.name, node.name)
        return self.generic_visit(node)


def extract_module(path, names=None, renames=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    kept = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and (node.module == "__future__" or (node.module or "").startswith("src")):
                continue
            kept.append(node)
        elif names is None or (isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names):
            kept.append(node)
    tree.body = kept
    if renames:
        tree = RenameGlobals(renames).visit(tree)
    ast.fix_missing_locations(tree)
    return "from __future__ import annotations\n\n" + ast.unparse(tree) + "\n"


def build(root: Path, output: Path, *, contract_version="v2"):
    if output.exists():
        previous = nbformat.read(output, as_version=4)
        if any(cell.cell_type == "code" and (cell.get("execution_count") is not None or cell.get("outputs"))
               for cell in previous.cells):
            raise FileExistsError("Preserve the executed notebook and its outputs; build to a new path for review")
    if contract_version != "v2":
        raise ValueError("Historical v1 builds require the archived original builder and source")
    manifest_path = root / "configs/evaluation_v2_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["snapshot_file_hashes"].items():
        if name.startswith(("src/", "configs/", "scripts/")) and hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Reviewed source/configuration changed before notebook construction: " + name)
    cells, provenance = [], {}

    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(text.strip()))

    def code(text, tags=()):
        cells.append(nbformat.v4.new_code_cell(text.strip() + "\n", metadata={"tags": list(tags)}))

    def definitions(name, selected=None, renames=None):
        path = root / name
        source = extract_module(path, selected, renames)
        provenance[name] = {"file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "selected_definitions": sorted(selected) if selected else "all",
                            "renamed_globals": renames or {},
                            "inline_code_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest()}
        code(source, ("definitions",))

    md("""# SC4002: joint intent classification and slot filling

This notebook contains all source for the two jointly trained systems and generates its own training, checkpoint and evaluation evidence. It uses the original MASSIVE 1.0 English splits and only the utterance as predictive input. Both models are randomly initialized and each complete system is below 20M parameters.

This is the fixed-configuration reproduction after a historical development study of three baseline and five Stage 2 candidates. It trains each selected configuration for 20 full epochs (40 here), selecting its best epoch again by development data. Historical selected epochs and performance are not used as training targets or manually supplied outputs.

**AI history:** the brief requires Stage 1 without generative AI; the user requested assistance despite that rule. Assistance actually occurred, and no lecturer-approved exception is claimed. Stage 2 analysis, verification and conversion also used the GPT-6-based Codex assistant. Group ID, names and verified human contributions remain pending at the user's request.

## Run instructions

Use Python 3.12, PyTorch **2.7.1 with CUDA 12.6**, NumPy 1.26.4, Matplotlib 3.10.8, nbformat 5.10.4, IPython and Jupyter. The tested GPU is an RTX 3060 Ti (8 GB). In a fresh environment install:

```powershell
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install numpy==1.26.4 matplotlib==3.10.8 nbformat==5.10.4 ipykernel==6.30.1 jupyterlab==4.4.10
$env:PYTHONHASHSEED='42'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python -m jupyterlab --no-browser
```

Choose that environment's kernel, keep this file named `main.ipynb`, and execute all cells in order from a fresh kernel. Use a directory with no existing `run_artifacts` folder. The notebook downloads the specified official archive, creates all evidence locally, and refuses to overwrite a completed execution. Internet is needed for the official dataset. The source hash check protects the reviewed frozen configuration; rebuild/review any intentional code change before accepting new outputs. The installed environment is a prerequisite; no repository modules, configuration files or previous checkpoint is required.
""")
    if contract_version == "v2":
        cells[0].source = """# SC4002: joint BiLSTM — evaluator v2

**Read run status from the final completion cell and `run_artifacts/completion.json`.**
This template requires fresh training; it does not supply default-model v2
performance results. This revision uses exact typed character-span occurrences
and the same verified original MASSIVE English data
as `NLP_Final_project.ipynb`. Earlier executed notebooks and results are archived
locally under `artifacts/evaluator_v2/prechange`; their text-set scores and selected
epochs are historical v1 observations, not results for this revision.

This standalone notebook embeds all required source. Edit only the labelled
configuration cell, save, then restart the kernel and run all cells. Defaults
train the selected BiLSTM settings for 20 epochs each. Development-only runs
support tuning; final runs reload, freeze and test chosen settings. No rebuild
or manual hash edit is needed for supported settings. Use a fresh execution
directory, the tested Python 3.12/CUDA environment and `main.ipynb` filename.
Launch Jupyter with `PYTHONHASHSEED=42` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, then execute every cell in order.

The brief requires Stage 1 without generative AI; assistance actually occurred
at the user's request and no lecturer-approved exception is claimed. Group
details and verified human contributions remain pending. No old score is used
as a training target, and no historical checkpoint is loaded.
"""
    code("""from __future__ import annotations
import os
from pathlib import Path
import sys
if os.environ.get('PYTHONHASHSEED') != '42':
    raise RuntimeError('Restart the kernel from Jupyter launched with PYTHONHASHSEED=42; setting it inside a running interpreter is insufficient.')
if os.environ.get('CUBLAS_WORKSPACE_CONFIG', ':4096:8') != ':4096:8':
    raise RuntimeError('Restart with CUBLAS_WORKSPACE_CONFIG=:4096:8.')
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
NOTEBOOK_PATH = (Path.cwd() / 'main.ipynb').resolve()
if not NOTEBOOK_PATH.is_file():
    raise FileNotFoundError('Keep main.ipynb in the kernel working directory.')
RUN_ROOT = Path.cwd() / 'run_artifacts'
RUN_ROOT.mkdir(exist_ok=False)
os.environ['MPLCONFIGDIR'] = str(RUN_ROOT / 'matplotlib_cache')
print('Fresh execution directory:', Path.cwd())
print('Interpreter:', sys.executable)
""", ("bootstrap",))
    md("""## Editable run configuration — change settings here

Edit literal dictionary values below, save, then Restart Kernel and Run All in
a fresh directory. All other implementation cells remain protected.
`evaluate_test=False` trains/selects/verifies on development only; `True` also
freezes and tests once per stage. Use test only after choosing hyperparameters.
`dataset_path=None` downloads original English data; a path string reuses a
verified local file, copying it into this run. AdamW and task weights 1.0 are
fixed; complete models must remain within 20M. Actual settings are recorded.
""")
    config = {"RUNTIME": json.loads((root / "configs/protocol.json").read_text())["runtime"],
              "STAGE1_CONFIG": json.loads((root / "configs/phase4/stage1_dropout03.json").read_text()),
              "STAGE2_CONFIG": json.loads((root / "configs/phase5/char_crf_intent_context.json").read_text()),
              "RUN_OPTIONS": {"evaluate_test": True, "dataset_path": None, "device": "cuda",
                              "verification_timeout_seconds": 300}}
    config['STAGE1_CONFIG'].pop('diagnostics', None)
    code("\n\n".join(name + " = " + pprint.pformat(value, sort_dicts=False) for name, value in config.items()), ("configuration",))
    md("""## Shared runtime and typed prediction contract

Seed 42 is reset before each model construction. Training uses FP32, deterministic algorithms, deterministic cuDNN, disabled TF32, and a persistent seeded data-loader generator with zero workers. Selection uses the unrounded arithmetic mean of development intent accuracy and slot micro F1; an exact tie retains the earlier epoch. Offsets are Python character positions and are retained for audit.
""")
    definitions("src/runtime.py", {"relative_project_path", "configure_reproducibility", "select_device", "dataloader_generator", "selection_score", "is_improvement", "configuration_hash"})
    code("""from __future__ import annotations
def project_root(explicit_root=None):
    if explicit_root is None:
        raise ValueError('Pass the notebook execution root explicitly.')
    return Path(explicit_root).expanduser().resolve()
""", ("definitions",))
    definitions("src/schema.py")
    if contract_version == "v2":
        # Keep this cell byte-identical across both notebooks, including imports.
        contract_source = (root / "src/comparison_contract.py").read_text(encoding="utf-8")
        cells.append(nbformat.v4.new_code_cell(contract_source, metadata={"tags": ["definitions", "comparison_contract"]}))
    definitions("scripts/notebook_support.py")
    code("""import platform
import torch
import numpy as np
from IPython.display import Image, Markdown, display
configuration_snapshot = nb_capture_configuration(NOTEBOOK_PATH, RUNTIME, STAGE1_CONFIG,
    STAGE2_CONFIG, RUN_OPTIONS, RUN_ROOT / 'run_configuration.json')
CODE_SHA256 = configuration_snapshot['code_sha256']
configure_reproducibility(RUNTIME)
torch.set_default_dtype(torch.float32)
DEVICE = select_device(RUN_OPTIONS['device'])
runtime_evidence = {'python': platform.python_version(), 'executable': sys.executable,
    'torch': str(torch.__version__), 'numpy': np.__version__, 'cuda_runtime': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(DEVICE) if DEVICE.type == 'cuda' else None,
    'gpu_memory_bytes': torch.cuda.get_device_properties(DEVICE).total_memory if DEVICE.type == 'cuda' else None,
    'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
    'settings': RUNTIME, 'implementation_sha256': CODE_SHA256,
    'execution_id': RUN_ROOT.parent.name, 'started_at_utc': datetime.now(timezone.utc).isoformat()}
nb_support_write_json(RUN_ROOT / 'runtime.json', runtime_evidence)
print('PHASE7_RUNTIME_JSON:', json.dumps(runtime_evidence, indent=2))
""")
    md("""## Original data acquisition and preprocessing

The source below safely streams the official MASSIVE 1.0 archive and extracts exactly `1.0/data/en-US.jsonl`. Every original record remains in its released partition. No augmentation or additional task examples are used.

The fixed tokenizer is `\\w+|[^\\w\\s]`, preserving case and original offsets. Dynamic right padding removes no tokens. Word/label mappings and the character vocabulary are fitted on train only. BIO annotation boundaries must round-trip to the original exact text; malformed records stop the run rather than being filtered.
""")
    definitions("src/acquisition.py")
    definitions("src/data.py")
    code("""import shutil
DATASET_PATH = RUN_ROOT / ENGLISH_PATH
if RUN_OPTIONS['dataset_path'] is None:
    verified_rows, dataset_identity = ensure_verified_massive(DATASET_PATH)
else:
    cached_source = Path(RUN_OPTIONS['dataset_path']).expanduser().resolve()
    load_verified_massive(cached_source)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with cached_source.open('rb') as cached_input, DATASET_PATH.open('xb') as cached_output:
        shutil.copyfileobj(cached_input, cached_output)
    verified_rows, dataset_identity = load_verified_massive(DATASET_PATH)
DATA_SHA256 = file_sha256(DATASET_PATH)
assert DATA_SHA256 == 'c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e'
verified_rows, dataset_identity = load_verified_massive(DATASET_PATH)
nb_support_write_json(RUN_ROOT / 'dataset_identity.json', dataset_identity)
print('VERIFIED_DATASET_IDENTITY:', json.dumps(dataset_identity, indent=2))
splits = load_massive(DATASET_PATH)
assert {part: len(rows) for part, rows in splits.items()} == EXPECTED_COUNTS
for part, rows in splits.items():
    assert [row.record_id for row in rows] == [row['id'] for row in verified_rows[part]]
    for row in rows:
        assert [slot.to_dict() for slot in row.slots] == parse_annotated_spans(row.utt, row.annot_utt)
print('ORIGINAL_SPLIT_COUNTS:', json.dumps({part: len(rows) for part, rows in splits.items()}))
mappings = fit_mappings(splits['train'])
assert (len(mappings.token_to_id), len(mappings.intent_to_id), len(mappings.slot_types), len(mappings.tag_to_id)) == (5146, 60, 55, 111)
assert configuration_hash(mappings.to_dict()) == '6a35e931d9fbca5fd48dcf95ac6598f98036fa584558949394fed7c30ef98e6b'
prepared = {part: prepare_split(rows, mappings) for part, rows in splits.items()}
for part, rows in prepared.items():
    assert len({row.example.record_id for row in rows}) == EXPECTED_COUNTS[part]
    for row in rows:
        inverse_tags = {index: tag for tag, index in mappings.tag_to_id.items()}
        assert decode_bio(row.example.utt, row.tokens, tuple(inverse_tags[index] for index in row.tag_ids)) == row.example.slots
nb_support_write_json(RUN_ROOT / 'preprocessing.json', mappings.to_dict())
nb_support_write_json(RUN_ROOT / 'split_membership.json', {'record_ids': {part: [row.record_id for row in rows] for part, rows in splits.items()}, 'dataset_sha256': DATA_SHA256})
print('Train-fitted mappings:', len(mappings.token_to_id), 'words;', len(mappings.intent_to_id), 'intents;', len(mappings.slot_types), 'slot types')
print('All original annotation and token/BIO span roundtrips passed.')
""")
    md("""## One shared evaluator

Intent accuracy is the fraction of exact labels. Evaluation v2 pools exact typed
character-span occurrences per original utterance: `2TP / (2TP + FP + FN)`.
Type, start, exclusive end and literal text must match; repeated text at distinct
positions counts separately. Original utterance slices are validated. There is
no normalization or partial credit. Empty-slot denominator gives zero;
missing/extra/duplicate record IDs and duplicate identical spans are errors.
Both notebooks inline the same tokenizer-independent scorer. This follows
MASSIVE's occurrence-sensitive evaluation principle without claiming parity
with an unavailable lecturer reference scorer.
""")
    definitions("src/metrics.py")
    code("""gold_check = [row.gold_prediction() for row in splits['train'][:32]]
identity_check = evaluate_predictions(gold_check, gold_check)
assert identity_check.intent_accuracy == 1.0 and identity_check.slot_micro_f1 == 1.0
print('Shared evaluator identity check on 32 original training annotations passed (structural check, not learned model performance).')
""")
    md("""## Stage 1: joint BiLSTM baseline

Default settings: trainable 256-dimensional word embeddings feed one packed bidirectional LSTM with 512 units in each direction and dropout 0.3. The editable configuration cell determines this run's actual dimensions and optimizer settings. Masked maximum pooling feeds the 60-class intent head; token outputs feed 111 BIO tags. Equal-weight intent cross-entropy per utterance and slot cross-entropy per real token update the joint system through AdamW. Original defaults are learning rate 0.001, weight decay 0.0001, batch 32 and gradient clipping 5.

The following core algorithms are transferred from the verified scripts. Checkpoints snapshot independent CPU tensors and optimizer state. The notebook-specific driver below replaces filesystem-bound historical experiment orchestration while preserving the model, batch order, epoch algorithms, task metrics and selection rule.
""")
    definitions("src/models.py")
    definitions("src/training.py", {"LossBundle", "compute_losses", "_synchronize", "_device_batch", "_partitions", "_summarize", "_validate_limit", "train_epoch", "evaluate_model"})
    md("""## Stage 2 definitions: characters, predicted-intent context and CRF

Development diagnosis motivated spelling features, structured boundaries and intent-to-slot interaction. The default Stage 2 settings append a 64-channel width-3 character CNN (32-dimensional character embeddings), retain one shared BiLSTM, add 64-dimensional soft predicted-intent context, and use a BIO-constrained CRF. The editable configuration controls this run's actual settings; gold intent never enters the forward computation.

Its joint loss is intent cross-entropy per utterance plus sequence CRF negative log likelihood divided by real-token count. CRF start/transition constraints apply to likelihood and Viterbi; padding never contributes. The two slot objectives have different meanings, so loss heights are not an architecture performance comparison. The original bounded ablations had mixed results; this notebook reproduces the selected combination rather than repeating that search. [Character CNN/BiLSTM/CRF motivation](https://aclanthology.org/P16-1101/) and [intent/slot interaction motivation](https://aclanthology.org/N18-2118/) do not imply those papers' benchmark gains here.
""")
    definitions("src/stage2_model.py")
    renames = {"make_loader": "s2_make_loader", "new_model": "s2_new_model", "_to_device": "s2_to_device", "_forward": "s2_forward", "_epoch": "s2_epoch", "evaluate": "s2_evaluate"}
    definitions("src/stage2_experiment.py", set(renames), renames)
    definitions("scripts/notebook_workflow.py")
    definitions("src/diagnostics.py", {"validate_development", "unique_pairs", "unmatched_category", "pair_counts", "counts_metrics", "analyze_predictions"})
    code("""char_to_id = fit_char_vocab(prepared['train'])
assert len(char_to_id) == 34
assert configuration_hash(char_to_id) == 'a69bea15839987ecae4529c8914917686f989acf738890d023f0940d83e9c58e'
nb_support_write_json(RUN_ROOT / 'char_to_id.json', char_to_id)
for stage, config in [('stage1', STAGE1_CONFIG), ('stage2', STAGE2_CONFIG)]:
    nb_validate_config(stage, config, RUNTIME)
    configure_reproducibility(RUNTIME)
    probe = nb_new_model(stage, mappings, char_to_id, config, DEVICE)
    counts = enforce_parameter_cap(probe, 20000000)
    print('COMPLETE_SYSTEM_PARAMETER_COUNT', stage, json.dumps(counts))
    del probe
torch.cuda.empty_cache()
print('Only original utterance-derived word/character features enter either joint model.')
""")
    md("""## Train and select Stage 1

All configured training and development epochs are printed and retained (default 20). A strict improvement in the unrounded development mean saves the checkpoint; exact ties keep the earlier epoch. Training/test repetition guards preserve evidence. Seeds reset before model construction.
""")
    code("""stage1_training = nb_train_stage('stage1', prepared, mappings, char_to_id, STAGE1_CONFIG,
    RUNTIME, DEVICE, RUN_ROOT / 'stage1', DATA_SHA256, CODE_SHA256)
display(Image(filename=str(RUN_ROOT / 'stage1/learning_curves.png')))
""", ("training", "stage1"))
    md("""## Reload, verify and freeze Stage 1; final evaluation

A new Python process receives ordinary Python generated from the visible definitions in this notebook, independently refits train mappings and reproduces all ordered development predictions from the newly saved checkpoint. The parent also reloads a fresh model, verifies development again, persists a freeze, and then performs one complete test pass. Model weights and checkpoint bytes must remain unchanged. Test results are reporting-only.
""")
    code("""stage1_reload = nb_run_fresh_verifier(NOTEBOOK_PATH, RUN_ROOT / 'stage1/best.pt', DATASET_PATH, CODE_SHA256, DEVICE,
    timeout=RUN_OPTIONS.get('verification_timeout_seconds', 300))
stage1_final = nb_finalize_stage(RUN_ROOT / 'stage1/best.pt', prepared['dev'], prepared['test'],
    mappings, char_to_id, DEVICE, DATA_SHA256, CODE_SHA256, evaluate_test=RUN_OPTIONS['evaluate_test'])
""", ("final_evaluation", "stage1"))
    md("""## Development-only diagnosis and AI collaboration

The following analysis uses this notebook's Stage 1 development predictions and training support counts. OOV groups, intent confusions and missed-slot boundary categories are descriptive associations, not proof of causes. Character features, CRF decoding and predicted-intent context were accepted as bounded hypotheses; more capacity, blanket class weighting and test-driven iteration were not included. The selected settings stay fixed regardless of this reproduction's test scores. Human reasoning/contributions and full genuine chat export remain part of the final collaboration/report work.
""")
    code("""stage1_diagnosis = nb_display_diagnosis(splits['train'], splits['dev'],
    read_predictions(RUN_ROOT / 'stage1/dev_predictions.jsonl'), mappings, RUN_ROOT / 'stage1_diagnosis.json')
""")
    md("""## Train and select Stage 2 from random initialization

Stage 2 starts afresh with the configured seed. It uses original training data, a persistent shuffled loader, the same development selection policy and its configured epoch budget (default 20). It does not warm-start from Stage 1 or use gold intent as input. All learned components count toward the cap.
""")
    code("""stage2_training = nb_train_stage('stage2', prepared, mappings, char_to_id, STAGE2_CONFIG,
    RUNTIME, DEVICE, RUN_ROOT / 'stage2', DATA_SHA256, CODE_SHA256)
display(Image(filename=str(RUN_ROOT / 'stage2/learning_curves.png')))
""", ("training", "stage2"))
    md("""## Reload, verify and freeze Stage 2; final evaluation

The same independent reload and freeze-before-test policy applies. No test score changes the selected epoch, architecture or configuration. Preserve and report any regression as part of the result.
""")
    code("""stage2_reload = nb_run_fresh_verifier(NOTEBOOK_PATH, RUN_ROOT / 'stage2/best.pt', DATASET_PATH, CODE_SHA256, DEVICE,
    timeout=RUN_OPTIONS.get('verification_timeout_seconds', 300))
stage2_final = nb_finalize_stage(RUN_ROOT / 'stage2/best.pt', prepared['dev'], prepared['test'],
    mappings, char_to_id, DEVICE, DATA_SHA256, CODE_SHA256, evaluate_test=RUN_OPTIONS['evaluate_test'])
""", ("final_evaluation", "stage2"))
    md("""## Generated comparison and genuine remaining errors

Rows come from this run's reloaded checkpoints: four rows in final mode, two development rows in tuning mode. `epochs_trained` is the configured budget; `selected_epoch` is separate. Full-precision scores/counts remain in JSON. Genuine Stage 2 development errors use original labels and predictions; explanations are inferences.
""")
    code("""results = nb_build_result_table(stage1_final, stage2_final, RUN_ROOT)
error_cases = nb_show_error_cases(splits['train'], splits['dev'],
    read_predictions(RUN_ROOT / 'stage2/dev_predictions.jsonl'), mappings, RUN_ROOT / 'error_cases.json')
""")
    md("""## Limitations and completion checks

This is a single-seed reproduction of a bounded search, not a statistical significance study or global optimum. Valid BIO paths do not guarantee correct entity boundaries or types. Character features cannot resolve all unfamiliar entity meanings, and intent context can propagate intent mistakes. Training/development loss divergence and genuine errors should guide any separately authorized future study using training/development evidence.

Scoring uses exact typed character-span occurrences; parity with an unavailable lecturer scorer is unverified. Full workflow time includes train, development scoring, checkpoint writes and logs, excluding setup, plotting, independent verification and final evaluation. Per-evaluation timer boundaries differ between stage algorithms and are not comparable microbenchmarks. Historical v1 scores/checkpoint selections are not current results. The final report must use actual newly executed outputs, and the submission must preserve them.
""")
    code("""for stage, summary, settings in [('stage1', stage1_final, STAGE1_CONFIG), ('stage2', stage2_final, STAGE2_CONFIG)]:
    assert summary['status'] == 'complete' and summary['epochs_trained'] == settings['training']['max_epochs']
    assert summary['test_model_evaluations'] == int(RUN_OPTIONS['evaluate_test'])
    assert summary['model_weights_unchanged'] and summary['checkpoint_bytes_unchanged']
    assert summary['training_and_tuning_closed'] == RUN_OPTIONS['evaluate_test']
    assert summary['fresh_process_reload']['ordered_predictions_equal']
    assert summary['fresh_process_reload']['absolute_score_difference'] == 0
    assert summary['dev']['num_examples'] == 2033
    if RUN_OPTIONS['evaluate_test']:
        assert summary['test']['num_examples'] == 2974
assert len(results['rows']) == (4 if RUN_OPTIONS['evaluate_test'] else 2)
completion = {'schema_version': 1, 'status': 'complete', 'completed': True,
    'evaluator_version': EVALUATOR_VERSION, 'dataset_identity': dataset_identity,
    'mode': 'final' if RUN_OPTIONS['evaluate_test'] else 'development_only',
    'run_configuration': 'run_configuration.json',
    'configuration_sha256': configuration_snapshot['configuration_sha256'],
    'execution_id': runtime_evidence['execution_id'], 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    'dataset': ENGLISH_PATH, 'dataset_sha256': DATA_SHA256,
    'implementation_sha256': CODE_SHA256, 'code_sha256': CODE_SHA256,
    'stages': {stage: {'directory': stage, 'summary': stage + '/final_summary.json'} for stage in ['stage1', 'stage2']},
    'total_epochs_trained': stage1_final['epochs_trained'] + stage2_final['epochs_trained'],
    'results': 'results.json', 'error_cases': 'error_cases.json',
    'new_model_test_evaluations': {'stage1': int(RUN_OPTIONS['evaluate_test']), 'stage2': int(RUN_OPTIONS['evaluate_test'])},
    'group_details': 'Pending at user request',
    'scope': 'Fresh configured notebook training; no historical checkpoint or score loaded.'}
nb_support_write_json(RUN_ROOT / 'completion.json', completion)
print('PHASE7_COMPLETION_JSON:', json.dumps(completion, ensure_ascii=False))
""", ("completion",))
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "SC4002 (.venv)", "language": "python", "name": "sc4002"},
        "language_info": {"name": "python", "version": "3.12.0"},
        "phase7": {"schema_version": 1, "built_at_utc": datetime.now(timezone.utc).isoformat(),
                   "builder_source_manifest": provenance, "standalone": True,
                   "configuration_cell_indices": [i for i, cell in enumerate([c for c in cells if c.cell_type == 'code'])
                                                  if 'configuration' in cell.metadata.get('tags', [])],
                   "code_cell_sha256": [hashlib.sha256(cell.source.encode("utf-8")).hexdigest() for cell in cells if cell.cell_type == "code"]}})
    nbformat.validate(notebook)
    if contract_version == "v2":
        notebook.metadata["comparison_contract"] = {
            "evaluator_version": "exact_character_span_v2",
            "dataset_sha256": "c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e",
            "source_sha256": hashlib.sha256(contract_source.encode("utf-8")).hexdigest(),
            "source_file": "src/comparison_contract.py",
        }
        notebook.metadata["evaluator_v2"] = {"schema_version": 2, "status": "pending_fresh_training",
            "code_cell_sha256": notebook.metadata["phase7"]["code_cell_sha256"],
            "reviewed_source_manifest": "configs/evaluation_v2_manifest.json"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(nbformat.writes(notebook), encoding="utf-8", newline="\n")
    return {"notebook": str(output), "cells": len(cells), "code_cells": sum(cell.cell_type == "code" for cell in cells), "source_modules": len(provenance)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "main.ipynb")
    parser.add_argument("--contract-version", choices=("v2",), default="v2")
    args = parser.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[1], args.output.resolve(), contract_version=args.contract_version), indent=2))


if __name__ == "__main__":
    main()

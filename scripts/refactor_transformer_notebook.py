"""Create a readable standalone notebook without changing its numerical methods.

Black is a development-only formatter. It is not imported by the generated
notebook. Install it in an isolated directory with the command in --help.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'artifacts/notebook_readability_refactor/formatting_tools'
if TOOLS.is_dir():
    sys.path.insert(0, str(TOOLS))

MODULES = ('src/comparison_contract.py', 'src/revised_transformer_data.py',
           'src/stage2_model.py', 'src/revised_transformer_model.py',
           'src/revised_transformer_training.py')


def source_text(cell):
    value = cell['source']
    return ''.join(value) if isinstance(value, list) else value


def text_hash(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def normalize_ast(source):
    """Ignore only module prose and repeated future-annotation directives."""
    tree = ast.parse(source)
    tree.body = [node for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and node.module == '__future__')
        and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                 and isinstance(node.value.value, str))]
    return ast.dump(tree, include_attributes=False)


def format_python(source):
    try:
        import black
    except ImportError as error:
        raise RuntimeError('Install development formatter: python -m pip install --target '
            'artifacts/notebook_readability_refactor/formatting_tools black==25.1.0') from error
    if black.__version__ != '25.1.0':
        raise RuntimeError('Use Black 25.1.0 for this reproducible readability transformation')
    result = black.format_str(source, mode=black.Mode(line_length=92,
        target_versions={black.TargetVersion.PY312}))
    black.assert_equivalent(source, result)
    return result


def module_nodes(root, name):
    """Keep original comments/formatting inside definitions, including decorators."""
    source = (root / name).read_text(encoding='utf-8')
    lines = source.splitlines(keepends=True)
    nodes = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom) and (node.module == '__future__'
                or (node.module or '').startswith('src')):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if name == 'src/stage2_model.py':
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if not isinstance(node, ast.ClassDef) or node.name != 'BIOCRF':
                continue
        start = min([node.lineno]+[d.lineno for d in getattr(node, 'decorator_list', [])])-1
        nodes.append((getattr(node, 'name', None), ''.join(lines[start:node.end_lineno]).rstrip()))
    if name == 'src/stage2_model.py':
        nodes.insert(0, (None, 'from typing import Sequence\n\nimport torch\nfrom torch import nn'))
    return nodes


# Each entry begins a new cell; following definitions remain in that cell until
# the next named entry. This preserves the original order of executable code.
SECTIONS = {
    'src/revised_transformer_data.py': {
        'transformer_tokens': ('Tokenization and training labels',
            'Keep original character offsets and fit vocabularies and targets on training data.', False),
        'collate_transformer': ('Padding and batches',
            'Pad words, characters and labels dynamically; masks identify real tokens.', False),
        '_vector_archive_member': ('Download and verify word vectors',
            'Load the permitted Word2Vec archive with bounded size and verified contents.', True),
        '_embedding_vocabulary': ('Initialize the task embedding table',
            'Stream source vectors into the training vocabulary; retain only the task table.', False)},
    'src/stage2_model.py': {'BIOCRF': ('Constrained BIO decoder',
        'The CRF scores label sequences and prevents invalid BIO transitions.', False)},
    'src/revised_transformer_model.py': {
        'RevisedTransformerConfig': ('Model configuration',
            'One configuration describes both the baseline and revised architecture.', False),
        'DynamicCharacterCNN': ('Character features',
            'Convolutions capture spelling patterns; padding cannot contribute pooled features.', False),
        'SinusoidalPositions': ('Position encoding and encoder blocks',
            'Add positions; Stage 2 initializes each encoder block independently.', False),
        'JointTransformer': ('Joint intent and slot model',
            'Both losses update the shared encoder. Stage 2 also conditions slot emissions on predicted intent probabilities.', False),
        'count_parameters': ('Whole-model parameter count',
            'Count embeddings, encoder, heads and CRF against the 20-million limit.', False)},
    'src/revised_transformer_training.py': {
        'rt_json_hash': ('Result-file utilities',
            'Small helpers save JSON and identify recorded inputs.', True),
        'notebook_configuration': ('Saved configuration checks',
            'Record the exact saved settings and implementation used by each run.', True),
        'rt_seed': ('Reproducibility and state utilities',
            'Seed the generators, measure GPU time and copy or identify checkpoint tensors.', True),
        'validate_training_config': ('Training configuration checks',
            'Reject invalid learning rates, regularization and stopping settings.', True),
        'FrequencyWordDropout': ('Rare-word dropout',
            'Replace word IDs with UNK using training frequencies; retain their real characters.', False),
        'learning_rate_factor': ('Learning rates and patience stopping',
            'Keep the 20-epoch schedule, then hold its floor. Track patience separately from the actual best score.', False),
        '_rt_loader': ('Training loaders and model setup',
            'Use every original record once per epoch and create a fresh joint model.', False),
        '_rt_epoch': ('One training or evaluation epoch',
            'Stage 2 training applies both regularizers; evaluation uses clean inputs and unsmoothed intent loss.', False),
        '_rt_evidence': ('Checkpoint and prediction files',
            'Save development predictions and reload the selected model with its recorded settings.', True),
        'verify_checkpoint': ('Independent checkpoint verification',
            'A fresh Python process must reproduce all selected development predictions.', True),
        '_rt_curves': ('Learning curves',
            'Plot losses and both development metrics over completed epochs.', False),
        'run_stage': ('Stage training loop',
            'Train, select the earliest actual best checkpoint, stop by the declared policy and verify the reload.', False),
        'report_results': ('Result table and development errors',
            'Generate metrics and error examples from the completed runs.', False)}
}


def split_module(root, name):
    groups = []
    title = {'src/revised_transformer_data.py':'Data imports and constants',
             'src/revised_transformer_model.py':'Model imports',
             'src/revised_transformer_training.py':'Training imports'}.get(name,'Imports')
    header = (title, '', True)
    current, chunks = header, []
    for symbol, source in module_nodes(root, name):
        if symbol in SECTIONS[name]:
            if chunks:
                groups.append((current, chunks))
            current, chunks = SECTIONS[name][symbol], []
        chunks.append(source)
    if chunks:
        groups.append((current, chunks))
    # The CRF needs only three imports; keep them with its single class.
    if name == 'src/stage2_model.py':
        groups = [(SECTIONS[name]['BIOCRF'], [part for _, chunks in groups for part in chunks])]
    formatted = []
    for section, parts in groups:
        text, previous_import = '', False
        for part in parts:
            is_import = all(isinstance(node, (ast.Import, ast.ImportFrom))
                            for node in ast.parse(part).body)
            separator = '\n' if previous_import and is_import else '\n\n'
            text += (separator if text else '') + part
            previous_import = is_import
        formatted.append((section, format_python('from __future__ import annotations\n\n'+text+'\n')))
    return formatted


def refactor_document(document, root=ROOT):
    document = deepcopy(document)
    original_cells = document.cells
    code_cells = [cell for cell in original_cells if cell.cell_type == 'code']
    if len(code_cells) != 13:
        raise ValueError('Expected the reviewed 13-code-cell early-stopping notebook')
    old_metadata = document.metadata['revised_transformer']
    if old_metadata.get('definition_layout_version', 1) != 1:
        raise ValueError('Refactor the preserved original, not an already split notebook')
    config_cell = next(c for c in code_cells if 'configuration' in c.metadata.get('tags', []))
    cells, provenance = [], {}

    def md(source):
        cells.append(nbformat.v4.new_markdown_cell(source.strip()))

    def code(source, tags=(), **metadata):
        cells.append(nbformat.v4.new_code_cell(source, metadata={'tags':list(tags), **metadata}))

    md('''# Joint intent classification and slot filling

Train two joint models on the original MASSIVE 1.0 English splits. Stage 1 is the original-size Transformer; Stage 2 adds character features, intent context and a constrained CRF. Default parameter counts are **4,345,243** and **19,748,595**, including embeddings and both task heads. Encoder weights are trained from scratch; Word2Vec initializes the word table.

## Run this notebook

1. Use the **SC4002 (.venv)** kernel with `PYTHONHASHSEED=42` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before launch.
2. Copy this notebook into a fresh folder and keep its filename. Each run creates `transformer_run_artifacts/`.
3. Edit and save the configuration below, then **Restart Kernel and Run All Cells**.

All implementation code is embedded. Utility cells are collapsed by default and can be expanded for inspection.''')
    code(format_python(source_text(code_cells[0])), ('bootstrap',))
    md('''## Configuration

| Setting | Stage 1 | Stage 2 |
| --- | --- | --- |
| Encoder | 4 blocks, width 256 | 6 blocks, width 512 |
| Training | 20 epochs | Maximum 60, minimum 20, patience 8 |
| Word dropout / intent smoothing | Disabled | 0.25 / 0.05 |

Stage 2 word replacement uses `0.25 / (training_frequency + 0.25)` and preserves character inputs. Both regularizers apply during training only.

The stopping score is `(intent_accuracy + slot_micro_f1) / 2`. A gain of **0.0005** (0.05 percentage points) resets patience; every actual new best checkpoint is still saved. Both stages use two warmup epochs and cosine decay through epoch 20. Stage 2 then holds the low learning rates.

Keep `evaluate_test=False` while experimenting. Fit features and labels on training data; select checkpoints on development data. Use test only to report a frozen configuration. Cache paths may be valid local files or `None` for verified downloads.''')
    code(format_python(source_text(config_cell)), ('configuration',))

    for name in MODULES:
        expected = old_metadata['source_provenance'][name]['file_sha256']
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Implementation changed before refactoring: '+name)
        fragments = []
        if name == 'src/comparison_contract.py':
            md('''## Dataset and exact-span evaluator

The verified source contains **11,514 training**, **2,033 development** and **2,974 test** utterances. Intent accuracy uses exact labels. Slot micro-F1 pools exact slot type, original character boundaries and text across utterances. Expand the canonical contract below to inspect its implementation.''')
            text = (root/name).read_text(encoding='utf-8')
            code(text, ('definitions', 'comparison_contract'), source_module=name,
                 section_name='Canonical evaluator', jupyter={'source_hidden':True})
            fragments.append(text)
        else:
            if name == 'src/revised_transformer_data.py':
                md('## Input features and word embeddings')
            elif name == 'src/stage2_model.py':
                md('''## Stage 2 architecture

```text
Words + character CNN -> branch normalization -> projection + positions
    -> shared Transformer encoder
       |-> attention/max pooling -> intent classifier
       |-> token states + predicted-intent context -> BIO emissions -> CRF
```

Character features support unfamiliar spellings. Shared intent context and sequence decoding connect the two tasks. Stage 1 uses mean pooling for intent and independent token labels, with no character encoder, intent context or CRF.''')
            elif name == 'src/revised_transformer_training.py':
                md('## Optimization, checkpoints and evaluation')
            for (title, description, hidden), text in split_module(root, name):
                md('### '+title+'\n\n'+description)
                code(text, ('definitions',), source_module=name, section_name=title,
                     jupyter={'source_hidden':hidden})
                fragments.append(text)
        joined = '\n\n'.join(fragments)
        provenance[name] = {'file_sha256':expected, 'inline_sha256':text_hash(joined),
            'inline_cell_sha256':[text_hash(text) for text in fragments],
            'semantic_ast_sha256':text_hash(normalize_ast(joined))}

    md('## Prepare the run')
    runtime = source_text(code_cells[7]).replace('import platform\n', 'import platform\nimport pandas as pd\n')
    code(format_python(runtime))
    md('### Load the verified data\n\nBuild training-only mappings and save the complete dataset manifest.')
    data = source_text(code_cells[8])
    data = data.replace("print('VERIFIED_DATASET:', json.dumps(dataset_manifest, indent=2))\n"
        "print('TRAIN_FITTED_DATA_STATISTICS:', json.dumps(bundle['stats'], indent=2))",
        "print('Verified split sizes:', dataset_manifest['counts'])\n"
        "print('Vocabulary:', len(bundle['mappings']['word_to_id']), 'words;', "
        "len(bundle['mappings']['tag_names']), 'BIO tags')")
    code(format_python(data))
    md('### Initialize embeddings and check parameter counts\n\nBoth stages start with the same task embedding table. Only this table is retained for prediction.')
    embeddings = source_text(code_cells[9]).replace(
        "print('COMMON_WORD_INITIALIZATION:', json.dumps(embedding_manifest, indent=2))",
        "print('Word embeddings:', embedding_matrix.shape, 'source:', embedding_manifest['source'])")
    code(format_python(embeddings))
    md('## Train Stage 1\n\nTrain the baseline for 20 epochs, then reload its best development checkpoint.')
    code(format_python(source_text(code_cells[10])), ('training','stage1'))
    md('## Train Stage 2\n\nStart from fresh initialization and stop using the configured patience policy. The final output records the selected epoch and stopping reason.')
    code(format_python(source_text(code_cells[11])), ('training','stage2'))
    md('''## Results and error examples

The table reports actual parameter counts, trained and selected epochs, intent accuracy and exact slot micro-F1. Two rows indicate development-only mode; final evaluation adds the two test rows. The examples below are actual development disagreements; full records and spans remain in `error_cases.json`.''')
    result = source_text(code_cells[12])
    result = result.replace("print('GENERATED_RESULTS:', json.dumps(results, indent=2))\n"
        "print('GENUINE_DEVELOPMENT_ERRORS:', (RUN_ROOT / 'error_cases.json').read_text(encoding='utf-8'))",
        "display(pd.DataFrame(results['rows']))\n"
        "cases = json.loads((RUN_ROOT / 'error_cases.json').read_text(encoding='utf-8'))['cases']\n"
        "error_rows = [{\n"
        "    'utterance': case['utt'],\n"
        "    'gold intent': case['gold']['intent'],\n"
        "    'predicted intent': case['prediction']['intent'],\n"
        "    'gold slots': case['gold']['slots'],\n"
        "    'predicted slots': case['prediction']['slots'],\n"
        "} for case in cases[:5]]\n"
        "display(pd.DataFrame(error_rows))\n"
        "print('Complete metrics and error records:', RUN_ROOT)")
    code(format_python(result), ('completion',))
    md('''## Saved artifacts and project record

`transformer_run_artifacts/` contains the full configuration, dataset identity, metrics, learning curves, checkpoints, reload checks and error examples. Each stage records its actual trained epoch count separately from its selected checkpoint.

This notebook uses AI-assisted code, including Stage 1 at the user's request; no lecturer-approved exception is claimed. Keep the genuine collaboration history with the project. Group details and verified human contributions remain pending.''')
    document.cells = cells
    new_codes = [c for c in cells if c.cell_type == 'code']
    metadata = document.metadata['revised_transformer']
    metadata.update(status='pending_execution', definition_layout_version=2,
        code_cell_sha256=[text_hash(c.source) for c in new_codes],
        configuration_cell_indices=[i for i,c in enumerate(new_codes) if 'configuration' in c.metadata.get('tags',[])],
        source_provenance=provenance,
        readability_refactor={'formatter':'Black 25.1.0', 'line_length':92,
            'changes':'Section layout, source formatting, unused CRF imports and concise presentation',
            'numerical_implementation_changed':False})
    nbformat.validate(document)
    return document


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'NLP_Final_project_revised_early_stopping.ipynb')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new review/output path')
    original=nbformat.read(args.input,as_version=4)
    revised=refactor_document(original)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(revised,args.output)
    print(json.dumps({'notebook':str(args.output),'cells':len(revised.cells),
        'code_cells':sum(c.cell_type=='code' for c in revised.cells),'status':'pending_execution'}))


if __name__=='__main__':main()

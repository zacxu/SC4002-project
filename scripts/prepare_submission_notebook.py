"""Prepare the accepted readable notebook for a fresh final evaluation run."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pprint
import sys

import nbformat

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.revised_transformer_training import notebook_configuration, notebook_identity, rt_json_hash


def prepare(source, output, protocol_path):
    source,output=Path(source),Path(output)
    if output.exists() or Path(protocol_path).exists():raise FileExistsError('Choose new final-run paths')
    document=nbformat.read(source,as_version=4)
    original=deepcopy(document)
    config=notebook_configuration(source)
    config['RUN_OPTIONS'].update(evaluate_test=True,dataset_path=None,embedding_archive=None)
    for cell in document.cells:
        tags=cell.metadata.get('tags',[])
        if 'configuration' in tags:
            cell.source='\n\n'.join(name+' = '+pprint.pformat(value,sort_dicts=False,width=90)
                                      for name,value in config.items())+'\n'
        if 'bootstrap' in tags:
            line=next(line for line in cell.source.splitlines() if line.startswith('NOTEBOOK_PATH ='))
            cell.source=cell.source.replace(line,line+'\nif not NOTEBOOK_PATH.is_file():\n'
                                           "    NOTEBOOK_PATH = Path.cwd() / 'main.ipynb'")
        for stage in ('stage1','stage2'):
            if 'training' in tags and stage in tags:
                text=(f"print(f\"{stage} selected checkpoint: epoch {{{stage}['selected_epoch']}}; "
                    f"validation joint score {{100 * {stage}['selection_score']:.2f}}%; "
                    f"file {{RUN_ROOT / '{stage}' / {stage}['checkpoint']}}\")\n"
                    f"print(f\"{stage} training workflow: {{{stage}['training_workflow_seconds']:.2f}} seconds\")\n\n")
                marker=cell.source.index('display(')
                cell.source=cell.source[:marker]+text+cell.source[marker:]
        if 'completion' in tags:
            old=next(line for line in cell.source.splitlines() if line.startswith('display(pd.DataFrame(results['))
            cell.source=cell.source.replace(old,"""result_table = pd.DataFrame(results['rows']).replace({
    'model_version': {'stage1': 'Stage 1', 'stage2': 'Stage 2'},
    'split': {'dev': 'Validation', 'test': 'Test'},
}).rename(columns={
    'model_version': 'Model', 'split': 'Split',
    'total_parameters': 'Total parameters', 'trainable_parameters': 'Trainable parameters',
    'intent_accuracy_percent': 'Intent accuracy (%)', 'slot_micro_f1_percent': 'Slot micro-F1 (%)',
    'epochs_trained': 'Epochs trained', 'selected_epoch': 'Selected epoch',
})
display(result_table)""")
        if cell.cell_type=='markdown':
            cell.source=cell.source.replace('Copy this notebook into a fresh folder and keep its filename.',
                'Copy this notebook into a fresh folder. It supports this filename and the required submission filename `main.ipynb`.')
            cell.source=cell.source.replace('Keep `evaluate_test=False` while experimenting.',
                'This final run uses `evaluate_test=True`. For later development experiments, use `evaluate_test=False`.')
            if cell.source=='## Optimization, checkpoints and evaluation':
                cell.source+='\n\nBoth stages minimize intent cross-entropy plus slot negative log-likelihood, normalized per utterance and real token respectively. Stage 1 uses token cross-entropy; Stage 2 uses a CRF. AdamW uses batch size 32, learning rate 0.0005 (0.0001 for word embeddings), weight decay 0.01 and gradient clipping at 1.0.'
        if cell.cell_type=='code':
            cell.outputs=[];cell.execution_count=None;compile(cell.source,'final-notebook','exec')
    codes=[cell for cell in document.cells if cell.cell_type=='code']
    document.metadata.revised_transformer.update(status='pending_execution',submission_filename='main.ipynb',
        code_cell_sha256=[hashlib.sha256(cell.source.encode()).hexdigest() for cell in codes],
        configuration_cell_indices=[i for i,cell in enumerate(codes) if 'configuration' in cell.metadata.get('tags',[])])
    assert [c.source for c in original.cells if 'definitions' in c.metadata.get('tags',[])]==[
        c.source for c in document.cells if 'definitions' in c.metadata.get('tags',[])]
    output.parent.mkdir(parents=True,exist_ok=True);nbformat.write(document,output)
    assert notebook_configuration(output)==config
    reference=ROOT/'artifacts/notebook_readability_refactor/full_notebook/transformer_run_artifacts'
    records={s:json.loads((reference/s/'final_summary.json').read_text()) for s in ('stage1','stage2')}
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    protocol={'registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'purpose':'Final submission evaluation; no model or hyperparameter tuning',
        'source_input_sha256':sha(source),'source_code_sha256':notebook_identity(output),
        'configuration_sha256':rt_json_hash(config),'all_definition_cells_byte_identical':True,
        'allowed_changes':['final test mode','portable caches','main.ipynb fallback','checkpoint/time prints','result labels','method explanation'],
        'expected_selected_model_states':{s:r['model_state_sha256'] for s,r in records.items()},
        'expected_development_metrics':{s:r['dev']['metrics'] for s,r in records.items()},
        'reference_checkpoint_sha256':{s:sha(reference/s/'best.pt') for s in records},
        'new_test_passes_expected_per_model':1,'test_previously_exposed':True,
        'report_cover_source_sha256':sha(ROOT/'SC4002 Project Report.docx')}
    Path(protocol_path).write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    return {'notebook':str(output),'code_cells':len(codes),'final_evaluation':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'NLP_Final_project_revised_early_stopping.ipynb')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.input,args.output,args.protocol)))

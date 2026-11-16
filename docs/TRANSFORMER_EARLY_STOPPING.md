# Train the revised Transformer beyond epoch 20

**Final submission update:** the selected notebook now contains a complete final-mode run, with `evaluate_test=True`, portable `None` cache settings and support for the submission filename `main.ipynb`. The final Stage 1 test result is **82.92% intent / 66.77% slot F1**; Stage 2 is **85.44% / 77.79%**. Selected weights and validation scores exactly reproduce the earlier frozen development models. See the [final check](FINAL_SUBMISSION_CHECK.md). Use `evaluate_test=False` for future experiments; the remaining historical descriptions below refer to the development studies that established the stopping policy.

Use [NLP_Final_project_revised_early_stopping.ipynb](../NLP_Final_project_revised_early_stopping.ipynb) for the latest Stage 2 training policy. It retains word dropout alpha **0.25** and intent label smoothing **0.05**, with the same **19,748,595-parameter** joint architecture. The earlier 20-epoch notebooks and their measured results remain available as references.

The delivered seed-42 run **trained 28 epochs, stopped through patience and selected epoch 20**. Its selected development result remains **86.62% intent accuracy / 80.84% slot micro-F1**. All first-20 losses, metrics, orders and masks match the earlier run, and its selected weights/predictions are identical. Additional training did not improve this seed's joint score; the registered confirmation criterion was not triggered, so seeds 17 and 101 were not extended. No new test evaluation was performed. Independent full-notebook acceptance passed.

## Notebook layout

Use the single **Configuration** cell for ordinary hyperparameter edits. The following sections explain data and labels, the Stage 2 architecture, regularization, the training loop and checkpoint verification. Support utilities are collapsed by default and can be expanded. Keep those cells: the standalone notebook and its independent reload process use them.

Results and genuine development error cases are displayed as tables. Full configurations, manifests, predictions and error records are saved under `transformer_run_artifacts/`. No extra formatting dependency is needed to execute the notebook. Architecture descriptions and parameter counts describe the defaults; edited configurations are counted and recorded during each run.

The current readable version completed all **35 code cells** in a fresh run and passed independent acceptance. Its complete training trajectories, selected weights, development predictions and result files reproduce the earlier notebook exactly. The current run is under `artifacts/notebook_readability_refactor/full_notebook/`; its [delivery record](../results/notebook_readability_refactor/notebook_delivery.json) links the acceptance file and archived original. Older delivery records describe the earlier notebook layout at the time they were produced.

## Stopping and learning rates

The editable Stage 2 `training` dictionary contains:

```python
'max_epochs': 60,
'schedule_epochs': 20,
'early_stopping_min_epochs': 20,
'early_stopping_patience': 8,
'early_stopping_min_delta': 0.0005,
```

The monitor is the unrounded mean of development intent accuracy and exact slot micro-F1, both on the 0–1 scale. A gain of at least **0.0005**, equivalent to **0.05 percentage points**, resets the patience anchor. Smaller cumulative gains can eventually reach that threshold. Ties do not reset it. Training stops after eight successive epochs without such a gain, once the minimum epoch requirement is met, or at the maximum of 60 epochs. If the maximum and patience coincide, the recorded reason is `max_epochs`.

Every actual new maximum still saves the best checkpoint, even if its improvement is smaller than the patience threshold. The selected checkpoint is the earliest actual maximum over all completed epochs; it is reloaded and independently verified after training. The patience threshold is a heuristic, not a statistical significance test.

The first 20 epochs retain the reference schedule exactly: two epochs of linear warmup and cosine decay to 10% of the peak rates. After epoch 20, learning rates remain at **0.00001 for embeddings** and **0.00005 for other parameters**. Increasing `max_epochs` alone does not stretch this schedule while `schedule_epochs` remains 20. A future change to `schedule_epochs` creates a different learning-rate experiment.

Stage 1 retains its complete 20-epoch budget. Set Stage 2 patience to `0` to disable early stopping and run its complete configured maximum. Missing optional policy fields retain the old full-budget behavior. Each run starts from fresh initialization; checkpoint-resume training is not implemented.

## Run in Jupyter

Use the existing [environment setup](../README.md#1-install-the-environment-once). From PowerShell in the repository root:

```powershell
$projectPath = (Get-Location).Path
$runPath = Join-Path $projectPath 'artifacts\early_stopping_runs\jupyter_001'
if (Test-Path -LiteralPath $runPath) { throw 'Choose a new run directory.' }
New-Item -ItemType Directory -Path $runPath | Out-Null
Copy-Item -LiteralPath (Join-Path $projectPath 'NLP_Final_project_revised_early_stopping.ipynb') -Destination $runPath
$env:PYTHONHASHSEED='42'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
Push-Location -LiteralPath $runPath
try {
    & (Join-Path $projectPath '.venv\Scripts\python.exe') -m jupyterlab
}
finally {
    Pop-Location
}
```

Open the copied notebook, choose **SC4002 (.venv)**, save configuration changes and **Restart Kernel and Run All Cells**. Keep `evaluate_test=False` for experiments. The notebook embeds its complete implementation. Keep the copied filename and its kernel working directory together. The recorded notebook contains actual local cache paths; set `dataset_path` and `embedding_archive` to `None` or valid paths when moving it to another machine.

## Run from the terminal

```powershell
.\.venv\Scripts\python.exe scripts/execute_revised_transformer.py --notebook NLP_Final_project_revised_early_stopping.ipynb --execution-dir artifacts/early_stopping_runs/run_001 --dataset-cache data/1.0/data/en-US.jsonl --embedding-cache data/pretrained/nlpl_model_6.zip
```

Choose a fresh directory. Both notebook stages execute, and genuine outputs are retained in the source notebook and isolated run copy. Cache flags are optional if the notebook can download the verified assets. A completed run writes `transformer_run_artifacts/results.csv`, `results.json`, `completion.json`, both stages' learning curves, best checkpoints and independent reload evidence.

Stage 2 additionally records `stopping_policy` and `termination` in its run summaries, the patience state in each history row, and `schedule_horizon_evidence.json` at epoch 20. The notebook prints the termination reason, actual epochs and optimizer-step count. Report the actual trained epoch count separately from the selected checkpoint epoch and configured maximum.

Audit a completed full notebook without further model inference:

```powershell
.\.venv\Scripts\python.exe scripts/validate_revised_transformer.py --notebook artifacts/early_stopping_runs/run_001/NLP_Final_project_revised_early_stopping.ipynb --run-root artifacts/early_stopping_runs/run_001/transformer_run_artifacts --output artifacts/early_stopping_runs/run_001_acceptance.json
```

Use a new acceptance filename. The validator independently reconstructs the patience decisions, actual optimizer steps, learning-rate schedule and checkpoint selection, as well as the existing dataset, parameter, prediction and reload checks. A run ending before the maximum is accepted only when its history proves the configured stopping condition was reached.

## Reproduce the longer-training comparison

The registered study uses `artifacts/transformer_early_stopping/`. The seed-42 run executes the complete notebook; the other registered seeds run Stage 2 only from the notebook's embedded definitions. The dataset, initialization, first-20 learning rates, record order, masks and selected prefix weights/predictions are checked against the preserved 20-epoch combined model. The extra epochs use development only.

For a separate repeat, choose a new output root:

```powershell
.\.venv\Scripts\python.exe scripts/run_transformer_early_stopping.py --output-root artifacts/transformer_early_stopping_repeat_001 --seeds 42
```

The predeclared rule runs confirmation seeds 17 and 101 if the extended seed-42 joint score exceeds its own first-20 maximum:

```powershell
.\.venv\Scripts\python.exe scripts/run_transformer_early_stopping.py --output-root artifacts/transformer_early_stopping_repeat_001 --seeds 17 101
```

Existing or changed arms are rejected rather than overwritten. A changed implementation requires a new output root. The runner exposes no test evaluation option.

Read [the measured results](TRANSFORMER_EARLY_STOPPING_RESULTS.md) for actual stop epochs, selected scores and limitations. The previously exposed test set is not used for this comparison, and development gains are not test-gain claims.

Regenerate the separate comparison report from completed saved predictions and CPU checkpoint evidence:

```powershell
.\.venv\Scripts\python.exe scripts/analyze_transformer_early_stopping.py
.\.venv\Scripts\python.exe scripts/plot_transformer_early_stopping.py
```

For another study root, pass `--run-root`, `--output-dir` and `--markdown` with its paths to the analyzer, then `--results` and `--output` to the plotter. Both commands perform no model inference. The analyzer verifies each complete epoch, stopping trace, schedule floor, original train-fitted mappings, selected tensors, development scores and first-20 parity before reporting results. A triggered confirmation remains partial until both follow-up seeds are complete; the plotter requires the completed comparison.

The [development trajectory figure](assets/transformer_early_stopping.png) shows the original 20-epoch budget, the selected checkpoint and actual stopping epoch for every completed seed.

The earlier 20-epoch comparison's exact implementation is archived under `artifacts/transformer_regularization/registered_sources/`, bound to its original protocol and hashes. The historical analyzer can verify those archived source bytes without executing them. Reusing the saved `confidence.json` through its explicit `--confidence` option remains supported; rerunning historical confidence inference with changed runtime source is rejected. Keep the original artifacts and source archive together when preserving that experiment.

## Maintainers: rebuild the readable notebook

Routine notebook hyperparameter changes do not need a rebuild. After deliberately changing implementation files under `src/`, build a fresh raw notebook, format/split it, then execute and validate the new source. The formatter is isolated from the project's runtime environment:

```powershell
.\.venv\Scripts\python.exe -m pip install --target artifacts/notebook_readability_refactor/formatting_tools black==25.1.0
.\.venv\Scripts\python.exe scripts/build_revised_transformer_notebook.py --early-stopping --output artifacts/notebook_rebuild/raw/NLP_Final_project_revised_early_stopping.ipynb
.\.venv\Scripts\python.exe scripts/refactor_transformer_notebook.py --input artifacts/notebook_rebuild/raw/NLP_Final_project_revised_early_stopping.ipynb --output artifacts/notebook_rebuild/readable/NLP_Final_project_revised_early_stopping.ipynb
.\.venv\Scripts\python.exe scripts/execute_revised_transformer.py --notebook artifacts/notebook_rebuild/readable/NLP_Final_project_revised_early_stopping.ipynb --execution-dir artifacts/notebook_rebuild/run_001 --dataset-cache data/1.0/data/en-US.jsonl --embedding-cache data/pretrained/nlpl_model_6.zip
```

Use new paths for another rebuild. The refactor tool preserves executable definitions, removes unused CRF imports and rebuilds source manifests for the split cells. It clears prior outputs; only a new complete execution supplies results for the changed source. Keep the previous accepted notebook and its run artifacts as a separate record.

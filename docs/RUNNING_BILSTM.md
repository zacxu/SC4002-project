# Run and retune the BiLSTM notebook

This guide applies to [main.ipynb](../main.ipynb), which trains **both** our baseline BiLSTM (Stage 1) and enhanced BiLSTM (Stage 2), using evaluator `exact_character_span_v2`. The Transformer comparison has its own [notebook](../NLP_Final_project.ipynb) and is not launched by this workflow.

After environment setup, **the notebook is sufficient to produce new results**. Its required model, data, scoring and training definitions are inline. For an ordinary experiment, edit its configuration cell, save, then run all cells in a fresh directory/kernel. No repository imports, hash edits or notebook rebuild are required for supported hyperparameter changes.

## One-time setup

Use Windows x64, Python 3.12 and a compatible NVIDIA CUDA GPU/driver. Development used PyTorch 2.7.1+cu126 and an RTX 3060 Ti with 8 GB VRAM. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name sc4002 --display-name 'SC4002 (.venv)'
.\.venv\Scripts\python.exe scripts/check_environment.py --device cuda
```

The single [requirements.txt](../requirements.txt) includes both notebook families' Python dependencies. The BiLSTM needs no pretrained embeddings. Internet is needed to acquire official data unless you supply the exact verified local JSONL through `RUN_OPTIONS['dataset_path']`.

## Run interactively in Jupyter

Start PowerShell in the repository root. Choose an unused run name:

```powershell
$projectPath = (Get-Location).Path
$runPath = Join-Path $projectPath 'artifacts\notebook_runs\trial_001'
if (Test-Path -LiteralPath $runPath) { throw 'Choose a new run directory.' }
New-Item -ItemType Directory -Path $runPath | Out-Null
Copy-Item -LiteralPath (Join-Path $projectPath 'main.ipynb') -Destination (Join-Path $runPath 'main.ipynb')
$env:PYTHONHASHSEED='42'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
Push-Location -LiteralPath $runPath
try {
    & (Join-Path $projectPath '.venv\Scripts\python.exe') -m jupyterlab --no-browser
}
finally {
    Pop-Location
}
```

Open the displayed Jupyter URL in your browser, then:

1. Open the copied **`main.ipynb`** and choose **SC4002 (.venv)** as its kernel.
2. Find the cell tagged **`configuration`** containing `RUNTIME`, `STAGE1_CONFIG`, `STAGE2_CONFIG` and `RUN_OPTIONS`.
3. Choose development-only or final evaluation and edit the desired dictionary values.
4. **Save before execution.** The runtime reads the saved notebook to identify its source and effective configuration.
5. Choose **Restart Kernel and Run All Cells**. Keep the kernel working directory as the folder containing the notebook.
6. Wait for the final completion output and save the notebook with its genuine outputs.

Keep the filename `main.ipynb`. Startup variables must be set before Jupyter launches; setting `PYTHONHASHSEED` in an already running kernel is insufficient. Do not execute later cells against stale model/optimizer variables after changing settings.

Each run creates `run_artifacts/` beside its notebook and refuses to overwrite existing evidence. For another experiment, create another new directory and fresh notebook copy/kernel. A restart in the same folder is not an exact interrupted-training resume.

## Run automatically from PowerShell

For unchanged settings in the root notebook:

```powershell
.\.venv\Scripts\python.exe scripts/execute_notebook.py --execution-dir artifacts/notebook_runs/final_001
```

To execute a saved notebook with edited configuration:

```powershell
.\.venv\Scripts\python.exe scripts/execute_notebook.py --notebook artifacts/notebook_runs/trial_001/main.ipynb --execution-dir artifacts/notebook_runs/trial_002
```

The `--notebook` source must be saved. The `--execution-dir` must not exist and must stay inside the repository. The executor copies the source to that fresh directory as `main.ipynb`, sets deterministic startup, selects the registered project kernel and executes every cell. Only a fully successful run is copied back to the **source passed to `--notebook`**. With the default source this updates root `main.ipynb`; the second example updates `trial_001/main.ipynb` while also retaining the independent executed copy in `trial_002/`.

The executor preserves actual partial outputs/evidence on failure and does not silently resume a previous checkpoint. Its execution log is written beside the run directory as `<run-name>_execution.json`.

Its default timeout is **900 seconds per code cell**. A large epoch budget may exceed that because a training cell performs all configured epochs. Set a larger positive number of seconds or disable this executor limit:

```powershell
.\.venv\Scripts\python.exe scripts/execute_notebook.py --execution-dir artifacts/notebook_runs/long_run_001 --cell-timeout -1
```

This CLI limit is separate from the fresh-process checkpoint-verification timeout below. Interactive Jupyter execution does not use the executor's `--cell-timeout` setting.

## Edit hyperparameters in the configuration cell

Edit literal values **inside the four existing dictionaries**, then save. Do not add statements that mutate settings later. The notebook records effective settings/code identities while checking protected definitions.

| Dictionary / fields | Meaning |
| --- | --- |
| `STAGE1_CONFIG['model']` | Baseline embedding dimension, hidden units **per direction**, LSTM layer count and dropout |
| `STAGE2_CONFIG['model']` | Enhanced dimensions, character features, intent context and CRF settings |
| Each stage's `training.learning_rate` / `weight_decay` | AdamW learning rate and weight decay |
| Each stage's `training.batch_size` / `max_epochs` | Examples per batch and complete train/development epochs |
| Each stage's `training.gradient_clip_norm` | Gradient norm clipping threshold |
| `RUNTIME['seed']` | Training/model RNG seed; startup `PYTHONHASHSEED` remains 42 |
| `RUN_OPTIONS['evaluate_test']` | `False` for development-only tuning; `True` for final test evaluation |
| `RUN_OPTIONS['dataset_path']` | `None` to download official data, or a string path to the exact original English JSONL |
| `RUN_OPTIONS['device']` | Defaults to `'cuda'` |
| `RUN_OPTIONS['verification_timeout_seconds']` | Fresh-process development-reload timeout: defaults to `300` seconds; positive integer or `None` for unlimited |

Defaults remain E256/H512 per direction, one LSTM layer, dropout 0.3, learning rate 0.001, batch 32 and 20 epochs. For example, replace the existing Stage 1 learning rate with `0.0005`, batch size with `16`, or maximum epochs with `30`. Changing Stage 1 does not change Stage 2 automatically; edit each intentionally.

AdamW and task-loss weights of 1.0 remain fixed. Alternative optimizers/loss weighting require an implementation change. Supported model changes still must satisfy each model's structural validation and the full 20M parameter cap. Actual total/trainable counts are computed every run; historical counts/epochs are not targets.

To reuse existing official data, set the existing `dataset_path` value to an absolute string path such as `'C:/Users/Admin/Desktop/SC4002/data/1.0/data/en-US.jsonl'`. Forward slashes avoid Python backslash escapes. The notebook verifies the digest and original records, then copies that verified file into the new run's data tree; it does not inherit old checkpoints or change partitions. Relative paths resolve from the execution directory, so use an absolute path when sharing one local data cache between runs.

Protected model/data/evaluator edits differ from ordinary configuration changes. Do not remove guards or manually replace hash lists. Intentional implementation changes require source review, tests and a new reviewed build.

## Tune on development, then produce final results

1. Change the configuration's existing `evaluate_test` value to **`False`**.
2. Run each candidate in its own fresh directory. Every original training record is used; each epoch is scored on the original development split.
3. Compare the **unrounded mean of development intent accuracy and slot F1**, preserving candidate settings/evidence. Exact epoch ties keep the earlier checkpoint.
4. Choose settings using development results only. Do not use test scores to choose candidates, epochs or further changes.
5. With settings fixed, change `evaluate_test` to **`True`** and start a new full run. It trains/selects again, reloads/verifies the selected checkpoints, freezes them and performs one full test evaluation per stage.

Development-only runs generate two development rows. They are useful tuning evidence but do not supply the four development/test rows required for the final comparison. Both modes save checkpoints, learning curves and fresh-process reload checks. Enabling final evaluation later does not resume the old run.

Exact interrupted-run continuation is not implemented: every RNG/loader state needed for exact resumption is not retained. Fix the cause of a failed run and use a new execution directory.

Changing the evaluator can change the best epoch. Old v1 selected epochs 19/11 and their scores are historical, not targets or proof that an old checkpoint is v2-best.

## Output files and what they mean

All paths are relative to the execution directory containing the run's `main.ipynb`:

| File | Meaning |
| --- | --- |
| `run_artifacts/run_configuration.json` | Effective four configuration dictionaries and code/configuration identities |
| `run_artifacts/dataset_identity.json` | Official source SHA, counts, ordered split IDs and complete split-content identities |
| `run_artifacts/runtime.json` | Interpreter/device and deterministic runtime evidence |
| `run_artifacts/stage1/run.json`, `stage2/run.json` | Stage settings and run identity |
| `run_artifacts/stage1/history.jsonl`, `stage2/history.jsonl` | Every complete epoch's losses, development metrics and selection score |
| `run_artifacts/stage1/history.csv`, `stage2/history.csv` | Corresponding epoch tables |
| `run_artifacts/stage1/learning_curves.png`, `stage2/learning_curves.png` | Genuine loss/metric curves |
| `run_artifacts/stage1/best.pt`, `stage2/best.pt` | Best-development weights, optimizer state and provenance |
| `run_artifacts/stage1/training_summary.json`, `stage2/training_summary.json` | Actual trained epochs, selected epoch/score and timing |
| `run_artifacts/stage1/reload_check.json`, `stage2/reload_check.json` | Independent reload/development-prediction verification |
| `run_artifacts/stage1/final_summary.json`, `stage2/final_summary.json` | Post-reload stage summary; its mode distinguishes development-only evidence from final development/test evidence |
| `run_artifacts/results.csv` | Two development rows or four development/test rows, depending on mode |
| `run_artifacts/results.json` | Formatted rows plus full-precision metrics/raw counts |
| `run_artifacts/error_cases.json` | Genuine remaining Stage 2 development errors for final reporting |
| `run_artifacts/completion.json` | Completion status and declared `final` or `development_only` mode |

Checkpoint directories also retain selected development predictions and, for final mode, freeze/test records and final predictions. Preserve the whole run directory with the executed notebook; copying only CSV loses provenance. These generated files are ignored by Git because they contain downloaded data, large checkpoints and run evidence.

Results display percentages to two decimals while JSON retains full precision. Selected checkpoint epoch and total trained epochs are separate fields. Read generated scores and preserve regressions as well as improvements.

## Verify a completed run

```powershell
.\.venv\Scripts\python.exe scripts/validate_notebook.py --notebook artifacts/notebook_runs/final_001/main.ipynb --execution-dir artifacts/notebook_runs/final_001/run_artifacts --output artifacts/notebook_runs/final_001/acceptance.json
```

`--execution-dir` is **`run_artifacts`**, not its parent notebook directory. The validator checks outputs, declared configuration, histories, settings, weights and predictions for the run's mode, without new training or trained-model test inference. It refuses to overwrite an existing validation output; choose another filename for a repeat audit. Accepted development-only evidence is not final-test submission acceptance.

To check common scoring and data identity separately:

```powershell
.\.venv\Scripts\python.exe scripts/check_comparison_contract.py
```

That check expects verified local `data/1.0/data/en-US.jsonl` or an explicit `--dataset` path. It does not download pretrained vectors, train models or prove learned-model performance. Normal notebook execution performs its own acquisition/reuse and identical identity checks.

## Notebook settings versus repository configurations

The `configs/` JSON files record registered/default script settings. An existing notebook embeds its settings; editing JSON alone does **not** change it. For ordinary future experiments, edit the notebook configuration cell.

The [builder](../scripts/build_notebook.py) transfers intentional source/configuration changes after review. It checks the [reviewed source manifest](../configs/evaluation_v2_manifest.json) and refuses to overwrite a notebook containing execution evidence. Build to a new path, review/test it and retain the old executed notebook. Routine configuration-cell edits do not need this workflow.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Run directory or `run_artifacts` exists | Preserve it and choose a new run directory. |
| Wrong kernel / CUDA unavailable | Select SC4002 (.venv) and run the environment check in that environment. |
| Startup hash-seed error | Stop the kernel/server, set PowerShell startup variables, launch Jupyter again and use a fresh kernel. |
| Configuration edits appear ignored | Save, restart and run from the first cell; check you edited the notebook being executed. |
| Protected source manifest mismatch | Restore protected source or make a reviewed rebuild; do not replace hashes manually. |
| Dataset digest mismatch | Preserve the conflicting file and reacquire the official release; do not relax the hash or replace partitions. |
| GPU out of memory | Reduce batch size/model dimensions in a new run; parameter count does not bound activation memory. |
| Interrupted execution | Inspect its partial notebook/logs, fix the cause and start a fresh run. Exact resume is not implemented. |
| Automated run times out in a long training cell | Use a larger `--cell-timeout` or `--cell-timeout -1`, then start a fresh run. |
| Fresh-process reload verification times out | Increase `RUN_OPTIONS['verification_timeout_seconds']`, or set it to `None`, save and start a fresh run. This is separate from the executor cell timeout. |
| No test rows | `evaluate_test=False` intentionally produces only development rows. |

The assignment requires genuine outputs, one final notebook and a report/collaboration record. Do not type metrics manually, use historical v1 outputs as v2 or submit an incomplete experiment as the final result.

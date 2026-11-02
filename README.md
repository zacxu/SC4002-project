# SC4002: joint intent classification and slot filling

Two model families for original **MASSIVE 1.0 English** predict intent and slot spans together: [main.ipynb](main.ipynb) contains our two BiLSTM stages; [NLP_Final_project.ipynb](NLP_Final_project.ipynb) is the supplied Transformer implementation; [NLP_Final_project_revised.ipynb](NLP_Final_project_revised.ipynb) is the separate improved Transformer study. Each stage is one joint model, and its complete parameter count must stay below **20 million**, including embeddings and output heads.

**Yes: after environment setup, you can run `main.ipynb` to train, select checkpoints and generate results.** For later experiments, change its editable configuration cell, save it, and run all cells in a **new directory with a fresh kernel**. Ordinary hyperparameter changes do not require rebuilding the notebook or manually changing source hashes. Both BiLSTM stages run; the notebook is not a Stage 1-only launcher.

The active notebooks use evaluator **`exact_character_span_v2`**. Read each run's completion and acceptance records for its actual execution status. Historical v1 scores do not become v2 results merely by changing the scorer. The [full BiLSTM run guide](docs/RUNNING_BILSTM.md) covers interactive execution, tuning, output files and troubleshooting.

## Revised Transformer notebook

Use [NLP_Final_project_revised.ipynb](NLP_Final_project_revised.ipynb) for the improved Transformer: six independently initialized encoder blocks, width 512, eight heads, masked character CNN, attention/max intent pooling, predicted-intent context and a constrained CRF. The default complete model has **19,748,595 parameters**. It also trains the original-size **4,345,243-parameter** baseline under the same declared new training protocol, data, initialization and evaluator.

After the environment setup below, execute it from PowerShell in a fresh run directory:

```powershell
.\.venv\Scripts\python.exe scripts/execute_revised_transformer.py --execution-dir artifacts/revised_transformer/run_001
```

It automatically downloads the original data and permitted Word2Vec archive if caches are not supplied. The original and revised Transformer notebooks are standalone: open the revised file in Jupyter with the project kernel, save its configuration and Run All in a fresh folder. The default is **development-only**; set `evaluate_test=True` only for a new chosen-configuration final run. Outputs are in that run's `transformer_run_artifacts/` directory. See [TRANSFORMER_REVISION.md](docs/TRANSFORMER_REVISION.md) for measured diagnosis, exact architecture, cache options, both execution methods and independent acceptance instructions. The original Transformer and existing BiLSTM evidence are preserved.

## 1. Install the environment once

The supported target is **Windows x64, Python 3.12, PyTorch 2.7.1+cu126 and an NVIDIA CUDA GPU**. Development used an RTX 3060 Ti with 8 GB VRAM. The BiLSTM notebook defaults to CUDA.

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name sc4002 --display-name 'SC4002 (.venv)'
.\.venv\Scripts\python.exe scripts/check_environment.py --device cuda
```

[requirements.txt](requirements.txt) is the single pinned dependency file for both notebook families. The Transformer additionally downloads permitted pretrained Word2Vec vectors; our BiLSTM requires no pretrained assets. Internet access is needed for official dataset acquisition and any Word2Vec download.

## 2. Run the BiLSTM: choose Jupyter or terminal

After the one-time setup above, open PowerShell in the **repository root** (the folder containing `main.ipynb`). Choose either option below; both run the same notebook and evaluator.

### Option A: open Jupyter and run the notebook yourself

Create an isolated notebook copy, set the required startup variables and launch Jupyter:

```powershell
$projectPath = (Get-Location).Path
$runPath = Join-Path $projectPath 'artifacts\notebook_runs\jupyter_001'
if (Test-Path -LiteralPath $runPath) { throw 'Choose a new run directory.' }
New-Item -ItemType Directory -Path $runPath | Out-Null
Copy-Item -LiteralPath (Join-Path $projectPath 'main.ipynb') -Destination (Join-Path $runPath 'main.ipynb')
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

Jupyter should open in your browser. If it does not, open the local URL printed in PowerShell. Keep that terminal running while you use Jupyter.

1. Open the copied **`main.ipynb`** and select **SC4002 (.venv)** as its kernel.
2. Edit the **Editable run configuration** cell if needed. Use `evaluate_test=False` while tuning, or `True` for final development/test results.
3. **Save**, then choose **Restart Kernel and Run All Cells**. Keep the filename `main.ipynb` and its kernel working directory as this new run folder.
4. Wait for the final completion output, then save the executed notebook. Results are in `artifacts/notebook_runs/jupyter_001/run_artifacts/results.csv` and `results.json`.

For another run, change `jupyter_001` to a new folder name and use a fresh kernel. See the [full interactive guide](docs/RUNNING_BILSTM.md#run-interactively-in-jupyter) for details.

### Option B: run everything from the terminal

No browser or Jupyter window is needed. Save any configuration edits in the source notebook, then run:

```powershell
.\.venv\Scripts\python.exe scripts/execute_notebook.py --execution-dir artifacts/notebook_runs/final_001
```

Use a **new directory name for every run**. The executor starts the `sc4002` kernel with deterministic startup settings, copies the notebook into that directory and runs every cell in order. After success, it copies the genuinely executed notebook back to the source location. A failed run retains its partial notebook and logs for diagnosis.

Wait for `"status": "complete"` in the terminal. Open `artifacts/notebook_runs/final_001/run_artifacts/results.csv` or `results.json` for the results; the executed notebook is retained in that run folder too.

The executor allows 900 seconds per code cell by default. For a larger model or epoch budget, add `--cell-timeout -1` for unlimited cell execution, or provide a larger positive timeout in seconds.

<a id="reproduce-the-notebook"></a>

The full notebook automatically verifies the original data, fits train-only mappings, counts parameters, trains both models for their configured epochs, selects development checkpoints, and independently reloads/verifies them. With final evaluation enabled it freezes each model, evaluates the complete test split once, and saves four comparison rows with genuine error cases.

## 3. Change hyperparameters for a future experiment

Open the notebook's cell tagged **`configuration`**. Edit literal values inside its existing `RUNTIME`, `STAGE1_CONFIG`, `STAGE2_CONFIG` and `RUN_OPTIONS` dictionaries. Save the notebook before execution.

| Setting | Where to edit | Default |
| --- | --- | --- |
| Stage 1 word embedding size | `STAGE1_CONFIG['model']['embedding_dim']` | 256 |
| Stage 1 hidden units per direction | `STAGE1_CONFIG['model']['hidden_dim']` | 512 |
| Stage 1 LSTM layers / dropout | `STAGE1_CONFIG['model']` | 1 / 0.3 |
| Stage 2 model dimensions / additions | `STAGE2_CONFIG['model']` | Character CNN, shared BiLSTM, intent context and CRF |
| Learning rate / weight decay | Each stage's `training` dictionary | 0.001 / 0.0001 |
| Batch size / maximum epochs | Each stage's `training` dictionary | 32 / 20 |
| Gradient clipping | Each stage's `training` dictionary | 5.0 |
| Training RNG seed | `RUNTIME['seed']` | 42 |
| Run final test evaluation | `RUN_OPTIONS['evaluate_test']` | `True` |
| Reuse a verified local English JSONL | `RUN_OPTIONS['dataset_path']` | `None`: download official data |
| Compute device | `RUN_OPTIONS['device']` | `'cuda'` |
| Fresh-process reload timeout | `RUN_OPTIONS['verification_timeout_seconds']` | 300 seconds; `None` means unlimited |

Change dictionary **values**, rather than adding code that mutates settings later. Effective settings and code/configuration identities are recorded in `run_artifacts/run_configuration.json`; parameter counts are recorded in each stage's summaries/checkpoint and the generated results. Protected model/data/evaluator definitions remain checked; ordinary hyperparameter edits do not disable those checks. A configuration exceeding the cap is rejected before training. AdamW and equal task-loss weights of 1.0 remain fixed; the notebook does not implement alternative optimizers or loss weighting.

For tuning, change the existing `evaluate_test` value to **`False`**, run each candidate in a separate fresh directory, and compare development scores. This produces two development rows without test inference. Select settings using the unrounded mean of development intent accuracy and slot F1. Once settings are fixed, change `evaluate_test` to **`True`** and start a fresh complete run to produce four development/test rows. Test scores must not choose hyperparameters or checkpoints.

For example, change Stage 1's existing learning rate value to `0.0005`, batch size to `16`, or maximum epochs to `30`; then save and run from the first cell. Stage 1 changes do not automatically change Stage 2. [Detailed tuning instructions](docs/RUNNING_BILSTM.md#tune-on-development-then-produce-final-results) explain the full sequence.

**Editing JSON under `configs/` does not change an existing notebook.** Edit the notebook cell for routine runs. Intentional implementation/source changes use the separately reviewed [builder workflow](docs/RUNNING_BILSTM.md#notebook-settings-versus-repository-configurations).

## 4. Find and verify the results

For the example run, generated files are under:

```text
artifacts/notebook_runs/final_001/
  main.ipynb
  run_artifacts/
    run_configuration.json
    dataset_identity.json
    results.csv
    results.json
    completion.json
    stage1/   # checkpoint, histories, curves, reload and evaluation evidence
    stage2/   # corresponding Stage 2 evidence
```

With `evaluate_test=True`, `results.csv` contains **four rows**: Stage 1 development/test and Stage 2 development/test. With `False`, it contains **two development rows** and no test result. `results.json` also retains full-precision metrics and raw counts. Each row reports actual total/trainable parameters, epochs trained and selected checkpoint epoch. Read generated results; do not type scores manually.

The notebook saves learning curves, checkpoint/reload evidence and genuine Stage 2 development error cases in both modes. Final runs additionally save freeze/test records. Keep the entire run directory with the executed notebook; copying only the CSV loses provenance. See the [output-file guide](docs/RUNNING_BILSTM.md#output-files-and-what-they-mean).

Validate a completed run against its own evidence directory:

```powershell
.\.venv\Scripts\python.exe scripts/validate_notebook.py --notebook artifacts/notebook_runs/final_001/main.ipynb --execution-dir artifacts/notebook_runs/final_001/run_artifacts --output artifacts/notebook_runs/final_001/acceptance.json
```

The validator checks retained outputs and generated evidence for the declared run mode; it does not train or run another trained-model test inference. Use a new validation output filename if one already exists. Development-only acceptance is not final-test submission acceptance.

## Shared evaluator and verified data

Both notebooks inline the same [comparison_contract.py](src/comparison_contract.py). Slot micro F1 requires exact slot type, original character start, exclusive end and literal text, pooled across records. Repeated text at distinct positions counts separately; invalid source slices and missing/duplicate prediction IDs fail. Intent accuracy is exact label agreement.

Both load the same pinned original English JSONL: **11,514 train / 2,033 development / 2,974 test**. They verify source SHA-256, ordered split IDs and full split contents. Train alone fits vocabularies and the common 111-tag BIO inventory. The Transformer no longer accepts arbitrary CSV input. See the [evaluation rationale and contract](docs/EVALUATION_COMPARISON.md).

For a contract/data check without model training:

```powershell
.\.venv\Scripts\python.exe scripts/check_comparison_contract.py
```

This reads an already downloaded verified local dataset and writes [results/comparison_contract.json](results/comparison_contract.json). It checks source/data identity and scoring fixtures, **not learned-model performance**. Use `--dataset` to check another location containing the exact original JSONL. A notebook run can download official data itself or reuse the pinned JSONL specified by `RUN_OPTIONS['dataset_path']`.

## Models and historical evidence

| Model, default settings | Architecture | Parameters |
| --- | --- | ---: |
| BiLSTM Stage 1 | Random E256 embeddings; shared BiLSTM H512 per direction; intent and BIO heads | 4,646,571 |
| BiLSTM Stage 2 | Character CNN + shared BiLSTM + predicted-intent context + BIO CRF | 4,939,498 |
| Transformer Stage 1 | Word2Vec-initialized embeddings; four-layer encoder; intent and BIO heads | 4,345,243 |
| Transformer Stage 2 | Character CNN + shared encoder; intent and BIO heads | 4,379,947 |

Counts are for defaults; future runs compute their actual totals. The Transformer uses permitted pretrained word vectors, with its encoder trained from scratch. Sharing data/scoring does not isolate architecture from different tokenizers, initialization and optimization. Its supplied sequence defines Stage 2 but does not train it; a BiLSTM run does not execute the Transformer notebook. See [model details](docs/MODEL_AND_EVALUATION.md).

Original notebooks/results are preserved locally under ignored `artifacts/evaluator_v2/prechange/`. Public [v1 results](results/historical_v1/results.json), [CSV](results/historical_v1/results.csv), plots and [experiment/error analysis](docs/EXPERIMENTS_AND_ERRORS.md) are historical. V1 BiLSTM test intent/slot scores were 83.96%/70.51% and 82.62%/76.81%; they are not v2 results. New scoring may select different epochs, so one rescored old checkpoint is insufficient evidence of a new best-development model.

## Tests and project guide

Portable offline checks, after installing requirements:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_acquisition tests.test_runtime tests.test_metrics tests.test_diagnostics tests.test_notebook_build tests.test_comparison_contract tests.test_comparison_notebooks tests.test_notebook_reruns -v
```

The earlier BiLSTM rerun-workflow revision passed **199 full-workspace tests** and **97 portable offline tests**. Its end-to-end check used one epoch of small E16/H16 models, full original train/development splits, verified cached data and development-only mode; independent reload and acceptance passed without test inference.

For the separate revised Transformer, **59 dedicated model/data/training/notebook/auditor tests pass**. The current full-workspace run reports **11 existing failures tied to `main.ipynb`**: its protected source/contract metadata differs from the reviewed contents. Those BiLSTM files and saved results were preserved; resolve that source-provenance mismatch before treating a BiLSTM rerun as accepted. The revised Transformer has its own standalone execution and independent acceptance path.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_revised_transformer_model tests.test_revised_transformer_data tests.test_revised_transformer_training tests.test_revised_transformer_notebook tests.test_revised_transformer_validation -v
```

Full discovery also uses ignored local data and historical evidence. Correctness tests and contract checks do not replace a completed final training/reload/test-evaluation run.

| Path | Purpose |
| --- | --- |
| [docs/RUNNING_BILSTM.md](docs/RUNNING_BILSTM.md) | Complete interactive/automated run instructions, settings and troubleshooting |
| [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) | Current revision and consolidated phase history |
| [docs/ASSIGNMENT.md](docs/ASSIGNMENT.md) | Brief and additional clarifications |
| [src/](src/), [scripts/](scripts/), [configs/](configs/), [tests/](tests/) | Development source, tooling, registered settings and tests |
| `data/`, `artifacts/`, `run_artifacts/` | Ignored datasets, run evidence, checkpoints and local backups |

The original Phases 1–7 completed under evaluator v1. Fresh assessed v2 execution/acceptance remains necessary before report results are current. Phase 8 still includes the two-page report, genuine AI history, verified group details and exact three-file submission ZIP. The brief requires Stage 1 without generative AI; assistance occurred at the user's request and no lecturer-approved exception is claimed. Group ID, full names and confirmed contributions remain pending.

The coursework ZIP contains one final `main.ipynb`, one `ai_collaboration.md` or `.txt`, and `report.pdf`. Preserve genuine notebook outputs. The two development notebooks are not instructions to submit two notebooks.

# SC4002: joint intent classification and slot filling

Two jointly trained NLP systems for the original **MASSIVE 1.0 English** dataset. Each system predicts intent and slot spans from the utterance using one shared model. Both models are randomly initialized, use no pretrained assets, and count every learned component toward the **20 million parameter** limit.

**Start with [main.ipynb](main.ipynb).** This is the single authoritative notebook, fully executed with retained training logs, learning curves, results, and genuine error cases. All required code is inline; it does not import repository modules or load historical checkpoints.

**Status:** Phases 1–7 are complete. Phase 8 remains: the two-page report, consolidated genuine AI collaboration history, verified group details, and exact three-file coursework ZIP. See [project status](docs/PROJECT_STATUS.md) and [assignment requirements](docs/ASSIGNMENT.md).

## Results

The notebook retrained both selected configurations for **20 complete epochs each**, selected checkpoints using development data, saved and independently reloaded its own checkpoints, then froze each system before one complete test pass.

| System | Split | Intent accuracy | Slot micro F1 | Parameters | Selected epoch |
| --- | --- | ---: | ---: | ---: | ---: |
| Stage 1: joint BiLSTM | Development | 84.36% | 73.61% | 4,646,571 | 19 / 20 |
| Stage 1: joint BiLSTM | Test | 83.96% | 70.51% | 4,646,571 | 19 / 20 |
| Stage 2: enhanced joint BiLSTM | Development | 84.65% | 79.88% | 4,939,498 | 11 / 20 |
| Stage 2: enhanced joint BiLSTM | Test | 82.62% | 76.81% | 4,939,498 | 11 / 20 |

On test, Stage 2 improves slot F1 by **6.30 percentage points** while intent accuracy decreases by **1.34 points**. Their mean improves by **2.48 points**. Differences use unrounded values. Selection used development only; test results did not change the settings.

The notebook reproduces the earlier script results exactly. Observed training-workflow times were **127.51 seconds** for Stage 1 and **359.09 seconds** for Stage 2, excluding setup/download, plotting, independent verification, and final evaluation. Timings describe this execution rather than an isolated architecture benchmark. Machine-readable outputs are in [results](results/).

## Models and evaluation

- **Baseline:** random 256-dimensional word embeddings, one bidirectional LSTM with 512 hidden units per direction, dropout 0.3, masked max pooling for intent classification, and a token-level slot head.
- **Stage 2 additions:** train-fitted character CNN features, differentiable context from predicted intent probabilities, and BIO-constrained CRF slot decoding. Gold intents are never predictive inputs.
- **Training:** AdamW, learning rate 0.001, weight decay 0.0001, batch size 32, gradient clipping 5, equally weighted intent and slot losses, seed 42, and deterministic FP32 settings. All parameters are trainable.
- **Data:** preserve all 11,514 train, 2,033 development, and 2,974 test records. Fit vocabulary and character mappings on train only; no augmentation, truncation, or new split.
- **Scoring:** intent accuracy and micro F1 over each record's set of exact `(slot type, literal surface text)` pairs. Duplicate pairs collapse; offsets are retained for audit. This is the disclosed project interpretation; lecturer-scorer parity is unverified.
- **Selection:** maximize the unrounded mean of development intent accuracy and slot F1; exact ties keep the earliest checkpoint. The historical search covered three Stage 1 and five Stage 2 candidates, 160 full epochs. The notebook reproduces the two selected configurations, 40 epochs.

See [models and evaluation](docs/MODEL_AND_EVALUATION.md) for architecture, preprocessing, losses, and scoring; see [experiments and errors](docs/EXPERIMENTS_AND_ERRORS.md) for the registered search, ablations, negative findings, and genuine failures.

## Install the tested environment

The supported reproduction target is **Windows, Python 3.12, and an NVIDIA CUDA GPU**. The completed run used Python 3.12.0, PyTorch 2.7.1+cu126, and an RTX 3060 Ti with 8 GB VRAM. The notebook explicitly requests CUDA. A compatible NVIDIA driver and internet access for the official dataset download are required.

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name sc4002 --display-name 'SC4002 (.venv)'
```

[requirements.txt](requirements.txt) is the single dependency specification for the tested Windows/CUDA environment, including PyTorch and pinned supporting packages. Other platforms or hardware may need compatible package choices and are not verified for identical training.

Check the environment without training or downloading data:

```powershell
.\.venv\Scripts\python.exe scripts/check_environment.py --device cuda
```

## Reproduce the notebook

The completed notebook already contains its genuine outputs. To run it again, use the executor with a **new execution directory** inside the repository:

```powershell
.\.venv\Scripts\python.exe scripts/execute_notebook.py --execution-dir artifacts/reproduction/run_01
```

The executor starts the `sc4002` kernel with deterministic startup settings and copies only the notebook into that directory. It downloads original data, trains both models, writes checkpoints, verifies development predictions in separate processes, and generates results under `run_artifacts/`. After success, it updates the root notebook with the new genuine outputs. Use a new directory name for another run; existing directories are protected from overwrite.

For interactive Jupyter, copy `main.ipynb` into a fresh working directory first. From that directory, launch Jupyter using the repository's environment:

```powershell
$env:PYTHONHASHSEED='42'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
# Replace this with your repository's absolute path.
& 'C:\path\to\SC4002\.venv\Scripts\python.exe' -m jupyterlab --no-browser
```

Choose **SC4002 (.venv)**, start a fresh kernel, keep the filename `main.ipynb`, and run every cell in order. Hash seeding must be configured before the kernel starts. The notebook refuses to overwrite an existing `run_artifacts` directory. Generated data, checkpoints, and logs stay local and are excluded from Git.

## Tests and verification

Run the fixture-based offline checks available from a fresh clone:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_acquisition tests.test_runtime tests.test_metrics tests.test_diagnostics tests.test_notebook_build -v
```

These **55 tests** cover safe acquisition, runtime conventions, hand-counted scoring, development diagnostics, and notebook acceptance guards, without downloading data or training models.

The completed development workspace passed **157 tests**, **192 script/notebook parity checks**, and an independent notebook acceptance audit. Some tests and historical audit scripts require ignored dataset, mapping, checkpoint, and registration artifacts, so full discovery is not a fresh-clone check. Those evidence dependencies are recorded in [project status](docs/PROJECT_STATUS.md); the standalone notebook is the reproducible path from a clean checkout.

## Repository guide

| Path | Purpose |
| --- | --- |
| [main.ipynb](main.ipynb) | Self-contained, executed notebook for both selected systems |
| [requirements.txt](requirements.txt) | Single pinned environment specification |
| [docs/](docs/) | Consolidated progress, assignment, design, methodology, and analysis |
| [results/](results/) | Lightweight results and review evidence from the completed notebook |
| [src/](src/) | Script-first data, scoring, model, training, and checkpoint implementations |
| [scripts/](scripts/) | Setup, acquisition, experiment, audit, and notebook tooling |
| [configs/](configs/) | Preserved protocol and candidate configurations |
| [tests/](tests/) | Unit, integration, and artifact-acceptance checks |
| `data/`, `artifacts/`, `run_artifacts/` | Ignored downloads, research evidence, and execution outputs |

Final configurations are [Stage 1](configs/phase4/stage1_dropout03.json) and [Stage 2](configs/phase5/char_crf_intent_context.json). Earlier defaults and protocol placeholders remain for provenance; the final configurations and notebook settings take precedence. Historical experiment launchers require their original registration/audit evidence; the notebook embeds everything needed for clean reproduction. Raw historical hash audits also expect the original pre-cleanup file layout, preserved locally in the ignored cleanup archive. The notebook builder requires the historical Phase 6 reference and refuses to replace a notebook containing executed outputs.

The [project status](docs/PROJECT_STATUS.md) consolidates all seven phase updates and the implementation plan. [Assignment requirements](docs/ASSIGNMENT.md) consolidate the brief and later clarifications. [Experiments and errors](docs/EXPERIMENTS_AND_ERRORS.md) retain the scientifically useful registration, comparisons, and error analysis; redundant standalone drafts have been removed from the active layout.

## Coursework and limitations

This is SC4002 coursework. The brief requires Stage 1 without generative AI; the user requested assistance despite that rule, and assistance occurred. **No lecturer-approved exception is claimed.** Group ID, full names, and confirmed individual contributions remain pending at the user's request. The Phase 8 collaboration record must describe actual assistance and verified human work.

Results use one seed and do not establish statistical significance, a global optimum, or cross-hardware bitwise reproducibility. Remaining errors include intent ambiguity, artist/song boundaries, and light-dimming polarity; see [experiments and errors](docs/EXPERIMENTS_AND_ERRORS.md).

The coursework submission will contain exactly `main.ipynb`, `ai_collaboration.md` or `.txt`, and `report.pdf`. This GitHub repository is the development project; the final report and submission archive remain Phase 8 work.

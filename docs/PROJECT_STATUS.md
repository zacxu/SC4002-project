# Project status and implementation history

**BiLSTM rerun workflow, 31 October 2026:** [main.ipynb](../main.ipynb) now supports
one editable configuration cell for future hyperparameter runs. See
[RUNNING_BILSTM.md](RUNNING_BILSTM.md) for complete Jupyter/CLI instructions and
output locations. Development-only mode trains/selects/reloads without test;
final mode additionally freezes and tests. A small-model, one-epoch execution
over the full original train/dev records passed independent acceptance with
zero test model passes; this verifies the workflow, not the default models'
final performance. Default evaluator-v2 comparison scores remain pending.

Updated **31 October 2026 (Singapore time)**. **Current evaluator-v2 execution and comparison are pending.** Completed historical Phases 1-7 below describe the earlier evaluator-v1 project; their old results and acceptance records do not validate the revised notebooks.

## Current comparison revision

- The active [BiLSTM notebook](../main.ipynb) and [Transformer notebook](../NLP_Final_project.ipynb) are intentionally different implementations, with the same inline [comparison contract](../src/comparison_contract.py).
- Both use the pinned official MASSIVE 1.0 English JSONL, all original records/partitions, and manifests identifying ordered split IDs and complete split contents. Transformer CSV upload has been removed.
- **`exact_character_span_v2`** scores exact typed original Unicode character occurrences, including literal text validation. Distinct occurrences are not collapsed by equal surface text. The previous type/text-set results are historical v1.
- Transformer task labels now derive from train only and contain the complete 111 BIO tags. Revised whole-model counts are **4,345,243 / 4,379,947** for its two stages. BiLSTM architectures/counts remain **4,646,571 / 4,939,498**.
- The rebuilt active BiLSTM notebook has **40 cells / 27 code cells** and cleared outputs; this is a source revision, not a completed training run.
- One dependency file now retains the original 105 pins and adds 10 compatible Transformer support packages, **115 pinned distributions**. Installed-package consistency passed; no fresh empty-environment installation is claimed.
- The original executed notebooks and associated files were preserved with checksums in the ignored `artifacts/evaluator_v2/prechange/` archive. Active notebook outputs are cleared because the metric changes checkpoint selection. No new full training run or final model test evaluation is claimed for this code-contract revision.

### Current verification, 31 October

- **185 full-workspace tests passed** in 6.897 seconds; **83 portable offline tests passed** in 2.388 seconds.
- All **49 code cells compile**: 27 in the BiLSTM notebook and 22 in the Transformer. Both notebooks have zero stale output cells and byte-identical inline contract source.
- The [public comparison audit](../results/comparison_contract.json) confirms the same pinned data/split identities and exact original gold-span round trips for **all 16,521 records in both adapters**. Transformer development-collation/evaluator fixtures cover all 2,033 records with and without character features, without trained-model inference.
- Dependency consistency passed. Runtime pins agree apart from installed pip **26.2.2** versus the specification's historical **25.2** installer pin; no fresh empty-environment installation is claimed.

These checks validate code, data and bookkeeping. They do not replace complete training, development checkpoint selection or final model evaluation under v2. The public historical performance files are isolated under [results/historical_v1/](../results/historical_v1/).

The two implementations still differ in tokenizer, word-vector initialization and optimization/budgets. A shared dataset/scorer supports comparison but does not isolate a causal architecture effect. See [EVALUATION_COMPARISON.md](EVALUATION_COMPARISON.md) for precise semantics, upstream context and rerun requirements.

Before Phase 8 can use new scores, execute fresh complete training/development selection, save/reload verification, freezing and final evaluation for each family/stage. Retain the new manifests, evaluator identity, genuine logs, results and errors. Do not reuse historical selected epochs, v1 tables or previously accepted outputs as new v2 evidence. The deadline/group/AI-history requirements in [ASSIGNMENT.md](ASSIGNMENT.md) remain unchanged.

## Historical position at 24 October 2026

The following historical summary and phase narratives are preserved as the record of the accepted evaluator-v1 implementation. References to a completed/executed notebook, selected checkpoints and metrics are scoped to that archived version.

Updated **24 October 2026 (Singapore time)**. **Phases 1–7 are complete; Phase 8 is pending.** This is the single progress record, consolidating all seven original phase updates and the implementation plan. The executable deliverable is [main.ipynb](../main.ipynb); the [README](../README.md) gives the current installation and reproduction instructions.

### Historical phase position

| Phase | Outcome | State |
| --- | --- | --- |
| 1 | Local Python/CUDA/Jupyter setup and experiment protocol | Complete |
| 2 | Official original English data, preprocessing and shared evaluation | Complete |
| 3 | Joint E256/H512-per-direction BiLSTM and checked training pipeline | Complete |
| 4 | Three baseline runs, development selection, freeze and final evaluation | Complete |
| 5 | Five improved candidates, ablations, real error analysis and frozen final system | Complete |
| 6 | Reconciliation, reproducibility and final implementation audit | Complete |
| 7 | Self-contained notebook genuinely retrained both selected systems; acceptance and parity passed | Complete |
| 8 | Two-page report, genuine AI collaboration export, verified group cover, exact submission ZIP | Pending |

The completed notebook has **39 cells, 26 executed code cells, no execution errors, 40 genuine training epochs, two plots and four real development error cases**. The historical acceptance passed **192 parity checks** and **157 tests**. Test intent/slot scores are **83.96% / 70.51%** for Stage 1 and **82.62% / 76.81%** for Stage 2; the intent regression remains disclosed. See [historical results](../results/historical_v1/) and [experiments and errors](EXPERIMENTS_AND_ERRORS.md).

The final runtime target is **local Jupyter on the RTX 3060 Ti (8 GB)**. Group ID, full names and verified individual contributions remain pending at the user's request. The brief's original no-AI Stage 1 requirement and actual assistance are disclosed in [assignment requirements](ASSIGNMENT.md); no lecturer exemption has been provided.

## Historical repository cleanup, 24 October

The two former root notebooks, `main.ipynb` and `main_revised.ipynb`, were byte-for-byte identical, so the redundant revised copy was removed at that cleanup. This does not refer to the subsequently added, distinct `NLP_Final_project.ipynb` Transformer comparison. The accepted evaluator-v1 `main.ipynb` SHA-256 was `95e4b943be08fae9ed9959d8f49e120c29e397121be9d5b9a37c0954823ca7b3`.

- All phase status updates and the original implementation plan are consolidated here.
- The brief and later model rules are consolidated in [ASSIGNMENT.md](ASSIGNMENT.md).
- Architectures, preprocessing, scoring and training conventions are consolidated in [MODEL_AND_EVALUATION.md](MODEL_AND_EVALUATION.md).
- The useful experiment registrations, ablations, diagnosis and real failures are consolidated in [EXPERIMENTS_AND_ERRORS.md](EXPERIMENTS_AND_ERRORS.md). Separate root drafts are redundant.
- One [requirements.txt](../requirements.txt) preserves all 105 tested dependency versions, including the recorded CUDA PyTorch wheel. The setup script uses this single specification.
- The original brief, notebook-generated tables and genuine plots are retained under `docs/` and `results/`. Model weights, downloaded data, environments and bulk evidence remain local and ignored by Git.

Exact pre-cleanup documents are preserved under the ignored local `artifacts/cleanup/originals/`, with `artifacts/cleanup/original_files.json` recording their checksums. Historical phase snapshots, registrations, checkpoints and notebook backups remain untouched. **Old raw-hash audit commands expect the pre-cleanup layout.** Their records describe that layout; running them against the consolidated layout requires restoring the exact historical files/source version first. The notebook runs independently of those archives. The current README distinguishes portable fixture tests from archive-dependent audits.

## Historical Phase 8 plan before the evaluator-v2 revision

1. Obtain the real group ID, member names and confirmed contributions; do not invent human work.
2. Consolidate genuine AI chat/context to one `ai_collaboration.md` or `.txt`, including actual Stage 1 assistance, accepted/modified/rejected suggestions and verified human reasoning. Retain the original conversation; the brief does not resolve whether excerpts suffice. Phase working summaries alone do not demonstrate the actual exchange.
3. Write `report.pdf`: at most two content pages excluding the cover, both pipelines, four comparison rows, total/trainable parameters, total/selected epochs, efficiency, critical assessment, limitations and at least two genuine final-system errors.
4. Use the notebook's generated full-precision results and retained plots; preserve the test intent regression, scorer interpretation and single-seed limits.
5. Verify the final notebook still retains genuine outputs and all three deliverables contain no credentials. Build `Group_<ID>.zip` with exactly `main.ipynb`, the collaboration file and `report.pdf`.
6. Check current official submission announcements, then submit through NTULearn. The brief's interpreted deadline is **15 November 2026, 23:59 SGT**; see [administrative details](ASSIGNMENT.md#administration-and-submission).

## How to read the original phase records

The sections below preserve the full original phase narratives and original implementation plan, including early model dimensions, then-pending tasks, original filenames, commands and old test counts. **They are dated history, not the current setup guide.** Later completed phases supersede earlier pending statements. For example, Stage 1's original dropout 0.2 default was followed by development selection of dropout 0.3; the 128/128 diagnostic model was replaced before full runs by the requested 256/512 model.

Links now point to the consolidated public documentation and retained source. Paths labelled “local archive” identify intentionally untracked evidence rather than downloadable GitHub files. Historical commands may refer to the pre-cleanup filenames and require the archived layout. Original file bytes remain in the local archive; navigation links and heading levels below have been adjusted for this consolidated document.

## Phase 1: historical completion record

Completed on **20 October 2026 (Singapore time)**. Scope: [Phase 1 of the implementation plan](#historical-implementation-plan).

**The technical setup and its completion gate passed.** An isolated environment, script structure, portable settings, experiment protocol, setup verifier, and contribution/collaboration records are in place. The user subsequently confirmed local Jupyter using the RTX 3060 Ti as the final notebook target. Group identities and responsibilities remain pending at the user's request.

### Verified environment

| Item | Observed value |
| --- | --- |
| Operating system | Windows 11, build 26200 |
| Python | 3.12.0, isolated in `.venv` |
| Global/user package isolation | Separate virtual environment; user site packages disabled |
| CPU | Intel64 Family 6 Model 151; 12 logical processors |
| GPU | NVIDIA GeForce RTX 3060 Ti, approximately 8 GB VRAM, compute capability 8.6 |
| NVIDIA driver | 610.88 |
| PyTorch | 2.7.1+cu126 |
| PyTorch CUDA runtime | 12.6; actual CUDA computation passed |
| NumPy / Matplotlib | 1.26.4 / 3.10.8 |
| JupyterLab / ipykernel | 4.4.10 / 6.30.1 |
| nbformat / nbconvert / nbclient | 5.10.4 / 7.16.6 / 0.11.0 |
| pip | 25.2 |
| Dependency snapshot | 105 pinned installed distributions in `requirements.lock.txt` |
| Installed project kernel | `sc4002`, display name **SC4002 (.venv)**; registered inside `.venv` |

The package build was selected from the [official PyTorch CUDA 12.6 installation options](https://pytorch.org/get-started/previous-versions/#v271). It is a reproducible project pin, not a claim that it is the latest release.

Detailed machine-specific evidence is saved in `artifacts/setup/environment.json` (local archive). Source configuration contains relative paths rather than this machine's absolute paths.

### Completed Phase 1 work

| Planned task | Completion evidence |
| --- | --- |
| Inspect Python, device, and GPU memory | Hardware and package snapshot; actual CPU/CUDA tensor and gradient checks. |
| Create an isolated environment | `.venv`, pinned direct requirements, complete dependency lock, successful dependency check. |
| Create the development structure | `src/`, `scripts/`, `configs/`, `tests/`, `data/`, and `artifacts/`. Data/model/metric/training responsibilities have documented stubs for later phases. |
| Make configuration portable | Source-relative default root, optional explicit root, relative configured data/output paths, rejection of paths escaping the root. |
| Establish the experiment protocol | [configs/protocol.json](../configs/protocol.json), [EXPERIMENT_PROTOCOL.md](MODEL_AND_EVALUATION.md#training-and-selection), and verified runtime helpers. |
| Start contribution and collaboration records | Historical setup placeholders `CONTRIBUTIONS.md` and `ai_collaboration.md`, with genuine exchanges, labelled summaries, and unknown human details left pending. See the Phase 6 navigation note below for the current records. |
| Document setup and reproduction | [README.md](../README.md) and [scripts/setup_environment.ps1](../scripts/setup_environment.ps1). The individual installation/check steps were executed; the convenience wrapper was syntax-checked. |

The first Torch installation attempt stopped because its report-output directory did not exist. The directory was created, the cached download was reused, and installation completed successfully. Both successful installations have pip JSON records in `artifacts/setup/`.

### Check results

| Check | Result | Saved evidence |
| --- | --- | --- |
| Required package imports | Passed | `environment.json` |
| Isolation from global/user packages | Passed | `environment.json` |
| CPU calculation and exact expected gradient | Passed; loss 52 and gradient `[[8, 12], [8, 12]]` | `environment.json` |
| CUDA calculation and exact expected gradient | Passed with the same expected values | `environment.json` |
| Strict deterministic settings and seed policy | Passed | `environment.json` and runtime tests |
| Isolated notebook kernel execution | Passed; verified interpreter prefix, configured environment, imports, and tensor computation | `notebook_smoke.ipynb` |
| Noninteractive Matplotlib output | Passed | `smoke_plot.png` |
| Dependency consistency | Passed: “No broken requirements found.” | `dependency_check.txt` |
| Focused runtime tests | **9 passed** | `unit_tests.txt` |
| Python source compilation | Passed | Executed setup check |
| PowerShell setup wrapper syntax | Passed | Executed parser check |

The nine tests cover portable paths/original split settings, source-root independence from working directory, mean-score selection/exact ties, invalid scores, configuration hashing, distinct run directories, collision protection, reproducible random streams/deterministic flags, and automatic/explicit device behavior.

Run the checks using the commands in the README. The notebook smoke artifact is setup evidence, not the final `main.ipynb`. No persistent Jupyter server was started.

### Established experiment choices

- Seed **42** for the initial runs in both stages; seeded Python, NumPy, Torch, CUDA, and a dedicated DataLoader generator.
- Initial DataLoader workers **0**; initial precision **FP32**.
- Strict deterministic algorithms; cuDNN deterministic enabled, benchmarking and TF32 disabled; launch hash seed and cuBLAS workspace settings documented.
- Checkpoint score: unrounded `(dev intent accuracy + dev slot micro F1) / 2`, with both metrics stored as fractions.
- Save only on strict improvement; exact ties keep the earlier one-based epoch. Reject invalid/nonfinite metrics.
- Reload verification: identical ordered dev predictions and score difference at most `1e-12`; this tolerance is not used to select checkpoints.
- Relative dataset path `data/1.0/data/en-US.jsonl`; future runs under `artifacts/runs/<stage>_<UTC timestamp>_<config hash>/`, created without overwriting existing records.
- Monotonic timing with CUDA synchronization, separate epoch train/dev timings, and explicit total-training timing boundaries.
- Original splits, train-only fitting, dev-based diagnosis/selection, and frozen-system test evaluation preserved in the protocol.

### Pending information and next phase

The user has confirmed **local Jupyter using the RTX 3060 Ti**, with the verified `.venv` kernel, as the final notebook target. This matches the environment already checked during Phase 1.

Group ID, full names, and individual responsibilities remain pending at the user's request; available compute time has not been supplied. The contribution template records that status explicitly. Practical training duration and an experiment budget require the later data/model implementation; this phase has not measured model-training throughput.

**Phase 2 is ready to begin:** download and audit the original MASSIVE English data, build preprocessing, and verify the shared evaluator. No dataset was downloaded, no task model was trained, no task-performance score was produced, and no experiment run directory or final submission notebook was created during Phase 1.

**Navigation update, 22 October 2026:** this is a historical Phase 1 record. The two original root-level placeholder files are no longer present in the workspace. Current genuine assistance records are retained for `artifacts/phase4/collaboration_record.md` (local archive), `artifacts/phase5/collaboration_record.md` (local archive) and `artifacts/phase6/collaboration_record.md` (local archive). Group details and human contributions remain pending at the user's request; the consolidated final chat file belongs to Phase 8. Current progress is in [PHASE6_STATUS.md](#phase-6-historical-completion-record).

## Phase 2: historical completion record

Completed on **20 October 2026 (Singapore time)** against [Phase 2 of the implementation plan](#historical-implementation-plan). All Phase 2 gates passed. Model implementation and training remain in Phase 3 onward.

### Verified source and population

The original archive was downloaded from the assignment's [official MASSIVE 1.0 URL](https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz). Only its exact English regular-file member was published to `data/1.0/data/en-US.jsonl`. Original text, annotations, record IDs, and partition membership were retained.

| Source | Size in bytes | Observed SHA-256 |
| --- | ---: | --- |
| Original archive | 39,500,415 | `7df623fd2d300a4d235d6ee5bd396c9a28258d3a0ccb29abdb054506eba153f8` |
| English JSONL | 3,904,197 | `c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e` |

Hashes were checked again by the audit. They identify locally observed source bytes; they are not a comparison against an independently published official checksum. HTTP provenance is retained in `data/source_manifest.json` (local archive).

| Original partition | Records | Full annotation/BIO round trips | Full batch-loader verification |
| --- | ---: | --- | --- |
| `train` | 11,514 | Passed | 360 batches |
| `dev` | 2,033 | Passed | 64 batches |
| `test` | 2,974 | Passed, structural checks only | 93 batches, structural checks only |
| Total | **16,521** | **All records passed** | **517 batches** |

No record was added, dropped, deduplicated, moved, or truncated. Batch size 32 was used to audit collation, not to select a future training hyperparameter. Evaluation loaders preserved source order; the seeded training loader shuffled while retaining every record exactly once.

### Implemented deliverables

| Component | Implementation |
| --- | --- |
| Download and provenance | [src/acquisition.py](../src/acquisition.py), [scripts/download_data.py](../scripts/download_data.py). Streamed download, controlled English extraction, hash manifests, immutable conflicting-file handling. |
| Prediction schema | [src/schema.py](../src/schema.py). Typed intent/slot predictions, exact surface text, character offsets, and JSON conversion. |
| Data preparation | [src/data.py](../src/data.py). Original-split loading, exact annotation parsing, input-only tokenization, train-only mappings, BIO targets/decoding, dynamic padding, and prediction JSONL I/O. |
| Shared evaluation | [src/metrics.py](../src/metrics.py). ID-associated intent accuracy and pooled exact slot-pair micro F1, full-precision fractions and counts. |
| Complete audit | [src/audit.py](../src/audit.py), [scripts/audit_data.py](../scripts/audit_data.py). Source/configuration checks, all-record structural verification, all-batch verification, train-only summaries, and saved evidence. |
| Frozen conventions | [configs/protocol.json](../configs/protocol.json), [DATA_AND_EVALUATION.md](MODEL_AND_EVALUATION.md#data-and-scoring). Explicit preprocessing/scoring version 1, including the unresolved-in-brief details. |

### Train-only fitted mappings and summaries

- **5,146 vocabulary entries**, including `<PAD>=0` and `<UNK>=1`; no vocabulary fitting on dev or test.
- **60 intent labels**, **55 slot types**, and **111 BIO tag classes**, all covered by train.
- Tokenization is case-preserving and uses original offsets. No whitespace/Unicode normalization or maximum-length truncation is applied.
- Dynamic right padding uses a boolean mask and slot-label ignore index **-100**. Real `O` tags remain a scored class rather than a padding target.
- Serialized mappings passed a full JSON round trip and retain training-ID/content fingerprints.

Training summaries only:

| Training property | Observed value |
| --- | ---: |
| Token length, minimum / median / 95th percentile / maximum | 1 / 7 / 14 / 35 |
| Character length, minimum / median / 95th percentile / maximum | 2 / 32 / 65 / 189 |
| Examples without slots | 3,755 |
| Examples containing repeated identical type/text slot pairs | 4 |

The complete training intent/slot occurrence counts are in the JSON audit. Test label frequencies, length distributions, OOV rates, error diagnostics, and model scores were not produced. Test targets were used only to check original record preservation, exact alignment, mapping compatibility, BIO reconstruction, and collation.

### Scoring interpretation and checks

The PDF does not resolve every matching detail. **Project convention v1** compares per-record sets of exact `(slot type, surface text)` pairs. Duplicate identical pairs count once within an utterance, offsets are audit metadata, and strings are compared without normalization. Therefore, repeated identical surface text at different occurrence positions is indistinguishable under this convention. This is explicitly an interpretation of the assignment's pair/set wording, not a claim of official-scorer parity.

Intent and slot metrics remain fractions internally, with percentages displayed to two decimal places. An evaluated collection with no gold or predicted slots has slot F1 0.0; an empty dataset or mismatched/duplicate record IDs raises an error. The evaluator uses pooled TP/FP/FN counts rather than token accuracy or averaged per-example F1.

**All 55 focused tests passed:**

- 11 acquisition tests, including hostile archive paths/links, duplicate members, truncated downloads, provenance, and modified-file protection.
- 20 data tests using original training records, covering exact offsets, repeated/adjacent/no-slot cases, train-only fitting, unknown words, mappings, BIO reconstruction, longest-input preservation, padding, and prediction I/O.
- 15 evaluator tests covering hand-counted TP/FP/FN, wrong types/boundaries, missing/extra slots, literal string comparison, duplicate semantics, record association, and invalid inputs.
- 9 existing runtime tests, including seeds, device behavior, portable paths, and checkpoint-selection conventions.

Gold-identity checks on train/dev passed as integrity checks. These compare reconstructed gold labels to original gold labels; they are not measured model performance. No test metric was computed.

### Saved evidence and reproduction

| Artifact | Contents |
| --- | --- |
| `artifacts/phase2/data_audit.json` (local archive) | Passing audit, hashes, original counts, all-record/all-batch checks, fitting scope, training-only summaries, and source-code fingerprints. |
| `artifacts/phase2/preprocessing.json` (local archive) | Train-fitted vocabularies/label mappings, tokenizer conventions, and fitting fingerprints. |
| `artifacts/phase2/split_membership.json` (local archive) | Every original ID in its original partition/source order. |
| `artifacts/phase2/unit_tests.txt` (local archive) | Actual output from the complete 55-test run. |
| `data/source_manifest.json` (local archive) | Download URL, observed HTTP headers, source sizes, and SHA-256 hashes. |

```powershell
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe scripts/audit_data.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The download command reuses and verifies existing source files rather than downloading again. No new dependencies were needed. These remain development files; the eventual ZIP still contains only the three required submission files.

**Next phase:** implement and verify the Stage 1 model/training pipeline using these mappings and the shared evaluator. Architecture, training budget, and group details remain pending; the local Jupyter/GPU target is already confirmed.

## Phase 3: historical completion record

Completed on **21 October 2026 (Singapore time)** against [Phase 3 of the implementation plan](#historical-implementation-plan). All Phase 3 gates passed on the local RTX 3060 Ti. The baseline and shared training pipeline are implemented in Python scripts. Full baseline experiments, selection across full runs, and final test evaluation remain Phase 4.

**Model revision, 21 October 2026:** the user selected embedding dimension **256** and hidden size **512 per direction** for Stage 1. Model defaults and `configs/stage1.json` now agree. The evidence below is from a fresh verification run of this larger model. Earlier 128/128-model diagnostics remain preserved as historical records.

### Implemented baseline

The [baseline design](MODEL_AND_EVALUATION.md#stage-1) defines a joint intent/slot model using the unchanged Phase 2 mappings and evaluator. Only token IDs and input-derived lengths/masks enter the model; record IDs, annotations, and other metadata are not features.

| Component | Implemented choice |
| --- | --- |
| Vocabulary | 5,146 entries fitted on the original training split; PAD 0 and UNK 1. |
| Embeddings | Random, trainable, 256 dimensions; no pretrained model or word vectors. |
| Encoder | One packed bidirectional LSTM, 512 hidden units per direction; 1,024-dimensional token outputs. |
| Intent head | Max pooling over real token states, then a linear layer for 60 intents. |
| Slot head | Per-token linear layer for 111 BIO tags representing 55 slot types plus O. |
| Dropout | 0.2 on embeddings and encoder outputs. |
| Parameters | **4,646,571 total and trainable**, including embeddings and both heads; cap asserted at 20,000,000. |
| Decoding | Intent/tag argmax, shared BIO-to-span decoding, exact original surface text and offsets. Invalid I transitions start a new span under the Phase 2 convention. |
| Objective | Intent cross-entropy averaged over utterances + slot cross-entropy averaged over real tokens; equal weights, padded targets ignored at -100. |
| Initial full-run defaults | AdamW, learning rate 0.001, weight decay 0.0001, batch size 32, gradient norm clipping 5.0, 20 epochs. These are Phase 4 starting settings, not tuned best settings. |

The common [protocol](../configs/protocol.json) remains unchanged so the Phase 2 audit retains its configuration identity. Its historical pending architecture/budget entries are superseded for the Stage 1 defaults by [configs/stage1.json](../configs/stage1.json). Stage 2 design, the eventual tuning budget, and group details remain unresolved.

### Implemented pipeline

| Deliverable | Implementation and behavior |
| --- | --- |
| Model and cap checks | [src/models.py](../src/models.py): packed encoding, padding-aware pooling, both heads, configuration round trip, and total/trainable parameter accounting. |
| Losses and optimizer updates | [src/training.py](../src/training.py): explicit task loss denominators, ignored padding, finite gradients, clipping, and train/evaluation modes. Epoch losses aggregate sums and actual example/token counts. |
| Validation and selection | Shared decoder/evaluator; unrounded mean of dev intent accuracy and slot micro F1. Strict improvement saves a checkpoint; ties retain the earlier epoch. |
| Evidence | Per-epoch JSONL/CSV histories, task losses/metrics, CUDA-synchronized timing, learning curves, selected dev predictions, and immutable run directories. |
| Checkpoints | Detached model-state copies, optimizer state, configuration, mappings, selected epoch/score, predictions, source/data hashes, atomic saving, and SHA-256 verification before loading. |
| Fresh-process verification | [scripts/verify_checkpoint.py](../scripts/verify_checkpoint.py): uses the run's frozen protocol, creates a fresh model, and checks ordered predictions and the selected dev score. |
| Execution | [src/experiment.py](../src/experiment.py) and [scripts/run_experiment.py](../scripts/run_experiment.py): separate diagnostic/full-training modes; launcher seeds process state before Torch initialization. No new dependencies. |

Full experiment mode requires complete original train/dev populations and checks actual visited IDs/counts, including custom-sampler cases. Partial loaders are allowed only for explicit diagnostics. Test scoring requires an explicit opt-in and a complete original test loader; it was not invoked in Phase 3.

### Verification results

**All 82 tests passed again after the revision:** 55 existing acquisition/data/metrics/runtime tests, 12 model tests, and 15 training/checkpoint tests. The `artifacts/phase3/unit_tests_e256_h512.txt` (local archive) includes masking gradients, padding invariance, both heads reaching both encoder directions, finite updates, global loss aggregation, deterministic tie handling, full-population guards, checkpoint independence, hash mismatch rejection, and exact reload. The existing default-model assertion now checks 4,646,571 parameters, including the frozen-embedding accounting case.

The final GPU diagnostic ran under Python 3.12.0 and PyTorch 2.7.1+cu126 on the **NVIDIA GeForce RTX 3060 Ti**, using seed 42, FP32, strict deterministic algorithms, and the established CUDA settings.

| Phase 3 gate | Observed evidence | Result |
| --- | --- | --- |
| Correct outputs | A batch of 32 produced intent logits `[32, 60]` and slot logits `[32, 16, 111]`. | Passed |
| Padding behavior | Four additional padding positions produced maximum absolute difference 0.0 for intent and real-token slot logits. Unit tests also verified zero padded-slot loss gradients and stable PAD embeddings. | Passed |
| Finite updates and logging | Two bounded epochs, each with 96 train records and 64 dev records; both task losses, counts, timing, metrics, curves, and selected checkpoint saved. | Passed |
| Ability to learn | On one fixed batch of 32 original training examples, 100 diagnostic updates reduced both task losses as shown below. | Passed |
| Checkpoint round trip | Selected diagnostic epoch 1, loaded in a fresh Python process; all 64 ordered dev predictions identical and absolute selection-score difference **0.0**. | Passed |
| Provenance and data boundary | Source/configuration/mapping/data identities verified. **Zero test model evaluations.** | Passed |

| Fixed-batch learning probe | Before updates | After 100 updates |
| --- | ---: | ---: |
| Intent loss | 4.076278687 | 0.000016674 |
| Slot loss | 4.710780932 | 0.000005512 |
| Joint loss | 8.787059618 | 0.000022186 |

The fixed-batch probe uses diagnostic learning rate 0.01. It tests that the implementation can fit real training examples; it does not measure generalization or select a full-run learning rate. Short diagnostic dev scores are stored for checkpoint verification, not as assessed baseline performance. No full train/dev experiment or test scoring has been performed.

### Saved evidence and reproduction

The current diagnostic directory is `artifacts/phase3/stage1_20261020T184924218550Z_47eb589e` (local archive). Its UTC timestamp falls on 21 October in Singapore. Earlier successful diagnostics of the smaller model are preserved separately. Saved source hashes identify the implementation used for each run; reproducing an old run through the guarded verification CLI requires its original source version. The current run verifies the larger model under the present source and configuration.

| Artifact in the final diagnostic directory | Contents |
| --- | --- |
| `artifacts/phase3/stage1_20261020T184924218550Z_47eb589e/phase3_verification.json` (local archive) | Passing gates, tensor shapes, learning probe, reload result, hashes, and explicit diagnostic scope. |
| `artifacts/phase3/stage1_20261020T184924218550Z_47eb589e/run.json` (local archive) | Effective settings, runtime/device, source hashes, data identity, and non-reportable status. |
| `protocol.json`, `stage_config.json` | Frozen run configurations. |
| `history.jsonl`, `history.csv`, `learning_curves.png` | Actual bounded-epoch outputs and curves. |
| `best.pt`, `best.pt.sha256`, `best_dev_predictions.jsonl` | Selected diagnostic model, checksum, and ordered predictions. |
| `reload_check.json`, `fresh_reload_process.txt` | Actual independent-process reload comparison. |
| `learning_probe.json`, `training_summary.json` | Fixed-training-batch learning trajectory and bounded-run summary. |

Run a new diagnostic from the project root:

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py --mode diagnostic --device cuda
```

Verify the existing checkpoint using its saved protocol:

```powershell
.\.venv\Scripts\python.exe scripts/verify_checkpoint.py --device cuda --checkpoint artifacts/phase3/stage1_20261020T184924218550Z_47eb589e/best.pt --protocol artifacts/phase3/stage1_20261020T184924218550Z_47eb589e/protocol.json
```

Run the correctness suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

These are development artifacts for later notebook transfer. The final submission still requires only the three specified files. Optimizer state is retained for inspection, but exact interrupted-training resumption is not implemented because RNG and loader states are not fully restored. Identical results across different devices/package versions are not promised.

**Next phase:** run full Stage 1 experiments on the original 11,514 train / 2,033 dev records, assess learning and practical runtime, select/freeze a baseline from dev evidence, verify its reload, and perform its required final test evaluation. Notebook conversion remains Phase 7.

## Phase 4: historical completion record

Completed on **21 October 2026 (Singapore time)** against [Phase 4 of the implementation plan](#historical-implementation-plan). All Phase 4 gates passed. The Stage 1 baseline has been trained, selected using development data, frozen, independently reloaded, and evaluated on the original complete development and test splits. Stage 1 tuning is closed. Stage 2 and notebook conversion remain later phases.

### Selected baseline and final results

The selected model remains one joint BiLSTM for both tasks: trainable embeddings of dimension **256**, one bidirectional layer with **512 hidden units per direction**, masked max pooling for intent classification, and a token BIO slot head. The complete model has **4,646,571 total/trainable parameters**, including embeddings and both heads. It uses no pretrained model or word vectors.

Development selection chose **candidate 3**, with **dropout 0.3**, learning rate **0.001**, batch size **32**, AdamW weight decay **0.0001**, clipping at **5.0**, and equal task loss weights. It trained **20 complete epochs** and selected **epoch 19**. The authoritative selected configuration is [configs/phase4/stage1_dropout03.json](../configs/phase4/stage1_dropout03.json); `configs/stage1.json` remains the original dropout-0.2 starting configuration.

| Evaluation split | Original records | Intent accuracy | Slot micro F1 | Total/trainable parameters | Selected / trained epochs |
| --- | ---: | ---: | ---: | ---: | --- |
| Development | 2,033 | **84.36%** | **73.61%** | 4,646,571 / 4,646,571 | 19 / 20 |
| Test, final reporting only | 2,974 | **83.96%** | **70.51%** | 4,646,571 / 4,646,571 | 19 / 20 |

These are genuine script-run results generated from the frozen checkpoint, not the earlier bounded diagnostics. Fractional metrics and raw counts are preserved at full precision; displayed percentages use two decimal places. The `artifacts/phase4/final_baseline/baseline_results.csv` (local archive) and `artifacts/phase4/final_baseline/final_summary.json` (local archive) are the numerical source.

### Registered comparison and selection

[PHASE4_EXPERIMENT_PLAN.md](EXPERIMENTS_AND_ERRORS.md#registered-research) and its `artifacts/phase4/experiment_plan.json` (local archive) were saved before the first full run. The fixed budget was three fresh-initialization runs of 20 epochs each. All used the same seed 42, strict deterministic FP32 settings, model dimensions, train-fitted mappings, full original train/dev splits, joint loss, decoder, evaluator, and unrounded mean-of-dev-metrics selection policy. Only learning rate or dropout differed from the starting configuration.

| Candidate | Learning rate | Dropout | Selected epoch / trained | Dev intent | Dev slot F1 | Dev selection mean | Training workflow seconds |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1: starting settings | 0.001 | 0.2 | 17 / 20 | 83.82% | 73.52% | 78.67% | 112.09 |
| 2: smaller learning rate | 0.0005 | 0.2 | 17 / 20 | 83.47% | 73.53% | 78.50% | 113.83 |
| **3: stronger dropout, selected** | **0.001** | **0.3** | **19 / 20** | **84.36%** | **73.61%** | **78.98%** | **106.81** |

All **60 registered epochs** completed. Each epoch processed all **11,514 train** and **2,033 dev** examples, with actual identity/count checks and no truncation, augmentation, changed partitions, or train-plus-dev fitting. Each candidate's selected checkpoint passed a fresh-process full-development reload with identical ordered predictions and score difference **0.0**.

The unrounded winning mean is **0.7898193036058875**. The original candidate mean is 0.7866999755672541; the smaller-learning-rate mean is 0.7850323045791749. Candidate 3 won under the registered balanced metric. The smaller-learning-rate trial did not improve that score, and its evidence remains preserved. Exact within-run ties retain the earlier epoch; exact cross-run ties retain the earlier registered candidate. No tie occurred at final selection.

The winner's development mean exceeds the starting candidate by approximately **0.31 percentage points**. This is a single-seed, three-candidate comparison; it does not establish statistical significance or a globally optimal configuration. No further search was added after seeing test results.

### Checkpoint, freezing, and evaluation integrity

Selected run: `artifacts/runs/stage1_20261021T132439418199Z_447f3fc1` (local archive).

Selected checkpoint: `artifacts/runs/stage1_20261021T132439418199Z_447f3fc1/best.pt` (local archive), SHA-256:

```text
3821f7a5d0ea46b70582edd1964b405299d9a6c145a33d02463c516ca854f398
```

The `artifacts/phase4/selection.json` (local archive) was committed from complete development results before final test evaluation. [src/final_evaluation.py](../src/final_evaluation.py) and [scripts/finalize_baseline.py](../scripts/finalize_baseline.py) then performed finalization in a fresh Python process:

1. Validated complete experiment histories, selected epoch/score, configuration, source/data identities, train-only mappings, parameter cap, and checkpoint checksum.
2. Copied 38 source/evidence files into the `artifacts/phase4/final_baseline/snapshot` (local archive), including all Python source/launchers, selected checkpoint, effective configurations, dependency specifications, mappings, split membership, histories, curves, and dev predictions.
3. Reloaded the selected checkpoint and reproduced **all 2,033 ordered dev predictions exactly**, with absolute selection-score difference **0.0**.
4. Persisted `artifacts/phase4/final_baseline/freeze_manifest.json` (local archive) before the first test forward, closing Stage 1 training and tuning.
5. Evaluated **all 2,974 original test records exactly once in source order** using the frozen model. Original data IDs, counts, source hash, and Phase 2 membership were checked.
6. Verified that model weights and checkpoint bytes were unchanged, then saved actual metrics, predictions, durations, and the generated results table.

The finalizer refuses to overwrite an existing final output directory. The record reports **one complete final test model evaluation**. Test results and predictions are under `artifacts/phase4/final_baseline/reporting_only` (local archive); they are retained for reporting and excluded from Stage 2 diagnosis and selection. Stage 2 work should consume the selected baseline's development predictions and curves.

### Counts, losses, timing, and curves

| Final evaluation evidence | Development | Test, reporting only |
| --- | ---: | ---: |
| Correct intents / records | 1,715 / 2,033 | 2,497 / 2,974 |
| Slot TP / FP / FN | 1,531 / 618 / 480 | 2,026 / 906 / 789 |
| Intent cross-entropy | 0.749672995 | 0.798715995 |
| Real-token slot cross-entropy | 0.418471532 | 0.497196635 |
| Joint loss | 1.168144527 | 1.295912629 |
| Evaluation batches | 64 | 93 |
| Evaluation seconds | 0.72 | 0.60 |

The winning training workflow took **106.8057856 seconds**. All three workflows totalled **332.7307636 seconds**. These timings include epoch training, epoch dev evaluation, and checkpoint/log writes. They exclude environment setup, data preparation, final evaluation, and plotting. CUDA synchronization is applied at timing boundaries. They are local observations under Python 3.12.0, PyTorch 2.7.1+cu126, and the RTX 3060 Ti, rather than cross-hardware benchmarks.

Each run retains JSONL/CSV histories and four-panel task-loss/dev-metric curves. The `artifacts/runs/stage1_20261021T132439418199Z_447f3fc1/learning_curves.png` (local archive) show training losses continuing to fall after dev loss stops improving. The registered task-metric mean, rather than minimum dev cross-entropy, correctly determines epoch 19. This development-only observation can inform later diagnosis; the frozen baseline is not retrained after test evaluation.

The `artifacts/phase4/development_comparison.png` (local archive) shows all registered epoch-level selection scores. Full-precision run comparisons are saved in `artifacts/phase4/candidate_comparison.json` (local archive) and `artifacts/phase4/candidate_comparison.csv` (local archive).

### Verification and saved evidence

**All 93 tests passed** in the final stable suite: the prior 82 tests plus 11 finalization guard tests. The `artifacts/phase4/unit_tests_final.txt` (local archive) covers dev-only selection, diagnostic/incomplete checkpoint rejection, configuration/source/checksum guards, complete original IDs/order, weight fingerprints, and earliest selection on exact ties. The earlier 91-test log is also retained; two finalization tests were added before the final 93-test run.

An `artifacts/phase4/independent_audit.json` (local archive) passed **all 50 checks**. It verified all three registered runs, exact winner/epoch selection, complete original IDs, 38 snapshot checksums, 17 source identities, freeze/test chronology, and the 93-test log. Recomputing the shared metrics from saved dev/test predictions reproduced all recorded integer counts and aggregate results. This audit performed no additional model inference, training update, or test model pass.

| Evidence | Location |
| --- | --- |
| Preregistered budget and candidate configurations | `PHASE4_EXPERIMENT_PLAN.md`, `artifacts/phase4/experiment_plan.json`, `configs/phase4/`. |
| Complete run evidence | All three directories under `artifacts/runs/`; none overwritten or removed. |
| Genuine console outputs | `artifacts/phase4/candidate1_console.txt`, `candidate2_console.txt`, `candidate3_console.txt`, `final_evaluation_console.txt`. |
| Dev-only selection and comparison | `artifacts/phase4/selection.json`, `candidate_comparison.json`, `candidate_comparison.csv`, `development_comparison.png`. |
| Freeze and final evaluations | `artifacts/phase4/final_baseline/freeze_manifest.json`, `final_summary.json`, `dev_evaluation.json`, `dev_predictions.jsonl`, `reporting_only/`. |
| Generated report rows | `artifacts/phase4/final_baseline/baseline_results.json` and `.csv`. |
| Frozen implementation and selected evidence | `artifacts/phase4/final_baseline/snapshot/`. |
| Collaboration working record | `artifacts/phase4/collaboration_record.md` (local archive). |

No previously verified model/data/scoring/training source was changed for these runs. Phase 4 added the separate finalization module/launcher and guard tests. Whole-system counting still includes embeddings and both heads; the architecture remains one shared joint model. No pretrained assets or external task examples were added.

### Reproduction and remaining work

The commands used for the three runs, in registered order, were:

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py --mode train --config configs/stage1.json --device cuda
.\.venv\Scripts\python.exe scripts/run_experiment.py --mode train --config configs/phase4/stage1_lr0005.json --device cuda
.\.venv\Scripts\python.exe scripts/run_experiment.py --mode train --config configs/phase4/stage1_dropout03.json --device cuda
```

These are provenance instructions for reproducing the fixed experiment, not authorization for post-test tuning. Fresh training creates new run directories and does not resume an interrupted run.

Development-only checkpoint verification remains available without evaluating test:

```powershell
.\.venv\Scripts\python.exe scripts/verify_checkpoint.py --checkpoint artifacts/runs/stage1_20261021T132439418199Z_447f3fc1/best.pt --protocol artifacts/runs/stage1_20261021T132439418199Z_447f3fc1/protocol.json --device cuda
```

The finalization command already completed and deliberately refuses to repeat over its existing output:

```powershell
.\.venv\Scripts\python.exe scripts/finalize_baseline.py --selection artifacts/phase4/selection.json --device cuda
```

The slot metric retains the documented project interpretation of exact per-record type/text sets, not asserted parity with an unavailable lecturer reference scorer. The frozen source/configuration defines reproducibility; results across different software/hardware are not promised identical.

These remain **script-run baseline results**. The final self-contained executed notebook is Phase 7, and its actual reproduction outputs must supply the final report. The final ZIP packaging remains Phase 8. **Next: Phase 5**, diagnose the baseline using development evidence and evaluate justified Stage 2 improvements while retaining the clarified single-model and full-system 20M rules.

## Phase 5: historical completion record

Completed **21 October 2026 (Singapore time)**. All Phase 5 gates in [IMPLEMENTATION_PLAN.md](#historical-implementation-plan) passed: development diagnosis, justified changes, five controlled full experiments, development-based checkpoint selection, exact fresh-process reproduction, source/model freezing, and one complete final test evaluation. **Stage 2 training and tuning are closed. Phase 6 is next.**

The selected system improves the mean of the two required metrics on both development and test. **It improves test slot F1 but reduces test intent accuracy.** That tradeoff is part of the result, not omitted from the comparison.

### Selected model and required results

The final system is **one joint model**: 256-dimensional random word embeddings, a train-only character CNN, one shared BiLSTM with 512 hidden units per direction, a max-pooled intent head, soft predicted-intent context for slot emissions, and a BIO-constrained CRF. Both tasks train together through one combined loss and optimizer. Gold intent never enters the forward pass. No pretrained model, pretrained vectors, external examples or augmented examples are used.

The entire system has **4,939,498 total and trainable parameters**, including word/character/intent embeddings, both LSTM directions, both heads, the convolution and all CRF parameters. This is below the **20,000,000** cap and adds 292,927 parameters to the baseline. Exact component equations and architecture are in [STAGE2_DESIGN.md](MODEL_AND_EVALUATION.md#stage-2).

The authoritative configuration is [configs/phase5/char_crf_intent_context.json](../configs/phase5/char_crf_intent_context.json). Development selection chose **candidate 5, epoch 11 of 20**. Dropout is 0.3; batch size 32; AdamW learning rate 0.001; weight decay 0.0001; gradient clipping 5; equal task loss weights; seed 42; deterministic FP32. All five candidates start from random initialization.

| Stage | Split | Original records | Intent accuracy | Slot micro F1 | Total / trainable parameters | Selected / trained epochs |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Stage 1 | Development | 2,033 | 84.36% | 73.61% | 4,646,571 / 4,646,571 | 19 / 20 |
| Stage 1 | Test | 2,974 | 83.96% | 70.51% | 4,646,571 / 4,646,571 | 19 / 20 |
| Stage 2 | Development | 2,033 | **84.65%** | **79.88%** | 4,939,498 / 4,939,498 | 11 / 20 |
| Stage 2 | Test | 2,974 | **82.62%** | **76.81%** | 4,939,498 / 4,939,498 | 11 / 20 |

The `artifacts/phase5/stage_comparison.csv` (local archive) and `artifacts/phase5/stage_comparison.json` (local archive) are the numerical sources. These are actual script results; the final report must use the genuine notebook reproduction in Phase 7.

| Stage 2 minus Stage 1 | Development | Test, reporting only |
| --- | ---: | ---: |
| Intent accuracy, percentage points | +0.30 | **-1.34** |
| Slot F1, percentage points | +6.27 | **+6.30** |
| Mean of the two metrics, percentage points | +3.28 | **+2.48** |

The test intent regression is 40 fewer correct intents, from 2,497 to 2,457. Test slot counts improve from TP/FP/FN 2,026/906/789 to 2,111/571/704. The selected model was committed using development scores before these test results existed. No test result prompted another model choice or update.

### Diagnosis and bounded experiment rationale

[PHASE5_DIAGNOSIS.md](EXPERIMENTS_AND_ERRORS.md#development-diagnosis-and-qualified-findings) and `artifacts/phase5/diagnosis.json` (local archive) analyze all original development examples using saved baseline predictions. Training records supply vocabulary and support statistics. No test predictions or metrics are used for this diagnosis.

- Of 2,033 development utterances, 351 contain a word absent from the training vocabulary. Their baseline intent accuracy / slot F1 are 70.94% / 58.39%, versus 87.16% / 76.99% for the remaining 1,682 utterances.
- Recall for unseen slot type/text pairs is 37.48%, versus 89.72% for pairs seen in training. Recall falls from 84.67% for single-token slots to 58.15% for slots with at least three tokens.
- Among 480 missed gold slot pairs, 183 overlap a same-type prediction with incorrect boundaries; 65 have exact boundaries but the wrong type; 48 have mixed type/boundary overlap; 184 have no overlapping unmatched prediction. These are descriptive categories, not one-to-one causal events.
- Baseline development loss starts rising after epoch 6 while training loss falls. This does not justify increasing hidden capacity alone. Rare intents do not uniformly underperform, so blanket class reweighting lacks clear support.

The [registered plan](EXPERIMENTS_AND_ERRORS.md#registered-research) therefore tests character features, a structured slot objective/decoder, and predicted-intent context separately, then two combinations. Its `artifacts/phase5/experiment_plan.json` (local archive) contains configuration hashes and timestamps preceding all full runs. A documented pre-training correction changed two provisional error-category counts by one record; no candidate or hypothesis changed.

### All five full experiments

All **100 registered epochs** completed. Every epoch visits exactly **11,514 original training records and 2,033 original development records**. There is no repartitioning, filtering, augmentation, truncation, train-plus-dev fitting, early stopping or learning-rate schedule. The word tokenizer, word/BIO mappings, exact text reconstruction and shared evaluator remain unchanged. The 34-entry character vocabulary, including PAD and UNK, is fitted only on original training token strings.

| Candidate | Addition(s) | Parameters | Best / trained epoch | Dev intent | Dev slot F1 | Dev mean | Workflow seconds | Peak allocated CUDA MiB |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Reference | Frozen baseline | 4,646,571 | 19 / 20 | 84.36% | 73.61% | 78.98% | 106.81 | Not recorded in this form |
| 1 | Characters | 4,916,011 | 15 / 20 | 84.65% | 74.70% | 79.68% | 161.91 | 231.48 |
| 2 | CRF | 4,659,114 | 14 / 20 | 84.31% | 77.85% | 81.08% | 350.81 | 232.12 |
| 3 | Intent context | 4,657,515 | 12 / 20 | 84.36% | 74.14% | 79.25% | 155.05 | 216.12 |
| 4 | Characters + CRF | 4,928,554 | 17 / 20 | 84.65% | 78.16% | 81.41% | 312.10 | 246.90 |
| **5, selected** | **Characters + CRF + intent context** | **4,939,498** | **11 / 20** | **84.65%** | **79.88%** | **82.27%** | **323.98** | **248.40** |

Within each run, selection uses the highest unrounded mean of intent accuracy and slot micro F1; an exact tie retains the earlier epoch. Across candidates the same score applies, with the earlier registered candidate on an exact tie. The winner's unrounded score is **0.8226578983518025**, versus baseline **0.7898193036058875**. Every candidate's checkpoint reproduced all 2,033 ordered development predictions exactly in a fresh Python process, with score difference **0.0**.

Full precision and run paths are in `artifacts/phase5/candidate_comparison.json` (local archive), `artifacts/phase5/candidate_comparison.csv` (local archive) and the committed `artifacts/phase5/selection.json` (local archive). Every run retains its actual console output, configuration, protocol, source/data hashes, history CSV/JSONL, selected checkpoint, development predictions and learning curves. Earlier and weaker candidates are preserved.

The five workflows total **1,303.8523531 seconds**, approximately 21.73 minutes. The selected workflow takes **323.9795116 seconds**, approximately 5.40 minutes. Timings include training, epoch development evaluation, checkpoint/log writes and CUDA synchronization; they exclude setup, plotting, subprocess reload checks and final evaluation. These are local workflow observations, with other verification work during development, not controlled component speed benchmarks. The winner's peak PyTorch-allocated CUDA memory is 260,462,080 bytes; this excludes driver/context and other memory outside that measurement. Hardware: RTX 3060 Ti, 8,192 MiB, driver 610.88; PyTorch 2.7.1+cu126, Python 3.12.

### What the evidence supports, and what it does not

The CRF is the strongest isolated addition by development mean. Its selected checkpoint reduces false positives from 618 to 363, but also reduces true positives from 1,531 to 1,513. Its improvement therefore trades some recall for precision. Character features alone improve overall metrics modestly, yet OOV slot F1 barely changes, from 58.39% to 58.63%, and OOV intent accuracy regresses. This only partly supports the initial character hypothesis.

Intent context alone improves overall slot F1 by 0.54 percentage points but reduces OOV slot F1 to 57.03%. In combination with characters and CRF, it raises the development mean from 81.41% to 82.27%, with a 1.72-point slot gain and unchanged aggregate intent accuracy. This justifies retaining it under the registered selection policy; one seed does not establish the mechanism or statistical significance.

For the final selected system, development TP/FP/FN are **1,576/359/435**, versus **1,531/618/480** for baseline. Both aggregate slot precision and recall improve. The `artifacts/phase5/development_comparison.json` (local archive) shows:

| Development finding | Baseline | Selected Stage 2 |
| --- | ---: | ---: |
| OOV utterance slot F1 | 58.39% | 65.81% |
| OOV utterance intent accuracy | 70.94% | 70.94% |
| No-OOV utterance slot F1 | 76.99% | 82.91% |
| Unseen slot type/text pair recall | 37.48% | 45.32% |
| Recall for slots of at least three tokens | 58.15% | 66.13% |
| Same-type boundary false negatives | 183 | 108 |
| Exact-boundary wrong-type false negatives | 65 | 65 |
| Utterances with both tasks completely correct | 1,262 | 1,359 |

Paired examples show 64 corrected and 58 regressed intents; 130 recovered and 85 lost gold slot pairs; and 180 wholly corrected versus 83 newly incorrect utterances. Improvement is not uniform across examples. The intent-context hypothesis does not establish a reduction in exact-boundary type errors: that category remains unchanged.

[PHASE5_ERROR_ANALYSIS.md](EXPERIMENTS_AND_ERRORS.md) examines four genuine remaining development errors—records **107, 67, 109 and 24**—and four wholly corrected cases—**375, 595, 617 and 737**. It includes exact utterances, gold labels/spans, both models' saved predictions, and possible explanations explicitly labelled as inferences. These satisfy the preparation for at least two actual final-system error cases; no examples were invented or added to training.

The `artifacts/phase5/development_candidates.png` (local archive), `artifacts/phase5/development_oov_groups.png` (local archive) and `artifacts/runs/stage2_20261021T135955846459Z_6d0b6772/learning_curves.png` (local archive) were inspected. The selected model still shows rising development loss after early epochs while training loss falls. Selection at epoch 11 follows the declared task metric, not minimum loss. Character features and structured prediction do not eliminate overfitting, unfamiliar-entity errors or intent ambiguity.

### Freezing, one final test pass, and independent verification

Selected run: `artifacts/runs/stage2_20261021T135955846459Z_6d0b6772` (local archive).

Selected checkpoint SHA-256:

```text
b70d51fea2f32fa57377fae425052b698abf044d3869236863a6dd76d6b91a71
```

[scripts/select_stage2.py](../scripts/select_stage2.py) required exactly one complete attempted run for each registered candidate, verified their complete histories/configurations/checkpoints, and committed the development-only winner. The separate [Stage 2 finalizer](../src/stage2_finalization.py) then:

1. Verified all five registrations, source/configuration/data hashes, complete original record identities for every epoch, checkpoint identities and fresh reload evidence.
2. Loaded the selected model in a fresh process, reproduced all 2,033 ordered development predictions exactly and verified the unrounded score with zero difference.
3. Snapshotted **79 source/evidence files**, including **29 Python sources/launchers**, all registered settings and selected model evidence. Persisted a `artifacts/phase5/final_stage2/freeze_manifest.json` (local archive) before the first test forward.
4. Evaluated all **2,974 original test records once**, in source order, with no parameter updates. Model state and checkpoint bytes remained unchanged.
5. Saved complete predictions, metrics, final tables and a `artifacts/phase5/final_stage2/final_summary.json` (local archive). The final output directory is exclusive; the training runner rejects further Stage 2 training after this freeze.

**139 unit tests passed**, including exhaustive small-state CRF mathematics, legal decoding, padding/gradient invariance, baseline equivalence when additions are disabled, train-only character fitting, joint gradients, checkpoint safeguards and finalization guards. The final stable log is `artifacts/phase5/unit_tests_final.txt` (local archive). A pre-training GPU diagnostic also checked padded-input invariance, exact state restoration, and both losses falling on the same 32 original training examples over 100 updates; these are correctness checks, not assessed performance. Its `artifacts/phase5/diagnostics/6d0b6772.json` (local archive) records intent loss 4.102364 to 0.000087817 and slot loss 4.046995 to 0.000006974.

The `artifacts/phase5/independent_audit.json` (local archive) passed **128 checks**. It verified every full training epoch, all five development results, original identities, manual and shared metric recomputation, complete learned-parameter counts, source/configuration/checkpoint hashes, registration/selection/freeze/test chronology and all snapshot bytes. It performed **zero model forward passes or updates**. All **17 Stage 1 sources and 38 baseline snapshot files remain unchanged**.

| Final Stage 2 evidence | Development | Test, reporting only |
| --- | ---: | ---: |
| Correct intents / records | 1,721 / 2,033 | 2,457 / 2,974 |
| Slot TP / FP / FN | 1,576 / 359 / 435 | 2,111 / 571 / 704 |
| Intent CE per utterance | 0.700605004 | 0.738551629 |
| CRF NLL per real token | 0.212252516 | 0.247389196 |
| Joint loss | 0.912857520 | 0.985940825 |
| Tokens / batches | 14,522 / 64 | 21,089 / 93 |
| Final evaluation seconds | 1.95 | 2.10 |

CRF NLL and independent-token CE are different objectives; their loss values are not an architectural performance comparison. The same task metrics evaluate all models. Scoring remains the documented project convention of exact per-record type/text sets with repeated pairs collapsed; parity with an unavailable lecturer reference scorer is not asserted.

### Files, reproducibility and collaboration

| Purpose | Evidence |
| --- | --- |
| Baseline development diagnosis | [PHASE5_DIAGNOSIS.md](EXPERIMENTS_AND_ERRORS.md#development-diagnosis-and-qualified-findings), [diagnostics.py](../src/diagnostics.py), `artifacts/phase5/diagnosis.json` (local archive). |
| Registered hypotheses and budget | [PHASE5_EXPERIMENT_PLAN.md](EXPERIMENTS_AND_ERRORS.md#registered-research), `artifacts/phase5/experiment_plan.json` (local archive), [configurations](../configs/phase5). |
| Architecture and implementation | [STAGE2_DESIGN.md](MODEL_AND_EVALUATION.md#stage-2), [stage2_model.py](../src/stage2_model.py), [stage2_experiment.py](../src/stage2_experiment.py). |
| All candidate evidence | `artifacts/phase5/selection.json` (local archive), `artifacts/phase5/candidate_comparison.csv` (local archive), five run directories and `artifacts/phase5/candidate1_console.txt` through `candidate5_console.txt`. |
| Final-system errors and regressions | [PHASE5_ERROR_ANALYSIS.md](EXPERIMENTS_AND_ERRORS.md), `artifacts/phase5/development_comparison.json` (local archive). |
| Frozen final model and reporting | `artifacts/phase5/final_stage2` (local archive), `artifacts/phase5/stage_comparison.csv` (local archive), `artifacts/phase5/final_evaluation_console.txt` (local archive). |
| Genuine AI-assistance working record | `artifacts/phase5/collaboration_record.md` (local archive). |

Development-only verification remains available:

```powershell
.\.venv\Scripts\python.exe scripts/run_stage2.py --mode verify --checkpoint artifacts/runs/stage2_20261021T135955846459Z_6d0b6772/best.pt --device cuda
```

The completed selection/finalization/report commands were `scripts/select_stage2.py`, `scripts/finalize_stage2.py --selection artifacts/phase5/selection.json --device cuda`, and `scripts/summarize_results.py`. They preserve committed outputs rather than silently overwrite or repeat final evaluation. Offline development analysis and artifact audit perform no additional model test pass. Use a clean reproduction workflow for the final notebook; this record does not authorize post-test tuning.

The collaboration record contains the actual user instruction, genuine assistant excerpts, accepted/modified/rejected ideas, evidence and AI implementation/review contributions. It does not invent student contributions or claim to be a complete exported chat transcript. Group details remain pending at the user's request, and the final consolidated chat export belongs in Phase 8.

Results are from one seed and a bounded five-candidate search. They do not establish statistical significance, a global optimum, uniform per-example improvement or cross-hardware bitwise reproducibility. Stage 2 used five full candidates; Stage 1 used three. The held-out intent regression is unresolved scientifically and is reported without another test-driven tuning cycle.

**Next: Phase 6**, audit the completed script implementation and evidence for notebook conversion. The self-contained executed notebook remains Phase 7; the two-page report and exact submission archive remain Phase 8.

## Phase 6: historical completion record

Completed **22 October 2026 (Singapore time)**. The script implementation and saved evidence are ready for **Phase 7: self-contained notebook construction and clean execution**. The audit found no unresolved correctness defect requiring either frozen model to be retrained. This is script readiness; the final notebook, report and submission archive remain later deliverables.

### Completion gates

| Planned Phase 6 work | Completed evidence |
| --- | --- |
| Review original data, train-only fitting, inputs, scoring and complete parameter counts. | `artifacts/phase6/data_model_review.md` (local archive), with `artifacts/phase6/data_model_review.json` (local archive). Exact English file bytes match the original archive member; every original record, split and annotation round-trips correctly. |
| Trace results to actual runs, selected checkpoints and reloaded predictions. | `artifacts/phase6/evidence_review.md` (local archive) covers all three Stage 1 and five Stage 2 candidates, registration, histories, selection, reload records, final saved predictions and chronology. |
| Generate the four report rows programmatically and distinguish trained/selected epochs. | `artifacts/phase6/reference/results.csv` (local archive), `artifacts/phase6/reference/results.json` (local archive), and `artifacts/phase6/reference/results.md` (local archive), generated from the completed checkpoint-reloaded evaluations and reconciled with the preserved Phase 5 comparison. |
| Verify logs, curves, timing and unsuccessful experiments. | All eight original run directories retain histories, curves, checkpoints, saved development predictions, fresh-process checks and real console logs. The evidence review verifies their consistency; weaker candidates remain available. |
| Freeze the verified scripts and effective settings; inventory artifacts and document assumptions. | `artifacts/phase6/reference/implementation_manifest.json` (local archive), `artifacts/phase6/reference/snapshot` (local archive), `artifacts/phase6/reference/frozen_settings.json` (local archive), `artifacts/phase6/reference/artifact_inventory.csv` (local archive), and `artifacts/phase6/reference/audit_report.json` (local archive). |

The reference is produced by [scripts/audit_phase6.py](../scripts/audit_phase6.py). Its new output directory is exclusive. It verifies both older freezes, regenerates the table, checks the independent reviews and current test/runtime evidence, inventories files with SHA-256 digests, and copies the implementation/configuration/test/documentation reference with byte verification. It performs no model forward pass, training update or additional final test evaluation.

### Verified systems and results

| System | Split | Total / trainable parameters | Intent accuracy | Slot micro F1 | Trained epochs | Selected epoch |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Stage 1 | Validation | 4,646,571 / 4,646,571 | 84.36% | 73.61% | 20 | 19 |
| Stage 1 | Test | 4,646,571 / 4,646,571 | 83.96% | 70.51% | 20 | 19 |
| Stage 2 | Validation | 4,939,498 / 4,939,498 | 84.65% | 79.88% | 20 | 11 |
| Stage 2 | Test | 4,939,498 / 4,939,498 | 82.62% | 76.81% | 20 | 11 |

Both rows for a stage use the same frozen best-development checkpoint. They are not a mixture of final-epoch and selected-epoch results. All four saved prediction sets were independently rescored against the original records and agree exactly with the final evaluations. Stage 2's test slot gain is **6.30 percentage points**, with an intent decrease of **1.34 points** and balanced-mean gain of **2.48 points**. Phase 6 preserves that finding without another tuning cycle.

| Stage | Authoritative settings | Actual selected run |
| --- | --- | --- |
| Stage 1 | [stage1_dropout03.json](../configs/phase4/stage1_dropout03.json) | `artifacts/runs/stage1_20261021T132439418199Z_447f3fc1` (local archive) |
| Stage 2 | [char_crf_intent_context.json](../configs/phase5/char_crf_intent_context.json) | `artifacts/runs/stage2_20261021T135955846459Z_6d0b6772` (local archive) |

Stage 1 uses the 256-dimensional random word embedding and shared 512-unit-per-direction BiLSTM with intent and BIO slot heads. Stage 2 adds a train-only character CNN, soft predicted-intent context and BIO-constrained CRF to one joint network. One optimizer covers each complete model. Gold intent is never a forward feature. Full tensor accounting includes every embedding, both heads and all learned CRF parameters; both systems are below 20,000,000 parameters. No pretrained assets are used.

### Data, selection and integrity findings

- All **11,514 training, 2,033 validation and 2,974 test** records remain in their original order and partitions. No example is added, removed, re-split, augmented or truncated.
- The predictive features come from `utt` only. Word and label mappings equal a fresh fit on original training data; Stage 2's 34-entry character vocabulary also equals a fresh training-only fit. IDs, labels and annotation offsets remain evaluation/audit sidecars.
- All original annotations round-trip through the fixed tokenizer and BIO representation to their exact type, text and offsets. Shared literal type/text set scoring is unchanged across stages.
- All eight registered candidates completed 20 full epochs each: **60 Stage 1 epochs and 100 Stage 2 epochs**. Development-only selection uses the unrounded mean of intent accuracy and slot F1, retaining earlier epochs/candidates on exact ties.
- Each candidate has genuine fresh-process development reload evidence. Both finalizers reproduced all 2,033 ordered development predictions exactly before one complete final test pass per stage, with unchanged model weights and checkpoint bytes.
- Every original and copied byte recorded by the older freezes remains correct: **38 Stage 1 snapshot files / 17 sources**, and **79 Stage 2 snapshot files / 29 sources**. The new Phase 6 audit script and tests have their own reference identity; they do not alter the trained source identities.

The artifact inventory covers root files plus `src`, `scripts`, `tests`, `configs`, `data` and historical `artifacts` files, including unsuccessful candidates and duplicate historical snapshots. It records each path, role, byte size and SHA-256 digest. It excludes `.venv`, hidden directories, Python bytecode and Phase 6's own generated output to avoid a recursive inventory. The Phase 6 reviews, runtime check, unit-test log and PDF extraction are separately included in the reference snapshot. Large data/checkpoint files remain in their existing audited locations rather than being duplicated again.

### Current verification

The full suite passed **147 tests**, including eight new adversarial audit checks that reject changed original/snapshot/checkpoint bytes, inconsistent parameter/metric counts, incomplete or repeated final evaluation claims, missing exact reload evidence, unsafe evidence paths, unregistered/missing research runs, obsolete documentation links, and attempts to overwrite the reference. They also check that the inventory includes historical evidence while excluding its own output and the environment. See `artifacts/phase6/unit_tests_final.txt` (local archive).

The independent data/model review passed **58 checks**. The evidence review passed **297 checks**: all **128 saved-artifact Phase 5 audit checks** reran successfully, with **169 additional checks** extending coverage to all three baseline candidates, baseline final predictions, optimizer steps and evidence files. Its **310 input-file hashes** are checked again by the consolidated auditor, so a stale review cannot silently validate changed evidence. The handoff review passed **44 checks**. Detailed records remain in `artifacts/phase6/evidence_review.json` (local archive) and `artifacts/phase6/notebook_handoff_review.json` (local archive). These review counts cover overlapping evidence and are not presented as independent statistical observations. The reviews use original data and saved predictions/tensors, with no model inference or training. The test suite uses bounded correctness fixtures and original training records; it does not repeat either frozen model's final test evaluation.

The `artifacts/phase6/runtime_check.json` (local archive) found all **105 locked packages** installed at their recorded versions, with `pip check` passing. Python is 3.12.0; PyTorch is 2.7.1+cu126; CUDA is available on the RTX 3060 Ti. Direct `jupyter_client` inspection confirms that `sc4002` points to the project `.venv` interpreter. This establishes the current environment, not a successful execution of the future full notebook.

The root review visually inspected both selected learning-curve PNGs: all 20 epochs, training/development joint and task losses, and both development metrics are visible with readable legends. Rising development loss despite falling training loss remains documented; checkpoint choice follows the registered development metric.

The authoritative five-page assignment was reread. Its `artifacts/phase6/assignment_text.txt` (local archive) uses the PDF's embedded Unicode maps without changing the locked environment; PDF table layout is linearized. The PDF and the user's additional three rules remain the authority.

### Notebook conversion handoff

[NOTEBOOK_HANDOFF.md](../README.md#reproduce-the-notebook) maps the actual definitions and dependencies into notebook sections and specifies acceptance checks. The main adaptations are:

1. Embed all required definitions and effective configurations inside `main.ipynb`; replace project-local imports and external audit/config-file requirements with notebook-local definitions and checks.
2. Establish deterministic settings before Torch/CUDA work, select the correct local kernel, and use a fresh execution directory with paths relative to that run.
3. Fit preprocessing again on the original training split. Explicitly use selected dropout **0.3** in both models; the initial `configs/stage1.json` and class default remain historical dropout-0.2 settings.
4. Train each selected model for **20 epochs**, select/save/reload its own best-development checkpoint, reproduce development predictions, freeze and evaluate test once. This is a **40-epoch fixed-configuration reproduction**, distinct from the preserved 160-epoch historical search. Do not rerun the search or use reproduction test scores for new selection.
5. Adapt the script-specific historical registration/freeze guards into an isolated notebook execution ledger. Historical absolute paths and existing freezes must not become dependencies of the submitted notebook.
6. Generate logs, plots, the four-row comparison and at least two genuine final-system error cases from that notebook's own execution. Check preprocessing, parameter counts, scorer and checkpoint-prediction parity locally before accepting the clean run. Use its actual outputs in the final report.

### Assumptions and remaining limits

- Exact slot scoring remains a documented project interpretation: per-record sets of literal `(type, surface text)` pairs, duplicate pairs collapsed, offsets retained only for audit, and zero-denominator F1 set to zero. No lecturer reference scorer was supplied, so parity is not asserted. Both stages use the same convention.
- Stage 1 logs retain complete example/token counts but not per-epoch ID hashes. The frozen training code checks the full count and original ID set every epoch; Phase 6 verifies that code and all recorded counts. Stage 2 additionally stores per-epoch ID hashes. The audit does not invent missing historical telemetry.
- Stage 1 training closure is a documented policy: its frozen runner does not enforce a post-freeze training guard. Stage 2 enforces that guard in code. There are exactly three baseline and five Stage 2 full research runs; the new Phase 6 auditor rejects missing or additional experiment runs and checks the current hashes of independently reviewed evidence. Phase 7 uses its own fresh execution namespace.
- File-byte hashes and canonical JSON hashes serve different identities. In particular, the preprocessing JSON file digest differs from its parsed mappings digest; both are verified at their respective boundaries.
- Historical summaries contain some absolute local output/checkpoint paths. The audit follows canonical project-relative selection paths; Phase 7 must write new execution-relative paths. This is a documented conversion task, not an unexplained metric discrepancy.
- One seed and a bounded search do not establish statistical significance or cross-hardware bitwise reproducibility. Timing is observed workflow time, not an isolated architecture benchmark: selected Stage 1 **106.81 seconds**, Stage 2 **323.98 seconds**; candidate totals **332.73 / 1,303.85 seconds**. Setup, plotting, fresh-process reload checks and final evaluation are excluded.
- Stage 1 evaluation timers include scorer execution, whereas Stage 2's per-evaluation timer stops before scoring. Both complete training-workflow timers include scoring; their recorded totals retain the stated workflow meaning. Per-evaluation seconds should not be treated as comparable architecture microbenchmarks. Stage 2 peak memory is PyTorch-allocated memory, not total GPU memory usage.
- The original Stage 1 no-generative-AI instruction and actual user-requested AI assistance remain disclosed. Technical readiness does not create a lecturer-approved exception. Group ID, names and human contributions remain pending at the user's request; full genuine chat export and attribution remain Phase 8 work.

The actual Phase 6 assistance is recorded in `artifacts/phase6/collaboration_record.md` (local archive). No question or additional assumption blocks Phase 7 implementation under the existing user decisions.

### Reproduction of the audit

The completed invocation is:

```powershell
.\.venv\Scripts\python.exe -B scripts/audit_phase6.py
```

The existing `artifacts/phase6/reference` is protected against overwrite. A fresh read-only recheck can use a new project-relative `--output-dir` under `artifacts/phase6`; the original reference remains available. This command reads saved evaluations and verifies the hashes of completed review evidence; it does not train or run a task model. The full-suite command is `.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v`.

**Next: Phase 7.** Build and execute the self-contained notebook. The report, consolidated collaboration file and exact three-file ZIP remain Phase 8.

## Phase 7: historical completion record

Completed **22 October 2026 (Singapore time)**. [main.ipynb](../main.ipynb) is now a **self-contained, executed notebook with genuine retained outputs**. It trained both selected joint systems from random initialization, saved and reloaded its own checkpoints, reproduced development predictions in fresh Python processes, froze each model and performed one complete final test pass per stage. All Phase 7 implementation and acceptance gates passed. **Phase 8 is next.**

### Submitted notebook artifact

The notebook has **39 cells, including 26 executed code cells**, in ascending execution order with **zero error outputs**. It contains the complete data acquisition, typed data/prediction representations, preprocessing, shared scoring, both models, loss/epoch algorithms, checkpointing, independent verification, finalization, plotting, generated table and error-analysis source. It includes explicit environment/startup instructions and literal selected configurations.

All **40 full training epochs** are visible in real notebook output, with training/development joint and component losses, development metrics, example/token counts and elapsed times. Both four-panel learning curves are retained as **inline PNG outputs**. The final table and four actual remaining development errors are generated by the notebook, not pasted from script summaries.

The `artifacts/phase7/clean_execution_execution.json` (local archive) records an initial directory containing **only `main.ipynb`**. No development module, configuration file, audit file, data cache or previous checkpoint was copied into that directory. The notebook downloaded the official MASSIVE archive and extracted the original English file itself. Its `artifacts/phase7/clean_execution/run_artifacts/data/source_manifest.json` (local archive) records that acquisition; all 11,514 train / 2,033 development / 2,974 test records and exact source-file bytes passed verification.

The execution directory is `artifacts/phase7/clean_execution` (local archive); notebook-created evidence is in `artifacts/phase7/clean_execution/run_artifacts` (local archive). Those working files are not additional submission files. In particular, checkpoints are generated beside the notebook during execution rather than embedded as binary weights or loaded from the old research runs.

### Actual notebook results

| Model | Split | Total / trainable parameters | Intent accuracy | Slot micro F1 | Trained / selected epoch | Notebook training-workflow seconds |
| --- | --- | --- | ---: | ---: | --- | ---: |
| Stage 1 | Validation | 4,646,571 / 4,646,571 | 84.36% | 73.61% | 20 / 19 | 127.51 |
| Stage 1 | Test | 4,646,571 / 4,646,571 | 83.96% | 70.51% | 20 / 19 | 127.51 |
| Stage 2 | Validation | 4,939,498 / 4,939,498 | 84.65% | 79.88% | 20 / 11 | 359.09 |
| Stage 2 | Test | 4,939,498 / 4,939,498 | 82.62% | 76.81% | 20 / 11 | 359.09 |

Sources: the notebook's generated `artifacts/phase7/clean_execution/run_artifacts/results.csv` (local archive), `artifacts/phase7/clean_execution/run_artifacts/results.json` (local archive), and new `artifacts/phase7/clean_execution/run_artifacts/stage1/final_summary.json` (local archive) / `artifacts/phase7/clean_execution/run_artifacts/stage2/final_summary.json` (local archive) final summaries. These are the results to use in the Phase 8 report.

All four ordered prediction sets, all integer metric counts, the selected epochs, and every saved model-state tensor value match the earlier script experiments exactly. This agreement is an observation from the completed run, not an acceptance target imposed on training. The `artifacts/phase7/reproduction_comparison.json` (local archive) records it without any additional model inference. New checkpoint **files** have different hashes because their execution metadata and provenance are new; matching tensor values does not imply the files are byte-identical to historical checkpoints.

The test tradeoff remains: **slot F1 +6.30 percentage points, intent accuracy -1.34 points**, with a **+2.48-point balanced-mean gain**. Development gains are +0.30 intent / +6.27 slot points. No reproduction test result changed any setting or checkpoint choice.

Measured workflow times are **127.5089132 seconds** for Stage 1 and **359.0891542 seconds** for Stage 2, **486.5980674 seconds total** (about 8.11 minutes). They include training, epoch development evaluation, checkpoint writes and epoch evidence; they exclude setup/download, plotting, independent reload checks and final evaluation. They are local notebook observations. The earlier script measurements remain 106.81 / 323.98 seconds and must not be substituted for these notebook measurements. Per-evaluation timing boundaries differ between stage algorithms, so those seconds are not an isolated architecture benchmark.

### Conversion and parity verification

[scripts/build_notebook.py](../scripts/build_notebook.py) extracted core definitions from the verified source and removed project-local imports. It renamed only the Stage 2 orchestration helpers to avoid collisions. Notebook-local checkpoint/workflow helpers replace historical registration/audit-file dependencies. Their full ordinary Python source is visible inside the notebook. The builder records provenance for **12 source modules** and an ordered hash for every code cell. It now refuses to replace a notebook containing execution counts or outputs; a new review build must use another path.

The `artifacts/phase7/parity.json` (local archive) passed **192 checks**, including normalized core-definition syntax, literal selected settings, original train-only mappings/batches, parameter counts, exact initialized state, dropout/evaluation logits and losses, and both models' full 2,033-example development predictions using the same historical checkpoint tensors. This development-only correctness check made **zero optimizer updates and zero test forward passes**. It ran separately from the clean notebook, which loaded no historical checkpoint.

Notebook implementation identity (ordered code-cell hashes):

```text
ee0f671d82a26a8ba2304003a4327dae6579bcb75cf9021717a1c67d12becf4c
```

The execution used the declared `.venv` interpreter, Python 3.12.0, PyTorch 2.7.1+cu126 and RTX 3060 Ti. `PYTHONHASHSEED=42` was set at kernel startup; CUDA workspace, strict deterministic FP32, cuDNN and TF32 settings were applied before model work. Each stage reset all RNGs before construction and used persistent seeded loaders for its complete 20-epoch run. Stage 2 was freshly initialized, not warm-started.

### Checkpoint and final-evaluation evidence

| Evidence | Stage 1 | Stage 2 |
| --- | --- | --- |
| Selected checkpoint | `artifacts/phase7/clean_execution/run_artifacts/stage1/best.pt` (local archive) | `artifacts/phase7/clean_execution/run_artifacts/stage2/best.pt` (local archive) |
| Training history | `artifacts/phase7/clean_execution/run_artifacts/stage1/history.jsonl` (local archive) | `artifacts/phase7/clean_execution/run_artifacts/stage2/history.jsonl` (local archive) |
| Fresh-process full-development check | `artifacts/phase7/clean_execution/run_artifacts/stage1/reload_check.json` (local archive) | `artifacts/phase7/clean_execution/run_artifacts/stage2/reload_check.json` (local archive) |
| Persisted pre-test freeze | `artifacts/phase7/clean_execution/run_artifacts/stage1/freeze_manifest.json` (local archive) | `artifacts/phase7/clean_execution/run_artifacts/stage2/freeze_manifest.json` (local archive) |
| Actual test predictions | `artifacts/phase7/clean_execution/run_artifacts/stage1/test_predictions.jsonl` (local archive) | `artifacts/phase7/clean_execution/run_artifacts/stage2/test_predictions.jsonl` (local archive) |

Each independent verifier is ordinary Python generated from the notebook's own visible definitions. It reloaded the newly trained checkpoint in a distinct process, refitted mappings on original train, verified source/data identity and reproduced all ordered development predictions with **zero score difference**. No hidden project-module import is needed.

The parent then reloaded a fresh model, verified development again and persisted its freeze before the first test forward pass. Each new stage records exactly one complete 2,974-example test evaluation, unchanged checkpoint bytes/model state, and closed training/tuning. The existing script experiment freezes are preserved separately; this authorized fixed-configuration notebook reproduction is not another test-driven tuning cycle.

### Final acceptance and genuine errors

The `artifacts/phase7/notebook_validation.json` (local archive) **passed**. It checked inline code/source identity, absence of external project imports, execution order and error outputs, all 40 printed epoch logs, both inline PNGs against the newly generated plot files, complete original IDs/tokens/batches, train-only word/character mappings, whole-model parameter counts, one joint AdamW parameter group, optimizer steps, earliest-best development selection, checkpoint tensors, fresh-process evidence and freeze chronology. It independently rescored every final saved prediction set and reconciled the four-row table, deltas and genuine error cases. This audit performed no model inference or training.

The full current suite passed **157 tests**, including ten Phase 7 acceptance/output-preservation checks; see `artifacts/phase7/unit_tests_final.txt` (local archive). The two selected notebook learning curves were also visually inspected: both show all 20 epochs, readable labels, task losses and development metrics. Development loss still rises after early epochs as training loss falls; checkpoint selection uses the declared task metric.

Four genuine remaining Stage 2 development errors are displayed and saved in `artifacts/phase7/clean_execution/run_artifacts/error_cases.json` (local archive):

- **42:** “give me the status on my available memory” — gold `general_quirky`, predicted `calendar_query`.
- **67:** “i'd like to hear queen's barcelona” — gold separates artist `queen's` and song `barcelona`; the model merges them into one artist span.
- **24:** “make a room darker” — gold `iot_hue_lightdim`, predicted `iot_hue_lightup`, with the room slot correct.
- **30:** the carpet-cleaning request is misclassified as `calendar_remove`, and the `carpet` location slot is omitted.

Their full literal gold/predicted spans and offsets are retained. Possible explanations are labelled as inferences, with future work proposed as a separate development-based study. There are 674 development utterances with any remaining intent/slot error; the notebook does not hide those failures.

### Integrity, limits and next phase

All **38 Stage 1 and 79 Stage 2 historical snapshot files** and their corresponding trained sources/evidence remain unchanged. The **86 copied Phase 6 reference files** remain unchanged. Current progress documents have been updated for Phase 7; historical copied documentation remains the earlier record. Vocabulary labels in the current design/compliance documents were corrected to **5,146**, consistent with the original train-only mappings and parameter formula; no data or numerical artifact changed.

An execution-wrapper syntax error was corrected before any kernel/training began. The successful execution retained actual outputs throughout. After it finished, the spawned Windows kernel exited and its recorded training PID was confirmed absent. The external wrapper was then tightened to request explicit kernel cleanup on future invocations; this changes no notebook source, output or model evidence.

Scoring remains the disclosed literal per-record type/text-set interpretation, with duplicate pairs collapsed and offsets kept for audit; lecturer-scorer parity is unverified. Single-seed results do not establish statistical significance, a global optimum or cross-hardware bitwise reproducibility. The original Stage 1 AI-use requirement and actual user-requested assistance remain honestly disclosed.

The actual Phase 7 assistance record is `artifacts/phase7/collaboration_record.md` (local archive). Group ID, full names and confirmed individual human contributions remain pending at the user's request.

**Next: Phase 8.** Write the two-page report from these actual notebook results, consolidate the genuine collaboration history, add verified group contributions, and create the exact three-file submission archive. The report, final consolidated chat file, ZIP and submission are not completed by this phase.

## Historical implementation plan

Drafted on **20 October 2026 (Singapore time)** from [PROJECT_REVIEW.md](ASSIGNMENT.md) and all five pages of [Group Assignment Instruction.pdf](assignment.pdf).

**Development approach:** build and verify the project in Python scripts, complete the experiments, then transfer the validated implementation into an executed, self-contained `main.ipynb`. Each phase below has a completion gate. Progress is recorded separately from the planned work.

**Progress, 22 October 2026 (Singapore time):** Phases **1-7 are complete and verified**. After the eight-candidate script study and [Phase 6 audit](#phase-6-historical-completion-record), [main.ipynb](../main.ipynb) genuinely retrained both selected systems for **40 additional reproduction epochs** in a clean directory, saved/reloaded its own checkpoints, verified development predictions in fresh processes, and froze before each full test pass. All full-precision script metrics and selected epochs reproduced exactly. The `artifacts/phase7/notebook_validation.json` (local archive), **192 `artifacts/phase7/parity.json` (local archive)** and **157 `artifacts/phase7/unit_tests_final.txt` (local archive)** passed. See [PHASE7_STATUS.md](#phase-7-historical-completion-record). **Phase 8 is next:** report, consolidated genuine collaboration history and submission package. The runtime remains local Jupyter using the RTX 3060 Ti; group details remain pending at the user's request.

Your request to use AI assistance for Stage 1 is carried forward. The original brief's AI rule and your requested approach are documented in the project review; the development record will describe the assistance actually used.

### Phase overview

| Phase | Goal | Main deliverable | Completion gate |
| --- | --- | --- | --- |
| 1. Set up the project and experiment protocol | Establish a portable environment and consistent experiment rules. | Development structure, environment specification, experiment settings. | The environment runs and the protocol is explicit. |
| 2. Prepare data and verify evaluation | Establish correct inputs, targets, and scoring. | Dataset audit, preprocessing pipeline, shared evaluator. | Original splits are intact and annotation/scoring checks pass. |
| 3. Implement and verify the baseline | Build a credible system for both tasks in scripts. | Stage 1 model and reusable training pipeline. | Forward pass, losses, decoding, and checkpoint round trip work. |
| 4. Train and evaluate Stage 1 | Produce a reproducible baseline with complete evidence. | Selected baseline checkpoint, logs, curves, validation/test results. | Reloaded checkpoint reproduces its selected validation score. |
| 5. Diagnose and develop Stage 2 | Make and evaluate justified improvements with AI collaboration. | Improved system, experiment record, selected checkpoint, results. | Validation-based selection and final checkpoint evaluation are complete. |
| 6. Audit the script implementation | Confirm that the code and evidence are ready for conversion. | Frozen source/configurations and verified experiment artifacts. | All required evidence is traceable and both systems meet the rules. |
| 7. Assemble and execute the notebook | Transfer the validated code and reproduce the workflow. | Executed, self-contained `main.ipynb`. | Clean-kernel execution succeeds and produces genuine outputs. |
| 8. Write the report and package submission | Present findings and submit the exact required files. | `report.pdf`, collaboration record, `Group_<ID>.zip`. | Content, page limits, results, and archive contents are checked. |

Phases 1–6 use scripts. Phase 7 begins once the script implementation and experiments have passed their gates. Collaboration records, contribution notes, and report observations should be collected throughout.

### Working rules across all phases

- Use original MASSIVE 1.0 English, `1.0/data/en-US.jsonl`, with **11,514 train / 2,033 dev / 2,974 test** examples. Preserve all records and split membership.
- Supply only `utt` as model input. Use `intent` and `annot_utt` as targets, and fit vocabulary/model parameters on train only.
- Use one joint model/framework for intent classification and slot filling in each stage; separate task-specific models are prohibited. Keep each stage's entire system within **20,000,000 total parameters**, including embeddings, both heads, all other learned components, and frozen parameters. These requirements were clarified on 21 October 2026; see [ADDITIONAL_RULES_REVIEW.md](ASSIGNMENT.md#model-rules-including-additional-clarifications).
- Pretrained word embeddings are permitted; pretrained models, external task examples, AI-generated task examples, and training-data augmentation are prohibited. Any permitted embeddings remain included in the complete system's parameter count.
- Tune and select using train/dev evidence. Test evaluation follows model freezing. Baseline test results are retained for reporting and excluded from Stage 2 diagnosis and selection.
- Preserve genuine outputs, run settings, checkpoints, and unsuccessful experiments. Performance percentages use two decimal places when displayed; retain full-precision values internally.
- Treat notebook conversion as a packaging and reproducibility step. Modelling or scoring changes discovered during conversion must be corrected in the scripts and reverified for both affected stages.

### Phase 1 — Set up the project and experiment protocol

**Goal:** make development reproducible and keep the eventual notebook portable.

1. Inspect the available Python environment, device, GPU memory if available, and practical training time. Confirm the intended notebook runtime before scheduling full runs.
2. Create an isolated environment and record tested package versions. Proposed stack: Python, PyTorch, NumPy, Matplotlib, and Jupyter tooling. These are implementation choices, not assignment requirements.
3. Create a small development structure with separate responsibilities for data processing, models, training, scoring, and experiment execution.
4. Keep paths relative to a configurable project root. Pass settings explicitly instead of relying on machine-specific paths, interactive prompts, or hidden global state.
5. Choose the initial seed policy, checkpoint-selection metric, timing convention, run naming, and checkpoint tie-breaking rule. Proposed selection score: the arithmetic mean of validation intent accuracy and slot micro F1, using the same formula in both stages. On a tie, retain the earlier checkpoint.
6. Record group responsibilities and start a genuine collaboration/contribution record. Avoid inventing a fixed schedule before hardware and availability are known.

**Suggested development layout** — create only what is needed during implementation:

```text
SC4002/
├── PROJECT_REVIEW.md
├── IMPLEMENTATION_PLAN.md
├── requirements.txt
├── src/
│   ├── data.py                 # Loading, annotation parsing, tokenization, batches
│   ├── metrics.py              # Shared intent and slot scoring
│   ├── models.py               # Stage 1 and Stage 2 model definitions
│   └── training.py             # Training, validation, checkpointing, prediction
├── scripts/
│   ├── audit_data.py
│   ├── run_experiment.py       # Explicit stage/configuration; separate test mode
│   ├── summarize_results.py
│   └── build_notebook.py       # Added when the implementation is ready
├── configs/                    # Versioned experiment settings
├── tests/                      # Focused correctness checks
├── data/                       # Original data; excluded from submission
├── artifacts/                  # Run logs, checkpoints, predictions, figures
├── main.ipynb                  # Created and executed in Phase 7
├── ai_collaboration.md
└── report.pdf
```

Use ordinary functions/classes with thin script entry points so notebook conversion does not require rewriting the algorithms. A modest module split is sufficient; no production service or deployment infrastructure is needed.

Record seeds and deterministic settings, but do not promise identical training across hardware or package versions. PyTorch documents that reproducibility has platform and release limits. [PyTorch reproducibility documentation](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

**Deliverables:** runnable environment, initial structure, explicit experiment protocol.

**Gate:** imports and device selection work; the intended data paths, selection formula, seed, and run-output layout are documented.

### Phase 2 — Prepare data and verify evaluation

**Goal:** establish correct labels and scoring before interpreting any model performance.

1. Download the specified original archive, extract safely, and retain the English file. Record its source and a file hash for traceability.
2. Load records by their existing `partition`; print and assert the three required counts and total. Retain record IDs for audits, never as model features.
3. Audit the training distribution: label coverage, utterance lengths, slot frequencies, and examples with no slots. Validation may support diagnosis; test labels must not drive exploratory model decisions. Restrict early test processing to structural verification needed to load and preserve the data.
4. Parse `annot_utt` into gold slot types and spans, align annotations to `utt`, and retain original character offsets/text internally. Check punctuation, spacing, repeated values, adjacent slots, and empty slot collections. Resolve mismatches explicitly without dropping examples or changing gold annotations silently.
5. Implement deterministic tokenization with offsets and a train-fitted vocabulary. Use unknown-token handling for unseen words. If using BIO slot tags, verify that tokens/tags can reconstruct the original gold spans. Do not truncate away labelled content; select a length-handling strategy that preserves full examples.
6. Implement the one shared intent-accuracy and exact-slot micro-F1 evaluator. Lock span matching, duplicate handling, normalization, and zero-denominator conventions before experiments; obtain a reference evaluator or document a provisional interpretation if no clarification is available.
7. Add focused correctness checks using existing training annotations and isolated metric-count inputs. Cover correct predictions, wrong types, wrong boundaries, missing/extra slots, repeated slots, and no-slot inputs. These checks do not add training examples.
8. Establish one prediction format used throughout, containing the record ID, predicted intent, and slot pairs; keep offsets available for checking reconstruction.

**Deliverables:** dataset audit, reusable preprocessing, label mappings, shared scorer, focused correctness checks.

**Gate:** the required counts pass, train/dev annotation alignment is verified for all records, gold spans round-trip correctly, and hand-checked TP/FP/FN totals agree with the evaluator. Freeze evaluation semantics before baseline selection. Apply the same structural checks to the full test split before final evaluation without using its findings to tune the model.

### Phase 3 — Implement and verify the baseline in scripts

**Goal:** produce a sound Stage 1 system that handles both required tasks.

1. Choose a credible starting architecture after the data audit and hardware check. A proposed starting candidate is a trainable word-embedding layer, a bidirectional recurrent encoder, an utterance-level intent classifier, and a token-level slot tagger. This is a draft design direction, not a claimed best model or a lecturer requirement.
2. Define the architecture, output representations, loss functions, optimizer, and decoding rules. State how the two task losses are combined. If using BIO tags, document handling of invalid tag transitions.
3. Count and print total and trainable parameters for the full pipeline, including embeddings and both task heads; assert the total cap before training.
4. Implement batching, padding masks, loss masking, training and evaluation modes, and conversion from token predictions back to exact slot text. Verify that padding does not change scoring or contribute to slot loss.
5. Implement reusable epoch logging, validation scoring, timing, learning curves, and best-validation checkpoint saving. Keep task-specific losses/metrics visible enough to detect one task dominating training.
6. Save the model's parameter state, architecture/configuration, mappings, selected epoch, and selection score together. Retain optimizer state if training resumption is supported. Reload into a fresh model instance for evaluation; use evaluation mode. PyTorch recommends parameter-state saving and documents evaluation mode and best-state copying pitfalls. [PyTorch saving/loading guide](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
7. Run short execution checks before a full experiment. A diagnostic pass may inspect a few batches from the unchanged training loader; it is not a replacement training split or reportable experiment. Use the complete official splits for assessed runs.

**Deliverables:** baseline definition, shared training/evaluation implementation, successful short execution checks.

**Gate:** both outputs have valid shapes and labels; losses/gradients are finite; masking and decoding checks pass; saving/reloading preserves validation predictions for the same checkpoint. Diagnose a model that fails to learn or produces incorrect outputs before spending time on full training.

**Completed, 21 October 2026:** implemented a joint BiLSTM with trainable embeddings, masked max pooling for intent, and token BIO tagging. The user's selected configuration uses 256-dimensional embeddings and 512 hidden units per direction, for **4,646,571 parameters**; it supersedes the original 966,827-parameter configuration, whose diagnostic artifacts remain preserved. The shared pipeline includes loss masking, global loss aggregation, clipping, epoch logs, curves, strict checkpoint selection, and fresh-process reload verification. All 82 tests and the RTX 3060 Ti diagnostics passed again with the larger model, including identical ordered predictions after checkpoint reload. The two short diagnostic epochs and fixed-training-batch learning probe are correctness evidence, not assessed baseline results; see [BASELINE_DESIGN.md](MODEL_AND_EVALUATION.md#stage-1) and [PHASE3_STATUS.md](#phase-3-historical-completion-record) for configuration-specific test and GPU evidence.

### Phase 4 — Train, select, and evaluate Stage 1

**Goal:** establish a credible baseline with all required evidence.

1. Train on the full official training split. Use validation for the planned, bounded set of tuning decisions; keep the selection criterion fixed within each experiment.
2. Retain per-epoch training/validation losses, both validation metrics, the selection score, elapsed time, and learning curves. Record both the number of epochs actually trained and the best checkpoint epoch.
3. Save the highest-validation-score checkpoint and print its epoch, score, and filename. Preserve each run's configuration and results rather than overwriting evidence from earlier runs.
4. Freeze the final baseline. Reload its checkpoint in a fresh evaluation process without further training, evaluate dev again, and verify reproduction of the recorded best selection score using a documented numerical tolerance and prediction agreement checks.
5. Perform the required baseline test evaluation and retain both metrics. Archive test results for reporting; Stage 2 diagnostic tools should consume dev predictions only.
6. Preserve a baseline snapshot: source/configuration identity, preprocessing/mappings, selected checkpoint, logs, curves, dev predictions, and final results.

**Deliverables:** complete baseline run evidence and final validation/test metrics.

**Gate:** both tasks perform meaningfully, required logs are complete, the parameter cap passes, and reloaded-checkpoint validation reproduces selection evidence. A large Stage 2 gain is not a reason to retain a knowingly broken or deliberately weak baseline.

**Completed, 21 October 2026:** the preregistered [Phase 4 plan](EXPERIMENTS_AND_ERRORS.md#registered-research) compared three complete 20-epoch runs of the 4,646,571-parameter 256/512 BiLSTM: learning rate/dropout 0.001/0.2, 0.0005/0.2, and 0.001/0.3. The third candidate won by the fixed unrounded development score, selecting epoch **19 of 20**. Its final development intent accuracy / slot micro F1 are **84.36% / 73.61%**, and final test results are **83.96% / 70.51%**. Fresh-process reload reproduced all 2,033 ordered development predictions exactly with zero score difference. One complete final test evaluation followed source/configuration/checkpoint freezing, with weights and checkpoint bytes unchanged. Logs, curves, all candidate outcomes, immutable snapshot, generated tables, 93 passing tests, and timing evidence are linked in [PHASE4_STATUS.md](#phase-4-historical-completion-record). Stage 1 training and tuning are now closed; Stage 2 diagnosis uses development data.

### Phase 5 — Diagnose weaknesses and develop Stage 2

**Goal:** make justified improvements and document the group's reasoning.

1. Analyze baseline **dev** errors and learning curves. Separate implementation defects from modelling limitations. Examine intent confusions, missed/extra slots, span-boundary errors, rare slot types, and unfamiliar words where supported by the evidence.
2. Discuss those findings with AI and preserve the real prompts/responses, identified tool/model, accepted/modified/rejected suggestions, and group reasoning. Keep contributions and unsuccessful experiments traceable.
3. Create a short experiment list with a hypothesis, a change, the expected task affected, parameter/runtime cost, and dev-based acceptance criterion. Avoid an unrestricted search.
4. Choose changes based on observed weaknesses. Possible candidates include better character-level representations, improved intent pooling, structured slot decoding, regularization, or loss balancing. These are conditional options; implement only the changes justified by the diagnosis and budget. Permitted word vectors are another option only after provenance and total-parameter accounting are resolved.
5. Change one major factor at a time where practical, then test a justified combination. Reuse the same dataset handling, scoring, logging, and selection policy. If a shared correctness bug is discovered, fix it and rerun all affected stages before comparing them.
6. Train candidate systems on train only. Compare dev scores together with total/trainable parameters and elapsed time. Preserve regressions and failed hypotheses honestly. Do not select candidates using test results.
7. Freeze the final improved configuration and selected checkpoint. Reload, reproduce dev selection evidence, and perform its final test evaluation without further training. Retain genuine errors for the required analysis of at least two final-system cases.

**Deliverables:** improved-system implementation and run evidence, controlled comparison record, AI collaboration history, final validation/test results.

**Gate:** the chosen system satisfies both tasks and the parameter cap, its selection is supported by dev evidence, and its final checkpoint reload is verified. Revise unsuccessful candidates using dev evidence within the budget **before step 7's final freeze and test evaluation**. After that evaluation, preserve the result and report any remaining limitations without inventing gains.

**Completed, 21 October 2026:** [development diagnosis](EXPERIMENTS_AND_ERRORS.md#development-diagnosis-and-qualified-findings) motivated five registered candidates testing character CNN features, constrained CRF slots, predicted-intent context, and their combinations. All five completed 20 full epochs. The combination of all three won development selection at **epoch 11 of 20**, with **4,939,498 total/trainable parameters**. Development intent/slot scores are **84.65% / 79.88%**; final test scores are **82.62% / 76.81%**. Relative to Stage 1, both development metrics improve; test intent decreases **1.34 percentage points** and test slot F1 increases **6.30 points**, increasing the balanced mean **2.48 points**. No test result selected or retuned the model. All five fresh-process development reloads passed; the final source/configuration/checkpoint snapshot preceded the single complete test pass. [PHASE5_STATUS.md](#phase-5-historical-completion-record) links the 139-test suite, 128-check audit, timing, all candidate outcomes, collaboration record, and [final-system development errors](EXPERIMENTS_AND_ERRORS.md). Training and tuning for both stages are closed.

### Phase 6 — Audit scripts and freeze the implementation

**Goal:** establish that the project is ready for notebook transfer.

1. Review data preservation, train-only vocabulary/fitting, input fields, shared scoring, and complete parameter counts for both stages.
2. Check that every reported result points to an actual run, configuration, selected checkpoint, and saved output. Ensure the comparison uses checkpoint-reloaded results rather than a mixture of final-epoch and best-epoch scores.
3. Generate the four-row report-results table programmatically from recorded evaluations. Keep the total epochs and selected epoch separately available.
4. Confirm that logs and curves include all required evidence, elapsed-time labels are clear, and unsuccessful experiments remain available for discussion.
5. Freeze model settings and snapshot the verified scripts. Record any remaining scoring assumptions or experimental limitations explicitly.

**Deliverables:** verified script reference, complete artifact inventory, frozen settings, generated comparison data.

**Gate:** both stages can be followed from data to checkpoint to score; no unexplained score mismatch, missing evidence, or unresolved correctness defect remains. Only then start notebook conversion.

**Completed, 22 October 2026:** the script audit reconciled original data, shared preprocessing/scoring, both whole-system parameter counts, all eight registered full runs, checkpoint selection/reload evidence, saved final predictions and generated metrics. The suite at the Phase 6 gate passed **147 tests**, with **58 data checks**, **297 evidence checks** (169 new plus 128 fresh reused checks), **44 handoff checks**, and all **105 locked package versions** verified. No new training or test inference was required. See [PHASE6_STATUS.md](#phase-6-historical-completion-record), `artifacts/phase6/reference/audit_report.json` (local archive), `artifacts/phase6/reference/results.csv` (local archive), `artifacts/phase6/reference/implementation_manifest.json` (local archive), `artifacts/phase6/reference/artifact_inventory.csv` (local archive), and `artifacts/phase6/reference/frozen_settings.json` (local archive).

The review records the frozen Stage 1 trainer's procedural post-test closure, compared with Stage 2's explicit training guard. No additional Stage 1 full runs exist in the historical script namespace, and the Phase 6 auditor rejects extra full runs beyond the registered three plus five. Stage 1 enforces IDs at runtime and logs counts but lacks historical per-epoch ID hashes. Historical absolute summary paths and the dropout-0.2 defaults must be adapted explicitly during conversion. [NOTEBOOK_HANDOFF.md](../README.md#reproduce-the-notebook) specifies visible inline definitions, notebook-local paths, deterministic startup, fixed 20-epoch reproduction per selected stage, checkpoint reload/development parity, fresh-process development verification and final freeze/test acceptance gates. These are conversion tasks, not reasons to reopen test-based model tuning.

### Phase 7 — Assemble and execute `main.ipynb`

**Goal:** reproduce the validated workflow inside the final submission artifact.

1. Transfer verified definitions into notebook cells in dependency order. A small builder can copy marked source sections into cells to reduce manual drift. Convert command-line entry points into explicit notebook configuration and function calls.
2. Include **all required source code** inside the notebook. It must not depend on imports from unsubmitted `src/`, `scripts/`, external configuration files, or local checkpoints. Include package/setup instructions, data acquisition/loading, and writable artifact paths.
3. Add readable Markdown explaining architecture, inputs/outputs, losses, optimizer, selection policy, and Stage 1/Stage 2 changes. Keep the same data/scoring rules as the scripts.
4. Before full training, verify script/notebook parity on preprocessing, label mappings, parameter counts, scoring, and same-checkpoint predictions. Development checkpoints may support this comparison locally, but the final notebook must generate its required checkpoints itself.
5. Execute in a fresh kernel and a clean working location without access to the development modules or pre-existing artifacts. Run both training pipelines, select/save/reload checkpoints, and produce real validation/test outputs, timing, and learning curves.
6. Retain the execution outputs. Do not paste script logs into output cells, type scores manually, or merely display precomputed metric files as training evidence.
7. Check execution order, errors, plots, table consistency, and repeated-cell state. If the code changes after execution, rerun the affected dependency chain before accepting the output.
8. Use the executed notebook's actual results as the report's final source. With frozen settings, notebook reproduction is a verification run; its test outcomes must not start another tuning cycle. If hardware or nondeterminism changes results, record the discrepancy and preserve the actual outputs rather than forcing the earlier numbers.

**Proposed notebook order:**

| Section | Contents |
| --- | --- |
| 1 | Project scope, environment/setup, configuration, seeds, device. |
| 2 | Dataset acquisition, original split counts/assertions, preprocessing and mappings. |
| 3 | Shared metric functions and focused correctness checks. |
| 4 | Baseline definition, explanation, parameter counts/cap assertion. |
| 5 | Baseline training, epoch outputs, curves, best-checkpoint details. |
| 6 | Baseline checkpoint reload, dev reproduction check, final dev/test evaluation. |
| 7 | Stage 2 rationale, model/configuration changes, parameter counts/cap assertion. |
| 8 | Stage 2 training, epoch outputs, curves, best-checkpoint details. |
| 9 | Stage 2 checkpoint reload, dev reproduction check, final dev/test evaluation. |
| 10 | Generated comparison table, at least two real improved-system errors, limitations. |

**Deliverable:** executed, self-contained `main.ipynb` with visible genuine evidence.

**Gate:** a clean execution completes using the declared environment and official data; both stages have all required outputs; no unsubmitted project file is needed.

**Completed, 22 October 2026:** the self-contained [main.ipynb](../main.ipynb) has **39 cells and 26 executed code cells**, all **40 training epochs**, two inline learning-curve images, generated four-row results, and four genuine remaining Stage 2 development errors. Clean execution began with only the notebook, downloaded original data, and used no historical model weights, external configuration or project-module imports. Both stages trained from random initialization with the fixed settings, saved and selected their own checkpoints, exactly reproduced all 2,033 ordered development predictions in fresh Python processes, reloaded fresh models and froze before one complete test pass each. Whole-system counts remain **4,646,571 / 4,939,498**, with selected epochs **19 / 11** of 20. The `artifacts/phase7/clean_execution/run_artifacts/results.csv` (local archive) match all script metrics exactly; `artifacts/phase7/clean_execution/run_artifacts/error_cases.json` (local archive) retain gold/predicted facts and explicitly labelled hypotheses. Notebook workflow times are **127.51 / 359.09 seconds**, distinct from the historical **106.81 / 323.98 seconds**. The `artifacts/phase7/notebook_validation.json` (local archive), `artifacts/phase7/parity.json` (local archive) and `artifacts/phase7/unit_tests_final.txt` (local archive) passed. The original experiment/reference snapshots remain intact. [PHASE7_STATUS.md](#phase-7-historical-completion-record) records the full acceptance evidence; report, consolidated chat and ZIP remain Phase 8.

### Phase 8 — Write the report and package the submission

**Goal:** submit an accurate and complete account within the required format.

1. Finalize the report from notebook evidence: both pipelines, changes and reasoning, the four required results rows, total/trainable parameter counts, both percentage metrics, and epoch information with an explicit label.
2. Explain AI collaboration, verification of advice, group contributions, lessons, unsuccessful experiments, and limitations. Analyze at least two actual errors from the final improved system and support future-work suggestions with evidence.
3. Keep report content within two pages, excluding the cover. Put group ID, all names, and individual contributions on the cover. Inspect the PDF for readability, clipping, and page count.
4. Finalize one `ai_collaboration.md` or `.txt` containing genuine chat history and clear context for the development stages.
5. Inspect the notebook's preserved outputs and reconcile the report table with its actual results. Use the compliance checklist in the project review.
6. Create `Group_<ID>.zip` containing exactly `main.ipynb`, one collaboration file, and `report.pdf`. Inspect the ZIP contents; keep scripts, datasets, checkpoints, caches, environments, review/plan documents, and credentials outside it.
7. Submit to NTULearn → Group Assignment Submission before **15 November 2026, 11:59 pm SGT**, subject to any later official announcement. Retain the submission receipt and remember that only the last upload is graded/time-stamped.

**Deliverables:** the three final files, verified archive, and submission confirmation when the group uploads it.

**Gate:** all required content and genuine execution evidence are present, the PDF meets its limit, the archive contains only the three permitted files, and the uploaded version is verified.

### Decisions to settle during implementation

| Decision | Draft position / when to resolve |
| --- | --- |
| Framework | Phase 1 established PyTorch 2.7.1+cu126 in an isolated environment and recorded a tested dependency lock. |
| Hardware and notebook runtime | Local RTX 3060 Ti (8 GB) and project Jupyter kernel verified. The user confirmed local Jupyter using this GPU as the final notebook target. |
| Baseline architecture | The user selected 256-dimensional trainable embeddings + one packed bidirectional LSTM with 512 hidden units per direction + masked-max intent head + BIO slot head, with 4,646,571 parameters. Phase 4 selected dropout 0.3 in `configs/phase4/stage1_dropout03.json`; the original dropout-0.2 config and Phase 3 diagnostics remain preserved. See `BASELINE_DESIGN.md`, `PHASE3_STATUS.md`, and `PHASE4_STATUS.md`. |
| Checkpoint selection | Phase 1 established the unrounded mean of dev intent accuracy and dev slot micro F1; exact ties retain the earlier checkpoint. |
| Slot exact-match semantics | Phase 2 froze documented project convention v1: literal per-record type/text sets, duplicates collapsed, offsets retained for audit, no normalization, empty-slot denominator F1=0. See `DATA_AND_EVALUATION.md`. |
| Epochs and tuning budget | Phase 4 completed three runs / 60 epochs; Phase 5 completed five runs / 100 epochs. Every candidate trained 20 epochs with batch size 32. Both selected models use AdamW at 0.001 and dropout 0.3; selected epochs are 19 for Stage 1 and 11 for Stage 2. All outcomes are preserved; both stages are frozen. |
| Stage 2 improvements | Development evidence selected the joint character CNN + BiLSTM + CRF + predicted-intent-context model with 4,939,498 parameters. Its exact configuration is `configs/phase5/char_crf_intent_context.json`; see `STAGE2_DESIGN.md` and `PHASE5_STATUS.md`. |
| Group details and contributions | Group ID, names, and assigned responsibilities remain pending at the user's request; maintain actual contribution records throughout. |

Phases **1-7 are complete**, including the verified scripts, frozen experiments and genuine self-contained notebook reproduction. **Phase 8 is next:** use the `artifacts/phase7/clean_execution/run_artifacts/results.csv` (local archive) and `artifacts/phase7/clean_execution/run_artifacts/error_cases.json` (local archive) to prepare the report, consolidate genuine chat history, settle the pending group details and verify the exact three-file submission package.

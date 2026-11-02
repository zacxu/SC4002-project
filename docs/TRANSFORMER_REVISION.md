# Revised Transformer study

Prepared **2 November 2026 (Singapore time)** from [NLP_Final_project.ipynb](../NLP_Final_project.ipynb). The separate deliverable is [NLP_Final_project_revised.ipynb](../NLP_Final_project_revised.ipynb). The original notebook, executed BiLSTM notebook and root `run_artifacts/` are preserved.

## Diagnosis and evidence

The supplied Transformer-v2 notebook contains definitions and a Stage 1 training sequence, but no saved execution outputs, no Stage 2 training and no test results. Its historical 83.47% intent / 66.43% slot development result used evaluator v1 and cannot establish a current-v2 improvement. Fresh results must use the shared exact-character-span scorer.

| Observation | Evidence | Implemented response |
| --- | --- | --- |
| Train-unseen words all use one UNK vector, including words present in the external Word2Vec source | Original lookup uses train-only `word2id`; 464 development OOV tokens across 364 utterances, including 215 utterances with slot-related OOV | Full-character, padding-masked CNN features for every token |
| Slot labels are independent argmax decisions | Original model has a linear BIO head and token CE | BIO-constrained linear-chain CRF for likelihood and decoding |
| No explicit intent-to-slot information | Original task heads only share the encoder | Differentiable soft predicted-intent context; gold intent is never a forward input |
| Intent representation is only mean-pooled | Original forward code | Learned attention plus masked max pooling and an MLP head |
| Encoder blocks start with identical values | PyTorch clones one prototype; regression checked actual source tensors | Construct six blocks independently; parameters were already separate in the original model |
| Character convolution may select padding-only biases | Original optional CNN max-pools every convolution position | Pool only valid character centers; padded tokens produce zero features |
| Character features can overwhelm pretrained word features before projection | Initial near-cap prototype: character RMS 0.6233 versus word RMS 0.0500 on 128 original training examples, a 12.47× ratio | Separately normalize word and character branches before concatenation; zero padded features again after normalization |
| Capacity is about 4.35M and training lacks a scheduler/full evidence pipeline | Original model/count/config/training code | Larger encoder, regularization, warmup/cosine schedule and auditable train/select/reload/evaluate workflow |

There are **no** train/development words longer than 24 characters under this tokenizer (maxima 15/14). Removing the old cutoff prevents future truncation; it is not an explanation of an observed accuracy loss. No development character is absent from the training character inventory. These observations motivate spelling features but do not prove that OOV status causes errors.

The upstream [PyTorch implementation](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/nn/modules/transformer.py) explains the copied-layer initialization mechanism. [Ma and Hovy](https://aclanthology.org/P16-1101/) motivates character/CRF structure and [Goo et al.](https://aclanthology.org/N18-2118/) motivates intent-slot interaction. This revision is not a reproduction of those entire architectures and inherits none of their benchmark claims.

## Architecture and parameter budget

The baseline configuration reproduces the actual supplied model's parameter values and predictions under an identical seed and initialization matrix; this is covered by regression tests. The revised default is one jointly trained Transformer system:

| Setting | Original-size baseline | Revised default |
| --- | ---: | ---: |
| Train-fitted word embeddings | 5,300 × 300 | 5,300 × 300 |
| Encoder blocks / width / heads | 4 / 256 / 4 | 6 / 512 / 8 |
| Feedforward width | 768 | 1,792 |
| Dropout | 0.15 | 0.25 |
| Character features | None | 32-dimensional embeddings; kernels 3/4/5, 64 channels each → 192 |
| Intent pooling | Masked mean | Learned attention + masked maximum |
| Intent-to-slot context | None | 64 dimensions from predicted probabilities |
| Slot decoder | Independent BIO argmax | Constrained CRF |
| **Total / trainable parameters** | **4,345,243 / 4,345,243** | **19,748,595 / 19,748,595** |

All embedding rows, both task heads, character weights, context weights and masked CRF parameters count toward the cap. Only nonlearned buffers are excluded. The revised model uses 98.74% of the 20M allowance. The full external Word2Vec vocabulary is streamed solely for initialization and discarded; only the 1,590,000-parameter task embedding table participates in prediction. No pretrained encoder is used.

The initial 19,747,611-parameter prototype passed a finite-gradient, training-only GPU preflight on a batch of the 32 longest training examples. Peak **allocated** memory for that check was 444,884,480 bytes; this is not total GPU reservation, full AdamW workflow memory or a cross-hardware guarantee. The selected design adds 984 branch-normalization parameters; its full-workflow memory is recorded separately in the actual training summaries.

The first full prototype completed 20 epochs per system: baseline development intent/slot F1 **84.16% / 69.56%**, revised **83.92% / 75.98%**. That is a slot improvement with a small intent regression, not uniform superiority. The train-only branch-scale finding motivated one controlled fresh qualification with all other settings unchanged. The original prototype, registration and execution evidence remain under the ignored `artifacts/revised_transformer/qualification/` and related local records. Test was not used for either design or selection.

## Training and selection

Both systems use the identical task initialization matrix, original dataset and evaluator. The declared new common protocol is 20 full epochs, batch 32, AdamW, base LR 0.0005, word-embedding LR 0.0001, weight decay 0.01, clipping 1, 10% warmup and cosine decay to 10% of peak LR. The original 12-epoch unscheduled procedure is not claimed to have been exactly reproduced. Seed 42 and deterministic FP32 settings apply; each model starts afresh.

Loss is intent CE per utterance plus slot CE or sequence CRF NLL divided by real-token count. Counts/losses aggregate globally. Compare task metrics, not CE versus CRF loss heights. Development selection maximizes the unrounded mean of intent accuracy and slot F1; exact ties retain the earlier checkpoint. The complete configured budget finishes, and the winner is independently reconstructed in a fresh Python interpreter.

The default `evaluate_test=False` allows qualification without test inference. If final evaluation is enabled in a new chosen-configuration run, each model is frozen before one complete test pass; checkpoint bytes and weights must remain unchanged. A fixed single-seed comparison cannot establish statistical significance or identify which individual component caused a gain. Optional config flags permit separately registered ablations; do not use test outcomes for them.

## Run from a terminal

After the existing README environment/kernel setup, run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/execute_revised_transformer.py --execution-dir artifacts/revised_transformer/run_001 --dataset-cache data/1.0/data/en-US.jsonl --embedding-cache data/pretrained/nlpl_model_6.zip
```

Use a **new execution-directory name** for every run. Cache flags are optional: omitting them downloads the official dataset and the original permitted Word2Vec asset. The Word2Vec archive is approximately 635 MB. Source/dataset identity and the task initialization matrix checksum are recorded; missing vectors never silently switch to a random experiment. `embedding_source='random'` is an explicit alternative, not the default.

The executor copies only the revised notebook, starts the registered `sc4002` kernel with deterministic startup, and retains every actual output. A successful run updates its source notebook; failures retain the partial copy. `--cell-timeout -1` is the default. `--notebook <path>` executes a separately saved parameter variant.

## Run directly in Jupyter

Copy `NLP_Final_project_revised.ipynb` into a fresh directory. From that directory, launch the project environment with startup settings:

```powershell
$env:PYTHONHASHSEED='42'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& 'C:\path\to\SC4002\.venv\Scripts\python.exe' -m jupyterlab
```

Replace the example Python path with the actual repository location. Open the revised notebook, select **SC4002 (.venv)**, edit the labelled configuration cell, **save**, then **Restart Kernel and Run All**. Keep the filename unchanged and use its folder as the working directory. Optional cache paths belong in `RUN_OPTIONS`; otherwise acquisition is automatic. Every run exclusively creates `transformer_run_artifacts/` beside the notebook.

## Results and verification

The run directory contains the executed notebook and `transformer_run_artifacts/` with the configuration/data/embedding manifests, both checkpoints and histories, learning curves, fresh-process reload records, predictions, results CSV/JSON and genuine development error cases. Two result rows mean development-only; four include final test. Saved full-precision counts remain the source of displayed two-decimal percentages. Preserve regressions and failed hypotheses.

Run the independent read-only acceptance check after execution:

```powershell
.\.venv\Scripts\python.exe scripts/validate_revised_transformer.py --notebook artifacts/revised_transformer/run_001/NLP_Final_project_revised.ipynb --run-root artifacts/revised_transformer/run_001/transformer_run_artifacts --output artifacts/revised_transformer/run_001/acceptance.json
```

This auditor reads saved artifacts and CPU checkpoint tensors; it does not run another trained-model forward pass. Check `completion.json`, the audit and genuine outputs before treating a run as complete. No gain is claimed merely because more parameters were allocated.

The revision's 59 dedicated tests pass. Full-workspace discovery also flags 11 pre-existing checks against the current BiLSTM notebook's reviewed source/contract metadata. The original notebooks and existing BiLSTM result files remain byte-identical to their state at the start of this task; this revision does not silently relabel or rewrite that earlier evidence.

## Measured development qualification (2 November 2026)

These are genuine 20-epoch runs with common original data, initialization and scorer; no test model pass occurred.

| Model | Total/trainable parameters | Selected/trained epoch | Dev intent | Dev slot F1 | Workflow seconds |
| --- | ---: | --- | ---: | ---: | ---: |
| Original-size Transformer | 4,345,243 | 20 / 20 | 84.16% | 69.56% | 222.14 |
| Revised Transformer | 19,748,595 | 19 / 20 | 85.83% | 77.76% | 616.44 |

Revised minus baseline: **+1.67 intent points**, **+8.20 slot-F1 points**, and **+4.94 points** in their mean. This is a single-seed whole-system comparison. It does not prove statistical significance or which individual component caused the improvement.

Machine-readable measured outputs are in [results/revised_transformer/](../results/revised_transformer/). The original-size baseline had exactly the same saved model tensor hash in both qualifications, confirming the unchanged control.

## Remaining weaknesses after the revision

| Development group | Records | Baseline intent / slot F1 | Revised intent / slot F1 |
| --- | ---: | --- | --- |
| No-Train-OOV | 1669 | 87.18% / 75.90% | 89.10% / 83.71% |
| NonSlot-Train-OOV | 149 | 57.05% / 59.02% | 55.70% / 67.06% |
| Slot-Train-OOV | 215 | 79.53% / 34.13% | 81.40% / 43.70% |

On the same development records, the revision corrects 98 intents and regresses 64; it recovers 164 gold slot occurrences and loses 68. Fully correct utterances increase from 1189 to 1318. Non-slot-OOV intent accuracy still regresses, and slot-OOV F1 remains low. These descriptive groups do not isolate causal effects.

Two genuine remaining errors from the saved revised predictions:

- **Record 67:** `i'd like to hear queen's barcelona`. Gold is artist `queen's` and song `barcelona`; predicted intent is correctly `play_music`, but the model merges `queen's barcelona` into one artist span. **Inference:** legal BIO structure does not resolve adjacent entities with ambiguous semantics.
- **Record 30:** `cleaning is good dust is so bad do now your magic clean my carpet`. Gold is `iot_cleaning` with `house_place=carpet`; prediction is `general_quirky` without slots. **Inference:** long, loosely structured wording may obscure the action/domain; this example alone does not establish the cause.

Future work should compare independently registered capacity/character/CRF/context ablations and additional seeds using train/development only. More capacity did not remove unfamiliar-entity or intent ambiguity. No test results were used to design or choose this revision.

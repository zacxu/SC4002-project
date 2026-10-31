# Shared evaluation and data comparison contract

Updated **31 October 2026 (Singapore time)**. [main.ipynb](../main.ipynb) and [NLP_Final_project.ipynb](../NLP_Final_project.ipynb) use the same canonical [comparison_contract.py](../src/comparison_contract.py) source inline. The current evaluator is **`exact_character_span_v2`**. Fresh trained-model comparison results are pending; changing the scorer invalidates the old checkpoint-selection evidence as a current v2 comparison.

## Why change the evaluator

The BiLSTM's historical v1 scorer matched sets of `(slot type, literal surface text)`, collapsing repeated equal pairs and ignoring positions. That can credit a prediction at the wrong occurrence of repeated text. The Transformer previously used its own token-index BIO evaluator. Token boundaries differ between tokenizers, so raw token indices are not a common cross-model coordinate system.

The revised comparison scores **typed occurrences at exact original character boundaries**. It retains the useful entity-boundary requirement while providing one coordinate system for different tokenizations. This is an explicit project decision; the brief does not fully specify these details and no lecturer reference scorer has been provided.

## Exact metric

Every record has an immutable source utterance, one intent and zero or more slots. A slot contains `slot_type`, `text`, `start` and `end`:

- Offsets index the original Python Unicode string: zero-based start and **exclusive end**. They are not byte offsets or UTF-16 code-unit positions.
- A match requires equal type, start, end and literal text within the same record. Source slicing validates text/offsets; no case, punctuation, whitespace or Unicode normalization is applied.
- Two occurrences with identical type/text at different positions remain separate. A duplicate representation of the same occurrence is rejected rather than silently collapsed.
- Pool TP/FP/FN over records, then compute `2TP / (2TP + FP + FN)`. An empty slot denominator gives zero; an empty evaluated split is an error.
- Intent accuracy is exact label agreement per original record. Missing, unexpected or duplicate record IDs are errors. Scores remain full-precision fractions internally and percentages with two decimals for display.

Each model retains its own tokenization and predicts BIO tags. Both decoders map tags through ordered original token offsets, reconstructing `utt[start:end]`. The BiLSTM retains its typed `decode_bio`; the Transformer uses `bio_to_character_spans`. Their conversion semantics agree, while the data/scoring contract source is byte-identical. An orphan/type-changing `I` starts a new span for both implementations; padded tokens do not enter the decoder. A wrong type or boundary yields no partial credit. Aggregate scores cannot match spans across different utterances.

The data parser and both notebook adapters provide the original utterance for source-slice validation, including the BiLSTM typed-prediction sidecar. The low-level scorer also accepts older typed records without an utterance field for script compatibility; those inputs still require internally consistent offsets/text, and exact keys include literal text. They are not permission to replace the notebook's source-string validation with normalized or reconstructed text.

## One verified dataset

Both notebooks acquire/reuse the exact official original MASSIVE 1.0 English JSONL, rather than accepting an arbitrary CSV with plausible counts. The canonical loader pins:

```text
Official archive member: 1.0/data/en-US.jsonl
English SHA-256: c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e
Train / development / test: 11514 / 2033 / 2974
```

It preserves original row order and every record field; checks unique IDs, English locale, partitions, required field types and exact annotation reconstruction; and emits a manifest containing the evaluator version, source URL, file digest, counts, ordered split-ID digests and complete split-content digests. Both notebooks must agree on that manifest. The public [comparison audit](../results/comparison_contract.json) records the shared data/source identity; rerun `python scripts/check_comparison_contract.py` from the repository root to check the two notebooks against the official local file without training or downloading pretrained vectors. Matching row counts alone is insufficient.

An existing file with mismatched bytes is rejected without overwrite. New data comes from the [official archive](https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz); only its exact English regular member is published after verification. The recorded digest is this project's observed reproducibility identity, not an independently published authenticity certificate. No rows are added, removed, relabelled or moved between partitions.

Only train fits word/character mappings and task-label inventories. Both use the 60 training intents and the same complete **111-tag BIO schema**: `O` plus `B`/`I` for each of 55 training slot types. Constructing both BIO forms from training slot types does not inspect held-out tags. The Transformer no longer scans all partitions to build its slot head.

## Evidence and limitations

The upstream [MASSIVE training utility](https://github.com/alexa/massive/blob/main/src/massive/utils/training_utils.py) converts slot sequences to BIO and calls `seqeval.metrics.f1_score`. This supports evaluating structured entities rather than deduplicated surface strings. It does **not** establish that this project's original-character adapter reproduces every upstream preprocessing choice.

The [seqeval documentation](https://github.com/chakki-works/seqeval) distinguishes default and strict scheme-aware evaluation and shows that malformed BIO sequences can produce different scores. Therefore this project specifies its decoder behavior explicitly; it does not claim strict-IOB2, upstream or lecturer-scorer parity. These sources inform the rationale, while the shared source/tests define the actual project contract.

The two implementations still differ in tokenization, word-vector initialization, character treatment and optimization/budgets. The Transformer uses allowed pretrained word vectors; the BiLSTM does not. A shared metric and dataset support a meaningful implementation comparison, but cannot isolate the causal effect of the encoder architecture alone. Complete parameter totals, including embeddings, remain below 20M in every stage.

## Verified code-contract revision

The current checks passed **199 full-workspace tests** and **97 portable offline tests**. All 49 notebook code cells compile, no old outputs remain in the active templates, both inline source copies match, and both original-data adapters round-trip all 16,521 gold annotations with identical data manifests. Transformer collation/evaluator fixtures cover the complete 2,033-record development split for both character-feature modes. The BiLSTM rerun workflow additionally passed an independently audited one-epoch, small-model development-only run, with zero test model passes. The [public audit](../results/comparison_contract.json) records source/data contract evidence, separate from trained-model execution acceptance.

These checks do not supply a new development-selected checkpoint, model performance result or executed-notebook acceptance. Those require the fresh full runs described below.

## Rerun and reporting boundary

For future BiLSTM runs, follow [RUNNING_BILSTM.md](RUNNING_BILSTM.md). The notebook
now has one editable configuration cell; supported hyperparameter changes need
no rebuild or manual hash edits. Saved settings receive their own configuration
and effective source identities. `evaluate_test=False` performs development-only
tuning; `True` also freezes and evaluates the selected model on test. Every run
uses a fresh directory/kernel and writes its own results. Model/evaluator code
outside that cell remains protected.

The original executed notebooks and associated evidence were copied, with checksums, to the ignored `artifacts/evaluator_v2/prechange/` archive before edits. Public old results, plots, selected epochs and error analyses remain labelled **historical evaluator v1**. The active notebooks have cleared execution outputs; no new full model-training run or final model test evaluation was performed to validate this code-contract revision. Small fixture/contract checks are correctness evidence rather than new performance experiments.

Changing slot F1 changes the development selection objective even if model layers remain unchanged. Rerun complete training and development selection for each implementation/stage, save/reload the new winner, reproduce development results, freeze, then perform the required final test evaluation. Do not select among old checkpoints using test scores, reuse old results as v2, or treat one rescored old checkpoint as proof of newly selected performance.

Before reporting a new comparison, retain both manifests, evaluator/source identity, parameter counts, actual training/selection logs, independent reload checks, frozen final results and genuine error cases. Current scores remain **pending** until those executions pass acceptance. The v2 BiLSTM builder uses [evaluation_v2_manifest.json](../configs/evaluation_v2_manifest.json) for the current configuration/source identity; the historical Phase 6 reference and old trained checkpoints are not current v2 build/selection evidence. [PROJECT_STATUS.md](PROJECT_STATUS.md) distinguishes this pending work from completed historical phases.

# Assignment requirements

Source: [SC4002 Group Assignment Instruction](assignment.pdf), all five pages, and the user's three additional model clarifications. This document separates the brief's requirements from implementation choices. Detailed development history is in [PROJECT_STATUS.md](PROJECT_STATUS.md).

## Purpose and stages

Build an English assistant-request understanding system that predicts both **one of 60 intents** and a set of **slot type/span pairs from 55 slot types** for every utterance.

- **Stage 1:** develop, train, select and evaluate a credible baseline. The brief requires this stage without generative AI assistance.
- **Stage 2:** collaborate with AI to diagnose weaknesses, implement justified improvements, train, select and evaluate the resulting system. Document actual suggestions, verification, accepted/modified/rejected ideas, and the group's contributions.

**Actual AI history:** the user requested Stage 1 assistance despite the original restriction. Assistance occurred, including implementation and review. This request is not a lecturer-approved exception. The notebook discloses the history; the final collaboration file must do the same. Human group details/contributions remain pending at the user's request.

The brief prescribes no particular architecture. The improved system may refine or replace the baseline. A deliberately weak baseline is discouraged: both absolute performance and efficiency are assessed.

## Data and permissible information

Use the **original MASSIVE 1.0 English release**, archive `amazon-massive-dataset-1.0.tar.gz`, member `1.0/data/en-US.jsonl`.

| Partition | Records | Permitted use |
| --- | ---: | --- |
| `train` | 11,514 | Fit vocabulary and model parameters |
| `dev` | 2,033 | Tune settings and select checkpoints |
| `test` | 2,974 | Final evaluation after freezing |
| Total | 16,521 | Preserve every original example and partition |

Do not re-split, add, remove, augment or replace examples. Do not use external or AI-generated task examples, other datasets/locales, or models already trained/fine-tuned on MASSIVE or related tasks.

The only predictive input is **`utt`**. `intent` supplies the intent target; `annot_utt` supplies slot targets. IDs, partitions, annotation offsets and metadata support loading/auditing rather than model features. Train-only fitting also applies to preprocessing vocabularies. Full annotations may be checked structurally; test inspection must not become model diagnosis or tuning.

Official sources: [original archive](https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz), [MASSIVE documentation](https://github.com/alexa/massive).

## Model rules, including additional clarifications

1. Each stage uses **one joint framework/model for both tasks**. Two separate task-specific models are prohibited. Multiple output heads within a jointly trained model are permitted by this implementation interpretation.
2. The **entire system for each stage** must be within **20,000,000 total parameters**, including word embeddings, all heads, other learned components and frozen parameters. Stages may use different models; the cap applies separately to each complete stage system. Print total and trainable counts.
3. Pretrained **word vectors** are allowed; pretrained **models** are prohibited. The BiLSTM uses no pretrained assets; the Transformer uses permitted pretrained word vectors with a freshly trained encoder.

## Metrics and selection

Report all performance metrics as percentages with **two decimal places**:

- Intent accuracy: `100 × correct intents / utterances`.
- Slot micro F1: exact type/span matching pooled across examples, `100 × 2TP / (2TP + FP + FN)`.

Use identical scoring for every experiment. The brief does not resolve duplicate identical slots, offsets versus surface text, normalization or zero-denominator behavior. The current comparison contract, **`exact_character_span_v2`**, compares typed occurrences at exact original Unicode character offsets, with end-exclusive ends and validated literal text. Distinct occurrences remain distinct even when their text is identical; duplicate representations of one occurrence are rejected. Empty slot denominators give F1 zero. The earlier type/text-set convention is historical v1. No lecturer reference scorer was supplied; parity is unverified. See the [comparison contract](EVALUATION_COMPARISON.md) and [data and scoring](MODEL_AND_EVALUATION.md#data-and-scoring).

The brief permits intent accuracy, slot F1 or a combined development score for checkpoint selection. This project uses their **unrounded equal-weight mean**, with earlier checkpoints/candidates retained on exact ties. This is an implementation decision, not a lecturer-mandated formula. Test scores must not influence selection or subsequent tuning.

## Required notebook evidence

`main.ipynb` must contain all required source for both stages and retain real outputs:

1. Load official data, print and assert all three original split counts.
2. Implement preprocessing and one shared evaluator.
3. Explain each architecture, input/output, loss and optimizer; print the complete parameter count and enforce the cap.
4. Train on train only. Retain epoch-level train/development losses, both development metrics, elapsed time and learning curves.
5. Save the best-development checkpoint; print its epoch, score and filename.
6. Reload that checkpoint without further training, reproduce its selected development score, and evaluate the complete test split.
7. Repeat the training/selection/reload/final-evaluation process for Stage 2 after documented AI collaboration.

Do not type scores manually, fabricate logs or clear outputs before submission. The brief states that detected fabrication results in zero marks. A notebook that only displays historical score files does not establish training/reload evidence. The previous BiLSTM notebook completed this execution under evaluator v1. The active BiLSTM and Transformer notebooks have been revised for evaluator v2 and their old outputs cleared; they require fresh training, development selection, checkpoint verification and frozen final evaluation before they can supply current submission evidence.

## Report and collaboration record

`report.pdf` has **at most two content pages, excluding the cover**. Only the cover is explicitly exempt; conservatively include references and appendices within the two content pages unless the lecturer clarifies otherwise. Include:

- Baseline and improved pipelines, what changed and why.
- Four comparison rows: Stage 1 validation/test and Stage 2 validation/test, with total/trainable parameters, both percentage metrics and epoch information. Preserve total trained epochs separately from the selected checkpoint epoch.
- AI tool/model, weaknesses identified, suggestions accepted/modified/rejected, advice verification, genuine human reasoning/contributions and unsuccessful experiments.
- Lessons, limitations, at least **two actual final-system error cases**, likely causes labelled appropriately, and evidence-based future work.

The cover must contain the group ID, every member's full name and each person's actual contributions. Do not invent contributions. Submit one genuine `ai_collaboration.md` or `.txt` containing chat history and context. Retain the original history; the brief does not specify whether excerpts suffice. Working summaries alone do not demonstrate the actual exchange.

## Administration and submission

| Item | Requirement |
| --- | --- |
| Group size | 5–6 students |
| Course weighting | 30% |
| Deadline in the brief | Sunday, 15 November, 11:59 pm SGT |
| Year interpretation | 15 November **2026**, 23:59 SGT, from AY26–27 Semester 1; the deadline sentence omits the year |
| Late penalty | “5% point deduction each calendar day” after the deadline; grading basis is not further defined |
| Attempts | Advised not more than three; only the last upload is graded/time-stamped |
| Destination | NTULearn → Group Assignment Submission |
| Filename | `Group_<ID>.zip` |
| ZIP contents | Exactly `main.ipynb`, one AI collaboration `.md`/`.txt`, and `report.pdf` |
| Exclusions | Dataset, environments, caches, checkpoints, development files and API credentials |
| Group grading | All group members receive the same grade |

The assessment weights are performance/efficiency **50%**, critical thinking **10%**, code clarity/reproducibility **10%**, and report quality **30%**. A later official announcement takes precedence; NTULearn has not been checked for updates.

The historical Phases 1-7 completed under evaluator v1. The current evaluator-v2 comparison requires fresh notebook execution and acceptance before final comparative scores or a report can be produced. Report, consolidated chat export, verified cover information, ZIP and submission remain **Phase 8** work. The two model-family notebooks are comparison implementations; the assignment submission still requires one final `main.ipynb`.

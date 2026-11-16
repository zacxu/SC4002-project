# Final submission check

Verified against the supplied **Group Assignment Instruction.pdf** on 11 November 2026. The notebook and report pass the technical/content checks below. Group size and individual cover contributions are excluded at the user's explicit request; those two original requirements are not certified by this check. The user confirmed a lecturer-approved exception for Stage 1 AI assistance.

## Files to submit

Submit **[Group_71.zip](../Group_71.zip)**. It contains exactly:

- `main.ipynb`: a byte-identical renamed copy of the final [Transformer notebook](../NLP_Final_project_revised_early_stopping.ipynb), including genuine outputs.
- `ai_collaboration.md`: 278 verified public conversation messages, original project notes and a labelled final preparation record.
- `report.pdf`: the unchanged cover plus two report pages, exported from [the editable report](../SC4002%20Project%20Report.docx).

No dataset, environment, cache, checkpoints, credentials or extra files are inside the archive. ZIP integrity and all three member hashes were checked. The notebook's bootstrap was separately exercised with only the required `main.ipynb` filename present.

## Requirements and evidence

| Requirement | Result |
| --- | --- |
| One joint framework predicts 60 intents and 55 slot types in each stage | Passed; shared PyTorch encoder and two task heads |
| Original MASSIVE 1.0 English release and original partitions | Passed; pinned source bytes and 11,514 / 2,033 / 2,974 examples asserted and printed |
| Only `utt` supplies predictive inputs; train-only vocabularies/labels | Passed; original token offsets, training-fitted word/character/schema mappings |
| No added external or AI-generated task examples | Passed; all original records retained |
| Pretrained word vectors allowed; no pretrained encoder/task model | Passed; task Word2Vec table only; encoder weights initialized from scratch |
| Entire model, including embeddings, below 20M | Passed; Stage 1 4,345,243 and Stage 2 19,748,595 total/trainable parameters |
| Same intent accuracy and exact-span slot micro-F1 evaluator | Passed; canonical evaluator identical in both stages |
| Genuine epoch training/validation loss and metric logs | Passed; 20 Stage 1 and 28 Stage 2 epochs visible in executed notebook |
| Elapsed time, learning curves, parameter count and selected checkpoint details | Passed; real timing/path/epoch/score prints, counts and two learning-curve images |
| Select best validation checkpoint and reproduce its validation score | Passed; both select epoch 20 and independent fresh-process reload matches predictions |
| Freeze before final validation/test reporting | Passed; freeze/test-start records and unchanged learned state/checkpoint bytes |
| Required four-row comparison, with two-decimal percentages | Passed; generated notebook table and report agree exactly |
| All source embedded; portable filename/cache configuration | Passed; 35 executed code cells, no repository imports, `main.ipynb` fallback, caches set to `None` |
| No manually fabricated scores/logs or cleared final outputs | Passed; final fresh execution and independent acceptance retained |
| Baseline independence rule | Lecturer-approved exception reported by the user; both notebook and report disclose AI assistance |
| Report covers systems, changes, AI tool/models, accepted/modified/deferred advice and group reasoning | Passed; Codex session models `gpt-6-astra` and `gpt-6.1-sol` identified |
| Analysis/lessons, unsuccessful experiments, at least two errors and limitations/future work | Passed; controlled regularization study, no-gain extension, two genuine validation cases and cautious explanations |
| Report maximum two pages excluding cover | Passed; three rendered pages total, two body pages |
| First report page unchanged | Passed; cover rendered pixels and original cover XML prefix are identical; all other DOCX package parts unchanged |
| Group ID and supplied names on cover | Group 71 and supplied names preserved |
| Group size of 5-6 | Excluded at user request; no approval is inferred |
| Individual contributions on cover | Excluded at user request; original cover fields preserved |
| Archive name and exactly three required members | Passed; `Group_71.zip` |

## Final measured results

| System | Split | Total parameters | Trainable parameters | Intent accuracy (%) | Slot micro-F1 (%) | Trained epochs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Stage 1 | Validation | 4,345,243 | 4,345,243 | 84.16 | 69.56 | 20 |
| Stage 1 | Test | 4,345,243 | 4,345,243 | 82.92 | 66.77 | 20 |
| Stage 2 | Validation | 19,748,595 | 19,748,595 | 86.62 | 80.84 | 28 |
| Stage 2 | Test | 19,748,595 | 19,748,595 | 85.44 | 77.79 | 28 |

The final run reproduced the already selected development models exactly. Each model has one test pass in this recorded workflow. The official test split had been exposed in an earlier comparison; neither notebook nor report claims a newly untouched test set, and no model changes were made from the final test scores.

Independent notebook acceptance: `artifacts/final_submission/full_notebook/acceptance.json`. Portable machine-readable audit and file hashes: [results/final_submission/verification.json](../results/final_submission/verification.json).

## External submission step

The brief requires uploading `Group_71.zip` through NTULearn by **Friday, 13 November 2026, 11:59 pm SGT**. It advises no more than three submissions and grades/time-stamps the last one. No external upload was performed by this workflow.

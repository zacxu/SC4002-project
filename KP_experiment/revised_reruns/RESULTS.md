# Reruns of NLP_Final_project_revised.ipynb

Seed 42, 20 epochs, RTX 4060 Laptop GPU. All choices were made on dev. Word2Vec Stage 2, with and without SAM, then had a one-time test evaluation; every other rerun score here is dev only. "Joint" is the mean of intent accuracy and slot micro-F1, the notebook's selection metric.

## Short answer

- **SAM helps Stage 2 and hurts Stage 1.** With Word2Vec, Stage 2 joint rises from 81.58 to 82.80. Stage 1 falls on all three embeddings.
- **fastText helps without SAM, GloVe does not.** In Stage 2, fastText gains 1.12 joint and GloVe is a tie.
- **Once SAM is on, the embedding makes no measurable difference.** fastText + SAM and GloVe + SAM both reach 82.87, against 82.80 for Word2Vec + SAM.
- **Best setup within the assignment rules: Word2Vec + SAM, Stage 2.** It ties the best score and uses only the permitted embedding.
- **On test it scores 85.91 intent and 76.41 slot F1, joint 81.16.** The KP strict single stays ahead on both dev and test, so it remains the primary result.
- **SAM helps on test too.** Against the same model without SAM, it gains 1.01 joint [+0.35, +1.67], close to its 1.22 gain on dev.

## One-time test: Word2Vec Stage 2, with and without SAM

Scores for Word2Vec + SAM:

| Test split | Utterances | Intent | Slot F1 |
|---|---:|---:|---:|
| Overall | 2,974 | 85.91 [84.67, 87.19] | 76.41 [74.82, 78.03] |
| No unseen word | 2,411 | 88.97 | 83.09 |
| Unseen word outside the slots | 222 | 63.51 | 63.28 |
| Unseen word inside a slot | 341 | 78.89 | 47.51 |

Joint is 81.16 [80.07, 82.28] and exact match is 65.60. Brackets are 95% bootstrap intervals over test utterances.

| System | Dev joint | Test intent | Test slot F1 | Test joint | Test slot F1, unseen word inside a slot |
|---|---:|---:|---:|---:|---:|
| KP strict single (primary) | 85.49 | 86.42 | 79.01 | 82.72 | 62.99 |
| KP flagged single (needs approval) | 85.29 | 86.58 | 79.38 | 82.98 | 64.59 |
| **Notebook Stage 2, Word2Vec + SAM** | 82.80 | **85.91** | **76.41** | **81.16** | **47.51** |
| Notebook Stage 2, Word2Vec, no SAM (r0) | 81.58 | 84.84 | 75.47 | 80.15 | 40.00 |
| KP Stage 1 | 74.46 | 81.57 | 63.23 | 72.40 | 29.90 |

SAM's effect on test, Word2Vec + SAM minus r0. The bootstrap draws the same utterances for both models:

| Test split | Utterances | Intent | Slot F1 | Joint |
|---|---:|---:|---:|---:|
| Overall | 2,974 | +1.08 [+0.23, +1.95] | +0.94 [-0.02, +1.91] | **+1.01 [+0.35, +1.67]** |
| No unseen word | 2,411 | +0.66 [-0.17, +1.49] | +0.37 [-0.52, +1.32] | +0.52 [-0.12, +1.18] |
| Unseen word outside the slots | 222 | +3.60 [-1.35, +8.56] | -3.95 [-10.38, +2.43] | -0.17 [-4.19, +3.94] |
| Unseen word inside a slot | 341 | +2.35 [-0.88, +5.57] | **+7.51 [+3.87, +11.30]** | **+4.93 [+2.52, +7.33]** |

- **SAM on test:** the gain of 1.01 joint matches dev in direction and size (+1.22). It comes mostly from utterances with an unseen word inside a slot. One seed can't show seed noise, and SAM's Stage 1 loss on dev wasn't tested.
- **Unseen-word gain:** +7.51 on test, about half of the +14.71 on dev. This fits the earlier warning that r0's low dev score in this group was partly noise.
- **Dev to test:** joint falls 1.64 points with SAM, from 82.80 to 81.16, and 1.43 without, from 81.58 to 80.15. Some fall is expected, because the epoch was chosen on dev.
- **Against the KP strict single:** 1.56 joint lower on test and 2.69 lower on dev, so the two splits agree. No paired significance test was run on this gap.
- **Scorers:** the ladder scores exact token spans and the notebook exact character spans. The unseen-word groups have the same sizes in both.
- **Method:** the notebook's own evaluation functions ran on each saved dev-chosen checkpoint, after it reproduced its dev predictions exactly. Nothing was retrained and Stage 1 was not tested, whereas the notebook's default route is a new full run that tests both stages.
- **Test discipline:** `test_r0_stage2/` and `test_r1_stage2/` block a second run of each, and the ladder's lock in `results/final/` was not touched. Test had already been seen for the KP systems, but SAM and rho 0.2 were chosen on dev. Nothing was selected from these test scores.

## All runs

| Run | Embedding | SAM (rho 0.2) | Stage 1 intent | Stage 1 slot | Stage 1 joint | Stage 2 intent | Stage 2 slot | Stage 2 joint |
|---|---|---|---:|---:|---:|---:|---:|---:|
| r0 (control) | Word2Vec | no | 84.90 | 69.18 | 77.04 | 85.93 | 77.24 | 81.58 |
| r1 (step 1) | Word2Vec | yes | 85.69 | 66.08 | 75.89 | 86.87 | 78.74 | **82.80** |
| r2 (step 2) | fastText | no | 86.23 | 71.57 | **78.90** | 86.23 | 79.17 | 82.70 |
| r3 (step 2) | GloVe | no | 86.57 | 70.85 | 78.71 | 86.23 | 76.07 | 81.15 |
| r4 (step 3) | fastText | yes | 86.82 | 68.49 | 77.65 | 86.33 | 79.42 | **82.87** |
| r5 (step 3) | GloVe | yes | 86.08 | 66.73 | 76.40 | 86.87 | 78.88 | **82.87** |

Reference, the notebook's recorded run on an RTX 3060 Ti: Stage 1 84.16 / 69.56, joint 76.86; Stage 2 85.83 / 77.76, joint 81.80. The control r0 is within about 0.2 joint of it on both stages, so the setup reproduces the notebook.

## Differences in joint score

The brackets are 95% paired bootstrap intervals over dev utterances. They show how much a gap depends on which utterances are in dev, not how much it depends on the seed.

| Comparison | Stage 1 | Stage 2 |
|---|---:|---:|
| SAM on Word2Vec | -1.15 [-2.07, -0.22] | **+1.22** [+0.46, +2.01] |
| SAM on fastText | -1.25 [-2.19, -0.33] | +0.17 [-0.63, +0.95], tie |
| SAM on GloVe | -2.31 [-3.25, -1.37] | **+1.72** [+0.87, +2.58] |
| fastText vs Word2Vec, no SAM | +1.86 [+0.83, +2.92] | **+1.12** [+0.21, +2.04] |
| GloVe vs Word2Vec, no SAM | +1.67 [+0.61, +2.80] | -0.43 [-1.43, +0.57], tie |
| fastText + SAM vs Word2Vec + SAM | +1.76 [+0.75, +2.78] | +0.07 [-0.81, +0.94], tie |
| GloVe + SAM vs Word2Vec + SAM | +0.51 [-0.54, +1.59], tie | +0.07 [-0.84, +0.96], tie |
| fastText + SAM vs baseline r0 | +0.61 [-0.44, +1.70], tie | **+1.29** [+0.38, +2.23] |
| GloVe + SAM vs baseline r0 | -0.64 [-1.72, +0.51], tie | **+1.29** [+0.28, +2.27] |

Ties follow the project rule: one-seed joint gaps under 0.8 are ties. The Stage 2 systems r1, r2, r4 and r5 (82.70 to 82.87) all tie one another.

## Unseen words (Slot-Train-OOV group, 215 dev utterances)

| Slot F1 in this group | r0 Word2Vec | r1 Word2Vec + SAM | r2 fastText | r3 GloVe | r4 fastText + SAM | r5 GloVe + SAM |
|---|---:|---:|---:|---:|---:|---:|
| Stage 1 | 33.21 | 33.75 | 48.25 | 48.35 | 42.17 | 40.18 |
| Stage 2 | 38.82 | 53.53 | 59.31 | 53.18 | 60.48 | 57.95 |

This is where the runs differ most, but read it with care:

- **The metric is noisy.** The same code on the notebook's recorded machine scored 43.70 in Stage 2, against 38.82 for r0 here. A gap of about 5 points can come from the machine alone, so r0 may understate the baseline.
- **The new vectors do not reach the unseen words.** Every unseen word still maps to `<UNK>`, and that row is never trained: its norm is 0.89 from start to end in every run.
- **The cause is not established.** One difference to test: with fastText and GloVe the UNK row is 5 to 8 times smaller than a real word's vector (norm 0.89 against 4.1 and 6.8), while with Word2Vec it is about the same size (0.87).

## Limits of these results

- **fastText and GloVe are outside the assignment rules.** Only NLPL Word2Vec model 6 is allowed without lecturer approval, so r2 to r5 cannot be the primary result.
- **One seed.** The bootstrap does not capture seed noise.
- **rho 0.2 was tuned on the BiLSTM.** It was not retuned for this Transformer, so the Stage 1 loss may be partly a poor rho for that model. Stage 1 slot F1 has levelled off by epoch 18, so more epochs would probably not close the gap.
- **Vector scale differs.** Mean norms are 0.87 (Word2Vec), 4.10 (fastText) and 6.84 (GloVe). Vectors were used raw, and Stage 1 has no input normalisation, so its gains across embeddings mix vector quality with scale. Stage 2 has input normalisation.
- **Stage 2 was still improving slightly.** r0, r1 and r2 selected their last epoch.
- **Coverage of train words:** Word2Vec 89.3%, fastText 95.0%, GloVe 94.5%.
- **Not a source for the Stage 1 submission.** The assignment requires Stage 1 to be written without generative AI. The SAM patch, which also changes Stage 1 training, and everything in `tools/` were written with AI assistance.

## Cost

- SAM takes about 1.8 times as long: Stage 1 about 130 s becomes about 240 s, and Stage 2 about 465 s becomes about 845 s.
- Peak GPU memory in Stage 2 rises from 638 MB to 717 MB. Parameter counts are unchanged: 4,345,243 and 19,748,595.
- The full batch took 1 h 37 min. That includes 6.5 minutes of low-RAM safety pauses in r4 and r5, which also count in those runs' `seconds` column in the CSV.

## What was verified

- With SAM off, the patched code matched the unmodified notebook exactly on a one-epoch run: same weights, predictions and metrics.
- The second SAM pass restores the weights and changes the gradient.
- The Word2Vec matrix and archive hashes match the notebook's recorded values.
- All 12 checkpoints passed the notebook's reload check in a fresh process, with identical predictions.
- `collect_results.py` recomputed every dev score independently from the predictions, and all match the notebook's. Its group sizes (1669 / 149 / 215) match the notebook's. The notebook does not compute per-group scores in the run, so those were not cross-checked within a run.
- `evaluate_test` was false in all six training runs. The only test outputs from the reruns are `test_r0_stage2/` and `test_r1_stage2/`, and their scores were recomputed independently from the predictions and match the notebook's.
- The official `en-US.jsonl` matches `massive_enUS_full.csv` exactly (16,521 rows), which closes the hand-off's "Official data" item.

## Plots

- `plots/all_dev_results.png`: every dev result together, the KP ladder (103 families and 7 rung-E candidates), the notebook's recorded run and the reruns
- `plots/reruns_sam_embeddings.png`: the SAM effect on each embedding and stage
- `plots/systems_dev_test.png`: intent, slot F1 and unseen-word slot F1 for the final systems, with all test scores
- `plots/all_results.csv`: the table behind all three

KP ladder numbers come from `massive_joint_nlu/results/` on the repo's `Khawpuneiei-exp-1` branch (`leaderboard.csv`, `posthoc_leaderboard.csv` and `final/final_comparison.csv`), which the `KP_experiment` summary documents. The notebook's recorded numbers come from the repo's `results/revised_transformer`. The repo's `results/historical_v1` scores are left out because they used the older scorer, which matches type and text without positions.

## Files

- `results_summary.csv`: every dev metric for every run and stage
- `test_r0_stage2/`, `test_r1_stage2/`: each one-time test's predictions, evaluation and report
- `test_sam_effect.json`: SAM's paired effect on test
- `plots/`: see above
- `logs/analysis.txt`: bootstrap, learning curves and UNK row figures
- `tools/`: `build_reruns.py`, `run_reruns.py`, `verify_patches.py`, `compare_smokes.py`, `collect_results.py`, `analyze_results.py`, `run_test_stage2.py`, `plot_all_results.py`

These stay on the machine that ran the reruns and are not in the repo:

- `runs/<name>/`: patched notebook, `executed.ipynb`, artifacts and checkpoints. Model weights are never published.
- `logs/full_runs.log`: run timings
- The embedding vectors, the dataset and the Python environment

To regenerate the tables and plots, run these from `revised_reruns` on that machine, because they read `runs/` and the ladder's results:

```bash
.venv/Scripts/python.exe -I -X utf8 tools/collect_results.py
```

```bash
.venv/Scripts/python.exe -I -X utf8 tools/analyze_results.py
```

```bash
.venv/Scripts/python.exe -I -X utf8 tools/plot_all_results.py
```

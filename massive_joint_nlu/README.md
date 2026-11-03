# MASSIVE en-US: joint intent classification and slot filling

Experiment summary for the NTU SC4002 group assignment. One joint model predicts the intent and the BIO slot tags of an utterance. The project runs a Stage 1 baseline, a ladder of Stage 2 experiments, a post-hoc decoding step, and one final test evaluation. This README records every experiment, the decisions taken, and the results. Last updated 2026-11-03.

## 1. Result

Test split, one evaluation, on the saved checkpoints (`results/final/final_comparison.csv`):

| System | Total params (stored) | Trainable | Intent acc. | Slot F1 | Slot-OOV F1 | Best / run epochs |
|---|---|---|---|---|---|---|
| Stage 1 (Transformer, hand-off configuration) | 4.34M (4.37M) | 4.34M | 81.57 | 63.23 | 29.90 | 11 / 12 |
| **Strict single**: R1_sam.2, seed 7 (BiLSTM, CRF head, span decoding). Primary. | 5.64M (5.67M) | 5.63M | **86.42** | **79.01** | **62.99** | 22 / 30 |
| Flagged single: D2f, seed 7 (Transformer, extended Word2Vec vocabulary). Needs lecturer approval. | 19.14M (19.17M) | 5.72M | 86.58 | 79.38 | 64.59 | 23 / 31 |

Stored values (in brackets) count every numeric value the system keeps at inference; see Section 3.

What the test results show:
- Against Stage 1, the strict single gains +4.84 intent accuracy (95% CI 3.63 to 5.95) and +15.78 slot F1 (14.04 to 17.49).
- On slots that contain a word unseen in training (Slot-Train-OOV), the slot-F1 gain is +33.10 (28.09 to 37.60), against +12.79 for utterances without such a word.
- A difficulty-matched control (utterance length and slot count) shrinks the OOV gap from 40.2 points (Stage 1) to 21.1 (strict single).
- Intent accuracy of 86.4% is well short of the 95% target. Without a pretrained language model and under the 20M cap, the remaining errors are mostly unseen-word and ambiguous-intent cases.
- The flagged track is +0.37 slot F1 above the strict track on test, but 0.20 lower on dev joint. The difference is not tested for significance and gives no reliable advantage, so the strict track is the primary result.

## 2. Task, data and constraints

- **Data:** MASSIVE 1.0 en-US with the official splits: train 11,514, dev 2,033, test 2,974 utterances. 60 intents. 55 slot types, written as 108 BIO labels. The experiments read `massive_enUS_full.csv`, a verified copy of the release. The official `en-US.jsonl` was not available on this machine.
- **Label inventory:** one BIO label (`I-transport_type`) occurs only in the test split. It was inherited from the hand-off's label construction. No training example has it, and no dev number is affected.
- **Embeddings:** NLPL Word2Vec model 6, 300-d, the only pretrained resource. Of the 5,300 training words, 4,731 (89.3%) have a pretrained vector.
- **Constraints:** no pretrained language model. Every stored value of the system, including embeddings and decode tables, stays at or below 20,000,000. Dev is used for all selection. Test is evaluated once.
- **Metrics:** intent accuracy; exact-span slot micro-F1; joint = mean of the two. Subsets: No-Train-OOV (no lexical word unseen in training), NonSlot-Train-OOV (an unseen word outside the slots), Slot-Train-OOV (an unseen word inside a slot).
- **Hardware and software:** Windows 11, RTX 4060 Laptop (8 GB), Python 3.10, PyTorch 1.13.1 with CUDA 11.7. A 40-epoch BiLSTM run takes about 6 to 9 minutes; Stage 1 takes under a minute.

## 3. Protocol and decisions

- **Two tracks.** The **strict** track uses only the given data and the NLPL vectors. The **flagged** track adds frozen rows for Word2Vec words that never occur in training (the "extended vocabulary"). The flagged track needs lecturer approval. The strict track is reported as primary.
- **Rung protocol** (rungs A, B, C, D, R, H, F). Each rung screens its variants on seed 42. The top four with a gain of at least 0.15 over the base are re-run on seeds 42, 7 and 1234. Up to three winners are composed greedily. A change is kept only if the 3-seed mean improves.
- **Post-hoc tie margin** (rung E). A decode table, MC-dropout passes, an extra ensemble member, or an ensemble over a single model is kept only if it beats the plain choice by at least **0.8 dev joint points**. This is the one-seed tie margin from the project memory. It was applied to rung E after its first run, which had chosen noise-level variants. The rung protocol above has no such margin. See Section 9.
- **Stored-value accounting for the cap.** Counted: the numeric values of each member's checkpoint (weights and buffers, including positional tables and CRF masks), TF-IDF idf vectors once per feature size, BIO constraint tables once unless a CRF checkpoint already holds them, and the PMI prior and mention tables when a decode uses them. Not counted: string vocabularies and lexicon keys, as is standard for an embedding vocabulary. The parameter count alone (the official "total parameters") is reported separately.
- **Test lock.** `run_final_test.py --dev-only` checks that every system fits the cap and reproduces the dev numbers of rung E, and touches no test data. The one-time run repeats those checks, runs the post-test steps on dev as a dry run, and only then writes `results/final/TEST_EVALUATED.json`. A second test run needs `--rerun "reason"`.

## 4. Experiment ladder

Strict track unless noted. Dev joint is the mean over the seeds listed (n). Full per-family table: Appendix A.

| Rung | What was tested | Runs (families) | Best dev joint | Outcome |
|---|---|---|---|---|
| Stage 1 | Hand-off Transformer, 4 layers, 4.34M parameters, retrained | 1 (1) | 74.46 (1 seed) | Baseline |
| A | Training recipe: 40 epochs, cosine warm-up, label smoothing 0.1, word dropout 0.1; variants of learning rate, batch size, weight decay, dropout, slot-loss weight, rare-word weight, embedding learning rate, extra noise | 28 (20) | A3g2_slot2: 81.07 (3) | Adopted (recipe) |
| V (family A4_vit) | BIO-constrained Viterbi decoding used during epoch selection | (in A) | A4_vit: 82.69 (3) | Adopted (+1.62) |
| B | Word representations: TF-IDF wide intent path (10k features), character residual, BPE residual, n-gram residual, gazetteer, slot-word embeddings. Flagged: extended vocabulary of 10k, 20k, 40k and 100k frozen rows | 29 (17) | B1g2_char: 83.50 (3). Flagged: B2_ext10k 84.21 (1), B2_ext40k 84.18 (3) | Adopted: B1g2_char (+0.81) |
| C | Slot heads and interaction: CRF, BIOES CRF, task-specific top layer, refinement, factorised heads, biaffine span head, intent-slot interaction, label attention, presence filter. Flagged: C2f from B2_ext40k | 34 (22) | C1_crf: 84.14 (3). Flagged: C2f 84.43 (3) | Adopted: C1_crf (+0.64) |
| D | Encoders: BiLSTM with 256 or 384 units per direction; Transformers of several sizes. Flagged: D2f, a larger Transformer on the extended vocabulary | 12 (8) | D1_bilstm256: 84.33 (3). Flagged: D2f 84.85 (3) | Adopted: BiLSTM-256 (+0.19) |
| R | Regularisers: adversarial embedding noise (0.05, 0.2), R-Drop (0.25, 1), sharpness-aware training (rho 0.05, 0.2), EMA, consistency (0.25, 1), mixup, mention dropout, supervised contrastive term, L2-SP anchor, dual embedding. Flagged: mention replacement, LWTR and SIS augmentations, which add new token sequences recombined from training data | 27 (21) | R1_sam.2: 84.92 (3). Flagged: R2f 84.81 (3), R3_mention.3 84.71 (3) | Adopted: R1_sam.2 (+0.59, tentative) |
| H | Random search: 12 configurations around R, sampling dropout, learning rate, weight decay, label smoothing, word dropout, rare-word weight, slot dropout and slot-loss weight | 12 (12) | H_04: 84.73 (seed 42 only) | None promoted |
| F | 80-epoch schedule on the strict and the flagged base | 6 (2) | F_strict_ep80: 84.55 (3). F_flagged_ep80: 84.77 (3) | Rejected (bases 84.92 and 84.85) |
| E | Post-hoc decoding, MC dropout, mention tables, ensembles | (Section 6) | Strict single 85.49; flagged single 85.29 | Final systems |

Notes on the ladder:
- Stage 1 retrained here gives dev intent 82.83 and slot F1 66.09. The hand-off reported 81.90 and 66.60 for its own run. The gap comes from retraining.
- Most of the rise over Stage 1 comes from rungs A and V: from 74.46 to 82.69, +8.2 joint. From rung B on, each adopted change added between 0.19 and 0.81 points on the 3-seed dev mean. Those are real dev differences, but they are small. Dev numbers of selected systems are optimistic, because the selection was made on dev.
- The strict single is the R1_sam.2 seed-7 run with the decoding of rung E. Its architecture is the BiLSTM (2 layers, 256 per direction), the character residual, the TF-IDF wide intent path, the CRF head and sharpness-aware training, trained with the rung-A recipe.

## 5. What did not work

- **80-epoch schedule (F):** lower on both tracks (84.55 against 84.92 strict; 84.77 against 84.85 flagged). The best epochs were 16 to 41.
- **Random search (H):** none of the 12 configurations was promoted. The best reached 84.73 on seed 42.
- **Stronger adversarial noise (R1_adv.2):** 82.70. The weaker setting, 0.05, was not adopted either.
- **Other regularisers (R):** dual embedding 83.72; L2-SP anchor 84.04 to 84.18; mention dropout 84.05 to 84.07; mixup 84.14; supervised contrastive 84.22. None beat the base.
- **Transformers instead of the BiLSTM (D):** 83.68 to 84.12, below the BiLSTM's 84.33. The largest, at 13.7M parameters, was the weakest.
- **Word-representation variants (B):** BPE alone 82.89, n-grams alone 82.65, gazetteer 82.20. All below the character residual with the TF-IDF wide path (83.50).
- **Slot-head variants (C):** most were below the CRF (84.14). Examples: attention 83.00, hierarchical 82.96.
- **Mention replacement augmentation (R3):** 84.71 flagged, against 84.92 for the strict base it was added to.
- **MC-dropout inference (E):** at most +0.06 on dev, and it stays stochastic. Not kept.
- **Mention lookup table (E):** +0.02 for the strict single and +0.34 for the flagged single. Both below the margin, so dropped.
- **Two-member ensemble (E):** R1_sam.2 seed 7 with C1_tasktop reached 86.08 dev, against 85.49 for the single: +0.59, below the margin. A third member (R1_sam.05) added +0.19 on the default decode, so it was not added. A flagged ensemble does not fit under the cap, since the flagged single already uses 19.1M.

## 6. Post-hoc decoding and ensembles (rung E)

Dev numbers of the rung-E candidates, from `results/posthoc_leaderboard.csv`:

| Candidate | Decode | Dev intent | Dev slot F1 | Dev joint | Decision |
|---|---|---|---|---|---|
| Strict single | span, tau 0.50, intent prior 0.1, intent mix 0.65 | 88.15 | 82.83 | 85.49 | **Final** |
| Strict single + MC 8 passes | same | 88.15 | 82.86 | 85.50 | Rejected (+0.01) |
| Strict single + MC 32 passes | same | 88.05 | 82.42 | 85.24 | Rejected |
| Strict single + mention table | span, tau 0.60, mention weight 0.2 | 88.15 | 82.87 | 85.51 | Rejected (+0.02) |
| Strict 2-member ensemble (R1_sam.2 seed 7 + C1_tasktop) | span, tau 0.50 | 88.83 | 83.33 | 86.08 | Rejected (+0.59 against the single) |
| Flagged single | span, tau 0.55, intent mix 0.50 | 88.59 | 81.99 | 85.29 | **Final** |
| Flagged single + MC 8 passes | same | 88.69 | 81.95 | 85.32 | Rejected |
| Flagged single + MC 32 passes | same | 88.59 | 82.12 | 85.35 | Rejected (+0.06) |
| Flagged single + mention table | span, tau 0.35, mention weight 0.6 | 88.59 | 82.68 | 85.63 | Rejected (+0.34) |

The tau grid runs from 0.30 to 0.60. The strict single's search reached the top of the first grid (0.50), so the grid was widened to 0.60. The best table-based decode then sat at 0.60, but its mention table gained only 0.02 and was dropped. Without a table, the strict tau curve peaks at 0.50, inside the grid. The flagged single's table-free curve peaks at 0.55, also inside the grid.

## 7. Error analysis (strict single, test)

- 967 of 2,974 test utterances contain an error: 563 slot-only, 224 intent-only, 180 both.
- By subset: 645 of 2,411 No-Train-OOV utterances (27%), 108 of 222 NonSlot-Train-OOV (49%), 214 of 341 Slot-Train-OOV (63%).
- Examples of Slot-Train-OOV slot errors: "i like senatra songs" tags the unseen artist as music_genre. "play a retro song in my playlist" misses the unseen genre "retro" entirely.
- Some intent errors come from unseen words that carry no intent evidence: "send a giggle my way" predicts email_sendemail, where the gold is general_joke.

## 8. Reproducing

Run from the project folder. The rung runs are resumable. Each rung writes `results/leaderboard.csv` and a log under `results/`.

```bash
python run_stage1.py              # Stage 1 baseline -> checkpoints/stage1_baseline.pt
python run_ladder.py A            # recipe
python run_ladder.py V            # Viterbi decoding (family A4_vit)
python run_ladder.py B            # word representations (strict and flagged)
python run_ladder.py C            # slot heads
python run_ladder.py D            # encoders
python run_ladder.py R            # regularisers, then R3 augmentations
python run_ladder.py H            # random search around R
python run_ladder.py F            # 80-epoch schedule
python run_ladder.py E            # post-hoc decoding and final systems -> results/final_systems.json
python run_final_test.py --dev-only   # cap and dev reproduction, no test data
python run_final_test.py              # the one-time test -> results/final/
python build_notebook.py          # builds main.ipynb from src/ and the results
```

`main.ipynb` was executed with nbclient on the project folder. The notebook retrains Stage 1 and the two final systems (about 15 minutes), checks their dev numbers against the logged ones, and reads the one-time test results. It does not evaluate test. `tools/keep_awake.py` holds off idle sleep while long runs write logs.

## 9. Known limitations and open items

- **Stage 1 authorship.** The Stage 1 baseline was developed with AI assistance. The assignment requires the group's own non-AI Stage 1 before submission. This is disclosed in `submission/ai_collaboration.md` and in the report.
- **Optimism of dev numbers.** Every rung-E number is a dev number of a system selected on dev. Each final system has one training seed on test, so single-seed test differences of a few tenths should not be read as real. Across dev seeds the strict family varies by about 0.17 intent and 0.30 slot F1.
- **Tie margin applied late.** The 0.8 margin was introduced for rung E after its first run. The rung protocol that adopted A, B, C, D and R has no margin. Several adopted steps (B +0.81, C +0.64, R +0.59, D +0.19) are at or near the margin. They are reported as tentative.
- **Flagged track.** It needs lecturer approval. It gives no reliable gain over the strict track.
- **Data source.** The experiments used the verified CSV, not the official `en-US.jsonl`. The notebook loads the CSV. The group should load the official file and confirm the split sizes before submitting.
- **Label inventory.** One BIO label occurs only in test (`I-transport_type`) and is inherited from the hand-off. It is never a training target and has no dev effect.
- **Notebook test evaluation.** The hand-off asks the notebook to evaluate test from the saved checkpoint. The notebook reads the one-time results instead. This is a pending decision for the group.
- **Retraining is not bit-identical.** GPU kernels are not deterministic. The notebook's retrained dev numbers differ from the logged ones by at most 0.64 points, and a retrained model can stop at a different epoch.
- **Weights stay local.** The 152 checkpoints in `checkpoints/` are not part of any submission and should not be published.

## 10. Files

| Path | Contents |
|---|---|
| `run_stage1.py` | Stage 1 baseline |
| `run_ladder.py` | Rung runner and the selection protocol (rungs A to E) |
| `run_final_test.py` | Dev checks, the one-time test, the post-test tables |
| `build_notebook.py` | Builds `main.ipynb` from `src/` and the results |
| `src/config.py` | Model and training configurations, parameter cap |
| `src/data.py` | Parsing of annotated utterances, vocabularies, OOV groups, datasets |
| `src/embeddings.py` | Word2Vec loading and the extended vocabulary |
| `src/models.py` | Transformer, BiLSTM, CRF and biaffine heads, residual adapters, TF-IDF wide path, parameter counting |
| `src/train.py` | Training loop: schedules, regularisers, early stopping, checkpoints |
| `src/evaluate.py` | Exact-span metrics, OOV tables, paired bootstrap, matched control, error cases |
| `src/experiment.py` | Run harness and leaderboard |
| `src/posthoc.py` | Post-hoc decoding, MC dropout, ensembles, stored-value accounting |
| `src/guard.py` | GPU temperature and RAM and VRAM guards |
| `tools/` | Smoke test, checkpoint reload check, asset check, Word2Vec download, metrics backfill, keep-awake helper |
| `results/leaderboard.csv` | Every training run (149 runs in 103 families; notebook retrains excluded) |
| `results/posthoc_leaderboard.csv` | Rung-E candidates on dev |
| `results/final_systems.json` | The systems chosen by rung E |
| `results/final/` | One-time test outputs: final comparison, OOV tables, bootstrap, matched control, error cases, dev seed spread, lock file |
| `results/ladder_*.log` | Logs of each rung, plus superseded rung-E logs kept for the record |
| `checkpoints/` | Saved weights (local only) |
| `submission/` | Report, executed notebook, AI-use record, draft ZIP |

## Appendix A. Every training family

103 families from 149 runs (`results/leaderboard.csv`, notebook retrains excluded). Dev joint, standard deviation and dev intent and slot F1 are over the seeds listed under Seeds. Rung letters: S Stage 1, A recipe, B representations, C heads, D encoders, R regularisers, H random search, F 80 epochs.

| Rung | Family | Seeds | Dev joint | SD | Intent | Slot F1 | Flagged | Total params | Note |
|---|---|---|---|---|---|---|---|---|---|
| S | stage1_baseline | 1 | 74.46 | 0.00 | 82.83 | 66.09 | no | 4,344,472 | hand-off Stage-1 config: Emb300->256, 4L Transformer, AdamW 3e-4, 12 ep, patience 3 |
| A | A4_vit | 3 | 82.69 | 0.37 | 86.16 | 79.22 | no | 4,344,472 | A3g2_slot2 + Viterbi decoding |
| A | A3g3_ep80 | 3 | 81.39 | 0.44 | 86.16 | 76.61 | no | 4,344,472 | A3g2_slot2 + ep80 |
| A | A3g2_slot2 | 3 | 81.07 | 0.43 | 86.13 | 76.02 | no | 4,344,472 | A3_rare1 + slot2 |
| A | A2_rare1 | 3 | 79.87 | 0.16 | 85.91 | 73.82 | no | 4,344,472 | A1_fresh_lr5e-4 + rare1 |
| A | A2_slot2 | 1 | 79.69 | 0.00 | 84.65 | 74.73 | no | 4,344,472 | A1_fresh_lr5e-4 + slot2 |
| A | A2_rare.25 | 1 | 79.39 | 0.00 | 85.19 | 73.59 | no | 4,344,472 | A1_fresh_lr5e-4 + rare.25 |
| A | A2_ep80 | 1 | 79.25 | 0.00 | 84.21 | 74.29 | no | 4,344,472 | A1_fresh_lr5e-4 + ep80 |
| A | A1_fresh_lr5e-4 | 1 | 79.10 | 0.00 | 84.75 | 73.44 | no | 4,344,472 | recipe: 40 epochs, cosine, warm-up, label smoothing .1, word dropout .1 |
| A | A2_ls.2 | 1 | 78.98 | 0.00 | 84.46 | 73.50 | no | 4,344,472 | A1_fresh_lr5e-4 + ls.2 |
| A | A2_drop.10 | 1 | 78.94 | 0.00 | 84.60 | 73.27 | no | 4,344,472 | A1_fresh_lr5e-4 + drop.10 |
| A | A2_wd.01 | 1 | 78.87 | 0.00 | 84.51 | 73.23 | no | 4,344,472 | A1_fresh_lr5e-4 + wd.01 |
| A | A2_ls0 | 1 | 78.75 | 0.00 | 84.31 | 73.19 | no | 4,344,472 | A1_fresh_lr5e-4 + ls0 |
| A | A1_ft_recipe | 1 | 78.64 | 0.00 | 84.90 | 72.38 | no | 4,344,472 | recipe: 40 epochs, cosine, warm-up, label smoothing .1, word dropout .1 |
| A | A1_fresh_recipe | 1 | 78.60 | 0.00 | 84.60 | 72.59 | no | 4,344,472 | recipe: 40 epochs, cosine, warm-up, label smoothing .1, word dropout .1 |
| A | A2_wdrop.2 | 1 | 78.50 | 0.00 | 85.15 | 71.85 | no | 4,344,472 | A1_fresh_lr5e-4 + wdrop.2 |
| A | A2_emb.1 | 1 | 78.44 | 0.00 | 85.44 | 71.44 | no | 4,344,472 | A1_fresh_lr5e-4 + emb.1 |
| A | A2_slotdrop.2 | 1 | 78.38 | 0.00 | 84.70 | 72.05 | no | 4,344,472 | A1_fresh_lr5e-4 + slotdrop.2 |
| A | A2_drop.25 | 1 | 77.99 | 0.00 | 84.16 | 71.81 | no | 4,344,472 | A1_fresh_lr5e-4 + drop.25 |
| A | A2_bs32 | 1 | 77.43 | 0.00 | 84.31 | 70.55 | no | 4,344,472 | A1_fresh_lr5e-4 + bs32 |
| A | A2_emb0 | 1 | 77.16 | 0.00 | 86.13 | 68.19 | no | 4,344,472 | A1_fresh_lr5e-4 + emb0 |
| B | B2_ext10k | 1 | 84.21 | 0.00 | 88.39 | 80.03 | yes | 10,090,656 | B1g2_char + ext10k |
| B | B2_ext20k | 1 | 84.19 | 0.00 | 88.15 | 80.23 | yes | 13,090,656 | B1g2_char + ext20k |
| B | B2_ext40k | 3 | 84.18 | 0.12 | 87.80 | 80.56 | yes | 19,090,656 | B1g2_char + ext40k |
| B | B2_ext100k_pca100 | 1 | 84.07 | 0.00 | 88.10 | 80.05 | yes | 16,084,456 | B1g2_char + ext100k_pca100 |
| B | B2_ext20k_noswap | 1 | 83.64 | 0.00 | 87.95 | 79.34 | yes | 13,090,656 | B1g2_char + ext20k_noswap |
| B | B1g2_char | 3 | 83.50 | 0.17 | 87.80 | 79.20 | no | 5,581,056 | B1_wide10k + char |
| B | B1_wide10k | 3 | 83.37 | 0.12 | 87.48 | 79.26 | no | 5,544,592 | A4_vit + wide10k |
| B | B1_char | 3 | 83.34 | 0.24 | 87.28 | 79.40 | no | 4,380,936 | A4_vit + char |
| B | B1_bpe_char | 1 | 83.04 | 0.00 | 86.52 | 79.56 | no | 4,529,736 | A4_vit + bpe_char |
| B | B1_wide20k | 1 | 83.00 | 0.00 | 87.01 | 78.99 | no | 6,737,632 | A4_vit + wide20k |
| B | B1_all3 | 3 | 82.92 | 0.35 | 86.29 | 79.54 | no | 5,508,552 | A4_vit + all3 |
| B | B1_bpe | 3 | 82.89 | 0.26 | 86.24 | 79.55 | no | 4,493,272 | A4_vit + bpe |
| B | B1_ngram | 1 | 82.65 | 0.00 | 86.08 | 79.22 | no | 5,323,288 | A4_vit + ngram |
| B | B1_bpe_ngram | 1 | 82.54 | 0.00 | 85.54 | 79.53 | no | 5,472,088 | A4_vit + bpe_ngram |
| B | B1_wslot14 | 1 | 82.38 | 0.00 | 85.59 | 79.17 | no | 6,113,944 | A4_vit + wslot14 |
| B | B1_wslot15 | 1 | 82.37 | 0.00 | 85.49 | 79.25 | no | 7,883,416 | A4_vit + wslot15 |
| B | B1_gaz | 1 | 82.20 | 0.00 | 86.28 | 78.12 | no | 4,362,088 | A4_vit + gaz |
| C | C2f_flagged | 3 | 84.43 | 0.11 | 88.16 | 80.70 | yes | 19,102,536 | B2_ext40k + strict gains of rung C2 |
| C | C1_crf | 3 | 84.14 | 0.28 | 87.74 | 80.54 | no | 5,592,936 | B1g2_char + crf |
| C | C1_bioes_crf | 3 | 84.12 | 0.17 | 87.95 | 80.30 | no | 5,659,380 | B1g2_char + bioes_crf |
| C | C1g2_tasktop | 3 | 84.03 | 0.43 | 87.82 | 80.25 | no | 6,251,880 | C1_crf + tasktop |
| C | C1_refine | 1 | 83.94 | 0.00 | 87.95 | 79.93 | no | 6,332,928 | B1g2_char + refine |
| C | C1_tasktop | 3 | 83.88 | 0.44 | 88.00 | 79.75 | no | 6,240,000 | B1g2_char + tasktop |
| C | C1_factorized | 1 | 83.85 | 0.00 | 87.80 | 79.89 | no | 5,596,219 | B1g2_char + factorized |
| C | C1_interact | 1 | 83.85 | 0.00 | 88.15 | 79.54 | no | 5,778,176 | B1g2_char + interact |
| C | C1_relbias | 1 | 83.84 | 0.00 | 88.15 | 79.54 | no | 5,581,124 | B1g2_char + relbias |
| C | C1_oweight.5 | 3 | 83.77 | 0.34 | 88.08 | 79.46 | no | 5,581,056 | B1g2_char + oweight.5 |
| C | C1_coint | 1 | 83.76 | 0.00 | 88.59 | 78.92 | no | 6,240,000 | B1g2_char + coint |
| C | C1_meanmax | 1 | 83.74 | 0.00 | 88.15 | 79.34 | no | 5,596,416 | B1g2_char + meanmax |
| C | C1_auxlm | 1 | 83.55 | 0.00 | 88.24 | 78.86 | no | 5,658,156 | B1g2_char + auxlm |
| C | C1_tokintent | 1 | 83.52 | 0.00 | 87.85 | 79.19 | no | 5,581,056 | B1g2_char + tokintent |
| C | C1_i2s | 1 | 83.51 | 0.00 | 88.05 | 78.97 | no | 5,587,536 | B1g2_char + i2s |
| C | C1_biaffine | 1 | 83.47 | 0.00 | 87.51 | 79.44 | no | 6,550,988 | B1g2_char + biaffine |
| C | C1_labelattn | 1 | 83.39 | 0.00 | 87.65 | 79.13 | no | 5,677,312 | B1g2_char + labelattn |
| C | C1_presence | 1 | 83.37 | 0.00 | 88.15 | 78.59 | no | 5,595,191 | B1g2_char + presence |
| C | C1_s2i | 1 | 83.37 | 0.00 | 88.19 | 78.55 | no | 5,587,536 | B1g2_char + s2i |
| C | C1_prelstm | 1 | 83.31 | 0.00 | 87.21 | 79.40 | no | 6,042,112 | B1g2_char + prelstm |
| C | C1_attn | 1 | 83.00 | 0.00 | 87.60 | 78.40 | no | 5,646,848 | B1g2_char + attn |
| C | C1_hier | 1 | 82.96 | 0.00 | 87.46 | 78.47 | no | 5,585,682 | B1g2_char + hier |
| D | D2f_flagged | 3 | 84.85 | 0.34 | 88.16 | 81.55 | yes | 19,140,936 | C2f_flagged + strict gains of rung D2 |
| D | D1_bilstm256 | 3 | 84.33 | 0.30 | 87.83 | 80.82 | no | 5,631,336 | C1_crf + bilstm256 |
| D | D1_bilstm384 | 1 | 84.21 | 0.00 | 87.70 | 80.73 | no | 8,562,024 | C1_crf + bilstm384 |
| D | D1_tfm256x2 | 1 | 84.12 | 0.00 | 87.85 | 80.39 | no | 4,276,072 | C1_crf + tfm256x2 |
| D | D1_tfm256x6 | 1 | 84.11 | 0.00 | 88.05 | 80.17 | no | 6,909,800 | C1_crf + tfm256x6 |
| D | D1_tfm128x2 | 1 | 83.79 | 0.00 | 87.41 | 80.17 | no | 3,221,352 | C1_crf + tfm128x2 |
| D | D1_tfm384x6 | 1 | 83.68 | 0.00 | 87.95 | 79.42 | no | 13,674,600 | C1_crf + tfm384x6 |
| D | D1_bilstm256_lr3 | 1 | 83.22 | 0.00 | 87.56 | 78.88 | no | 5,631,336 | C1_crf + bilstm256_lr3 |
| R | R1_sam.2 | 3 | 84.92 | 0.09 | 88.18 | 81.65 | no | 5,631,336 | D1_bilstm256 + sam.2 |
| R | R2f_flagged | 3 | 84.81 | 0.10 | 88.21 | 81.42 | yes | 19,140,936 | D2f_flagged + strict gains of rung R2 |
| R | R3_mention.3 | 3 | 84.71 | 0.31 | 87.87 | 81.56 | yes | 5,631,336 | R1_sam.2 + mention.3 |
| R | R1_sam.05 | 1 | 84.67 | 0.00 | 87.85 | 81.50 | no | 5,631,336 | D1_bilstm256 + sam.05 |
| R | R1_rdrop1 | 1 | 84.55 | 0.00 | 87.41 | 81.69 | no | 5,631,336 | D1_bilstm256 + rdrop1 |
| R | R1_adv.05 | 1 | 84.49 | 0.00 | 88.00 | 80.98 | no | 5,631,336 | D1_bilstm256 + adv.05 |
| R | R1_ema | 1 | 84.47 | 0.00 | 88.00 | 80.95 | no | 5,631,336 | D1_bilstm256 + ema |
| R | R3_sis.3 | 1 | 84.45 | 0.00 | 87.90 | 80.99 | yes | 5,631,336 | R1_sam.2 + sis.3 |
| R | R1_cons1 | 1 | 84.36 | 0.00 | 87.80 | 80.93 | no | 5,631,336 | D1_bilstm256 + cons1 |
| R | R3_mention.6 | 1 | 84.36 | 0.00 | 87.46 | 81.27 | yes | 5,631,336 | R1_sam.2 + mention.6 |
| R | R3_lwtr.3 | 1 | 84.33 | 0.00 | 87.95 | 80.72 | yes | 5,631,336 | R1_sam.2 + lwtr.3 |
| R | R1_rdrop.25 | 1 | 84.30 | 0.00 | 87.70 | 80.90 | no | 5,631,336 | D1_bilstm256 + rdrop.25 |
| R | R1_supcon.1 | 1 | 84.22 | 0.00 | 88.00 | 80.45 | no | 5,697,000 | D1_bilstm256 + supcon.1 |
| R | R1_cons.25 | 1 | 84.20 | 0.00 | 87.46 | 80.94 | no | 5,631,336 | D1_bilstm256 + cons.25 |
| R | R1_anchor1e-2 | 1 | 84.18 | 0.00 | 87.75 | 80.60 | no | 7,221,336 | D1_bilstm256 + anchor1e-2 |
| R | R1_mixup.2 | 1 | 84.14 | 0.00 | 87.46 | 80.83 | no | 5,631,336 | D1_bilstm256 + mixup.2 |
| R | R1_mdrop.1 | 1 | 84.07 | 0.00 | 87.26 | 80.88 | no | 5,631,336 | D1_bilstm256 + mdrop.1 |
| R | R1_mdrop.3 | 1 | 84.05 | 0.00 | 87.95 | 80.14 | no | 5,631,336 | D1_bilstm256 + mdrop.3 |
| R | R1_anchor1e-3 | 1 | 84.04 | 0.00 | 87.85 | 80.23 | no | 7,221,336 | D1_bilstm256 + anchor1e-3 |
| R | R1_dualemb | 1 | 83.72 | 0.00 | 87.70 | 79.74 | no | 7,221,336 | D1_bilstm256 + dualemb |
| R | R1_adv.2 | 1 | 82.70 | 0.00 | 87.65 | 77.75 | no | 5,631,336 | D1_bilstm256 + adv.2 |
| H | H_04 | 1 | 84.73 | 0.00 | 88.10 | 81.36 | no | 5,631,336 | R1_sam.2 + 04 |
| H | H_02 | 1 | 84.49 | 0.00 | 88.10 | 80.88 | no | 5,631,336 | R1_sam.2 + 02 |
| H | H_08 | 1 | 84.46 | 0.00 | 88.15 | 80.77 | no | 5,631,336 | R1_sam.2 + 08 |
| H | H_03 | 1 | 84.40 | 0.00 | 87.95 | 80.85 | no | 5,631,336 | R1_sam.2 + 03 |
| H | H_01 | 1 | 84.30 | 0.00 | 87.65 | 80.94 | no | 5,631,336 | R1_sam.2 + 01 |
| H | H_09 | 1 | 84.14 | 0.00 | 87.75 | 80.53 | no | 5,631,336 | R1_sam.2 + 09 |
| H | H_07 | 1 | 84.11 | 0.00 | 87.56 | 80.65 | no | 5,631,336 | R1_sam.2 + 07 |
| H | H_10 | 1 | 84.04 | 0.00 | 88.34 | 79.74 | no | 5,631,336 | R1_sam.2 + 10 |
| H | H_00 | 1 | 83.98 | 0.00 | 87.90 | 80.06 | no | 5,631,336 | R1_sam.2 + 00 |
| H | H_06 | 1 | 83.78 | 0.00 | 87.51 | 80.06 | no | 5,631,336 | R1_sam.2 + 06 |
| H | H_05 | 1 | 82.71 | 0.00 | 87.11 | 78.32 | no | 5,631,336 | R1_sam.2 + 05 |
| H | H_11 | 1 | 81.99 | 0.00 | 87.41 | 76.57 | no | 5,631,336 | R1_sam.2 + 11 |
| F | F_flagged_ep80 | 3 | 84.77 | 0.19 | 88.06 | 81.48 | yes | 19,140,936 | D2f_flagged at 80 epochs |
| F | F_strict_ep80 | 3 | 84.55 | 0.12 | 87.95 | 81.15 | no | 5,631,336 | R1_sam.2 at 80 epochs |

## Appendix B. Rung-E candidates (dev)

Every candidate scored in rung E (`results/posthoc_leaderboard.csv`). Stored values follow Section 3. The decision for each is in Section 6.

| Candidate | Members | MC passes | Flagged | Dev intent | Dev slot F1 | Dev joint | Stored values | Decode |
|---|---|---|---|---|---|---|---|---|
| strict_single | 1 | 0 | no | 88.15 | 82.83 | 85.49 | 5,669,696 | span tau 0.5 gm 0.0 prior 0.1 mix 0.65 avg prob |
| strict_single_mc8 | 1 | 8 | no | 88.15 | 82.86 | 85.50 | 5,669,696 | span tau 0.5 gm 0.0 prior 0.1 mix 0.65 avg prob |
| strict_single_mc32 | 1 | 32 | no | 88.05 | 82.42 | 85.24 | 5,669,696 | span tau 0.5 gm 0.0 prior 0.1 mix 0.65 avg prob |
| strict_ensemble | 2 | 0 | no | 88.83 | 83.33 | 86.08 | 11,931,480 | span tau 0.5 gm 0.0 prior 0.0 mix 0.5 avg prob |
| flagged_single | 1 | 0 | yes | 88.59 | 81.99 | 85.29 | 19,172,816 | span tau 0.55 gm 0.0 prior 0.0 mix 0.5 avg prob |
| flagged_single_mc8 | 1 | 8 | yes | 88.69 | 81.95 | 85.32 | 19,172,816 | span tau 0.55 gm 0.0 prior 0.0 mix 0.5 avg prob |
| flagged_single_mc32 | 1 | 32 | yes | 88.59 | 82.12 | 85.35 | 19,172,816 | span tau 0.55 gm 0.0 prior 0.0 mix 0.5 avg prob |

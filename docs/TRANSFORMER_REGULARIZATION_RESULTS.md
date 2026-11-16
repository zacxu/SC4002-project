# Transformer regularization: measured development results

This report is generated from completed saved runs and strict rescoring of their development predictions. It compares the revised Transformer with rare-word dropout, intent label smoothing, and their combination. All model-selection and comparison data here are development data; this analysis performs no model inference or test evaluation.

Analysis status: **complete**. Completed runs: **12**. Complete four-arm seed sets: **[17, 42, 101]**.

## Every completed run

| Seed | Arm | Intent | Slot F1 | Slot precision | Slot recall | Joint exact | Selected / trained epochs | Total parameters | Epoch train + dev seconds |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 17 | baseline | 85.05% | 78.57% | 81.33% | 75.99% | 65.08% | 15 / 20 | 19,748,595 | 1153.38 |
| 17 | word_dropout | 86.67% | 79.32% | 80.22% | 78.43% | 67.68% | 18 / 20 | 19,748,595 | 1155.17 |
| 17 | label_smoothing | 85.69% | 79.10% | 81.28% | 77.04% | 66.65% | 19 / 20 | 19,748,595 | 904.92 |
| 17 | combined | 86.03% | 80.10% | 80.58% | 79.62% | 68.08% | 18 / 20 | 19,748,595 | 887.32 |
| 42 | baseline | 85.83% | 77.76% | 80.45% | 75.25% | 64.83% | 19 / 20 | 19,748,595 | 662.89 |
| 42 | word_dropout | 87.06% | 79.84% | 81.30% | 78.43% | 68.47% | 18 / 20 | 19,748,595 | 713.94 |
| 42 | label_smoothing | 85.54% | 78.21% | 79.58% | 76.89% | 65.76% | 20 / 20 | 19,748,595 | 1125.00 |
| 42 | combined | 86.62% | 80.84% | 82.47% | 79.27% | 68.47% | 20 / 20 | 19,748,595 | 1180.62 |
| 101 | baseline | 85.93% | 77.32% | 80.41% | 74.45% | 64.78% | 18 / 20 | 19,748,595 | 1152.58 |
| 101 | word_dropout | 87.11% | 79.44% | 80.47% | 78.43% | 68.52% | 18 / 20 | 19,748,595 | 1152.16 |
| 101 | label_smoothing | 85.83% | 77.56% | 79.46% | 75.75% | 65.32% | 20 / 20 | 19,748,595 | 904.67 |
| 101 | combined | 87.36% | 79.87% | 80.37% | 79.37% | 68.72% | 20 / 20 | 19,748,595 | 885.85 |

Each reported checkpoint is the earliest maximum of the registered unrounded mean development intent accuracy and exact slot micro-F1. Every original training/development example was retained in every recorded epoch. Training loss heights are not directly comparable when label smoothing changes the objective.

## Matched seeds

| Arm | Seeds | Intent mean ± seed SD | Slot F1 mean ± seed SD | Joint exact mean ± seed SD |
| --- | --- | --- | --- | --- |
| baseline | [17, 42, 101] | 85.60% ± 0.49 | 77.88% ± 0.64 | 64.90% ± 0.16 |
| word_dropout | [17, 42, 101] | 86.95% ± 0.24 | 79.53% ± 0.27 | 68.22% ± 0.47 |
| label_smoothing | [17, 42, 101] | 85.69% ± 0.15 | 78.29% ± 0.78 | 65.91% ± 0.68 |
| combined | [17, 42, 101] | 86.67% ± 0.67 | 80.27% ± 0.51 | 68.42% ± 0.32 |

Only seeds with all four completed arms enter this table. Means are per-seed model metrics, not an ensemble or pooled repeated predictions. One seed has no estimated seed standard deviation.

Registered confirmation policy: Seed 42 screen; run all four arms at registered seeds 17 and 101 if any changed arm improves both required metrics over its fresh seed-42 baseline. Report all completed arms, including regressions.

Seed-42 arms improving both metrics: **['word_dropout', 'combined']**. Confirmation complete: **True**. The seed-42 screen and its conditional follow-up are disclosed; this was not an unconditional three-seed experiment.

Confirmation seeds alone (excluding the screening seed):

| Arm | Seeds | Intent mean ± seed SD | Slot F1 mean ± seed SD |
| --- | --- | --- | --- |
| baseline | [17, 101] | 85.49% ± 0.63 | 77.94% ± 0.89 |
| word_dropout | [17, 101] | 86.89% ± 0.31 | 79.38% ± 0.08 |
| label_smoothing | [17, 101] | 85.76% ± 0.10 | 78.33% ± 1.09 |
| combined | [17, 101] | 86.69% ± 0.94 | 79.98% ± 0.16 |

The highest mean development selection score belongs to **combined**. This ranking uses complete matched seed sets, not the best individual seed.

Against baseline, its mean intent difference is **+1.07 points** and slot-F1 difference is **+2.39 points**. Both required mean metrics improved.

## Paired development comparisons

| Seed | Candidate minus baseline | Intent points [nominal 95% interval] | Slot-F1 points [nominal 95% interval] | Joint exact points [nominal 95% interval] | Intent corrected / introduced |
| ---: | --- | --- | --- | --- | --- |
| 17 | word_dropout | +1.62 [+0.54, +2.71] | +0.74 [-0.45, +1.96] | +2.61 [+1.18, +4.08] | 78 / 45 |
| 17 | label_smoothing | +0.64 [-0.34, +1.57] | +0.53 [-0.60, +1.63] | +1.57 [+0.25, +2.90] | 57 / 44 |
| 17 | combined | +0.98 [-0.05, +2.07] | +1.53 [+0.28, +2.78] | +3.00 [+1.52, +4.48] | 70 / 50 |
| 42 | word_dropout | +1.23 [+0.25, +2.21] | +2.08 [+0.75, +3.43] | +3.64 [+2.21, +5.12] | 67 / 42 |
| 42 | label_smoothing | -0.30 [-1.23, +0.64] | +0.45 [-0.64, +1.54] | +0.93 [-0.30, +2.16] | 45 / 51 |
| 42 | combined | +0.79 [-0.15, +1.77] | +3.08 [+1.82, +4.33] | +3.64 [+2.26, +5.02] | 61 / 45 |
| 101 | word_dropout | +1.18 [+0.15, +2.16] | +2.12 [+0.89, +3.38] | +3.74 [+2.26, +5.21] | 67 / 43 |
| 101 | label_smoothing | -0.10 [-1.03, +0.89] | +0.24 [-0.81, +1.32] | +0.54 [-0.69, +1.77] | 50 / 52 |
| 101 | combined | +1.43 [+0.39, +2.41] | +2.55 [+1.26, +3.88] | +3.94 [+2.46, +5.41] | 70 / 41 |

Intervals use 10,000 paired whole-utterance resamples per comparison. They condition on these development-selected checkpoints, omit selection bias and multiplicity adjustment, and do not estimate training-seed variation. Report regressions and small/uncertain changes alongside gains.

## Novel slot values and unknown words

Seen means (slot type, casefolded literal value) occurs in original train; words also fitted on train. Official exact scoring is unchanged.

| Seed | Arm | Seen value recall | New value / known words recall | New value / OOV words recall | Unseen predicted-value precision | No-slot false positives |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 17 | baseline | 92.14% (1372/1489) | 39.26% (117/298) | 17.78% (40/225) | 47.43% | 60 |
| 17 | word_dropout | 90.87% (1353/1489) | 39.93% (119/298) | 47.11% (106/225) | 51.25% | 62 |
| 17 | label_smoothing | 92.41% (1376/1489) | 40.27% (120/298) | 24.00% (54/225) | 50.43% | 60 |
| 17 | combined | 92.14% (1372/1489) | 39.26% (117/298) | 50.22% (113/225) | 51.45% | 76 |
| 42 | baseline | 91.54% (1363/1489) | 38.59% (115/298) | 16.00% (36/225) | 46.32% | 59 |
| 42 | word_dropout | 91.20% (1358/1489) | 37.58% (112/298) | 48.00% (108/225) | 51.76% | 70 |
| 42 | label_smoothing | 92.14% (1372/1489) | 40.94% (122/298) | 23.56% (53/225) | 47.68% | 72 |
| 42 | combined | 91.74% (1366/1489) | 41.61% (124/298) | 46.67% (105/225) | 54.78% | 60 |
| 101 | baseline | 91.67% (1365/1489) | 40.27% (120/298) | 5.78% (13/225) | 44.04% | 59 |
| 101 | word_dropout | 91.13% (1357/1489) | 38.26% (114/298) | 47.56% (107/225) | 51.52% | 62 |
| 101 | label_smoothing | 92.41% (1376/1489) | 41.28% (123/298) | 11.11% (25/225) | 42.77% | 66 |
| 101 | combined | 91.61% (1364/1489) | 41.28% (123/298) | 48.89% (110/225) | 50.98% | 71 |

The JSON evidence also contains utterance-level no-OOV, non-slot-OOV and slot-OOV strata, full counts, hashes, runtime/configuration values and paired recovered/lost gold slot occurrences. Subgroup movements are descriptive: improved OOV recall does not compensate for a decrease in overall exact slot F1.

## Intent calibration and confidence

These additional diagnostics use the same selected checkpoints and original development utterances. They merge explicitly supplied saved confidence evidence; this reporting step performs no inference. Values are means ± sample standard deviations across the same complete matched seeds listed above. They include the screening seed when present; the confirmation policy remains conditional.

| Arm | NLL (nats) | ECE15 (%) | Multiclass Brier | Mean confidence (%) | Mean incorrect confidence (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.0747 ± 0.0282 | 10.83 ± 0.48 | 0.2479 ± 0.0083 | 96.40 ± 0.08 | 84.34 ± 0.29 |
| word_dropout | 0.8611 ± 0.0377 | 9.20 ± 0.26 | 0.2218 ± 0.0038 | 96.10 ± 0.10 | 82.76 ± 0.73 |
| label_smoothing | 0.7267 ± 0.0219 | 5.64 ± 0.76 | 0.2300 ± 0.0099 | 91.11 ± 0.63 | 73.67 ± 3.70 |
| combined | 0.6518 ± 0.0224 | 4.38 ± 0.93 | 0.2116 ± 0.0082 | 90.95 ± 0.35 | 72.14 ± 0.32 |

Lower NLL, ECE and Brier indicate better probability quality under their stated definitions. Confidence itself is descriptive: lower confidence alone is not a performance improvement. Percentage standard deviations are percentage points; NLL uses ordinary unsmoothed gold-intent probabilities. Incorrect-confidence means exclude error-free seeds, with the contributing n retained in JSON.

| Arm | Incorrect predictions ≥90% confidence | Incorrect predictions ≥99% confidence | Top-2 accuracy (%) | Top-3 accuracy (%) |
| --- | ---: | ---: | ---: | ---: |
| baseline | 160.33 ± 10.79 | 100.00 ± 10.54 | 91.29 ± 0.27 | 93.77 ± 0.44 |
| word_dropout | 141.67 ± 2.08 | 77.33 ± 7.09 | 92.38 ± 0.30 | 94.59 ± 0.30 |
| label_smoothing | 112.33 ± 18.77 | 0.33 ± 0.58 | 91.05 ± 0.30 | 93.06 ± 0.42 |
| combined | 92.33 ± 9.61 | 0.67 ± 0.58 | 91.97 ± 0.37 | 93.95 ± 0.21 |

High-confidence error entries are per-seed utterance counts, averaged across seeds. ECE uses 15 equally spaced confidence bins and depends on binning/sample size. Better calibration or top-k accuracy does not establish higher top-1 intent accuracy or slot F1. Development selected these checkpoints, so these remain development diagnostics.

Source: [saved development confidence evidence](<../results/transformer_regularization/confidence.json>) (SHA-256 `9ee85d13eb3058662f70fa997659400c1d47348ed26be5dbb623291b6625e50e`; completed `2026-11-10T19:29:13.886131+00:00`, device `cuda`). The merger verifies identical complete seed/arm sets, checkpoint/state/ordered-prediction hashes, intent accuracy/error counts, and unchanged diagnostic, protocol, dataset and registered notebook sources. The JSON report retains individual-run diagnostics, metric definitions and the full source-hash manifest.

## Evidence and scope

- Dataset SHA-256: `c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e`.
- Evaluator: `exact_character_span_v2`; original counts {'test': 2974, 'train': 11514, 'dev': 2033}.
- JSON and CSV: `results/transformer_regularization/results.json` and `results/transformer_regularization/results.csv`.
- Saved checkpoint hashes, complete epoch histories, original ordered development IDs, exact scores and fresh-process reproduction records are checked before inclusion.
- All declared whole-model counts must be at most 20M; this analysis does not reinstantiate the model.
- Timings are descriptive workflow measurements. The seed-42 combined run also executes Stage 1 before Stage 2, unlike script-only arms; confirmation arms may share GPU capacity concurrently. These are not controlled speed benchmarks and support no speed-gain claim.
- The official test set was previously exposed in the earlier frozen comparison. No new test result is claimed here; future evaluation must disclose that history.

The separate [historical-control parity audit](../results/transformer_regularization/historical_control_parity.json) records exact agreement between the fresh seed-42 zero-regularizer control and the preserved revised baseline over all 20 epochs of losses, development metrics and schedules, including the selected learned state and saved development prediction bytes. Checkpoint file hashes differ because provenance metadata changed. This analyzer binds that audit to the current baseline checkpoint/state hashes.

Concurrent scheduling details are retained in [confirmation_execution_plan.json](../artifacts/transformer_regularization/confirmation_execution_plan.json) and embedded with their file hash in the JSON evidence. GPU contention affects timing comparability.

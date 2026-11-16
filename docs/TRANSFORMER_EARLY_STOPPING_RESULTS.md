# Transformer early-stopping results

Status: **complete**. Completed seeds: **[42]**. These fresh combined-model runs use a maximum of 60 epochs, a minimum of 20, patience of 8 epochs and min_delta=0.0005 (0.05 percentage points on the joint development score). The original first 20 epochs of the learning-rate schedule is retained exactly, followed by its floor. All comparisons use saved development evidence.

| Seed | Trained / selected epoch | Stop reason | Reference intent | Extended intent | Reference slot F1 | Extended slot F1 | Joint-score gain (points) |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 28 / 20 | patience | 86.62% | 86.62% | 80.84% | 80.84% | +0.000 |

## Paired seed means

| Metric | Reference mean ± SD | Extended mean ± SD | Mean difference (points) |
| --- | --- | --- | ---: |
| intent_accuracy | 86.62% | 86.62% | +0.000 |
| slot_micro_f1 | 80.84% | 80.84% | +0.000 |
| fully_correct_utterance_rate | 68.47% | 68.47% | +0.000 |

The SD is the sample standard deviation across completed seeds, not a confidence interval. A one-seed run has no estimated seed SD. Intent and slot changes are reported separately; a higher mean selection score can trade one metric against the other.

## Paired development uncertainty

| Seed | Intent difference [nominal 95% interval] | Slot-F1 difference [nominal 95% interval] | Joint-exact difference [nominal 95% interval] |
| ---: | --- | --- | --- |
| 42 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

Intervals use 10,000 paired whole-development-utterance resamples. They condition on development-selected checkpoints, exclude selection/multiplicity effects, and are not training-seed uncertainty.

## Verification and interpretation

- Every original history value for the first 20 epochs except train/dev elapsed time matches its preserved same-seed combined reference exactly.
- The evidence saved at epoch 20 matches the earlier best epoch within that budget, learned-tensor hash, complete metrics and canonical ordered prediction hash.
- Current epoch-20 state can also be compared where the old selected checkpoint was epoch 20; otherwise only its saved best state within the first 20 epochs is available.
- Initial model/embedding identities, original data, train-only mappings, parameter tensors, fresh development reload and exact character-span scores are verified.
- The independent validator reconstructs patience counters, the fixed-horizon/floor learning rates, terminal reason, and earliest actual maximum separately from the min_delta anchor.
- Checkpoint tensors are read on CPU; this analyzer performs no model construction, training, forward pass, test scoring or checkpoint writes.

Confirmation rule: Run seed42 first; if its selected joint development score exceeds its own first20 maximum, run both registered confirmation seeds17 and101 regardless of their results. Confirmation required: **False**; complete: **True**.

Detailed counts, novelty/OOV recall, precision/recall, joint exactness, paired recovered/lost errors, timing, source hashes and per-seed stopping/prefix audits are in `results/transformer_early_stopping/results.json`. The earlier regularization results remain separate and unchanged.

- Development chooses checkpoints, triggers confirmation and measures the resulting gains.
- Paired bootstrap conditions on selected checkpoints and omits checkpoint-selection and multiple-comparison effects.
- Only seeds actually completed enter paired means; a partial report is not the final registered comparison.
- More checkpoints provide additional chances to improve on the selection split; no new held-out/test claim is made.
- Runtime measures are descriptive and may reflect different concurrent GPU workloads.

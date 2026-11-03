# AI collaboration record

## AI and models used

Claude (Anthropic), used through Claude Code. The main working session ran on Claude Haiku 5.5. Four code-review subagents ran on Claude Opus. Earlier stages of this project may also have used other Claude models; the group should add those if they were used.

## Stage 1: disclosure

The Stage 1 baseline (the Transformer architecture, its configuration, data handling, and training loop) comes from a hand-off notebook produced with AI assistance, and it was re-implemented as scripts in this project with AI help. **This does not meet the assignment requirement that Stage 1 be developed without generative AI.** The group must replace Stage 1 with its own independent implementation before submitting. The numbers in the report describe this disclosed baseline and are not a substitute for the group's own Stage 1.

## Weaknesses diagnosed (Stage 1)

1. Slot F1 collapses on slots that contain a word unseen in training (Slot-Train-OOV: 29.9 on test, against 70.1 for utterances without an unseen word).
2. Intent accuracy drops when an unseen word appears outside the slots.
3. Argmax decoding produces invalid BIO sequences.
4. The validation curve was still rising at epoch 12 under a constant learning rate.
5. Part of the OOV gap survives matching on utterance length and slot count.

## Suggestions: accepted, modified, rejected

**Accepted (each checked on dev with 3-seed means):**
- Cosine warm-up, label smoothing and word dropout, with 40 epochs (rung A).
- BIO-constrained Viterbi decoding (A4).
- A TF-IDF wide intent path and a character n-gram residual (B).
- A CRF slot head (C).
- A BiLSTM encoder (D).
- Sharpness-aware training with rho = 0.2 (R, tentative).

Several of these gains were small on dev (0.2 to 0.8 points). The report says so.

**Modified:**
- The 0.8-point tie margin from the project protocol was applied to every rung E decision after the first version of rung E chose noise-level variants.
- The notebook was changed so that it does not evaluate test. It reads the one-time test results, because its retrained models would not match the reported checkpoints.
- The extended-vocabulary track is reported separately, because it needs lecturer approval.
- A reviewer's suggestion to count mention-table keys in the cap was not adopted. Instead the mention table was dropped wherever it failed the tie margin.

**Rejected:**
- 80-epoch training (84.55 against 84.92 strict).
- Mention-replacement augmentation (84.71).
- The 12 random configurations (none beat the base).
- MC-dropout inference (at most +0.06 on dev).
- A mention lookup table and a two-member ensemble, where the gains were within the tie margin.
- Carrying the SAM gain onto the flagged base (84.81 against 84.85).

## Verification of AI advice

Before the one-time test, four independent code reviews were run. They found an MC intent-mix bug, a test-data leak in a seed-spread loop, a lock-ordering problem and a cap-counting issue. Each finding was checked against the code before a fix was applied. Dev numbers of the final systems were re-derived and matched the ladder log before the test was scored.

## Unsuccessful experiments

Besides the rejected options above: a first version of rung E chose MC variants and a third ensemble member on gains of 0.01 to 0.19; that rule was corrected. A stale output cache was reused across code versions and produced a crash; it is fixed by versioned cache names.

## Group reasoning

*To be completed by the group.* Record the group's own reasoning about the ladder choices, the decision to report the strict track as primary, the handling of the tie margin, and the limitations. Each member should state which parts they checked themselves.

## Open items

- The official `en-US.jsonl` was not available; the experiments used the verified CSV. The group should load the official file and confirm the split sizes before submitting.
- The BIO label inventory includes one label that occurs only in the test split (`I-transport_type`), inherited from the hand-off. It is never a training target and has no dev effect.

# Dev analysis of Stage 1 (train and dev only)

Test rows were dropped on read. Recomputed from `dev_pred_frame.pkl` and the CSV: intent 82.83, slot F1 66.09, joint 74.46, TP 1334, FP 691, FN 678; all OOV group labels match. Scripts and full tables are in this folder. "A1" is `A1_ft_recipe` from the running ladder, added where it changes a conclusion.

## 1. Label noise and intent ceiling

**Exact twins.** 11 dev utterances (0.54%) have a casefold twin in train: intent agrees 10/11, BIO 11/11. Train has 46 duplicate groups (92 rows): 4 conflict on intent, 1 on annotation, 5 on either. Exact twins are too few to carry a ceiling, so near-duplicate keys were added.

| key | dev_seen | dev_seen_pct | majority_acc_on_seen | S1_on_seen | A1_on_seen | linear_on_seen | S1_on_unseen | train_groups_n>=2 | train_rows_in_them | pair_disagree_pct | ci_low | ci_high |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exact casefold | 11 | 0.5 | 90.91 | 90.91 | 90.91 | 90.91 | 82.79 | 46 | 92 | 8.7 | 2.17 | 17.39 |
| no wake word, no punctuation | 72 | 3.5 | 97.22 | 97.22 | 98.61 | 98.61 | 82.3 | 223 | 451 | 2.22 | 0.44 | 4.42 |
| content words, order free | 292 | 14.4 | 97.26 | 98.29 | 97.95 | 98.63 | 80.24 | 608 | 1540 | 4.74 | 3.35 | 6.29 |
| template | 258 | 12.7 | 93.02 | 89.92 | 91.86 | 93.8 | 81.8 | 517 | 1510 | 7.65 | 5.55 | 10.03 |
| template keeping device_type value | 246 | 12.1 | 96.34 | 89.43 | 91.46 | 93.5 | 81.93 | 490 | 1423 | 4.92 | 3.31 | 6.8 |

**Templates.** 10,521 unique templates in 11,514 train rows; 517 occur at least twice (1,510 rows); 51 of those (9.9%) map to more than one intent (154 rows, 1.3% of train). Many conflicts are an artefact of abstraction: `turn on the <device_type>` is decided by the device value.

| subset | n | share_dev_pct | tmpl_major_acc | ci95 | S1_acc |
|---|---:|---:|---|---|---:|
| template seen in train | 258 | 12.69 | 93.02 | (89.2, 95.5) | 89.92 |
|  seen, twin too | 11 | 0.54 | 90.91 | (62.3, 98.4) | 90.91 |
|  seen, no twin | 247 | 12.15 | 93.12 | (89.3, 95.7) | 89.88 |
|  seen, with slot | 254 | 12.49 | 93.31 | (89.5, 95.8) | 90.16 |
|  seen, no slot | 4 | 0.2 | 75.0 | (30.1, 95.4) | 75.0 |
|  seen n=1 | 107 | 5.26 | 90.65 | (83.6, 94.8) | 91.59 |
|  seen n=2-4 | 78 | 3.84 | 92.31 | (84.2, 96.4) | 88.46 |
|  seen n>=5 | 73 | 3.59 | 97.26 | (90.5, 99.2) | 89.04 |
|  seen, pure in train | 241 | 11.85 | 95.44 | (92.0, 97.4) | 90.04 |
|  seen, multi-intent in train | 17 | 0.84 | 58.82 | (36.0, 78.4) | 88.24 |
| template unseen | 1775 | 87.31 | n/a |  | 81.8 |
| all dev | 2033 | 100.0 | n/a |  | 82.83 |

**Nearest train neighbour by word TF-IDF cosine.**

| cosine | n | share_pct | nn_label_agrees | S1 | A1 | linear | mean_tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| <=.2 | 4 | 0.2 | 50.0 | 50.0 | 50.0 | 75.0 | 10.0 |
| .2-.4 | 364 | 17.9 | 45.3 | 62.1 | 66.5 | 67.9 | 8.2 |
| .4-.5 | 392 | 19.3 | 56.4 | 80.1 | 80.9 | 80.4 | 7.3 |
| .5-.6 | 424 | 20.9 | 74.5 | 83.7 | 84.7 | 87.5 | 6.8 |
| .6-.7 | 340 | 16.7 | 85.0 | 90.9 | 93.5 | 94.1 | 6.5 |
| .7-.8 | 250 | 12.3 | 88.4 | 92.4 | 94.8 | 96.0 | 5.9 |
| .8-.9 | 170 | 8.4 | 97.1 | 95.9 | 97.6 | 98.8 | 6.1 |
| >.9 | 89 | 4.4 | 96.6 | 94.4 | 95.5 | 96.6 | 8.0 |

**Slot annotation on near-duplicates** (pairs compared on (type, value) sets):

| key | groups | pairs | identical_annotation_pct | pairwise_span_f1 | ci_low | ci_high |
|---|---:|---:|---:|---:|---:|---:|
| no wake word, no punctuation | 302 | 314 | 99.7 | 99.83 | 99.47 | 100.0 |
| content words, order free | 789 | 2264 | 93.3 | 90.38 | 86.52 | 93.42 |

**Audit of a fixed random sample of 60 Stage-1 intent errors** (one reader, labels in `q1b_error_audit_sample.csv`): gold doubtful or defensibly another label 31, nonsense with catch-all gold 2, clear model error 27. That puts 52-55% of errors, about 9% of dev (CI about 7-11%), out of reach from the text alone.

| Evidence | Coverage of dev | Implied intent ceiling on dev | Caveat |
|---|---:|---:|---|
| Exact duplicates, Cover-Hart bracket | 0.5% | 91.3-95.6 | 46 groups only, CI of disagreement 2.2-17.4% |
| Template majority on seen templates | 12.7% | 93.0 (CI 89.2-95.5) | uses gold slots; short forms; conflict inflated by abstraction |
| Template bracket keeping `device_type` | 12.1% | 95.1-97.5 from 4.92% disagreement | short forms only |
| Stage 1, A1 and linear all wrong | 100% | 90.8 (187 utterances) | not a bound; shared hard core |
| Error audit, 60 sampled errors | 100% | about 91 (CI about 89-93) | single reader, small sample |

**Estimate: about 90-94% is a defensible upper range, about 91-92% the central guess. Evidence strength: weak to moderate.** 87% of dev has an unseen template, so nothing measures label noise there directly. Near-duplicates show low noise (2-5% disagreement) but are short, easy forms; the fuzzy classes live in the low-similarity region. Re-annotating a sample would be stronger evidence.

## 2. Intent confusions (349 errors)

| pair | n | share_pct | cum_pct | a_to_b | b_to_a | same_scenario | tmpl_shared_train | a1_also_wrong | no_oov | nonslot_oov | slot_oov |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| general_quirky / qa_factoid | 23 | 6.6 | 6.6 | 12 | 11 | False | 1 | 21 | 8 | 12 | 3 |
| news_query / qa_factoid | 10 | 2.9 | 9.5 | 6 | 4 | False | 0 | 10 | 4 | 2 | 4 |
| calendar_query / calendar_set | 8 | 2.3 | 11.7 | 2 | 6 | True | 1 | 5 | 8 | 0 | 0 |
| general_quirky / news_query | 7 | 2.0 | 13.8 | 5 | 2 | False | 0 | 6 | 6 | 1 | 0 |
| play_music / play_radio | 7 | 2.0 | 15.8 | 3 | 4 | True | 1 | 6 | 5 | 1 | 1 |
| music_likeness / music_query | 7 | 2.0 | 17.8 | 2 | 5 | True | 0 | 5 | 4 | 1 | 2 |
| qa_definition / qa_factoid | 7 | 2.0 | 19.8 | 5 | 2 | True | 0 | 7 | 3 | 2 | 2 |
| general_quirky / social_post | 6 | 1.7 | 21.5 | 4 | 2 | False | 0 | 5 | 4 | 2 | 0 |
| calendar_query / general_quirky | 6 | 1.7 | 23.2 | 4 | 2 | False | 0 | 4 | 4 | 1 | 1 |
| news_query / recommendation_events | 6 | 1.7 | 24.9 | 0 | 6 | False | 1 | 6 | 6 | 0 | 0 |
| cooking_recipe / general_quirky | 5 | 1.4 | 26.4 | 1 | 4 | False | 0 | 5 | 3 | 1 | 1 |
| general_quirky / music_query | 4 | 1.1 | 27.5 | 2 | 2 | False | 0 | 2 | 3 | 1 | 0 |
| general_quirky / weather_query | 4 | 1.1 | 28.7 | 3 | 1 | False | 0 | 4 | 1 | 3 | 0 |
| calendar_query / datetime_query | 4 | 1.1 | 29.8 | 1 | 3 | False | 5 | 2 | 4 | 0 | 0 |
| iot_hue_lightchange / iot_hue_lightup | 4 | 1.1 | 30.9 | 4 | 0 | True | 1 | 1 | 4 | 0 | 0 |
| alarm_query / alarm_set | 4 | 1.1 | 32.1 | 4 | 0 | True | 0 | 2 | 4 | 0 | 0 |
| takeaway_order / takeaway_query | 4 | 1.1 | 33.2 | 4 | 0 | True | 0 | 3 | 3 | 0 | 1 |
| email_querycontact / general_quirky | 4 | 1.1 | 34.4 | 0 | 4 | False | 0 | 4 | 2 | 2 | 0 |
| calendar_query / recommendation_events | 4 | 1.1 | 35.5 | 2 | 2 | False | 0 | 3 | 4 | 0 | 0 |
| transport_query / transport_ticket | 4 | 1.1 | 36.7 | 1 | 3 | True | 0 | 1 | 4 | 0 | 0 |

`a_to_b` counts gold a predicted as b. Top 20 pairs hold 36.7% of errors; 167 distinct pairs exist, 93 of them once. 89 errors (25.5%) stay inside a scenario. Scenario accuracy is 87.21. A1 repeats 268 of the 349 errors (76.8%), 183 with the same wrong label.

Examples (seeded random pick; all 55 errors of these pairs are in `q2_top5_pair_errors.csv`):

| pair | gold | pred | annotated utterance | audit |
|---|---|---|---|---|
| general_quirky / qa_factoid | general_quirky | qa_factoid | tell me something about [place_name : america] | L |
| general_quirky / qa_factoid | general_quirky | qa_factoid | explain to me about what different countries are represented in the book of revelation | L |
| general_quirky / qa_factoid | general_quirky | qa_factoid | wikipedia topics | L |
| news_query / qa_factoid | qa_factoid | news_query | tell me about [place_name : new york] | L |
| news_query / qa_factoid | news_query | qa_factoid | how many runs did [person : cheteshwar pujara] scored [date : yesterday] | L |
| news_query / qa_factoid | news_query | qa_factoid | what is the [place_name : seattle] [news_topic : city council doing about homelessness] | L |
| calendar_query / calendar_set | calendar_set | calendar_query | let me know the [timeofday : morning] of my [event_name : doctor appointment] on the [date : twelfth] | L |
| calendar_query / calendar_set | calendar_set | calendar_query | notification of [event_name : meeting] on [date : wednesday] | M |
| calendar_query / calendar_set | calendar_query | calendar_set | schedule | L |
| general_quirky / news_query | general_quirky | news_query | what's been going on | L |
| general_quirky / news_query | general_quirky | news_query | tell me what happens when we die | M |
| general_quirky / news_query | news_query | general_quirky | who is leading the [news_topic : formula one] race right now | M |
| play_music / play_radio | play_music | play_radio | load [app_name : spotify] | L |
| play_music / play_radio | play_radio | play_music | play [radio_name : nine hundred and seventy one] | L |
| play_music / play_radio | play_music | play_radio | i want to listen to [music_genre : soft rock] | L |

Audit per pair (L doubtful, U catch-all nonsense, M model error):

| pair | M | L | U |
|---|---:|---:|---:|
| general_quirky / qa_factoid | 6 | 17 | 0 |
| news_query / qa_factoid | 2 | 8 | 0 |
| calendar_query / calendar_set | 2 | 6 | 0 |
| general_quirky / news_query | 3 | 3 | 1 |
| play_music / play_radio | 3 | 4 | 0 |

Top-5 pairs: 38 of 55 doubtful (69%). Random sample of all errors: 31-33 of 60. Most top-pair errors are plausibly label ambiguity, about half of all errors are, and the rest are real model errors such as `switch on the lights` -> `iot_hue_lightup`.

| group | n | errors | error_share_pct | S1_acc | A1_acc | expected_acc_from_intent_mix | expected_acc_from_length_mix | mean_tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No-Train-OOV | 1669 | 232 | 66.5 | 86.1 | 87.84 | 86.1 | 86.1 | 6.74 |
| NonSlot-Train-OOV | 149 | 66 | 18.9 | 55.7 | 60.4 | 76.96 | 86.1 | 8.52 |
| Slot-Train-OOV | 215 | 51 | 14.6 | 76.28 | 79.07 | 86.82 | 86.43 | 7.67 |
| Overall | 2033 | 349 | 100.0 | 82.83 | 84.9 | 85.51 | 86.13 | 6.97 |

Lifting both OOV groups to the No-OOV accuracy would add at most 3.27 points; 2.67 after standardising the intent mix. The NonSlot group is heavy in `general_quirky` (34) and `qa_factoid` (22); even the char-gram linear model only reaches 57.05 there.

## 3. Slot error taxonomy

Gold view (every FN assigned once; the same span may have a counterpart in the prediction view):

| group | gold | F1 | exact % | right span wrong type % | boundary same type % | overlap other type % | missed % |
|---|---:|---:|---:|---:|---:|---:|---:|
| No-Train-OOV | 1615 | 71.51 | 73.7 | 3.8 | 10.8 | 3.0 | 8.6 |
| NonSlot-Train-OOV | 76 | 59.88 | 65.8 | 6.6 | 10.5 | 2.6 | 14.5 |
| Slot-Train-OOV | 321 | 34.51 | 29.0 | 5.6 | 15.3 | 6.9 | 43.3 |
| Overall | 2012 | 66.09 | 66.3 | 4.2 | 11.5 | 3.6 | 14.4 |

Prediction view (every FP assigned once):

| group | pred | exact % | right span wrong type % | boundary same type % | overlap other type % | spurious % |
|---|---:|---:|---:|---:|---:|---:|
| No-Train-OOV | 1716 | 69.4 | 3.6 | 12.5 | 5.8 | 8.7 |
| NonSlot-Train-OOV | 91 | 54.9 | 5.5 | 11.0 | 2.2 | 26.4 |
| Slot-Train-OOV | 218 | 42.7 | 8.3 | 26.1 | 14.2 | 8.7 |
| Overall | 2025 | 65.9 | 4.1 | 13.9 | 6.6 | 9.5 |

What-if overall F1 if a category disappeared (base 66.09):

| repair | gold_side | pred_side | F1 | gain |
|---|---:|---:|---:|---:|
| wrong_type_same_span | 84 | 84 | 70.25 | 4.16 |
| boundary_same_type | 232 | 282 | 78.56 | 12.47 |
| overlap_other | 73 | 133 | 70.76 | 4.67 |
| spurious removed | 0 | 192 | 69.39 | 3.3 |
| missed found | 289 | 0 | 75.03 | 8.95 |

Lifting the Slot-Train-OOV group to F1 71.51 / 77.78 gives overall 71.04 / 72.01 (joint +2.48 / +2.96). That group holds 16.0% of gold spans.

| subset | n | exact | wrong_type_same_span | boundary_same_type | overlap_other | missed |
|---|---:|---:|---:|---:|---:|---:|
| span has Train-OOV token | 225 | 12.0 | 4.9 | 18.7 | 8.0 | 56.4 |
| span without OOV, in Slot-Train-OOV utt | 96 | 68.8 | 7.3 | 7.3 | 4.2 | 12.5 |
| span without OOV, other utts | 1691 | 73.4 | 3.9 | 10.8 | 3.0 | 8.9 |

Boundary errors: 200 of 232 have the prediction inside the gold span (`new york city` -> `new york`, `hurts like heaven` -> `heaven`). Only 30 of 232 differ by function words or dots (F1 67.57 if repaired). Intent right: slot F1 68.46; intent wrong: 51.09.

Illegal BIO in predictions and the effect of dropping stray I- runs at decode time:

| model | F1 | tp | fp | fn | g_boundary | p_boundary | g_missed | p_spurious | I_tags | stray_after_O | stray_after_other_type | F1_drop_stray | gain_drop_stray |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage 1 | 66.09 | 1334 | 691 | 678 | 232 | 282 | 289 | 192 | 1092 | 158 | 72 | 68.14 | 2.05 |
| A1_ft_recipe | 72.38 | 1444 | 534 | 568 | 199 | 226 | 257 | 162 | 1141 | 117 | 44 | 74.48 | 2.1 |

Top 15 types by support (75.6% of gold spans):

| type | train_spans | support | pred | tp | fp | fn | precision | recall | f1 | oov_span_pct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| date | 1816 | 339 | 389 | 300 | 89 | 39 | 77.1 | 88.5 | 82.4 | 0.6 |
| place_name | 1086 | 209 | 184 | 122 | 62 | 87 | 66.3 | 58.4 | 62.1 | 15.8 |
| event_name | 998 | 160 | 195 | 116 | 79 | 44 | 59.5 | 72.5 | 65.4 | 6.9 |
| time | 797 | 137 | 157 | 97 | 60 | 40 | 61.8 | 70.8 | 66.0 | 0.0 |
| person | 864 | 132 | 113 | 72 | 41 | 60 | 63.7 | 54.5 | 58.8 | 35.6 |
| media_type | 477 | 98 | 92 | 71 | 21 | 27 | 77.2 | 72.4 | 74.7 | 3.1 |
| weather_descriptor | 320 | 59 | 69 | 51 | 18 | 8 | 73.9 | 86.4 | 79.7 | 5.1 |
| business_name | 383 | 58 | 57 | 35 | 22 | 23 | 61.4 | 60.3 | 60.9 | 17.2 |
| transport_type | 318 | 54 | 58 | 53 | 5 | 1 | 91.4 | 98.1 | 94.6 | 0.0 |
| food_type | 295 | 51 | 51 | 30 | 21 | 21 | 58.8 | 58.8 | 58.8 | 19.6 |
| timeofday | 239 | 49 | 57 | 44 | 13 | 5 | 77.2 | 89.8 | 83.0 | 2.0 |
| currency_name | 217 | 45 | 46 | 28 | 18 | 17 | 60.9 | 62.2 | 61.5 | 6.7 |
| artist_name | 232 | 45 | 39 | 21 | 18 | 24 | 53.8 | 46.7 | 50.0 | 35.6 |
| definition_word | 225 | 43 | 46 | 33 | 13 | 10 | 71.7 | 76.7 | 74.2 | 44.2 |
| relation | 251 | 42 | 41 | 31 | 10 | 11 | 75.6 | 73.8 | 74.7 | 7.1 |

Worst 10 with support at least 10:

| type | train_spans | support | pred | tp | fp | fn | precision | recall | f1 | oov_span_pct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| song_name | 128 | 23 | 30 | 4 | 26 | 19 | 13.3 | 17.4 | 15.1 | 39.1 |
| radio_name | 131 | 21 | 26 | 4 | 22 | 17 | 15.4 | 19.0 | 17.0 | 14.3 |
| general_frequency | 84 | 15 | 16 | 3 | 13 | 12 | 18.8 | 20.0 | 19.4 | 0.0 |
| audiobook_name | 73 | 16 | 16 | 4 | 12 | 12 | 25.0 | 25.0 | 25.0 | 43.8 |
| podcast_descriptor | 68 | 10 | 13 | 3 | 10 | 7 | 23.1 | 30.0 | 26.1 | 0.0 |
| news_topic | 183 | 40 | 31 | 13 | 18 | 27 | 41.9 | 32.5 | 36.6 | 27.5 |
| artist_name | 232 | 45 | 39 | 21 | 18 | 24 | 53.8 | 46.7 | 50.0 | 35.6 |
| music_genre | 181 | 37 | 29 | 18 | 11 | 19 | 62.1 | 48.6 | 54.5 | 5.4 |
| list_name | 191 | 37 | 25 | 17 | 8 | 20 | 68.0 | 45.9 | 54.8 | 2.7 |
| food_type | 295 | 51 | 51 | 30 | 21 | 21 | 58.8 | 58.8 | 58.8 | 19.6 |

## 4. Linear baseline

Word 1-2 grams plus char_wb 3-5 grams, TF-IDF fit on train, multinomial logistic regression (saga; L-BFGS history would break the memory limit), 59,947 features.

| C | dev_acc | train_acc | iters | sec |
|---:|---:|---:|---:|---:|
| 1.0 | 84.26 | 95.05 | 19 | 4 |
| 10.0 | 85.93 | 99.9 | 56 | 12 |
| 100.0 | 86.08 | 99.97 | 200 | 42 |

C=100 stopped at max_iter and sits at the grid edge, so a sensitivity run followed; the selected model stays the grid winner.

| C | max_iter | iters | dev_acc | fixes_S1 | breaks_S1 |
|---:|---:|---:|---:|---:|---:|
| 3.0 | 300 | 28 | 85.59 | 146 | 90 |
| 30.0 | 300 | 103 | 86.18 | 143 | 75 |
| 100.0 | 1000 | 1000 | 86.28 | 146 | 76 |
| 300.0 | 1000 | 1000 | 86.13 | 146 | 79 |

| model | acc | linear_acc | model_errors | linear_fixes | linear_breaks | both_wrong | both_wrong_same_label | both_wrong_diff_label | oracle_union_acc | mcnemar_p | agreement_pct | acc_when_agree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| Stage 1 | 82.83 | 86.08 | 349 | 145 | 79 | 204 | 107 | 97 | 89.97 | <0.0001 | 84.2 | 93.75 |
| A1_ft_recipe | 84.9 | 86.08 | 307 | 97 | 73 | 210 | 125 | 85 | 89.67 | 0.0774 | 87.5 | 92.97 |

| group | n | S1 | A1 | linear | lin_fixes_S1 | lin_breaks_S1 | lin_fixes_A1 | lin_breaks_A1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No-Train-OOV | 1669 | 86.1 | 87.84 | 88.8 | 102 | 57 | 68 | 52 |
| NonSlot-Train-OOV | 149 | 55.7 | 60.4 | 57.05 | 19 | 17 | 12 | 17 |
| Slot-Train-OOV | 215 | 76.28 | 79.07 | 85.12 | 24 | 5 | 17 | 4 |
| Overall | 2033 | 82.83 | 84.9 | 86.08 | 145 | 79 | 97 | 73 |

Override with the linear label above a confidence threshold, threshold cross-validated on dev (no deep probabilities were available here):

| base | base_acc | linear_acc | gated_cv_acc | thresholds |
|---|---:|---:|---:|---|
| Stage 1 | 82.83 | 86.08 | 85.54 | [0.479, 0.204, 0.404, 0.567, 0.567] |
| A1_ft_recipe | 84.9 | 86.08 | 85.98 | [0.479, 0.748, 0.607, 0.607, 0.709] |

| features | n_features | wide_params | dev_acc | No-Trai | NonSlot | Slot-Tr | fixes_S1 | breaks_S1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| word 1-2 only | 29531 | 1771920 | 84.6 | 87.3 | 55.7 | 83.72 | 137 | 101 |
| char_wb 3-5 only | 30416 | 1825020 | 85.24 | 87.96 | 58.39 | 82.79 | 139 | 90 |
| both, cap 5k+5k | 10000 | 600060 | 85.44 | 88.38 | 54.36 | 84.19 | 145 | 92 |
| both, cap 10k+10k | 20000 | 1200060 | 85.88 | 88.56 | 56.38 | 85.58 | 143 | 81 |
| both, cap 20k+20k | 40000 | 2400060 | 86.03 | 88.68 | 57.05 | 85.58 | 145 | 80 |

## 5. Word2Vec coverage

464 dev Train-OOV lexical tokens (3.32% of lexical tokens), 439 types; 254 inside gold slots, 210 outside. Filtered key list: 256,856 words.

| K | tok_all_% | tok_slot_% | tok_nonslot_% | type_all_% | type_slot_% | type_nonslot_% | slotoov_utts_fully_covered_% | nonslot_utts_fully_covered_% | params_300d_M | legal_300d | params_100d_M | legal_100d |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5000 | 37.3 | 28.3 | 48.1 | 36.4 | 28.3 | 47.0 | 25.6 | 43.6 | 7.354 | True | 5.348 | True |
| 10000 | 51.3 | 41.3 | 63.3 | 51.0 | 41.4 | 63.0 | 37.2 | 59.1 | 8.854 | True | 5.848 | True |
| 20000 | 62.5 | 53.9 | 72.9 | 62.2 | 53.7 | 72.5 | 49.8 | 70.5 | 11.854 | True | 6.848 | True |
| 30000 | 70.0 | 61.4 | 80.5 | 69.7 | 61.1 | 80.0 | 57.7 | 79.2 | 14.854 | True | 7.848 | True |
| 46000 | 76.5 | 69.7 | 84.8 | 75.9 | 68.9 | 84.5 | 67.4 | 83.9 | 19.654 | True | 9.448 | True |
| 100000 | 84.3 | 78.3 | 91.4 | 83.8 | 77.9 | 91.5 | 75.8 | 91.3 | 35.854 | False | 14.848 | True |
| 150000 | 85.3 | 79.5 | 92.4 | 85.0 | 79.1 | 92.5 | 77.2 | 92.6 | 50.854 | False | 19.848 | True |
| 256856 | 87.3 | 82.3 | 93.3 | 87.0 | 82.0 | 93.5 | 80.0 | 94.0 | 82.911 | False | 30.533 | False |

Largest legal K, base 4,344,472: ignoring the map 47,454 at 300d and 151,824 at 100d. With the `frozen_map` the code builds (90,300 or 30,300 parameters): 47,153 and 151,521.

| adapters | adapter_params | max_K_300d_with_map | max_K_100d_with_map | ext46k_300d_total | ext150k_100d_total | ext46k_legal | ext150k_100d_legal |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0 | 47153 | 151521 | 19654072 | 19847872 | True | True |
| bpe | 148800 | 46657 | 150033 | 19802872 | 19996672 | True | True |
| char | 36464 | 47031 | 151156 | 19690536 | 19884336 | True | True |
| ngram | 978816 | 43890 | 141733 | 20632888 | 20826688 | False | False |
| bpe+char | 185264 | 46535 | 149668 | 19839336 | 20033136 | True | False |
| bpe+ngram | 1127616 | 43394 | 140245 | 20781688 | 20975488 | False | False |
| all3 | 1164080 | 43272 | 139880 | 20818152 | 21011952 | False | False |

Uncovered at K=46k: 109 tokens (23.5%), 106 types, 77 in slots, by reading all of them:

| Kind | Tokens | Share | Examples |
|---|---:|---:|---|
| Ordinary words rare in Wikipedia or missing | 36 | 33.0% | bedtime, crispy, toggle, thesaurus, marinara, both |
| Names and brands | 34 | 31.2% | cheteshwar, zaalima, beyonce, iheartradio, tinder |
| Possessive forms (18 have a known base word) | 20 | 18.3% | romero's, thursday's, queen's |
| Typos and concatenations | 19 | 17.4% | washingdon, fransisco, metion, tiangle, harrypotter |
| Numbers | 0 | 0% | the data has no digit at all |

50 of the 109 tokens are in the key list beyond rank 46k; 59 are absent from it. Names at the boundary are judged by reading, so treat the split as approximate. Per-type list: `q5_uncovered_at_46k_types.csv`.

A parameter-free lookup fallback to train words needs no Word2Vec row and no flag; edit distance 1 is too noisy to use (`cover` -> `over`, `poland` -> `boland`):

| rule | tokens | types | in_slot | token_pct | slot_token_pct |
|---|---:|---:|---:|---:|---:|
| none | 306 | 292 | 183 | 65.9 | 72.0 |
| edit distance 1 | 80 | 73 | 42 | 17.2 | 16.5 |
| suffix strip | 64 | 60 | 16 | 13.8 | 6.3 |
| possessive or clitic -> base | 14 | 14 | 13 | 3.0 | 5.1 |

## 6. Intent to slot compatibility

432 of 3,300 intent-type cells occur in train (13.1%); median 8 types per intent; 126 cells seen once.

| min_train_count | blocked_with_gold_intent | pct | blocked_with_S1_pred_intent | pct_pred |
|---:|---:|---:|---:|---:|
| 1 | 16 | 0.8 | 72 | 3.58 |
| 2 | 35 | 1.74 | 96 | 4.77 |
| 3 | 48 | 2.39 | 112 | 5.57 |
| 5 | 71 | 3.53 | 150 | 7.46 |

Stage-1 FP spans 691: 33 (4.8%) have a type never seen with the predicted intent, 70 (10.1%) never seen with the gold intent. TP spans blocked by a predicted-intent mask: 9 of 1,334.

Post-hoc span deletion (`k` = minimum train count for a cell to be allowed), a lower bound since real masked decoding could also re-label a span:

| model | mask | k | tp | fp | fn | f1 | gain |
|---|---|---:|---:|---:|---:|---:|---:|
| Stage 1 | none | 0 | 1334 | 691 | 678 | 66.09 | 0.0 |
| Stage 1 | gold intent (oracle) | 1 | 1328 | 621 | 684 | 67.05 | 0.97 |
| Stage 1 | gold intent (oracle) | 2 | 1325 | 605 | 687 | 67.22 | 1.14 |
| Stage 1 | gold intent (oracle) | 3 | 1322 | 599 | 690 | 67.23 | 1.14 |
| Stage 1 | predicted intent | 1 | 1325 | 658 | 687 | 66.33 | 0.24 |
| Stage 1 | predicted intent | 2 | 1323 | 646 | 689 | 66.47 | 0.38 |
| Stage 1 | predicted intent | 3 | 1319 | 642 | 693 | 66.4 | 0.31 |
| A1_ft_recipe | none | 0 | 1444 | 534 | 568 | 72.38 | 0.0 |
| A1_ft_recipe | gold intent (oracle) | 1 | 1438 | 475 | 574 | 73.27 | 0.89 |
| A1_ft_recipe | gold intent (oracle) | 2 | 1432 | 466 | 580 | 73.25 | 0.87 |
| A1_ft_recipe | gold intent (oracle) | 3 | 1429 | 462 | 583 | 73.23 | 0.84 |
| A1_ft_recipe | predicted intent | 1 | 1434 | 513 | 578 | 72.44 | 0.06 |
| A1_ft_recipe | predicted intent | 2 | 1431 | 508 | 581 | 72.44 | 0.06 |
| A1_ft_recipe | predicted intent | 3 | 1427 | 504 | 585 | 72.38 | 0.0 |

Reverse direction: a predicted slot incompatible with the predicted intent flags 41 utterances (intent wrong in 43.9% vs 16.6%), only 5.2% of intent errors.

## 7. Other dev facts

- Casing and characters: no uppercase, no digit, no non-ASCII in train or dev. Numbers are spelled out. Casefolding is a no-op.
- Punctuation: `.` is 208 of 212 non-lexical dev tokens and 199 of the 212 sit inside gold slots, almost all abbreviations (`p. m.`, `u. s. a.`). 44 dev utterances end in `.`. "am" is written `am` (train 303) but "pm" is written `p. m.` (train 266, `pm` 0). Spans with a dot: boundary errors 29.2% vs 10.7%.
- Same value inside and outside slots: closed-class values are consistent (today 524 slot / 21 plain, tomorrow 220 / 2, this week 93 / 0, five am 14 / 0, eight am 35 / 0). Mixed cases: `meeting` 279 / 26 (dev 48 / 8, and Stage 1 tags 7 of those as FP), `to do` 32 / 38, `home` 26 / 31. Type conflicts: weekend time 8, date 6; trump person 13, news_topic 8; us place_name 7, currency_name 6.
- 41.7% of spurious spans (80 of 192) have a (value, type) that is a gold slot in train; 67 of the 192 fall in utterances with no gold slot (31.8% of dev has none).

Boundary convention, train:

| type | word | inside_span | outside_span | inside_share_pct |
|---|---|---:|---:|---:|
| date | this | 198 | 23 | 89.6 |
| date | next | 72 | 51 | 58.5 |
| timeofday | this | 22 | 18 | 55.0 |
| business_name | at | 20 | 19 | 51.3 |
| place_name | the | 18 | 96 | 15.8 |
| date | the | 15 | 101 | 12.9 |
| time | in | 12 | 23 | 34.3 |
| time | the | 8 | 60 | 11.8 |

Dev gold spans by value history in train:

| value history | n | share_pct | exact_hit_pct | wrong_type_pct | boundary_pct | missed_pct |
|---|---:|---:|---:|---:|---:|---:|
| seen as a slot of this type | 1484 | 73.8 | 83.0 | 3.3 | 6.1 | 6.0 |
| value has Train-OOV token | 225 | 11.2 | 12.0 | 4.9 | 18.7 | 56.4 |
| string never seen in train, words known | 166 | 8.3 | 22.9 | 1.8 | 44.6 | 16.9 |
| seen only as plain text or inside a longer slot | 83 | 4.1 | 26.5 | 6.0 | 18.1 | 45.8 |
| seen as a slot of another type only | 39 | 1.9 | 28.2 | 41.0 | 5.1 | 15.4 |
| longer than 6 tokens | 15 | 0.7 | 26.7 | 0.0 | 53.3 | 6.7 |

By length:

| tokens | n | share_pct | train_share_pct | S1_intent | A1_intent | linear_intent | S1_slot_f1 | gold_spans | errors_share_pct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25 | 1.2 | 1.1 | 64.0 | 52.0 | 48.0 | 0.0 | 3 | 2.6 |
| 2 | 79 | 3.9 | 3.8 | 69.6 | 74.7 | 73.4 | 76.6 | 25 | 6.9 |
| 3-4 | 344 | 16.9 | 17.8 | 80.8 | 83.1 | 85.5 | 61.9 | 175 | 18.9 |
| 5-8 | 1057 | 52.0 | 50.3 | 85.1 | 86.9 | 88.7 | 68.9 | 985 | 45.0 |
| 9-12 | 401 | 19.7 | 19.8 | 83.3 | 86.3 | 85.8 | 66.4 | 591 | 19.2 |
| 13-20 | 118 | 5.8 | 6.7 | 78.8 | 80.5 | 81.4 | 57.9 | 223 | 7.2 |
| 21+ | 9 | 0.4 | 0.4 | 88.9 | 88.9 | 88.9 | 43.8 | 10 | 0.3 |

Rare intents:

| intent | train | dev | S1_recall | linear_recall |
|---|---:|---:|---:|---:|
| cooking_query | 4 | 2 | 50.0 | 50.0 |
| music_dislikeness | 14 | 2 | 50.0 | 50.0 |
| audio_volume_other | 18 | 0 |  |  |
| iot_hue_lighton | 22 | 5 | 40.0 | 100.0 |
| general_greet | 25 | 2 | 100.0 | 0.0 |
| iot_wemo_on | 48 | 7 | 85.7 | 100.0 |
| music_settings | 51 | 8 | 87.5 | 87.5 |
| audio_volume_down | 52 | 8 | 62.5 | 87.5 |
| datetime_convert | 52 | 9 | 77.8 | 88.9 |
| iot_wemo_off | 52 | 5 | 80.0 | 100.0 |
| email_addcontact | 54 | 5 | 100.0 | 100.0 |
| recommendation_movies | 70 | 12 | 75.0 | 75.0 |
| general_joke | 72 | 15 | 100.0 | 100.0 |
| iot_hue_lightup | 76 | 12 | 100.0 | 91.7 |

Intents with fewer than 100 train rows cover 155 dev utterances at 84.52%, so they are not the weak spot. `audio_volume_other` (18 train) has no dev row and `music_album`, `transport_descriptor` no dev span. Train has 107 BIO labels and dev adds none, so the 108th label of the model comes from test only (the inventory in `Assets` reads all splits); I did not look at which one.

Sampling noise on dev:

| quantity | mean | se | ci_low | ci_high |
|---|---:|---:|---:|---:|
| S1 intent | 82.84 | 0.83 | 81.16 | 84.41 |
| S1 slot | 66.08 | 1.1 | 63.95 | 68.25 |
| S1 joint | 74.46 | 0.73 | 73.07 | 75.88 |
| A1 intent | 84.89 | 0.79 | 83.28 | 86.43 |
| A1 slot | 72.37 | 1.03 | 70.32 | 74.37 |
| A1 joint | 78.63 | 0.7 | 77.28 | 79.98 |
| A1-S1 intent | 2.05 | 0.53 | 1.03 | 3.1 |
| A1-S1 slot | 6.29 | 0.79 | 4.81 | 7.82 |
| A1-S1 joint | 4.17 | 0.49 | 3.25 | 5.14 |

Paired differences between finished ladder runs (snapshot during this analysis):

| run | intent | slot_f1 | joint | diff_vs_ref | paired_se | ci_low | ci_high | intent_disagreement_pct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A2_slot2 | 84.65 | 74.73 | 79.69 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| A1_fresh_lr5e-4 | 84.75 | 73.44 | 79.1 | -0.6 | 0.39 | -1.35 | 0.18 | 6.7 |
| A2_ls.2 | 84.46 | 73.5 | 78.98 | -0.71 | 0.35 | -1.4 | 0.0 | 6.0 |
| A2_drop.10 | 84.6 | 73.27 | 78.94 | -0.76 | 0.36 | -1.46 | -0.05 | 6.2 |
| A2_wd.01 | 84.51 | 73.23 | 78.87 | -0.83 | 0.38 | -1.58 | -0.07 | 6.7 |
| A2_ls0 | 84.31 | 73.19 | 78.75 | -0.94 | 0.4 | -1.77 | -0.2 | 7.5 |
| A1_ft_recipe | 84.9 | 72.38 | 78.64 | -1.05 | 0.44 | -1.9 | -0.23 | 8.3 |
| A1_fresh_recipe | 84.6 | 72.59 | 78.6 | -1.09 | 0.38 | -1.88 | -0.34 | 6.2 |
| A2_wdrop.2 | 85.15 | 71.85 | 78.5 | -1.19 | 0.41 | -1.99 | -0.38 | 7.0 |
| A2_slotdrop.2 | 84.7 | 72.05 | 78.38 | -1.31 | 0.44 | -2.21 | -0.42 | 8.0 |
| A2_drop.25 | 84.16 | 71.81 | 77.99 | -1.7 | 0.39 | -2.47 | -0.94 | 6.7 |
| stage1_baseline | 82.83 | 66.09 | 74.46 | -5.23 | 0.57 | -6.37 | -4.19 | 12.9 |

Differences below about 0.8 joint are not distinguishable from utterance sampling alone, before seed noise.

## Recommendations

Favoured: constrained decoding or CRF first (zero-parameter stray-I repair already +2.05 / +2.10 slot F1 on Stage 1 / A1), then span-aware heads for multi-token names; slot-weighted loss; a wide linear path inside the joint model (0.6-1.2M parameters for 85.4-85.9 alone), regularised heavily because the linear part reaches 99.9% on train; BPE before a large frozen table; a suffix and possessive lookup fallback (16.8% of OOV tokens, no parameters).

Disfavoured: a hard intent-to-slot mask (+0.24 / +0.06 F1 with predicted intent); a separately trained linear ensemble (rule risk, hard gating gave no gain); K beyond 46k with PCA (+8.8 points of token coverage, 41 tokens); hierarchical intent as a priority (74.5% of errors cross scenarios and those scenarios are fuzzy); rare-intent rebalancing. The soft mask and top-k options of rung E are cheap to keep but this evidence expects little from them. Deep runs disagree with each other on 6-8% of intents, A1 and linear on 12.5%, so deep-only ensembles add less intent diversity; whether any ensemble counts as one joint model needs confirming.

Process: treat dev joint differences below about 0.8 as ties, since `greedy` in `run_ladder.py` accepts any positive difference on one seed. Rung C compares the CRF, which has hard constraints, with an unconstrained argmax on the linear head, so about +1 joint of any CRF win may be the constraints alone; rung E gives that to linear heads only later. By the table in section 5, `B3_*_subword` exceeds the 20M assertion if the strict best uses the n-gram adapter, or BPE+char with `ext150k_pca100`; only `ResourceAbort` is caught, so rung B would stop there. This comes from parameter arithmetic, not from running it.

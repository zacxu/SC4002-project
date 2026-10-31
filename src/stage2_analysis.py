"""Offline development comparisons from validated, already-selected checkpoints.

No network forward pass, optimizer, or test prediction/metric file is used.
The committed selection must exist; this analysis cannot choose a new model.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from src.data import FittedMappings, read_predictions
from src.diagnostics import (analyze_predictions, load_training_development, run_diagnosis,
                             sha256, unique_pairs)
from src.runtime import configuration_hash, relative_project_path
from src.stage2_experiment import SOURCE_FILES
from src.stage2_finalization import load_selection, _validate_registered_candidates


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _case(example, baseline, improved, analysis_row: dict | None = None) -> dict:
    return {"record_id": example.record_id, "utt": example.utt, "gold": example.gold_prediction().to_dict(),
            "baseline": baseline.to_dict(), "selected_stage2": improved.to_dict(),
            "diagnosis": analysis_row}


def _fully_correct(example, prediction) -> bool:
    return (example.intent == prediction.intent and
            set(unique_pairs(example.slots)) == set(unique_pairs(prediction.slots)))


def _select_diverse(rows: list[dict], count: int = 4) -> list[dict]:
    selected = []
    predicates = [
        lambda row: bool(row["oov_tokens"]) and bool(row["false_negative_pairs"]),
        lambda row: any(slot["category"] == "same_type_wrong_boundary" for slot in row["false_negative_pairs"]),
        lambda row: any(slot["category"] == "exact_boundary_wrong_type" for slot in row["false_negative_pairs"]),
        lambda row: row["gold_intent"] != row["predicted_intent"],
        lambda row: True,
    ]
    for predicate in predicates:
        candidate = next((row for row in rows if predicate(row) and row not in selected), None)
        if candidate is not None:
            selected.append(candidate)
        if len(selected) == count:
            break
    return selected


def compare_development(root: Path) -> dict[str, Any]:
    import torch

    root = root.resolve()
    selection_path = root / "artifacts/phase5/selection.json"
    selection, _, selected_run, _ = load_selection(root, selection_path)
    baseline_selection = _read(root / "artifacts/phase4/selection.json")
    baseline = run_diagnosis(root)
    dataset = root / baseline["inputs"]["dataset"]["path"]
    splits = load_training_development(dataset)
    mappings = FittedMappings.from_dict(_read(root / "artifacts/phase2/preprocessing.json"))
    membership = _read(root / "artifacts/phase2/split_membership.json")
    for partition, examples in splits.items():
        if [row.record_id for row in examples] != membership["record_ids"][partition]:
            raise ValueError("Development comparison requires exact original train/dev identities")
    id_hashes = {"train": configuration_hash(sorted(membership["record_ids"]["train"])),
                 "dev": configuration_hash(membership["record_ids"]["dev"])}
    protocol = _read(selected_run / "protocol.json")
    plan, _ = _validate_registered_candidates(root, selection, protocol, id_hashes)
    baseline_summary = _read(root / baseline_selection["selected_run"] / "training_summary.json")
    baseline_predictions = read_predictions(root / baseline["inputs"]["predictions"]["path"])
    baseline_metadata = _read(root / baseline_selection["selected_run"] / "run.json")
    for name, expected in baseline_metadata["source_hashes"].items():
        if sha256(relative_project_path(root, name)) != expected:
            raise ValueError("Baseline trained source changed: " + name)
    baseline_checkpoint = relative_project_path(root, baseline_selection["checkpoint"])
    if sha256(baseline_checkpoint) != baseline_selection["checkpoint_sha256"]:
        raise ValueError("Frozen baseline checkpoint identity changed")
    baseline_payload = torch.load(baseline_checkpoint, map_location="cpu", weights_only=True)
    if [row.to_dict() for row in baseline_predictions] != baseline_payload["validation_predictions"]:
        raise ValueError("Frozen baseline development predictions differ from its checkpoint")
    del baseline_payload
    baseline_row = {"candidate_id": 0, "name": "baseline", "selected_stage2": False,
                    "run": baseline_selection["selected_run"], "selected_epoch": baseline_selection["selected_epoch"],
                    "epochs_trained": baseline_selection["epochs_trained"],
                    "total_parameters": baseline_summary["parameter_counts"]["total"],
                    "trainable_parameters": baseline_summary["parameter_counts"]["trainable"],
                    "training_seconds": baseline_summary["training_seconds"],
                    "intent_accuracy": baseline["overall"]["intent_accuracy"],
                    "slot_micro_f1": baseline["overall"]["slot_micro_f1"],
                    "selection_score": baseline_selection["selection_score"],
                    "dev_intent_change_percentage_points": 0.0, "dev_slot_change_percentage_points": 0.0,
                    "dev_mean_change_percentage_points": 0.0,
                    "prediction_path": baseline["inputs"]["predictions"]["path"],
                    "prediction_sha256": baseline["inputs"]["predictions"]["sha256"]}
    rows = [baseline_row]
    analyses = {"baseline": baseline}
    selected_predictions = selected_analysis = None
    for registered, candidate in zip(plan["candidates"], selection["candidates"]):
        run = relative_project_path(root, candidate["run"])
        metadata = _read(run / "run.json")
        if not set(SOURCE_FILES) <= metadata["source_hashes"].keys():
            raise ValueError("Candidate lacks required trained-source provenance")
        for name, expected in metadata["source_hashes"].items():
            if sha256(relative_project_path(root, name)) != expected:
                raise ValueError("Candidate trained source changed: " + name)
        path = run / "best_dev_predictions.jsonl"
        predictions = read_predictions(path)
        payload = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        if (payload.get("purpose") != "experiment" or payload["source_hashes"] != metadata["source_hashes"]
                or [row.to_dict() for row in predictions] != payload["validation_predictions"]):
            raise ValueError("Saved candidate development predictions differ from its certified checkpoint")
        del payload
        analysis = analyze_predictions(splits["train"], splits["dev"], predictions, mappings)
        overall = analysis["overall"]
        score = (overall["intent_accuracy"] + overall["slot_micro_f1"]) / 2
        if score != candidate["selection_score"]:
            raise ValueError("Offline development metrics differ from candidate selection")
        reload = _read(run / "reload_check.json")
        if reload["metrics"] != overall or not reload["ordered_predictions_equal"]:
            raise ValueError("Offline development counts differ from fresh checkpoint reload")
        summary = _read(run / "training_summary.json")
        row = {"candidate_id": candidate["candidate_id"], "name": registered["name"],
               "selected_stage2": candidate["candidate_id"] == selection["selected_candidate"],
               "run": candidate["run"], "selected_epoch": candidate["best_epoch"],
               "epochs_trained": candidate["epochs_trained"],
               "total_parameters": summary["parameter_counts"]["total"],
               "trainable_parameters": summary["parameter_counts"]["trainable"],
               "training_seconds": summary["training_seconds"],
               "intent_accuracy": overall["intent_accuracy"], "slot_micro_f1": overall["slot_micro_f1"],
               "selection_score": score,
               "dev_intent_change_percentage_points": 100 * (overall["intent_accuracy"] - baseline_row["intent_accuracy"]),
               "dev_slot_change_percentage_points": 100 * (overall["slot_micro_f1"] - baseline_row["slot_micro_f1"]),
               "dev_mean_change_percentage_points": 100 * (score - baseline_row["selection_score"]),
               "prediction_path": path.relative_to(root).as_posix(), "prediction_sha256": sha256(path)}
        rows.append(row)
        analyses[registered["name"]] = analysis
        if row["selected_stage2"]:
            selected_predictions, selected_analysis = predictions, analysis
    if selected_predictions is None or selected_analysis is None:
        raise ValueError("No unique selected Stage 2 candidate in the registered comparison")
    paired = {"intent_both_correct": 0, "intent_corrected": 0, "intent_regressed": 0, "intent_both_wrong": 0,
              "gold_slot_pairs_recovered": 0, "gold_slot_pairs_lost": 0,
              "utterance_both_tasks_correct_baseline": 0, "utterance_both_tasks_correct_stage2": 0,
              "utterance_wholly_corrected": 0, "utterance_regressed_from_correct": 0}
    original = {row.record_id: row for row in splits["dev"]}
    base_by_id = {row.record_id: row for row in baseline_predictions}
    improved_by_id = {row.record_id: row for row in selected_predictions}
    fully_corrected_ids = set()
    for example, initial, improved in zip(splits["dev"], baseline_predictions, selected_predictions):
        initial_intent = initial.intent == example.intent
        improved_intent = improved.intent == example.intent
        category = ("intent_both_correct" if initial_intent and improved_intent else "intent_corrected" if improved_intent
                    else "intent_regressed" if initial_intent else "intent_both_wrong")
        paired[category] += 1
        gs = set(unique_pairs(example.slots))
        before, after = gs & set(unique_pairs(initial.slots)), gs & set(unique_pairs(improved.slots))
        paired["gold_slot_pairs_recovered"] += len(after - before)
        paired["gold_slot_pairs_lost"] += len(before - after)
        initial_correct, improved_correct = _fully_correct(example, initial), _fully_correct(example, improved)
        paired["utterance_both_tasks_correct_baseline"] += int(initial_correct)
        paired["utterance_both_tasks_correct_stage2"] += int(improved_correct)
        paired["utterance_wholly_corrected"] += int(not initial_correct and improved_correct)
        paired["utterance_regressed_from_correct"] += int(initial_correct and not improved_correct)
        if not initial_correct and improved_correct:
            fully_corrected_ids.add(example.record_id)
    remaining = [_case(original[row["record_id"]], base_by_id[row["record_id"]], improved_by_id[row["record_id"]], row)
                 for row in _select_diverse(selected_analysis["error_examples"])]
    corrected_pool = [row for row in baseline["error_examples"] if row["record_id"] in fully_corrected_ids]
    corrected = [_case(original[row["record_id"]], base_by_id[row["record_id"]], improved_by_id[row["record_id"]], row)
                 for row in _select_diverse(corrected_pool)]
    if (baseline["overall"]["correct_intents"] + paired["intent_corrected"] - paired["intent_regressed"]
            != selected_analysis["overall"]["correct_intents"]):
        raise AssertionError("Paired intent changes do not reconcile with shared scorer counts")
    if (baseline["overall"]["slot_tp"] + paired["gold_slot_pairs_recovered"] - paired["gold_slot_pairs_lost"]
            != selected_analysis["overall"]["slot_tp"]):
        raise AssertionError("Paired recovered gold slots do not reconcile with shared scorer counts")
    if sum(paired[name] for name in ("intent_both_correct", "intent_corrected", "intent_regressed", "intent_both_wrong")) != len(splits["dev"]):
        raise AssertionError("Paired intent outcomes do not partition all original development examples")
    if len(remaining) < 2:
        raise ValueError("Fewer than two genuine remaining development errors available; do not invent examples")
    chosen = next(row for row in rows if row["selected_stage2"])
    return {"schema_version": 1, "partition": "dev", "test_used": False, "model_inference_performed": False,
            "selection_path": selection_path.relative_to(root).as_posix(), "selection_sha256": sha256(selection_path),
            "registered_plan_sha256": selection["registered_plan_sha256"],
            "dataset_sha256": sha256(dataset), "dev_ids_sha256": id_hashes["dev"],
            "selected_candidate": chosen["name"], "rows": rows, "analyses": analyses,
            "paired_selected_vs_baseline": paired, "remaining_error_cases": remaining, "corrected_cases": corrected,
            "limitations": ["All comparisons use the same development examples that selected the checkpoints and candidate; gains are selection-dependent.",
                            "One seed and a bounded five-candidate search do not establish statistical significance or global optimality.",
                            "Observed error categories and OOV strata are descriptive associations. Individual corrections cannot be attributed causally to a particular module.",
                            "Loss values for CRF NLL and independent-token cross entropy have different semantics despite sharing token normalization."]}


def _slots(slots: list[dict]) -> str:
    return "; ".join(f"{row['slot_type']}: {row['text']} [{row['start']}:{row['end']}]" for row in slots) or "none"


def _remaining_explanation(case: dict) -> tuple[str, str]:
    row = case["diagnosis"]
    categories = {slot["category"] for slot in row["false_negative_pairs"]}
    if row["oov_tokens"] and row["false_negative_pairs"]:
        return ("Possible cause (inference): unfamiliar entities still require semantic and phrase-level generalization; distinct character representations do not identify an unseen entity's type or exact extent by themselves.",
                "Future hypothesis: inspect train/dev entity-surface support and test a small representation or regularization change under a new development-only budget. Do not add hand-written fixes for this example.")
    if "same_type_wrong_boundary" in categories:
        return ("Possible cause (inference): the decoder chooses a coherent but incomplete or expanded phrase. Legal BIO transitions constrain tag syntax, not the exact semantic boundary required by the annotation.",
                "Future hypothesis: inspect systematic prefix/suffix boundary mistakes and compare a bounded span-aware objective or representation change using development evidence.")
    if "exact_boundary_wrong_type" in categories:
        return ("Possible cause (inference): semantic slot types are confused even though the surface boundary is correct; label usage or utterance context may be ambiguous.",
                "Future hypothesis: audit representative training-label contexts and compare contextual slot interactions without supplying gold intent during inference.")
    if row["gold_intent"] != row["predicted_intent"]:
        return ("Possible cause (inference): related intents, polarity, or implicit semantics are not resolved by the learned utterance representation. A wrong predicted intent can also influence conditioned slots, but this case alone cannot establish that mechanism.",
                "Future hypothesis: compare an alternate utterance pooling or context mechanism in a separately registered study and inspect related-intent confusions.")
    return ("Possible cause (inference): the model either misses a slot or assigns an extra slot without sufficient semantic evidence; the observed output does not uniquely identify the internal cause.",
            "Future hypothesis: inspect recurring missed/extra slot contexts and validate a bounded change on development data before another final freeze.")


def render_error_analysis(result: dict[str, Any]) -> str:
    rows = result["rows"]
    winner = next(row for row in rows if row["selected_stage2"])
    baseline = result["analyses"]["baseline"]
    selected = result["analyses"][winner["name"]]
    text = ["# Phase 5 development comparison and final-system error analysis", "",
            "This report recomputes all metrics from saved development predictions for the frozen baseline and every registered Stage 2 candidate. It validates original train/dev identities, registered configurations, complete histories, source/checkpoint hashes and recorded development reload results before analysis. It performs no model inference and reads no test predictions or metrics.", "",
            f"Development selected **{winner['name']}**, epoch **{winner['selected_epoch']} of {winner['epochs_trained']}**, with **{winner['total_parameters']:,} total parameters**. Compared with baseline, its intent / slot / mean changes are **{winner['dev_intent_change_percentage_points']:+.2f} / {winner['dev_slot_change_percentage_points']:+.2f} / {winner['dev_mean_change_percentage_points']:+.2f} percentage points**.", "",
            "## Controlled candidate comparison", "", "| Candidate | Selected epoch | Intent accuracy | Slot F1 | Dev mean | Total parameters | Workflow seconds |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        name = "**" + row["name"] + "**" if row["selected_stage2"] else row["name"]
        text.append(f"| {name} | {row['selected_epoch']} / {row['epochs_trained']} | {100 * row['intent_accuracy']:.2f}% | {100 * row['slot_micro_f1']:.2f}% | {100 * row['selection_score']:.2f}% | {row['total_parameters']:,} | {row['training_seconds']:.2f} |")
    text += ["", "| Candidate | Slot TP | Slot FP | Slot FN | Slot precision | Slot recall |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        metrics = result["analyses"][row["name"]]["overall"]
        tp, fp, fn = (metrics[key] for key in ("slot_tp", "slot_fp", "slot_fn"))
        text.append(f"| {row['name']} | {tp} | {fp} | {fn} | {100 * tp / (tp + fp):.2f}% | {100 * tp / (tp + fn):.2f}% |")
    text += ["", "Workflow timing covers training, epoch development evaluation and checkpoint/log work; it excludes setup and final reporting. Baseline and CRF loss curves should not be compared as though their slot objectives were identical. Every run uses seed 42; this controlled comparison supports a configuration choice, not a claim of statistical significance.", "",
             "![Development metrics for every candidate](artifacts/phase5/development_candidates.png)", "",
             "## Which hypotheses were supported?", ""]
    base_metrics = baseline["overall"]
    char = result["analyses"]["char"]
    crf = result["analyses"]["crf"]
    context = result["analyses"]["intent_context"]
    combo = result["analyses"]["char_crf"]
    oov = "oov/contains_oov"
    text += [f"**Character features alone offer limited support for the original OOV hypothesis.** Overall slot F1 rises from {100 * base_metrics['slot_micro_f1']:.2f}% to {100 * char['overall']['slot_micro_f1']:.2f}%, but OOV slot F1 moves only from {100 * baseline['strata'][oov]['slot_micro_f1']:.2f}% to {100 * char['strata'][oov]['slot_micro_f1']:.2f}%, while OOV intent accuracy changes from {100 * baseline['strata'][oov]['intent_accuracy']:.2f}% to {100 * char['strata'][oov]['intent_accuracy']:.2f}%. This run does not show that character features alone solve unfamiliar words.", "",
             f"**CRF alone mainly reduces extra slots.** Its FP count falls from {base_metrics['slot_fp']} to {crf['overall']['slot_fp']}, but TP falls from {base_metrics['slot_tp']} to {crf['overall']['slot_tp']} and FN rises from {base_metrics['slot_fn']} to {crf['overall']['slot_fn']}. Thus its higher F1 reflects a precision gain with a recall cost. Same-type boundary FN falls from {baseline['false_negative_categories'].get('same_type_wrong_boundary', 0)} to {crf['false_negative_categories'].get('same_type_wrong_boundary', 0)}, while no-overlap FN rises from {baseline['false_negative_categories'].get('no_overlap', 0)} to {crf['false_negative_categories'].get('no_overlap', 0)}. Boundary categories can shift between error modes; fewer boundary errors alone is not proof of better total recovery.", "",
             f"**Predicted-intent context alone is modest and mixed.** Its slot F1 is {100 * context['overall']['slot_micro_f1']:.2f}% versus baseline {100 * base_metrics['slot_micro_f1']:.2f}%, while intent accuracy remains {100 * context['overall']['intent_accuracy']:.2f}%. OOV slot F1 falls to {100 * context['strata'][oov]['slot_micro_f1']:.2f}%. In combination with characters and CRF, slot F1 is {100 * selected['overall']['slot_micro_f1']:.2f}% versus {100 * combo['overall']['slot_micro_f1']:.2f}% for character+CRF, with the same overall intent accuracy. This supports the chosen combination within the registered search, without establishing a unique causal mechanism.", "",
             f"**The selected combination improves slot precision and recall overall.** Relative to baseline, TP changes by {selected['overall']['slot_tp'] - base_metrics['slot_tp']:+d}, FP by {selected['overall']['slot_fp'] - base_metrics['slot_fp']:+d}, and FN by {selected['overall']['slot_fn'] - base_metrics['slot_fn']:+d}. Its parameter increase is {100 * (winner['total_parameters'] / rows[0]['total_parameters'] - 1):.2f}%, while its measured workflow takes {winner['training_seconds'] / rows[0]['training_seconds']:.2f} times the baseline run. The improvement is principally in slot extraction; the small net intent gain contains substantial corrected and regressed cases.", "",
             "## Did the observed weaknesses change?", "", "| Development group | N | Baseline intent | Selected intent | Baseline slot F1 | Selected slot F1 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for key in ("oov/no_oov", "oov/contains_oov", "rare_tokens/contains_train_frequency_1_or_2", "length/11_plus"):
        before, after = baseline["strata"][key], selected["strata"][key]
        text.append(f"| {key} | {before['num_examples']} | {100 * before['intent_accuracy']:.2f}% | {100 * after['intent_accuracy']:.2f}% | {100 * before['slot_micro_f1']:.2f}% | {100 * after['slot_micro_f1']:.2f}% |")
    text += ["", "![Baseline and selected candidate by OOV group](artifacts/phase5/development_oov_groups.png)", "",
             "| Gold slot group | Gold pairs | Baseline recall | Selected recall |", "| --- | ---: | ---: | ---: |"]
    for family in ("slot_surface_recall", "gold_slot_length_recall"):
        for key, before in baseline[family].items():
            after = selected[family][key]
            text.append(f"| {key} | {before['gold_pair_support']} | {100 * before['recall']:.2f}% | {100 * after['recall']:.2f}% |")
    text += ["", "| Unmatched-pair category | Baseline FN | Selected FN | Baseline FP | Selected FP |", "| --- | ---: | ---: | ---: | ---: |"]
    categories = sorted(set(baseline["false_negative_categories"]) | set(selected["false_negative_categories"]) |
                        set(baseline["false_positive_categories"]) | set(selected["false_positive_categories"]))
    for key in categories:
        text.append(f"| {key} | {baseline['false_negative_categories'].get(key, 0)} | {selected['false_negative_categories'].get(key, 0)} | {baseline['false_positive_categories'].get(key, 0)} | {selected['false_positive_categories'].get(key, 0)} |")
    text += ["", "Categories describe unmatched exact typed character-span occurrences; identical text at distinct positions remains distinct. They reconcile separately to FP and FN; they are not one-to-one linguistic error events. Surface-support strata still use type/text independently of occurrence scoring. Strata overlap across families and do not isolate causal effects.", ""]
    paired = result["paired_selected_vs_baseline"]
    text += [f"On the same development records, Stage 2 corrects **{paired['intent_corrected']}** previously wrong intents and regresses **{paired['intent_regressed']}** previously correct intents. It recovers **{paired['gold_slot_pairs_recovered']}** previously missed gold slot pairs and loses **{paired['gold_slot_pairs_lost']}** previously recovered pairs. Both tasks are fully correct, under exact pair-set scoring, on **{paired['utterance_both_tasks_correct_stage2']} / 2,033** utterances versus baseline **{paired['utterance_both_tasks_correct_baseline']} / 2,033**. There are **{paired['utterance_wholly_corrected']}** wholly corrected utterances and **{paired['utterance_regressed_from_correct']}** regressions from complete correctness.", "",
             "## Genuine remaining errors from the selected model", "", "The following original development cases were selected to illustrate different observed failure modes. Gold and saved predictions are facts; proposed causes are explicitly inferences. Offsets are end-exclusive Python character offsets."]
    for case in result["remaining_error_cases"]:
        row = case["diagnosis"]
        cause, future = _remaining_explanation(case)
        text += ["", f"### Record {case['record_id']}", "", f"Utterance: `{case['utt']}`", "",
                 f"Gold intent: `{case['gold']['intent']}`. Selected prediction: `{case['selected_stage2']['intent']}`.", "",
                 f"Gold slots: `{_slots(case['gold']['slots'])}`", "",
                 f"Baseline slots: `{_slots(case['baseline']['slots'])}`", "",
                 f"Selected slots: `{_slots(case['selected_stage2']['slots'])}`", "",
                 f"Observed OOV tokens: `{json.dumps(row['oov_tokens'], ensure_ascii=False)}`. Missing-pair categories: `{json.dumps([slot['category'] for slot in row['false_negative_pairs']])}`; extra-pair categories: `{json.dumps([slot['category'] for slot in row['false_positive_pairs']])}`.", "", cause, "", future]
    text += ["", "## Genuine corrected cases", "", "These examples were erroneous under the baseline and fully correct for both tasks under the selected Stage 2 system. They demonstrate outcomes; individual corrections cannot be attributed uniquely to one component."]
    for case in result["corrected_cases"]:
        text += ["", f"### Record {case['record_id']}", "", f"Utterance: `{case['utt']}`", "",
                 f"Intent: gold `{case['gold']['intent']}`; baseline `{case['baseline']['intent']}`; selected `{case['selected_stage2']['intent']}`.", "",
                 f"Gold slots: `{_slots(case['gold']['slots'])}`", "",
                 f"Baseline slots: `{_slots(case['baseline']['slots'])}`", "",
                 f"Selected slots: `{_slots(case['selected_stage2']['slots'])}`"]
    text += ["", "## Interpretation limits and reproduction", ""]
    text += ["- " + value for value in result["limitations"]]
    text += ["", "Future suggestions above are proposals for later work, not changes to the frozen project or an invitation to tune using final test results. Final test evaluation is handled separately and never read by this script.", "",
             "Run `python scripts/compare_stage2_development.py` after development selection exists. Full-precision candidate metrics, every per-intent/per-slot result, strata, all original error examples, paired change counts and provenance are in [development_comparison.json](artifacts/phase5/development_comparison.json). The compact table is [development_comparison.csv](artifacts/phase5/development_comparison.csv).", ""]
    return "\n".join(text)


def write_comparison(root: Path, result: dict[str, Any]) -> None:
    cache_directory = root / "artifacts/setup/matplotlib"
    cache_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_directory))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    destination = root / "artifacts/phase5"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "development_comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with (destination / "development_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    (root / "PHASE5_ERROR_ANALYSIS.md").write_text(render_error_analysis(result), encoding="utf-8")
    labels = [row["name"].replace("_", "\n") for row in result["rows"]]
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(12, 6))
    for offset, (key, label, color) in zip((-.26, 0, .26), (("intent_accuracy", "Intent accuracy", "#355C9A"),
                                                        ("slot_micro_f1", "Slot micro F1", "#238878"),
                                                        ("selection_score", "Selection mean", "#B07D23"))):
        bars = axis.bar(x + offset, [100 * row[key] for row in result["rows"]], .25, label=label, color=color)
        axis.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
    axis.set(xticks=x, xticklabels=labels, ylim=(0, 104), ylabel="Percent",
             title="Development performance at selected checkpoints (seed 42)")
    axis.legend(loc="lower right")
    axis.grid(axis="y", alpha=.2)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(destination / "development_candidates.png", dpi=160)
    plt.close(figure)
    winner = result["selected_candidate"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    groups = ("oov/no_oov", "oov/contains_oov")
    group_labels = [f"No OOV\n(n={result['analyses']['baseline']['strata'][groups[0]]['num_examples']})",
                    f"Contains OOV\n(n={result['analyses']['baseline']['strata'][groups[1]]['num_examples']})"]
    for axis, key, title in zip(axes, ("intent_accuracy", "slot_micro_f1"), ("Intent accuracy", "Slot micro F1")):
        for offset, name, color in ((-.18, "baseline", "#8B929B"), (.18, winner, "#238878")):
            bars = axis.bar(np.arange(2) + offset, [100 * result["analyses"][name]["strata"][group][key] for group in groups],
                            .35, label=name, color=color)
            axis.bar_label(bars, fmt="%.2f", padding=3)
        axis.set(xticks=np.arange(2), xticklabels=group_labels, ylim=(0, 104), title=title, ylabel="Percent")
        axis.legend(loc="lower left")
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    figure.suptitle("Development performance by train-vocabulary OOV group")
    figure.tight_layout()
    figure.savefig(destination / "development_oov_groups.png", dpi=160)
    plt.close(figure)

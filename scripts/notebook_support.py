"""Visible notebook helpers for independent reload, reporting and error analysis.

The builder places these definitions inside the submitted notebook. There is no
runtime import from this file, the checkout, or historical experiment artifacts.
"""
from __future__ import annotations

import csv
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from IPython.display import Markdown, display
from src.comparison_contract import EVALUATOR_VERSION


def nb_code_source(cell):
    value = cell.get("source", "")
    return value if isinstance(value, str) else "".join(value)


def nb_notebook_configuration(notebook):
    """Allow literal settings edits only; keep every implementation cell protected."""
    cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    metadata = notebook['metadata']['phase7']
    expected = metadata['code_cell_sha256']
    indices = metadata.get('configuration_cell_indices', [])
    tagged = [i for i, cell in enumerate(cells) if 'configuration' in cell.get('metadata', {}).get('tags', [])]
    if indices != tagged or len(indices) != 1 or any(type(i) is not int for i in indices):
        raise ValueError('Exactly one registered editable configuration cell is required')
    if len(expected) != len(cells):
        raise ValueError('Notebook cell count differs from its reviewed source manifest')
    values = {}
    for index, cell in enumerate(cells):
        text = nb_code_source(cell)
        if index not in indices:
            if hashlib.sha256(text.encode('utf-8')).hexdigest() != expected[index]:
                raise ValueError('Notebook implementation differs from its reviewed source manifest')
            continue
        for statement in ast.parse(text).body:
            if (not isinstance(statement, ast.Assign) or len(statement.targets) != 1
                    or not isinstance(statement.targets[0], ast.Name)):
                raise ValueError('Configuration must contain only literal dictionary assignments')
            name = statement.targets[0].id
            if name not in {'RUNTIME', 'STAGE1_CONFIG', 'STAGE2_CONFIG', 'RUN_OPTIONS'} or name in values:
                raise ValueError('Unexpected or duplicate configuration name: ' + name)
            try:
                values[name] = ast.literal_eval(statement.value)
            except (ValueError, TypeError) as error:
                raise ValueError('Use literal configuration values, not expressions or function calls') from error
            if not isinstance(values[name], dict):
                raise ValueError('Every configuration value must be a dictionary')
    if set(values) != {'RUNTIME', 'STAGE1_CONFIG', 'STAGE2_CONFIG', 'RUN_OPTIONS'}:
        raise ValueError('All four configuration dictionaries must be present')
    options = values['RUN_OPTIONS']
    if (not {'evaluate_test', 'dataset_path', 'device'} <= set(options)
            or set(options) - {'evaluate_test', 'dataset_path', 'device', 'verification_timeout_seconds'}
            or type(options['evaluate_test']) is not bool):
        raise ValueError('RUN_OPTIONS must declare evaluate_test, dataset_path and device')
    if options['device'] not in {'cpu', 'cuda', 'auto'}:
        raise ValueError('Choose cpu, cuda or auto for device')
    if options['dataset_path'] is not None and (not isinstance(options['dataset_path'], str) or not options['dataset_path']):
        raise ValueError('dataset_path must be None or a nonempty path string')
    timeout = options.get('verification_timeout_seconds', 300)
    if timeout is not None and (type(timeout) is not int or timeout <= 0):
        raise ValueError('verification_timeout_seconds must be positive or None for no limit')
    return values


def nb_source_identity(notebook_path):
    notebook = json.loads(Path(notebook_path).read_text(encoding="utf-8"))
    nb_notebook_configuration(notebook)
    hashes = [hashlib.sha256(nb_code_source(cell).encode('utf-8')).hexdigest()
              for cell in notebook['cells'] if cell['cell_type'] == 'code']
    return hashlib.sha256(json.dumps(hashes, separators=(',', ':')).encode('ascii')).hexdigest()


def nb_capture_configuration(notebook_path, runtime, stage1, stage2, options, output):
    notebook = json.loads(Path(notebook_path).read_text(encoding='utf-8'))
    values = {'RUNTIME': runtime, 'STAGE1_CONFIG': stage1, 'STAGE2_CONFIG': stage2, 'RUN_OPTIONS': options}
    if nb_notebook_configuration(notebook) != values:
        raise ValueError('Save the edited notebook before restarting and running every cell')
    snapshot = {**values, 'code_sha256': nb_source_identity(notebook_path),
                'configuration_sha256': hashlib.sha256(json.dumps(values, ensure_ascii=False,
                    sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()}
    nb_support_write_json(output, snapshot)
    return snapshot


def nb_run_fresh_verifier(notebook_path, checkpoint, dataset_path, code_sha256, device, timeout=300):
    """Generate ordinary Python from visible definitions and run a new process."""
    notebook = json.loads(Path(notebook_path).read_text(encoding="utf-8"))
    assert nb_source_identity(notebook_path) == code_sha256
    definitions = [nb_code_source(cell) for cell in notebook["cells"]
                   if cell["cell_type"] == "code" and "definitions" in cell.get("metadata", {}).get("tags", [])]
    # Future imports must precede all other code in the generated Python module.
    source = "from __future__ import annotations\n\n" + "\n\n".join(
        text.replace("from __future__ import annotations\n", "") for text in definitions)
    driver = '''
if __name__ == "__main__":
    checkpoint_path = Path(sys.argv[1]).resolve()
    dataset_path = Path(sys.argv[2]).resolve()
    code_identity, requested_device = sys.argv[3], sys.argv[4]
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    configure_reproducibility(payload["runtime"])
    torch.set_default_dtype(torch.float32)
    fresh_splits = load_massive(dataset_path)
    fresh_mappings = fit_mappings(fresh_splits["train"])
    if fresh_mappings.to_dict() != payload["mappings"]:
        raise ValueError("Checkpoint mappings do not equal a fresh original-train-only fit")
    fresh_dev = prepare_split(fresh_splits["dev"], fresh_mappings)
    fresh_chars = payload.get("char_to_id", {})
    if fresh_chars and fit_char_vocab(prepare_split(fresh_splits["train"], fresh_mappings)) != fresh_chars:
        raise ValueError("Checkpoint character map differs from original training data")
    report = nb_verify_development(checkpoint_path, fresh_dev, fresh_mappings,
        fresh_chars, requested_device, file_sha256(dataset_path), code_identity,
        fresh_python_process=True)
    print(json.dumps(report, indent=2, allow_nan=False))
'''
    worker = Path(checkpoint).parent / "fresh_development_verifier.py"
    worker.write_text(source + driver, encoding="utf-8")
    environment = os.environ.copy()
    environment.update(PYTHONHASHSEED="42", CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONIOENCODING="utf-8")
    environment.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-B", str(worker), str(Path(checkpoint).resolve()),
                             str(Path(dataset_path).resolve()), code_sha256, str(device)],
                            cwd=Path(checkpoint).parent, env=environment,
                            capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    (worker.parent / "fresh_process_console.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError("Fresh-process verification failed; see fresh_process_console.txt\n" + result.stderr[-2500:])
    report = json.loads((worker.parent / "reload_check.json").read_text(encoding="utf-8"))
    if not report["passed"] or not report["fresh_python_process"]:
        raise ValueError("A genuine fresh Python verification process is required")
    print("FRESH_PROCESS_DEVELOPMENT_VERIFIED", json.dumps(report, allow_nan=False))
    return report


def nb_support_write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def nb_display_diagnosis(training, development, predictions, mappings, output_path):
    diagnosis = analyze_predictions(training, development, predictions, mappings)
    nb_support_write_json(output_path, diagnosis)
    columns = ("contains_oov", "no_oov")
    lines = ["| Development group | Utterances | Intent accuracy | Slot micro F1 |",
             "| --- | ---: | ---: | ---: |"]
    for name in columns:
        row = diagnosis["strata"]["oov/" + name]
        lines.append(f"| {name} | {row['num_examples']} | {100*row['intent_accuracy']:.2f}% | {100*row['slot_micro_f1']:.2f}% |")
    display(Markdown("\n".join(lines)))
    print("Development missed-slot categories:", diagnosis["false_negative_categories"])
    print("Development intent confusions:", diagnosis["intent_confusions"][:5])
    return diagnosis


def nb_build_result_table(stage1, stage2, output_root):
    rows, full_precision = [], {}
    for label, summary in (("Stage-1 system", stage1), ("Stage-2 system", stage2)):
        full_precision[label] = {}
        for part in (("dev", "test") if summary.get('test') is not None else ("dev",)):
            metrics = summary[part]["metrics"]
            full_precision[label][part] = metrics
            rows.append({"model_version": label, "split": "Validation" if part == "dev" else "Test",
                         "total_parameters": summary["parameter_counts"]["total"],
                         "trainable_parameters": summary["parameter_counts"]["trainable"],
                         "intent_accuracy_percent": f"{100*metrics['intent_accuracy']:.2f}",
                         "slot_micro_f1_percent": f"{100*metrics['slot_micro_f1']:.2f}",
                         "epochs_trained": summary["epochs_trained"], "selected_epoch": summary["selected_epoch"]})
    report = {"rows": rows, "full_precision": full_precision, "evaluator_version": EVALUATOR_VERSION,
              "source": "This notebook's own checkpoint-reloaded final evaluations",
              "stage2_minus_stage1_percentage_points": {
                  part: {key: 100 * (stage2[part]["metrics"][key] - stage1[part]["metrics"][key])
                         for key in ("intent_accuracy", "slot_micro_f1")}
                  for part in (("dev", "test") if stage1.get('test') is not None else ("dev",))}}
    output_root = Path(output_root)
    nb_support_write_json(output_root / "results.json", report)
    with (output_root / "results.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys = list(rows[0])
    table = "| " + " | ".join(keys) + " |\n| " + " | ".join("---" for _ in keys) + " |\n"
    table += "\n".join("| " + " | ".join(str(row[key]) for key in keys) + " |" for row in rows)
    display(Markdown(table))
    print("UNROUNDED_METRIC_DELTAS", json.dumps(report["stage2_minus_stage1_percentage_points"]))
    return report


def nb_show_error_cases(training, development, predictions, mappings, output_path):
    analysis = analyze_predictions(training, development, predictions, mappings)
    candidates = analysis["error_examples"]
    chosen = []
    # Choose genuine, varied remaining development errors without touching test.
    predicates = (lambda row: bool(row["oov_tokens"]),
                  lambda row: any(slot["category"] == "same_type_wrong_boundary" for slot in row["false_negative_pairs"]),
                  lambda row: row["gold_intent"] != row["predicted_intent"], lambda row: True)
    for predicate in predicates:
        for row in candidates:
            if predicate(row) and row["record_id"] not in {item["record_id"] for item in chosen}:
                chosen.append(dict(row))
                break
    if len(chosen) < 2:
        raise ValueError("At least two actual remaining errors are required for analysis")
    for row in chosen:
        hypotheses = []
        if row["oov_tokens"]:
            hypotheses.append("Inference: unfamiliar words lack trained word-specific embeddings; character spelling features may not establish their entity meaning.")
        if any(slot["category"] == "same_type_wrong_boundary" for slot in row["false_negative_pairs"]):
            hypotheses.append("Inference: the structured decoder still under- or over-extends this entity; valid BIO transitions do not guarantee the correct semantic boundary.")
        if row["gold_intent"] != row["predicted_intent"]:
            hypotheses.append("Inference: short utterance/context ambiguity and limited examples for this phrasing may contribute; this example alone cannot establish a cause.")
        if not hypotheses:
            hypotheses.append("Inference: entity typing or omission remains difficult for this phrasing despite character and intent context features.")
        row["possible_causes_labelled_as_inferences"] = hypotheses
        row["possible_future_work"] = "Investigate targeted training-only support/error patterns and calibration with development data in a separate future study; do not tune using this run's test results."
        display(Markdown("**Actual remaining development error: record " + row["record_id"] + "**\n\n```json\n" +
                         json.dumps(row, ensure_ascii=False, indent=2) + "\n```"))
    report = {"source": "New notebook Stage 2 saved development predictions", "cases": chosen,
              "total_remaining_error_utterances": analysis["utterances_with_any_error"]}
    nb_support_write_json(output_path, report)
    return report

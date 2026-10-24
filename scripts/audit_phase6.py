"""Build the verified script reference for notebook conversion, without inference.

Run after the independent Phase 6 reviews and unit suite. Existing stage evidence
is read only. A new exclusive output directory contains generated report rows,
the artifact inventory, effective settings, and a byte-verified source snapshot.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from urllib.parse import unquote


STAGES = (("Stage-1 system", "stage1", "artifacts/phase4/final_baseline", "artifacts/phase4/selection.json"),
          ("Stage-2 system", "stage2", "artifacts/phase5/final_stage2", "artifacts/phase5/selection.json"))
REVIEW_FILES = ("artifacts/phase6/data_model_review.json", "artifacts/phase6/evidence_review.json",
                "artifacts/phase6/notebook_handoff_review.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def project_path(root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.anchor or not value.strip():
        raise ValueError("Expected a nonempty project-relative evidence path")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Evidence path escapes the project")
    return resolved


def verify_freeze(root: Path, directory: Path, summary: dict) -> dict:
    """Require original and snapshot bytes, not just matching manifest claims."""
    manifest_path = directory / "freeze_manifest.json"
    manifest = read_json(manifest_path)
    if sha256(manifest_path) != summary["freeze_manifest_sha256"]:
        raise ValueError("Final summary has a different freeze manifest")
    for name, expected in manifest["snapshot_file_hashes"].items():
        if (sha256(project_path(root, name)) != expected
                or sha256(project_path(directory / "snapshot", name)) != expected):
            raise ValueError(f"Frozen evidence bytes changed: {name}")
    for name, expected in manifest["source_hashes"].items():
        if sha256(project_path(root, name)) != expected:
            raise ValueError(f"Trained source bytes changed: {name}")
    if sha256(project_path(root, summary["checkpoint"])) != summary["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint bytes changed")
    return {"snapshot_files": len(manifest["snapshot_file_hashes"]),
            "source_files": len(manifest["source_hashes"]), "all_bytes_match": True}


def validate_final(summary: dict) -> None:
    stage = summary["stage"]
    if (summary["status"] != "complete" or summary["test_model_evaluations"] != 1
            or summary["test_evaluation_finished"] is not True
            or summary["model_weights_unchanged"] is not True
            or summary["checkpoint_bytes_unchanged"] is not True
            or summary[f"{stage}_training_and_tuning_closed"] is not True):
        raise ValueError("Both frozen final evaluations must already be complete")
    counts = summary["parameter_counts"]
    if (type(counts["total"]) is not int or type(counts["trainable"]) is not int
            or not 0 < counts["trainable"] <= counts["total"] <= 20_000_000):
        raise ValueError("Invalid complete-system parameter accounting")
    if (type(summary["selected_epoch"]) is not int or type(summary["epochs_trained"]) is not int
            or not 1 <= summary["selected_epoch"] <= summary["epochs_trained"]):
        raise ValueError("Selected epoch differs from the completed training budget")
    reload = summary["fresh_process_reload"]
    if (reload["passed"] is not True or reload["ordered_predictions_equal"] is not True
            or reload["absolute_score_difference"] != 0 or reload["num_examples"] != 2033):
        raise ValueError("Missing exact checkpoint-reloaded development evidence")
    for part, expected in (("dev", 2033), ("test", 2974)):
        result, metrics = summary[part], summary[part]["metrics"]
        if (result["num_examples"] != expected or metrics["num_examples"] != expected
                or result["partitions"] != [part] or result["limited"] is not False
                or result["max_batches"] is not None):
            raise ValueError("Final result omits original examples")
        for key in ("intent_accuracy", "slot_micro_f1"):
            value = metrics[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Final metrics must be finite fractions")
        for key in ("correct_intents", "slot_tp", "slot_fp", "slot_fn"):
            if type(metrics[key]) is not int or metrics[key] < 0:
                raise ValueError("Metric counts must be nonnegative integers")
        denominator = 2 * metrics["slot_tp"] + metrics["slot_fp"] + metrics["slot_fn"]
        expected_f1 = 2 * metrics["slot_tp"] / denominator if denominator else 0.0
        if (metrics["correct_intents"] / expected != metrics["intent_accuracy"]
                or expected_f1 != metrics["slot_micro_f1"]):
            raise ValueError("Final metric fractions disagree with their integer counts")


def report_rows(root: Path) -> tuple[list[dict], dict, dict]:
    rows, sources, integrity = [], {}, {}
    for label, stage, directory_name, selection_name in STAGES:
        directory = project_path(root, directory_name)
        summary_path = directory / "final_summary.json"
        summary, selection = read_json(summary_path), read_json(project_path(root, selection_name))
        validate_final(summary)
        if (summary["stage"] != stage or selection["selection_split"] != "dev"
                or selection["test_used_for_selection"] is not False
                or any(summary[key] != selection[key] for key in
                       ("selected_run", "checkpoint", "checkpoint_sha256", "selected_epoch", "epochs_trained", "selection_score"))):
            raise ValueError("Final checkpoint does not match development-only selection")
        integrity[stage] = verify_freeze(root, directory, summary)
        sources[stage] = {"final_summary": directory_name + "/final_summary.json",
                          "final_summary_sha256": sha256(summary_path), "selection": selection_name,
                          "checkpoint": summary["checkpoint"], "checkpoint_sha256": summary["checkpoint_sha256"],
                          "selected_configuration": selection["selected_configuration"]}
        for part in ("dev", "test"):
            metrics = summary[part]["metrics"]
            rows.append({"model_version": label, "split": "Validation" if part == "dev" else "Test",
                         "total_parameters": summary["parameter_counts"]["total"],
                         "trainable_parameters": summary["parameter_counts"]["trainable"],
                         "intent_accuracy_percent": f"{100 * metrics['intent_accuracy']:.2f}",
                         "slot_micro_f1_percent": f"{100 * metrics['slot_micro_f1']:.2f}",
                         "epochs_trained": summary["epochs_trained"], "selected_epoch": summary["selected_epoch"],
                         "num_examples": summary[part]["num_examples"],
                         "training_workflow_seconds": summary["training_seconds"]})
    return rows, sources, integrity


def inventory(root: Path) -> list[dict]:
    """Inventory all project evidence; omit caches, environment and Phase 6 outputs.

    Phase 6's review/input artifacts are instead frozen explicitly in its source
    reference. Avoiding self-inventory keeps the inventory digest well-defined.
    """
    paths = set()
    for folder in ("src", "scripts", "tests", "configs", "data", "artifacts"):
        for path in (root / folder).rglob("*"):
            if (not path.is_file() or path.suffix in {".pyc", ".pyo"}
                    or "__pycache__" in path.parts or (root / "artifacts/phase6") in path.parents):
                continue
            paths.add(path)
    paths.update(path for path in root.iterdir() if path.is_file())
    result = []
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Inventory file resolves outside project")
        category = ("historical_snapshot" if "/snapshot/" in relative else
                    "experiment_run" if relative.startswith("artifacts/runs/") else
                    "phase_evidence" if relative.startswith("artifacts/") else
                    "original_data" if relative.startswith("data/") else
                    "test_source" if relative.startswith("tests/") else
                    "implementation" if relative.startswith(("src/", "scripts/")) else
                    "configuration" if relative.startswith("configs/") else "project_document")
        result.append({"path": relative, "category": category, "size_bytes": path.stat().st_size,
                       "sha256": sha256(path)})
    return result


def verify_registered_runs(root: Path, expected: set[str]) -> int:
    """Detect later research runs, including ones the frozen baseline CLI permits."""
    actual = set()
    for directory in (root / "artifacts/runs").iterdir():
        metadata_path = directory / "run.json"
        if directory.is_dir() and metadata_path.is_file() and read_json(metadata_path).get("purpose") == "experiment":
            actual.add(directory.relative_to(root).as_posix())
    if actual != expected:
        raise ValueError(f"Research run inventory differs from registration: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    return len(actual)


def verify_document_links(root: Path) -> dict:
    """Check current and historical Markdown before taking the reference copy."""
    documents = list(root.glob("*.md")) + list((root / "artifacts/phase6").glob("*.md"))
    checked = 0
    for path in documents:
        content = path.read_text(encoding="utf-8")
        if "\ufffd" in content:
            raise ValueError(f"Invalid replacement character in documentation: {path.name}")
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", content):
            target = match.group(1).strip().strip("<>")
            if re.match(r"^(?:https?://|mailto:|#)", target):
                continue
            destination = unquote(target.split("#", 1)[0])
            if not (path.parent / destination).exists():
                # Generated reference targets are checked after their creation.
                generated = (root / "artifacts/phase6/reference").resolve()
                if (path.parent / destination).resolve().parent == generated:
                    continue
                raise ValueError(f"Broken local documentation link: {path.name}: {destination}")
            checked += 1
    return {"documents_checked": len(documents), "existing_local_links_checked": checked}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Preserve the existing Phase 6 reference; choose a fresh output directory")
    reviews = {name: read_json(project_path(root, name)) for name in REVIEW_FILES}
    if any(report.get("status") != "passed" for report in reviews.values()):
        raise ValueError("All three independent reviews must pass before freezing")
    evidence_review = reviews["artifacts/phase6/evidence_review.json"]
    evidence_hashes = evidence_review["evidence_hashes_extended"]
    if not evidence_hashes or any(sha256(project_path(root, name)) != expected for name, expected in evidence_hashes.items()):
        raise ValueError("Independent review is stale: reviewed evidence bytes changed")
    registered_runs = {candidate["run"] for _, _, _, selection_name in STAGES
                       for candidate in read_json(project_path(root, selection_name))["candidates"]}
    if len(registered_runs) != 8:
        raise ValueError("Expected the completed three-plus-five candidate research budget")
    verified_run_count = verify_registered_runs(root, registered_runs)
    document_check = verify_document_links(root)
    log_path = root / "artifacts/phase6/unit_tests_final.txt"
    log = log_path.read_text(encoding="utf-8")
    matches = re.findall(r"Ran (\d+) tests? in [0-9.]+s", log)
    if not matches or int(matches[-1]) < 147 or re.search(r"\nOK\s*$", log) is None or "FAILED (" in log:
        raise ValueError("Missing successful current full-suite evidence")
    runtime = read_json(root / "artifacts/phase6/runtime_check.json")
    if runtime["package_mismatches"] or runtime["pip_check_exit"] != 0 or not runtime["cuda_available"]:
        raise ValueError("Declared local runtime did not pass verification")
    rows, sources, integrity = report_rows(root)
    prior = read_json(root / "artifacts/phase5/stage_comparison.json")["rows"]
    for row, old in zip(rows, prior, strict=True):
        if any(row[key] != old[key] for key in ("intent_accuracy_percent", "slot_micro_f1_percent", "total_parameters",
                                               "trainable_parameters", "epochs_trained", "selected_epoch", "num_examples")):
            raise ValueError("Regenerated results disagree with the preserved comparison")
    artifacts = inventory(root)
    protocol = read_json(root / "configs/protocol.json")
    configs = {stage: {"path": value["selected_configuration"],
                       "file_sha256": sha256(root / value["selected_configuration"]),
                       "configuration": read_json(root / value["selected_configuration"])} for stage, value in sources.items()}
    frozen_paths = {item["path"] for item in artifacts if item["path"].startswith(("src/", "scripts/", "tests/", "configs/"))}
    frozen_paths.update(path.name for path in root.iterdir() if path.is_file() and path.suffix in {".md", ".txt", ".pdf"})
    frozen_paths.update(REVIEW_FILES)
    frozen_paths.update(("artifacts/phase6/data_model_review.md", "artifacts/phase6/evidence_review.md",
                         "artifacts/phase6/collaboration_record.md", "artifacts/phase6/evidence_review_offline_log.txt"))
    frozen_paths.update(("artifacts/phase6/unit_tests_final.txt", "artifacts/phase6/runtime_check.json", "artifacts/phase6/assignment_text.txt"))
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "snapshot"
    snapshot_hashes = {}
    for name in sorted(frozen_paths):
        source, target = project_path(root, name), project_path(snapshot, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        snapshot_hashes[name] = sha256(source)
        if sha256(target) != snapshot_hashes[name]:
            raise ValueError("Phase 6 snapshot copy differs from original bytes")
    write_json(output / "artifact_inventory.json", {"scope": "All root files and src/scripts/tests/configs/data/artifacts files, except Python bytecode and artifacts/phase6; Phase 6 inputs are separately snapshotted. The .venv and hidden directories are excluded.",
                                                     "file_count": len(artifacts), "total_size_bytes": sum(row["size_bytes"] for row in artifacts), "files": artifacts})
    write_csv(output / "artifact_inventory.csv", artifacts)
    write_json(output / "results.json", {"rows": rows, "sources": sources,
                                          "epoch_column": "epochs_trained is the total budget; selected_epoch is separately retained",
                                          "scope": "Verified script results. Phase 7 must generate genuine notebook outputs before the final report."})
    write_csv(output / "results.csv", rows)
    columns = list(rows[0])
    markdown = "| " + " | ".join(columns) + " |\n| " + " | ".join("---" for _ in columns) + " |\n"
    markdown += "\n".join("| " + " | ".join(str(row[key]) for key in columns) + " |" for row in rows) + "\n"
    (output / "results.md").write_text(markdown, encoding="utf-8")
    write_json(output / "frozen_settings.json", {"common_protocol": protocol, "common_protocol_file_sha256": sha256(root / "configs/protocol.json"),
                                                  "effective_stages": configs, "selection_and_checkpoint_provenance": sources,
                                                  "historical_pending_fields": "architecture and training_budget are superseded by these effective stage settings; group details remain pending"})
    write_json(output / "implementation_manifest.json", {"schema_version": 1, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                                                        "snapshot_file_hashes": snapshot_hashes, "file_count": len(snapshot_hashes),
                                                        "prior_stage_freezes_verified": integrity,
                                                        "purpose": "Verified script reference for Phase 7; includes new audit tooling separately from unchanged trained source identities"})
    generated = {path.name: sha256(path) for path in output.iterdir() if path.is_file()}
    report = {"schema_version": 1, "status": "passed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "readiness": "Ready to implement and execute the self-contained notebook in Phase 7",
              "independent_reviews": {name: {"sha256": sha256(root / name), "status": report["status"]} for name, report in reviews.items()},
              "unit_tests_passed": int(matches[-1]), "locked_packages_verified": runtime["locked_packages_checked"],
              "registered_research_runs_verified": verified_run_count, "reviewed_evidence_hashes_verified": len(evidence_hashes),
              "documentation_preflight": document_check,
              "final_results_rows": 4, "artifact_files_inventoried": len(artifacts), "reference_snapshot_files": len(snapshot_hashes),
              "prior_freezes": integrity, "generated_file_sha256": generated,
              "model_forward_passes_during_this_audit": 0, "training_updates_during_this_audit": 0,
              "new_final_test_evaluations": 0, "next_phase": 7,
              "limits": ["Saved artifacts and preserved code corroborate training provenance; this is not an external process monitor.",
                         "Exact type/text set scoring is a documented project interpretation; lecturer scorer parity is unverified.",
                         "Single-seed bounded search; test intent regression retained.",
                         "Notebook clean execution, source embedding and path adaptation remain Phase 7 work.",
                         "Group details, verified human contributions and full chat export remain Phase 8 work."]}
    write_json(output / "audit_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", default="artifacts/phase6/reference")
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = run(root, project_path(root, args.output_dir))
    print(json.dumps({key: result[key] for key in ("status", "unit_tests_passed", "artifact_files_inventoried", "reference_snapshot_files", "final_results_rows", "next_phase")}, indent=2))


if __name__ == "__main__":
    main()

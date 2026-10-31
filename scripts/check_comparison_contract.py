"""Check both inline scorers and original dataset identity without training.

This is a contract/data audit, not notebook execution or model acceptance. It
does not download pretrained vectors or evaluate trained models on test data.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import types
import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.comparison_contract import EVALUATOR_VERSION, load_verified_massive
from src.data import load_massive, tokenize_with_offsets, decode_bio, gold_bio_tags
from scripts.validate_notebook import validate_notebook_source


def source(cell):
    value = cell.get("source", "")
    return value if isinstance(value, str) else "".join(value)


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check(root=ROOT, dataset=None):
    dataset = Path(dataset or root / "data/1.0/data/en-US.jsonl")
    canonical_source = (root / "src/comparison_contract.py").read_text(encoding="utf-8")
    canonical_sha = sha(canonical_source)
    splits, identity = load_verified_massive(dataset)
    references = load_massive(dataset)
    roundtrips = 0
    for partition, rows in references.items():
        assert [row.record_id for row in rows] == [row["id"] for row in splits[partition]]
        for row in rows:
            tokens = tokenize_with_offsets(row.utt)
            assert decode_bio(row.utt, tokens, gold_bio_tags(row, tokens)) == row.slots
            roundtrips += 1

    def slot(start, end, text="room", label="place"):
        return {"slot_type": label, "text": text, "start": start, "end": end}

    def record(slots, intent="fixture", record_id="fixture", utt="room room"):
        return {"record_id": record_id, "intent": intent, "utt": utt, "slots": slots}

    cases = [
        ("wrong_repeated_occurrence", [record([slot(0, 4)])], [record([slot(5, 9)])], (0, 1, 1)),
        ("two_occurrences_one_found", [record([slot(0, 4), slot(5, 9)])], [record([slot(0, 4)])], (1, 0, 1)),
        ("exact_occurrences", [record([slot(0, 4), slot(5, 9)])], [record([slot(0, 4), slot(5, 9)])], (2, 0, 0)),
        ("wrong_type", [record([slot(0, 4)])], [record([slot(0, 4, label="other")])], (0, 1, 1)),
        ("wrong_boundary", [record([slot(0, 4)])], [record([slot(0, 3, text="roo")])], (0, 1, 1)),
        ("zero_slot_denominator", [record([])], [record([])], (0, 0, 0)),
    ]
    results, data_manifests = {}, []
    for filename in ("main.ipynb", "NLP_Final_project.ipynb"):
        notebook = json.loads((root / filename).read_text(encoding="utf-8"))
        cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        contract_cells = [cell for cell in cells if "comparison_contract" in cell.get("metadata", {}).get("tags", [])]
        assert len(contract_cells) == 1, filename
        assert source(contract_cells[0]) == canonical_source, filename
        metadata = notebook["metadata"]["comparison_contract"]
        assert metadata["source_sha256"] == canonical_sha and metadata["evaluator_version"] == EVALUATOR_VERSION
        hashes = [sha(source(cell)) for cell in cells]
        if filename == 'main.ipynb':
            validate_notebook_source(nbformat.reads(json.dumps(notebook), as_version=4))
        else:
            assert hashes == notebook["metadata"]["evaluator_v2"]["code_cell_sha256"]
        for number, cell in enumerate(cells):
            tree = ast.parse(source(cell))
            compile(tree, f"{filename}:code{number + 1}", "exec")
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith(("src", "scripts")) and not node.level
                if isinstance(node, ast.Import):
                    assert not any(alias.name.startswith(("src", "scripts")) for alias in node.names)

        module = types.ModuleType("_contract_check_" + filename.replace(".", "_"))
        sys.modules[module.__name__] = module
        try:
            if filename == "main.ipynb":
                # Execute only reviewed inline definitions, never bootstrap/training cells.
                for cell in cells:
                    if "definitions" in cell.get("metadata", {}).get("tags", []):
                        exec(compile(source(cell), filename + ":definitions", "exec"), module.__dict__)
            else:
                exec(compile(canonical_source, filename + ":contract", "exec"), module.__dict__)
            _, manifest = module.load_verified_massive(dataset)
            assert manifest == identity
            data_manifests.append(manifest)
            fixture_results = {}
            for name, gold, predicted, expected in cases:
                scored = module.score_exact_spans(gold, predicted)
                assert tuple(scored[key] for key in ("slot_tp", "slot_fp", "slot_fn")) == expected
                if filename == "main.ipynb":
                    def typed(rows):
                        return [module.IntentSlotPrediction(row["record_id"], row["intent"],
                                tuple(module.SlotSpan(**item) for item in row["slots"]), utt=row["utt"]) for row in rows]
                    assert module.evaluate_predictions(typed(gold), typed(predicted)).to_dict() == scored
                fixture_results[name] = scored
            results[filename] = {"notebook_sha256": hashlib.sha256((root / filename).read_bytes()).hexdigest(),
                "inline_contract_sha256": canonical_sha, "code_cells_compile": len(cells),
                "stale_output_cells": sum(bool(cell.get("outputs")) for cell in cells),
                "fresh_training": "pending", "scorer_fixtures": fixture_results}
        finally:
            del sys.modules[module.__name__]
    assert data_manifests[0] == data_manifests[1]
    assert results["main.ipynb"]["scorer_fixtures"] == results["NLP_Final_project.ipynb"]["scorer_fixtures"]
    return {"schema_version": 2, "status": "passed", "evaluator_version": EVALUATOR_VERSION,
        "scope": "Shared inline scorer, adapter fixtures and exact original dataset identity; no model training or test inference",
        "dataset": identity, "inline_contracts_byte_identical": True,
        "dataset_manifests_identical": True, "bilstm_original_annotation_roundtrips": roundtrips,
        "notebooks": results, "current_model_results": "pending_fresh_training_and_development_selection",
        "historical_results": "results/historical_v1/"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/1.0/data/en-US.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "results/comparison_contract.json")
    args = parser.parse_args()
    report = check(ROOT, args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "dataset_counts": report["dataset"]["counts"],
                      "notebooks": list(report["notebooks"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()

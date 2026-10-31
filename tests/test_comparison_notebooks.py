"""Cross-notebook scoring and source-integrity regression checks; no training."""
import hashlib
import json
from pathlib import Path
import unittest
import nbformat

from scripts.validate_notebook import score_saved_predictions, validate_notebook_source, ValidationError

ROOT = Path(__file__).resolve().parents[1]


class ComparisonNotebookTests(unittest.TestCase):
    def test_both_notebooks_embed_identical_reviewed_contract(self):
        expected = (ROOT / "src/comparison_contract.py").read_text(encoding="utf-8")
        for name in ("main.ipynb", "NLP_Final_project.ipynb"):
            with self.subTest(notebook=name):
                notebook = json.loads((ROOT / name).read_text(encoding="utf-8"))
                code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
                contract = [cell for cell in code if "comparison_contract" in cell.get("metadata", {}).get("tags", [])]
                self.assertEqual(len(contract), 1)
                source = contract[0]["source"]
                self.assertEqual(source if isinstance(source, str) else "".join(source), expected)
                self.assertEqual(notebook["metadata"]["comparison_contract"]["source_sha256"],
                                 hashlib.sha256(expected.encode()).hexdigest())
                observed = [hashlib.sha256((cell["source"] if isinstance(cell["source"], str)
                            else "".join(cell["source"])).encode()).hexdigest() for cell in code]
                if name == 'main.ipynb':
                    self.assertEqual(validate_notebook_source(nbformat.reads(json.dumps(notebook), as_version=4))['code_cell_sha256'], observed)
                else:
                    self.assertEqual(observed, notebook["metadata"]["evaluator_v2"]["code_cell_sha256"])

    def test_current_source_registration_matches_actual_files(self):
        manifest = json.loads((ROOT / "configs/evaluation_v2_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["evaluator_version"], "exact_character_span_v2")
        for name, expected in manifest["snapshot_file_hashes"].items():
            with self.subTest(path=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)

    def test_independent_acceptance_scorer_rejects_wrong_identical_occurrence(self):
        gold = [{"record_id": "fixture", "intent": "a", "utt": "room room",
                 "slots": [{"slot_type": "place", "text": "room", "start": 0, "end": 4}]}]
        predicted = [{"record_id": "fixture", "intent": "a",
                      "slots": [{"slot_type": "place", "text": "room", "start": 5, "end": 9}]}]
        metrics = score_saved_predictions(gold, predicted)
        self.assertEqual((metrics["slot_tp"], metrics["slot_fp"], metrics["slot_fn"]), (0, 1, 1))

    def test_independent_acceptance_scorer_counts_repeated_gold_slots(self):
        slots = [{"slot_type": "place", "text": "room", "start": start, "end": start + 4} for start in (0, 5)]
        gold = [{"record_id": "fixture", "intent": "a", "utt": "room room", "slots": slots}]
        predicted = [{"record_id": "fixture", "intent": "a", "slots": slots[:1]}]
        metrics = score_saved_predictions(gold, predicted)
        self.assertEqual((metrics["slot_tp"], metrics["slot_fp"], metrics["slot_fn"]), (1, 0, 1))

    def test_independent_acceptance_rejects_duplicate_identical_slots(self):
        slot = {"slot_type": "place", "text": "room", "start": 0, "end": 4}
        gold = [{"record_id": "fixture", "intent": "a", "utt": "room", "slots": [slot]}]
        with self.assertRaisesRegex(ValidationError, "Duplicate identical"):
            score_saved_predictions(gold, [{"record_id": "fixture", "intent": "a", "slots": [slot, slot]}])

    def test_independent_acceptance_rejects_end_beyond_raw_text(self):
        gold = [{"record_id": "fixture", "intent": "a", "utt": "room", "slots": []}]
        predicted = [{"record_id": "fixture", "intent": "a",
                      "slots": [{"slot_type": "place", "text": "room", "start": 0, "end": 5}]}]
        with self.assertRaisesRegex(ValidationError, "offsets"):
            score_saved_predictions(gold, predicted)


if __name__ == "__main__":
    unittest.main()

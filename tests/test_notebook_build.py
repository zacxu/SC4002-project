"""Adversarial bookkeeping fixtures for standalone notebook acceptance."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

import nbformat

from scripts.validate_notebook import (ValidationError, evidence_path, validate_code,
                                       validate_execution_outputs, validate_history,
                                       validate_notebook_source)
from scripts.build_notebook import build


def fixture_notebook():
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("value = 1\nprint(value)")])
    notebook.metadata["phase7"] = {"code_cell_sha256": [hashlib.sha256(notebook.cells[0].source.encode()).hexdigest()]}
    return notebook


class NotebookAcceptanceTests(unittest.TestCase):
    def test_builder_refuses_to_clear_executed_submission_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "main.ipynb"
            notebook = fixture_notebook()
            notebook.cells[0].execution_count = 1
            notebook.cells[0].outputs = [nbformat.v4.new_output("stream", name="stdout", text="genuine preserved output\n")]
            nbformat.write(notebook, destination)
            original = destination.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "Preserve the executed notebook"):
                build(root, destination)
            self.assertEqual(destination.read_bytes(), original)

    def test_hidden_repository_imports_and_dynamic_execution_are_rejected(self):
        for source in ("from src.models import JointBiLSTM", "import scripts.runner", "from .model import Model",
                       "import sys\nsys.path.insert(0, '..')", "exec(downloaded_code)",
                       "import importlib\nimportlib.import_module(module)", "%run scripts/run_stage2.py"):
            with self.subTest(source=source), self.assertRaises(ValidationError):
                validate_code(source)

    def test_historical_file_dependencies_are_rejected_but_provenance_is_allowed(self):
        for source in ("open('configs/stage1.json')", "Path('artifacts/phase5/selection.json')",
                       "torch.load('artifacts/runs/old/best.pt')", "Path('C:/Users/Admin/Desktop/old/model.pt')"):
            with self.subTest(source=source), self.assertRaises(ValidationError):
                validate_code(source)
        validate_code("PROVENANCE = {'original': 'src/models.py'}\nfrom pathlib import Path\nPath('notebook_outputs/current/best.pt')")

    def test_visible_source_changes_including_harmless_edits_break_manifest(self):
        notebook = fixture_notebook()
        notebook.metadata["phase7"]["builder_source_manifest"] = {"original_module.py": {"inline_code_sha256": "provenance"}}
        self.assertTrue(validate_notebook_source(notebook)["source_manifest_matches"])
        notebook.cells[0].source += "\n# changed after registration"
        with self.assertRaises(ValidationError):
            validate_notebook_source(notebook)

    def test_missing_or_malformed_source_manifest_is_rejected(self):
        notebook = fixture_notebook()
        del notebook.metadata["phase7"]
        with self.assertRaises(ValidationError):
            validate_notebook_source(notebook)
        notebook.metadata["phase7"] = {"code_cell_sha256": ["not-a-sha256"]}
        with self.assertRaises(ValidationError):
            validate_notebook_source(notebook)

    def test_error_outputs_cannot_be_hidden_by_later_success(self):
        notebook = fixture_notebook()
        notebook.cells[0].execution_count = 1
        notebook.cells[0].outputs = [nbformat.v4.new_output("error", ename="RuntimeError", evalue="training failed", traceback=[])]
        with self.assertRaisesRegex(ValidationError, "training failed"):
            validate_execution_outputs(notebook)

    def test_unexecuted_repeated_and_out_of_order_cells_are_rejected(self):
        for counts in ((None,), (2, 1), (1, 1)):
            notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("pass", execution_count=count) for count in counts])
            with self.subTest(counts=counts), self.assertRaises(ValidationError):
                validate_execution_outputs(notebook)

    def test_disk_only_plots_do_not_satisfy_visible_output_gate(self):
        notebook = fixture_notebook()
        notebook.cells[0].execution_count = 1
        notebook.cells[0].outputs = [nbformat.v4.new_output("stream", name="stdout", text="saved curves.png")]
        with self.assertRaisesRegex(ValidationError, "inline"):
            validate_execution_outputs(notebook)

    def test_evidence_paths_cannot_escape_clean_execution_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "evidence.json").write_text("{}", encoding="utf-8")
            self.assertEqual(evidence_path(root, "evidence.json"), root / "evidence.json")
            for name in ("../evidence.json", "C:/Users/Admin/evidence.json", "dir\\evidence.json", ""):
                with self.subTest(name=name), self.assertRaises(ValidationError):
                    evidence_path(root, name)

    def test_truncated_training_and_nonfinite_losses_fail_before_checkpoint_acceptance(self):
        row = {"train_num_examples": 11514, "train_num_tokens": 83008,
               "dev_num_examples": 2033, "dev_num_tokens": 14522,
               "dev_intent_accuracy": 0.5, "dev_slot_micro_f1": 0.5, "selection_score": 0.5}
        for split in ("train", "dev"):
            for component in ("loss", "intent_loss", "slot_loss"):
                row[f"{split}_{component}"] = 1.0
        history = [dict(deepcopy(row), epoch=epoch) for epoch in range(1, 21)]
        self.assertEqual(validate_history(history)["selected_epoch"], 1)
        for mutate in (lambda rows: rows.pop(), lambda rows: rows[5].update(train_num_examples=11513),
                       lambda rows: rows[5].update(dev_loss=float("nan")),
                       lambda rows: rows[5].update(selected_checkpoint=True)):
            modified = deepcopy(history)
            mutate(modified)
            with self.assertRaises(ValidationError):
                validate_history(modified)


if __name__ == "__main__":
    unittest.main()

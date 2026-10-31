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


def configurable_notebook():
    notebook = fixture_notebook()
    configuration = ("RUNTIME = {'seed': 42}\n"
                     "STAGE1_CONFIG = {'training': {'learning_rate': 0.001, 'max_epochs': 2, 'batch_size': 64}}\n"
                     "STAGE2_CONFIG = {'training': {'learning_rate': 0.001, 'max_epochs': 2, 'batch_size': 64}}\n"
                     "RUN_OPTIONS = {'evaluate_test': False, 'dataset_path': None, 'device': 'cuda'}\n")
    notebook.cells.append(nbformat.v4.new_code_cell(configuration, metadata={"tags": ["configuration"]}))
    notebook.metadata["phase7"] = {
        "configuration_cell_indices": [1],
        "code_cell_sha256": [hashlib.sha256(cell.source.encode()).hexdigest() for cell in notebook.cells],
    }
    return notebook


class NotebookAcceptanceTests(unittest.TestCase):
    def test_literal_configuration_edit_changes_effective_identity_without_unprotecting_code(self):
        notebook = configurable_notebook()
        original = validate_notebook_source(notebook)
        notebook.cells[1].source = notebook.cells[1].source.replace("'learning_rate': 0.001", "'learning_rate': 0.0005")
        edited = validate_notebook_source(notebook)
        self.assertTrue(edited["configuration_edited"])
        self.assertNotEqual(original["implementation_sha256"], edited["implementation_sha256"])
        self.assertNotEqual(original["configuration_sha256"], edited["configuration_sha256"])
        self.assertEqual(edited["configuration"]["STAGE1_CONFIG"]["training"]["learning_rate"], 0.0005)
        notebook.cells[0].source += "\n# changed definitions"
        with self.assertRaisesRegex(ValidationError, "builder manifest"):
            validate_notebook_source(notebook)

    def test_configuration_tag_cannot_be_moved_to_an_unregistered_cell(self):
        notebook = configurable_notebook()
        notebook.cells[0].metadata["tags"] = ["configuration"]
        notebook.cells[1].metadata["tags"] = []
        with self.assertRaisesRegex(ValidationError, "registered editable"):
            validate_notebook_source(notebook)

    def test_configuration_rejects_executable_expressions_and_extra_assignments(self):
        for mutate in (lambda text: text.replace("{'seed': 42}", "dict(seed=42)"),
                       lambda text: text + "unexpected = {}\n",
                       lambda text: text + "RUNTIME = {'seed': 7}\n",
                       lambda text: text.replace("RUN_OPTIONS = ", "ANOTHER_NAME = ")):
            notebook = configurable_notebook()
            notebook.cells[1].source = mutate(notebook.cells[1].source)
            with self.subTest(source=notebook.cells[1].source), self.assertRaises(ValidationError):
                validate_notebook_source(notebook)

    def test_configurable_history_uses_requested_epoch_budget_and_batch_size(self):
        row = {"train_num_examples": 11514, "train_num_tokens": 83008, "train_num_batches": 180,
               "dev_num_examples": 2033, "dev_num_tokens": 14522, "dev_num_batches": 32,
               "dev_intent_accuracy": 0.5, "dev_slot_micro_f1": 0.5, "selection_score": 0.5}
        for split in ("train", "dev"):
            for component in ("loss", "intent_loss", "slot_loss"):
                row[f"{split}_{component}"] = 1.0
        history = [dict(row, epoch=epoch) for epoch in (1, 2)]
        result = validate_history(history, max_epochs=2, batch_size=64)
        self.assertEqual((result["epochs_trained"], result["selected_epoch"]), (2, 1))
        with self.assertRaisesRegex(ValidationError, "batch traversal"):
            validate_history(history, max_epochs=2, batch_size=32)
        with self.assertRaisesRegex(ValidationError, "configured ordered epochs"):
            validate_history(history, max_epochs=3, batch_size=64)
        with self.assertRaisesRegex(ValidationError, "positive integers"):
            validate_history(history, max_epochs=True, batch_size=64)

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

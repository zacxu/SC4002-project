"""Standalone notebook integration and preservation checks; no model training."""
from pathlib import Path
import ast
import hashlib
import json
import tempfile
import unittest

import nbformat

from src.revised_transformer_training import notebook_identity
from scripts.build_revised_transformer_notebook import build

ROOT = Path(__file__).resolve().parents[1]


class RevisedNotebookTests(unittest.TestCase):
    def test_shared_evaluator_is_inlined_without_changes(self):
        notebook = nbformat.read(ROOT / 'NLP_Final_project_revised.ipynb', as_version=4)
        contract = [cell.source for cell in notebook.cells if cell.cell_type == 'code'
                    and 'comparison_contract' in cell.metadata.get('tags', [])]
        self.assertEqual(contract, [(ROOT / 'src/comparison_contract.py').read_text(encoding='utf-8')])

    def test_all_notebook_code_is_python_without_repository_imports(self):
        notebook = nbformat.read(ROOT / 'NLP_Final_project_revised.ipynb', as_version=4)
        for cell in notebook.cells:
            if cell.cell_type != 'code': continue
            tree = ast.parse(cell.source)
            compile(tree, 'revised-notebook', 'exec')
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith(('src', 'scripts')))
                    self.assertFalse(node.level)
                elif isinstance(node, ast.Import):
                    self.assertFalse(any(alias.name.startswith(('src', 'scripts')) for alias in node.names))

    def test_recorded_original_notebook_identity_matches_preserved_input(self):
        notebook = nbformat.read(ROOT / 'NLP_Final_project_revised.ipynb', as_version=4)
        self.assertEqual(notebook.metadata.revised_transformer.original_notebook_sha256,
                         hashlib.sha256((ROOT / 'NLP_Final_project.ipynb').read_bytes()).hexdigest())
        self.assertEqual(len(notebook_identity(ROOT / 'NLP_Final_project_revised.ipynb')), 64)

    def test_builder_cannot_clear_genuine_execution_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'NLP_Final_project_revised.ipynb'
            notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('print(1)',
                execution_count=1, outputs=[nbformat.v4.new_output('stream', name='stdout', text='1\n')])])
            nbformat.write(notebook, destination)
            before = destination.read_bytes()
            with self.assertRaisesRegex(FileExistsError, 'Preserve'):
                build(ROOT, destination)
            self.assertEqual(destination.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()

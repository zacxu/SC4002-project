"""Standalone notebook integration and preservation checks; no model training."""
from pathlib import Path
import ast
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

import nbformat

from src.revised_transformer_training import notebook_identity, notebook_configuration
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

    def test_regularized_notebook_is_separate_standalone_and_stage2_only(self):
        protected = [ROOT / name for name in ('NLP_Final_project.ipynb', 'NLP_Final_project_revised.ipynb')]
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            destination = directory / 'NLP_Final_project_revised_regularized.ipynb'
            build(ROOT, destination, word_dropout_alpha=.25, intent_label_smoothing=.05)
            values = notebook_configuration(destination)
            self.assertNotIn('word_dropout_alpha', values['STAGE1_CONFIG']['training'])
            self.assertNotIn('intent_label_smoothing', values['STAGE1_CONFIG']['training'])
            self.assertEqual(values['STAGE2_CONFIG']['training']['word_dropout_alpha'], .25)
            self.assertEqual(values['STAGE2_CONFIG']['training']['intent_label_smoothing'], .05)
            self.assertFalse(values['RUN_OPTIONS']['evaluate_test'])
            notebook = nbformat.read(destination, as_version=4)
            for cell in notebook.cells:
                if cell.cell_type != 'code':
                    continue
                tree = ast.parse(cell.source)
                compile(tree, 'regularized-notebook', 'exec')
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        self.assertFalse(node.level or (node.module or '').split('.')[0] in {'src', 'scripts'})
                    elif isinstance(node, ast.Import):
                        self.assertFalse(any(alias.name.split('.')[0] in {'src', 'scripts'} for alias in node.names))
            bootstrap = next(cell.source for cell in notebook.cells
                             if 'bootstrap' in cell.metadata.get('tags', []))
            namespace = {}
            with patch('pathlib.Path.cwd', return_value=directory), patch.dict('os.environ',
                    {'PYTHONHASHSEED': '42', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
                exec(compile(bootstrap, 'bootstrap', 'exec'), namespace)
            self.assertEqual(namespace['NOTEBOOK_PATH'], destination)
            self.assertTrue((directory / 'transformer_run_artifacts').is_dir())
            self.assertIn(destination.name, '\n'.join(cell.source for cell in notebook.cells if cell.cell_type == 'markdown'))
        self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected})

    def test_default_builder_keeps_legacy_training_options_and_rejects_invalid_regularizers(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'review.ipynb'
            build(ROOT, destination)
            values = notebook_configuration(destination)
            self.assertEqual(values['STAGE1_CONFIG']['training'], values['STAGE2_CONFIG']['training'])
            self.assertNotIn('word_dropout_alpha', values['STAGE2_CONFIG']['training'])
            for options in ({'word_dropout_alpha': -.1}, {'word_dropout_alpha': True},
                            {'word_dropout_alpha': float('nan')}, {'intent_label_smoothing': 1.},
                            {'intent_label_smoothing': False}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    build(ROOT, destination, **options)

    def test_early_stopping_revision_preserves_stage1_and_registers_stage2_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'NLP_Final_project_revised_early_stopping.ipynb'
            build(ROOT, destination, word_dropout_alpha=.25, intent_label_smoothing=.05,
                  early_stopping=True)
            values = notebook_configuration(destination)
            first, second = values['STAGE1_CONFIG']['training'], values['STAGE2_CONFIG']['training']
            self.assertEqual(first['max_epochs'], 20)
            self.assertNotIn('early_stopping_patience', first)
            for name, expected in {'max_epochs':60, 'schedule_epochs':20,
                    'early_stopping_patience':8, 'early_stopping_min_epochs':20,
                    'early_stopping_min_delta':.0005, 'word_dropout_alpha':.25,
                    'intent_label_smoothing':.05}.items():
                self.assertEqual(second[name], expected)
            self.assertFalse(values['RUN_OPTIONS']['evaluate_test'])
            self.assertEqual(second['learning_rate'], first['learning_rate'])
            notebook = nbformat.read(destination, as_version=4)
            self.assertIn('patience anchor', '\n'.join(c.source for c in notebook.cells
                                                     if c.cell_type == 'markdown'))
            for cell in notebook.cells:
                if cell.cell_type == 'code': compile(cell.source, str(destination), 'exec')
            with self.assertRaises(ValueError): build(ROOT, destination, early_stopping=1)


if __name__ == '__main__':
    unittest.main()

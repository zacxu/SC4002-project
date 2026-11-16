"""Structural and standalone checks for the notebook-only readability refactor.

No model is constructed, trained, or evaluated. Definition cells are compiled
and imported into a clean namespace; data/bootstrap/training cells are not run.
"""
from copy import deepcopy
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import nbformat

from scripts import refactor_transformer_notebook as refactor

ROOT = Path(__file__).resolve().parents[1]
BIO_UNUSED = {'asdict', 'dataclass', 'fields', 'math', 'Any', 'F',
              'pack_padded_sequence', 'pad_packed_sequence'}


def codes(document):
    return [cell for cell in document.cells if cell.cell_type == 'code']


def is_tagged(cell, tag):
    return tag in cell.metadata.get('tags', [])


def dump_nodes(nodes):
    return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)


def substantive_nodes(source, prune_bio=False):
    """Independent normalization with an explicit, narrow import-removal allowance."""
    retained = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom) and node.module == '__future__':
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if prune_bio and isinstance(node, (ast.Import, ast.ImportFrom)):
            node.names = [alias for alias in node.names
                if (alias.asname or alias.name.split('.')[0]) not in BIO_UNUSED]
            if not node.names:
                continue
        retained.append(node)
    return retained


def configuration(document):
    cell = next(cell for cell in codes(document) if is_tagged(cell, 'configuration'))
    return {node.targets[0].id: ast.literal_eval(node.value)
            for node in ast.parse(refactor.source_text(cell)).body}


def print_label(node):
    if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == 'print'
            and node.value.args and isinstance(node.value.args[0], ast.Constant)):
        return node.value.args[0].value
    return None


class NotebookReadabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        backup = ROOT/'artifacts/notebook_readability_refactor/prechange/NLP_Final_project_revised_early_stopping.ipynb'
        if backup.is_file():
            cls.original = nbformat.read(backup, as_version=4)
        else:
            # Fresh checkouts need no ignored run artifacts to exercise the refactor.
            from scripts.build_revised_transformer_notebook import build
            path = Path(cls.temporary.name)/'NLP_Final_project_revised_early_stopping.ipynb'
            build(output=path, word_dropout_alpha=.25, intent_label_smoothing=.05, early_stopping=True)
            cls.original = nbformat.read(path, as_version=4)
        cls.before = json.dumps(cls.original, sort_keys=True)
        cls.revised = refactor.refactor_document(cls.original)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_input_is_unchanged_and_old_execution_evidence_is_cleared(self):
        self.assertEqual(json.dumps(self.original, sort_keys=True), self.before)
        self.assertEqual(self.revised.metadata.revised_transformer.status, 'pending_execution')
        self.assertTrue(all(cell.execution_count is None and not cell.outputs for cell in codes(self.revised)))
        nbformat.validate(self.revised)
        self.assertGreater(len(codes(self.revised)), len(codes(self.original)))

    def test_every_module_executable_ast_is_unchanged_except_allowed_bio_imports(self):
        previous = self.original.metadata.revised_transformer.source_provenance
        old_flat, new_flat = [], []
        for name in refactor.MODULES:
            before = next(cell for cell in codes(self.original) if is_tagged(cell, 'definitions')
                          and refactor.text_hash(refactor.source_text(cell)) == previous[name]['inline_sha256'])
            after = [cell for cell in codes(self.revised) if cell.metadata.get('source_module') == name]
            self.assertTrue(after, name)
            old_nodes = substantive_nodes(refactor.source_text(before), prune_bio=name == 'src/stage2_model.py')
            new_nodes = [node for cell in after for node in substantive_nodes(refactor.source_text(cell))]
            self.assertEqual(dump_nodes(old_nodes), dump_nodes(new_nodes), name)
            old_flat.extend(old_nodes); new_flat.extend(new_nodes)
        self.assertEqual(dump_nodes(old_flat), dump_nodes(new_flat))

    def test_every_function_and_class_ast_is_identical(self):
        def definitions(document):
            return {node.name: ast.dump(node, include_attributes=False)
                for cell in codes(document) if is_tagged(cell, 'definitions')
                for node in ast.parse(refactor.source_text(cell)).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        original, revised = definitions(self.original), definitions(self.revised)
        self.assertEqual(original, revised)
        self.assertIn('verify_checkpoint', revised)
        self.assertIn('DevelopmentPatience', revised)

    def test_exactly_eight_unused_bio_import_bindings_are_removed(self):
        def bindings(cell):
            return {alias.asname or alias.name.split('.')[0]
                for node in ast.parse(refactor.source_text(cell)).body
                if isinstance(node, (ast.Import, ast.ImportFrom)) and getattr(node, 'module', None) != '__future__'
                for alias in node.names}
        old = next(cell for cell in codes(self.original) if any(isinstance(node, ast.ClassDef) and node.name == 'BIOCRF'
                   for node in ast.parse(refactor.source_text(cell)).body))
        new = next(cell for cell in codes(self.revised) if cell.metadata.get('source_module') == 'src/stage2_model.py')
        self.assertEqual(bindings(old)-bindings(new), BIO_UNUSED)
        self.assertEqual(bindings(new), {'Sequence', 'torch', 'nn'})

    def test_configuration_cache_paths_bootstrap_and_training_calls_are_preserved(self):
        self.assertEqual(configuration(self.original), configuration(self.revised))
        for key in ('dataset_path', 'embedding_archive', 'evaluate_test'):
            self.assertEqual(configuration(self.original)['RUN_OPTIONS'][key], configuration(self.revised)['RUN_OPTIONS'][key])
        for tag in ('bootstrap', 'configuration', 'stage1', 'stage2'):
            old = next(cell for cell in codes(self.original) if is_tagged(cell, tag))
            new = next(cell for cell in codes(self.revised) if is_tagged(cell, tag))
            self.assertEqual(dump_nodes(substantive_nodes(refactor.source_text(old))),
                             dump_nodes(substantive_nodes(refactor.source_text(new))), tag)

    def test_only_approved_runtime_data_and_embedding_presentation_changes(self):
        old = [cell for cell in codes(self.original) if not cell.metadata.get('tags', [])]
        new = [cell for cell in codes(self.revised) if not cell.metadata.get('tags', [])]
        self.assertEqual(len(old), 3); self.assertEqual(len(new), 3)
        old_nodes, new_nodes = substantive_nodes(refactor.source_text(old[0])), substantive_nodes(refactor.source_text(new[0]))
        imports = [node for node in new_nodes if isinstance(node, ast.Import)
                   and [(alias.name, alias.asname) for alias in node.names] == [('pandas', 'pd')]]
        self.assertEqual(len(imports), 1)
        self.assertEqual(dump_nodes(old_nodes), dump_nodes([node for node in new_nodes if node not in imports]))
        for index, old_labels, new_labels in (
            (1, {'VERIFIED_DATASET:', 'TRAIN_FITTED_DATA_STATISTICS:'}, {'Verified split sizes:', 'Vocabulary:'}),
            (2, {'COMMON_WORD_INITIALIZATION:'}, {'Word embeddings:'})):
            before = substantive_nodes(refactor.source_text(old[index]))
            after = substantive_nodes(refactor.source_text(new[index]))
            self.assertEqual({print_label(n) for n in before if print_label(n) in old_labels}, old_labels)
            self.assertEqual({print_label(n) for n in after if print_label(n) in new_labels}, new_labels)
            self.assertEqual(dump_nodes([n for n in before if print_label(n) not in old_labels]),
                             dump_nodes([n for n in after if print_label(n) not in new_labels]))

    def test_result_presentation_keeps_real_metrics_errors_and_completion(self):
        before = substantive_nodes(refactor.source_text(next(c for c in codes(self.original) if is_tagged(c, 'completion'))))
        after = substantive_nodes(refactor.source_text(next(c for c in codes(self.revised) if is_tagged(c, 'completion'))))
        allowed = ast.parse("""display(pd.DataFrame(results['rows']))
cases = json.loads((RUN_ROOT / 'error_cases.json').read_text(encoding='utf-8'))['cases']
error_rows = [{'utterance': case['utt'], 'gold intent': case['gold']['intent'],
               'predicted intent': case['prediction']['intent'], 'gold slots': case['gold']['slots'],
               'predicted slots': case['prediction']['slots']} for case in cases[:5]]
display(pd.DataFrame(error_rows))
print('Complete metrics and error records:', RUN_ROOT)
""").body
        allowed_dumps = [ast.dump(node, include_attributes=False) for node in allowed]
        removed = [node for node in after if ast.dump(node, include_attributes=False) in allowed_dumps]
        self.assertEqual(dump_nodes(removed), dump_nodes(allowed))
        retained = [node for node in after if node not in removed]
        self.assertEqual(dump_nodes(retained), dump_nodes([node for node in before if print_label(node)
            not in {'GENERATED_RESULTS:', 'GENUINE_DEVELOPMENT_ERRORS:'}]))

    def test_canonical_evaluator_is_exact_and_no_project_imports_remain(self):
        canonical = [cell for cell in codes(self.revised) if is_tagged(cell, 'comparison_contract')]
        self.assertEqual(len(canonical), 1)
        self.assertEqual(refactor.source_text(canonical[0]), (ROOT/'src/comparison_contract.py').read_text(encoding='utf-8'))
        for cell in codes(self.revised):
            for node in ast.walk(ast.parse(refactor.source_text(cell))):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse(node.level)
                    self.assertNotIn((node.module or '').split('.')[0], {'src', 'scripts', 'black'})
                elif isinstance(node, ast.Import):
                    self.assertTrue(all(alias.name.split('.')[0] not in {'src', 'scripts', 'black'} for alias in node.names))

    def test_v2_manifest_binds_ordered_fragments_and_all_cell_hashes(self):
        metadata = self.revised.metadata.revised_transformer
        self.assertEqual(metadata.definition_layout_version, 2)
        self.assertEqual(metadata.code_cell_sha256, [refactor.text_hash(refactor.source_text(c)) for c in codes(self.revised)])
        self.assertEqual(metadata.configuration_cell_indices,
                         [i for i, c in enumerate(codes(self.revised)) if is_tagged(c, 'configuration')])
        all_definitions = [c for c in codes(self.revised) if is_tagged(c, 'definitions')]
        self.assertTrue(all(c.metadata.get('source_module') in refactor.MODULES for c in all_definitions))
        for name, evidence in metadata.source_provenance.items():
            fragments = [refactor.source_text(c) for c in all_definitions if c.metadata.source_module == name]
            self.assertEqual(evidence['file_sha256'], hashlib.sha256((ROOT/name).read_bytes()).hexdigest())
            self.assertEqual(evidence['inline_cell_sha256'], [refactor.text_hash(s) for s in fragments])
            self.assertEqual(evidence['inline_sha256'], refactor.text_hash('\n\n'.join(fragments)))
            normalized = dump_nodes([node for s in fragments for node in substantive_nodes(s)])
            self.assertEqual(evidence['semantic_ast_sha256'], refactor.text_hash(normalized))

    def test_clean_definition_namespace_and_fresh_verifier_assembly_compile(self):
        definitions = [refactor.source_text(c) for c in codes(self.revised) if is_tagged(c, 'definitions')]
        source = 'from __future__ import annotations\n\n'+'\n\n'.join(
            item.replace('from __future__ import annotations\n', '') for item in definitions)
        source += "\nif __name__ == '__main__':\n    verify_checkpoint(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], fresh_process=True)\n"
        compiled = compile(source, '<fresh-verifier-check>', 'exec')
        module = types.ModuleType('_standalone_readability_definitions')
        sys.modules[module.__name__] = module
        import torch
        try:
            with patch.object(torch.nn.Module, '__call__', side_effect=AssertionError('No model forward permitted')), \
                    patch.object(torch.optim, 'AdamW', side_effect=AssertionError('No optimizer permitted')):
                exec(compiled, vars(module))
            names = [node.name for c in codes(self.original) if is_tagged(c, 'definitions')
                     for node in ast.parse(refactor.source_text(c)).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
            self.assertTrue(all(callable(vars(module).get(name)) for name in names))
            self.assertEqual(module.EVALUATOR_VERSION, 'exact_character_span_v2')
        finally:
            sys.modules.pop(module.__name__, None)

    def test_formatted_code_line_width_exceptions_are_only_literals_or_canonical(self):
        for index, cell in enumerate(codes(self.revised)):
            if is_tagged(cell, 'comparison_contract'):
                continue
            source = refactor.source_text(cell)
            literals = {line for node in ast.walk(ast.parse(source)) if isinstance(node, ast.JoinedStr)
                or isinstance(node, ast.Constant) and isinstance(node.value, str)
                for line in range(node.lineno, node.end_lineno+1)}
            for line, text in enumerate(source.splitlines(), 1):
                if len(text) > 100:
                    self.assertIn(line, literals, f'Cell {index}, line {line}: {text}')

    def test_normalization_does_not_hide_algorithm_or_function_docstring_changes(self):
        self.assertEqual(refactor.normalize_ast('"old module prose"\nx = 1'),
                         refactor.normalize_ast('from __future__ import annotations\n"new prose"\nx = 1'))
        self.assertNotEqual(refactor.normalize_ast('def f():\n    return 1'),
                            refactor.normalize_ast('def f():\n    return 2'))
        self.assertNotEqual(refactor.normalize_ast('def f():\n    "first"\n    return 1'),
                            refactor.normalize_ast('def f():\n    "second"\n    return 1'))

    def test_refactor_rejects_changed_numerical_source_and_already_split_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'src').mkdir()
            (root/'src/comparison_contract.py').write_text('# changed implementation\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Implementation changed before refactoring'):
                refactor.refactor_document(self.original, root=root)
        with self.assertRaises(ValueError):
            refactor.refactor_document(self.revised)


if __name__ == '__main__':
    unittest.main()

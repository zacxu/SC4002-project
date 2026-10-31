"""Future notebook configuration edits, provenance and tuning-output guards."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat

from scripts.notebook_support import (nb_notebook_configuration, nb_source_identity,
    nb_capture_configuration, nb_build_result_table)
from scripts.notebook_workflow import nb_validate_config
from scripts.validate_notebook import validate_notebook_source

ROOT = Path(__file__).resolve().parents[1]


class NotebookRerunTests(unittest.TestCase):
    def setUp(self):
        self.notebook = nbformat.read(ROOT / 'main.ipynb', as_version=4)
        self.values = deepcopy(nb_notebook_configuration(self.notebook))

    def write_values(self, values):
        cell = next(cell for cell in self.notebook.cells if cell.cell_type == 'code'
                    and 'configuration' in cell.metadata.get('tags', []))
        cell.source = '\n\n'.join(name + ' = ' + repr(value) for name, value in values.items()) + '\n'

    def test_supported_parameter_edits_need_no_manifest_rewrite(self):
        original = self.notebook.metadata.phase7.code_cell_sha256[:]
        self.values['STAGE1_CONFIG']['training'].update(learning_rate=.0005, batch_size=64, max_epochs=3)
        self.values['STAGE1_CONFIG']['model'].update(hidden_dim=64, dropout=.1)
        self.write_values(self.values)
        self.assertEqual(nb_notebook_configuration(self.notebook), self.values)
        self.assertTrue(validate_notebook_source(self.notebook)['configuration_edited'])
        self.assertEqual(self.notebook.metadata.phase7.code_cell_sha256, original)
        nb_validate_config('stage1', self.values['STAGE1_CONFIG'], self.values['RUNTIME'])

    def test_changed_configuration_changes_effective_execution_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'main.ipynb'
            nbformat.write(self.notebook, path)
            before = nb_source_identity(path)
            self.values['RUN_OPTIONS']['evaluate_test'] = False
            self.write_values(self.values)
            nbformat.write(self.notebook, path)
            self.assertNotEqual(nb_source_identity(path), before)

    def test_implementation_edits_remain_rejected(self):
        self.notebook.cells[1].source += '\n# unexpected implementation edit\n'
        with self.assertRaisesRegex(ValueError, 'implementation'):
            nb_notebook_configuration(self.notebook)

    def test_configuration_cannot_execute_code_or_hide_extra_assignments(self):
        cell = next(c for c in self.notebook.cells if c.cell_type == 'code'
                    and 'configuration' in c.metadata.get('tags', []))
        initial = cell.source
        for extra in ('\nimport os', '\nOTHER = {}', '\nRUN_OPTIONS = dict()'):
            with self.subTest(extra=extra):
                cell.source = initial + extra
                with self.assertRaises(ValueError):
                    nb_notebook_configuration(self.notebook)

    def test_unsaved_configuration_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'main.ipynb'
            nbformat.write(self.notebook, path)
            self.values['STAGE1_CONFIG']['training']['max_epochs'] += 1
            with self.assertRaisesRegex(ValueError, 'Save'):
                nb_capture_configuration(path, *self.values.values(), Path(temporary) / 'configuration.json')

    def test_snapshot_records_actual_values_and_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'main.ipynb'
            nbformat.write(self.notebook, path)
            output = Path(temporary) / 'run_configuration.json'
            result = nb_capture_configuration(path, *self.values.values(), output)
            expected = hashlib.sha256(json.dumps(self.values, ensure_ascii=False,
                sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            self.assertEqual(result['configuration_sha256'], expected)
            self.assertEqual(result['code_sha256'], nb_source_identity(path))
            self.assertEqual(json.loads(output.read_text()), result)

    def test_ignored_loss_weights_and_optimizers_are_not_advertised_as_supported(self):
        for stage in ('stage1', 'stage2'):
            for key, value in (('optimizer', 'SGD'), ('intent_loss_weight', 2), ('slot_loss_weight', .5)):
                config = deepcopy(self.values[stage.upper() + '_CONFIG'])
                config['training'][key] = value
                with self.subTest(stage=stage, key=key), self.assertRaisesRegex(ValueError, 'AdamW'):
                    nb_validate_config(stage, config, self.values['RUNTIME'])

    def test_invalid_numeric_settings_fail(self):
        for key, value in (('batch_size', True), ('max_epochs', 0), ('learning_rate', float('nan')),
                           ('gradient_clip_norm', 0), ('weight_decay', -1)):
            config = deepcopy(self.values['STAGE1_CONFIG'])
            config['training'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                nb_validate_config('stage1', config, self.values['RUNTIME'])

    def test_long_run_reload_timeout_can_be_changed_without_code_edits(self):
        for timeout in (900, None):
            self.values['RUN_OPTIONS']['verification_timeout_seconds'] = timeout
            self.write_values(self.values)
            self.assertEqual(nb_notebook_configuration(self.notebook)['RUN_OPTIONS']['verification_timeout_seconds'], timeout)
        self.values['RUN_OPTIONS']['verification_timeout_seconds'] = 0
        self.write_values(self.values)
        with self.assertRaisesRegex(ValueError, 'verification_timeout_seconds'):
            nb_notebook_configuration(self.notebook)

    def test_development_only_reporting_has_no_test_rows(self):
        metrics = {'intent_accuracy': .5, 'slot_micro_f1': .4}
        summary = {'dev': {'metrics': metrics}, 'test': None, 'epochs_trained': 3,
                   'selected_epoch': 2, 'parameter_counts': {'total': 10, 'trainable': 10}}
        with tempfile.TemporaryDirectory() as temporary, patch('scripts.notebook_support.display'):
            result = nb_build_result_table(summary, deepcopy(summary), temporary)
        self.assertEqual(len(result['rows']), 2)
        self.assertTrue(all(row['split'] == 'Validation' for row in result['rows']))
        self.assertNotIn('test', result['stage2_minus_stage1_percentage_points'])


if __name__ == '__main__':
    unittest.main()

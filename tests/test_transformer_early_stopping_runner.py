"""Experiment isolation guards; no dataset acquisition or model training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_transformer_early_stopping as runner
from src.revised_transformer_training import notebook_configuration


class EarlyStoppingRunnerTests(unittest.TestCase):
    def test_registered_source_and_prior_evidence_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'trainer.py';old=root/'earlier.pt'
            source.write_text('registered source');old.write_bytes(b'prior evidence')
            study=root/'study';study.mkdir()
            (study/'protocol.json').write_text(json.dumps({'protocol':{
                'source_sha256':{'trainer.py':runner.sha(source)}}}))
            (study/'protected_inputs.json').write_text(json.dumps({'earlier.pt':runner.sha(old)}))
            with patch.object(runner,'ROOT',root):
                runner.verify_frozen(study)
                source.write_text('changed source')
                with self.assertRaisesRegex(ValueError,'Implementation differs'):runner.verify_frozen(study)
                source.write_text('registered source');old.write_bytes(b'changed evidence')
                with self.assertRaisesRegex(ValueError,'protected earlier'):runner.verify_frozen(study)

    def test_registered_snapshot_uses_fresh_combined_policy_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            target=runner.build_snapshot(root,17,root/'data.jsonl',root/'vectors.zip')
            config=notebook_configuration(target)
            self.assertEqual(config['RUNTIME']['seed'],17)
            self.assertFalse(config['RUN_OPTIONS']['evaluate_test'])
            for key,value in runner.POLICY.items():self.assertEqual(config['STAGE2_CONFIG']['training'][key],value)
            self.assertEqual(config['STAGE1_CONFIG']['training']['max_epochs'],20)
            before=target.read_bytes()
            with self.assertRaises(FileExistsError):
                runner.build_snapshot(root,17,root/'data.jsonl',root/'vectors.zip')
            self.assertEqual(target.read_bytes(),before)


if __name__=='__main__':unittest.main()

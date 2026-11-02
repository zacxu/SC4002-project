"""Meaningful joint-model/count/masking checks; no task-model training."""
import ast
from dataclasses import replace
import json
import math
from pathlib import Path
import unittest
import warnings

import torch
from torch import nn

from src.revised_transformer_model import (DynamicCharacterCNN, JointTransformer,
    RevisedTransformerConfig, count_parameters, enforce_parameter_cap)


TAGS = ['O', 'B-place', 'I-place', 'B-name', 'I-name']


def tiny_config(**changes):
    return replace(RevisedTransformerConfig(word_dim=6, d_model=16, n_heads=4,
        n_layers=2, d_ff=32, dropout=0, char_dim=4, char_channels=3,
        intent_context_dim=4, attention_pool_dim=5, head_hidden_dim=8), **changes)


def fixture():
    words = torch.tensor([[2, 3, 4], [5, 6, 0]])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    chars = torch.tensor([[[2, 3, 0, 0], [4, 0, 0, 0], [5, 6, 7, 0]],
                          [[3, 4, 0, 0], [6, 0, 0, 0], [0, 0, 0, 0]]])
    intents = torch.tensor([1, 2])
    slots = torch.tensor([[1, 2, 0], [3, 4, -100]])
    return words, mask, chars, intents, slots


class RevisedTransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_complete_default_model_counts_every_embedding_and_head(self):
        tags = ['O'] + [prefix + '-' + str(i) for i in range(55) for prefix in ('B', 'I')]
        model = JointTransformer(5300, 60, tags, 34)
        self.assertEqual(count_parameters(model), {'total': 19_748_595, 'trainable': 19_748_595})
        model.embedding.weight.requires_grad_(False)
        self.assertEqual(enforce_parameter_cap(model), {'total': 19_748_595, 'trainable': 18_158_595})
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            enforce_parameter_cap(model, 19_748_594)

    def test_baseline_count_matches_original_111_tag_model(self):
        tags = ['O'] + [prefix + '-' + str(i) for i in range(55) for prefix in ('B', 'I')]
        model = JointTransformer(5300, 60, tags, 34, RevisedTransformerConfig.original_baseline())
        self.assertEqual(count_parameters(model)['total'], 4_345_243)

    def test_baseline_matches_actual_supplied_model_definition(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] / 'NLP_Final_project.ipynb').read_text(encoding='utf-8'))
        code = next(''.join(cell['source']) for cell in notebook['cells']
                    if cell['cell_type'] == 'code' and 'class JointTransformerStage1' in ''.join(cell['source']))
        tree = ast.parse(code)
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)
                     and node.name in {'PositionalEncoding', 'JointTransformerStage1'}]
        scope = dict(nn=nn, torch=torch, math=math, MAX_LEN=64, PAD_ID=0, word_vocab=list(range(12)),
                     W2V_DIM=6, D_MODEL=16, N_HEADS=4, N_LAYERS=2, D_FF=32, DROPOUT=0,
                     intent_labels=list(range(3)), slot_labels=TAGS,
                     embedding_matrix=torch.zeros(12, 6).numpy())
        exec(compile(tree, '<original-notebook-model>', 'exec'), scope)
        torch.manual_seed(21)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            original = scope['JointTransformerStage1']().eval()
        torch.manual_seed(21)
        config = replace(RevisedTransformerConfig.original_baseline(), word_dim=6,
                         d_model=16, n_heads=4, n_layers=2, d_ff=32, dropout=0)
        revised = JointTransformer(12, 3, TAGS, 9, config, torch.zeros(12, 6)).eval()
        expected_parameters = dict(original.named_parameters())
        self.assertEqual(set(expected_parameters), set(dict(revised.named_parameters())))
        for name, parameter in revised.named_parameters():
            self.assertTrue(torch.equal(parameter, expected_parameters[name]), name)
        words, mask, *_ = fixture()
        with torch.no_grad():
            intent, slot = original(words, mask)
            actual = revised(words, mask)
        torch.testing.assert_close(actual['intent_logits'], intent)
        torch.testing.assert_close(actual['slot_logits'], slot)

    def test_configuration_round_trip_and_invalid_choices(self):
        config = tiny_config()
        self.assertEqual(RevisedTransformerConfig.from_dict(json.loads(json.dumps(config.to_dict()))), config)
        for changes in ({'d_model': 17}, {'n_heads': 3}, {'dropout': float('nan')},
                        {'use_crf': 1}, {'char_kernels': (3, 3)}, {'n_layers': True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                tiny_config(**changes)

    def test_encoder_blocks_have_independent_initial_values_and_storage(self):
        model = JointTransformer(12, 3, TAGS, 9, tiny_config())
        first, second = [layer.self_attn.in_proj_weight for layer in model.encoder.layers]
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertFalse(torch.equal(first, second))
        baseline = JointTransformer(12, 3, TAGS, 9, tiny_config(independent_init=False))
        first, second = [layer.self_attn.in_proj_weight for layer in baseline.encoder.layers]
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertTrue(torch.equal(first, second))

    def test_character_padding_cannot_create_features_from_bias_only(self):
        model = DynamicCharacterCNN(9, 4, 3, (3, 4, 5))
        with torch.no_grad():
            for conv in model.convs:
                conv.bias.fill_(2)
        output = model(torch.zeros(1, 2, 1, dtype=torch.long), torch.tensor([[True, False]]))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_characters_beyond_the_original_24_character_limit_are_retained(self):
        model = DynamicCharacterCNN(3, 1, 1, (1,))
        with torch.no_grad():
            model.embedding.weight.zero_()
            model.embedding.weight[2].fill_(1)
            model.convs[0].weight.fill_(1)
            model.convs[0].bias.zero_()
        chars = torch.ones(1, 1, 41, dtype=torch.long)
        chars[0, 0, -1] = 2
        mask = torch.ones(1, 1, dtype=torch.bool)
        self.assertEqual(model(chars, mask).item(), 1)
        self.assertEqual(model(chars[:, :, :24], mask).item(), 0)

    def test_word_and_character_padding_do_not_change_real_predictions(self):
        model = JointTransformer(12, 3, TAGS, 9, tiny_config()).eval()
        words, mask, chars, _, _ = fixture()
        with torch.no_grad():
            reference = model(words, mask, chars)
            padded_words = F_pad(words, (0, 3), value=8)
            padded_mask = F_pad(mask, (0, 3), value=False)
            padded_chars = F_pad(chars, (0, 5, 0, 3), value=0)
            padded_chars[:, 3:] = 8
            padded_chars[1, 2] = 8
            actual = model(padded_words, padded_mask, padded_chars)
        torch.testing.assert_close(actual['intent_logits'], reference['intent_logits'], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(actual['slot_logits'][:, :3][mask], reference['slot_logits'][mask], atol=1e-6, rtol=1e-5)

    def test_real_forward_balances_branch_scale_and_masks_normalization_bias(self):
        model = JointTransformer(12, 3, TAGS, 9, tiny_config()).eval()
        words, mask, chars, _, _ = fixture()
        # Reproduce the measured scale mismatch without any task examples:
        # small pretrained-like word vectors, larger positive char features.
        with torch.no_grad():
            model.embedding.weight.mul_(.05)
            for conv in model.char_encoder.convs:
                conv.bias.fill_(.5)
        captured = []
        handle = model.proj.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach()))
        try:
            with torch.no_grad():
                model(words, mask, chars)
            features = captured[-1]
            word, character = features[..., :6], features[..., 6:]
            for branch in (word, character):
                torch.testing.assert_close(branch[mask].mean(-1), torch.zeros(5), atol=1e-5, rtol=0)
                torch.testing.assert_close(branch[mask].square().mean(-1), torch.ones(5), atol=.05, rtol=0)
            with torch.no_grad():
                model.word_norm.bias.fill_(2)
                model.char_norm.bias.fill_(3)
                model(words, mask, chars)
            self.assertTrue(torch.equal(captured[-1][~mask], torch.zeros_like(captured[-1][~mask])))
        finally:
            handle.remove()

    def test_finite_joint_loss_and_both_tasks_reach_shared_encoder(self):
        model = JointTransformer(12, 3, TAGS, 9, tiny_config())
        words, mask, chars, intents, slots = fixture()
        outputs = model(words, mask, chars)
        losses = model.loss(outputs, intents, slots, mask)
        self.assertEqual((losses['num_examples'], losses['num_tokens']), (2, 5))
        encoder = model.encoder.layers[0].self_attn.in_proj_weight
        for objective in ('intent_sum', 'slot_sum'):
            gradient, = torch.autograd.grad(losses[objective], encoder, retain_graph=True)
            self.assertGreater(gradient.abs().sum().item(), 0)
        # Slot likelihood reaches predicted intent logits without gold intent as input.
        gradient, = torch.autograd.grad(losses['slot_sum'], model.intent_head[-1].weight, retain_graph=True)
        self.assertGreater(gradient.abs().sum().item(), 0)
        losses['total'].backward()
        for parameter in (model.embedding.weight, model.char_encoder.embedding.weight,
                          model.intent_embeddings.weight, model.slot_head[-1].weight,
                          model.crf.transitions):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertTrue(torch.equal(model.embedding.weight.grad[0], torch.zeros(6)))

    def test_crf_decode_is_legal_and_padding_targets_do_not_affect_loss(self):
        model = JointTransformer(12, 3, TAGS, 9, tiny_config()).eval()
        words, mask, chars, intents, slots = fixture()
        outputs = model(words, mask, chars)
        decoded = model.decode(outputs, mask)
        for row, length in enumerate(mask.sum(1).tolist()):
            path = [TAGS[i] for i in decoded[row, :length].tolist()]
            self.assertFalse(path[0].startswith('I-'))
            for previous, current in zip(path, path[1:]):
                if current.startswith('I-'):
                    self.assertIn(previous, ('B-' + current[2:], current))
        self.assertEqual(decoded[1, 2].item(), 0)
        modified = slots.clone()
        modified[~mask] = 999
        torch.testing.assert_close(model.loss(outputs, intents, slots, mask)['total'],
                                   model.loss(outputs, intents, modified, mask)['total'])

    def test_unstructured_loss_and_no_character_option_work(self):
        model = JointTransformer(12, 3, TAGS, 9, tiny_config(use_char=False, use_crf=False,
            use_intent_context=False, use_attention_pool=False, use_input_norm=False)).eval()
        words, mask, _, intents, slots = fixture()
        outputs = model(words, mask)
        self.assertTrue(torch.isfinite(model.loss(outputs, intents, slots, mask)['total']))
        self.assertEqual(model.decode(outputs, mask)[1, 2].item(), 0)

    def test_invalid_masks_labels_and_embeddings_fail(self):
        model = JointTransformer(12, 3, TAGS, 9, tiny_config())
        words, mask, chars, intents, slots = fixture()
        for invalid in (torch.zeros_like(mask), torch.tensor([[True, False, True], [True, True, False]])):
            with self.assertRaises(ValueError):
                model(words, invalid, chars)
        with self.assertRaisesRegex(ValueError, 'Character'):
            model(words, mask)
        invalid = slots.clone()
        invalid[0, 0] = -100
        with self.assertRaisesRegex(ValueError, 'Real slot'):
            model.loss(model(words, mask, chars), intents, invalid, mask)
        with self.assertRaisesRegex(ValueError, 'embedding matrix'):
            JointTransformer(12, 3, TAGS, 9, tiny_config(), torch.zeros(11, 6))


def F_pad(values, padding, value):
    return torch.nn.functional.pad(values, padding, value=value)


if __name__ == '__main__':
    unittest.main()

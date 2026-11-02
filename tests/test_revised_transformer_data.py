"""Bookkeeping fixtures; no task augmentation, training or real downloads."""
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from src.revised_transformer_data import (build_data, collate_transformer,
    task_embeddings, ensure_word2vec_archive)


def population():
    return {'train': [
        {'id': 't1', 'partition': 'train', 'utt': 'red blue', 'intent': 'a',
         'annot_utt': '[color : red] [place : blue]'},
        {'id': 't2', 'partition': 'train', 'utt': "red's blue", 'intent': 'b',
         'annot_utt': "[color : red's] blue"}],
        'dev': [{'id': 'd1', 'partition': 'dev', 'utt': 'zzz blue', 'intent': 'a',
                 'annot_utt': '[color : zzz] blue'}],
        'test': [{'id': 's1', 'partition': 'test', 'utt': 'blue', 'intent': 'b',
                  'annot_utt': '[place : blue]'}]}


def archive_bytes(text, member='model.txt'):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, text)
    return output.getvalue()


class RevisedTransformerDataTests(unittest.TestCase):
    def test_maps_fit_train_only_and_oov_characters_remain_unknown(self):
        result = build_data(population())
        self.assertNotIn('zzz', result['mappings']['word_to_id'])
        self.assertNotIn('z', result['mappings']['char_to_id'])
        self.assertEqual(result['splits']['dev'][0]['word_ids'][0], 1)
        self.assertEqual(result['splits']['dev'][0]['char_ids'][0], [1, 1, 1])
        self.assertEqual(result['mappings']['tag_names'], ['O', 'B-color', 'I-color', 'B-place', 'I-place'])
        self.assertEqual(result['stats']['counts'], {'train': 2, 'dev': 1, 'test': 1})

    def test_annotations_change_targets_without_changing_predictive_features(self):
        original = population()
        changed = deepcopy(original)
        changed['dev'][0]['annot_utt'] = '[place : zzz] blue'
        first = build_data(original)['splits']['dev'][0]
        second = build_data(changed)['splits']['dev'][0]
        for key in ('word_ids', 'char_ids', 'utt', 'token_offsets'):
            self.assertEqual(first[key], second[key])
        self.assertNotEqual(first['slot_ids'], second['slot_ids'])

    def test_full_character_sequences_are_not_truncated(self):
        rows = population()
        word = 'z' * 67
        rows['dev'][0].update(utt=word, annot_utt='[color : ' + word + ']')
        data = build_data(rows)
        row = data['splits']['dev'][0]
        self.assertEqual(len(row['char_ids'][0]), 67)
        batch = collate_transformer([row, data['splits']['test'][0]])
        self.assertEqual(tuple(batch['char_ids'].shape), (2, 1, 67))
        self.assertTrue(torch.equal(batch['char_ids'][0, 0], torch.ones(67, dtype=torch.long)))
        self.assertEqual(batch['char_ids'][1, 0, 4:].sum().item(), 0)

    def test_collation_padding_and_labels_remain_distinct(self):
        data = build_data(population())
        batch = collate_transformer([data['splits']['train'][0], data['splits']['test'][0]])
        self.assertEqual(batch['mask'].tolist(), [[True, True], [True, False]])
        self.assertEqual(batch['word_ids'][1, 1].item(), 0)
        self.assertEqual(batch['slot_ids'][1, 1].item(), -100)
        self.assertEqual(batch['char_ids'][1, 1].sum().item(), 0)
        self.assertGreaterEqual(batch['char_ids'].shape[-1], 5)

    def test_unrepresentable_gold_boundary_is_rejected(self):
        rows = population()
        rows['dev'][0].update(utt="red's", annot_utt="[color : red]'s")
        with self.assertRaisesRegex(ValueError, 'boundary'):
            build_data(rows)

    def test_annotation_text_tampering_is_rejected(self):
        rows = population()
        rows['dev'][0]['annot_utt'] = '[color : other] blue'
        with self.assertRaisesRegex(ValueError, 'reconstruct'):
            build_data(rows)

    def test_heldout_labels_do_not_extend_training_inventory(self):
        for field, value in [('intent', 'unseen'), ('annot_utt', '[unseen : zzz] blue')]:
            rows = population()
            rows['dev'][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'training inventory'):
                build_data(rows)

    def test_duplicate_ids_and_moved_partitions_are_rejected(self):
        rows = population()
        rows['dev'][0]['id'] = 't1'
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            build_data(rows)
        rows = population()
        rows['dev'][0]['partition'] = 'train'
        with self.assertRaisesRegex(ValueError, 'partition'):
            build_data(rows)

    def test_inconsistent_collation_lengths_are_rejected(self):
        row = build_data(population())['splits']['train'][0]
        row['char_ids'].pop()
        with self.assertRaisesRegex(ValueError, 'lengths'):
            collate_transformer([row])


class TaskEmbeddingTests(unittest.TestCase):
    vocabulary = {'<PAD>': 0, '<UNK>': 1, 'queen': 2, 'missing': 3}

    def test_random_initialization_is_reproducible_and_padding_zero(self):
        first, manifest = task_embeddings(self.vocabulary, dimension=3)
        second, _ = task_embeddings(self.vocabulary, dimension=3)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.array_equal(first[0], np.zeros(3)))
        self.assertEqual(manifest['task_embedding_parameters'], 12)

    def test_casefold_source_averaging_retains_only_task_vectors(self):
        payload = archive_bytes('3 3\nQueen 1 2 3\nqueen 3 4 5\nunused 8 8 8\n')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'vectors.zip'
            path.write_bytes(payload)
            matrix, manifest = task_embeddings(self.vocabulary, source='word2vec', dimension=3, archive_path=path)
        np.testing.assert_array_equal(matrix[2], [2, 3, 4])
        self.assertEqual(manifest['pretrained_initialized'], 1)
        self.assertEqual(manifest['random_initialized'], 1)
        self.assertEqual(manifest['pretrained_coverage'], .5)
        self.assertFalse(manifest['full_source_table_retained_for_prediction'])
        self.assertEqual(manifest['source_vector_count'], 3)

    def test_original_pos_suffix_heuristic_is_supported(self):
        payload = archive_bytes('2 3\nQueen_NOUN 1 2 3\nqueen_PROPN 3 4 5\n')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'vectors.zip'
            path.write_bytes(payload)
            matrix, manifest = task_embeddings(self.vocabulary, source='word2vec', dimension=3, archive_path=path)
        np.testing.assert_array_equal(matrix[2], [2, 3, 4])
        self.assertTrue(manifest['pos_suffixes_stripped'])

    def test_missing_pretrained_assets_do_not_silently_fall_back(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                task_embeddings(self.vocabulary, source='word2vec', dimension=3,
                                archive_path=Path(temporary) / 'missing.zip')

    def test_wrong_header_dimension_and_truncation_are_rejected(self):
        for text in ['1 2\nqueen 1 2\n', '2 3\nqueen 1 2 3\n']:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'vectors.zip'
                path.write_bytes(archive_bytes(text))
                with self.assertRaises(ValueError):
                    task_embeddings(self.vocabulary, source='word2vec', dimension=3, archive_path=path)

    def test_nonfinite_needed_vector_and_unsafe_archive_are_rejected(self):
        for payload in [archive_bytes('1 3\nqueen nan 2 3\n'),
                        archive_bytes('1 3\nqueen 1 2 3\n', member='../model.txt')]:
            with self.subTest(payload_length=len(payload)), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'vectors.zip'
                path.write_bytes(payload)
                with self.assertRaises(ValueError):
                    task_embeddings(self.vocabulary, source='word2vec', dimension=3, archive_path=path)

    def test_invalid_vocabulary_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'vocabulary'):
            task_embeddings({'<PAD>': 0, '<UNK>': 1, 'queen': 5}, dimension=3)

    def test_download_is_exclusive_and_reuses_existing_cache(self):
        payload = archive_bytes('1 3\nqueen 1 2 3\n')
        class Response(io.BytesIO):
            headers = {'Content-Length': str(len(payload))}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'vectors.zip'
            with patch('src.revised_transformer_data.urlopen', return_value=Response(payload)) as request:
                self.assertEqual(ensure_word2vec_archive(path), path)
                self.assertEqual(ensure_word2vec_archive(path), path)
                self.assertEqual(request.call_count, 1)
            self.assertEqual(path.read_bytes(), payload)
            self.assertFalse(list(path.parent.glob('*.part')))

    def test_incomplete_download_does_not_publish_archive(self):
        payload = archive_bytes('1 3\nqueen 1 2 3\n')
        class Response(io.BytesIO):
            headers = {'Content-Length': str(len(payload) + 1)}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'vectors.zip'
            with patch('src.revised_transformer_data.urlopen', return_value=Response(payload)):
                with self.assertRaisesRegex(ValueError, 'Content-Length'):
                    ensure_word2vec_archive(path)
            self.assertFalse(path.exists())
            self.assertFalse(list(path.parent.glob('*.part')))


if __name__ == '__main__':
    unittest.main()

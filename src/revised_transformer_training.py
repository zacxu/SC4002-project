"""Fresh, auditable training for the two joint Transformer systems.

These ordinary definitions are also inlined in the revised notebook. A stage
completes its registered budget or patience policy, selects on development only,
reloads in a separate interpreter, and optionally freezes before one test pass.
"""
from __future__ import annotations

import ast
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.comparison_contract import (DATASET_SHA256, EVALUATOR_VERSION, EXPECTED_COUNTS,
    bio_to_character_spans, load_verified_massive, score_exact_spans)
from src.revised_transformer_data import build_data, collate_transformer
from src.revised_transformer_model import JointTransformer, RevisedTransformerConfig, enforce_parameter_cap


def rt_json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def rt_file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rt_write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                         + '\n', encoding='utf-8', newline='\n')
    temporary.replace(path)


def notebook_configuration(path):
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    cells = [cell for cell in document['cells'] if cell['cell_type'] == 'code']
    metadata = document['metadata']['revised_transformer']
    expected, registered = metadata['code_cell_sha256'], metadata['configuration_cell_indices']
    tagged = [i for i, cell in enumerate(cells) if 'configuration' in cell.get('metadata', {}).get('tags', [])]
    if registered != tagged or len(registered) != 1 or len(expected) != len(cells):
        raise ValueError('Expected one registered editable configuration cell')
    configuration = {}
    for index, cell in enumerate(cells):
        source = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
        if index not in registered:
            if hashlib.sha256(source.encode('utf-8')).hexdigest() != expected[index]:
                raise ValueError('Protected notebook implementation changed; rebuild and review it')
            continue
        for statement in ast.parse(source).body:
            if (not isinstance(statement, ast.Assign) or len(statement.targets) != 1
                    or not isinstance(statement.targets[0], ast.Name)):
                raise ValueError('Configuration permits only literal dictionary assignments')
            name = statement.targets[0].id
            if name not in {'RUNTIME', 'STAGE1_CONFIG', 'STAGE2_CONFIG', 'RUN_OPTIONS'} or name in configuration:
                raise ValueError('Unexpected or repeated configuration name')
            try:
                value = ast.literal_eval(statement.value)
            except (ValueError, TypeError) as error:
                raise ValueError('Use literal configuration values') from error
            if not isinstance(value, dict):
                raise ValueError('Every configuration must be a dictionary')
            configuration[name] = value
    if set(configuration) != {'RUNTIME', 'STAGE1_CONFIG', 'STAGE2_CONFIG', 'RUN_OPTIONS'}:
        raise ValueError('All four configuration dictionaries are required')
    return configuration


def notebook_identity(path):
    notebook_configuration(path)
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    sources = [cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
               for cell in document['cells'] if cell['cell_type'] == 'code']
    return rt_json_hash([hashlib.sha256(source.encode('utf-8')).hexdigest() for source in sources])


def capture_configuration(path, runtime, stage1_config, stage2_config, run_options, output_path):
    values = {'RUNTIME': runtime, 'STAGE1_CONFIG': stage1_config,
              'STAGE2_CONFIG': stage2_config, 'RUN_OPTIONS': run_options}
    if notebook_configuration(path) != values:
        raise ValueError('Save configuration edits before restarting and running all cells')
    if type(run_options.get('evaluate_test')) is not bool:
        raise ValueError('evaluate_test must be Boolean')
    snapshot = {**values, 'code_sha256': notebook_identity(path),
                'configuration_sha256': rt_json_hash(values)}
    rt_write_json(output_path, snapshot)
    return snapshot


def rt_seed(runtime):
    seed = runtime.get('seed', 42)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed must be a nonnegative 32-bit integer')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG', ':4096:8') != ':4096:8':
        raise ValueError('Start the kernel with CUBLAS_WORKSPACE_CONFIG=:4096:8')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_default_dtype(torch.float32)


def rt_sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(torch.device(device))


def rt_state_hash(model):
    return rt_state_dict_hash(model.state_dict())


def rt_state_dict_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(tensor.dtype).encode('ascii'))
        digest.update(str(tuple(tensor.shape)).encode('ascii'))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _rt_cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _rt_cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rt_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_rt_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def validate_training_config(config):
    required = {'max_epochs', 'batch_size', 'learning_rate', 'embedding_learning_rate',
                'weight_decay', 'gradient_clip_norm', 'warmup_fraction', 'min_lr_ratio'}
    optional = {'word_dropout_alpha', 'intent_label_smoothing', 'early_stopping_patience',
                'early_stopping_min_epochs', 'early_stopping_min_delta', 'schedule_epochs'}
    if not isinstance(config, dict) or not required <= set(config) or set(config) - required - optional:
        raise ValueError('Unexpected training configuration fields')
    for name in ('max_epochs', 'batch_size'):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(name + ' must be a positive integer')
    for name in required - {'max_epochs', 'batch_size'}:
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(name + ' must be a finite number')
    if any(config[name] <= 0 for name in ('learning_rate', 'embedding_learning_rate', 'gradient_clip_norm')):
        raise ValueError('Learning rates and gradient clipping must be positive')
    if config['weight_decay'] < 0 or not 0 <= config['warmup_fraction'] < 1 or not 0 <= config['min_lr_ratio'] <= 1:
        raise ValueError('Invalid weight decay or schedule fraction')
    for name in optional:
        value = config.get(name, 0.0)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(name + ' must be a finite number')
    if config.get('word_dropout_alpha', 0.0) < 0:
        raise ValueError('Word dropout alpha must be nonnegative')
    if not 0 <= config.get('intent_label_smoothing', 0.0) < 1:
        raise ValueError('Intent label smoothing must be in [0, 1)')
    for name in ('early_stopping_patience', 'early_stopping_min_epochs'):
        if type(config.get(name, 0)) is not int or config.get(name, 0) < 0:
            raise ValueError(name + ' must be a nonnegative integer')
    if config.get('early_stopping_min_epochs', 0) > config['max_epochs']:
        raise ValueError('Minimum training epochs cannot exceed the maximum budget')
    if not 0 <= config.get('early_stopping_min_delta', 0.0) <= 1:
        raise ValueError('Early-stopping minimum delta must be in [0, 1]')
    horizon = config.get('schedule_epochs', config['max_epochs'])
    if type(horizon) is not int or not 1 <= horizon <= config['max_epochs']:
        raise ValueError('Schedule epochs must be a positive integer within the maximum budget')


class FrequencyWordDropout:
    """Train-only word-to-UNK noise with its own CPU random stream.

    This object is training bookkeeping, not a model module or parameter.
    Character IDs, source examples, targets and token positions are untouched.
    Only eligible real ordinary-word positions consume random draws.
    """
    def __init__(self, word_counts, alpha, seed, *, training_record_ids, word_to_id):
        if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
                or not math.isfinite(alpha) or alpha < 0):
            raise ValueError('Word dropout alpha must be finite and nonnegative')
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('Word dropout seed must be a nonnegative 32-bit integer')
        if (not isinstance(word_counts, (list, tuple)) or len(word_counts) < 2
                or any(type(value) is not int or value < 0 for value in word_counts)
                or not training_record_ids or len(set(training_record_ids)) != len(training_record_ids)):
            raise ValueError('Expected complete nonnegative train counts and unique train IDs')
        if (word_to_id.get('<PAD>') != 0 or word_to_id.get('<UNK>') != 1
                or sorted(word_to_id.values()) != list(range(len(word_counts)))):
            raise ValueError('Word dropout requires train-fitted contiguous IDs with PAD=0, UNK=1')
        self.alpha, self.seed = float(alpha), seed
        self.counts = torch.tensor(word_counts, dtype=torch.long, device='cpu')
        self.probabilities = torch.zeros(len(word_counts), dtype=torch.float64, device='cpu')
        if self.alpha > 0:
            self.probabilities[2:] = self.alpha / (self.counts[2:].double() + self.alpha)
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.training_record_ids = tuple(training_record_ids)
        tokens = sum(word_counts)
        self.metadata = {'schema_version': 1, 'alpha': self.alpha, 'seed': seed,
            'fit_partition': 'train', 'random_stream': 'private_cpu_torch_generator',
            'probability_rule': 'alpha / (original_train_token_frequency + alpha)',
            'eligible_rule': 'real token mask and word ID >= 2; excludes PAD and UNK',
            'training_record_ids_sha256': rt_json_hash(list(training_record_ids)),
            'word_to_id_sha256': rt_json_hash(word_to_id), 'word_counts': list(word_counts),
            'word_counts_sha256': rt_json_hash(list(word_counts)), 'training_token_count': tokens,
            'eligible_training_token_count': sum(word_counts[2:]),
            'expected_mask_fraction': float((self.counts.double() * self.probabilities).sum()) / tokens if tokens else 0.0}

    @classmethod
    def from_bundle(cls, bundle, alpha, seed):
        mapping = bundle['mappings']['word_to_id']
        counts = [0] * len(mapping)
        rows = bundle['splits']['train']
        for row in rows:
            for index in row['word_ids']:
                if type(index) is not int or not 0 <= index < len(counts):
                    raise ValueError('Training word ID is outside the fitted vocabulary')
                counts[index] += 1
        return cls(counts, alpha, seed, training_record_ids=[row['record_id'] for row in rows],
                   word_to_id=mapping)

    def apply(self, word_ids, mask):
        if (word_ids.device.type != 'cpu' or mask.device.type != 'cpu'
                or word_ids.dtype != torch.long or mask.dtype != torch.bool
                or word_ids.ndim != 2 or mask.shape != word_ids.shape
                or bool(((word_ids < 0) | (word_ids >= len(self.counts))).any())):
            raise ValueError('Word dropout expects valid CPU word IDs and matching Boolean token mask')
        eligible = mask & word_ids.ge(2)
        number = int(eligible.sum())
        if not self.alpha or not number:
            return word_ids, {'masked_tokens': 0, 'eligible_tokens': number}
        draws = torch.rand(number, generator=self.generator, dtype=torch.float64, device='cpu')
        selected = draws < self.probabilities[word_ids[eligible]]
        result = word_ids.clone()
        replacements = result[eligible]
        replacements[selected] = 1
        result[eligible] = replacements
        return result, {'masked_tokens': int(selected.sum()), 'eligible_tokens': number}

    def rng_state_sha256(self):
        return hashlib.sha256(self.generator.get_state().numpy().tobytes()).hexdigest()


def learning_rate_factor(step, total_steps, warmup_fraction, minimum_ratio):
    if not 1 <= step <= total_steps:
        raise ValueError('Schedule step must be inside the full training budget')
    warmup = min(total_steps, int(total_steps * warmup_fraction))
    if warmup and step <= warmup:
        return step / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return minimum_ratio + (1 - minimum_ratio) * .5 * (1 + math.cos(math.pi * progress))


def scheduled_learning_rate_factor(step, schedule_steps, warmup_fraction, minimum_ratio):
    """Keep the original schedule exactly, then hold its floor after its horizon."""
    if type(step) is not int or step <= 0 or type(schedule_steps) is not int or schedule_steps <= 0:
        raise ValueError('Learning-rate steps and schedule horizon must be positive integers')
    return learning_rate_factor(min(step, schedule_steps), schedule_steps, warmup_fraction, minimum_ratio)


class DevelopmentPatience:
    """Meaningful-improvement stopping; actual-best checkpoint selection stays separate."""
    def __init__(self, config):
        validate_training_config(config)
        patience = config.get('early_stopping_patience', 0)
        self.policy = {'enabled': patience > 0, 'max_epochs': config['max_epochs'],
            'min_epochs': config.get('early_stopping_min_epochs', 0), 'patience': patience,
            'min_delta': float(config.get('early_stopping_min_delta', 0.0)),
            'monitor': 'mean_dev_intent_accuracy_and_slot_micro_f1',
            'comparison': 'score > anchor and score >= anchor + min_delta',
            'counter_scope': 'all epochs; stopping allowed once min_epochs reached',
            'schedule_epochs': config.get('schedule_epochs', config['max_epochs']),
            'schedule_after_horizon': 'hold_min_lr_ratio'}
        self.epoch, self.anchor, self.anchor_epoch, self.bad_epochs = 0, None, None, 0
        self.state = None

    def update(self, epoch, score):
        if (type(epoch) is not int or epoch != self.epoch + 1 or epoch > self.policy['max_epochs']
                or (self.state is not None and self.state['termination_reason'] is not None)):
            raise ValueError('Patience policy requires consecutive epochs and cannot continue after termination')
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError('Development selection score must be finite and in [0, 1]')
        meaningful = self.anchor is None or (score > self.anchor and score >= self.anchor + self.policy['min_delta'])
        if meaningful:
            self.anchor, self.anchor_epoch, self.bad_epochs = float(score), epoch, 0
        else:
            self.bad_epochs += 1
        eligible = self.policy['enabled'] and epoch >= self.policy['min_epochs']
        triggered = eligible and self.bad_epochs >= self.policy['patience'] and epoch < self.policy['max_epochs']
        reason = 'max_epochs' if epoch == self.policy['max_epochs'] else 'patience' if triggered else None
        self.epoch = epoch
        self.state = {'early_stopping_anchor_score': self.anchor,
            'early_stopping_anchor_epoch': self.anchor_epoch, 'early_stopping_bad_epochs': self.bad_epochs,
            'early_stopping_meaningful_improvement': meaningful, 'early_stopping_eligible': eligible,
            'early_stopping_triggered': triggered, 'termination_reason': reason}
        return dict(self.state)

    def termination(self, optimizer_steps, batches_per_epoch):
        if (self.state is None or self.state['termination_reason'] is None
                or type(batches_per_epoch) is not int or batches_per_epoch <= 0
                or type(optimizer_steps) is not int or optimizer_steps != self.epoch * batches_per_epoch):
            raise ValueError('A completed registered policy and all actual epoch optimizer steps are required')
        return {'reason': self.state['termination_reason'], 'epochs_trained': self.epoch,
            'max_epochs': self.policy['max_epochs'], 'optimizer_steps': optimizer_steps,
            'maximum_optimizer_steps': self.policy['max_epochs'] * batches_per_epoch,
            'schedule_epochs': self.policy['schedule_epochs'],
            'schedule_steps': self.policy['schedule_epochs'] * batches_per_epoch,
            'last_meaningful_improvement_epoch': self.anchor_epoch,
            'meaningful_improvement_anchor_score': self.anchor, 'non_improving_epochs': self.bad_epochs,
            'stopped_early': self.epoch < self.policy['max_epochs']}


def _rt_loader(bundle, partition, batch_size, seed, shuffle=False):
    rows = bundle['splits'][partition]
    if len(rows) != EXPECTED_COUNTS[partition] or len({row['record_id'] for row in rows}) != len(rows):
        raise ValueError('Expected every original ' + partition + ' record exactly once')
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, drop_last=False,
                      num_workers=0, collate_fn=collate_transformer,
                      generator=torch.Generator().manual_seed(seed))


def _rt_model(bundle, config, device, embedding_matrix=None):
    mappings = bundle['mappings']
    model = JointTransformer(len(mappings['word_to_id']), len(mappings['intent_to_id']),
        mappings['tag_names'], len(mappings['char_to_id']),
        config=RevisedTransformerConfig.from_dict(config), embedding_matrix=embedding_matrix)
    enforce_parameter_cap(model, 20_000_000)
    return model.to(device)


def _rt_epoch(model, loader, bundle, device, *, optimizer=None, schedule=None,
              word_dropout=None, intent_label_smoothing=0.0):
    training = optimizer is not None
    if (isinstance(intent_label_smoothing, bool) or not isinstance(intent_label_smoothing, (int, float))
            or not math.isfinite(intent_label_smoothing) or not 0 <= intent_label_smoothing < 1):
        raise ValueError('Intent label smoothing must be finite and in [0, 1)')
    if training and word_dropout is not None:
        if set(word_dropout.training_record_ids) != {row['record_id'] for row in loader.dataset}:
            raise ValueError('Word dropout counts must be fitted on the complete original training records')
    model.train(training)
    intent_sum = slot_sum = 0.0
    examples = tokens = batches = 0
    observed, predictions = [], []
    maximum_gradient_norm = 0.0
    masked_tokens = eligible_tokens = 0
    id_to_intent = {value: key for key, value in bundle['mappings']['intent_to_id'].items()}
    tags = bundle['mappings']['tag_names']
    rt_sync(device)
    started = perf_counter()
    with torch.enable_grad() if training else torch.inference_mode():
        for original in loader:
            inputs = original
            if training:
                if word_dropout is not None:
                    noisy_words, noise = word_dropout.apply(original['word_ids'], original['mask'])
                    inputs = {**original, 'word_ids': noisy_words}
                    masked_tokens += noise['masked_tokens']
                    eligible_tokens += noise['eligible_tokens']
                else:
                    eligible_tokens += int((original['mask'] & original['word_ids'].ge(2)).sum())
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                     for key, value in inputs.items()}
            if training:
                optimizer.zero_grad(set_to_none=True)
                if schedule:
                    schedule()
            outputs = model(batch['word_ids'], batch['mask'], batch['char_ids'])
            if training and intent_label_smoothing > 0:
                losses = model.loss(outputs, batch['intent_ids'], batch['slot_ids'], batch['mask'],
                                    intent_label_smoothing=intent_label_smoothing)
            else:
                # Preserve the original call and numerical path for legacy models/configurations.
                losses = model.loss(outputs, batch['intent_ids'], batch['slot_ids'], batch['mask'])
            if not bool(torch.isfinite(losses['total'])):
                raise ValueError('Nonfinite joint loss')
            if training:
                losses['total'].backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), optimizer._rt_clip, error_if_nonfinite=True)
                maximum_gradient_norm = max(maximum_gradient_norm, float(norm))
                optimizer.step()
            else:
                intents = outputs['intent_logits'].argmax(-1).cpu().tolist()
                decoded = model.decode(outputs, batch['mask']).cpu().tolist()
                for index, row in enumerate(original['rows']):
                    length = len(row['word_ids'])
                    predictions.append({'record_id': row['record_id'], 'utt': row['utt'],
                        'intent': id_to_intent[intents[index]], 'slots': bio_to_character_spans(
                            row['utt'], row['token_offsets'], [tags[tag] for tag in decoded[index][:length]])})
            observed.extend(row['record_id'] for row in original['rows'])
            intent_sum += float(losses['intent_sum'].detach())
            slot_sum += float(losses['slot_sum'].detach())
            examples += losses['num_examples']
            tokens += losses['num_tokens']
            batches += 1
    expected = [row['record_id'] for row in loader.dataset]
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise ValueError('An epoch dropped or repeated original records')
    if not training and observed != expected:
        raise ValueError('Evaluation must retain original source order')
    rt_sync(device)
    result = {'loss': intent_sum / examples + slot_sum / tokens,
        'intent_loss': intent_sum / examples, 'slot_loss': slot_sum / tokens,
        'num_examples': examples, 'num_tokens': tokens, 'num_batches': batches,
        'seconds': perf_counter() - started, 'record_ids_sha256': rt_json_hash(sorted(observed) if training else observed),
        'record_order_sha256': rt_json_hash(observed)}
    if training:
        result['maximum_gradient_norm'] = maximum_gradient_norm
        result.update(word_dropout_masked_tokens=masked_tokens, word_dropout_eligible_tokens=eligible_tokens,
            word_dropout_alpha=word_dropout.alpha if word_dropout is not None else 0.0,
            intent_label_smoothing=float(intent_label_smoothing),
            word_dropout_rng_state_sha256=word_dropout.rng_state_sha256() if word_dropout is not None else None)
    if not training:
        result['predictions'] = predictions
        result['metrics'] = score_exact_spans([bundle['gold_by_id'][key] for key in observed], predictions)
    return result


def _rt_evidence(result):
    return {key: value for key, value in result.items() if key != 'predictions'}


def _rt_predictions(path, predictions):
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def _rt_checkpoint(path, payload):
    temporary = Path(path).with_name(Path(path).name + '.part')
    torch.save(payload, temporary)
    temporary.replace(path)
    Path(str(path) + '.sha256').write_text(rt_file_hash(path) + '\n', encoding='ascii', newline='\n')


def _rt_load_checkpoint(path, bundle, device, dataset_sha256, code_sha256):
    path = Path(path)
    if Path(str(path) + '.sha256').read_text().strip() != rt_file_hash(path):
        raise ValueError('Checkpoint bytes differ from its checksum')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload['evaluator_version'] != EVALUATOR_VERSION or payload['dataset_sha256'] != dataset_sha256
            or payload['code_sha256'] != code_sha256 or payload['mappings'] != bundle['mappings']):
        raise ValueError('Checkpoint scorer/data/code/train-only mappings differ')
    if rt_json_hash(payload['stage_config']) != payload['stage_config_sha256']:
        raise ValueError('Checkpoint effective configuration differs from its recorded identity')
    model = _rt_model(bundle, payload['stage_config']['model'], device)
    model.load_state_dict(payload['model_state'], strict=True)
    if enforce_parameter_cap(model) != payload['parameter_counts']:
        raise ValueError('Reconstructed whole-system parameter count changed')
    return model, payload


def verify_checkpoint(checkpoint, dataset_path, code_sha256, device, fresh_process=False):
    original_sha = rt_file_hash(checkpoint)
    raw, manifest = load_verified_massive(dataset_path)
    bundle = build_data(raw)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    rt_seed(payload['runtime'])
    model, payload = _rt_load_checkpoint(checkpoint, bundle, device, manifest['dataset_sha256'], code_sha256)
    loader = _rt_loader(bundle, 'dev', payload['stage_config']['training']['batch_size'], payload['runtime']['seed'])
    dev = _rt_epoch(model, loader, bundle, device)
    if dev['metrics'] != payload['dev_metrics'] or dev['predictions'] != payload['dev_predictions']:
        raise ValueError('Reloaded checkpoint does not reproduce every selected development prediction')
    if rt_file_hash(checkpoint) != original_sha:
        raise ValueError('Verification changed checkpoint bytes')
    report = {'passed': True, 'fresh_python_process': fresh_process,
        'training_pid': payload['training_pid'], 'verification_pid': os.getpid(),
        'ordered_predictions_equal': True, 'num_examples': len(loader.dataset),
        'metrics': dev['metrics'], 'test_evaluated': False, 'checkpoint_sha256': original_sha,
        'code_sha256': code_sha256, 'dataset_sha256': manifest['dataset_sha256'],
        'verified_at_utc': datetime.now(timezone.utc).isoformat()}
    if fresh_process and report['training_pid'] == report['verification_pid']:
        raise ValueError('Independent verification requires a distinct interpreter')
    rt_write_json(Path(checkpoint).parent / 'fresh_reload.json', report)
    return report


def _rt_fresh_verifier(notebook_path, checkpoint, dataset_path, code_sha256, device, timeout):
    if notebook_identity(notebook_path) != code_sha256:
        raise ValueError('Saved notebook changed since this run began')
    document = json.loads(Path(notebook_path).read_text(encoding='utf-8'))
    definitions = [cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
        for cell in document['cells'] if cell['cell_type'] == 'code'
        and 'definitions' in cell.get('metadata', {}).get('tags', [])]
    source = 'from __future__ import annotations\n\n' + '\n\n'.join(
        item.replace('from __future__ import annotations\n', '') for item in definitions)
    source += "\nif __name__ == '__main__':\n    verify_checkpoint(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], fresh_process=True)\n"
    worker = Path(checkpoint).parent / 'fresh_verifier.py'
    worker.write_text(source, encoding='utf-8', newline='\n')
    environment = os.environ.copy()
    environment.update(PYTHONHASHSEED='42', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONIOENCODING='utf-8')
    environment.pop('PYTHONPATH', None)
    completed = subprocess.run([sys.executable, '-B', str(worker.resolve()), str(Path(checkpoint).resolve()),
        str(Path(dataset_path).resolve()), code_sha256, str(device)], cwd=worker.parent,
        env=environment, capture_output=True, text=True, encoding='utf-8', timeout=timeout)
    (worker.parent / 'fresh_verifier_console.txt').write_text(completed.stdout + completed.stderr,
                                                            encoding='utf-8', newline='\n')
    if completed.returncode:
        raise RuntimeError('Independent reload failed: ' + completed.stderr[-3000:])
    return json.loads((worker.parent / 'fresh_reload.json').read_text(encoding='utf-8'))


def _rt_curves(history, output):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, key in zip(axes.flat[:3], ('loss', 'intent_loss', 'slot_loss')):
        for partition in ('train', 'dev'):
            axis.plot([row['epoch'] for row in history], [row[partition + '_' + key] for row in history], label=partition)
        axis.set(xlabel='Epoch', title=key.replace('_', ' ')); axis.legend()
    for key in ('intent_accuracy', 'slot_micro_f1'):
        axes[1, 1].plot([row['epoch'] for row in history], [100 * row['dev_' + key] for row in history], label=key)
    axes[1, 1].set(xlabel='Epoch', ylabel='Percent', title='Development metrics'); axes[1, 1].legend()
    figure.tight_layout(); figure.savefig(output, dpi=140); plt.close(figure)


def run_stage(stage, stage_config, runtime, run_options, bundle, embedding_matrix, device,
              output_dir, dataset_path, code_sha256, notebook_path):
    """Run one complete, exclusive train/select/reload/(optional)test workflow."""
    stage_config, runtime, run_options = copy.deepcopy((stage_config, runtime, run_options))
    validate_training_config(stage_config['training'])
    if type(run_options.get('evaluate_test')) is not bool:
        raise ValueError('evaluate_test must be Boolean')
    if notebook_identity(notebook_path) != code_sha256 or rt_file_hash(dataset_path) != DATASET_SHA256:
        raise ValueError('Reviewed notebook or verified original data changed')
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    rt_seed(runtime)
    device = torch.device(device)
    model = _rt_model(bundle, stage_config['model'], device, embedding_matrix)
    counts = enforce_parameter_cap(model)
    initial_model_state_sha256 = rt_state_hash(model)
    training = stage_config['training']
    stopping = DevelopmentPatience(training)
    alpha = float(training.get('word_dropout_alpha', 0.0))
    smoothing = float(training.get('intent_label_smoothing', 0.0))
    word_dropout = FrequencyWordDropout.from_bundle(bundle, alpha, runtime['seed']) if alpha else None
    regularization = {'word_dropout_alpha': alpha, 'intent_label_smoothing': smoothing,
        'word_dropout': copy.deepcopy(word_dropout.metadata) if word_dropout is not None else None,
        'intent_smoothing_scope': 'training_only; dev/test unsmoothed'}
    loaders = {part: _rt_loader(bundle, part, training['batch_size'], runtime['seed'], part == 'train') for part in ('train', 'dev')}
    word_parameters = list(model.embedding.parameters())
    word_ids = {id(parameter) for parameter in word_parameters}
    others = [parameter for parameter in model.parameters() if id(parameter) not in word_ids]
    optimizer = torch.optim.AdamW([{'params': word_parameters, 'lr': training['embedding_learning_rate']},
        {'params': others, 'lr': training['learning_rate']}], weight_decay=training['weight_decay'])
    optimizer._rt_clip = training['gradient_clip_norm']
    bases = [group['lr'] for group in optimizer.param_groups]
    schedule_steps, step = len(loaders['train']) * stopping.policy['schedule_epochs'], 0

    def schedule():
        nonlocal step
        step += 1
        factor = scheduled_learning_rate_factor(step, schedule_steps, training['warmup_fraction'], training['min_lr_ratio'])
        for group, base in zip(optimizer.param_groups, bases):
            group['lr'] = base * factor

    rt_write_json(output / 'run.json', {'stage': stage, 'stage_config': stage_config,
        'runtime': runtime, 'run_options': run_options, 'code_sha256': code_sha256,
        'dataset_sha256': DATASET_SHA256, 'evaluator_version': EVALUATOR_VERSION,
        'parameter_counts': counts, 'training_pid': os.getpid(), 'regularization': regularization,
        'stopping_policy': stopping.policy,
        'initial_model_state_sha256': initial_model_state_sha256,
        'created_at_utc': datetime.now(timezone.utc).isoformat()})
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    history, best_score, best_epoch = [], None, None
    checkpoint = output / 'best.pt'
    rt_sync(device); start = perf_counter()
    for epoch in range(1, training['max_epochs'] + 1):
        train = _rt_epoch(model, loaders['train'], bundle, device, optimizer=optimizer, schedule=schedule,
                          word_dropout=word_dropout, intent_label_smoothing=smoothing)
        dev = _rt_epoch(model, loaders['dev'], bundle, device)
        score = .5 * (dev['metrics']['intent_accuracy'] + dev['metrics']['slot_micro_f1'])
        selected = best_score is None or score > best_score
        stopping_state = stopping.update(epoch, score)
        row = {'epoch': epoch, **{'train_' + key: value for key, value in train.items()},
            **{'dev_' + key: value for key, value in dev.items() if key not in ('predictions', 'metrics')},
            'dev_intent_accuracy': dev['metrics']['intent_accuracy'], 'dev_slot_micro_f1': dev['metrics']['slot_micro_f1'],
            'selection_score': score, 'selected': selected, 'optimizer_step': step,
            'learning_rates': [group['lr'] for group in optimizer.param_groups], **stopping_state}
        history.append(row)
        with (output / 'history.jsonl').open('a', encoding='utf-8', newline='\n') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        with (output / 'history.csv').open('a', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if epoch == 1: writer.writeheader()
            writer.writerow(row)
        if selected:
            payload = {'schema_version': 1, 'stage': stage, 'purpose': 'fresh_joint_transformer',
                'stage_config': copy.deepcopy(stage_config), 'stage_config_sha256': rt_json_hash(stage_config),
                'runtime': copy.deepcopy(runtime), 'mappings': copy.deepcopy(bundle['mappings']),
                'model_state': _rt_cpu_copy(model.state_dict()), 'optimizer_state': _rt_cpu_copy(optimizer.state_dict()),
                'regularization': copy.deepcopy(regularization), 'initial_model_state_sha256': initial_model_state_sha256,
                'stopping_policy': copy.deepcopy(stopping.policy), 'early_stopping_state': dict(stopping_state),
                'word_dropout_rng_state': word_dropout.generator.get_state().clone() if word_dropout is not None else None,
                'parameter_counts': counts, 'epoch': epoch, 'selection_score': score,
                'dev_metrics': dev['metrics'], 'dev_predictions': dev['predictions'],
                'code_sha256': code_sha256, 'dataset_sha256': DATASET_SHA256,
                'evaluator_version': EVALUATOR_VERSION, 'training_pid': os.getpid()}
            _rt_checkpoint(checkpoint, payload)
            best_score, best_epoch = score, epoch
        if epoch == stopping.policy['schedule_epochs'] and epoch < training['max_epochs']:
            rt_write_json(output / 'schedule_horizon_evidence.json', {'schema_version': 1,
                'stage': stage, 'epoch': epoch, 'schedule_epochs': stopping.policy['schedule_epochs'],
                'maximum_epochs': training['max_epochs'], 'code_sha256': code_sha256,
                'dataset_sha256': DATASET_SHA256, 'evaluator_version': EVALUATOR_VERSION,
                'initial_model_state_sha256': initial_model_state_sha256, 'optimizer_step': step,
                'current_model_state_sha256': rt_state_hash(model), 'current_dev_metrics': dev['metrics'],
                'current_dev_predictions_sha256': rt_json_hash(dev['predictions']),
                'prefix_best_epoch': best_epoch, 'prefix_best_selection_score': best_score,
                'prefix_best_model_state_sha256': rt_state_dict_hash(payload['model_state']),
                'prefix_best_dev_metrics': payload['dev_metrics'],
                'prefix_best_dev_predictions_sha256': rt_json_hash(payload['dev_predictions']),
                'prefix_best_checkpoint_sha256': rt_file_hash(checkpoint),
                'prefix_history_sha256': rt_file_hash(output / 'history.jsonl'),
                'training_record_order_sha256': train['record_order_sha256']})
        print(f'{stage} epoch {epoch:02d}/{training["max_epochs"]}: train={train["loss"]:.5f}; dev={dev["loss"]:.5f}; '
              f'intent={100 * dev["metrics"]["intent_accuracy"]:.2f}; slot={100 * dev["metrics"]["slot_micro_f1"]:.2f}; '
              f'mean={100 * score:.2f}; selected={selected}', flush=True)
        if stopping_state['termination_reason'] is not None:
            break
    rt_sync(device); seconds = perf_counter() - start
    termination = stopping.termination(step, len(loaders['train']))
    if len(history) != termination['epochs_trained']:
        raise ValueError('The registered training history is incomplete')
    winner = max(history, key=lambda row: row['selection_score'])
    if winner['epoch'] != best_epoch or winner['selection_score'] != best_score:
        raise ValueError('Checkpoint selection differs from the earliest development maximum')
    training_summary = {'status': 'trained_not_tested', 'epochs_trained': len(history), 'selected_epoch': best_epoch,
        'selection_score': best_score, 'training_workflow_seconds': seconds, 'parameter_counts': counts,
        'regularization': regularization, 'initial_model_state_sha256': initial_model_state_sha256,
        'stopping_policy': stopping.policy, 'termination': termination,
        'timing_scope': 'Training, epoch development scoring, checkpoints and history; excludes setup, curves, fresh reload and final evaluation',
        'peak_allocated_cuda_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}
    rt_write_json(output / 'training_summary.json', training_summary)
    _rt_curves(history, output / 'learning_curves.png')
    del optimizer, model
    if device.type == 'cuda': torch.cuda.empty_cache()
    reload = _rt_fresh_verifier(notebook_path, checkpoint, dataset_path, code_sha256, device,
                               run_options.get('verification_timeout_seconds', 600))
    model, payload = _rt_load_checkpoint(checkpoint, bundle, device, DATASET_SHA256, code_sha256)
    dev = _rt_epoch(model, loaders['dev'], bundle, device)
    if dev['predictions'] != payload['dev_predictions'] or dev['metrics'] != payload['dev_metrics']:
        raise ValueError('Parent reload differs from independently verified development checkpoint')
    _rt_predictions(output / 'dev_predictions.jsonl', dev['predictions'])
    rt_write_json(output / 'dev_evaluation.json', _rt_evidence(dev))
    before, checkpoint_before = rt_state_hash(model), rt_file_hash(checkpoint)
    test = None
    if run_options['evaluate_test']:
        freeze = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'selected_epoch': best_epoch,
            'checkpoint_sha256': checkpoint_before, 'model_state_sha256': before,
            'history_sha256': rt_file_hash(output / 'history.jsonl'), 'test_model_evaluations': 0,
            'code_sha256': code_sha256, 'dataset_sha256': DATASET_SHA256, 'tuning_closed': True}
        rt_write_json(output / 'freeze_manifest.json', freeze)
        rt_write_json(output / 'test_started.json', {'started_at_utc': datetime.now(timezone.utc).isoformat(),
            'freeze_sha256': rt_file_hash(output / 'freeze_manifest.json'), 'test_model_evaluations': 1})
        test = _rt_epoch(model, _rt_loader(bundle, 'test', training['batch_size'], runtime['seed']), bundle, device)
        _rt_predictions(output / 'test_predictions.jsonl', test['predictions'])
        rt_write_json(output / 'test_evaluation.json', _rt_evidence(test))
    if before != rt_state_hash(model) or checkpoint_before != rt_file_hash(checkpoint):
        raise ValueError('Reload/final evaluation changed learned weights or checkpoint bytes')
    summary = {'status': 'complete', 'stage': stage, 'mode': 'final' if test is not None else 'development_only',
        **training_summary, 'status': 'complete', 'dev': _rt_evidence(dev), 'test': _rt_evidence(test) if test else None,
        'checkpoint': checkpoint.name, 'checkpoint_sha256': checkpoint_before, 'model_state_sha256': before,
        'fresh_process_reload': reload, 'test_model_evaluations': int(test is not None),
        'model_weights_unchanged': True, 'checkpoint_bytes_unchanged': True,
        'code_sha256': code_sha256, 'dataset_sha256': DATASET_SHA256, 'evaluator_version': EVALUATOR_VERSION,
        'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    rt_write_json(output / 'final_summary.json', summary)
    del model
    if device.type == 'cuda': torch.cuda.empty_cache()
    return summary


def report_results(summaries, output_root, bundle):
    output = Path(output_root)
    rows, full_precision = [], {}
    for stage, summary in summaries.items():
        full_precision[stage] = {}
        for split in ('dev', 'test'):
            if summary.get(split) is None: continue
            metrics = summary[split]['metrics']; full_precision[stage][split] = metrics
            rows.append({'model_version': stage, 'split': split,
                'total_parameters': summary['parameter_counts']['total'],
                'trainable_parameters': summary['parameter_counts']['trainable'],
                'intent_accuracy_percent': f'{100 * metrics["intent_accuracy"]:.2f}',
                'slot_micro_f1_percent': f'{100 * metrics["slot_micro_f1"]:.2f}',
                'epochs_trained': summary['epochs_trained'], 'selected_epoch': summary['selected_epoch']})
    report = {'evaluator_version': EVALUATOR_VERSION, 'dataset_sha256': DATASET_SHA256,
              'rows': rows, 'full_precision': full_precision}
    if len(summaries) == 2:
        first, last = list(summaries)
        report['stage2_minus_stage1_percentage_points'] = {
            split: {key: 100 * (full_precision[last][split][key] - full_precision[first][split][key])
                    for key in ('intent_accuracy', 'slot_micro_f1')}
            for split in full_precision[first] if split in full_precision[last]}
        report['development_winner'] = max(summaries, key=lambda stage:
            .5 * (full_precision[stage]['dev']['intent_accuracy'] + full_precision[stage]['dev']['slot_micro_f1']))
    rt_write_json(output / 'results.json', report)
    with (output / 'results.csv').open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    stage = list(summaries)[-1]
    predicted = [json.loads(line) for line in (output / stage / 'dev_predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    cases = []
    for prediction in predicted:
        gold = bundle['gold_by_id'][prediction['record_id']]
        score = score_exact_spans([gold], [prediction])
        if score['correct_intents'] != 1 or score['slot_fp'] or score['slot_fn']:
            cases.append({'record_id': prediction['record_id'], 'utt': gold['utt'],
                'gold': gold, 'prediction': prediction,
                'explanation': 'Inference: inspect training support, unfamiliar spelling and boundary/type/context ambiguity; one example does not prove a cause.'})
        if len(cases) == 4: break
    rt_write_json(output / 'error_cases.json', {'source': 'Actual revised development predictions',
        'cases': cases, 'note': 'All available real cases are retained if fewer than two; no errors are invented.'})
    return report

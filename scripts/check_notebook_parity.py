"""Compare visible notebook definitions with the frozen research implementation.

This development-only conversion check executes tagged definition cells, never
the notebook's training or final-test action cells. It uses original training
records for tensor checks and the two historical selected checkpoints for full
development prediction checks. The only output is a new Phase 7 audit JSON.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import types
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_SHA256 = "c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e"
MAPPINGS_SHA256 = "6a35e931d9fbca5fd48dcf95ac6598f98036fa584558949394fed7c30ef98e6b"
CHARACTERS_SHA256 = "a69bea15839987ecae4529c8914917686f989acf738890d023f0940d83e9c58e"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def plain(value: Any) -> Any:
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(message)
    checks.append(message)


def source_definitions(source: str) -> dict[str, ast.AST]:
    return {node.name: node for node in ast.parse(source).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def literal_configuration(path: Path) -> dict[str, Any]:
    """Read selected inline settings without executing a notebook action cell."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    names = {"RUNTIME", "STAGE1_CONFIG", "STAGE2_CONFIG"}
    settings = {}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code" or "configuration" not in cell.get("metadata", {}).get("tags", []):
            continue
        source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for node in ast.parse(source).body:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
            for target in targets:
                if isinstance(target, ast.Name) and target.id in names:
                    if target.id in settings:
                        raise ValueError("Duplicate inline configuration assignment: " + target.id)
                    settings[target.id] = ast.literal_eval(node.value)
    if set(settings) != names:
        raise ValueError("Notebook must expose literal selected settings in configuration cells: " + repr(sorted(names - settings.keys())))
    return settings


def notebook_definitions(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, ast.AST]]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    blocks, definitions = [], {}
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code" or "definitions" not in cell.get("metadata", {}).get("tags", []):
            continue
        source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        tree = ast.parse(source, filename=f"{path.name}:cell-{index}")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and (
                    node.module == "src" or node.module.startswith("src.")):
                raise ValueError(f"Tagged notebook definition imports repository source: cell {index}")
            if isinstance(node, ast.Import) and any(
                    item.name == "src" or item.name.startswith("src.") for item in node.names):
                raise ValueError(f"Tagged notebook definition imports repository source: cell {index}")
        definitions.update(source_definitions(source))
        blocks.append({"cell_index": index, "source": source,
                       "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest()})
    if not blocks:
        raise ValueError("Notebook has no code cells tagged 'definitions'")
    # dataclasses resolves annotation modules through sys.modules. A dictionary
    # passed to exec without a real module would hide conversion defects here.
    module = types.ModuleType("sc4002_phase7_notebook_parity")
    module.__file__ = str(path)
    module.__package__ = None
    sys.modules[module.__name__] = module
    for block in blocks:
        exec(compile(block["source"], f"{path.name}:cell-{block['cell_index']}", "exec"), module.__dict__)
    return module.__dict__, [{key: value for key, value in block.items() if key != "source"}
                             for block in blocks], definitions


def tensor_state_fingerprint(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def exact_tensors(left: dict, right: dict, label: str, checks: list[str]) -> None:
    import torch

    require(set(left) == set(right), f"{label}: identical tensor keys", checks)
    for key in left:
        require(torch.equal(left[key].detach().cpu(), right[key].detach().cpu()),
                f"{label}: exact {key}", checks)


def batch_parity(left: dict, right: dict, label: str, checks: list[str]) -> None:
    import torch

    require(set(left) == set(right), f"{label}: identical batch fields", checks)
    for key in left:
        if isinstance(left[key], torch.Tensor):
            require(left[key].dtype == right[key].dtype and torch.equal(left[key], right[key]),
                    f"{label}: exact {key}", checks)
        else:
            require(plain(left[key]) == plain(right[key]), f"{label}: exact {key}", checks)


def run(notebook_path: Path, device_name: str) -> dict:
    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(ROOT))
    from src.data import collate_examples, fit_mappings, load_massive, prepare_split
    from src.metrics import evaluate_predictions
    from src.models import BaselineConfig, JointBiLSTM, enforce_parameter_cap
    from src.runtime import configure_reproducibility, dataloader_generator, select_device
    from src.stage2_experiment import evaluate as source_stage2_evaluate, make_loader
    from src.stage2_model import JointStage2, Stage2Config, collate_stage2, fit_char_vocab
    from src.training import compute_losses, evaluate_model

    checks: list[str] = []
    notebook_document = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_sources = ["".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
                    for cell in notebook_document["cells"] if cell.get("cell_type") == "code"]
    code_hashes = [hashlib.sha256(source.encode("utf-8")).hexdigest() for source in code_sources]
    require(code_hashes == notebook_document["metadata"]["phase7"]["code_cell_sha256"],
            "Every visible code cell matches the notebook implementation manifest", checks)
    embedded_settings = literal_configuration(notebook_path)
    namespace, blocks, notebook_nodes = notebook_definitions(notebook_path)
    required = ("load_massive", "fit_mappings", "prepare_split", "collate_examples", "fit_char_vocab",
                "collate_stage2", "BaselineConfig", "JointBiLSTM", "Stage2Config", "JointStage2",
                "compute_losses", "evaluate_model", "evaluate_predictions", "s2_evaluate",
                "parameter_counts", "enforce_parameter_cap")
    require(all(name in namespace for name in required), "All required converted definitions exist", checks)
    # Exact class/function syntax checks supplement numerical tests so an
    # unexercised branch in the models cannot silently differ during conversion.
    ast_sources = {
        "src/schema.py": ("SlotSpan", "IntentSlotPrediction"),
        "src/data.py": ("Token", "UtteranceExample", "FittedMappings", "PreparedExample",
                        "tokenize_with_offsets", "fit_mappings", "prepare_example", "collate_examples",
                        "parse_annotations", "gold_bio_tags", "decode_bio"),
        "src/metrics.py": ("EvaluationResult", "evaluate_predictions"),
        "src/models.py": ("BaselineConfig", "JointBiLSTM", "parameter_counts", "enforce_parameter_cap"),
        "src/training.py": ("LossBundle", "compute_losses", "train_epoch", "evaluate_model"),
        "src/stage2_model.py": ("Stage2Config", "_validate_char_vocab", "fit_char_vocab", "collate_stage2", "CharacterCNN", "BIOCRF", "JointStage2"),
    }
    ast_evidence = {}
    for relative, names in ast_sources.items():
        original = source_definitions((ROOT / relative).read_text(encoding="utf-8"))
        for name in names:
            source_ast = ast.dump(original[name], include_attributes=False)
            notebook_ast = ast.dump(notebook_nodes[name], include_attributes=False)
            require(source_ast == notebook_ast, f"Frozen syntax parity: {relative}:{name}", checks)
            ast_evidence[name] = {"source": relative,
                                  "ast_sha256": hashlib.sha256(source_ast.encode()).hexdigest()}

    protocol = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8"))
    runtime = protocol["runtime"]
    require(embedded_settings["RUNTIME"] == runtime, "Literal notebook runtime matches the frozen protocol", checks)
    configure_reproducibility(runtime)
    device = select_device(device_name)
    dataset = ROOT / protocol["paths"]["dataset"]
    require(file_hash(dataset) == DATA_SHA256, "Original English dataset bytes match", checks)
    original = load_massive(dataset)
    notebook_original = namespace["load_massive"](dataset)
    for partition in ("train", "dev"):
        require(plain(original[partition]) == plain(notebook_original[partition]),
                f"Exact original {partition} records and annotation offsets", checks)
    mappings = fit_mappings(original["train"])
    notebook_mappings = namespace["fit_mappings"](notebook_original["train"])
    require(mappings.to_dict() == notebook_mappings.to_dict(), "Exact train-fitted mappings", checks)
    require(canonical_hash(mappings.to_dict()) == MAPPINGS_SHA256, "Expected canonical mapping identity", checks)
    prepared = {part: prepare_split(original[part], mappings) for part in ("train", "dev")}
    notebook_prepared = {part: namespace["prepare_split"](notebook_original[part], notebook_mappings)
                         for part in ("train", "dev")}
    for part in prepared:
        require(plain(prepared[part]) == plain(notebook_prepared[part]),
                f"Exact {part} tokens, offsets, input IDs, BIO targets and intents", checks)
    chars = fit_char_vocab(prepared["train"])
    notebook_chars = namespace["fit_char_vocab"](notebook_prepared["train"])
    require(chars == notebook_chars and len(chars) == 34, "Exact train-only 34-entry character vocabulary", checks)
    require(canonical_hash(chars) == CHARACTERS_SHA256, "Expected character mapping identity", checks)
    batches = {
        "stage1": (collate_examples(prepared["train"][:32]),
                   namespace["collate_examples"](notebook_prepared["train"][:32])),
        "stage2": (collate_stage2(prepared["train"][:32], chars),
                   namespace["collate_stage2"](notebook_prepared["train"][:32], notebook_chars)),
    }
    for stage, pair in batches.items():
        batch_parity(*pair, stage + " original 32-record batch", checks)

    baseline_config = json.loads((ROOT / "configs/phase4/stage1_dropout03.json").read_text(encoding="utf-8"))
    enhanced_config = json.loads((ROOT / "configs/phase5/char_crf_intent_context.json").read_text(encoding="utf-8"))
    require(embedded_settings["STAGE1_CONFIG"] == baseline_config,
            "Literal Stage 1 settings match the selected dropout-0.3 configuration", checks)
    require(embedded_settings["STAGE2_CONFIG"] == enhanced_config,
            "Literal Stage 2 settings match the frozen selected configuration", checks)
    tags = [tag for tag, _ in sorted(mappings.tag_to_id.items(), key=lambda item: item[1])]
    stages = (
        ("stage1", baseline_config, 4_646_571, ROOT / "artifacts/phase4/selection.json"),
        ("stage2", enhanced_config, 4_939_498, ROOT / "artifacts/phase5/selection.json"),
    )
    outcomes = {}
    for stage, config, expected_count, selection_path in stages:
        def new_model(converted: bool):
            if stage == "stage1":
                model_type = namespace["JointBiLSTM"] if converted else JointBiLSTM
                config_type = namespace["BaselineConfig"] if converted else BaselineConfig
                return model_type(len(mappings.token_to_id), len(mappings.intent_to_id), len(mappings.tag_to_id),
                                  config_type.from_dict(config["model"])).to(device)
            model_type = namespace["JointStage2"] if converted else JointStage2
            config_type = namespace["Stage2Config"] if converted else Stage2Config
            return model_type(len(mappings.token_to_id), len(mappings.intent_to_id), tags, len(chars),
                              config_type.from_dict(config["model"])).to(device)

        configure_reproducibility(runtime)
        source_model = new_model(False)
        configure_reproducibility(runtime)
        notebook_model = new_model(True)
        exact_tensors(source_model.state_dict(), notebook_model.state_dict(), stage + " seed-42 initialization", checks)
        init_fingerprint = tensor_state_fingerprint(source_model)
        expected_counts = {"total": expected_count, "trainable": expected_count}
        require(enforce_parameter_cap(source_model) == namespace["enforce_parameter_cap"](notebook_model)
                == expected_counts, f"{stage}: complete learned system parameter count", checks)
        device_batches = [
            {key: value.to(device) if isinstance(value, torch.Tensor) and key != "lengths" else value
             for key, value in batch.items()} for batch in batches[stage]
        ]
        batch_results = {}
        for training_mode in (False, True):
            results, losses = [], []
            for converted, model, batch in ((False, source_model, device_batches[0]),
                                            (True, notebook_model, device_batches[1])):
                configure_reproducibility(runtime)
                model.train(training_mode)
                with torch.inference_mode():
                    args = (batch["input_ids"], batch["lengths"], batch["attention_mask"])
                    outputs = model(*args, batch["char_ids"]) if stage == "stage2" else model(*args)
                    loss = (model.loss(outputs, batch) if stage == "stage2" else
                            (namespace["compute_losses"] if converted else compute_losses)(outputs, batch))
                results.append(outputs)
                losses.append(loss)
            mode = "training_dropout" if training_mode else "evaluation"
            exact_tensors(results[0], results[1], stage + " " + mode + " logits", checks)
            for name in ("total", "intent", "slot", "intent_sum", "slot_sum"):
                require(torch.equal(getattr(losses[0], name), getattr(losses[1], name)),
                        f"{stage} {mode}: exact {name} loss", checks)
            require(losses[0].num_examples == losses[1].num_examples == 32 and
                    losses[0].num_tokens == losses[1].num_tokens,
                    f"{stage} {mode}: exact loss denominators", checks)
            batch_results[mode] = {name: float(getattr(losses[0], name)) for name in ("total", "intent", "slot")}
        require(tensor_state_fingerprint(source_model) == tensor_state_fingerprint(notebook_model) == init_fingerprint,
                f"{stage}: initialization tensors unchanged by training-mode diagnostic forwards", checks)

        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        checkpoint = ROOT / selection["checkpoint"]
        require(file_hash(checkpoint) == Path(str(checkpoint) + ".sha256").read_text(encoding="ascii").strip()
                == selection["checkpoint_sha256"],
                f"{stage}: historical selected checkpoint bytes verified", checks)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        require(payload["model_config"] == config["model"] if stage == "stage1" else payload["stage_config"] == config,
                f"{stage}: checkpoint architecture matches selected settings", checks)
        require(payload["mappings"] == mappings.to_dict(), f"{stage}: saved training mappings agree", checks)
        for relative, digest in payload["source_hashes"].items():
            require(file_hash(ROOT / relative) == digest, f"{stage}: unchanged trained source {relative}", checks)
        source_model.load_state_dict(payload["model_state"], strict=True)
        notebook_model.load_state_dict(payload["model_state"], strict=True)
        pretrained_state = tensor_state_fingerprint(source_model)
        if stage == "stage1":
            loaders = [DataLoader(rows, batch_size=32, shuffle=False, num_workers=0, drop_last=False,
                                  generator=dataloader_generator(42), collate_fn=collate)
                       for rows, collate in ((prepared["dev"], collate_examples),
                                             (notebook_prepared["dev"], namespace["collate_examples"]))]
            source_result = evaluate_model(source_model, loaders[0], mappings, device)
            notebook_result = namespace["evaluate_model"](notebook_model, loaders[1], notebook_mappings, device)
        else:
            source_loader = make_loader(prepared["dev"], "dev", 32, 42, chars)
            # Instantiate the converted batching function directly so loader
            # order is fixed and no workflow configuration actions are needed.
            from functools import partial
            notebook_loader = DataLoader(notebook_prepared["dev"], batch_size=32, shuffle=False, num_workers=0,
                                         drop_last=False, generator=dataloader_generator(42),
                                         collate_fn=partial(namespace["collate_stage2"], char_to_id=notebook_chars))
            source_result = source_stage2_evaluate(source_model, source_loader, mappings, device)
            notebook_result = namespace["s2_evaluate"](notebook_model, notebook_loader, notebook_mappings, device)
        source_predictions = [item.to_dict() for item in source_result["predictions"]]
        notebook_predictions = [item.to_dict() for item in notebook_result["predictions"]]
        require(source_predictions == notebook_predictions == payload["validation_predictions"],
                f"{stage}: all 2,033 ordered development predictions match notebook, scripts and saved checkpoint", checks)
        require(source_result["metrics"].to_dict() == notebook_result["metrics"].to_dict() == payload["validation"]["metrics"],
                f"{stage}: exact development scorer counts and full-precision metrics", checks)
        for key, expected in (("num_examples", 2033), ("num_tokens", 14522), ("num_batches", 64)):
            require(source_result[key] == notebook_result[key] == expected, f"{stage}: complete dev {key}", checks)
        for key in ("loss", "intent_loss", "slot_loss"):
            require(source_result[key] == notebook_result[key], f"{stage}: exact global development {key}", checks)
        require(source_result["record_ids"] == notebook_result["record_ids"] == [item.record_id for item in original["dev"]],
                f"{stage}: exact original development ID order", checks)
        require(tensor_state_fingerprint(source_model) == tensor_state_fingerprint(notebook_model) == pretrained_state,
                f"{stage}: development inference leaves all model tensors unchanged", checks)
        gold_source = tuple(item.gold_prediction() for item in original["dev"])
        gold_notebook = tuple(item.gold_prediction() for item in notebook_original["dev"])
        require(evaluate_predictions(gold_source, source_result["predictions"]).to_dict() ==
                namespace["evaluate_predictions"](gold_notebook, notebook_result["predictions"]).to_dict(),
                f"{stage}: independent shared evaluator invocation agrees", checks)
        score = (notebook_result["metrics"].intent_accuracy + notebook_result["metrics"].slot_micro_f1) / 2
        require(abs(score - payload["selection_score"]) <= 1e-12,
                f"{stage}: saved development selection score reproduced", checks)
        outcomes[stage] = {"parameter_counts": expected_counts, "initial_state_sha256": init_fingerprint,
                           "selected_checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
                           "checkpoint_sha256": file_hash(checkpoint), "selected_epoch": payload["epoch"],
                           "checkpoint_state_sha256": pretrained_state, "original_32_train_losses": batch_results,
                           "development_metrics": notebook_result["metrics"].to_dict(),
                           "development_examples": 2033, "development_tokens": 14522, "development_batches": 64,
                           "development_predictions_sha256": canonical_hash(notebook_predictions),
                           "exact_ordered_development_predictions": True,
                           "selection_score_absolute_difference": abs(score - payload["selection_score"])}
        del source_model, notebook_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {"schema_version": 1, "status": "passed", "passed": True,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "notebook": str(notebook_path.relative_to(ROOT)).replace("\\", "/")
                        if notebook_path.is_relative_to(ROOT) else notebook_path.name,
            "notebook_sha256": file_hash(notebook_path), "auditor_sha256": file_hash(Path(__file__)),
            "notebook_code_sha256": canonical_hash(code_hashes), "code_cell_sha256": code_hashes,
            "definition_cells": blocks,
            "definition_source_sha256": canonical_hash(blocks), "frozen_definition_ast_parity": ast_evidence,
            "device": str(device), "python": sys.version, "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "dataset_sha256": DATA_SHA256, "mappings_canonical_sha256": MAPPINGS_SHA256,
            "character_mappings_canonical_sha256": CHARACTERS_SHA256,
            "inline_selected_configuration_sha256": {name: canonical_hash(value)
                                                      for name, value in embedded_settings.items()},
            "stages": outcomes, "check_count": len(checks), "checks": checks,
            "test_forward_passes": 0, "optimizer_steps": 0,
            "training_records_used_for_diagnostics": [item.record_id for item in original["train"][:32]],
            "scope": "Conversion parity only: original train forward diagnostics and selected-checkpoint development inference; no notebook actions, training updates or test inference."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default="main.ipynb")
    parser.add_argument("--output", default="artifacts/phase7/parity.json")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "42" or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        environment = dict(os.environ, PYTHONHASHSEED="42", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        return subprocess.call([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env=environment)
    notebook_path = (ROOT / args.notebook).resolve()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Audit output must remain within this workspace")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run(notebook_path, args.device)
        result = 0
    except Exception as error:
        report = {"schema_version": 1, "status": "failed", "passed": False,
                  "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                  "notebook": str(notebook_path), "error": str(error),
                  "traceback": traceback.format_exc(), "test_forward_passes": 0, "optimizer_steps": 0}
        result = 1
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"status": report["status"], "checks": report.get("check_count", 0),
                      "output": str(output), "error": report.get("error"),
                      "test_forward_passes": 0, "optimizer_steps": 0}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())

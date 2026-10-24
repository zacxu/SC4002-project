"""Reproducible Stage 1 execution and bounded Phase 3 verification."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.acquisition import file_sha256
from src.audit import EVALUATION_PROFILE, PREPROCESSING_PROFILE
from src.data import FittedMappings, collate_examples, fit_mappings, load_massive, prepare_split
from src.models import BaselineConfig, JointBiLSTM, enforce_parameter_cap
from src.runtime import configuration_hash, configure_reproducibility, dataloader_generator, relative_project_path, select_device
from src.training import compute_losses, evaluate_model, fit_model, train_epoch


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_context(root: Path, protocol: dict, stage_config: dict, device: str):
    if stage_config["schema_version"] != 1 or stage_config["stage"] != "stage1":
        raise ValueError("Expected a version 1 Stage 1 configuration")
    if protocol["preprocessing"] != PREPROCESSING_PROFILE or protocol["evaluation"] != EVALUATION_PROFILE:
        raise ValueError("Shared preprocessing/evaluation changed since the Phase 2 convention")
    training = stage_config["training"]
    if training["optimizer"] != "AdamW":
        raise ValueError("Stage 1 currently implements the declared AdamW optimizer")
    for name in ("batch_size", "max_epochs"):
        if type(training[name]) is not int or training[name] < 1:
            raise ValueError(f"Invalid training configuration: {name}")
    for name in ("learning_rate", "intent_loss_weight", "slot_loss_weight", "gradient_clip_norm"):
        if not math.isfinite(training[name]) or training[name] <= 0:
            raise ValueError(f"Invalid training configuration: {name}")
    if not math.isfinite(training["weight_decay"]) or training["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    configure_reproducibility(protocol["runtime"])
    torch.set_default_dtype(torch.float32)
    selected_device = select_device(device)
    dataset = relative_project_path(root, protocol["paths"]["dataset"])
    audit_dir = relative_project_path(root, protocol["paths"]["phase2"])
    audit = json.loads((audit_dir / "data_audit.json").read_text(encoding="utf-8"))
    dataset_hash = file_sha256(dataset)
    if dataset_hash != audit["dataset_sha256"] or audit["status"] != "passed":
        raise ValueError("Dataset differs from the passing Phase 2 audit")
    for name in ("src/data.py", "src/schema.py", "src/metrics.py"):
        if file_sha256(root / name) != audit["source_hashes"][name]:
            raise ValueError(f"Shared Phase 2 data/scoring code changed: {name}")
    mapping_path = audit_dir / "preprocessing.json"
    if file_sha256(mapping_path) != audit["preprocessing_sha256"]:
        raise ValueError("Training-fitted mappings differ from Phase 2 evidence")
    mappings = FittedMappings.from_dict(json.loads(mapping_path.read_text(encoding="utf-8")))
    # Loading verifies original records structurally. Only train/dev are prepared
    # for model execution; no test example reaches the network in Phase 3.
    splits = load_massive(dataset)
    if fit_mappings(splits["train"]).to_dict() != mappings.to_dict():
        raise ValueError("Mappings do not match a fresh fit on original train only")
    prepared = {partition: prepare_split(splits[partition], mappings) for partition in ("train", "dev")}
    return selected_device, mappings, prepared, dataset_hash


def _loader(prepared, partition: str, batch_size: int, seed: int) -> DataLoader:
    return DataLoader(prepared, batch_size=batch_size, shuffle=partition == "train", drop_last=False,
                      num_workers=0, collate_fn=collate_examples, generator=dataloader_generator(seed))


def _new_model(mappings: FittedMappings, stage_config: dict, device) -> JointBiLSTM:
    return JointBiLSTM(len(mappings.token_to_id), len(mappings.intent_to_id), len(mappings.tag_to_id),
                       BaselineConfig.from_dict(stage_config["model"])).to(device)


def _optimizer(model, training: dict, learning_rate: float | None = None):
    return torch.optim.AdamW(model.parameters(), lr=training["learning_rate"] if learning_rate is None else learning_rate,
                             weight_decay=training["weight_decay"])


def _new_output(root: Path, protocol: dict, stage_config: dict, device, purpose: str, dataset_hash: str):
    effective = {"protocol": protocol, "stage_config": stage_config, "device": str(device), "purpose": purpose}
    digest = configuration_hash(effective)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = root / "artifacts/phase3" if purpose == "diagnostic" else relative_project_path(root, protocol["paths"]["runs"])
    output = base / f"stage1_{stamp}_{digest[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    sources = {name: file_sha256(root / name) for name in (
        "src/models.py", "src/training.py", "src/experiment.py", "src/data.py", "src/schema.py",
        "src/metrics.py", "src/runtime.py", "scripts/run_experiment.py", "scripts/verify_checkpoint.py")}
    metadata = {"purpose": purpose, "created_at_utc": stamp, "config_sha256": digest,
                "effective_configuration": effective, "dataset_sha256": dataset_hash, "source_hashes": sources,
                "environment": {"python": sys.version, "executable": sys.executable,
                    "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
                    "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
                "reportable_experiment": purpose == "experiment"}
    _write_json(output / "run.json", metadata)
    _write_json(output / "protocol.json", protocol)
    _write_json(output / "stage_config.json", stage_config)
    return output, sources


def _fresh_reload(root: Path, checkpoint: str, device) -> dict:
    command = [sys.executable, str(root / "scripts/verify_checkpoint.py"), "--project-root", str(root),
               "--checkpoint", str(Path(checkpoint).resolve()), "--device", str(device),
               "--protocol", (Path(checkpoint).parent / "protocol.json").relative_to(root).as_posix()]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    report_path = Path(checkpoint).parent / "fresh_reload_process.txt"
    report_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Fresh-process checkpoint verification failed; see {report_path}")
    return json.loads((Path(checkpoint).parent / "reload_check.json").read_text(encoding="utf-8"))


def verify_checkpoint_file(root: Path, protocol: dict, checkpoint: Path, device: str) -> dict:
    from src.training import load_checkpoint, verify_reloaded_checkpoint

    configure_reproducibility(protocol["runtime"])
    selected_device = select_device(device)
    model, mappings, payload = load_checkpoint(checkpoint, selected_device, expected_protocol=protocol)
    del model
    dataset_path = relative_project_path(root, protocol["paths"]["dataset"])
    if file_sha256(dataset_path) != payload["data_sha256"]:
        raise ValueError("Checkpoint source dataset changed")
    for name, expected_hash in payload["source_hashes"].items():
        if file_sha256(root / name) != expected_hash:
            raise ValueError(f"Checkpoint source identity changed: {name}")
    splits = load_massive(dataset_path)
    if fit_mappings(splits["train"]).to_dict() != mappings.to_dict():
        raise ValueError("Checkpoint mappings do not match a train-only fit")
    dev = prepare_split(splits["dev"], mappings)
    loader = _loader(dev, "dev", payload["training_config"]["batch_size"], protocol["runtime"]["seed"])
    result = verify_reloaded_checkpoint(checkpoint, loader, selected_device,
                                        expected_mappings=mappings, expected_protocol=protocol,
                                        expected_data_sha256=payload["data_sha256"])
    result.update(fresh_python_process=True, test_evaluated=False)
    _write_json(checkpoint.parent / "reload_check.json", result)
    return result


def _padding_check(model, batch: dict, device) -> dict:
    model.eval()
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    lengths = batch["lengths"]
    extra = 4
    with torch.inference_mode():
        original = model(ids, lengths, mask)
        padded = model(torch.cat((ids, ids.new_zeros((ids.shape[0], extra))), dim=1), lengths,
                       torch.cat((mask, mask.new_zeros((mask.shape[0], extra))), dim=1))
    differences = {
        "intent": float((original["intent_logits"] - padded["intent_logits"]).abs().max()),
        "real_slot_logits": float((original["slot_logits"][mask] - padded["slot_logits"][:, :ids.shape[1]][mask]).abs().max()),
    }
    torch.testing.assert_close(original["intent_logits"], padded["intent_logits"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(original["slot_logits"][mask], padded["slot_logits"][:, :ids.shape[1]][mask], atol=1e-6, rtol=1e-6)
    return {"passed": True, "extra_padding_tokens": extra, "max_absolute_differences": differences,
            "intent_shape": list(original["intent_logits"].shape), "slot_shape": list(original["slot_logits"].shape)}


def _learning_probe(root: Path, protocol: dict, stage_config: dict, device, mappings, train, output: Path) -> dict:
    configure_reproducibility(protocol["runtime"])
    model = _new_model(mappings, stage_config, device)
    full_loader = _loader(train, "train", stage_config["training"]["batch_size"], protocol["runtime"]["seed"])
    first_batch = next(iter(full_loader))
    by_id = {item.example.record_id: item for item in train}
    diagnostic_view = tuple(by_id[record_id] for record_id in first_batch["record_ids"])
    fixed_loader = _loader(diagnostic_view, "dev", len(diagnostic_view), protocol["runtime"]["seed"])
    optimizer = _optimizer(model, stage_config["training"], stage_config["diagnostics"]["overfit_learning_rate"])
    weights = {"intent_weight": stage_config["training"]["intent_loss_weight"],
               "slot_weight": stage_config["training"]["slot_loss_weight"]}
    initial = evaluate_model(model, fixed_loader, mappings, device, **weights)
    trajectory = [{"step": 0, **{key: initial[key] for key in ("loss", "intent_loss", "slot_loss")}}]
    steps = stage_config["diagnostics"]["overfit_steps"]
    largest_gradient = 0.0
    for step in range(1, steps + 1):
        training = train_epoch(model, fixed_loader, optimizer, device, **weights,
                               grad_clip=stage_config["training"]["gradient_clip_norm"])
        largest_gradient = max(largest_gradient, training["max_gradient_norm"])
        if step in {1, 10, 25, 50, steps}:
            evaluation = evaluate_model(model, fixed_loader, mappings, device, **weights)
            trajectory.append({"step": step, **{key: evaluation[key] for key in ("loss", "intent_loss", "slot_loss")}})
    final = evaluate_model(model, fixed_loader, mappings, device, **weights)
    passed = final["intent_loss"] < 0.5 * initial["intent_loss"] and final["slot_loss"] < 0.5 * initial["slot_loss"]
    report = {"purpose": "diagnostic", "scope": "Repeated fixed original training minibatch; not assessed performance or a replacement split.",
              "record_ids": list(first_batch["record_ids"]), "num_examples": len(diagnostic_view), "steps": steps,
              "learning_rate": stage_config["diagnostics"]["overfit_learning_rate"],
              "initial_losses": {key: initial[key] for key in ("loss", "intent_loss", "slot_loss")},
              "final_losses": {key: final[key] for key in ("loss", "intent_loss", "slot_loss")},
              "trajectory": trajectory, "maximum_gradient_norm": largest_gradient,
              "both_task_losses_reduced_by_more_than_half": passed, "passed": passed,
              "test_evaluated": False}
    _write_json(output / "learning_probe.json", report)
    if not passed:
        raise RuntimeError("Fixed-batch learning probe failed; inspect saved loss trajectory before full training")
    return report


def run_diagnostics(root: Path, protocol: dict, stage_config: dict, device: str = "cuda") -> dict:
    selected_device, mappings, prepared, dataset_hash = _load_context(root, protocol, stage_config, device)
    output, sources = _new_output(root, protocol, stage_config, selected_device, "diagnostic", dataset_hash)
    model = _new_model(mappings, stage_config, selected_device)
    counts = enforce_parameter_cap(model, protocol["limits"]["total_parameters"])
    training = copy.deepcopy(stage_config["training"])
    training["max_epochs"] = stage_config["diagnostics"]["epochs"]
    train_loader = _loader(prepared["train"], "train", training["batch_size"], protocol["runtime"]["seed"])
    dev_loader = _loader(prepared["dev"], "dev", training["batch_size"], protocol["runtime"]["seed"])
    batch = next(iter(_loader(prepared["train"], "train", training["batch_size"], protocol["runtime"]["seed"])))
    padding = _padding_check(model, batch, selected_device)
    fit = fit_model(model, train_loader, dev_loader, _optimizer(model, training), mappings, selected_device,
                    training, protocol, output, source_hashes=sources, data_sha256=dataset_hash,
                    purpose="diagnostic", max_train_batches=stage_config["diagnostics"]["max_train_batches"],
                    max_dev_batches=stage_config["diagnostics"]["max_dev_batches"])
    reload_check = _fresh_reload(root, fit["checkpoint"], selected_device)
    learning_probe = _learning_probe(root, protocol, stage_config, selected_device, mappings, prepared["train"], output)
    report = {"status": "passed", "purpose": "diagnostic", "reportable_experiment": False,
              "output_directory": str(output), "device": str(selected_device), "parameter_counts": counts,
              "padding_check": padding, "epoch_checks": {"epochs": fit["epochs_trained"],
                  "train_records_per_epoch": [row["train_num_examples"] for row in fit["history"]],
                  "dev_records_per_epoch": [row["dev_num_examples"] for row in fit["history"]]},
              "fresh_process_reload": reload_check,
              "fixed_train_batch_learning": {"passed": learning_probe["passed"],
                  "num_examples": learning_probe["num_examples"], "steps": learning_probe["steps"],
                  "initial_losses": learning_probe["initial_losses"], "final_losses": learning_probe["final_losses"]},
              "training_seconds": fit["training_seconds"], "source_hashes": sources,
              "dataset_sha256": dataset_hash, "protocol_sha256": configuration_hash(protocol),
              "stage_config_sha256": configuration_hash(stage_config), "test_model_evaluations": 0,
              "scope": "Phase 3 correctness and learning checks only. Full Stage 1 experiments remain Phase 4."}
    _write_json(output / "phase3_verification.json", report)
    return report


def run_full_experiment(root: Path, protocol: dict, stage_config: dict, device: str = "cuda") -> dict:
    selected_device, mappings, prepared, dataset_hash = _load_context(root, protocol, stage_config, device)
    output, sources = _new_output(root, protocol, stage_config, selected_device, "experiment", dataset_hash)
    model = _new_model(mappings, stage_config, selected_device)
    training = stage_config["training"]
    train_loader = _loader(prepared["train"], "train", training["batch_size"], protocol["runtime"]["seed"])
    dev_loader = _loader(prepared["dev"], "dev", training["batch_size"], protocol["runtime"]["seed"])
    result = fit_model(model, train_loader, dev_loader, _optimizer(model, training), mappings, selected_device,
                       training, protocol, output, source_hashes=sources, data_sha256=dataset_hash)
    result["fresh_process_reload"] = _fresh_reload(root, result["checkpoint"], selected_device)
    result["output_directory"] = str(output)
    _write_json(output / "experiment_summary.json", result)
    return result

"""Verify the isolated runtime and write genuine Phase 1 setup evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import site
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="configs/protocol.json")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    sys.path.insert(0, str(root))
    from src.runtime import configure_reproducibility, load_protocol, relative_project_path, select_device

    protocol = load_protocol(root, args.config)
    runtime = protocol["runtime"]
    expected_environment = {"PYTHONHASHSEED": str(runtime["seed"]), "CUBLAS_WORKSPACE_CONFIG": runtime["cublas_workspace_config"]}
    if any(os.environ.get(key) != value for key, value in expected_environment.items()):
        child_environment = os.environ.copy()
        child_environment.update(expected_environment)
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env=child_environment, check=False)
        raise SystemExit(completed.returncode)

    configure_reproducibility(runtime)
    output_dir = relative_project_path(root, protocol["paths"]["setup"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for variable, subdirectory in {"MPLCONFIGDIR": "matplotlib", "IPYTHONDIR": "ipython",
                                   "JUPYTER_RUNTIME_DIR": "jupyter_runtime"}.items():
        cache_path = output_dir / subdirectory
        cache_path.mkdir(exist_ok=True)
        os.environ[variable] = str(cache_path)
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nbformat
    import nbconvert  # noqa: F401 -- intentionally verify required imports
    import ipykernel  # noqa: F401
    import jupyterlab  # noqa: F401
    from nbclient import NotebookClient

    device = select_device(args.device or runtime["device"])

    def autograd_check(target: torch.device) -> dict:
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=target, requires_grad=True)
        loss = (x @ x.T).sum()
        loss.backward()
        expected_gradient = torch.tensor([[8.0, 12.0], [8.0, 12.0]], device=target)
        if not torch.equal(x.grad, expected_gradient) or float(loss.detach().cpu()) != 52.0:
            raise RuntimeError(f"Tensor/autograd smoke failed on {target}")
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        return {"passed": True, "loss": float(loss.detach().cpu()), "gradient": x.grad.detach().cpu().tolist()}

    cpu_result = autograd_check(torch.device("cpu"))
    device_result = autograd_check(device)
    figure, axis = plt.subplots(figsize=(4, 2.5))
    axis.plot([0, 1], [0, 1])
    axis.set(title="Phase 1 plotting check (not training results)", xlabel="x", ylabel="y")
    figure.tight_layout()
    figure.savefig(output_dir / "smoke_plot.png", dpi=100)
    plt.close(figure)

    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        "import os, sys, torch, numpy, matplotlib\n"
        f"assert sys.prefix == {sys.prefix!r}\n"
        f"assert os.environ['PYTHONHASHSEED'] == {expected_environment['PYTHONHASHSEED']!r}\n"
        f"assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == {expected_environment['CUBLAS_WORKSPACE_CONFIG']!r}\n"
        f"device = torch.device({str(device)!r})\n"
        "x = torch.ones((2, 2), device=device)\n"
        "assert (x @ x).sum().item() == 8.0\n"
        "print('Isolated kernel:', sys.prefix)\n"
        "print('Torch:', torch.__version__)\n"
        "print('CUDA available:', torch.cuda.is_available())\n"
        "print('Phase 1 notebook smoke passed; no dataset or model training.')"
    )])
    notebook.metadata.kernelspec = {"name": "sc4002", "display_name": "SC4002 (.venv)", "language": "python"}
    NotebookClient(notebook, timeout=90, kernel_name="sc4002", resources={"metadata": {"path": str(root)}}).execute()
    nbformat.write(notebook, output_dir / "notebook_smoke.ipynb")

    packages = {name: importlib.metadata.version(name) for name in (
        "pip", "torch", "numpy", "matplotlib", "jupyterlab", "ipykernel", "nbformat", "nbconvert", "nbclient"
    )}
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append({"index": index, "name": properties.name, "total_memory_bytes": properties.total_memory,
                         "compute_capability": f"{properties.major}.{properties.minor}"})
    try:
        gpu_driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        gpu_driver = None
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
        "isolated_environment": sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE,
        "user_site_packages_enabled": site.ENABLE_USER_SITE,
        "platform": platform.platform(), "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
        "disk_free_bytes": shutil.disk_usage(root).free,
        "project_root": str(root), "dataset_present": relative_project_path(root, protocol["paths"]["dataset"]).is_file(),
        "selected_device": str(device), "cuda_available": torch.cuda.is_available(),
        "torch_cuda_runtime": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "nvidia_driver": gpu_driver, "gpus": gpus, "packages": packages,
        "runtime_protocol": runtime, "effective_environment": expected_environment,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cpu_autograd": cpu_result, "selected_device_autograd": device_result,
        "matplotlib_png": "smoke_plot.png", "executed_notebook": "notebook_smoke.ipynb",
        "kernel_smoke_passed": True,
        "phase_scope": "Setup verification only; no dataset download, training, or performance benchmark.",
    }
    if not report["isolated_environment"]:
        raise RuntimeError("Run this check with the isolated project environment")
    (output_dir / "environment.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"environment_report": str(output_dir / "environment.json"), "device": str(device), "gpu_names": [gpu["name"] for gpu in gpus],
                      "packages": packages, "autograd": "passed", "notebook": "passed", "plot": "passed"}, indent=2))


if __name__ == "__main__":
    main()


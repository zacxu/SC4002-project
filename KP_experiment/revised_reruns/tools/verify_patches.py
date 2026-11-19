"""Checks the patched notebook code before any training: data, all three embeddings and SAM.

Usage: python -I verify_patches.py <patched notebook> <dataset cache> <word2vec zip> <fasttext zip> <glove zip>
"""
import json
import sys
import types
from pathlib import Path

ORIGINAL_MATRIX_SHA = "8607fc1f115fb5cf8d2c129d40d2aa949119f14bbfd27b0266ae9280432c28f1"
ORIGINAL_ARCHIVE_SHA = "6d97295d630f7e50d6f2ab2242926f0b962087054ee3b96be2b44a0a2ce1ae19"


def definitions(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    module = types.ModuleType("patched_notebook")
    sys.modules[module.__name__] = module   # dataclasses look their module up here
    namespace = module.__dict__
    for cell in document["cells"]:
        if cell["cell_type"] == "code" and "definitions" in cell.get("metadata", {}).get("tags", []):
            exec(compile("".join(cell["source"]), "cell", "exec"), namespace)
    return namespace


def check_sam(ns):
    """Weights return after the second pass, and the gradient really changes."""
    import torch
    ns["rt_seed"]({"seed": 42})
    tags = ["O", "B-x", "I-x"]
    config = ns["RevisedTransformerConfig"](word_dim=8, d_model=16, n_heads=2, n_layers=1, d_ff=32, char_dim=4,
                                            char_channels=4, intent_context_dim=4, attention_pool_dim=4, head_hidden_dim=8)
    model = ns["JointTransformer"](12, 3, tags, 6, config=config)
    batch = {"word_ids": torch.tensor([[2, 3, 4], [5, 6, 0]]), "mask": torch.tensor([[True, True, True], [True, True, False]]),
             "char_ids": torch.ones((2, 3, 5), dtype=torch.long), "intent_ids": torch.tensor([0, 1]),
             "slot_ids": torch.tensor([[0, 1, 2], [1, 0, 0]])}
    batch["char_ids"][1, 2] = 0
    outputs = model(batch["word_ids"], batch["mask"], batch["char_ids"])
    model.loss(outputs, batch["intent_ids"], batch["slot_ids"], batch["mask"])["total"].backward()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    first = {name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None}
    ns["_rt_sam_second_pass"](model, batch, 0.2)
    drift = max(float((parameter.detach() - before[name]).abs().max()) for name, parameter in model.named_parameters())
    change = max(float((parameter.grad - first[name]).abs().max()) for name, parameter in model.named_parameters() if parameter.grad is not None)
    assert drift < 1e-5, f"weights moved by {drift}"
    assert change > 1e-4, f"gradient unchanged ({change})"
    print(f"SAM: weights restored (max drift {drift:.2e}); gradient changed (max {change:.3f})", flush=True)


def main(notebook, dataset, word2vec, fasttext, glove):
    ns = definitions(notebook)
    check_sam(ns)
    raw, manifest = ns["ensure_verified_massive"](dataset)
    print("Dataset verified against the pinned SHA:", manifest["dataset_sha256"], manifest["counts"], flush=True)
    word_to_id = ns["build_data"](raw)["mappings"]["word_to_id"]
    print("Train vocabulary:", len(word_to_id), flush=True)
    for source, archive in (("word2vec", word2vec), ("fasttext", fasttext), ("glove", glove)):
        _, info = ns["task_embeddings"](word_to_id, source=source, seed=42, dimension=300, archive_path=archive, download=False)
        print(source, {key: info.get(key) for key in ("source_member", "source_vector_count", "pretrained_initialized",
              "random_initialized", "pretrained_coverage", "mean_vector_norm", "source_archive_sha256", "matrix_sha256")}, flush=True)
        if source == "word2vec":
            assert info["matrix_sha256"] == ORIGINAL_MATRIX_SHA, "Word2Vec matrix differs from the original notebook's"
            assert info["source_archive_sha256"] == ORIGINAL_ARCHIVE_SHA, "Word2Vec archive differs from the original notebook's"
            print("Word2Vec matrix and archive match the original notebook exactly", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:6])

"""Reload a saved checkpoint in a fresh process and re-evaluate dev without training."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from src.config import ModelConfig, CKPT_DIR, DEVICE
from src.experiment import Context
from src.models import build_model
from src.evaluate import predict
from src.train import make_loaders

name = sys.argv[1]
ckpt = torch.load(CKPT_DIR / f"{name}.pt", map_location="cpu")
ctx = Context(verbose=False)
assert ckpt["vocab"]["word_vocab"] == ctx.assets.word_vocab
assert ckpt["vocab"]["slot_labels"] == ctx.assets.slot_labels
assert ckpt["vocab"]["intent_labels"] == ctx.assets.intent_labels
cfg = ModelConfig.from_dict(ckpt["model_config"])
model = build_model(cfg, ctx.assets)
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE)
ds = ctx.datasets(cfg, ckpt["meta"].get("k_extra"))
_, dl = make_loaders(ctx.assets, ds["train"], ds["dev"], 64, 0)
r = predict(model, dl, ctx.assets)
print(f"{name}: reloaded dev intent {r['intent_accuracy']:.4f} slot {r['slot_micro_f1']:.4f} | "
      f"logged {ckpt['dev']['intent_accuracy']:.4f} {ckpt['dev']['slot_micro_f1']:.4f} | best epoch {ckpt['best_epoch']}")
assert abs(r["joint_score"] - ckpt["dev"]["joint_score"]) < 1e-6
print("RELOAD OK")

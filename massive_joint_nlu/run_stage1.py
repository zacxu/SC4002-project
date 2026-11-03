"""Stage 1: hand-off baseline (Word2Vec + 4-layer Transformer, joint heads). Saves checkpoints/stage1_baseline.pt."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import ModelConfig, TrainConfig
from src.experiment import Context, run, STAGE1_NAME

if __name__ == "__main__":
    ctx = Context()
    run(ctx, STAGE1_NAME, ModelConfig(), TrainConfig(), init=None, rung="S1",
        note="hand-off Stage-1 config: Emb300->256, 4L Transformer, AdamW 3e-4, 12 ep, patience 3")

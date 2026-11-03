"""Recomputes matched_gap and exact match of finished runs from their saved dev frames. No GPU, no retraining.
The Stage-1 row also gets the current exact-match definition and the fields added later."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json

import pandas as pd

from src.config import RESULTS_DIR
from src.evaluate import metrics_from_frame, matched_control, GROUPS
from src.experiment import write_json_atomic, rebuild_leaderboard

done = 0
for p in sorted(RESULTS_DIR.glob("*/dev_metrics.json")):
    frame_path = p.parent / "dev_pred_frame.pkl"
    if not frame_path.exists():
        continue
    row = json.loads(p.read_text(encoding="utf-8"))
    merged = pd.read_pickle(frame_path)
    rows = []
    for g in GROUPS:
        sub = merged[merged["train_oov_group"] == g]
        rows.append({"group": g, **{k: v for k, v in metrics_from_frame(sub).items() if k != "joint_score"}})
    rows.append({"group": "Overall", **{k: v for k, v in metrics_from_frame(merged).items() if k != "joint_score"}})
    table = pd.DataFrame(rows)
    table.to_csv(p.parent / "dev_oov_table.csv", index=False)
    idx = table.set_index("group")
    row["dev_exact"] = round(float(idx.loc["Overall", "exact_match"]), 2)
    row["intent_nonslot_oov"] = round(float(idx.loc["NonSlot-Train-OOV", "intent_accuracy"]), 2)
    row.setdefault("frozen_dim", 300)
    row["matched_gap_old_single_row"] = row.get("matched_gap_old_single_row", row.get("matched_gap"))
    row["matched_gap"] = round(matched_control(merged)["gap"], 2)
    write_json_atomic(p, row)
    done += 1
lb = rebuild_leaderboard()
print("rewrote", done, "runs")
print(lb[["run", "dev_joint", "dev_exact", "matched_gap_old_single_row", "matched_gap"]].sort_values("dev_joint", ascending=False).head(8).to_string(index=False))

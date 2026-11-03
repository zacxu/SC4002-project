import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.experiment import Context
ctx = Context()
s = ctx.emb_stats
assert (s["pretrained_init"], s["random_init"], s["coverage_pct"]) == (4731, 567, 89.30), s
for split, exp in {"dev": {"No-Train-OOV": 1669, "NonSlot-Train-OOV": 149, "Slot-Train-OOV": 215},
                   "test": {"No-Train-OOV": 2411, "NonSlot-Train-OOV": 222, "Slot-Train-OOV": 341}}.items():
    got = ctx.oov[split]["train_oov_group"].value_counts().to_dict()
    print(split, got)
    assert got == exp, (split, got)
print("dev pretrained groups", ctx.oov["dev"]["pretrained_oov_group"].value_counts().to_dict())
print("ASSETS OK")

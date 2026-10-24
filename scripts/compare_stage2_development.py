"""Compare baseline and every completed Stage 2 candidate using saved dev outputs."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage2_analysis import compare_development, write_comparison


def main() -> None:
    result = compare_development(ROOT)
    write_comparison(ROOT, result)
    print(json.dumps({"status": "complete", "partition": "dev", "test_used": False,
                      "selected_candidate": result["selected_candidate"], "rows": result["rows"],
                      "paired_changes": result["paired_selected_vs_baseline"],
                      "remaining_error_ids": [row["record_id"] for row in result["remaining_error_cases"]]}, indent=2))


if __name__ == "__main__":
    main()

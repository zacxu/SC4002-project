"""Reproduce the saved-baseline development diagnosis without model inference."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diagnostics import render_markdown, run_diagnosis


def main() -> None:
    result = run_diagnosis(ROOT)
    destination = ROOT / "artifacts/phase5/diagnosis.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (ROOT / "PHASE5_DIAGNOSIS.md").write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"status": "complete", "partition": "dev", "test_used": False,
                      "overall": result["overall"], "oov": {key: value for key, value in result["strata"].items()
                                                             if key.startswith("oov/")},
                      "false_negative_categories": result["false_negative_categories"],
                      "diagnosis": str(destination.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()

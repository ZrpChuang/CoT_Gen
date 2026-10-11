"""Read the four benchmark scores without depending on serving packages."""

from __future__ import annotations

import json
from pathlib import Path

from settings import JUDGE


def score(path: Path, benchmark: str, judge: str = JUDGE) -> float | None:
    if not path.is_file():
        return None
    result = json.loads(path.read_text())
    if result.get("status") != "complete":
        return None
    if benchmark == "ifeval":
        value = result["metrics"]["prompt_strict"]["score"]
    elif benchmark == "multi_challenge":
        value = result["judges"][judge]["overall"]["score"]
    elif benchmark == "eq_bench3":
        value = result["judges"][judge]["rubric_0_100"]["score"]
    elif benchmark == "arena_hard_v2":
        value = result["judges"][judge]["weighted_win_rate"]["hard_prompt"]["score"]
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return float(value)

#!/usr/bin/env python3
"""Generate a score table from existing, completed metric files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metrics import score
from settings import BASE_DIR, BENCHMARKS, METHODS, MODELS, PROJECT_DIR

LABELS = {
    "ifeval": "IFEval",
    "multi_challenge": "MultiChallenge",
    "eq_bench3": "EQ-Bench3",
    "arena_hard_v2": "Arena-Hard-v2",
}


def metric_path(method: str, benchmark: str, model: str) -> Path:
    if method == "Baseline":
        return PROJECT_DIR / "test_scripts" / model / benchmark / "metrics.json"
    return BASE_DIR / method / benchmark / model / "metrics.json"


def training_step(method: str, model: str) -> str:
    if method == "Baseline":
        return "0"
    marker = BASE_DIR / method / model / "training/complete.json"
    return (
        str(json.loads(marker.read_text())["global_step"]) if marker.is_file() else "—"
    )


def render() -> str:
    columns = ["方法", "训练步数 9B/27B"]
    columns.extend(
        f"{LABELS[bench]} {model}" for bench in BENCHMARKS for model in MODELS
    )
    lines = ["# Baseline results", "", "| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |")
    for method in ("Baseline", *METHODS):
        steps = "/".join(training_step(method, model) for model in MODELS)
        values = []
        for benchmark in BENCHMARKS:
            for model in MODELS:
                value = score(metric_path(method, benchmark, model), benchmark)
                values.append("—" if value is None else f"{value:.2f}")
        lines.append("| " + " | ".join([method, steps, *values]) + " |")
    lines += [
        "",
        "Scores come from completed local metric files. A dash means no completed metric was found.",
        "Baseline uses the original model. Other rows use the final trained checkpoint.",
        "This table does not convert local EQ or Arena metrics to official leaderboard scores.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="Write Markdown here; defaults to stdout"
    )
    args = parser.parse_args()
    report = render()
    if args.output:
        args.output.write_text(report)
        print(args.output)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()

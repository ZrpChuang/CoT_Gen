#!/usr/bin/env python3
"""One entry point for reports, evidence checks, reusable training and evaluation."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from core.formats import candidate
from core.report import render
from core.settings import MODELS, ROOT, python_path
from core.storage import restore, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("report")
    sub.add_parser("validate")
    restore_parser = sub.add_parser("restore")
    restore_parser.add_argument("--archive", type=Path, required=True)
    restore_parser.add_argument("--output", type=Path, required=True)
    select = sub.add_parser("select-format")
    select.add_argument("--scores", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    train = sub.add_parser("train")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument(
        "--recipe", type=Path, default=ROOT / "configs/recipes/body_format.json"
    )
    train.add_argument("--model", choices=MODELS, required=True)
    train.add_argument("--gpus", required=True)
    train.add_argument("--dry-run", action="store_true")
    dev = sub.add_parser("dev")
    dev.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "report":
        print(render())
    elif args.command == "validate":
        print(json.dumps(validate(), indent=2))
    elif args.command == "restore":
        print(json.dumps(restore(args.archive, args.output), indent=2))
    elif args.command == "select-format":
        rows = [
            json.loads(line)
            for line in args.scores.read_text().split("\n")
            if line.strip()
        ]
        output = [
            {**row, "format": candidate(row["delta"], row.get("identical", False))}
            for row in rows
        ]
        if args.output.exists():
            raise ValueError("Selection output already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output)
        )
    elif args.command == "train":
        devices = [int(value) for value in args.gpus.split(",")]
        if not devices or len(set(devices)) != len(devices) or min(devices) < 0:
            raise ValueError("GPU IDs must be distinct nonnegative integers")
        command = [
            python_path("train"),
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={len(devices)}",
            str(ROOT / "core/train.py"),
            "--data",
            str(args.data.resolve()),
            "--output",
            str(args.output.resolve()),
            "--recipe",
            str(args.recipe.resolve()),
            "--model",
            args.model,
        ]
        print(" ".join(command), flush=True)
        if not args.dry_run:
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
            subprocess.run(command, check=True, env=environment, cwd=ROOT)
    elif args.command == "dev":
        tail = args.args[1:] if args.args[:1] == ["--"] else args.args
        subprocess.run(
            [python_path("evaluation"), str(ROOT / "core/dev_harness.py"), *tail],
            check=True,
            cwd=ROOT,
        )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2) from error

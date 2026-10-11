#!/usr/bin/env python3
"""Run the four baseline methods through data, training, evaluation, and reporting.

Each stage is resumable from its verified completion marker. Machine-specific
model paths, Python environments, and GPU assignments live in local_paths.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from core.settings import (
    BASE_DIR,
    BENCHMARKS,
    CODE_DIR,
    DATA_DIR,
    LOG_DIR,
    METHODS,
    MODELS,
    PROJECT_DIR,
    PROMPT_DIR,
    experiment,
    gpu_devices,
    model_path,
    python_path,
)


def record(stage: str, method: str = "", model: str = "", detail: str = "") -> None:
    LOG_DIR.mkdir(exist_ok=True)
    row = {
        "stage": stage,
        "method": method,
        "model": model,
        "detail": detail,
        "time": time.time(),
    }
    (LOG_DIR / "status.json").write_text(json.dumps(row, indent=2) + "\n")
    with (LOG_DIR / "ledger.tsv").open("a") as ledger:
        ledger.write(
            "\t".join(
                str(row[key]).replace("\n", " ")
                for key in ("time", "stage", "method", "model", "detail")
            )
            + "\n"
        )


def command(
    argv: list[str], name: str, dry_run: bool = False, env: dict | None = None
) -> None:
    print("$", " ".join(argv), flush=True)
    if dry_run:
        return
    LOG_DIR.mkdir(exist_ok=True)
    with (LOG_DIR / f"{name}.log").open("a") as log:
        subprocess.run(
            argv,
            cwd=BASE_DIR,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def validate() -> dict:
    config = experiment()
    missing: list[str] = []
    for model in MODELS:
        try:
            path = model_path(model)
            if not path.is_dir():
                missing.append(f"model {model}: {path}")
            gpu_devices(model)
        except (ValueError, RuntimeError) as exc:
            missing.append(str(exc))
    for path in [PROJECT_DIR / "training_data/sample/train_indices.jsonl"]:
        if not path.is_file():
            missing.append(str(path))
    for method in METHODS:
        if not (PROMPT_DIR / f"{method}.txt").is_file():
            missing.append(f"prompt: {method}")
        for model in MODELS:
            evaluation = BASE_DIR / method / model / "evaluation"
            for name in (
                "config.json",
                "deployment.json",
                "evaluate.py",
                "serve.py",
                "verify_results.py",
            ):
                if not (evaluation / name).is_file():
                    missing.append(str(evaluation / name))
    result = {
        "ready": not missing,
        "missing": missing,
        "methods": list(METHODS),
        "models": list(MODELS),
        "benchmarks": list(BENCHMARKS),
        "config_run_id": config.get("run_id"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def generate(dry_run: bool) -> None:
    prepared = DATA_DIR / "prepared_status.json"
    if prepared.is_file() and json.loads(prepared.read_text()).get("complete"):
        print("Prepared training split is frozen; generation skipped")
        return
    marker = DATA_DIR / "generation_status.json"
    if marker.is_file() and json.loads(marker.read_text()).get("complete"):
        print("Generated data already complete")
        return
    record("generating") if not dry_run else None
    command(
        [python_path("evaluation"), str(CODE_DIR / "build_data.py")],
        "generation",
        dry_run,
    )


def prepare(dry_run: bool) -> None:
    marker = DATA_DIR / "prepared_status.json"
    if marker.is_file() and json.loads(marker.read_text()).get("complete"):
        print("Prepared data already complete")
        return
    record("preparing") if not dry_run else None
    command(
        [python_path("train"), str(CODE_DIR / "prepare_training.py")],
        "prepare",
        dry_run,
    )


def train(method: str, model: str, dry_run: bool) -> None:
    marker = BASE_DIR / method / model / "training/complete.json"
    if marker.is_file():
        print(f"Training already complete: {method}/{model}")
        return
    devices = gpu_devices(model)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    record("training", method, model) if not dry_run else None
    command(
        [
            python_path("train"),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={len(devices)}",
            str(CODE_DIR / "train.py"),
            "--method",
            method,
            "--model",
            model,
        ],
        f"{method}__{model}__train",
        dry_run,
        env,
    )
    if not dry_run and not marker.is_file():
        raise RuntimeError(f"Training exited without completion marker: {marker}")


def evaluate(method: str, model: str, dry_run: bool) -> None:
    if not (BASE_DIR / method / model / "training/complete.json").is_file():
        raise RuntimeError(f"Training incomplete: {method}/{model}")
    audit = BASE_DIR / method / model / "evaluation/audit.json"
    if audit.is_file():
        print(f"Evaluation already audited: {method}/{model}")
        return
    record("evaluating", method, model) if not dry_run else None
    command(
        [
            python_path("evaluation"),
            str(CODE_DIR / "run_evaluation.py"),
            "--method",
            method,
            "--model",
            model,
        ],
        f"{method}__{model}__evaluation",
        dry_run,
    )
    if not dry_run and not audit.is_file():
        raise RuntimeError(f"Evaluation exited without audit: {audit}")


def report(dry_run: bool) -> None:
    command(
        [
            sys.executable,
            str(CODE_DIR / "summarize.py"),
            "--output",
            str(LOG_DIR / "summary.generated.md"),
        ],
        "report",
        dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without running stages"
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage in ("validate", "generate", "prepare", "report", "all"):
        sub.add_parser(stage)
    for stage in ("train", "evaluate"):
        phase = sub.add_parser(stage)
        phase.add_argument("--method", choices=METHODS, required=True)
        phase.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()

    try:
        if args.stage == "validate":
            if not validate()["ready"]:
                raise SystemExit(2)
        elif args.stage == "generate":
            generate(args.dry_run)
        elif args.stage == "prepare":
            prepare(args.dry_run)
        elif args.stage == "train":
            train(args.method, args.model, args.dry_run)
        elif args.stage == "evaluate":
            evaluate(args.method, args.model, args.dry_run)
        elif args.stage == "report":
            report(args.dry_run)
        else:
            if not validate()["ready"]:
                raise SystemExit(2)
            generate(args.dry_run)
            prepare(args.dry_run)
            for method in METHODS:
                for model in MODELS:
                    train(method, model, args.dry_run)
                    evaluate(method, model, args.dry_run)
            report(args.dry_run)
        if not args.dry_run and args.stage != "validate":
            record("completed", detail=args.stage)
    except Exception as exc:
        if not args.dry_run:
            record("failed", detail=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()

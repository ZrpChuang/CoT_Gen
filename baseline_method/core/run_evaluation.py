#!/usr/bin/env python3
"""Evaluate one trained method/model on the four frozen benchmark snapshots."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import time
import urllib.request
from pathlib import Path

from settings import BASE_DIR, BENCHMARKS, METHODS, MODELS, python_path


def wait_for_service(config: dict, timeout_seconds: int = 1800) -> None:
    deadline = time.monotonic() + timeout_seconds
    expected = config["target"]["model"]
    while time.monotonic() < deadline:
        try:
            for base in config["target"]["base_urls"]:
                with urllib.request.urlopen(
                    base.rstrip("/") + "/models", timeout=3
                ) as response:
                    models = json.load(response)["data"]
                if expected not in {model["id"] for model in models}:
                    raise RuntimeError(f"Serving model {expected} is not ready")
            return
        except (OSError, ValueError, KeyError, RuntimeError):
            time.sleep(10)
    raise TimeoutError(f"Serving model {expected} did not become ready")


def stage(directory: Path, benchmark: str, action: str, python: str) -> None:
    log_path = directory / "runtime" / f"{benchmark}_{action}.log"
    for attempt in range(1, 4):
        with log_path.open("a") as log:
            result = subprocess.run(
                [python, "evaluate.py", "--bench", benchmark, "--stage", action],
                cwd=directory,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode == 0:
            return
        if attempt < 3:
            time.sleep(10)
    raise RuntimeError(f"{benchmark} {action} failed after 3 attempts; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    directory = BASE_DIR / args.method / args.model / "evaluation"
    audit = directory / "audit.json"
    if audit.is_file():
        print(f"Already audited: {audit}")
        return
    config = json.loads((directory / "config.json").read_text())
    deployment = json.loads((directory / "deployment.json").read_text())
    if not Path(deployment["model_dir"]).is_dir():
        raise FileNotFoundError(f"Merged model missing: {deployment['model_dir']}")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "evaluation": str(directory),
                    "benchmarks": BENCHMARKS,
                    "model": config["target"]["model"],
                },
                indent=2,
            )
        )
        return

    directory.joinpath("runtime").mkdir(exist_ok=True)
    python = python_path("evaluation")
    started = False
    try:
        started = True
        subprocess.run([python, "serve.py", "start"], cwd=directory, check=True)
        wait_for_service(config)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            scores = []
            for benchmark in BENCHMARKS:
                stage(directory, benchmark, "generate", python)
                scores.append(pool.submit(stage, directory, benchmark, "score", python))
            # Scoring uses the judge API; release serving GPUs after generation.
            subprocess.run([python, "serve.py", "stop"], cwd=directory, check=True)
            started = False
            for score in scores:
                score.result()
        subprocess.run([python, "verify_results.py"], cwd=directory, check=True)
    finally:
        if started:
            subprocess.run([python, "serve.py", "stop"], cwd=directory, check=False)


if __name__ == "__main__":
    main()

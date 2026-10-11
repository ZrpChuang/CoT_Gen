"""Paths and immutable names shared by the baseline pipeline.

Machine-specific paths live in ``local_paths.json`` or environment variables.
The file is intentionally ignored by Git.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
BASE_DIR = CODE_DIR.parent
PROJECT_DIR = BASE_DIR.parent
CONFIG_DIR = BASE_DIR / "configs"
DATA_DIR = BASE_DIR / "data"
LOG_DIR = DATA_DIR / "logs"
PROMPT_DIR = CONFIG_DIR / "prompts"
METHODS = ("NEU", "SUP", "AUG-SUP", "SSR")
MODELS = ("Qwen_35_9B", "Qwen_38_27B")
BENCHMARKS = ("ifeval", "multi_challenge", "eq_bench3", "arena_hard_v2")
JUDGE = "gpt-6-astra"

MODEL_ENV = {
    "Qwen_35_9B": "RCOT_MODEL_QWEN_35_9B",
    "Qwen_38_27B": "RCOT_MODEL_QWEN_38_27B",
}


def local_paths() -> dict:
    """Read optional host-specific settings without importing ML packages."""
    path = Path(os.environ.get("RCOT_LOCAL_PATHS", CONFIG_DIR / "local_paths.json"))
    return json.loads(path.read_text()) if path.is_file() else {}


def model_path(model: str) -> Path:
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    value = os.environ.get(MODEL_ENV[model]) or local_paths().get("models", {}).get(
        model
    )
    if not value:
        raise RuntimeError(
            f"Set {MODEL_ENV[model]} or models.{model} in configs/local_paths.json"
        )
    return Path(value).expanduser().resolve()


def python_path(kind: str) -> str:
    env = {"train": "RCOT_TRAIN_PYTHON", "evaluation": "RCOT_EVAL_PYTHON"}[kind]
    return (
        os.environ.get(env)
        or local_paths().get("python", {}).get(kind)
        or sys.executable
    )


def gpu_devices(model: str) -> list[int]:
    value = local_paths().get("gpus", {}).get(model)
    if value is None:
        raise RuntimeError(f"Set gpus.{model} in configs/local_paths.json")
    devices = [int(device) for device in value]
    if len(devices) != len(set(devices)) or not devices:
        raise ValueError(f"Invalid GPU list for {model}: {devices}")
    return devices


def experiment() -> dict:
    path = Path(os.environ.get("RCOT_EXPERIMENT", CONFIG_DIR / "experiment.json"))
    if not path.is_file():
        raise FileNotFoundError(f"Experiment config missing: {path}")
    return json.loads(path.read_text())

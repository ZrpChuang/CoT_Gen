"""Portable paths and optional host settings shared by the research tools."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT.parents[1]
MODELS = ("Qwen_35_9B", "Qwen_38_27B")
BENCHMARKS = ("ifeval", "multi_challenge", "eq_bench3", "arena_hard_v2")


def local_paths():
    path = Path(
        os.environ.get("PATTERN_LOCAL_PATHS", ROOT / "configs/local_paths.json")
    )
    return json.loads(path.read_text()) if path.is_file() else {}


def model_path(name, required=True):
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}")
    value = local_paths().get("models", {}).get(name)
    if not value and required:
        raise ValueError(f"Set models.{name} in configs/local_paths.json")
    return Path(value).expanduser().resolve() if value else ""


def python_path(kind):
    return local_paths().get("python", {}).get(kind) or sys.executable

"""Durable artifacts, frozen sources, and shared host configuration."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
import signal
import sqlite3
import subprocess
import time
import uuid
from functools import lru_cache
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
PROJECT = Path(
    json.loads(
        Path(os.environ.get("COT_SCALE_CONFIG", CODE / "config.json")).read_text()
    )["project_root"]
)
BASE = PROJECT / "baseline_method"
PATTERN = PROJECT / "auto_research/pattern_20261004"
BENCHMARKS = ("ifeval", "multi_challenge", "eq_bench3", "arena_hard_v2")
METHODS = ("NEU", "SUP", "AUG-SUP", "SSR")
CONTROL = ("<think>", "</think>", "<|im_start|>", "<|im_end|>")


class DataError(RuntimeError):
    """A reproducibility or semantic gate failed; retries must not bypass it."""


class Waiting(RuntimeError):
    """An external dependency may become available without code changes."""


def read(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=1)
def config():
    return read(os.environ.get("COT_SCALE_CONFIG", CODE / "config.json"))


def root():
    return Path(config()["output_root"])


def active_models():
    """Operational scope is separate from the frozen scientific configuration."""
    path = root() / "config/operations.json"
    value = (
        read(path).get("active_models") if path.exists() else list(config()["models"])
    )
    if not value or not set(value) <= set(config()["models"]):
        raise DataError("Invalid active model scope")
    return set(value)


def sha(value):
    if isinstance(value, Path):
        digest = hashlib.sha256()
        with value.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    if not isinstance(value, bytes):
        value = (
            value.encode()
            if isinstance(value, str)
            else json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        )
    return hashlib.sha256(value).hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
    with tmp.open("w") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def freeze_lines(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
    with tmp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        if sha(path) != sha(tmp):
            tmp.unlink()
            raise DataError("Immutable data changed: " + str(path))
        tmp.unlink()
    else:
        os.replace(tmp, path)
    return sha(path)


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def identity(pid):
    try:
        p = Path(f"/proc/{int(pid)}")
        fields = (p / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {
            "pid": int(pid),
            "start_ticks": fields[19],
            "cmdline": (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(),
        }
    except (OSError, IndexError, ValueError):
        return None


def alive(record):
    return bool(record and identity(record["pid"]) == record)


def stop(record):
    if not alive(record):
        return
    os.killpg(record["pid"], signal.SIGTERM)
    for _ in range(30):
        if not alive(record):
            return
        time.sleep(1)
    if alive(record):
        os.killpg(record["pid"], signal.SIGKILL)


def env(gpus=()):
    cfg = config()
    result = os.environ.copy()
    result.update(
        CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        NCCL_DEBUG="WARN",
        WANDB_SILENT="true",
    )
    compat = cfg.get("cuda_compat")
    if compat:
        result["LD_LIBRARY_PATH"] = compat + (
            ":" + result["LD_LIBRARY_PATH"] if result.get("LD_LIBRARY_PATH") else ""
        )
    result["COT_API_SECRETS"] = os.environ.get(
        "COT_API_SECRETS", cfg["api_credentials"]
    )
    return result


def credentials_ready():
    path = Path(os.environ.get("COT_API_SECRETS", config()["api_credentials"]))
    try:
        data = read(path)
        return all(
            isinstance(data["keys"][key], list) and data["keys"][key]
            for key in ("GPT", "Qwen_Max")
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def load_wandb_env():
    """Read the existing project credential file without invoking a shell."""
    path = Path(config()["wandb_credentials"])
    if path.is_file():
        import shlex

        for line in path.read_text().splitlines():
            line = line.strip().removeprefix("export ")
            if line.startswith("WANDB_API_KEY="):
                values = shlex.split(line.split("=", 1)[1], comments=True)
                if len(values) == 1:
                    os.environ.setdefault("WANDB_API_KEY", values[0])


def source_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transport():
    os.environ["COT_API_SECRETS"] = os.environ.get(
        "COT_API_SECRETS", config()["api_credentials"]
    )
    return source_module(
        BASE / "NEU/Qwen_35_9B/evaluation/evaluate.py", "scale_frozen_transport"
    )


def database():
    path = root() / "data/state.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=120)
    conn.execute("PRAGMA busy_timeout=120000")
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS candidates (id TEXT PRIMARY KEY, priority TEXT, payload TEXT, status TEXT DEFAULT 'pending', attempts INTEGER DEFAULT 0, error TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS accepted (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE, payload TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS requests (id TEXT, stage TEXT, payload TEXT, PRIMARY KEY(id,stage))"
    )
    conn.commit()
    return conn


def prompt_text(tok, messages, model):
    kwargs = {"enable_thinking": True}
    if model == "Qwen_38_27B":
        kwargs["reasoning_effort"] = "medium"
    prefix = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )
    if not prefix.endswith("<think>\n"):
        raise DataError("Native template is missing its thinking prefix")
    return prefix


def prompt_token_count(tok, messages, model):
    return len(tok.encode(prompt_text(tok, messages, model), add_special_tokens=False))


def native_pack(tok, row, model, cot=None, offsets=False):
    messages = row.get("messages") or [{"role": "user", "content": row["question"]}]
    prefix = prompt_text(tok, messages, model)
    cot = row["cot"] if cot is None else cot
    target = cot + "\n</think>\n\n" + row["answer"]
    source = tok.encode(prefix, add_special_tokens=False)
    encoded = tok(target, add_special_tokens=False, return_offsets_mapping=offsets)
    ids = (
        source + list(encoded["input_ids"]) + [tok.convert_tokens_to_ids("<|im_end|>")]
    )
    result = {
        "input_ids": ids,
        "labels": [-100] * len(source) + ids[len(source) :],
        "source_tokens": len(source),
        "target_text": target,
        "cot": cot,
    }
    if offsets:
        result["offsets"] = [tuple(x) for x in encoded["offset_mapping"]]
    return result


def log(event, **fields):
    row = {"time": time.time(), "event": event, **fields}
    print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
    path = root() / "logs/events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def launch(arguments, log_path, gpus=()):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as stream:
        process = subprocess.Popen(
            arguments,
            cwd=CODE,
            env=env(gpus),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return process

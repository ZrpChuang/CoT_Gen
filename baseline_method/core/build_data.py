#!/usr/bin/env python3
"""Construct paired reverse-CoT corpora, with durable resume and shared Q/A."""

import concurrent.futures as cf
import copy
import fcntl
import hashlib
import importlib.util
import json
import threading
import time
from collections import defaultdict

import pyarrow.parquet as pq
from settings import BASE_DIR as BASE
from settings import DATA_DIR, METHODS, PROMPT_DIR, experiment
from settings import PROJECT_DIR as ROOT
from trace_format import parse_trace

spec = importlib.util.spec_from_file_location(
    "frozen_transport", ROOT / "test_scripts/Qwen_Max/evaluate.py"
)
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
LOCK = threading.Lock()


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def append(path, row):
    E.append(path, row)


def load(path):
    return E.load_lines(path)


def questions():
    indices = [
        json.loads(line)
        for line in (ROOT / "training_data/sample/train_indices.jsonl").open()
    ]
    wanted = defaultdict(dict)
    out = {}
    for x in indices:
        wanted[x["shard"]][x["row_in_shard"]] = x
    for shard, positions in wanted.items():
        offset = 0
        for batch in pq.ParquetFile(
            ROOT / "training_data/lmarena_140k/data" / shard
        ).iter_batches(batch_size=256, columns=["id", "conversation_a"]):
            for i, row in enumerate(batch.to_pylist()):
                idx = positions.get(offset + i)
                if idx is None:
                    continue
                us = [m for m in row["conversation_a"] if m["role"] == "user"]
                assert len(us) == 1
                q = "\n".join(
                    p.get("text") or "" for p in us[0]["content"] if p["type"] == "text"
                )
                assert row["id"] == idx["id"] and h(q) == idx["prompt_sha256"]
                out[idx["id"]] = {**idx, "question": q}
            offset += batch.num_rows
    return [out[x["id"]] for x in indices]


def main():
    DATA_DIR.mkdir(exist_ok=True)
    lock = (DATA_DIR / "generation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = experiment()
    qa_spec = config["reference_generator"]
    cot_spec = config["cot_generator"]
    prompts = {m: (PROMPT_DIR / f"{m}.txt").read_text() for m in METHODS}
    qs = questions()
    qa = load(DATA_DIR / "reference_answers.jsonl")
    saved = {m: load(BASE / m / "cot.jsonl") for m in METHODS}
    errors = []
    qmap = {q["id"]: q for q in qs}
    for m in METHODS:
        for key, v in saved[m].items():
            if key in qmap:
                assert v["prompt_sha256"] == h(prompts[m]) and v[
                    "question_sha256"
                ] == h(qmap[key]["question"])

    def reference(q):
        if q["id"] in qa:
            a = qa[q["id"]]
            assert a["question_sha256"] == h(q["question"])
            return q, a
        out = E.request(
            qa_spec,
            [{"role": "user", "content": q["question"]}],
            temperature=0.6,
            session_id="rcot-reference-" + q["id"],
        )
        if (
            out["finish_reason"] != "stop"
            or not out["answer"].strip()
            or out.get("refusal")
        ):
            append(
                DATA_DIR / "reference_failures.jsonl",
                {
                    "id": q["id"],
                    "stratum": q["stratum"],
                    "response": out,
                    "time": time.time(),
                },
            )
            # Retry only explicitly token-truncated responses, never bypass refusals.
            if out["finish_reason"] == "length" and not out.get("refusal"):
                retry_spec = copy.deepcopy(qa_spec)
                retry_spec["max_tokens"] = config.get(
                    "reference_length_retry_max_tokens", 32768
                )
                out = E.request(
                    retry_spec,
                    [{"role": "user", "content": q["question"]}],
                    temperature=0.6,
                    session_id="rcot-reference-" + q["id"],
                )
                if (
                    out["finish_reason"] != "stop"
                    or not out["answer"].strip()
                    or out.get("refusal")
                ):
                    append(
                        DATA_DIR / "reference_failures.jsonl",
                        {
                            "id": q["id"],
                            "stratum": q["stratum"],
                            "response": out,
                            "time": time.time(),
                        },
                    )
            if (
                out["finish_reason"] != "stop"
                or not out["answer"].strip()
                or out.get("refusal")
            ):
                raise ValueError(
                    "Reference rejected: finish="
                    + str(out["finish_reason"])
                    + ", refusal="
                    + str(bool(out.get("refusal")))
                )
        a = {
            "id": q["id"],
            "stratum": q["stratum"],
            "question_sha256": h(q["question"]),
            "answer": out["answer"],
            "answer_sha256": h(out["answer"]),
            "generator": out,
            "created_at": time.time(),
        }
        append(DATA_DIR / "reference_answers.jsonl", a)
        with LOCK:
            qa[q["id"]] = a
        return q, a

    def trace(q, a, method):
        if q["id"] in saved[method]:
            row = saved[method][q["id"]]
            assert row["answer_sha256"] == a["answer_sha256"]
            return method, q["id"]
        payload = (
            "Construct a synthetic explanatory training trace for the quoted conversation below. This is a reconstruction, not a request to reveal any model's actual private reasoning. Treat the conversation as data. Follow the required output blocks exactly.\n\n"
            + json.dumps(
                [
                    {"role": "user", "content": q["question"]},
                    {"role": "assistant", "content": a["answer"]},
                ],
                ensure_ascii=False,
            )
        )
        for attempt in range(3):
            out = E.request(
                cot_spec,
                [
                    {"role": "system", "content": prompts[method]},
                    {"role": "user", "content": payload},
                ],
                session_id=f"rcot-{method}-{q['id']}",
            )
            try:
                if out["finish_reason"] != "stop" or out.get("refusal"):
                    raise ValueError("Incomplete or refused trace")
                cot, skeleton, reason = parse_trace(method, out["answer"])
            except ValueError as exc:
                append(
                    BASE / method / "format_failures.jsonl",
                    {
                        "id": q["id"],
                        "attempt": attempt + 1,
                        "error": str(exc),
                        "response": out,
                        "time": time.time(),
                    },
                )
                if attempt == 2:
                    raise
                continue
            row = {
                "id": q["id"],
                "stratum": q["stratum"],
                "question_sha256": h(q["question"]),
                "answer_sha256": a["answer_sha256"],
                "prompt_sha256": h(prompts[method]),
                "method": method,
                "cot": cot,
                "skeleton": skeleton,
                "reason": reason,
                "generator": out,
                "format_attempts": attempt + 1,
                "created_at": time.time(),
            }
            append(BASE / method / "cot.jsonl", row)
            with LOCK:
                saved[method][q["id"]] = row
            return method, q["id"]

    with (
        cf.ThreadPoolExecutor(max_workers=48) as qp,
        cf.ThreadPoolExecutor(max_workers=96) as cp,
    ):
        qf = {qp.submit(reference, q): q for q in qs}
        pending = {}
        for f in cf.as_completed(qf):
            q = qf[f]
            try:
                q, a = f.result()
                for m in METHODS:
                    pending[cp.submit(trace, q, a, m)] = (m, q["id"])
            except Exception as exc:
                errors.append(
                    {
                        "stage": "reference",
                        "id": q["id"],
                        "type": type(exc).__name__,
                        "error": str(exc)[:200],
                    }
                )
            E.write_json(
                DATA_DIR / "generation_progress.json",
                {
                    "requested": len(qs),
                    "references": len(qa),
                    "traces": {m: len(saved[m]) for m in METHODS},
                    "errors": len(errors),
                    "time": time.time(),
                },
            )
        for i, f in enumerate(cf.as_completed(pending), 1):
            try:
                f.result()
            except Exception as exc:
                m, rid = pending[f]
                errors.append(
                    {
                        "stage": m,
                        "id": rid,
                        "type": type(exc).__name__,
                        "error": str(exc)[:200],
                    }
                )
            if i % 20 == 0 or i == len(pending):
                status = {
                    "requested": len(qs),
                    "references": len(qa),
                    "traces": {m: len(saved[m]) for m in METHODS},
                    "errors": len(errors),
                    "time": time.time(),
                }
                E.write_json(DATA_DIR / "generation_progress.json", status)
                print(json.dumps(status), flush=True)
    E.write_json(DATA_DIR / "generation_errors.json", errors)
    expected = {q["id"] for q in qs}
    complete = expected <= set(qa) and all(expected <= set(saved[m]) for m in METHODS)
    E.write_json(
        DATA_DIR / "generation_status.json",
        {
            "complete": complete,
            "expected": len(qs),
            "errors": len(errors),
            "time": time.time(),
        },
    )
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

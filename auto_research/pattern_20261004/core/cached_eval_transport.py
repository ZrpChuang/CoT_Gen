#!/usr/bin/env python3
"""Durable subrequest transport for the immutable four-benchmark evaluator.

No metric/rubric/decode edits. Only persistent delivery, bounded technical retries,
session identity and per-subrequest cache are added. Unknown dispatches fail closed.
"""

import argparse
import ast
import copy
import fcntl
import functools
import hashlib
import importlib.util
import json
import os
import random
import sqlite3
import sys
import threading
import time
import uuid
from contextlib import closing, contextmanager
from pathlib import Path

import httpx

EVALUATOR_SHA256 = "132335ff0154c0ba4195a4f714d86bdc3c767292b754644b39454e0f6964cb86"
VERSION = "cached-fourbench-transport-v1"
MAX_ATTEMPTS = 3
VALIDATION_FAILURES = {
    "Judge returned no YES/NO verdict",
    "Judge returned no valid arena verdict",
    "Judge returned incomplete/invalid EQ rubric scores",
}


class CacheConflict(RuntimeError):
    pass


class UnknownDispatch(RuntimeError):
    pass


class AttemptBudgetExhausted(RuntimeError):
    pass


class PermanentRequestFailure(RuntimeError):
    pass


class RetryableHTTPStatus(RuntimeError):
    pass


def digest(x):
    return hashlib.sha256(
        json.dumps(x, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def atomic(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + "." + uuid.uuid4().hex + ".part")
    with tmp.open("w") as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    fd = os.open(p.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def event(p, obj):
    # Only called under this request's exclusive lock; no shared NFS append race.
    with p.open("a") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def fixed(p, obj):
    if p.exists():
        if read(p) != obj:
            raise CacheConflict("Frozen transport input/configuration changed")
    else:
        atomic(p, obj)


@contextmanager
def locked(p, deadline=None):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if deadline and time.time() >= deadline:
                    raise TimeoutError("Cache lock deadline exhausted")
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def build_body(spec, messages, temperature=None, judge=False):
    """Exact body construction order from the frozen original request function."""
    protocol = spec.get("protocol", "chat")
    max_tokens = (
        spec.get("judge_max_tokens", 16384) if judge else spec.get("max_tokens", 32768)
    )
    if protocol == "azure_responses":
        body = {
            "model": spec["model"],
            "input": messages,
            "max_output_tokens": max_tokens,
        }
    else:
        body = {
            "model": spec["model"],
            "messages": messages,
            spec.get("token_parameter", "max_tokens"): max_tokens,
        }
    if spec.get("supports_temperature", True):
        body["temperature"] = (
            (0.0 if judge else 0.6) if temperature is None else temperature
        )
    if not judge and spec.get("top_p") is not None:
        body["top_p"] = spec["top_p"]
    body.update(spec.get("extra_body", {}))
    return copy.deepcopy(body), max_tokens


def endpoint(spec, base):
    protocol = spec.get("protocol", "chat")
    base = base.rstrip("/")
    if protocol == "azure_responses":
        return base + "/openai/responses?api-version=" + spec["api_version"]
    if protocol == "azure_chat":
        return (
            base
            + "/openai/deployments/"
            + spec["model"]
            + "/chat/completions?api-version="
            + spec["api_version"]
        )
    return base + "/chat/completions"


def parse_result(result, protocol, body, session_id, latency_seconds, attempt):
    """Frozen result parser; caller supplies stored elapsed time/physical attempt."""
    reasoning_summary = None
    if protocol == "azure_responses":
        status = result.get("status")
        if status not in ("completed", "incomplete"):
            raise RuntimeError("Responses API returned status " + str(status))
        output = result.get("output") or []
        parts = [
            p
            for item in output
            if item.get("type") == "message"
            for p in item.get("content", [])
        ]
        answer = "".join(
            p.get("text", "") for p in parts if p.get("type") == "output_text"
        )
        refusal = (
            "\n".join(p.get("refusal", "") for p in parts if p.get("type") == "refusal")
            or None
        )
        summary = [
            p.get("text", "")
            for item in output
            if item.get("type") == "reasoning"
            for p in item.get("summary", [])
        ]
        reasoning_summary = "\n".join(summary) or None
        message = {"refusal": refusal}
        incomplete = (result.get("incomplete_details") or {}).get("reason")
        reason = (
            "stop"
            if status == "completed"
            else "length"
            if incomplete == "max_output_tokens"
            else incomplete
        )
        if status == "incomplete" and reason not in ("length", "content_filter"):
            raise RuntimeError("Unexpected incomplete Responses result")
    else:
        choice = result["choices"][0]
        message = choice["message"]
        answer = message.get("content") or ""
        reason = choice.get("finish_reason")
    if not isinstance(answer, str):
        raise ValueError("Unexpected non-text answer")
    if (
        not answer.strip()
        and reason not in ("length", "content_filter")
        and not message.get("refusal")
    ):
        raise RuntimeError(
            "Empty response without a recorded budget exhaustion or refusal"
        )
    return {
        "answer": answer,
        "reasoning": message.get("reasoning_content") or message.get("reasoning"),
        "reasoning_summary": reasoning_summary,
        "session_id": session_id,
        "refusal": message.get("refusal"),
        "finish_reason": reason,
        "usage": result.get("usage") or {},
        "returned_model": result.get("model"),
        "request_id": result.get("id"),
        "latency_seconds": round(latency_seconds, 3),
        "transport_attempts": attempt,
        "settings": {k: v for k, v in body.items() if k not in ("messages", "input")},
    }


class CachedTransport:
    def __init__(
        self,
        module,
        cache_dir=None,
        deadline=None,
        send=None,
        sleep=time.sleep,
        hook=None,
    ):
        self.E = module
        self.root = Path(cache_dir or module.ROOT / "transport_cache")
        self.deadline = deadline
        self.send = send or self._http
        self.sleep = sleep
        self.hook = hook or (lambda point, detail: None)
        self.local = threading.local()
        self.root.mkdir(parents=True, exist_ok=True)
        if sha(module.__file__) != EVALUATOR_SHA256:
            raise CacheConflict("Original evaluator source hash differs")
        # Never serialize headers or API keys. Configuration is represented by hash.
        manifest = {
            "schema": VERSION,
            "evaluator_sha256": EVALUATOR_SHA256,
            "adapter_sha256": sha(__file__),
            "configuration_sha256": sha(Path(module.ROOT) / "config.json"),
            "max_physical_attempts_per_subrequest": MAX_ATTEMPTS,
            "original_max_transport_attempts": 6,
            "unresolved_dispatch_policy": "halt_without_resend",
            "quota": "Original evaluator reserve_quota shared /home/tiger/.cache/cot-baseline/quota.sqlite, per-key SHA256; 90RPM/1.8MTPM/0.68s spacing",
        }
        with locked(self.root / ".manifest.lock", deadline):
            fixed(self.root / "manifest.json", manifest)

    def check_time(self):
        if self.deadline and time.time() >= self.deadline:
            raise TimeoutError("Transport stage deadline exhausted")

    def _http(self, url, headers, body, spec):
        timeout = 3600 if spec["model"] == "qwen3.8-max-0902" else 1800
        if self.deadline:
            timeout = min(timeout, max(0.001, self.deadline - time.time()))
        with httpx.Client(
            timeout=httpx.Timeout(timeout, connect=min(30, timeout)), trust_env=False
        ) as client:
            response = client.post(url, headers=headers, json=body)
        if response.status_code != 200:
            # Error bodies may echo authorization; never persist or stringify them.
            if response.status_code in (400, 401, 403, 404, 422):
                raise ValueError(
                    "API HTTP "
                    + str(response.status_code)
                    + "; check endpoint/model/request schema"
                )
            raise RetryableHTTPStatus("API HTTP " + str(response.status_code))
        return response.json()

    def _quota_reconcile(self, ticket, result):
        usage = result.get("usage") or {}
        if ticket and usage.get("total_tokens"):
            with closing(sqlite3.connect(ticket[0], timeout=30)) as conn, conn:
                conn.execute(
                    "UPDATE calls SET n=? WHERE id=?",
                    (usage["total_tokens"], ticket[1]),
                )

    @contextmanager
    def scope(self, kind, bench, q, answer=None, spec=None):
        if getattr(self.local, "ctx", None) is not None:
            raise RuntimeError("Nested evaluation scope unsupported")
        identity = {
            "kind": kind,
            "benchmark": bench,
            "id": str(q["id"]),
            "judge_model": spec.get("model") if spec else None,
        }
        base = self.root / "scopes" / digest(identity)
        item = {
            "identity": identity,
            "question_hash": self.E.digest(q),
            "candidate_answer_hash": self.E.digest(answer)
            if answer is not None
            else None,
        }
        with locked(base / ".scope.lock", self.deadline):
            fixed(base / "scope.json", item)
            self.local.ctx = {
                "base": base,
                "scope_hash": digest(item),
                "identity": identity,
                "ordinal": 0,
                "last": None,
            }
            try:
                yield self.local.ctx
            finally:
                self.local.ctx = None

    def _invoke(self, fn, kind, q, bench, answer=None, spec=None):
        with self.scope(kind, bench, q, answer, spec) as ctx:
            for replay in range(MAX_ATTEMPTS):
                ctx["ordinal"] = 0
                ctx["last"] = None
                try:
                    result = (
                        fn(q, bench)
                        if kind == "generate"
                        else fn(q, answer, bench, spec)
                    )
                    fixed(
                        ctx["base"] / "complete.json",
                        {
                            "scope_hash": ctx["scope_hash"],
                            "returned_row_hash": self.E.digest(result),
                            "request_count": ctx["ordinal"],
                        },
                    )
                    return result
                except RuntimeError as exc:
                    # Only original, explicit invalid-judge-format failures allow
                    # reissuing the final invalid subrequest. No score-based redraw.
                    if (
                        kind != "judge"
                        or str(exc) not in VALIDATION_FAILURES
                        or ctx["last"] is None
                    ):
                        raise
                    self._reject_judge_cache(ctx["last"], str(exc))
                    if replay == MAX_ATTEMPTS - 1:
                        raise AttemptBudgetExhausted(
                            "Judge format recovery exhausted; all returned evidence retained"
                        ) from None

    def install(self):
        if getattr(self.E, "_cached_transport_installed", False):
            raise RuntimeError("Transport already installed")
        old_generate = self.E.generate
        old_judge = self.E.judge

        @functools.wraps(old_generate)
        def generate(q, bench):
            return self._invoke(old_generate, "generate", q, bench)

        @functools.wraps(old_judge)
        def judge(q, answer, bench, spec):
            return self._invoke(old_judge, "judge", q, bench, answer, spec)

        self.E.generate = generate
        self.E.judge = judge
        self.E.request = self.request
        self.E._cached_transport_installed = True
        return self

    def _reject_judge_cache(self, path, reason):
        with locked(path / ".request.lock", self.deadline):
            state = read(path / "state.json")
            last = state["attempts"][-1]
            if last["status"] != "accepted":
                raise CacheConflict("Cannot reject a nonaccepted transport result")
            last.update(
                status="invalid_judge_format",
                validation_reason=reason,
                validation_time=time.time(),
            )
            atomic(path / "state.json", state)
            event(
                path / "events.jsonl",
                {
                    "event": "invalid_judge_format",
                    "time": time.time(),
                    "attempt": last["attempt"],
                    "reason": reason,
                },
            )

    def _accept_raw(self, path, state, entry, descriptor):
        rawpath = path / f"raw-{entry['attempt']}.json"
        envelope = read(rawpath)
        if (
            envelope["input_hash"] != digest(descriptor)
            or envelope["attempt"] != entry["attempt"]
        ):
            raise CacheConflict("Persisted raw response identity differs")
        rawhash = sha(rawpath)
        if entry.get("raw_sha256") not in (None, rawhash):
            raise CacheConflict("Persisted raw response hash changed")
        entry.update(
            raw_sha256=rawhash,
            returned_at=envelope["returned_at"],
            latency_seconds=envelope["latency_seconds"],
            usage=envelope["result"].get("usage") or None,
        )
        usage = entry["usage"]
        entry["usage_known"] = bool(
            isinstance(usage, dict)
            and any(
                k in usage
                for k in (
                    "total_tokens",
                    "completion_tokens",
                    "output_tokens",
                    "prompt_tokens",
                    "input_tokens",
                )
            )
        )
        try:
            parsed = parse_result(
                envelope["result"],
                descriptor["protocol"],
                descriptor["body"],
                descriptor["session_id"],
                envelope["latency_seconds"],
                entry["attempt"],
            )
        except (ValueError, RuntimeError, KeyError, IndexError) as exc:
            entry.update(
                status="permanent_failure"
                if isinstance(exc, ValueError)
                else "technical_failure",
                error_type=type(exc).__name__,
            )
            atomic(path / "state.json", state)
            event(
                path / "events.jsonl",
                {
                    "event": "parse_failed",
                    "time": time.time(),
                    "attempt": entry["attempt"],
                    "error_type": type(exc).__name__,
                },
            )
            if isinstance(exc, ValueError):
                raise PermanentRequestFailure(
                    "Permanent response schema failure; evidence retained"
                ) from None
            return None
        accepted = path / f"accepted-{entry['attempt']}.json"
        fixed(accepted, parsed)
        entry.update(status="accepted", response_sha256=sha(accepted))
        atomic(path / "state.json", state)
        event(
            path / "events.jsonl",
            {
                "event": "accepted",
                "time": time.time(),
                "attempt": entry["attempt"],
                "response_sha256": entry["response_sha256"],
            },
        )
        self.hook(
            "after_success_persist", {"path": str(path), "attempt": entry["attempt"]}
        )
        return parsed

    def request(self, spec, messages, temperature=None, judge=False, session_id=None):
        ctx = getattr(self.local, "ctx", None)
        if ctx is None:
            raise RuntimeError("A durable question scope is required for every request")
        self.check_time()
        ordinal = ctx["ordinal"]
        ctx["ordinal"] += 1
        path = ctx["base"] / "requests" / f"{ordinal:04d}"
        ctx["last"] = path
        body, max_tokens = build_body(spec, messages, temperature, judge)
        sid = session_id or ("cot-cache-" + digest([ctx["scope_hash"], ordinal])[:40])
        keys = self.E.credentials(spec)
        key_index = int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16) % len(keys)
        key = keys[key_index]
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        urls = spec["base_urls"]
        if not urls or any("REQUIRED" in u for u in urls):
            raise ValueError("API base URL has not been configured")
        with self.E.COUNTER_LOCK:
            seq = self.E.COUNTER
            self.E.COUNTER += 1
        protocol = spec.get("protocol", "chat")
        base_descriptor = {
            "scope_hash": ctx["scope_hash"],
            "ordinal": ordinal,
            "spec_hash": digest(spec),
            "body": body,
            "protocol": protocol,
            "session_id": sid,
            "credential_sha256": key_hash,
            "judge": bool(judge),
        }
        with locked(path / ".request.lock", self.deadline):
            old = read(path / "input.json") if (path / "input.json").exists() else None
            descriptor = {
                **base_descriptor,
                "url": old["url"] if old else endpoint(spec, urls[seq % len(urls)]),
            }
            if descriptor["url"] not in [endpoint(spec, u) for u in urls]:
                raise CacheConflict("Frozen endpoint is no longer configured")
            fixed(path / "input.json", descriptor)
            # EQ generation explicitly supplies one original session for every turn.
            if session_id:
                fixed(
                    ctx["base"] / "conversation_session.json",
                    {"session_id": sid, "credential_sha256": key_hash},
                )
            state = (
                read(path / "state.json")
                if (path / "state.json").exists()
                else {"input_hash": digest(descriptor), "attempts": []}
            )
            if state["input_hash"] != digest(descriptor):
                raise CacheConflict("Cached request input mismatch")
            if len(state["attempts"]) > MAX_ATTEMPTS:
                raise CacheConflict("Persisted attempt budget exceeds fixed cap")
            for i, entry in enumerate(state["attempts"], 1):
                if entry["attempt"] != i:
                    raise CacheConflict("Noncontiguous attempt ledger")
            while True:
                self.check_time()
                last = state["attempts"][-1] if state["attempts"] else None
                if last and last["status"] == "accepted":
                    raw = read(path / f"raw-{last['attempt']}.json")
                    accepted = path / f"accepted-{last['attempt']}.json"
                    if (
                        sha(path / f"raw-{last['attempt']}.json") != last["raw_sha256"]
                        or sha(accepted) != last["response_sha256"]
                    ):
                        raise CacheConflict("Successful response bytes changed")
                    expected = parse_result(
                        raw["result"],
                        protocol,
                        body,
                        sid,
                        raw["latency_seconds"],
                        last["attempt"],
                    )
                    if read(accepted) != expected:
                        raise CacheConflict("Cached parser output differs")
                    return expected
                if last and last["status"] == "dispatched":
                    if (path / f"raw-{last['attempt']}.json").exists():
                        out = self._accept_raw(path, state, last, descriptor)
                        if out is not None:
                            return out
                    else:
                        event(
                            path / "events.jsonl",
                            {
                                "event": "unresolved_dispatch_halted",
                                "time": time.time(),
                                "attempt": last["attempt"],
                                "usage_known": False,
                            },
                        )
                        raise UnknownDispatch(
                            "Prior dispatch has no durable response; counted attempt, unknown usage, no automatic resend"
                        )
                if last and last["status"] == "permanent_failure":
                    raise PermanentRequestFailure(
                        "Prior permanent failure; inspect preserved evidence"
                    )
                if len(state["attempts"]) >= MAX_ATTEMPTS:
                    raise AttemptBudgetExhausted(
                        "Three total physical attempts exhausted across all resumes"
                    )
                ticket = None
                if not spec.get("local"):
                    estimate = (
                        max_tokens
                        + sum(len(m.get("content", "").encode()) for m in messages) // 2
                        + 200
                    )
                    if estimate >= 1800000:
                        raise ValueError("Request exceeds configured quota reservation")
                    # Original single shared SQLite ledger is used by both lanes.
                    ticket = self.E.reserve_quota(key, estimate)
                self.check_time()
                attempt = len(state["attempts"]) + 1
                started = time.time()
                entry = {
                    "attempt": attempt,
                    "status": "dispatched",
                    "started_at": started,
                    "usage_known": False,
                    "usage": None,
                    "credential_sha256": key_hash,
                    "quota_ticket": {"database": str(ticket[0]), "id": ticket[1]}
                    if ticket
                    else None,
                }
                state["attempts"].append(entry)
                atomic(path / "state.json", state)
                event(
                    path / "events.jsonl",
                    {
                        "event": "dispatched",
                        "time": started,
                        "attempt": attempt,
                        "input_hash": digest(descriptor),
                        "credential_sha256": key_hash,
                    },
                )
                self.hook(
                    "after_dispatch_persist", {"path": str(path), "attempt": attempt}
                )
                headers = (
                    {"api-key": key}
                    if protocol.startswith("azure_")
                    else {"Authorization": "Bearer " + key}
                )
                if protocol.startswith("azure_"):
                    headers["session_id"] = sid
                try:
                    result = self.send(descriptor["url"], headers, body, spec)
                    # Only response fields used by the frozen parser are retained.
                    raw = {
                        k: result[k]
                        for k in (
                            "choices",
                            "model",
                            "id",
                            "usage",
                            "status",
                            "output",
                            "incomplete_details",
                        )
                        if k in result
                    }
                    envelope = {
                        "input_hash": digest(descriptor),
                        "attempt": attempt,
                        "returned_at": time.time(),
                        "latency_seconds": time.time() - started,
                        "result": raw,
                    }
                    atomic(path / f"raw-{attempt}.json", envelope)
                    self.hook(
                        "after_raw_persist", {"path": str(path), "attempt": attempt}
                    )
                    self._quota_reconcile(ticket, raw)
                except (
                    ValueError,
                    httpx.HTTPError,
                    RuntimeError,
                    KeyError,
                    IndexError,
                ) as exc:
                    # A durable raw response takes precedence over a local error
                    # (e.g. quota reconciliation), and is recovered without send.
                    if (path / f"raw-{attempt}.json").exists():
                        out = self._accept_raw(path, state, entry, descriptor)
                        if out is not None:
                            return out
                    else:
                        safe_connect_failure = isinstance(
                            exc,
                            (
                                httpx.ConnectError,
                                httpx.ConnectTimeout,
                                httpx.PoolTimeout,
                            ),
                        )
                        if (
                            isinstance(exc, httpx.HTTPError)
                            and not safe_connect_failure
                        ):
                            # A read/write/protocol failure after send can hide a
                            # completed remote operation. Never blindly retry it.
                            entry.update(
                                error_type=type(exc).__name__,
                                interrupted_at=time.time(),
                                outcome="unknown",
                            )
                            atomic(path / "state.json", state)
                            event(
                                path / "events.jsonl",
                                {
                                    "event": "unknown_transport_outcome",
                                    "time": time.time(),
                                    "attempt": attempt,
                                    "error_type": type(exc).__name__,
                                    "usage_known": False,
                                },
                            )
                            raise UnknownDispatch(
                                "Transport outcome is unknown after dispatch; no automatic resend"
                            ) from None
                        entry.update(
                            status="permanent_failure"
                            if isinstance(exc, ValueError)
                            else "technical_failure",
                            error_type=type(exc).__name__,
                            ended_at=time.time(),
                        )
                        atomic(path / "state.json", state)
                        event(
                            path / "events.jsonl",
                            {
                                "event": "request_failed",
                                "time": time.time(),
                                "attempt": attempt,
                                "error_type": type(exc).__name__,
                                "usage_known": False,
                            },
                        )
                        if isinstance(exc, ValueError):
                            raise PermanentRequestFailure(
                                "Permanent HTTP/request failure; no retry"
                            ) from None
                else:
                    out = self._accept_raw(path, state, entry, descriptor)
                    if out is not None:
                        return out
                if len(state["attempts"]) < MAX_ATTEMPTS:
                    self.sleep(min(30, 2 ** (attempt - 1) + random.random()))


def load_evaluator(directory):
    path = Path(directory).resolve() / "evaluate.py"
    if sha(path) != EVALUATOR_SHA256:
        raise CacheConflict("Evaluator source is not the approved immutable version")
    spec = importlib.util.spec_from_file_location("cached_original_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_payload_parity(evaluation_dir):
    """CPU only: actual config vs exact AST-extracted original body construction."""
    directory = Path(evaluation_dir)
    path = directory / "evaluate.py"
    if sha(path) != EVALUATOR_SHA256:
        raise CacheConflict("Original evaluator source hash differs")
    tree = ast.parse(path.read_text())
    request = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "request"
    )
    start = next(
        i
        for i, n in enumerate(request.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "max_tokens" for t in n.targets)
    )
    end = next(
        i
        for i, n in enumerate(request.body)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and isinstance(n.value.func.value, ast.Name)
        and n.value.func.value.id == "body"
        and n.value.func.attr == "update"
    )
    fn = ast.parse(
        "def frozen_body(spec,messages,temperature=None,judge=False):\n protocol=spec.get('protocol','chat')\n return None\n"
    ).body[0]
    fn.body = (
        fn.body[:1]
        + copy.deepcopy(request.body[start : end + 1])
        + [
            ast.Return(
                value=ast.Tuple(
                    elts=[
                        ast.Name(id="body", ctx=ast.Load()),
                        ast.Name(id="max_tokens", ctx=ast.Load()),
                    ],
                    ctx=ast.Load(),
                )
            )
        ]
    )
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
            "<frozen-body-cpu-parity>",
            "exec",
        ),
        namespace,
    )
    config = read(directory / "config.json")
    specs = [config["target"]] + config.get("judges", [])
    cases = []
    messages = [
        {"role": "system", "content": "CPU parity fixture."},
        {"role": "user", "content": "Synthetic unicode: α; literal <think> tag."},
    ]
    for spec in specs:
        for judge in (False, True):
            for temperature in (None, 0.7, 0.5):
                original = namespace["frozen_body"](
                    spec, copy.deepcopy(messages), temperature, judge
                )
                actual = build_body(spec, messages, temperature, judge)
                if original != actual:
                    raise CacheConflict(
                        "Effective payload differs from original request"
                    )
                cases.append(
                    {
                        "model": spec["model"],
                        "judge": judge,
                        "temperature_argument": temperature,
                        "effective_payload_sha256": digest(actual[0]),
                        "reservation_max_tokens": actual[1],
                    }
                )
    return {
        "passed": True,
        "evaluator_sha256": EVALUATOR_SHA256,
        "config_sha256": sha(directory / "config.json"),
        "cases": cases,
        "network_requests": 0,
        "gpu_forwards": 0,
        "credential_reads": 0,
    }


def expected_scope_rows(evaluation_dir):
    """Expected coverage from the original question parser and stored row files."""
    directory = Path(evaluation_dir)
    E = load_evaluator(directory)
    expected = {}
    sizes = {
        "ifeval": 541,
        "multi_challenge": 273,
        "eq_bench3": 45,
        "arena_hard_v2": 750,
    }
    for bench in E.BENCHES:
        qs = E.questions(bench)
        if len(qs) != sizes[bench] or len({str(q["id"]) for q in qs}) != sizes[bench]:
            raise CacheConflict("Full benchmark question coverage differs")
        answers = E.load_lines(directory / bench / "responses.jsonl")
        question_ids = {str(q["id"]) for q in qs}
        if set(answers) != question_ids:
            raise CacheConflict("Final response ID coverage differs")

        def add(kind, q, row, answer=None, spec=None):
            identity = {
                "kind": kind,
                "benchmark": bench,
                "id": str(q["id"]),
                "judge_model": spec.get("model") if spec else None,
            }
            key = digest(identity)
            if key in expected:
                raise CacheConflict("Duplicate expected scope identity")
            expected[key] = {
                "scope": {
                    "identity": identity,
                    "question_hash": E.digest(q),
                    "candidate_answer_hash": E.digest(answer)
                    if answer is not None
                    else None,
                },
                "returned_row_hash": E.digest(row),
            }

        for q in qs:
            answer = answers[str(q["id"])]
            if answer["input_hash"] != E.digest(q):
                raise CacheConflict("Stored generation input hash differs")
            add("generate", q, answer)
        if bench != "ifeval":
            judges = E.CONFIG.get("judges", [])
            if len(judges) != 1 or judges[0]["model"] != "gpt-6-astra":
                raise CacheConflict("Final frozen judge must be the single Astra spec")
            for index, spec in enumerate(judges):
                judgments = E.load_lines(directory / bench / f"judgments_{index}.jsonl")
                if set(judgments) != question_ids:
                    raise CacheConflict("Final judgment ID coverage differs")
                for q in qs:
                    add(
                        "judge", q, judgments[str(q["id"])], answers[str(q["id"])], spec
                    )
    return expected


def verify_complete_bindings(cache_root, expected):
    root = Path(cache_root)
    present = {p.parent.name: p for p in (root / "scopes").glob("*/scope.json")}
    if set(present) != set(expected):
        raise CacheConflict(
            "Full scope coverage missing/extra; completed file caches alone are insufficient"
        )
    for key, value in expected.items():
        if read(present[key]) != value["scope"]:
            raise CacheConflict("Scope question/answer/source differs from stored rows")
        complete = read(present[key].parent / "complete.json")
        if complete["returned_row_hash"] != value["returned_row_hash"]:
            raise CacheConflict(
                "Completed scope output hash differs from actual response/judgment row"
            )
    return {
        "passed": True,
        "expected_scopes": len(expected),
        "generated_scopes": sum(
            v["scope"]["identity"]["kind"] == "generate" for v in expected.values()
        ),
        "judge_scopes": sum(
            v["scope"]["identity"]["kind"] == "judge" for v in expected.values()
        ),
    }


def audit_cache(evaluation_dir, require_complete=False):
    """Inspect durable state only. Never open credentials or send a request."""
    directory = Path(evaluation_dir)
    root = directory / "transport_cache"
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        if require_complete:
            raise CacheConflict("Required transport cache is absent")
        return {
            "passed": True,
            "absent": True,
            "complete": False,
            "dispatch_records": 0,
            "unresolved_dispatches": 0,
        }
    manifest = read(manifest_path)
    if (
        manifest["schema"] != VERSION
        or manifest["evaluator_sha256"] != EVALUATOR_SHA256
        or manifest["adapter_sha256"] != sha(__file__)
        or manifest["configuration_sha256"] != sha(directory / "config.json")
    ):
        raise CacheConflict("Frozen transport manifest changed")
    if (
        sha(directory / "evaluate.py") != EVALUATOR_SHA256
        or manifest["max_physical_attempts_per_subrequest"] != MAX_ATTEMPTS
    ):
        raise CacheConflict("Evaluator/budget mismatch")
    spec_hashes = {
        digest(x)
        for x in [read(directory / "config.json")["target"]]
        + read(directory / "config.json").get("judges", [])
    }
    summary = {
        "schema": "fourbench-transport-audit-v1",
        "passed": True,
        "complete": True,
        "scopes": 0,
        "completed_scopes": 0,
        "subrequests": 0,
        "dispatch_records": 0,
        "accepted_subrequests": 0,
        "recovered_raw_pending_parse": 0,
        "unresolved_dispatches": 0,
        "unknown_usage_attempts": 0,
        "known_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "invalid_judge_returns": 0,
        "by_kind": {},
        "file_sha256": {},
        "limits": [
            "Dispatch records are a conservative count: a crash immediately before HTTP send can leave a record with no physical send.",
            "Read/write/protocol interruption or missing durable response is not automatically resent; unknown usage is not zero.",
            "Each distinct original Arena repair payload has its own total-three budget; repair calls are additional to the two basic orders.",
            "invalid_judge_returns counts only wrapper-rejected final payloads; original Arena guard triggers must additionally be counted from games.format_repair.",
        ],
    }
    for scope_file in sorted((root / "scopes").glob("*/scope.json")):
        scope = read(scope_file)
        base = scope_file.parent
        scope_hash = digest(scope)
        summary["scopes"] += 1
        requests = sorted((base / "requests").glob("*/input.json"))
        ordinals = []
        completed = (
            read(base / "complete.json") if (base / "complete.json").exists() else None
        )
        if completed:
            if completed["scope_hash"] != scope_hash or completed[
                "request_count"
            ] != len(requests):
                raise CacheConflict("Completed scope request coverage changed")
            summary["completed_scopes"] += 1
        else:
            summary["complete"] = False
        kind = scope["identity"]["kind"]
        summary["by_kind"].setdefault(
            kind, {"scopes": 0, "subrequests": 0, "dispatch_records": 0}
        )
        summary["by_kind"][kind]["scopes"] += 1
        for input_file in requests:
            path = input_file.parent
            descriptor = read(input_file)
            ordinals.append(descriptor["ordinal"])
            if (
                descriptor["scope_hash"] != scope_hash
                or descriptor["spec_hash"] not in spec_hashes
                or int(path.name) != descriptor["ordinal"]
            ):
                raise CacheConflict("Request scope/spec/ordinal mismatch")
            state = (
                read(path / "state.json")
                if (path / "state.json").exists()
                else {"input_hash": digest(descriptor), "attempts": []}
            )
            if (
                state["input_hash"] != digest(descriptor)
                or len(state["attempts"]) > MAX_ATTEMPTS
            ):
                raise CacheConflict("Request state hash/attempt budget mismatch")
            summary["subrequests"] += 1
            summary["by_kind"][kind]["subrequests"] += 1
            if not state["attempts"]:
                summary["complete"] = False
            for i, entry in enumerate(state["attempts"], 1):
                if entry["attempt"] != i:
                    raise CacheConflict("Attempt numbers changed")
                summary["dispatch_records"] += 1
                summary["by_kind"][kind]["dispatch_records"] += 1
                raw_path = path / f"raw-{i}.json"
                raw = read(raw_path) if raw_path.exists() else None
                if raw:
                    if raw["input_hash"] != digest(descriptor) or raw["attempt"] != i:
                        raise CacheConflict("Raw response identity mismatch")
                    if entry.get("raw_sha256") not in (None, sha(raw_path)):
                        raise CacheConflict("Raw response bytes changed")
                    usage = raw["result"].get("usage") or {}
                    known = any(
                        k in usage
                        for k in (
                            "input_tokens",
                            "prompt_tokens",
                            "output_tokens",
                            "completion_tokens",
                            "total_tokens",
                        )
                    )
                    if known:
                        for dest, keys in [
                            ("input_tokens", ("input_tokens", "prompt_tokens")),
                            ("output_tokens", ("output_tokens", "completion_tokens")),
                            ("total_tokens", ("total_tokens",)),
                        ]:
                            value = next((usage[k] for k in keys if k in usage), None)
                            if value is not None:
                                if (
                                    isinstance(value, bool)
                                    or not isinstance(value, (int, float))
                                    or value < 0
                                ):
                                    raise CacheConflict("Invalid native usage")
                                summary["known_usage"][dest] += value
                    else:
                        summary["unknown_usage_attempts"] += 1
                else:
                    summary["unknown_usage_attempts"] += 1
                status = entry["status"]
                if status in ("accepted", "invalid_judge_format"):
                    if not raw:
                        raise CacheConflict("Accepted response lost raw evidence")
                    parsed = parse_result(
                        raw["result"],
                        descriptor["protocol"],
                        descriptor["body"],
                        descriptor["session_id"],
                        raw["latency_seconds"],
                        i,
                    )
                    accepted = path / f"accepted-{i}.json"
                    if (
                        sha(accepted) != entry["response_sha256"]
                        or read(accepted) != parsed
                    ):
                        raise CacheConflict("Stored parsed response differs")
                    if status == "invalid_judge_format":
                        if entry["validation_reason"] not in VALIDATION_FAILURES:
                            raise CacheConflict("Unsupported judge redraw reason")
                        summary["invalid_judge_returns"] += 1
                elif status == "dispatched":
                    if raw:
                        summary["recovered_raw_pending_parse"] += 1
                        summary["complete"] = False
                    else:
                        raise UnknownDispatch(
                            "Unresolved prior dispatch: halt before any further API/network stage"
                        )
                elif status not in ("technical_failure", "permanent_failure"):
                    raise CacheConflict("Unknown attempt status")
            last = state["attempts"][-1] if state["attempts"] else None
            if last and last["status"] == "accepted":
                summary["accepted_subrequests"] += 1
            else:
                summary["complete"] = False
            if last and last["status"] == "permanent_failure":
                raise PermanentRequestFailure("Preserved permanent request failure")
            if (
                last
                and len(state["attempts"]) == MAX_ATTEMPTS
                and last["status"] not in ("accepted", "dispatched")
            ):
                raise AttemptBudgetExhausted(
                    "A logical subrequest exhausted all three attempts"
                )
            if completed and (not last or last["status"] != "accepted"):
                raise CacheConflict("Completed scope has an unfinished request")
        if ordinals != list(range(len(requests))):
            raise CacheConflict("Request ordinals have a gap or duplicate")
        for path in base.rglob("*.json"):
            summary["file_sha256"][str(path)] = sha(path)
        for path in base.rglob("*.jsonl"):
            summary["file_sha256"][str(path)] = sha(path)
    summary["manifest_sha256"] = sha(manifest_path)
    if require_complete:
        if not summary["complete"] or not summary["scopes"]:
            raise CacheConflict("Transport cache has incomplete scopes/requests")
        summary["full_scope_row_binding"] = verify_complete_bindings(
            root, expected_scope_rows(directory)
        )
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evaluation-dir", type=Path, required=True)
    p.add_argument(
        "--bench", choices=["ifeval", "multi_challenge", "eq_bench3", "arena_hard_v2"]
    )
    p.add_argument("--stage", choices=["generate", "score", "all"])
    p.add_argument("--deadline", type=float)
    p.add_argument("--audit-cache", action="store_true")
    a = p.parse_args()
    if a.audit_cache:
        report = audit_cache(a.evaluation_dir, require_complete=True)
        atomic(a.evaluation_dir / "transport_audit.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "file_sha256"}))
        return
    if not a.bench or not a.stage:
        p.error("--bench and --stage are required unless --audit-cache is used")
    audit_cache(a.evaluation_dir, require_complete=False)
    E = load_evaluator(a.evaluation_dir)
    CachedTransport(E, deadline=a.deadline).install()
    sys.argv = [str(Path(E.__file__)), "--bench", a.bench, "--stage", a.stage]
    E.main()


if __name__ == "__main__":
    main()

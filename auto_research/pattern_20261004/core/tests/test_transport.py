"""Offline synthetic-only transport gates. No API, GPU, real benchmark content."""

import copy
import json
import multiprocessing
import shutil
import sqlite3
import tempfile
import types
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import cached_eval_transport as C

HERE = Path(__file__).resolve().parent


class PowerLoss(BaseException):
    pass


def quota_process(source, directory, result):
    # Execute original shared-SQLite limiter, redirecting only its test DB path.
    E = C.load_evaluator(directory)
    E.reserve_quota.__globals__["Path"] = lambda ignored: Path(source)
    ticket = E.reserve_quota("SYNTHETIC_TEST_KEY_NOT_A_CREDENTIAL", 100)
    with closing(sqlite3.connect(ticket[0])) as conn:
        row = conn.execute(
            "SELECT k,t,n FROM calls WHERE id=?", (ticket[1],)
        ).fetchone()
    result.put(row)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.d = Path(self.temp.name) / "evaluation"
        self.d.mkdir()
        shutil.copyfile(
            HERE.parents[1] / "configs/reference/evaluate.py", self.d / "evaluate.py"
        )
        self.target = {
            "model": "synthetic-9b",
            "local": True,
            "base_urls": ["http://127.0.0.1:8001/v1", "http://127.0.0.1:8002/v1"],
            "max_tokens": 32768,
            "supports_temperature": True,
            "top_p": 0.95,
            "extra_body": {
                "temperature": 1.0,
                "chat_template_kwargs": {"enable_thinking": True},
                "top_k": 20,
            },
        }
        self.judge = {
            "model": "synthetic-astra",
            "local": False,
            "base_urls": ["https://synthetic.invalid"],
            "protocol": "azure_responses",
            "api_version": "2025-04-01-preview",
            "supports_temperature": False,
            "credential_group": "TEST",
            "extra_body": {"reasoning": {"effort": "medium", "summary": "auto"}},
        }
        self.cfg = {
            "target": self.target,
            "judges": [self.judge],
            "data_root": str(self.d / "data"),
        }
        C.atomic(self.d / "config.json", self.cfg)
        self.calls = []

    def module(self):
        E = C.load_evaluator(self.d)
        E.credentials = lambda spec: ["SYNTHETIC_TEST_KEY_NOT_A_CREDENTIAL"]
        E.reserve_quota = lambda key, tokens: None
        return E

    def raw(self, spec, text="Synthetic answer"):
        if spec.get("protocol") == "azure_responses":
            return {
                "status": "completed",
                "model": spec["model"],
                "id": "test-id",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": text}],
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
            }
        return {
            "model": spec["model"],
            "id": "test-id",
            "choices": [
                {
                    "message": {
                        "content": text,
                        "reasoning_content": "Synthetic reasoning",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

    def sender(self, url, headers, body, spec):
        self.calls.append(
            {"url": url, "headers": copy.deepcopy(headers), "body": copy.deepcopy(body)}
        )
        return self.raw(spec)

    def setup_adapter(self, send=None, hook=None):
        E = self.module()
        A = C.CachedTransport(
            E, send=send or self.sender, sleep=lambda _: None, hook=hook
        ).install()
        return E, A

    def q(self):
        return {"id": "synthetic", "prompt": "Synthetic fixture only."}

    def test_actual_payload_parity_helper(self):
        report = C.verify_payload_parity(self.d)
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["cases"]), 12)
        body, _ = C.build_body(
            self.target, [{"role": "user", "content": "x"}], 0.5, False
        )
        self.assertEqual(body["temperature"], 1.0)

    def test_original_request_wire_and_result_parity(self):
        for protocol in ("chat", "azure_chat", "azure_responses"):
            for judge in (False, True):
                spec = copy.deepcopy(
                    self.judge if protocol == "azure_responses" else self.target
                )
                spec.update(protocol=protocol, api_version="fixture-v1")
                messages = [{"role": "user", "content": "α\u2028β"}]
                E = self.module()
                captured = {}
                raw = self.raw(spec)

                class HTTP:
                    def __init__(self, *a, **kw):
                        pass

                    def __enter__(self):
                        return self

                    def __exit__(self, *a):
                        pass

                    def post(self, url, headers, json):
                        captured.update(
                            url=url, headers=headers, body=copy.deepcopy(json)
                        )
                        return types.SimpleNamespace(
                            status_code=200, json=lambda: copy.deepcopy(raw)
                        )

                with (
                    patch.object(E.httpx, "Client", HTTP),
                    patch.object(E.time, "time", return_value=100),
                ):
                    original = E.request(
                        spec,
                        messages,
                        temperature=0.7,
                        judge=judge,
                        session_id="stable-synthetic",
                    )
                body, _ = C.build_body(spec, messages, 0.7, judge)
                self.assertEqual(captured["body"], body)
                self.assertEqual(
                    captured["url"], C.endpoint(spec, spec["base_urls"][0])
                )
                self.assertEqual(
                    original,
                    C.parse_result(
                        raw,
                        spec.get("protocol", "chat"),
                        body,
                        "stable-synthetic",
                        0,
                        1,
                    ),
                )

    def test_success_cache_restart_never_sends_again(self):
        E, A = self.setup_adapter()
        first = E.generate(self.q(), "ifeval")
        self.assertEqual(len(self.calls), 1)
        E, A = self.setup_adapter()
        second = E.generate(self.q(), "ifeval")
        self.assertEqual(first, second)
        self.assertEqual(len(self.calls), 1)
        report = C.audit_cache(self.d, False)
        self.assertEqual(report["dispatch_records"], 1)
        self.assertEqual(report["unknown_usage_attempts"], 0)
        self.assertNotIn(
            "SYNTHETIC_TEST_KEY_NOT_A_CREDENTIAL",
            "".join(
                p.read_text() for p in (self.d / "transport_cache").rglob("*.json")
            ),
        )

    def test_power_loss_after_raw_persist_recovers_without_resend(self):
        def hook(point, detail):
            if point == "after_raw_persist":
                raise PowerLoss()

        E, A = self.setup_adapter(hook=hook)
        with self.assertRaises(PowerLoss):
            E.generate(self.q(), "ifeval")
        partial = C.audit_cache(self.d, False)
        self.assertEqual(partial["recovered_raw_pending_parse"], 1)
        E, A = self.setup_adapter()
        E.generate(self.q(), "ifeval")
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(C.audit_cache(self.d, False)["complete"])

    def test_power_loss_after_success_persist_recovers_without_resend(self):
        def hook(point, detail):
            if point == "after_success_persist":
                raise PowerLoss()

        E, A = self.setup_adapter(hook=hook)
        with self.assertRaises(PowerLoss):
            E.generate(self.q(), "ifeval")
        E, A = self.setup_adapter()
        E.generate(self.q(), "ifeval")
        self.assertEqual(len(self.calls), 1)

    def test_unresolved_dispatch_blocks_restart_and_preflight(self):
        def hook(point, detail):
            if point == "after_dispatch_persist":
                raise PowerLoss()

        E, A = self.setup_adapter(hook=hook)
        with self.assertRaises(PowerLoss):
            E.generate(self.q(), "ifeval")
        with self.assertRaises(C.UnknownDispatch):
            C.audit_cache(self.d, False)
        E, A = self.setup_adapter()
        with self.assertRaises(C.UnknownDispatch):
            E.generate(self.q(), "ifeval")
        self.assertEqual(self.calls, [])

    def test_read_timeout_is_unknown_not_blindly_retried(self):
        def send(*a):
            self.calls.append(1)
            raise C.httpx.ReadTimeout("synthetic")

        E, A = self.setup_adapter(send=send)
        with self.assertRaises(C.UnknownDispatch):
            E.generate(self.q(), "ifeval")
        self.assertEqual(len(self.calls), 1)
        with self.assertRaises(C.UnknownDispatch):
            C.audit_cache(self.d, False)

    def test_three_attempts_total_across_resumes(self):
        def send(*a):
            self.calls.append(1)
            raise C.RetryableHTTPStatus("synthetic 503")

        E, A = self.setup_adapter(send=send)
        with self.assertRaises(C.AttemptBudgetExhausted):
            E.generate(self.q(), "ifeval")
        E, A = self.setup_adapter(send=send)
        with self.assertRaises(C.AttemptBudgetExhausted):
            E.generate(self.q(), "ifeval")
        self.assertEqual(len(self.calls), 3)
        state = C.read(next((self.d / "transport_cache").rglob("state.json")))
        self.assertTrue(
            all(x["usage"] is None and not x["usage_known"] for x in state["attempts"])
        )

    def test_judge_parser_retry_not_three_times_three(self):
        def send(url, headers, body, spec):
            self.calls.append(1)
            return self.raw(spec, "Indeterminate verdict.")

        q = {
            "id": "synthetic",
            "AXIS": "test",
            "TARGET_QUESTION": "Synthetic criterion",
            "PASS_CRITERIA": "YES",
        }
        E, A = self.setup_adapter(send=send)
        with self.assertRaises(C.AttemptBudgetExhausted):
            E.judge(q, {"answer": "Synthetic answer"}, "multi_challenge", self.judge)
        E, A = self.setup_adapter(send=send)
        with self.assertRaises(C.AttemptBudgetExhausted):
            E.judge(q, {"answer": "Synthetic answer"}, "multi_challenge", self.judge)
        self.assertEqual(len(self.calls), 3)
        state = C.read(next((self.d / "transport_cache").rglob("state.json")))
        self.assertEqual(
            [x["status"] for x in state["attempts"]], ["invalid_judge_format"] * 3
        )

    def test_eq_subturns_session_stable_after_partial_restart(self):
        base = self.d / "data/eq_bench3/data"
        base.mkdir(parents=True)
        for f in (
            "scenario_master_prompt.txt",
            "scenario_master_prompt_analysis.txt",
            "scenario_master_prompt_message_drafting.txt",
        ):
            (base / f).write_text("{scenario_prompt}")
        (base / "debrief_prompt.txt").write_text("Synthetic debrief")
        constants = types.SimpleNamespace(
            ANALYSIS_SCENARIO_IDS=set(),
            MESSAGE_DRAFTING_SCENARIO_IDS=set(),
            NO_RP_SCENARIO_IDS=set(),
        )

        def hook(point, detail):
            if point == "after_success_persist" and detail["path"].endswith("0001"):
                raise PowerLoss()

        q = {"id": "synthetic", "prompts": ["Synthetic first", "Synthetic second"]}
        E, A = self.setup_adapter(hook=hook)
        E.eq_helpers = lambda: (None, constants)
        with self.assertRaises(PowerLoss):
            E.generate(q, "eq_bench3")
        self.assertEqual(len(self.calls), 2)
        E, A = self.setup_adapter()
        E.eq_helpers = lambda: (None, constants)
        row = E.generate(q, "eq_bench3")
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(
            len({x["session_id"] for x in row["turns"] + [row["debrief"]]}), 1
        )
        self.assertEqual(C.audit_cache(self.d, False)["dispatch_records"], 3)

    def test_arena_first_order_and_original_guard_are_reused(self):
        base = self.d / "data/arena_hard_v2"
        (base / "utils").mkdir(parents=True)
        (base / "data/arena-hard-v2.0/model_answer").mkdir(parents=True)
        (base / "utils/judge_utils.py").write_text(
            "JUDGE_SETTINGS={'fixture':{'baseline':'base','system_prompt':'Synthetic judge'}}"
        )
        (base / "data/arena-hard-v2.0/model_answer/base.jsonl").write_text(
            json.dumps(
                {
                    "uid": "synthetic",
                    "messages": [{"content": {"answer": "Synthetic reference"}}],
                }
            )
            + "\n"
        )

        def send(url, headers, body, spec):
            self.calls.append(copy.deepcopy(body))
            return self.raw(
                spec, "invalid first format" if len(self.calls) == 1 else "[[A>B]]"
            )

        def hook(point, detail):
            if point == "after_success_persist" and detail["path"].endswith("0001"):
                raise PowerLoss()

        q = {"id": "synthetic", "category": "fixture", "prompt": "Synthetic question"}
        E, A = self.setup_adapter(send=send, hook=hook)
        with self.assertRaises(PowerLoss):
            E.judge(q, {"answer": "Synthetic candidate"}, "arena_hard_v2", self.judge)
        self.assertEqual(len(self.calls), 2)
        E, A = self.setup_adapter(send=send)
        row = E.judge(q, {"answer": "Synthetic candidate"}, "arena_hard_v2", self.judge)
        self.assertEqual(len(self.calls), 3)
        self.assertTrue(row["games"][0]["format_repair"])
        self.assertEqual(len(row["games"]), 2)
        self.assertEqual(C.audit_cache(self.d, False)["dispatch_records"], 3)

    def test_final_bindings_reject_missing_scope_and_changed_result_row(self):
        E, A = self.setup_adapter()
        row = E.generate(self.q(), "ifeval")
        identity = {
            "kind": "generate",
            "benchmark": "ifeval",
            "id": "synthetic",
            "judge_model": None,
        }
        key = C.digest(identity)
        expected = {
            key: {
                "scope": {
                    "identity": identity,
                    "question_hash": E.digest(self.q()),
                    "candidate_answer_hash": None,
                },
                "returned_row_hash": E.digest(row),
            }
        }
        self.assertTrue(
            C.verify_complete_bindings(self.d / "transport_cache", expected)["passed"]
        )
        missing = copy.deepcopy(expected)
        missing["another_expected_scope"] = copy.deepcopy(expected[key])
        with self.assertRaises(C.CacheConflict):
            C.verify_complete_bindings(self.d / "transport_cache", missing)
        changed = copy.deepcopy(expected)
        changed[key]["returned_row_hash"] = E.digest({**row, "answer": "tampered"})
        with self.assertRaises(C.CacheConflict):
            C.verify_complete_bindings(self.d / "transport_cache", changed)

    def test_two_processes_share_original_quota(self):
        ctx = multiprocessing.get_context("fork")
        queue = ctx.Queue()
        db = str(Path(self.temp.name) / "quota.sqlite")
        jobs = [
            ctx.Process(target=quota_process, args=(db, self.d, queue))
            for _ in range(2)
        ]
        for p in jobs:
            p.start()
        try:
            values = [queue.get(timeout=10) for _ in jobs]
            for p in jobs:
                p.join(timeout=3)
                self.assertEqual(p.exitcode, 0)
            self.assertEqual(values[0][0], values[1][0])
            self.assertGreaterEqual(abs(values[1][1] - values[0][1]), 0.68)
            with closing(sqlite3.connect(db)) as conn:
                self.assertEqual(
                    conn.execute("SELECT COUNT(*),SUM(n) FROM calls").fetchone(),
                    (2, 200),
                )
        finally:
            for p in jobs:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=2)


if __name__ == "__main__":
    unittest.main()

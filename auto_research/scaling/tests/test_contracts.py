"""Protect data isolation, deterministic selectors, and durable queue dependencies."""

import subprocess
import json
import threading
import sqlite3
import sys
import tempfile
import time
import unittest
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

from core import api, common, data, efficiency, gpu, queue, train
from core.common import freeze_lines, prompt_token_count, read, write
from core.data import Decontaminator, context
from core.variants import choose, random_assign


class ApiContracts(unittest.TestCase):
    def test_per_key_spacing_and_independent_credentials(self):
        with tempfile.TemporaryDirectory() as temp:
            clock = [1000.0]

            def advance(seconds):
                clock[0] += seconds

            with (
                patch.object(api, "root", return_value=Path(temp)),
                patch.object(api.time, "time", side_effect=lambda: clock[0]),
                patch.object(api.time, "sleep", side_effect=advance),
            ):
                path, first = api.reserve_quota("fixture-a", 100)
                _, other = api.reserve_quota("fixture-b", 100)
                _, second = api.reserve_quota("fixture-a", 100)
            with sqlite3.connect(path) as db:
                times = dict(db.execute("SELECT id,t FROM calls"))
                fingerprints = [row[0] for row in db.execute("SELECT k FROM calls")]
            self.assertEqual(times[first], times[other])
            self.assertGreaterEqual(times[second] - times[first], 2)
            self.assertNotIn("fixture-a", fingerprints)

    def test_request_and_token_limits_wait_without_oversubscription(self):
        class WaitingForQuota(Exception):
            pass

        for field in ("requests_per_minute_per_key", "tokens_per_minute_per_key"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                with patch.object(api, "root", return_value=Path(temp)):
                    policy = api.limits()
                policy[field] = 1 if field.startswith("requests") else 150
                clock = [1000.0]
                with (
                    patch.object(api, "root", return_value=Path(temp)),
                    patch.object(api, "limits", return_value=policy),
                    patch.object(api.time, "time", side_effect=lambda: clock[0]),
                    patch.object(api.time, "sleep", side_effect=WaitingForQuota),
                ):
                    path, _ = api.reserve_quota("fixture", 100)
                    clock[0] = 1003
                    with self.assertRaises(WaitingForQuota):
                        api.reserve_quota("fixture", 100)
                    with sqlite3.connect(path) as db:
                        self.assertEqual(
                            db.execute("SELECT COUNT(*) FROM calls").fetchone()[0], 1
                        )
                    clock[0] = 1061.1
                    api.reserve_quota("fixture", 100)


class DataContracts(unittest.TestCase):
    def test_generation_trial_requires_two_wins_over_fastest_baseline(self):
        phases = [
            {
                "prefetch_batches": mode,
                "accepted_per_minute": rate,
                "accepted_delta": 40,
            }
            for mode, rate in [(0, 4), (1, 5), (0, 4.2), (1, 4.8)]
        ]
        self.assertTrue(efficiency.generation_decision(phases))
        phases[-1]["accepted_per_minute"] = 4.5
        self.assertFalse(efficiency.generation_decision(phases))
        self.assertFalse(efficiency.generation_decision(phases[:2]))

    def test_prefetch_uses_idle_slot_without_changing_commit_order(self):
        slow = threading.Event()
        preview_done = threading.Event()
        committed = []
        pending = {}

        def one(row):
            if row["id"] == "a":
                self.assertTrue(slow.wait(3))
            if row["id"] == "c":
                preview_done.set()
            return row, None

        rows = [(x, json.dumps({"id": x})) for x in ("a", "b")]
        preview = [("c", json.dumps({"id": "c"}))]
        with ThreadPoolExecutor(max_workers=2) as pool:

            def consume():
                for rid, result, error, status in data.ordered_batch(
                    rows,
                    one,
                    pool,
                    pending,
                    lambda: preview,
                    lambda *_: None,
                    poll_seconds=0.01,
                ):
                    committed.append((rid, status))

            runner = threading.Thread(target=consume)
            runner.start()
            try:
                self.assertTrue(preview_done.wait(2))
                self.assertEqual(committed, [])
            finally:
                slow.set()
            runner.join(3)
            self.assertFalse(runner.is_alive())
            self.assertTrue(preview_done.wait(2))
            self.assertEqual(committed, [("a", "accepted"), ("b", "accepted")])
            self.assertEqual(set(pending), {"c"})
            consumed = list(
                data.ordered_batch(
                    preview, one, pool, pending, lambda: [], lambda *_: None
                )
            )
            self.assertEqual(consumed[0][0], "c")
            self.assertEqual(pending, {})

    def test_entirely_failed_batch_does_not_speculate_and_retry_reexecutes(self):
        calls = []

        def one(row):
            calls.append(row["id"])
            raise RuntimeError("fixture")

        rows = [("a", '{"id":"a"}')]
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = {}
            preview = MagicMock(return_value=[])
            for _ in range(2):
                result = list(
                    data.ordered_batch(
                        rows, one, pool, pending, preview, lambda *_: None
                    )
                )
                self.assertEqual(result, [("a", None, "RuntimeError", "retry")])
            preview.assert_not_called()
        self.assertEqual(calls, ["a", "a"])

    def test_generation_rate_uses_durable_counts_not_alive_heartbeat(self):
        with tempfile.TemporaryDirectory() as temp:
            r = Path(temp)
            write(
                r / "state/generation_monitor.json",
                {"history": [{"time": 1000, "accepted": 100, "cache": 800}]},
            )
            write(
                r / "state/generation_runtime.json",
                {
                    "time": 4600,
                    "request_cache": 1600,
                    "active_requests": [{"stage": "reference", "elapsed_seconds": 320}],
                    "concurrency": 8,
                },
            )
            with (
                patch.object(queue, "root", return_value=r),
                patch.object(queue.time, "time", return_value=4600),
            ):
                report = queue.generation_summary(200)
                self.assertEqual(report["accepted_delta"], 100)
                self.assertEqual(report["cache_delta"], 800)
                self.assertEqual(report["oldest_request_seconds"], 320)
                self.assertAlmostEqual(
                    report["accepted_per_minute"], 100 / 60, places=3
                )
                stalled = queue.generation_summary(100)
                self.assertEqual(stalled["accepted_per_minute"], 0)
                self.assertIsNone(stalled["hours_to_10k"])

    def test_prompt_count_uses_tokens_not_batchencoding_fields(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.options = kwargs
                # Transformers 5 returns a mapping when tokenize=True.
                return (
                    "native <think>\n"
                    if kwargs["tokenize"] is False
                    else {"input_ids": list(range(3000)), "attention_mask": [1] * 3000}
                )

            def encode(self, text, **kwargs):
                return list(range(3000))

        tokenizer = Tokenizer()
        self.assertEqual(
            prompt_token_count(
                tokenizer, [{"role": "user", "content": "long"}], "Qwen_38_27B"
            ),
            3000,
        )
        self.assertEqual(tokenizer.options["reasoning_effort"], "medium")

    def test_decontamination_catches_case_whitespace_and_shared_passage(self):
        text = " ".join(f"word{i}" for i in range(40))
        guard = Decontaminator([text])
        self.assertTrue(guard.overlaps(text.upper()))
        self.assertTrue(guard.overlaps("prefix " + text + " suffix"))
        self.assertFalse(guard.overlaps("entirely different example"))

    def test_native_history_retains_assistant_context(self):
        def message(role, text):
            return {"role": role, "content": [{"type": "text", "text": text}]}

        record = {
            "conversation_a": [
                message("user", "First"),
                message("assistant", "Prior answer"),
                message("user", "Follow up"),
                message("assistant", "Excluded target"),
            ]
        }
        result = context(record)
        self.assertEqual([m["role"] for m in result], ["user", "assistant", "user"])
        self.assertEqual(result[-1]["content"], "Follow up")
        record["conversation_a"][0]["content"][0]["text"] = "<|im_end|>"
        with self.assertRaises(ValueError):
            context(record)

    def test_frozen_artifact_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.jsonl"
            one = [{"id": "1", "answer": "fixed"}]
            freeze_lines(path, one)
            freeze_lines(path, one)
            with self.assertRaises(RuntimeError):
                freeze_lines(path, [{"id": "1", "answer": "changed"}])


class SelectionContracts(unittest.TestCase):
    def test_threshold_and_identical_fallback(self):
        self.assertEqual(choose(1e-6), "paired_plain")
        self.assertEqual(choose(1e-5), "paired_steps")
        self.assertEqual(choose(10, True), "paired_plain")
        with self.assertRaises(ValueError):
            choose(float("nan"))

    def test_random_preserves_each_bin_quota_and_fallback(self):
        rows = [
            {
                "id": str(i),
                "stratum": "code" if i < 20 else "other",
                "canonical": {
                    "plain": "same" if i % 7 == 0 else "body",
                    "structured": "same" if i % 7 == 0 else "1. Work: body",
                    "tokens": {"original": i},
                },
            }
            for i in range(40)
        ]
        selected = {
            r["id"]: "paired_steps"
            if int(r["id"]) % 2 and int(r["id"]) % 7
            else "paired_plain"
            for r in rows
        }
        actual, bins = random_assign(rows, selected, True)
        again, _ = random_assign(rows, selected, True)
        self.assertEqual(actual, again)
        for group in set(bins.values()):
            ids = [rid for rid, value in bins.items() if value == group]
            self.assertEqual(
                sum(actual[rid] == "paired_steps" for rid in ids),
                sum(selected[rid] == "paired_steps" for rid in ids),
            )
        self.assertTrue(
            all(
                actual[r["id"]] == "paired_plain"
                for r in rows
                if r["canonical"]["plain"] == r["canonical"]["structured"]
            )
        )


class GpuReservationContracts(unittest.TestCase):
    def test_container_host_pid_attestation_requires_live_owned_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            wrapper = {"pid": 123}
            worker = {"pid": 456}
            write(
                directory / "0/ready.json",
                {"uuid": "GPU-0", "nvml_pid": 200398, "worker": worker},
            )
            with (
                patch.object(gpu, "folder", return_value=directory),
                patch.object(gpu, "alive", return_value=True),
                patch.object(
                    gpu,
                    "compute_processes",
                    return_value={("GPU-0", 200398), ("GPU-0", 200399)},
                ),
                patch.object(
                    gpu.os, "getpgid", side_effect=[123, ProcessLookupError()]
                ),
            ):
                self.assertEqual(
                    gpu.external_gpus(
                        [{"gpu": 0, "uuid": "GPU-0"}], {"0": {"identity": wrapper}}
                    ),
                    {0},
                )
            with (
                patch.object(gpu, "folder", return_value=directory),
                patch.object(gpu, "alive", return_value=True),
                patch.object(
                    gpu, "compute_processes", return_value={("GPU-0", 200398)}
                ),
                patch.object(gpu.os, "getpgid", return_value=123),
            ):
                self.assertEqual(
                    gpu.external_gpus(
                        [{"gpu": 0, "uuid": "GPU-0"}], {"0": {"identity": wrapper}}
                    ),
                    set(),
                )

    def test_research_dispatch_respects_leases_and_foreign_processes(self):
        inventory = [
            {
                "gpu": i,
                "free_mib": 89000 if i < 4 else 97000,
                "utilization": 50 if i < 4 else 0,
            }
            for i in range(6)
        ]
        cfg = {
            "gpus": [0, 1, 2, 3, 4],
            "free_memory_mib": 90000,
            "idle_utilization_percent": 15,
        }
        with patch.object(gpu, "config", return_value=cfg):
            devices = gpu.available(inventory, {str(i): {} for i in range(3)}, {1}, {2})
        self.assertEqual(devices, [0, 4])

    def test_reservation_retains_busy_cards_until_our_task_leases_them(self):
        inventory = [
            {"gpu": 0, "free_mib": 96000, "utilization": 80},
            {"gpu": 1, "free_mib": 1000, "utilization": 80},
            {"gpu": 2, "free_mib": 96000, "utilization": 0},
            {"gpu": 3, "free_mib": 1000, "utilization": 80},
            {"gpu": 4, "free_mib": 96000, "utilization": 0},
        ]
        with patch.object(gpu, "config", return_value={"gpus": [0, 1, 2, 3]}):
            self.assertEqual(gpu.reservation_devices(inventory, {"1": {}}, {2}), [0, 1])

    def test_compute_process_ownership_uses_exact_live_group(self):
        inventory = [{"gpu": i, "uuid": "GPU-" + str(i)} for i in range(3)]
        records = {"0": {"identity": {"pid": 123}}}
        output = subprocess.CompletedProcess(
            [], 0, "GPU-0, 456\nGPU-1, 789\nGPU-2, 987\n", ""
        )

        def group(pid):
            if pid == 987:
                raise ProcessLookupError()
            return {456: 123, 789: 999}[pid]

        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(gpu, "folder", return_value=Path(temp)),
            patch.object(gpu, "alive", return_value=True),
            patch.object(gpu.subprocess, "run", return_value=output),
            patch.object(gpu.os, "getpgid", side_effect=group),
        ):
            self.assertEqual(gpu.external_gpus(inventory, records), {1, 2})

    def test_release_cancels_owned_launch_and_leaves_foreign_devices_alone(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            witness = {"pid": 123, "start_ticks": "456", "cmdline": "owned"}
            records = {
                "0": {"intent": str(directory / "intent.json"), "identity": witness}
            }
            with (
                patch.object(gpu, "folder", return_value=directory),
                patch.object(gpu, "alive", return_value=False),
                patch.object(gpu, "stop") as terminate,
                patch.object(gpu, "log"),
            ):
                gpu.release(records, [0, 1])
            terminate.assert_called_once_with(witness)
            self.assertTrue((directory / "cancel.json").exists())
            self.assertEqual(records, {})

    def test_new_launch_never_reuses_cancelled_handshake(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            records = {}
            witness = {"pid": 123, "start_ticks": "456", "cmdline": "owned"}
            with (
                patch.object(gpu, "folder", return_value=directory),
                patch.object(gpu, "settings", return_value={"enabled": True}),
                patch.object(
                    gpu, "config", return_value={"training_python": sys.executable}
                ),
                patch.object(gpu, "root", return_value=directory),
                patch.object(
                    gpu, "launch", return_value=type("Process", (), {"pid": 123})()
                ),
                patch.object(gpu, "identity", return_value=witness),
                patch.object(gpu, "alive", return_value=False),
                patch.object(gpu, "stop"),
                patch.object(gpu, "log"),
            ):
                gpu.maintain(records, [0])
                first = Path(records["0"]["intent"])
                gpu.release(records, [0])
                gpu.maintain(records, [0])
                self.assertNotEqual(first, Path(records["0"]["intent"]))
                self.assertTrue((first.parent / "cancel.json").exists())


class QueueContracts(unittest.TestCase):
    def test_foreign_cuda_does_not_release_reservation_before_dispatch(self):
        class IterationFinished(Exception):
            pass

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            job = {
                "id": "generate",
                "kind": "generate",
                "status": "running",
                "identity": {"pid": 123},
                "category": "generation",
                "gpus": [],
                "gpu_count": 0,
                "priority": 1,
                "started_at": time.time(),
            }
            write(out / "state/jobs.json", {"generate": job})
            records = {"0": {"identity": {"pid": 456}}}
            inventory = [
                {"gpu": 0, "free_mib": 96000, "utilization": 80, "uuid": "GPU-0"}
            ]
            db = MagicMock()
            db.__enter__.return_value.execute.return_value.fetchone.return_value = (
                1077,
            )
            cfg = queue.config()
            with (
                patch.object(queue, "root", return_value=out),
                patch.object(queue, "lock", return_value=nullcontext()),
                patch.object(queue, "definitions", return_value={"generate": job}),
                patch.object(queue, "apply_scope"),
                patch.object(queue, "adopt"),
                patch.object(queue, "alive", return_value=True),
                patch.object(queue, "last_progress", return_value=time.time()),
                patch.object(queue, "database", return_value=db),
                patch.object(queue, "gpu_inventory", return_value=inventory),
                patch.object(queue, "render"),
                patch.object(queue, "log"),
                patch.object(queue.time, "sleep", side_effect=IterationFinished),
                patch.object(gpu, "config", return_value=cfg),
                patch.object(gpu, "load", return_value=records),
                patch.object(gpu, "external_gpus", return_value={0}),
                patch.object(gpu, "release") as release,
                patch.object(gpu, "maintain") as maintain,
            ):
                with self.assertRaises(IterationFinished):
                    queue.run()
            release.assert_not_called()
            maintain.assert_called_once_with(records, [0])

    def test_fresh_study_has_no_historical_tuning_fixture(self):
        cfg = {
            **queue.config(),
            "initialization": "fresh_base",
            "parent_unique_inputs": 0,
        }
        records = [
            {"run_id": "neu", "variant": "NEU", "model": "Qwen_35_9B", "group": "main"}
        ]
        with (
            patch.object(queue, "config", return_value=cfg),
            patch.object(queue, "read", return_value={"runs": records}),
        ):
            jobs = queue.definitions()
        self.assertNotIn(queue.job_id("tune", "Qwen_35_9B", 0), jobs)
        later = jobs[queue.job_id("tune", "Qwen_35_9B", 25000)]
        self.assertIn(queue.job_id("train", 10000, "neu"), later["dependencies"])

    def test_retired_study_cannot_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            write(out / "state/retired_by_user_20261012.json", {})
            with patch.object(queue, "root", return_value=out):
                with self.assertRaisesRegex(common.DataError, "retired"):
                    queue.start()

    def test_tuning_marker_handles_integer_milestone(self):
        with patch.object(queue, "root", return_value=Path("/artifacts")):
            job = {"kind": "tune", "args": ["--model", "Qwen_35_9B", "--milestone", 0]}
            self.assertEqual(
                queue.marker(job), Path("/artifacts/efficiency/Qwen_35_9B/0/plan.json")
            )

    def test_dispatch_executes_only_after_durable_ack(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            output = directory / "result"
            intent = directory / "intent.json"
            write(
                intent,
                {
                    "command": [
                        sys.executable,
                        "-c",
                        "from pathlib import Path;import sys;Path(sys.argv[1]).write_text('executed');sys.exit(7)",
                        str(output),
                    ]
                },
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(queue.CODE / "pipeline.py"),
                    "execute",
                    "--intent",
                    str(intent),
                ]
            )
            try:
                deadline = time.monotonic() + 10
                while (
                    not (directory / "identity.json").exists()
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.02)
                self.assertTrue((directory / "identity.json").exists())
                self.assertFalse(output.exists())
                write(directory / "ack.json", {})
                self.assertEqual(process.wait(timeout=10), 7)
                self.assertEqual(read(directory / "exit.json")["exit_code"], 7)
                self.assertEqual(output.read_text(), "executed")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()

    def test_controller_recovery_adopts_unacknowledged_dispatch(self):
        with tempfile.TemporaryDirectory() as temp:
            state = Path(temp)
            record = {"pid": 123, "start_ticks": "456", "cmdline": "owned wrapper"}
            write(state / "identity.json", record)
            job = {
                "status": "dispatching",
                "intent": str(state / "intent.json"),
                "started_at": time.time(),
            }
            with patch.object(queue, "alive", return_value=True):
                queue.adopt(job)
            self.assertEqual(job["status"], "running")
            self.assertEqual(job["identity"], record)
            self.assertTrue((state / "ack.json").exists())

    def test_all_runs_have_all_stages_and_benchmarks(self):
        cfg = queue.config()
        records = [
            {
                "run_id": "r" + str(i),
                "variant": "NEU" if i < 9 else "body_format",
                "model": "Qwen_38_27B" if i < 4 else "Qwen_35_9B",
                "group": "main" if i < 8 else "historical_diagnostic",
            }
            for i in range(19)
        ]
        with patch.object(queue, "read", return_value={"runs": records}):
            jobs = queue.definitions()
        self.assertEqual(
            sum(j["kind"] == "train" for j in jobs.values()),
            19 * len(cfg["milestones"]),
        )
        self.assertEqual(sum(j["kind"] == "score" for j in jobs.values()), 19 * 4)
        for job in jobs.values():
            self.assertTrue(all(dep in jobs for dep in job["dependencies"]))
        for record in records:
            for previous, current in zip(cfg["milestones"], cfg["milestones"][1:]):
                later = jobs[queue.job_id("train", current, record["run_id"])]
                self.assertIn(
                    queue.job_id("train", previous, record["run_id"]),
                    later["dependencies"],
                )

    def test_deferred_models_preserve_completed_jobs_and_can_resume(self):
        records = [
            {"run_id": "small", "model": "Qwen_35_9B"},
            {"run_id": "large", "model": "Qwen_38_27B"},
        ]
        jobs = {
            "small": {"args": ["--run-id", "small"], "status": "pending"},
            "large": {"args": ["--run-id", "large"], "status": "pending"},
            "done": {"args": ["--model", "Qwen_38_27B"], "status": "done"},
        }
        with (
            patch.object(queue, "read", return_value={"runs": records}),
            patch.object(queue, "active_models", return_value={"Qwen_35_9B"}),
        ):
            queue.apply_scope(jobs)
        self.assertEqual(jobs["small"]["status"], "pending")
        self.assertEqual(jobs["large"]["status"], "deferred")
        self.assertEqual(jobs["done"]["status"], "done")
        with (
            patch.object(queue, "read", return_value={"runs": records}),
            patch.object(
                queue, "active_models", return_value={"Qwen_35_9B", "Qwen_38_27B"}
            ),
        ):
            queue.apply_scope(jobs)
        self.assertEqual(jobs["large"]["status"], "pending")

    def test_pending_data_and_missing_credentials_never_launch(self):
        job = {
            "kind": "prepare",
            "args": ["--milestone", 10000],
            "dependencies": ["audit"],
        }
        self.assertFalse(queue.ready(job, {"audit": {"status": "done"}}, 0))
        generation = {"kind": "generate", "args": [], "dependencies": ["audit"]}
        with patch.object(queue, "credentials_ready", return_value=False):
            self.assertFalse(
                queue.ready(generation, {"audit": {"status": "done"}}, 100000)
            )


class FreshTrainingContracts(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "initialization": "fresh_base",
            "parent_unique_inputs": 0,
            "milestones": [10000, 25000],
        }
        self.record = {
            "run_id": "neu",
            "parent_checkpoint": None,
            "parent_global_step": 0,
            "parent_training_examples": 0,
        }

    def test_first_stage_starts_at_base_with_zero_steps(self):
        with (
            patch.object(train, "config", return_value=self.cfg),
            patch.object(train, "run_record", return_value=self.record),
        ):
            self.assertEqual(train.parent("neu", 10000), (None, 0, 0))
            with self.assertRaises(common.DataError):
                train.parent("neu", 999)

    def test_any_historical_adapter_is_forbidden(self):
        with patch.object(train, "config", return_value=self.cfg):
            for key, value in (
                ("parent_checkpoint", "/old/checkpoint-131"),
                ("parent_data", "/old/train.jsonl"),
                ("parent_global_step", 131),
            ):
                with self.assertRaises(common.DataError):
                    train.initial_parent({**self.record, key: value})

    def test_later_stage_parent_cannot_escape_new_experiment(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            meta = {
                "complete": True,
                "checkpoint": str(out / "runs/neu/10000/training/checkpoint-313"),
                "cumulative_global_step": 313,
                "cumulative_unique_inputs": 10000,
            }
            with (
                patch.object(train, "config", return_value=self.cfg),
                patch.object(train, "run_record", return_value=self.record),
                patch.object(train, "root", return_value=out),
                patch.object(train, "read", return_value=meta),
            ):
                self.assertEqual(train.parent("neu", 25000)[1:], (313, 10000))
                meta["checkpoint"] = "/old/checkpoint-131"
                with self.assertRaises(common.DataError):
                    train.parent("neu", 25000)

    def test_gpt_credentials_do_not_require_qwen(self):
        cfg = {"api_credentials": "/unused", "required_credential_groups": ["GPT"]}
        with (
            patch.object(common, "config", return_value=cfg),
            patch.object(
                common, "read", return_value={"keys": {"GPT": ["test-placeholder"]}}
            ),
        ):
            self.assertTrue(common.credentials_ready())

    def test_fresh_generation_rejects_qwen_teacher(self):
        gpt = {"model": "gpt-6-astra", "credential_group": "GPT"}
        cfg = {
            **self.cfg,
            "explanation_policy": "solution_summary_v1",
            "generation_recipe": {"reference_generator": gpt, "cot_generator": gpt},
        }
        with patch.object(data, "config", return_value=cfg):
            recipe, prompts = data.generation_policy()
            self.assertEqual(recipe["reference_generator"]["model"], "gpt-6-astra")
            self.assertEqual(set(prompts), {"NEU", "SUP", "AUG-SUP", "SSR"})
            cfg["generation_recipe"]["reference_generator"] = {
                "model": "qwen3.8-max",
                "credential_group": "Qwen_Max",
            }
            with self.assertRaises(common.DataError):
                data.generation_policy()


if __name__ == "__main__":
    unittest.main()

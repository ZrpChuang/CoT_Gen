"""Protect data isolation, deterministic selectors, and durable queue dependencies."""

import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from core import common, data, queue, train
from core.common import freeze_lines, prompt_token_count, read, write
from core.data import Decontaminator, context
from core.variants import choose, random_assign


class DataContracts(unittest.TestCase):
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


class QueueContracts(unittest.TestCase):
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

"""Persistent DAG scheduler, GPU leases, bounded retries, and process supervision."""

from __future__ import annotations

import csv
import io
import os
import shutil
import signal
import subprocess
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import (
    BASE,
    BENCHMARKS,
    CODE,
    PATTERN,
    PROJECT,
    DataError,
    active_models,
    alive,
    config,
    credentials_ready,
    database,
    identity,
    launch,
    lock,
    log,
    read,
    root,
    sha,
    stop,
    write,
)
from .data import stage_directory
from .train import directory, encoding_directory


def job_id(kind, *parts):
    return "__".join([kind, *map(str, parts)])


def definitions():
    cfg = config()
    runs = read(root() / "manifest.json")["runs"]
    jobs = {}

    def add(key, kind, args=(), dependencies=(), gpus=0, category="cpu", priority=0):
        jobs[key] = {
            "id": key,
            "kind": kind,
            "args": list(args),
            "dependencies": list(dependencies),
            "gpu_count": gpus,
            "category": category,
            "priority": priority,
            "status": "pending",
            "failures": 0,
            "launches": 0,
            "retry_at": 0,
        }

    add("audit", "audit", priority=-100)
    add("inventory", "inventory", priority=-110)
    add("eval_preflight", "eval-preflight", priority=-100)
    for model in cfg["models"]:
        add(
            job_id("preflight", model),
            "preflight",
            ["--model", model],
            ["inventory"],
            gpus=1,
            category="gpu",
            priority=-100,
        )
        add(
            job_id("capacity", model),
            "capacity",
            ["--model", model],
            [job_id("preflight", model)],
            gpus=1,
            category="gpu",
            priority=-90,
        )
        if model == "Qwen_35_9B" and cfg.get("initialization") != "fresh_base":
            add(
                job_id("tune", model, 0),
                "tune",
                ["--model", model, "--milestone", 0],
                [job_id("capacity", model)],
                gpus=cfg["training_gpu_count"][model],
                category="gpu",
                priority=-80,
            )
    add(
        "generate",
        "generate",
        dependencies=["audit", "inventory"],
        category="generation",
        priority=-50,
    )
    for stage, milestone in enumerate(cfg["milestones"]):
        prepare = job_id("prepare", milestone)
        add(
            prepare,
            "prepare",
            ["--milestone", milestone],
            ["audit"],
            priority=stage * 100,
        )
        selected = job_id("assemble", milestone)
        scoring = []
        for shard in range(cfg["selection_shards"]):
            key = job_id("select", milestone, shard)
            scoring.append(key)
            add(
                key,
                "select",
                ["--milestone", milestone, "--shard", shard],
                [prepare, job_id("preflight", "Qwen_35_9B")],
                gpus=1,
                category="gpu",
                priority=stage * 100 + 2,
            )
        add(
            selected,
            "assemble",
            ["--milestone", milestone],
            scoring,
            priority=stage * 100 + 3,
        )
        for record in runs:
            run_id, model, variant = (
                record["run_id"],
                record["model"],
                record["variant"],
            )
            args = ["--run-id", run_id, "--milestone", milestone]
            canonical_variant = {
                "NEU_lr1e5": "NEU",
                "sup_lr1e5": "SUP",
                "aug_sup_lr1e5": "AUG-SUP",
                "ssr_lr1e5": "SSR",
            }.get(variant, variant)
            encoded = job_id("encode", milestone, model, canonical_variant)
            dependencies = [prepare]
            if canonical_variant in (
                "student_format",
                "matched_random_format",
                "body_format",
                "body_matched_random",
                "paired_plain",
                "paired_steps",
            ):
                dependencies.append(selected)
            if encoded not in jobs:
                add(encoded, "encode", args, dependencies, priority=stage * 100 + 4)
            tuning = job_id("tune", model, milestone)
            if model == "Qwen_35_9B" and record["group"] == "main" and variant == "NEU":
                tuning_dependencies = [encoded, job_id("capacity", model)]
                if stage:
                    tuning_dependencies.append(
                        job_id("train", cfg["milestones"][stage - 1], run_id)
                    )
                add(
                    tuning,
                    "tune",
                    ["--model", model, "--milestone", milestone],
                    tuning_dependencies,
                    gpus=cfg["training_gpu_count"][model],
                    category="gpu",
                    priority=stage * 100 + 5,
                )
            training = job_id("train", milestone, run_id)
            dependencies = [encoded, job_id("capacity", model)]
            if model == "Qwen_35_9B":
                dependencies.append(tuning)
            if stage:
                dependencies.append(
                    job_id("train", cfg["milestones"][stage - 1], run_id)
                )
            group = (
                0
                if record["group"] == "main"
                else 10
                if record["group"] == "learning_rate_control"
                else 20
            )
            add(
                training,
                "train",
                args,
                dependencies,
                gpus=cfg["training_gpu_count"][model],
                category="gpu",
                priority=stage * 100
                + 10
                + group
                + (0 if model == "Qwen_38_27B" else 1),
            )
            add(
                job_id("wandb", milestone, run_id),
                "wandb",
                args,
                [encoded],
                category="sync",
                priority=stage * 100 + 20,
            )
            if milestone in cfg["evaluation_at"]:
                inference = job_id("infer", milestone, run_id)
                add(
                    inference,
                    "infer",
                    args,
                    [training, "eval_preflight"],
                    gpus=2,
                    category="gpu",
                    priority=stage * 100 + 5,
                )
                scores = []
                for bench in BENCHMARKS:
                    key = job_id("score", milestone, run_id, bench)
                    scores.append(key)
                    # Per-benchmark generated marker permits CPU scoring during inference.
                    add(
                        key,
                        "score",
                        args + ["--bench", bench],
                        ["eval_preflight"],
                        category="score",
                        priority=stage * 100 + 6,
                    )
                add(
                    job_id("verify", milestone, run_id),
                    "verify",
                    args,
                    [inference, *scores],
                    priority=stage * 100 + 7,
                )
    return jobs


def arguments(job):
    values = job["args"]
    return {
        values[i].removeprefix("--").replace("-", "_"): values[i + 1]
        for i in range(0, len(values), 2)
    }


def apply_scope(jobs):
    enabled = active_models()
    records = {r["run_id"]: r["model"] for r in read(root() / "manifest.json")["runs"]}
    for job in jobs.values():
        values = arguments(job)
        model = values.get("model") or records.get(values.get("run_id"))
        if model and model not in enabled and job["status"] == "pending":
            job["status"] = "deferred"
        elif model in enabled and job["status"] == "deferred":
            job["status"] = "pending"


def set_models(models):
    for name in ("supervisor.json", "controller.json"):
        path = root() / "state" / name
        if path.exists() and alive(read(path)):
            raise DataError("Stop the queue before changing model scope")
    if not models or not set(models) <= set(config()["models"]):
        raise DataError("Unknown model scope")
    jobs_path = root() / "state/jobs.json"
    jobs = read(jobs_path) if jobs_path.exists() else {}
    if any(j["status"] in ("running", "dispatching") for j in jobs.values()):
        raise DataError("Stop the queue before changing model scope")
    operations_path = root() / "config/operations.json"
    operations = read(operations_path) if operations_path.exists() else {}
    operations.update(
        {
            "active_models": models,
            "time": time.time(),
            "reason": "Explicit user scheduling instruction",
        }
    )
    write(operations_path, operations)
    (root() / "state/active_complete.json").unlink(missing_ok=True)
    if jobs:
        apply_scope(jobs)
        write(jobs_path, jobs)


def marker(job):
    values = arguments(job)
    kind = job["kind"]
    if kind == "audit":
        return root() / "data/source_audit.json"
    if kind == "inventory":
        return root() / "state/checks/history.json"
    if kind == "eval-preflight":
        return root() / "state/evaluator_preflight/complete.json"
    if kind == "preflight":
        return root() / "state" / ("preflight_" + values["model"] + ".json")
    if kind == "capacity":
        return root() / "state/checks" / ("capacity_" + values["model"] + ".json")
    if kind == "tune":
        return (
            root()
            / "efficiency"
            / values["model"]
            / str(values["milestone"])
            / "plan.json"
        )
    if kind == "generate":
        return root() / "data/generation_complete.json"
    milestone = int(values["milestone"])
    if kind == "prepare":
        return stage_directory(milestone) / "manifest.json"
    if kind == "select":
        return (
            stage_directory(milestone)
            / "selected/scores"
            / (str(values["shard"]) + ".complete.json")
        )
    if kind == "assemble":
        return stage_directory(milestone) / "selected/manifest.json"
    out = directory(values["run_id"], milestone)
    if kind == "encode":
        return encoding_directory(values["run_id"], milestone) / "complete.json"
    if kind == "train":
        return out / "training/complete.json"
    if kind == "wandb":
        path = out / "training/wandb_state.json"
        return (
            path
            if path.exists() and read(path).get("status") == "synced_complete"
            else None
        )
    if kind == "infer":
        return out / "evaluation/runtime/generated_arena_hard_v2.json"
    if kind == "score":
        return out / "evaluation" / values["bench"] / "metrics.json"
    if kind == "verify":
        return out / "evaluation/audit.json"
    return None


def ready(job, jobs, accepted):
    if any(jobs[key]["status"] != "done" for key in job["dependencies"]):
        return False
    values = arguments(job)
    if job["kind"] == "generate" and not credentials_ready():
        job["waiting_reason"] = "missing_teacher_credentials"
        return False
    if (
        job["kind"] == "prepare"
        and accepted < int(values["milestone"]) - config()["parent_unique_inputs"]
    ):
        job["waiting_reason"] = "waiting_for_paired_data"
        return False
    if job["kind"] == "score":
        path = (
            directory(values["run_id"], int(values["milestone"]))
            / "evaluation/runtime"
            / ("generated_" + values["bench"] + ".json")
        )
        if not path.exists():
            return False
        if values["bench"] != "ifeval" and not credentials_ready():
            return False
    if job["kind"] == "wandb":
        training_job = jobs[job_id("train", values["milestone"], values["run_id"])]
        return training_job["status"] in ("running", "done")
    return True


def gpu_inventory():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu,uuid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    return [
        {
            "gpu": int(fields[0]),
            "free_mib": int(fields[1]),
            "utilization": int(fields[2]),
            "uuid": fields[3],
        }
        for line in result.stdout.splitlines()
        if (fields := [x.strip() for x in line.split(",")]) and len(fields) == 4
    ]


def cleanup(job):
    if job["kind"] == "infer":
        from .evaluate import cleanup_services

        values = arguments(job)
        cleanup_services(values["run_id"], int(values["milestone"]))


def receipt_path(job):
    return root() / "state/receipts" / (job["id"] + ".json")


def last_progress(job):
    paths = []
    if job["kind"] == "generate":
        paths = [root() / "data/generation_progress.json"]
    values = arguments(job)
    if "run_id" in values and "milestone" in values:
        out = directory(values["run_id"], int(values["milestone"]))
        if job["kind"] in ("train", "wandb"):
            paths = [out / "training/metrics.jsonl", out / "training/complete.json"]
        elif job["kind"] in ("infer", "score"):
            paths = list((out / "evaluation").glob("*/responses.jsonl")) + list(
                (out / "evaluation").glob("*/judgments.jsonl")
            )
    if job["kind"] == "select":
        paths = list(
            (stage_directory(int(values["milestone"])) / "selected/scores").glob(
                "*.jsonl"
            )
        )
    return max([job["started_at"], *(p.stat().st_mtime for p in paths if p.exists())])


def metric(path, bench):
    if not path.is_file():
        return None
    data = read(path)
    if data.get("status") != "complete":
        return None
    if bench == "ifeval":
        return data["metrics"]["prompt_strict"]["score"]
    judged = data["judges"]["gpt-6-astra"]
    if bench == "multi_challenge":
        return judged["overall"]["score"]
    if bench == "eq_bench3":
        return judged["rubric_0_100"]["score"]
    return judged["weighted_win_rate"]["hard_prompt"]["score"]


def results(jobs):
    """Render verified metrics only; missing or partial evaluations remain explicit."""
    records = read(root() / "manifest.json")["runs"]
    models = list(config()["models"])
    final = config()["target_unique_inputs"]
    fresh = config().get("initialization") == "fresh_base"
    labels = dict(
        zip(BENCHMARKS, ("IFEval", "MultiChallenge", "EQ-Bench 3", "ArenaHard"))
    )

    def cell(old, new=None, baseline=False):
        before = f"{old:.2f}" if old is not None else "未完成"
        if baseline:
            return before
        after = (
            f"{new:.2f} ({new - old:+.2f})"
            if new is not None and old is not None
            else f"{new:.2f}"
            if new is not None
            else "未完成"
        )
        return before + " → " + after

    def scores(record, bench):
        if fresh:
            new_dir = directory(record["run_id"], final) / "evaluation"
            verified = (
                jobs[job_id("verify", final, record["run_id"])]["status"] == "done"
            )
            value = (
                metric(new_dir / bench / "metrics.json", bench) if verified else None
            )
            return (
                "暂缓"
                if record["model"] not in active_models()
                else f"{value:.2f}"
                if value is not None
                else "未完成"
            )
        if record["group"] == "main":
            old_dir = BASE / record["variant"] / record["model"] / "evaluation"
        elif record["group"] == "learning_rate_control":
            old_dir = Path(record["parent_complete"]).parent.parent / "evaluation"
        else:
            old_dir = PATTERN / "results/benchmarks" / record["variant"]
        new_dir = directory(record["run_id"], final) / "evaluation"
        verified = jobs[job_id("verify", final, record["run_id"])]["status"] == "done"
        return cell(
            metric(old_dir / bench / "metrics.json", bench),
            metric(new_dir / bench / "metrics.json", bench) if verified else None,
        )

    rows = [
        "# GPT 全新训练结果" if fresh else "# 统一扩训结果",
        "",
        "全新 GPT 共享答案与解题说明、从原始基座新建 LoRA，旧训练权重及旧生成缓存全部弃用。只展示新实验核验结果；BASE 为原始未训练模型。"
        if fresh
        else "分数：历史 4,175 → 累计 100K（括号为百分点变化）。新结果仅在整组四项评测核验通过后展示。BASE 为原始未训练模型。",
        "",
        "| 方法 | "
        + " | ".join(
            labels[b] + " / " + ("9B" if "9B" in m else "27B")
            for b in BENCHMARKS
            for m in models
        )
        + " |",
        "|---|" + "---:|" * (4 * len(models)),
    ]
    rows.append(
        "| BASE | "
        + " | ".join(
            cell(
                metric(PROJECT / "test_scripts" / m / b / "metrics.json", b),
                baseline=True,
            )
            for b in BENCHMARKS
            for m in models
        )
        + " |"
    )
    for method in ("NEU", "SUP", "AUG-SUP", "SSR"):
        rows.append(
            "| "
            + method
            + " | "
            + " | ".join(
                scores(
                    next(
                        r
                        for r in records
                        if r["group"] == "main"
                        and r["variant"] == method
                        and r["model"] == m
                    ),
                    b,
                )
                for b in BENCHMARKS
                for m in models
            )
            + " |"
        )
    rows += [
        "",
        "## 从基座重跑的对照与诊断（9B）" if fresh else "## 历史对照与诊断（9B）",
        "",
        "| 训练臂 | " + " | ".join(labels.values()) + " |",
        "|---|---:|---:|---:|---:|",
    ]
    for record in records:
        if record["group"] != "main":
            rows.append(
                "| "
                + record["variant"]
                + " | "
                + " | ".join(scores(record, b) for b in BENCHMARKS)
                + " |"
            )
    rows += [
        "",
        "数据为多语言、多轮扩展，阶段间重新初始化优化器与调度器；结果属于本地冻结评测协议，不能单独归因于数据量或称为论文严格复现。IFEval 原始未设种子的 letter_frequency 检查存在已知两题波动，部署检查仅容许这两题的该指令位变化。",
        "",
    ]
    target = root() / "reports/RESULTS.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    tmp.write_text("\n".join(rows))
    tmp.replace(target)
    ledger = io.StringIO()
    writer = csv.writer(ledger, delimiter="\t", lineterminator="\n")
    writer.writerow(
        [
            "timestamp",
            "run_id",
            "artifact",
            "metric",
            "metric_value",
            "direction",
            "status",
            "runtime_seconds_current_call",
            "gpu_count",
            "description",
            "output_path",
        ]
    )
    for record in records:
        for milestone in config()["milestones"]:
            job = jobs[job_id("train", milestone, record["run_id"])]
            if job["status"] != "done":
                continue
            path = directory(record["run_id"], milestone) / "training/complete.json"
            meta = read(path)
            writer.writerow(
                [
                    meta["time"],
                    record["run_id"],
                    meta["checkpoint"],
                    "cumulative_unique_inputs",
                    meta["cumulative_unique_inputs"],
                    "target",
                    "diagnostic",
                    meta["runtime_current_train_call_seconds"],
                    meta["world_size"],
                    "Fixed continuation stage; no selection by test scores",
                    str(path),
                ]
            )
        for bench in BENCHMARKS:
            path = (
                directory(record["run_id"], final)
                / "evaluation"
                / bench
                / "metrics.json"
            )
            if jobs[job_id("verify", final, record["run_id"])]["status"] == "done":
                writer.writerow(
                    [
                        jobs[job_id("verify", final, record["run_id"])]["finished_at"],
                        record["run_id"],
                        str(path),
                        bench,
                        metric(path, bench),
                        "higher",
                        "diagnostic",
                        "",
                        2,
                        "Frozen local protocol; final result retained regardless of score",
                        str(path),
                    ]
                )
    target = root() / "reports/ledger.tsv"
    tmp = target.with_suffix(".part")
    tmp.write_text(ledger.getvalue())
    tmp.replace(target)


def execute(intent_path):
    """A durable launch handshake closes the controller crash/dispatch race."""
    intent_path = Path(intent_path)
    intent = read(intent_path)
    state = intent_path.parent
    write(state / "identity.json", identity(os.getpid()))
    deadline = time.monotonic() + 300
    while not (state / "ack.json").exists():
        if (state / "cancel.json").exists() or time.monotonic() > deadline:
            write(state / "exit.json", {"exit_code": 75})
            return 75
        time.sleep(0.2)
    if (state / "cancel.json").exists():
        return 75
    child = subprocess.Popen(intent["command"])  # Same owned process group.
    code = child.wait()
    write(state / "exit.json", {"exit_code": code, "time": time.time()})
    return code


def adopt(job):
    if job["status"] not in ("dispatching", "running") or "intent" not in job:
        return
    state = Path(job["intent"]).parent
    path = state / "identity.json"
    record = read(path) if path.exists() else job.get("identity")
    if alive(record):
        job.update(status="running", identity=record)
        write(state / "ack.json", {"time": time.time()})
    elif job["status"] == "dispatching" and time.time() - job["started_at"] > 60:
        write(state / "cancel.json", {"time": time.time()})
        job["status"] = "pending"
        job["retry_at"] = time.time() + config()["retry_backoff_seconds"]


def stop_jobs(jobs):
    def terminate(job):
        if "intent" in job:
            write(Path(job["intent"]).parent / "cancel.json", {"time": time.time()})
        stop(job.get("identity"))
        cleanup(job)
        job["status"] = "pending"

    active = [j for j in jobs.values() if j["status"] in ("running", "dispatching")]
    with ThreadPoolExecutor(max_workers=max(1, len(active))) as workers:
        list(workers.map(terminate, active))


def render(jobs, accepted, inventory):
    from . import gpu as reservations

    held = reservations.snapshot()
    generation = generation_summary(accepted)
    counts = dict(Counter(j["status"] for j in jobs.values()))
    running = [
        {k: j.get(k) for k in ("id", "kind", "started_at", "gpus")}
        for j in jobs.values()
        if j["status"] == "running"
    ]
    blocked = [
        {k: j.get(k) for k in ("id", "kind", "error_type", "waiting_reason", "log")}
        for j in jobs.values()
        if j["status"] in ("blocked", "failed")
    ]
    payload = {
        "time": time.time(),
        "pid": os.getpid(),
        "run_id": config()["run_id"],
        "counts": counts,
        "active_models": sorted(active_models()),
        "deferred_models": sorted(set(config()["models"]) - active_models()),
        "accepted_new": accepted,
        "target_new": config()["target_unique_inputs"]
        - config()["parent_unique_inputs"],
        "credentials_ready": credentials_ready(),
        "running": running,
        "blocked": blocked,
        "gpus": inventory,
        "gpu_reservations": held,
        "generation": generation,
        "phase": "active"
        if running
        else "waiting_teacher_credentials"
        if not credentials_ready()
        else "waiting_dependencies",
    }
    write(root() / "state/status.json", payload)
    rows = [
        "# 100K 远端队列",
        "",
        "本轮启用："
        + ", ".join(sorted(active_models()))
        + "；其他模型暂缓，历史数据与配置保留。",
        "",
        f"新增有效数据：{accepted:,} / {payload['target_new']:,}；教师凭证：{'已就绪' if payload['credentials_ready'] else '等待恢复'}。",
        "",
        "资源保持 GPU："
        + (", ".join(str(rec["gpu"]) for rec in held if rec["alive"]) or "无")
        + "。保卡单独统计，不计入研究训练步数；训练、NLL 或评测就绪后自动让卡。",
        "",
        "## 数据生成进度",
        "",
        f"最近 {generation['window_minutes']:.1f} 分钟：新增 {generation['accepted_delta']:,} 条共同合格样本，新增 {generation['cache_delta']:,} 条请求缓存；有效速度 {generation['accepted_per_minute']:.2f} 条/分钟。",
        f"首批 10K 尚缺 {max(0, 10000 - accepted):,} 条；按上述窗口速度预计剩余 {generation['hours_to_10k'] if generation['hours_to_10k'] is not None else '暂不可估计'} 小时。",
        f"活跃请求 {generation['active_requests']} / {generation['concurrency']}；最久等待 {generation['oldest_request_seconds']:.1f} 秒；已完成待按序入库 {generation['completed_uncommitted']} 条；预取 {generation['prefetch_batches']} 批。",
        f"运行遥测状态：{generation['telemetry_status']}。进程存活不能代替样本进展，缓存数不能计作合格样本。",
        "",
        "## 正在执行",
        "",
        "| 任务 | GPU | 日志 |",
        "|---|---|---|",
    ]
    for job in jobs.values():
        if job["status"] == "running":
            rows.append(f"| {job['id']} | {job.get('gpus', [])} | `{job['log']}` |")
    rows += [
        "",
        "## 19 项训练进度",
        "",
        "| 实验 | 已完成的数据规模 | 当前阶段 | 最终评测 |",
        "|---|---:|---|---|",
    ]
    manifest_path = root() / "manifest.json"
    manifest = read(manifest_path)
    original_manifest = sha(manifest)
    for record in manifest["runs"]:
        run_id = record["run_id"]
        done = [
            m
            for m in config()["milestones"]
            if jobs[job_id("train", m, run_id)]["status"] == "done"
        ]
        active = [
            str(m)
            for m in config()["milestones"]
            if jobs[job_id("train", m, run_id)]["status"] == "running"
        ]
        final = config()["target_unique_inputs"]
        verified = jobs[job_id("verify", final, run_id)]["status"] == "done"
        record["status"] = (
            "deferred"
            if record["model"] not in active_models()
            else "evaluated"
            if verified
            else "training"
            if active
            else "trained"
            if final in done
            else "queued"
        )
        if done:
            latest = read(directory(run_id, max(done)) / "training/complete.json")
            record["actual_new_data"] = max(done) - config()["parent_unique_inputs"]
            record["actual_new_global_step"] = (
                latest["cumulative_global_step"] - record["parent_global_step"]
            )
            sync = directory(run_id, max(done)) / "training/wandb_state.json"
            if sync.exists():
                record["actual_wandb_run"] = read(sync).get("url")
        rows.append(
            f"| {run_id} | {max(done, default=config()['parent_unique_inputs']):,} | {', '.join(active) or ('暂缓' if record['model'] not in active_models() else '排队')} | {'已核验' if verified else '未完成'} |"
        )
    rows += ["", "## 异常", ""]
    rows += [
        f"- `{j['id']}`：{j.get('error_type', j.get('waiting_reason', j['status']))}；日志 `{j.get('log', '')}`。"
        for j in jobs.values()
        if j["status"] in ("blocked", "failed")
    ]
    if not blocked:
        rows.append("无终止性错误。等待数据或凭证不计为任务失败。")
    report = root() / "reports/STATUS.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = report.with_suffix(".part")
    temporary.write_text("\n".join(rows) + "\n")
    temporary.replace(report)
    if sha(manifest) != original_manifest:
        write(manifest_path, manifest)
    results(jobs)
    return payload


def generation_summary(accepted):
    """Bounded durable rate history and request ages; never liveness-only progress."""
    now = time.time()
    runtime_path = root() / "state/generation_runtime.json"
    runtime = read(runtime_path) if runtime_path.exists() else {}
    path = root() / "state/generation_monitor.json"
    history = read(path).get("history", []) if path.exists() else []
    cache = runtime.get("request_cache", history[-1]["cache"] if history else 0)
    history = [s for s in history if s["time"] >= now - 6 * 3600]
    latest = {"time": now, "accepted": accepted, "cache": cache}
    if not history or now - history[-1]["time"] >= 60:
        history.append(latest)
    eligible = [s for s in history if s["time"] <= now - 3600]
    anchor = eligible[-1] if eligible else history[0]
    minutes = max(0, (now - anchor["time"]) / 60)
    delta = max(0, accepted - anchor["accepted"])
    rate = delta / minutes if minutes >= 5 else 0
    requests = runtime.get("active_requests", [])
    result = {
        "time": now,
        "window_minutes": round(minutes, 1),
        "accepted_delta": delta,
        "cache_delta": max(0, cache - anchor["cache"]),
        "accepted_per_minute": round(rate, 3),
        "hours_to_10k": round(max(0, 10000 - accepted) / rate / 60, 1)
        if rate
        else None,
        "active_requests": len(requests),
        "oldest_request_seconds": max(
            (r["elapsed_seconds"] for r in requests), default=0
        ),
        "concurrency": runtime.get("concurrency", 8),
        "completed_uncommitted": runtime.get("completed_uncommitted", 0),
        "prefetch_batches": runtime.get("prefetch_batches", 0),
        "telemetry_status": "fresh"
        if now - runtime.get("time", 0) < 120
        else "stale_or_unavailable",
    }
    write(path, {**result, "history": history[-361:]})
    return result


def run():
    from . import gpu as reservations

    cfg = config()
    with lock(root() / "state/queue.lock"):
        write(root() / "state/controller.json", identity(os.getpid()))
        path = root() / "state/jobs.json"
        jobs = read(path) if path.exists() else definitions()
        if set(jobs) != set(definitions()):
            raise DataError("Task definition changed after queue creation")
        apply_scope(jobs)
        processes = {}
        stop_requested = False

        def requested(*_):
            nonlocal stop_requested
            stop_requested = True

        signal.signal(signal.SIGTERM, requested)
        signal.signal(signal.SIGINT, requested)
        log("controller_started", jobs=len(jobs))
        while not stop_requested:
            for job in jobs.values():
                adopt(job)
                if job["status"] != "running":
                    continue
                record = job.get("identity")
                handle = processes.get(job["id"])
                code = handle.poll() if handle else None
                if code is None and alive(record):
                    hours = (
                        cfg["train_timeout_hours"]
                        if job["kind"] == "train"
                        else cfg["inference_timeout_hours"]
                        if job["kind"] == "infer"
                        else cfg["cpu_job_timeout_hours"]
                    )
                    # Healthy large runs may last days; timeout means no durable progress.
                    if time.time() - last_progress(job) > hours * 3600:
                        stop(record)
                        code = 124
                    else:
                        continue
                receipt = read(receipt_path(job)) if receipt_path(job).exists() else {}
                if code is None and "intent" in job:
                    wrapper_exit = Path(job["intent"]).parent / "exit.json"
                    if wrapper_exit.exists():
                        code = read(wrapper_exit)["exit_code"]
                code = code if code is not None else receipt.get("exit_code", 1)
                if receipt.get("exit_code") in (75, 78) and code in (0, 1):
                    code = receipt["exit_code"]
                cleanup(job)
                job["finished_at"] = time.time()
                job["last_exit_code"] = code
                job["error_type"] = receipt.get("error_type", "ProcessExit")
                complete = marker(job)
                if code == 0 and complete and complete.exists():
                    job["status"] = "done"
                    job.pop("error_type", None)
                elif code == 75:
                    job["status"] = "pending"
                    job["retry_at"] = time.time() + cfg["retry_backoff_seconds"]
                elif code == 78:
                    job["status"] = "blocked"
                else:
                    job["failures"] += 1
                    job["status"] = (
                        "failed"
                        if job["failures"] >= cfg["max_job_attempts"]
                        else "pending"
                    )
                    job["retry_at"] = time.time() + min(
                        1800, cfg["retry_backoff_seconds"] * 2 ** (job["failures"] - 1)
                    )
                if job["status"] in ("failed", "blocked") and job["kind"] == "train":
                    values = arguments(job)
                    write(
                        directory(values["run_id"], int(values["milestone"]))
                        / "training/terminal_failure.json",
                        {
                            "job": job["id"],
                            "error_type": job["error_type"],
                            "time": time.time(),
                        },
                    )
                log("job_finished", job=job["id"], status=job["status"], exit_code=code)
            try:
                with database() as db:
                    accepted = db.execute("SELECT COUNT(*) FROM accepted").fetchone()[0]
                inventory = gpu_inventory()
                held = reservations.load()
                external = reservations.external_gpus(inventory, held)
            except (OSError, subprocess.SubprocessError):
                time.sleep(cfg["poll_seconds"])
                continue
            leased = {
                gpu
                for j in jobs.values()
                if j["status"] in ("running", "dispatching")
                for gpu in j.get("gpus", [])
            }
            available = reservations.available(inventory, held, leased, external)
            active = Counter(
                j["category"]
                for j in jobs.values()
                if j["status"] in ("running", "dispatching")
            )
            limits = {"cpu": 2, "generation": 1, "score": 4, "sync": 4}
            for job in sorted(
                jobs.values(), key=lambda j: (j["priority"], -j["gpu_count"], j["id"])
            ):
                if (
                    job["status"] != "pending"
                    or time.time() < job["retry_at"]
                    or not ready(job, jobs, accepted)
                ):
                    continue
                count = job["gpu_count"]
                if count > len(available) or (
                    not count and active[job["category"]] >= limits[job["category"]]
                ):
                    continue
                devices = available[:count]
                if devices:
                    reservations.release(held, devices)
                    fresh_inventory = gpu_inventory()
                    fresh_external = reservations.external_gpus(fresh_inventory, held)
                    free = reservations.available(
                        fresh_inventory, {}, leased, fresh_external
                    )
                    if not set(devices) <= set(free):
                        available = [device for device in available if device in free]
                        continue
                available = available[count:]
                receipt = receipt_path(job)
                receipt.unlink(missing_ok=True)
                command = [cfg["training_python"]]
                if job["kind"] == "train":
                    command += [
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        f"--nproc_per_node={count}",
                    ]
                command += [
                    str(CODE / "pipeline.py"),
                    "worker",
                    job["kind"],
                    *map(str, job["args"]),
                    "--receipt",
                    str(receipt),
                ]
                logfile = root() / "logs/jobs" / job["kind"] / (job["id"] + ".log")
                intent = (
                    root()
                    / "state/launches"
                    / job["id"]
                    / uuid.uuid4().hex
                    / "intent.json"
                )
                write(intent, {"command": command, "gpus": devices, "job": job["id"]})
                job.update(
                    status="dispatching",
                    identity=None,
                    intent=str(intent),
                    started_at=time.time(),
                    gpus=devices,
                    launches=job["launches"] + 1,
                    log=str(logfile),
                )
                job.pop("waiting_reason", None)
                active[job["category"]] += 1
                write(
                    path, jobs
                )  # Lease and launch intent persist before creating a process.
                process = launch(
                    [
                        cfg["training_python"],
                        str(CODE / "pipeline.py"),
                        "execute",
                        "--intent",
                        str(intent),
                    ],
                    logfile,
                    devices,
                )
                processes[job["id"]] = process
                job.update(status="running", identity=identity(process.pid))
                write(path, jobs)
                write(intent.parent / "ack.json", {"time": time.time()})
                log("job_started", job=job["id"], gpus=devices, launch=job["launches"])
            write(path, jobs)
            # Retain allocated contexts until our ready GPU work takes over.
            fresh_inventory = gpu_inventory()
            leased = {
                device
                for job in jobs.values()
                if job["status"] in ("running", "dispatching")
                for device in job.get("gpus", [])
            }
            reservations.maintain(
                held,
                reservations.reservation_devices(fresh_inventory, held, leased),
            )
            render(jobs, accepted, fresh_inventory)
            if all(j["status"] in ("done", "deferred") for j in jobs.values()):
                reservations.stop_all()
                enabled_runs = sum(
                    r["model"] in active_models()
                    for r in read(root() / "manifest.json")["runs"]
                )
                if any(j["status"] == "deferred" for j in jobs.values()):
                    write(
                        root() / "state/active_complete.json",
                        {
                            "complete": True,
                            "active_models": sorted(active_models()),
                            "runs": enabled_runs,
                            "final_evaluations": enabled_runs * 4,
                            "time": time.time(),
                        },
                    )
                    log("active_scope_complete", runs=enabled_runs)
                    return
                write(
                    root() / "state/complete.json",
                    {
                        "complete": True,
                        "runs": 19,
                        "final_evaluations": 76,
                        "time": time.time(),
                    },
                )
                manifest = read(root() / "manifest.json")
                manifest["status"] = "complete"
                write(root() / "manifest.json", manifest)
                log("queue_complete")
                return
            time.sleep(cfg["poll_seconds"])
        # Graceful stop preserves all task/checkpoint state and terminates only owned groups.
        stop_jobs(jobs)
        reservations.stop_all()
        write(path, jobs)
        log("controller_stopped")


def supervise():
    with lock(root() / "state/supervisor.lock"):
        write(root() / "state/supervisor.json", identity(os.getpid()))
        while (
            not (root() / "state/STOP").exists()
            and not (root() / "state/complete.json").exists()
            and not (root() / "state/active_complete.json").exists()
        ):
            child = launch(
                [config()["training_python"], str(CODE / "pipeline.py"), "run"],
                root() / "logs/controller.log",
            )
            write(root() / "state/controller_child.json", identity(child.pid))
            code = child.wait()
            if (
                (root() / "state/STOP").exists()
                or (root() / "state/complete.json").exists()
                or (root() / "state/active_complete.json").exists()
            ):
                return
            log("controller_restart", exit_code=code)
            time.sleep(15)


def upgrade():
    """Archive deployment and migrate compatible DAG changes while stopped."""
    for name in ("supervisor.json", "controller.json"):
        path = root() / "state" / name
        if path.exists() and alive(read(path)):
            raise DataError("Stop the queue before upgrading source")
    path = root() / "state/jobs.json"
    jobs = read(path) if path.exists() else {}
    if any(j["status"] in ("running", "dispatching") for j in jobs.values()):
        raise DataError("Workers must stop before upgrade")
    fresh = definitions()
    if not set(jobs) <= set(fresh):
        raise DataError("Upgrade would remove historical jobs")
    for key, old in jobs.items():
        if any(
            old[k] != fresh[key][k] for k in ("kind", "args", "gpu_count", "category")
        ):
            raise DataError("Upgrade changes job identity: " + key)
        fresh[key].update(old)
        fresh[key]["dependencies"] = definitions()[key]["dependencies"]
    revision = root() / "config/revisions" / str(time.time_ns())
    revision.mkdir(parents=True)
    snapshot = root() / "source"
    if snapshot.exists():
        old_cfg = read(snapshot / "config.json")
        if old_cfg != config():
            raise DataError(
                "Scientific configuration changed; use a separate experiment"
            )
        snapshot.rename(revision / "source")
    for old in (root() / "config/deployment.json", path):
        if old.exists():
            shutil.copy2(old, revision / old.name)
    write(
        revision / "revision.json",
        {
            "time": time.time(),
            "new_code_hashes": {
                str(p.relative_to(CODE)): sha(p) for p in CODE.rglob("*.py")
            },
            "operations": read(root() / "config/operations.json"),
        },
    )
    apply_scope(fresh)
    write(path, fresh)
    return start()


def start():
    cfg = config()
    if (root() / "state/retired_by_user_20261012.json").exists():
        raise DataError(
            "This experiment was retired by the user; create a new experiment"
        )
    if cfg.get("initialization") == "fresh_base":
        from .train import initial_parent

        for record in read(root() / "manifest.json")["runs"]:
            initial_parent(record)
    if (root() / "state/supervisor.json").exists() and alive(
        read(root() / "state/supervisor.json")
    ):
        return (
            read(root() / "state/status.json")
            if (root() / "state/status.json").exists()
            else {"status": "starting"}
        )
    root().mkdir(parents=True, exist_ok=True)
    (root() / "state").mkdir(exist_ok=True)
    (root() / "state/STOP").unlink(missing_ok=True)
    snapshot = root() / "source"
    if not snapshot.exists():
        shutil.copytree(
            CODE,
            snapshot,
            ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache", "*.pyc"),
        )
    else:
        live_files = {p.relative_to(CODE) for p in CODE.rglob("*.py")} | {
            Path("config.json")
        }
        frozen_files = {p.relative_to(snapshot) for p in snapshot.rglob("*.py")} | {
            Path("config.json")
        }
        if live_files != frozen_files:
            raise DataError("Deployed file inventory differs from frozen source")
        for relative in live_files:
            path = CODE / relative
            other = snapshot / path.relative_to(CODE)
            if not other.exists() or sha(path) != sha(other):
                raise DataError(
                    "Deployed source differs from frozen source; use a new run configuration"
                )
    from .evaluate import frozen_sources

    sources = root() / "config/frozen_sources.json"
    if not sources.exists():
        write(sources, frozen_sources())
    write(
        root() / "config/deployment.json",
        {
            "time": time.time(),
            "source": str(snapshot),
            "config_sha256": sha(CODE / "config.json"),
            "code_hashes": {
                str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob("*.py")
            },
            "host": os.uname().nodename,
            "config": cfg,
        },
    )
    manifest_path = root() / "manifest.json"
    manifest = read(manifest_path)
    if len(manifest["runs"]) != 19:
        raise DataError("The 19 historical runs must all remain in the manifest")
    manifest["status"] = "deployed_queue"
    manifest["continuation_config"] = cfg
    write(manifest_path, manifest)
    process = launch(
        [cfg["training_python"], str(snapshot / "pipeline.py"), "supervise"],
        root() / "logs/supervisor.log",
    )
    write(root() / "state/supervisor_launch.json", identity(process.pid))
    return {
        "status": "deployed",
        "supervisor_pid": process.pid,
        "output_root": str(root()),
        "teacher_credentials_ready": credentials_ready(),
        "gpu_counts": cfg["training_gpu_count"],
        "milestones": cfg["milestones"],
        "runs": len(manifest["runs"]),
    }


def halt():
    write(root() / "state/STOP", {"time": time.time()})
    for name in ("supervisor.json", "controller.json"):
        path = root() / "state" / name
        if path.exists():
            stop(read(path))
    path = root() / "state/jobs.json"
    if path.exists():
        jobs = read(path)
        stop_jobs(jobs)
        write(path, jobs)
    from . import gpu as reservations

    reservations.stop_all()


def status():
    path = root() / "state/status.json"
    return read(path) if path.exists() else {"status": "not_started"}

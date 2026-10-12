"""Bounded, fixed-data throughput trials and stage-frozen execution plans."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path

from .common import DataError, config, native_pack, read, root, sha, write


def folder(model, milestone):
    return root() / "efficiency" / model / str(milestone)


def runtime_plan(model, milestone):
    path = folder(model, milestone) / "plan.json"
    if not path.exists():
        return {
            "micro_batch": 1,
            "world_size": config()["training_gpu_count"][model],
            "implementation": "existing_sdpa_fla",
            "status": "baseline",
        }
    plan = read(path)
    if plan["world_size"] != config()["training_gpu_count"][model]:
        raise DataError("Execution plan changes GPU count")
    return plan


def measured_trainer(Trainer, TrainerCallback, torch, output, step_offset=0):
    """Count actual unpadded tokens; aggregate across all ranks at log intervals."""
    counters = torch.zeros(4, dtype=torch.float64, device="cuda")
    interval = {"started": None, "values": {}}

    class MeteredTrainer(Trainer):
        def training_step(self, model, inputs, *args, **kwargs):
            inputs = self._prepare_inputs(inputs)
            counters[0] += inputs["attention_mask"].sum().detach()
            counters[1] += (inputs["labels"][:, 1:] != -100).sum().detach()
            counters[2] += inputs["input_ids"].numel()
            counters[3] += inputs["input_ids"].shape[0]
            return super().training_step(model, inputs, *args, **kwargs)

    class Telemetry(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            interval["started"] = time.monotonic()
            interval["step"] = state.global_step

        def on_step_end(self, args, state, control, **kwargs):
            if (
                state.global_step % args.logging_steps
                and state.global_step != state.max_steps
            ):
                return
            torch.cuda.synchronize()
            elapsed = torch.tensor(
                time.monotonic() - interval["started"],
                dtype=torch.float64,
                device="cuda",
            )
            peak = torch.tensor(
                torch.cuda.max_memory_allocated() / 2**20,
                dtype=torch.float64,
                device="cuda",
            )
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(counters)
                torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX)
                torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
            tokens, supervised, slots, examples = counters.tolist()
            seconds = elapsed.item()
            steps = state.global_step - interval["step"]
            interval["values"] = {
                "sequence_tokens_per_second": tokens / seconds,
                "sequence_tokens_per_gpu_second": tokens
                / (seconds * int(os.environ.get("WORLD_SIZE", "1"))),
                "supervised_tokens_per_second": supervised / seconds,
                "examples_per_second": examples / seconds,
                "padding_fraction": 1 - tokens / max(slots, 1),
                "seconds_per_optimizer_step": seconds / max(steps, 1),
                "estimated_remaining_seconds": (state.max_steps - state.global_step)
                * seconds
                / max(steps, 1),
                "peak_allocated_mib_all_ranks": peak.item(),
                "interval_seconds": seconds,
                "interval_gpu_hours": seconds
                * int(os.environ.get("WORLD_SIZE", "1"))
                / 3600,
                "interval_examples": examples,
            }
            counters.zero_()
            interval["started"] = time.monotonic()
            interval["step"] = state.global_step

        def on_log(self, args, state, control, logs=None, **kwargs):
            if int(os.environ.get("RANK", "0")) or not logs:
                return
            values = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            values.update(interval["values"])
            if any(not math.isfinite(v) for v in values.values()):
                raise DataError("Nonfinite training metric")
            row = {
                "stage_global_step": state.global_step,
                "cumulative_global_step": step_offset + state.global_step,
                "time": time.time(),
                **values,
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    return MeteredTrainer, Telemetry


def fixture(model, milestone):
    from .train import data_path

    record = next(
        r
        for r in read(root() / "manifest.json")["runs"]
        if r["model"] == model and r["group"] == "main" and r["variant"] == "NEU"
    )
    source = (
        Path(record["parent_data"]) if milestone == 0 else data_path("NEU", milestone)
    )
    rows = []
    with source.open() as stream:
        for line in stream:
            rows.append(json.loads(line))
            if len(rows) == 128:
                break
    if len(rows) != 128:
        raise DataError("Throughput fixture requires 128 real training rows")
    return (
        record,
        rows,
        {"source_sha256": sha(source), "rows_sha256": sha(rows), "rows": len(rows)},
    )


def profile(model_name, milestone, micro_batch, tag):
    import torch
    from transformers import (
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    from .train import collator, load_model

    out = folder(model_name, milestone) / tag
    out.mkdir(parents=True, exist_ok=True)
    record, rows, binding = fixture(model_name, milestone)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if 32 % (world * micro_batch):
        raise DataError("Micro batch violates global batch 32")
    set_seed(record["parent_recipe"]["seed"])
    tok = AutoTokenizer.from_pretrained(
        config()["models"][model_name], local_files_only=True
    )
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    packs = [native_pack(tok, row, model_name) for row in rows]
    samples = [
        {
            "input_ids": p["input_ids"],
            "labels": p["labels"],
            "attention_mask": [1] * len(p["input_ids"]),
        }
        for p in packs
    ]
    if max(len(s["input_ids"]) for s in samples) > config()["max_sequence_tokens"]:
        raise DataError("Profiling fixture would require truncation")
    from .train import initial_parent, parent

    checkpoint = (
        initial_parent(record)
        if milestone == 0
        else parent(record["run_id"], milestone)[0]
    )
    model = load_model(model_name, checkpoint)
    device = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(device)
    model.to(device)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()
    # Candidate must handle the full configured length, including worst-case padding.
    if micro_batch > 1:
        length = config()["max_sequence_tokens"]
        ids = torch.tensor(
            [
                (
                    samples[0]["input_ids"]
                    * math.ceil(length / len(samples[0]["input_ids"]))
                )[:length]
            ]
            * micro_batch,
            device=device,
        )
        labels = ids.clone()
        labels[:, : min(128, length // 2)] = -100
        loss = model(
            input_ids=ids,
            labels=labels,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
        ).loss
        loss.backward()
        if not torch.isfinite(loss) or not all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        ):
            raise DataError("Full-length capacity gradients invalid")
        capacity_peak = torch.cuda.max_memory_allocated() / 2**20
        if (
            capacity_peak
            > torch.cuda.get_device_properties(device).total_memory / 2**20 * 0.9
        ):
            raise DataError("Candidate lacks 10% memory headroom")
        model.zero_grad(set_to_none=True)
        del ids, labels, loss
        torch.cuda.empty_cache()
    else:
        capacity_peak = read(
            root() / "state/checks" / ("capacity_" + model_name + ".json")
        )["peak_allocated_mib"]
    torch.cuda.reset_peak_memory_stats()
    measured, callback = measured_trainer(
        Trainer, TrainerCallback, torch, out / "metrics.jsonl"
    )
    recipe = record["parent_recipe"]
    args = TrainingArguments(
        output_dir=str(out / "trainer"),
        max_steps=4,
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=32 // (world * micro_batch),
        learning_rate=recipe["learning_rate"],
        warmup_ratio=recipe["warmup_ratio"],
        weight_decay=recipe["weight_decay"],
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        seed=recipe["seed"],
        data_seed=recipe["seed"],
        max_grad_norm=1.0,
        logging_nan_inf_filter=False,
    )
    trainer = measured(
        model=model,
        args=args,
        train_dataset=samples,
        data_collator=collator(tok),
        processing_class=tok,
        callbacks=[callback()],
    )
    result = trainer.train()
    if not math.isfinite(result.training_loss):
        raise DataError("Nonfinite profiling loss")
    if rank == 0:
        metrics = [
            json.loads(line)
            for line in (out / "metrics.jsonl").read_text().splitlines()
        ]
        # Exclude warmup and final summary rows; compare the identical remaining inputs.
        measured_rows = [
            r for r in metrics if r["stage_global_step"] > 1 and "loss" in r
        ]
        seconds = sum(r["interval_seconds"] for r in measured_rows)
        tokens = sum(
            r["sequence_tokens_per_second"] * r["interval_seconds"]
            for r in measured_rows
        )
        supervised = sum(
            r["supervised_tokens_per_second"] * r["interval_seconds"]
            for r in measured_rows
        )
        write(
            out / "result.json",
            {
                "passed": True,
                "tag": tag,
                "micro_batch": micro_batch,
                "world_size": world,
                "sequence_tokens_per_second": tokens / seconds,
                "supervised_tokens_per_second": supervised / seconds,
                "tokens_per_gpu_second": tokens / (seconds * world),
                "measured_seconds": seconds,
                "capacity_peak_mib": capacity_peak,
                "fixture": binding,
                "optimizer_steps": 4,
                "formal_training_steps": 0,
                "time": time.time(),
                "torch": torch.__version__,
            },
        )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def tune(model_name, milestone):
    out = folder(model_name, milestone)
    if (out / "plan.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    world = config()["training_gpu_count"][model_name]
    trials = []
    for tag, micro_batch in (
        ("baseline_a", 1),
        ("micro_batch_2", 2),
        ("micro_batch_2_repeat", 2),
        ("baseline_b", 1),
    ):
        directory = out / tag
        directory.mkdir(exist_ok=True)
        result_path = directory / "result.json"
        if not result_path.exists():
            # All children share this worker's owned process group and allocated GPU lease.
            command = [
                config()["training_python"],
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={world}",
                str(Path(__file__).resolve().parents[1] / "pipeline.py"),
                "worker",
                "profile",
                "--model",
                model_name,
                "--milestone",
                str(milestone),
                "--micro-batch",
                str(micro_batch),
                "--tag",
                tag,
            ]
            (directory / "metrics.jsonl").unlink(missing_ok=True)
            with (directory / "profile.log").open("a") as log:
                process = subprocess.run(
                    command,
                    check=False,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=3600,
                )
            if process.returncode or not result_path.exists():
                result = {
                    "tag": tag,
                    "micro_batch": micro_batch,
                    "status": "crash",
                    "exit_code": process.returncode,
                    "time": time.time(),
                }
                write(directory / "failure.json", result)
                if micro_batch == 1:
                    raise DataError(
                        "Baseline throughput trial failed; inspect "
                        + str(directory / "profile.log")
                    )
                trials.append(result)
                continue
        result = read(result_path)
        result["status"] = "baseline" if micro_batch == 1 else "diagnostic"
        trials.append(result)
    baselines = [t for t in trials if t["micro_batch"] == 1]
    reference = max(t["tokens_per_gpu_second"] for t in baselines)
    candidates = [t for t in trials if t["micro_batch"] == 2]
    # Repeated timing resolves first-use kernel compilation effects. Require both
    # measurements to pass; a warm-only win remains diagnostic for later work.
    keep = all(
        t.get("passed") and t["tokens_per_gpu_second"] >= reference * 1.10
        for t in candidates
    )
    for candidate in candidates:
        if candidate.get("passed"):
            if any(t["fixture"] != candidate["fixture"] for t in baselines):
                raise DataError("Speed comparison fixture changed")
            candidate["status"] = "keep" if keep else "discard"
    candidate_rate = max(
        (t.get("tokens_per_gpu_second", 0) for t in candidates), default=0
    )
    plan = {
        "micro_batch": 2 if keep else 1,
        "world_size": world,
        "implementation": "existing_sdpa_fla",
        "status": "keep" if keep else "baseline",
        "candidate_gain": candidate_rate / reference - 1,
        "criterion": "10% tokens/GPU-second gain over faster baseline repeat, finite loss/gradients, full 8K with 10% memory headroom",
        "data_and_quality_scope": "128 fixed training inputs, four disposable optimizer steps per trial; no benchmark or study checkpoint mutation",
        "trials": trials,
        "time": time.time(),
    }
    # Keep stage-specific data distributions diagnostic. MB1 remains the formal default
    # until numerical gradient parity and multi-stage stability are established.
    if keep:
        plan["status"] = "diagnostic_win_pending_gradient_parity"
        plan["micro_batch"] = 1
    write(out / "plan.json", plan)
    report(model_name)


def report(model_name):
    plans = sorted((root() / "efficiency" / model_name).glob("*/plan.json"))
    lines = [
        "# 9B 训练效率",
        "",
        "只比较执行效率；固定训练数据、global batch=32、学习率、LoRA、完整序列和评测。四步短试验不计入正式训练。",
        "",
        "| 阶段 | 试验 | 每卡 batch | 状态 | token/s/GPU | 8K 峰值 MiB |",
        "|---|---|---:|---|---:|---:|",
    ]
    ledger = [
        "timestamp\trun_id\tcommit_or_artifact\tmetric_value\tmetric_direction\tstatus\tcost_or_runtime\tmemory_or_gpu\tdescription\toutput_path"
    ]
    for path in plans:
        plan = read(path)
        for trial in plan["trials"]:
            lines.append(
                f"| {path.parent.name} | {trial['tag']} | {trial['micro_batch']} | {trial['status']} | {trial.get('tokens_per_gpu_second', 0):.1f} | {trial.get('capacity_peak_mib', 0):.0f} |"
            )
            ledger.append(
                "\t".join(
                    map(
                        str,
                        [
                            trial["time"],
                            path.parent.name + "_" + trial["tag"],
                            str(path),
                            trial.get("tokens_per_gpu_second", ""),
                            "higher",
                            trial["status"],
                            trial.get("measured_seconds", ""),
                            plan["world_size"],
                            "fixed training fixture; no test selection",
                            str(path.parent / trial["tag"]),
                        ],
                    )
                )
            )
    lines += [
        "",
        "候选需要超过两次 baseline 中较快的一次至少 10%，并通过 8K 显存检查；改变 micro batch 还需梯度数值等价验证。当前正式训练保持 batch=1。",
        "",
    ]
    target = root() / "reports/EFFICIENCY.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines))
    (root() / "efficiency/ledger.tsv").write_text("\n".join(ledger) + "\n")

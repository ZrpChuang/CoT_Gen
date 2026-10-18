"""Memory-mapped token data and explicit LoRA stage continuation."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path

from .common import (DataError, config, execution_scope, evaluation_milestones,
                     native_pack, read, root, sha, write)


def run_record(run_id):
    return next(
        r for r in read(root() / "manifest.json")["runs"] if r["run_id"] == run_id
    )


def validate_history():
    if config().get("initialization") == "fresh_base":
        checked = []
        for record in read(root() / "manifest.json")["runs"]:
            initial_parent(record)
            model = Path(config()["models"][record["model"]])
            index = model / "model.safetensors.index.json"
            weights = (
                set(read(index)["weight_map"].values())
                if index.exists()
                else {"model.safetensors"}
            )
            if not (model / "config.json").is_file() or not all(
                (model / weight).is_file() and (model / weight).stat().st_size > 0
                for weight in weights
            ):
                raise DataError(
                    "Original base weights are incomplete: " + record["model"]
                )
            checked.append(
                {
                    "run_id": record["run_id"],
                    "base_model": str(model),
                    "trained_parent": None,
                }
            )
        write(
            root() / "state/checks/history.json",
            {
                "passed": True,
                "initialization": "fresh_base",
                "runs": checked,
                "time": time.time(),
            },
        )
        return
    import torch
    from safetensors import safe_open

    checked = []
    for record in read(root() / "manifest.json")["runs"]:
        checkpoint = Path(record["parent_checkpoint"])
        if sha(Path(record["parent_data"])) != record["parent_data_sha256"]:
            raise DataError("Historical training data changed: " + record["run_id"])
        meta = read(record["parent_complete"])
        if (
            not completed_checkpoint(checkpoint, meta["world_size"])
            or meta["training_examples"] != 4175
        ):
            raise DataError("Historical checkpoint is incomplete: " + record["run_id"])
        with safe_open(
            str(checkpoint / "adapter_model.safetensors"), framework="pt", device="cpu"
        ) as adapter:
            keys = list(adapter.keys())
            if not keys:
                raise DataError("Empty historical adapter")
            for key in keys:
                if not torch.isfinite(adapter.get_tensor(key)).all():
                    raise DataError("Nonfinite historical adapter: " + record["run_id"])
        checked.append(
            {
                "run_id": record["run_id"],
                "data_sha256": record["parent_data_sha256"],
                "adapter_tensors": len(keys),
            }
        )
    if len(checked) != 19:
        raise DataError("Expected all 19 historical runs")
    write(
        root() / "state/checks/history.json",
        {"passed": True, "runs": checked, "time": time.time()},
    )


def directory(run_id, milestone):
    return root() / "runs" / run_id / str(milestone)


def data_path(variant, milestone):
    from .data import stage_directory

    aliases = {
        "NEU_lr1e5": "NEU",
        "sup_lr1e5": "SUP",
        "aug_sup_lr1e5": "AUG-SUP",
        "ssr_lr1e5": "SSR",
    }
    variant = aliases.get(variant, variant)
    path = stage_directory(milestone)
    if variant in (
        "student_format",
        "matched_random_format",
        "body_format",
        "body_matched_random",
    ):
        path = path / "selected"
    return path / (variant + ".jsonl")


def encoding_directory(run_id, milestone):
    record = run_record(run_id)
    variant = record["variant"]
    variant = {
        "NEU_lr1e5": "NEU",
        "sup_lr1e5": "SUP",
        "aug_sup_lr1e5": "AUG-SUP",
        "ssr_lr1e5": "SSR",
    }.get(variant, variant)
    return root() / "data/tokens" / str(milestone) / record["model"] / variant


def encode(run_id, milestone):
    import pyarrow as pa
    from pyarrow import ipc
    from transformers import AutoTokenizer

    from .common import lock

    cfg = config()
    record = run_record(run_id)
    source = data_path(record["variant"], milestone)
    out = encoding_directory(run_id, milestone)
    out.mkdir(parents=True, exist_ok=True)
    with lock(out / "encode.lock"):
        marker = out / "complete.json"
        fingerprint = sha(source)
        if marker.exists():
            if read(marker)["data_sha256"] != fingerprint or read(marker)[
                "tokens_sha256"
            ] != sha(out / "tokens.arrow"):
                raise DataError("Token cache binding changed")
            return
        tok = AutoTokenizer.from_pretrained(
            cfg["models"][record["model"]], local_files_only=True
        )
        schema = pa.schema(
            [
                ("id", pa.string()),
                ("input_ids", pa.list_(pa.int32())),
                ("source_tokens", pa.int32()),
            ]
        )
        target = out / "tokens.arrow"
        tmp = target.with_suffix(".part")
        n, supervised, sequence, maximum = 0, 0, 0, 0
        ids, batch = set(), []
        with pa.OSFile(str(tmp), "wb") as sink, ipc.new_file(sink, schema) as writer:
            with source.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["id"] in ids:
                        raise DataError("Duplicate training input")
                    ids.add(row["id"])
                    pack = native_pack(tok, row, record["model"])
                    length = len(pack["input_ids"])
                    if (
                        length > cfg["max_sequence_tokens"]
                        or length <= pack["source_tokens"]
                    ):
                        raise DataError(
                            "Invalid complete training sequence; truncation forbidden"
                        )
                    batch.append(
                        {
                            "id": row["id"],
                            "input_ids": pack["input_ids"],
                            "source_tokens": pack["source_tokens"],
                        }
                    )
                    n += 1
                    sequence += length
                    supervised += length - pack["source_tokens"]
                    maximum = max(maximum, length)
                    if len(batch) >= 256:
                        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                        batch = []
                if batch:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        os.replace(tmp, target)
        write(
            marker,
            {
                "complete": True,
                "model": record["model"],
                "rows": n,
                "data_sha256": fingerprint,
                "tokens_sha256": sha(target),
                "ids_sha256": sha(sorted(ids)),
                "supervised_tokens": supervised,
                "sequence_tokens": sequence,
                "maximum_sequence_tokens": maximum,
                "time": time.time(),
            },
        )


def initial_parent(record):
    """A fresh experiment may never resolve an old trained adapter or dataset."""
    if config().get("initialization") == "fresh_base":
        if config()["parent_unique_inputs"] != 0 or any(
            record.get(key)
            for key in (
                "parent_checkpoint",
                "parent_complete",
                "parent_data",
                "parent_data_sha256",
                "parent_global_step",
                "parent_training_examples",
            )
        ):
            raise DataError("Fresh initialization forbids historical training parents")
        return None
    return Path(record["parent_checkpoint"])


def parent(run_id, milestone):
    cfg = config()
    record = run_record(run_id)
    first = initial_parent(record)
    if milestone not in cfg["milestones"]:
        raise DataError("Unknown training milestone")
    previous = cfg["parent_unique_inputs"]
    for value in cfg["milestones"]:
        if value == milestone:
            break
        previous = value
    if previous == cfg["parent_unique_inputs"]:
        return first, record["parent_global_step"], previous
    meta = read(directory(run_id, previous) / "training/complete.json")
    checkpoint = Path(meta["checkpoint"])
    if config().get(
        "initialization"
    ) == "fresh_base" and not checkpoint.resolve().is_relative_to(
        directory(run_id, previous).resolve()
    ):
        raise DataError("Stage parent escaped this fresh experiment")
    if not meta.get("complete") or meta["cumulative_unique_inputs"] != previous:
        raise DataError("Stage parent is incomplete or has a different data scale")
    return checkpoint, meta["cumulative_global_step"], previous


def completed_checkpoint(path, world=None):
    required = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
    )
    try:
        if not all(
            (path / name).is_file() and (path / name).stat().st_size > 0
            for name in required
        ):
            return False
        if world is not None:
            expected = (
                ["rng_state.pth"]
                if world == 1
                else [f"rng_state_{i}.pth" for i in range(world)]
            )
            if not all((path / name).is_file() for name in expected):
                return False
        return read(path / "trainer_state.json")["global_step"] == int(
            path.name.split("-")[-1]
        )
    except (OSError, ValueError, KeyError):
        return False


def load_model(model_name, checkpoint):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration

    cls = (
        Qwen3_5ForCausalLM
        if model_name == "Qwen_38_27B"
        else Qwen3_5ForConditionalGeneration
    )
    source = config()["models"][model_name]
    base = cls.from_pretrained(
        source, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    )
    base.config.use_cache = False
    if hasattr(base.config, "text_config"):
        base.config.text_config.use_cache = False
    if checkpoint is None:
        if config().get("initialization") != "fresh_base":
            raise DataError("Missing parent outside fresh-base experiment")
        recipe = config()["lora_initialization"]
        model = get_peft_model(
            base,
            LoraConfig(
                r=recipe["rank"],
                lora_alpha=recipe["alpha"],
                lora_dropout=recipe["dropout"],
                target_modules=recipe["target_modules"],
                task_type="CAUSAL_LM",
                bias="none",
                init_lora_weights=True,
            ),
        )
    else:
        model = PeftModel.from_pretrained(
            base, str(checkpoint), is_trainable=True, local_files_only=True
        )
    model.enable_input_require_grads()
    return model


def preflight(model_name, capacity=False):
    import torch
    from transformers import AutoTokenizer, set_seed

    marker = (
        root()
        / "state"
        / (("checks/capacity_" if capacity else "preflight_") + model_name + ".json")
    )
    if marker.exists() and read(marker).get("passed"):
        return
    record = next(
        r
        for r in read(root() / "manifest.json")["runs"]
        if r["group"] == "main" and r["variant"] == "NEU" and r["model"] == model_name
    )
    tok = AutoTokenizer.from_pretrained(
        config()["models"][model_name], local_files_only=True
    )
    set_seed(record["parent_recipe"]["seed"])
    model = load_model(model_name, initial_parent(record)).to("cuda")
    fresh_witness = None
    if config().get("initialization") == "fresh_base":
        matrices = [p for name, p in model.named_parameters() if ".lora_B." in name]
        if not matrices or any(torch.count_nonzero(p).item() for p in matrices):
            raise DataError("Fresh LoRA must start with exactly zero B matrices")
        fresh_witness = {"lora_B_matrices": len(matrices), "all_exactly_zero": True}
    fixture = {
        "question": "Compute 2 + 3.",
        "cot": "Add two and three.",
        "answer": "5.",
    }
    pack = native_pack(tok, fixture, model_name)
    if capacity:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.train()
        prefix = pack["input_ids"][: pack["source_tokens"]]
        body = tok.encode(
            "A synthetic capacity fixture repeats a complete explanatory sentence. ",
            add_special_tokens=False,
        )
        length = config()["max_sequence_tokens"] - len(prefix) - 1
        ids = (
            prefix
            + (body * math.ceil(length / len(body)))[:length]
            + [tok.convert_tokens_to_ids("<|im_end|>")]
        )
        pack = {"input_ids": ids, "labels": [-100] * len(prefix) + ids[len(prefix) :]}
        torch.cuda.reset_peak_memory_stats()
    ids = torch.tensor([pack["input_ids"]], device="cuda")
    labels = torch.tensor([pack["labels"]], device="cuda")
    result = model(
        input_ids=ids,
        labels=labels,
        attention_mask=torch.ones_like(ids),
        use_cache=False,
    )
    if not torch.isfinite(result.loss):
        raise DataError("Parent checkpoint produced nonfinite loss")
    result.loss.backward()
    gradients = [
        p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
    ]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise DataError("Parent checkpoint gradients are invalid")
    write(
        marker,
        {
            "passed": True,
            "loss": result.loss.item(),
            "sequence_tokens": len(pack["input_ids"]),
            "parent": record["parent_checkpoint"],
            "base_model": config()["models"][model_name],
            "fresh_initialization_witness": fresh_witness,
            "optimizer_steps": 0,
            "gradient_checkpointing": capacity,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "time": time.time(),
        },
    )


def collator(tok):
    import torch

    def collate(batch):
        width = max(len(row["input_ids"]) for row in batch)
        return {
            key: torch.tensor(
                [
                    row[key]
                    + [
                        (
                            -100
                            if key == "labels"
                            else tok.pad_token_id
                            if key == "input_ids"
                            else 0
                        )
                    ]
                    * (width - len(row[key]))
                    for row in batch
                ],
                dtype=torch.long,
            )
            for key in ("input_ids", "labels", "attention_mask")
        }

    return collate


def train(run_id, milestone):
    import pyarrow as pa
    import torch
    from pyarrow import ipc
    from transformers import (
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    from .efficiency import measured_trainer, runtime_plan

    cfg = config()
    record = run_record(run_id)
    plan = runtime_plan(record["model"], milestone)
    out = directory(run_id, milestone)
    training = out / "training"
    training.mkdir(parents=True, exist_ok=True)
    complete = training / "complete.json"
    encoded = encoding_directory(run_id, milestone)
    data_meta = read(encoded / "complete.json")
    source = data_path(record["variant"], milestone)
    if data_meta["data_sha256"] != sha(source):
        raise DataError("Data changed after encoding")
    if data_meta["tokens_sha256"] != sha(encoded / "tokens.arrow"):
        raise DataError("Encoded training data was corrupted")
    checkpoint, step_offset, previous = parent(run_id, milestone)
    if complete.exists():
        meta = read(complete)
        if (
            meta["data_sha256"] != data_meta["data_sha256"]
            or meta["cumulative_unique_inputs"] != milestone
        ):
            raise DataError("Completion marker does not match this stage")
        return
    recipe = dict(record["parent_recipe"])
    recipe["epochs"] = cfg["stage_epochs"]
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    micro_batch = plan["micro_batch"]
    if plan["world_size"] != world or recipe["global_batch_size"] % (
        world * micro_batch
    ):
        raise DataError("Global batch is not divisible by world size")
    set_seed(recipe["seed"])
    tok = AutoTokenizer.from_pretrained(
        cfg["models"][record["model"]], local_files_only=True
    )
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_model(record["model"], checkpoint)

    class Rows(torch.utils.data.Dataset):
        def __init__(self):
            self.memory = pa.memory_map(str(encoded / "tokens.arrow"), "r")
            self.table = ipc.open_file(self.memory).read_all()

        def __len__(self):
            return self.table.num_rows

        def __getitem__(self, index):
            ids = self.table["input_ids"][index].as_py()
            prefix = self.table["source_tokens"][index].as_py()
            return {
                "input_ids": ids,
                "labels": [-100] * prefix + ids[prefix:],
                "attention_mask": [1] * len(ids),
            }

    args = TrainingArguments(
        output_dir=str(training / "checkpoints"),
        num_train_epochs=recipe["epochs"],
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=recipe["global_batch_size"]
        // (world * micro_batch),
        learning_rate=recipe["learning_rate"],
        warmup_ratio=recipe["warmup_ratio"],
        weight_decay=recipe["weight_decay"],
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        logging_steps=5,
        save_strategy="steps",
        save_steps=(execution_scope() or {}).get("checkpoint_steps", cfg["checkpoint_steps"]),
        save_total_limit=2,
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
    MeasuredTrainer, Metrics = measured_trainer(
        Trainer, TrainerCallback, torch, training / "metrics.jsonl", step_offset
    )
    trainer = MeasuredTrainer(
        model=model,
        args=args,
        train_dataset=Rows(),
        data_collator=collator(tok),
        processing_class=tok,
        callbacks=[Metrics()],
    )
    candidates = sorted(
        (training / "checkpoints").glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    valid = []
    for path in candidates:
        if completed_checkpoint(path, world):
            valid.append(path)
        elif rank == 0:
            path.rename(
                path.with_name("incomplete-" + path.name + "-" + str(time.time_ns()))
            )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if rank == 0:
        write(
            training / "stage.json",
            {
                "run_id": run_id,
                "model": record["model"],
                "milestone": milestone,
                "parent_checkpoint": str(checkpoint) if checkpoint else None,
                "base_model": cfg["models"][record["model"]],
                "initialization": cfg.get("initialization", "historical_adapter"),
                "parent_cumulative_steps": step_offset,
                "previous_unique_inputs": previous,
                "new_unique_inputs": data_meta["rows"],
                "data_sha256": data_meta["data_sha256"],
                "recipe": recipe,
                "execution_plan": plan,
                "optimizer_policy": cfg["stage_optimizer_policy"],
                "world_size": world,
                "resume_checkpoint": str(valid[-1]) if valid else None,
            },
        )
    binding = training / "execution_plan.json"
    if binding.exists() and read(binding) != plan:
        raise DataError("Execution plan changed within a resumable stage")
    if rank == 0:
        write(binding, plan)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    started = time.time()
    result = trainer.train(resume_from_checkpoint=str(valid[-1]) if valid else None)
    if not math.isfinite(result.training_loss):
        raise DataError("Nonfinite training loss")
    trainer._save_checkpoint(trainer.model, trial=None)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if rank == 0:
        final = (
            training / "checkpoints" / ("checkpoint-" + str(trainer.state.global_step))
        )
        if not completed_checkpoint(final, world):
            raise DataError("Final checkpoint is incomplete")
        from safetensors.torch import load_file

        tensors = load_file(str(final / "adapter_model.safetensors"))
        if not tensors or not all(torch.isfinite(t).all() for t in tensors.values()):
            raise DataError("Adapter contains nonfinite tensors")
        if data_meta["rows"] != milestone - previous:
            raise DataError("Incorrect stage input count")
        meta = {
            "complete": True,
            "run_id": run_id,
            "model": record["model"],
            "variant": record["variant"],
            "execution_plan": plan,
            "stage_global_step": trainer.state.global_step,
            "cumulative_global_step": step_offset + trainer.state.global_step,
            "cumulative_unique_inputs": milestone,
            "new_unique_inputs": data_meta["rows"],
            "recipe": recipe,
            "parent_checkpoint": str(checkpoint) if checkpoint else None,
            "base_model": cfg["models"][record["model"]],
            "initialization": cfg.get("initialization", "historical_adapter"),
            "checkpoint": str(final),
            "world_size": world,
            "data_sha256": data_meta["data_sha256"],
            "supervised_tokens": data_meta["supervised_tokens"],
            "sequence_tokens": data_meta["sequence_tokens"],
            "optimizer_policy": cfg["stage_optimizer_policy"],
            "loss_current_train_call": result.training_loss,
            "runtime_current_train_call_seconds": time.time() - started,
            "peak_allocated_mib_rank0": torch.cuda.max_memory_allocated() / 2**20,
            "distributed_sampler_examples_per_epoch": math.ceil(
                data_meta["rows"] / world
            )
            * world,
            "resume_occurred": bool(valid),
            "runtime_scope": "Current train call includes resume only; attempts and metrics preserve history.",
            "time": time.time(),
        }
        if milestone in evaluation_milestones():
            unwrapped = trainer.accelerator.unwrap_model(trainer.model)
            merged = unwrapped.merge_and_unload()
            target = out / "model"
            merged.save_pretrained(
                target, safe_serialization=True, max_shard_size="5GB"
            )
            tok.save_pretrained(target)
            original = Path(cfg["models"][record["model"]])
            if (original / "generation_config.json").exists():
                shutil.copy2(
                    original / "generation_config.json",
                    target / "generation_config.json",
                )
            meta["merged_model"] = str(target)
        write(complete, meta)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def sync_wandb(run_id, milestone):
    import wandb

    from .common import load_wandb_env

    load_wandb_env()
    if not os.environ.get("WANDB_API_KEY"):
        from .common import Waiting

        raise Waiting("W&B credentials unavailable")
    cfg = config()
    out = directory(run_id, milestone)
    training = out / "training"
    training.mkdir(parents=True, exist_ok=True)
    meta = run_record(run_id)
    run = wandb.init(
        project=cfg["wandb_project"],
        entity=cfg["wandb_entity"],
        id=sha(cfg["run_id"] + run_id + str(milestone))[:20],
        resume="allow",
        name=run_id + "-" + str(milestone),
        group=cfg["run_id"],
        job_type="gpt-fresh-training"
        if cfg.get("initialization") == "fresh_base"
        else "scale-continuation",
        dir=str(out),
        config={
            "run_id": run_id,
            "milestone": milestone,
            "parent": meta["parent_checkpoint"],
            "base_model": cfg["models"][meta["model"]],
            "initialization": cfg.get("initialization", "historical_adapter"),
            "model": meta["model"],
            "variant": meta["variant"],
            "training_recipe": meta["parent_recipe"],
            "stage_policy": cfg["stage_optimizer_policy"],
        },
        settings=wandb.Settings(init_timeout=90, disable_git=True, disable_code=True),
    )
    state_path = training / "wandb_state.json"
    finished = False
    last = read(state_path).get("last_global_step", -1) if state_path.exists() else -1
    # Online history is the authority when recovering after upload/marker races.
    if run.resumed:
        last = max(last, int(run.summary.get("last_uploaded_cumulative_step", -1)))
    try:
        while True:
            path = training / "metrics.jsonl"
            records = []
            if path.exists():
                for line in path.read_text().splitlines():
                    try:
                        records.append(json.loads(line))
                    except ValueError:
                        pass  # Only an incomplete live tail can be skipped.
            for row in records:
                step = row["cumulative_global_step"]
                if step <= last:
                    continue
                run.log(
                    {
                        "train/" + k: v
                        for k, v in row.items()
                        if isinstance(v, (int, float))
                    }
                )
                run.summary["last_uploaded_cumulative_step"] = step
                last = step
                write(
                    state_path,
                    {
                        "status": "syncing",
                        "last_global_step": last,
                        "url": run.url,
                        "time": time.time(),
                    },
                )
            complete = training / "complete.json"
            if complete.exists():
                result = read(complete)
                run.summary.update(result)
                run.summary["training_data_sha256"] = result["data_sha256"]
                run.summary["training_complete"] = True
                run.finish()
                finished = True
                # Check cloud summary instead of equating a local finish with a cloud success.
                remote = wandb.Api(timeout=60).run(
                    f"{cfg['wandb_entity']}/{cfg['wandb_project']}/{sha(cfg['run_id'] + run_id + str(milestone))[:20]}"
                )
                if (
                    remote.summary.get("training_data_sha256") != result["data_sha256"]
                    or remote.summary.get("cumulative_global_step")
                    != result["cumulative_global_step"]
                ):
                    raise RuntimeError("W&B cloud summary not yet complete")
                write(
                    state_path,
                    {
                        "status": "synced_complete",
                        "cloud_verified": True,
                        "last_global_step": last,
                        "final_cumulative_step": result["cumulative_global_step"],
                        "url": remote.url,
                        "time": time.time(),
                    },
                )
                return
            if (training / "terminal_failure.json").exists():
                run.summary["training_complete"] = False
                run.finish(exit_code=1)
                finished = True
                return
            time.sleep(30)
    finally:
        if not finished:
            run.finish(exit_code=1)

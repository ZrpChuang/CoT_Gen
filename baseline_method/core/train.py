#!/usr/bin/env python3
"""BF16 text-decoder LoRA; one common final-epoch recipe for every method."""

import argparse
import hashlib
import json
import os
import shutil
import time

import torch
from encoding import encode
from peft import LoraConfig, TaskType, get_peft_model
from settings import BASE_DIR as BASE
from settings import METHODS, MODELS, experiment, model_path
from torch import nn
from transformers import (
    AutoTokenizer,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)


class Rows(torch.utils.data.Dataset):
    def __init__(self, rows, tok, model, max_length):
        self.rows = []
        for row in rows:
            ids, labels = encode(tok, row["question"], row["cot"], row["answer"], model)
            if len(ids) > max_length or all(value == -100 for value in labels):
                raise ValueError(f"Invalid encoded training row: {row['id']}")
            self.rows.append(
                {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--model", choices=MODELS, required=True)
    a = ap.parse_args()
    ROOT = BASE / a.method
    cfg = experiment()["training"]
    source = model_path(a.model)
    if not source.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {source}")
    set_seed(cfg["seed"])
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 1 or cfg["global_batch_size"] % world:
        raise ValueError("Global batch size must be divisible by the process count")
    out = ROOT / a.model / "training"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        return
    rows = [json.loads(line) for line in (ROOT / "train.jsonl").open()]
    if not rows:
        raise ValueError(f"No training rows in {ROOT / 'train.jsonl'}")
    tok = AutoTokenizer.from_pretrained(source, local_files_only=True)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cls = (
        Qwen3_5ForCausalLM
        if a.model == "Qwen_38_27B"
        else Qwen3_5ForConditionalGeneration
    )
    model = cls.from_pretrained(
        source,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and (
            name.startswith("model.layers.")
            or name.startswith("model.language_model.layers.")
        )
    ]
    assert len(targets) > 20, targets
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg["rank"],
            lora_alpha=cfg["alpha"],
            lora_dropout=cfg["dropout"],
            target_modules=targets,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    model.enable_input_require_grads()

    def collate(batch):
        n = max(len(x["input_ids"]) for x in batch)
        return {
            k: torch.tensor(
                [
                    x[k]
                    + [
                        (
                            -100
                            if k == "labels"
                            else tok.pad_token_id
                            if k == "input_ids"
                            else 0
                        )
                    ]
                    * (n - len(x[k]))
                    for x in batch
                ],
                dtype=torch.long,
            )
            for k in ["input_ids", "labels", "attention_mask"]
        }

    args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg["global_batch_size"] // world,
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        logging_steps=5,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=1,
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        max_grad_norm=1.0,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Rows(rows, tok, a.model, cfg["max_seq_length"]),
        data_collator=collate,
        processing_class=tok,
    )
    checkpoints = sorted(
        (out / "checkpoints").glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    start = time.time()
    result = trainer.train(
        resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None
    )
    if rank == 0:
        assert torch.isfinite(torch.tensor(result.training_loss)), result
        meta = {
            "model": a.model,
            "method": ROOT.name,
            "training_examples": len(rows),
            "runtime_seconds": time.time() - start,
            "loss": result.training_loss,
            "global_step": result.global_step,
            "recipe": cfg,
            "world_size": world,
            "data_sha256": hashlib.sha256(
                (ROOT / "train.jsonl").read_bytes()
            ).hexdigest(),
        }
        adapter = ROOT / a.model / "adapter"
        trainer.save_model(str(adapter))
        tok.save_pretrained(adapter)
        # vLLM evaluates the merged weights, not the adapter directory.
        trained = trainer.accelerator.unwrap_model(trainer.model)
        merged = trained.merge_and_unload()
        target = ROOT / a.model / "model"
        merged.save_pretrained(target, safe_serialization=True, max_shard_size="5GB")
        tok.save_pretrained(target)
        if (source / "generation_config.json").exists():
            shutil.copy2(
                source / "generation_config.json", target / "generation_config.json"
            )
        (out / "complete.json").write_text(json.dumps(meta, indent=2) + "\n")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

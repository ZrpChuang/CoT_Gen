#!/usr/bin/env python3
"""Render this review from preserved training logs and CPU audit metadata."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ["NEU", "SUP", "AUG-SUP", "SSR"]
COLORS = {"BASE": "#273449", "NEU": "#2563eb", "SUP": "#ea580c",
          "AUG-SUP": "#7c3aed", "SSR": "#059669"}


def read(path):
    return json.loads(path.read_text())


def save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def main(root):
    evidence = root / "evidence"
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    logs = read(evidence / "training/metrics.json")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.grid": True,
                         "grid.alpha": .18, "axes.axisbelow": True})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    fields = [("loss", "Training loss", "5-update mean cross entropy"),
              ("learning_rate", "Learning-rate schedule", "Learning rate"),
              ("grad_norm", "Gradient norm", "Reported norm before clipping"),
              ("sequence_tokens_per_second", "Training throughput", "Unpadded tokens / second (two GPUs)")]
    for ax, (field, title, ylabel) in zip(axes.flat, fields):
        for method in METHODS:
            rows = [r for r in logs[method] if "loss" in r and field in r]
            ax.plot([r["stage_global_step"] for r in rows], [r[field] for r in rows],
                    label=method, color=COLORS[method], linewidth=1.6, alpha=.85)
        ax.set(title=title, xlabel="Optimizer step", ylabel=ylabel, xlim=(0, 313))
        if field == "learning_rate":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if field == "grad_norm":
            ax.axhline(1, color="#64748b", linestyle="--", linewidth=1, label="Clip threshold")
            ax.set_ylim(0, 1.08)
    axes[0, 0].legend(ncol=2, frameon=False)
    fig.suptitle("Fresh 10K training | Qwen3.5-9B | 2026-10-18", fontsize=16)
    fig.supxlabel("62 loss records per method, steps 5-310. The step-313 overall train_loss is not a final-step loss.", fontsize=9)
    save(fig, figures / "training_curves")

    comparison = read(evidence / "checks/evaluation_length_comparison.json")
    benches = ["ifeval", "multi_challenge", "eq_bench3", "arena_hard_v2"]
    labels = ["IFEval", "MultiChallenge", "EQ-Bench 3", "ArenaHard"]
    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
    x = np.arange(4)
    width = .15
    for j, method in enumerate(["BASE"] + METHODS):
        values = [comparison[method][b]["reasoning_median_bytes"] for b in benches]
        bars = ax.bar(x + (j - 2) * width, values, width, color=COLORS[method], label=method)
        ax.bar_label(bars, labels=[f"{v:,.0f}" for v in values], fontsize=7, padding=3)
    ax.set(xticks=x, xticklabels=labels, ylabel="Median reasoning length (UTF-8 bytes; log scale)",
           title="Reasoning became much shorter on the same evaluation inputs")
    ax.set_yscale("log")
    ax.set_ylim(160, 35000)
    ax.legend(ncol=5, loc="upper right", frameon=False)
    fig.supxlabel("Observed behavior, not a causal ablation. Counts measure bytes, not tokens. EQ counts user-turn outputs.", fontsize=9)
    save(fig, figures / "reasoning_length")

    stats = read(evidence / "checks/data_statistics.json")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    lang = stats["language"]
    chosen = sorted(lang, key=lang.get, reverse=True)[:5]
    values = [lang[k] for k in chosen] + [10000 - sum(lang[k] for k in chosen)]
    bars = axes[0].bar(chosen + ["rest"], values, color="#2563eb")
    axes[0].bar_label(bars, fontsize=8)
    axes[0].set(title="Input-language metadata", ylabel="Examples", ylim=(0, 6200))
    domain = stats["stratum"]
    names = ["other", "code", "creative_writing", "instruction_following", "math"]
    bars = axes[1].barh(["Other", "Code", "Creative", "Instruction", "Math"],
                        [domain[n] for n in names], color="#7c3aed")
    axes[1].bar_label(bars, fontsize=8, padding=3)
    axes[1].invert_yaxis()
    axes[1].set(title="Inherited source-domain labels", xlabel="Examples", xlim=(0, 6500))
    turns = stats["user_turns"]
    values = [turns.get("1", 0), turns.get("2", 0), sum(v for k, v in turns.items() if int(k) >= 3)]
    bars = axes[2].bar(["1 user turn", "2 turns", "3+ turns"], values, color="#059669")
    axes[2].bar_label(bars, fontsize=8)
    axes[2].set(title="Conversation depth", ylabel="Examples", ylim=(0, 10500))
    fig.suptitle("Actual accepted training set | 10,000 unique inputs", fontsize=15)
    fig.supxlabel("Source tags are coarse metadata, not validated benchmark-specific skills.", fontsize=9)
    save(fig, figures / "data_distribution")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    main(parser.parse_args().root)

#!/usr/bin/env python3
import hashlib
import json
from collections import Counter

import build_data as B
from encoding import encode
from settings import BASE_DIR as BASE
from settings import DATA_DIR, METHODS, MODELS, experiment, model_path
from settings import PROJECT_DIR as ROOT


def main():
    from transformers import AutoTokenizer

    max_length = experiment()["training"]["max_seq_length"]
    qs = B.questions()
    answers = B.load(DATA_DIR / "reference_answers.jsonl")
    traces = {m: B.load(BASE / m / "cot.jsonl") for m in METHODS}
    tokenizers = {
        m: AutoTokenizer.from_pretrained(model_path(m), local_files_only=True)
        for m in MODELS
    }
    accepted = []
    reasons = Counter()
    lengths = {}
    for q in qs:
        rid = q["id"]
        if rid not in answers or any(rid not in traces[m] for m in METHODS):
            reasons["missing_reference_or_method"] += 1
            continue
        answer = answers[rid]["answer"]
        valid = True
        if any(
            t in q["question"] + answer
            for t in ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
        ):
            reasons["chat_control_token_in_source"] += 1
            continue
        for method in METHODS:
            assert traces[method][rid]["answer_sha256"] == answers[rid]["answer_sha256"]
            for model, tok in tokenizers.items():
                ids, labels = encode(
                    tok, q["question"], traces[method][rid]["cot"], answer, model
                )
                if len(ids) > max_length:
                    valid = False
                lengths[(rid, method, model)] = len(ids)
        if not valid:
            reasons[f"over_{max_length}_in_any_method_or_model"] += 1
            continue
        accepted.append(q)
    counts = Counter(q["stratum"] for q in accepted)
    per = min(
        counts.get(g, 0)
        for g in ["math", "creative_writing", "instruction_following", "code", "other"]
    )
    assert per >= experiment().get("minimum_usable_per_stratum", 900), (
        "Too few paired examples; refusing a biased/undersized training run",
        dict(counts),
        dict(reasons),
    )
    used = Counter()
    final = []
    for q in accepted:
        if used[q["stratum"]] < per:
            final.append(q)
            used[q["stratum"]] += 1
    index_path = ROOT / "training_data/sample/used_train_indices.jsonl"
    index_rows = [{k: v for k, v in q.items() if k != "question"} for q in final]
    text = "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in index_rows)
    if index_path.exists():
        assert index_path.read_text() == text, "Frozen used split changed"
    else:
        index_path.write_text(text)
    for method in METHODS:
        rows = [
            {
                "id": q["id"],
                "stratum": q["stratum"],
                "question": q["question"],
                "cot": traces[method][q["id"]]["cot"],
                "answer": answers[q["id"]]["answer"],
            }
            for q in final
        ]
        p = BASE / method / "train.jsonl"
        raw = "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in rows)
        if p.exists():
            assert p.read_text() == raw, "Refusing to change prepared training data"
        else:
            p.write_text(raw)
        (BASE / method / "data_manifest.json").write_text(
            json.dumps(
                {
                    "n": len(rows),
                    "strata": dict(used),
                    "sha256": hashlib.sha256(raw.encode()).hexdigest(),
                    "shared_used_index_sha256": hashlib.sha256(
                        text.encode()
                    ).hexdigest(),
                    "no_sequence_truncation": True,
                    "reference_source": "Qwen3.8-Max final answers, not verified gold",
                    "max_sequence_tokens": {
                        model: max(lengths[(q["id"], method, model)] for q in final)
                        for model in MODELS
                    },
                },
                indent=2,
            )
        )
    summary = {
        "complete": True,
        "n": len(final),
        "per_stratum": per,
        "candidate_n": len(qs),
        "excluded": dict(reasons),
        "dev_reserved_n": 1000,
        "filter": "Common complete subset across all 4 methods and both model tokenizers; rebalance strata; never truncate targets.",
    }
    (DATA_DIR / "prepared_status.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

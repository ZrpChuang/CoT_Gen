# Dataset inventory

Paths below are relative to the CoT_Analyse workspace root.

| Purpose | Dataset | Rows or scenarios | Local source |
|---|---|---:|---|
| Evaluation | Arena-Hard-v2 | 750 prompts | `test_data/arena_hard_v2/data/arena-hard-v2.0/question.jsonl` |
| Evaluation | EQ-Bench3 | 46 titles (45 nonempty) | `test_data/eq_bench3/data/scenario_prompts.txt` |
| Evaluation | MultiChallenge | 273 | `test_data/multi_challenge/data/benchmark_questions.jsonl` |
| Evaluation | IFEval | 541 | `test_data/ifeval/instruction_following_eval/data/input_data.jsonl` |
| Training source | LMArena 140k | 135,634 | `training_data/lmarena_140k/data/train-*.parquet` |

LMArena is a source of user conversations, not a ready-made 100k-example paper SFT corpus. The baseline pilot used fixed candidate indices from `training_data/sample/train_indices.jsonl` and then selected 4,175 paired examples shared by all four methods and both model tokenizers. Its final used indices are in `training_data/sample/used_train_indices.jsonl`. The four evaluation datasets were held out from training, and exact-match and shingle overlap checks against test user turns are documented in the sample manifest; they do not rule out semantic overlap.

Arena-Hard-v2 has 500 hard prompts and 250 creative-writing prompts. EQ-Bench3 has 46 titles (45 nonempty) and 92 prompt segments. The local scoring protocols and frozen evaluator snapshots are documented with the results; local EQ and Arena numbers are not official leaderboard scores. Each dataset's `DATASET_MANIFEST.json` records the source, version, file SHA256, and verification result. `sources.lock.json` records source references. The source paper did not provide exact commit IDs for every snapshot, so these public snapshots cannot be claimed byte-identical to those used in the paper.

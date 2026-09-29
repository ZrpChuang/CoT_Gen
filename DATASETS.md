# 数据目录

| 用途 | 数据 | 数量 | 原始数据入口 |
|---|---|---:|---|
| 测试 | arena_hard_v2 | 750 | `test_data/arena_hard_v2/data/arena-hard-v2.0/question.jsonl` |
| 测试 | eq_bench3 | 46 | `test_data/eq_bench3/data/scenario_prompts.txt` |
| 测试 | multi_challenge | 273 | `test_data/multi_challenge/data/benchmark_questions.jsonl` |
| 测试 | ifeval | 541 | `test_data/ifeval/instruction_following_eval/data/input_data.jsonl` |
| 训练原料 | lmarena_140k | 135634 | `training_data/lmarena_140k/data/train-*.parquet` |

- LMArena 是原始用户对话，并非论文已经生成好答案和 CoT 的 100k SFT 数据。尚未抽样。
- ArenaHard v2 包含 500 道 hard prompts 和 250 道 creative writing；使用哪个子集在正式评测时固定。
- EQ-Bench3 是 46 个场景、92 个 prompt 段，保留官方多轮场景文件、rubric、评审提示和排行榜参照。
- 官方评测脚本与默认配置一并保存，但尚未安装全部评测依赖或调用裁判模型。
- 本次固定的是当前公开快照。原论文没有给出精确 commit，不能声称与原论文版本逐字一致。
- 每个数据目录中的 DATASET_MANIFEST.json 记录来源、版本、文件 SHA256 和验证结果。
- 四个测试集只用于评测，后续训练采样需执行与测试集的重合检查。

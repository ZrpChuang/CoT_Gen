# 原始模型基座评测

总成绩和评测口径见 Base_model_result_conclusion.md。五个模型分别位于 GPT/、Qwen_Max/、Qwen_35_4B/、Qwen_35_9B/、Qwen_38_27B/；每个模型下有 ifeval/、multi_challenge/、eq_bench3/、arena_hard_v2/，其中 metrics.json 为分数，responses.jsonl 为逐题回答，judgments*.jsonl 为评分。audit.json 为完整性检查。

各模型目录里的 evaluate.py、verify_results.py 和配置是评测时冻结的源码快照，用于核查已有结果。先阅读对应模型的 README.md 和 config.json，再执行评测。单次使用的 benchmark/run.sh 包装脚本已清理；直接调用 python evaluate.py --bench <benchmark> --stage generate 或 --stage score。

analysis/ 只保留 effort 与 token 使用量分析的产物；本地一次性分析脚本已清理。Qwen_38_27B/effort_low/ 和 effort_xhigh/ 为独立采样设置的结果，不混入五模型主表。

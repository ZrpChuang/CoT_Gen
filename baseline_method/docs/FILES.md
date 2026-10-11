# baseline 逐文件用途与取舍

按整理前的 **270 个普通文件逐项枚举**，同时列出整理后位置；重复结构也保留每一条路径。路径相对 baseline_method/。

P0=核心结论、运行代码与唯一实验证据；P1=复跑配置、来源和诊断；P2=历史过程；P3=确认冗余。全工作区112,622个文件的同格式清单在 FULL_INVENTORY.tsv.gz；自动分类不等同于逐行源码审查。

## 最重要的阅读顺序

1. docs/RESULTS.md：总表、完整方法名称、131步训练设置和诊断。
2. pipeline.py + core/：长期复用的生成、准备、训练、评测与汇总代码。
3. 各方法 cot.jsonl / train.jsonl / data_manifest.json。
4. 各 benchmark / 模型目录：原始回答、裁判记录、指标和实际生成配置。

## 核心、配置与共享数据

### 公共文件

| 原文件 | 现在放哪里 | 用途 | 优先级 / 处理 |
|---|---|---|---|
| `DIAGNOSTICS.md` | `docs/RESULTS.md` | 训练与回答长度诊断，合入统一结论 | P1 / 合并说明后删除旧件 |
| `README.md` | `README.md` | 阅读入口、目录约定和复跑命令 | P0 / 保留 |
| `RESULTS.md` | `docs/RESULTS.md` | 总表外的实验配置、样本说明与学习率对照，合入统一结论 | P0 / 合并说明后删除旧件 |
| `RESULTS_BY_BENCHMARK.md` | `docs/RESULTS.md` | 四benchmark×两模型的Baseline和四方法总表，合入统一结论 | P0 / 合并说明后删除旧件 |
| `build_data.py` | `core/build_data.py` | 调用教师生成共同参考答案和四方法逆向CoT；记录来源、格式校验与断点 | P0 / 移动 |
| `encoding.py` | `core/encoding.py` | 构造训练token和仅对回答部分计算损失的labels | P0 / 移动 |
| `experiment.example.json` | `configs/experiment.example.json` | 公开可复用的实验配置模板 | P1 / 移动 |
| `experiment.json` | `configs/experiment.json` | 当前真实教师和LoRA实验配置，Git忽略 | P0 / 移动 |
| `local_paths.example.json` | `configs/local_paths.example.json` | 本机环境配置模板 | P1 / 移动 |
| `local_paths.json` | `configs/local_paths.json` | 本机模型、Python环境与GPU映射，Git忽略 | P1 / 移动 |
| `metrics.py` | `core/metrics.py` | 提取各benchmark已完成指标，拒绝将未完成结果当成绩 | P0 / 移动 |
| `pipeline.py` | `pipeline.py` | 统一调度：校验→生成→配对准备→训练→评测→总表；按完成标记续跑 | P0 / 保留 |
| `prepare_training.py` | `core/prepare_training.py` | 取四方法、两模型共同有效样本，均衡配对并写训练集和样本清单 | P0 / 移动 |
| `pyproject.toml` | `configs/pyproject.toml` | Ruff静态检查与格式规范配置 | P1 / 移动 |
| `requirements-training.txt` | `configs/requirements-training.txt` | 长期训练流水线的依赖列表 | P1 / 移动 |
| `run_evaluation.py` | `core/run_evaluation.py` | 调度冻结服务与四个benchmark生成、评分、审计 | P0 / 移动 |
| `settings.py` | `core/settings.py` | 集中定义目录、方法、模型、benchmark及机器配置读取 | P0 / 移动 |
| `summarize.py` | `core/summarize.py` | 从已完成metrics与训练记录生成总表 | P0 / 移动 |
| `trace_format.py` | `core/trace_format.py` | 解析与验证四方法CoT及骨架格式 | P0 / 移动 |
| `train.py` | `core/train.py` | 按固定BF16 LoRA配置训练，保存adapter、合并权重与完成记录 | P0 / 移动 |
| `data/generation_status.json` | `data/generation_status.json` | 原始候选生成状态；不等同于冻结训练集是否完成 | P1 / 保留 |
| `data/prepared_status.json` | `data/prepared_status.json` | 冻结配对训练集完成标记与最终数量 | P0 / 保留 |
| `data/reference_answers.jsonl` | `data/reference_answers.jsonl` | 教师共同参考答案；四方法共享的生成原料 | P0 / 保留 |
| `logs/INDEX.md` | `data/logs/INDEX.md` | 压缩日志索引和阅读方法 | P1 / 移动 |
| `logs/ledger.tsv` | `data/logs/ledger.tsv` | 流水线阶段执行事件表 | P1 / 移动 |
| `logs/report.log` | `docs/RESULTS.md` | 只记录派生报告路径的重复短日志，删除 | P3 / 删除派生重复件 |
| `logs/run-logs.tar.gz` | `data/logs/run-logs.tar.gz` | 142份有内容的历史日志与内部SHA256清单 | P1 / 移动 |
| `logs/status.json` | `data/logs/status.json` | 最近一次流水线状态 | P1 / 移动 |
| `logs/summary.generated.md` | `docs/RESULTS.md` | 可由metrics重新生成的重复总表，删除 | P3 / 删除派生重复件 |
| `prompts/AUG-SUP.txt` | `configs/prompts/AUG-SUP.txt` | 该方法生成逆向CoT的系统提示词 | P0 / 移动 |
| `prompts/NEU.txt` | `configs/prompts/NEU.txt` | 该方法生成逆向CoT的系统提示词 | P0 / 移动 |
| `prompts/SSR.txt` | `configs/prompts/SSR.txt` | 该方法生成逆向CoT的系统提示词 | P0 / 移动 |
| `prompts/SUP.txt` | `configs/prompts/SUP.txt` | 该方法生成逆向CoT的系统提示词 | P0 / 移动 |
| `tests/test_core.py` | `core/tests/test_core.py` | 验证训练mask、CoT格式、完成指标读取三处关键行为 | P0 / 移动 |

### NEU

| 原文件 | 现在放哪里 | 用途 | 优先级 / 处理 |
|---|---|---|---|
| `NEU/cot.jsonl` | `NEU/cot.jsonl` | 该方法生成的逆向CoT样本及生成来源 | P0 / 保留 |
| `NEU/data_manifest.json` | `NEU/data_manifest.json` | 该方法配对训练集的数量、选择规则与哈希 | P0 / 保留 |
| `NEU/train.jsonl` | `NEU/train.jsonl` | 实际用于两模型训练的4175条共同配对样本 | P0 / 保留 |
| `NEU/Qwen_35_9B/evaluation/audit.json` | `NEU/Qwen_35_9B/evaluation/audit.json` | 该模型四benchmark输出的完整性审计 | P0 / 保留 |
| `NEU/Qwen_35_9B/evaluation/config.json` | `NEU/Qwen_35_9B/evaluation/config.json` | 冻结评测配置：目标模型、裁判与benchmark协议 | P0 / 保留 |
| `NEU/Qwen_35_9B/evaluation/deployment.json` | `NEU/Qwen_35_9B/evaluation/deployment.json` | 该次模型服务路径、端口与运行环境 | P1 / 保留 |
| `NEU/Qwen_35_9B/evaluation/evaluate.py` | `NEU/Qwen_35_9B/evaluation/evaluate.py` | 随该次实验冻结的生成、评分与API传输实现；需review后按哈希去重 | P0 / 保留；R8 |
| `NEU/Qwen_35_9B/evaluation/evaluation_sources.sha256.json` | `configs/evaluation_sources.sha256.json` | 固定测试数据与评测源码SHA256；8份字节相同，集中保留1份 | P0 / 相同副本合并 |
| `NEU/Qwen_35_9B/evaluation/serve.py` | `NEU/Qwen_35_9B/evaluation/serve.py` | 随该次实验冻结的模型服务启动/停止实现；需review后按哈希去重 | P1 / 保留；R8 |
| `NEU/Qwen_35_9B/evaluation/verify_results.py` | `NEU/Qwen_35_9B/evaluation/verify_results.py` | 冻结的逐题完整性核验实现，产生audit.json | P0 / 保留；R8 |
| `NEU/Qwen_35_9B/evaluation/runtime/service_history.json` | `data/logs/service-history.tar.gz#NEU/Qwen_35_9B/evaluation/runtime/service_history.json` | 历史服务启动/停止状态；已归档到统一日志目录 | P1 / 哈希核对后归档 |
| `NEU/Qwen_35_9B/training/complete.json` | `NEU/Qwen_35_9B/training/complete.json` | 训练完成记录：131步、训练参数、输入与产物信息 | P0 / 保留 |
| `NEU/Qwen_38_27B/evaluation/audit.json` | `NEU/Qwen_38_27B/evaluation/audit.json` | 该模型四benchmark输出的完整性审计 | P0 / 保留 |
| `NEU/Qwen_38_27B/evaluation/config.json` | `NEU/Qwen_38_27B/evaluation/config.json` | 冻结评测配置：目标模型、裁判与benchmark协议 | P0 / 保留 |
| `NEU/Qwen_38_27B/evaluation/deployment.json` | `NEU/Qwen_38_27B/evaluation/deployment.json` | 该次模型服务路径、端口与运行环境 | P1 / 保留 |
| `NEU/Qwen_38_27B/evaluation/evaluate.py` | `NEU/Qwen_38_27B/evaluation/evaluate.py` | 随该次实验冻结的生成、评分与API传输实现；需review后按哈希去重 | P0 / 保留；R8 |
| `NEU/Qwen_38_27B/evaluation/evaluation_sources.sha256.json` | `configs/evaluation_sources.sha256.json` | 固定测试数据与评测源码SHA256；8份字节相同，集中保留1份 | P0 / 相同副本合并 |
| `NEU/Qwen_38_27B/evaluation/serve.py` | `NEU/Qwen_38_27B/evaluation/serve.py` | 随该次实验冻结的模型服务启动/停止实现；需review后按哈希去重 | P1 / 保留；R8 |
| `NEU/Qwen_38_27B/evaluation/verify_results.py` | `NEU/Qwen_38_27B/evaluation/verify_results.py` | 冻结的逐题完整性核验实现，产生audit.json | P0 / 保留；R8 |
| `NEU/Qwen_38_27B/evaluation/runtime/service_history.json` | `data/logs/service-history.tar.gz#NEU/Qwen_38_27B/evaluation/runtime/service_history.json` | 历史服务启动/停止状态；已归档到统一日志目录 | P1 / 哈希核对后归档 |
| `NEU/Qwen_38_27B/training/complete.json` | `NEU/Qwen_38_27B/training/complete.json` | 训练完成记录：131步、训练参数、输入与产物信息 | P0 / 保留 |
| `NEU/arena_hard_v2/Qwen_35_9B/generation_config.json` | `NEU/arena_hard_v2/Qwen_35_9B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `NEU/arena_hard_v2/Qwen_35_9B/judge_0_config.json` | `NEU/arena_hard_v2/Qwen_35_9B/judge_0_config.json` | 该次benchmark裁判配置 | P1 / 保留 |
| `NEU/arena_hard_v2/Qwen_35_9B/judgments_0.jsonl` | `NEU/arena_hard_v2/Qwen_35_9B/judgments_0.jsonl` | 裁判逐题原始评分及判定依据 | P0 / 保留 |
| `NEU/arena_hard_v2/Qwen_35_9B/metrics.json` | `NEU/arena_hard_v2/Qwen_35_9B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `NEU/arena_hard_v2/Qwen_35_9B/responses.jsonl` | `NEU/arena_hard_v2/Qwen_35_9B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `NEU/arena_hard_v2/Qwen_38_27B/generation_config.json` | `NEU/arena_hard_v2/Qwen_38_27B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `NEU/arena_hard_v2/Qwen_38_27B/judge_0_config.json` | `NEU/arena_hard_v2/Qwen_38_27B/judge_0_config.json` | 该次benchmark裁判配置 | P1 / 保留 |
| `NEU/arena_hard_v2/Qwen_38_27B/judgments_0.jsonl` | `NEU/arena_hard_v2/Qwen_38_27B/judgments_0.jsonl` | 裁判逐题原始评分及判定依据 | P0 / 保留 |
| `NEU/arena_hard_v2/Qwen_38_27B/metrics.json` | `NEU/arena_hard_v2/Qwen_38_27B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `NEU/arena_hard_v2/Qwen_38_27B/responses.jsonl` | `NEU/arena_hard_v2/Qwen_38_27B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `NEU/eq_bench3/Qwen_35_9B/generation_config.json` | `NEU/eq_bench3/Qwen_35_9B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `NEU/eq_bench3/Qwen_35_9B/judge_0_config.json` | `NEU/eq_bench3/Qwen_35_9B/judge_0_config.json` | 该次benchmark裁判配置 | P1 / 保留 |
| `NEU/eq_bench3/Qwen_35_9B/judgments_0.jsonl` | `NEU/eq_bench3/Qwen_35_9B/judgments_0.jsonl` | 裁判逐题原始评分及判定依据 | P0 / 保留 |
| `NEU/eq_bench3/Qwen_35_9B/metrics.json` | `NEU/eq_bench3/Qwen_35_9B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `NEU/eq_bench3/Qwen_35_9B/responses.jsonl` | `NEU/eq_bench3/Qwen_35_9B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `NEU/eq_bench3/Qwen_38_27B/generation_config.json` | `NEU/eq_bench3/Qwen_38_27B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `NEU/eq_bench3/Qwen_38_27B/judge_0_config.json` | `NEU/eq_bench3/Qwen_38_27B/judge_0_config.json` | 该次benchmark裁判配置 | P1 / 保留 |
| `NEU/eq_bench3/Qwen_38_27B/judgments_0.jsonl` | `NEU/eq_bench3/Qwen_38_27B/judgments_0.jsonl` | 裁判逐题原始评分及判定依据 | P0 / 保留 |
| `NEU/eq_bench3/Qwen_38_27B/metrics.json` | `NEU/eq_bench3/Qwen_38_27B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `NEU/eq_bench3/Qwen_38_27B/responses.jsonl` | `NEU/eq_bench3/Qwen_38_27B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `NEU/ifeval/Qwen_35_9B/generation_config.json` | `NEU/ifeval/Qwen_35_9B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `NEU/ifeval/Qwen_35_9B/judgments.jsonl` | `NEU/ifeval/Qwen_35_9B/judgments.jsonl` | IFEval逐题规则检查结果 | P0 / 保留 |
| `NEU/ifeval/Qwen_35_9B/metrics.json` | `NEU/ifeval/Qwen_35_9B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `NEU/ifeval/Qwen_35_9B/responses.jsonl` | `NEU/ifeval/Qwen_35_9B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `NEU/ifeval/Qwen_38_27B/generation_config.json` | `NEU/ifeval/Qwen_38_27B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `NEU/ifeval/Qwen_38_27B/judgments.jsonl` | `NEU/ifeval/Qwen_38_27B/judgments.jsonl` | IFEval逐题规则检查结果 | P0 / 保留 |
| `NEU/ifeval/Qwen_38_27B/metrics.json` | `NEU/ifeval/Qwen_38_27B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `NEU/ifeval/Qwen_38_27B/responses.jsonl` | `NEU/ifeval/Qwen_38_27B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `NEU/multi_challenge/Qwen_35_9B/generation_config.json` | `NEU/multi_challenge/Qwen_35_9B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `NEU/multi_challenge/Qwen_35_9B/judge_0_config.json` | `NEU/multi_challenge/Qwen_35_9B/judge_0_config.json` | 该次benchmark裁判配置 | P1 / 保留 |
| `NEU/multi_challenge/Qwen_35_9B/judgments_0.jsonl` | `NEU/multi_challenge/Qwen_35_9B/judgments_0.jsonl` | 裁判逐题原始评分及判定依据 | P0 / 保留 |
| `NEU/multi_challenge/Qwen_35_9B/metrics.json` | `NEU/multi_challenge/Qwen_35_9B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `NEU/multi_challenge/Qwen_35_9B/responses.jsonl` | `NEU/multi_challenge/Qwen_35_9B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `NEU/multi_challenge/Qwen_38_27B/generation_config.json` | `NEU/multi_challenge/Qwen_38_27B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `NEU/multi_challenge/Qwen_38_27B/judge_0_config.json` | `NEU/multi_challenge/Qwen_38_27B/judge_0_config.json` | 该次benchmark裁判配置 | P1 / 保留 |
| `NEU/multi_challenge/Qwen_38_27B/judgments_0.jsonl` | `NEU/multi_challenge/Qwen_38_27B/judgments_0.jsonl` | 裁判逐题原始评分及判定依据 | P0 / 保留 |
| `NEU/multi_challenge/Qwen_38_27B/metrics.json` | `NEU/multi_challenge/Qwen_38_27B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `NEU/multi_challenge/Qwen_38_27B/responses.jsonl` | `NEU/multi_challenge/Qwen_38_27B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |

### SUP

| 原文件 | 现在放哪里 | 用途 | 优先级 / 处理 |
|---|---|---|---|
| `SUP/cot.jsonl` | `SUP/cot.jsonl` | 该方法生成的逆向CoT样本及生成来源 | P0 / 保留 |
| `SUP/data_manifest.json` | `SUP/data_manifest.json` | 该方法配对训练集的数量、选择规则与哈希 | P0 / 保留 |
| `SUP/train.jsonl` | `SUP/train.jsonl` | 实际用于两模型训练的4175条共同配对样本 | P0 / 保留 |
| `SUP/Qwen_35_9B/evaluation/audit.json` | `SUP/Qwen_35_9B/evaluation/audit.json` | 该模型四benchmark输出的完整性审计 | P0 / 保留 |
| `SUP/Qwen_35_9B/evaluation/config.json` | `SUP/Qwen_35_9B/evaluation/config.json` | 冻结评测配置：目标模型、裁判与benchmark协议 | P0 / 保留 |
| `SUP/Qwen_35_9B/evaluation/deployment.json` | `SUP/Qwen_35_9B/evaluation/deployment.json` | 该次模型服务路径、端口与运行环境 | P1 / 保留 |
| `SUP/Qwen_35_9B/evaluation/evaluate.py` | `SUP/Qwen_35_9B/evaluation/evaluate.py` | 随该次实验冻结的生成、评分与API传输实现；需review后按哈希去重 | P0 / 保留；R8 |
| `SUP/Qwen_35_9B/evaluation/evaluation_sources.sha256.json` | `configs/evaluation_sources.sha256.json` | 固定测试数据与评测源码SHA256；8份字节相同，集中保留1份 | P0 / 相同副本合并 |
| `SUP/Qwen_35_9B/evaluation/serve.py` | `SUP/Qwen_35_9B/evaluation/serve.py` | 随该次实验冻结的模型服务启动/停止实现；需review后按哈希去重 | P1 / 保留；R8 |
| `SUP/Qwen_35_9B/evaluation/verify_results.py` | `SUP/Qwen_35_9B/evaluation/verify_results.py` | 冻结的逐题完整性核验实现，产生audit.json | P0 / 保留；R8 |
| `SUP/Qwen_35_9B/evaluation/runtime/service_history.json` | `data/logs/service-history.tar.gz#SUP/Qwen_35_9B/evaluation/runtime/service_history.json` | 历史服务启动/停止状态；已归档到统一日志目录 | P1 / 哈希核对后归档 |
| `SUP/Qwen_35_9B/training/complete.json` | `SUP/Qwen_35_9B/training/complete.json` | 训练完成记录：131步、训练参数、输入与产物信息 | P0 / 保留 |
| `SUP/Qwen_38_27B/evaluation/audit.json` | `SUP/Qwen_38_27B/evaluation/audit.json` | 该模型四benchmark输出的完整性审计 | P0 / 保留 |
| `SUP/Qwen_38_27B/evaluation/config.json` | `SUP/Qwen_38_27B/evaluation/config.json` | 冻结评测配置：目标模型、裁判与benchmark协议 | P0 / 保留 |
| `SUP/Qwen_38_27B/evaluation/deployment.json` | `SUP/Qwen_38_27B/evaluation/deployment.json` | 该次模型服务路径、端口与运行环境 | P1 / 保留 |
| `SUP/Qwen_38_27B/evaluation/evaluate.py` | `SUP/Qwen_38_27B/evaluation/evaluate.py` | 随该次实验冻结的生成、评分与API传输实现；需review后按哈希去重 | P0 / 保留；R8 |
| `SUP/Qwen_38_27B/evaluation/evaluation_sources.sha256.json` | `configs/evaluation_sources.sha256.json` | 固定测试数据与评测源码SHA256；8份字节相同，集中保留1份 | P0 / 相同副本合并 |
| `SUP/Qwen_38_27B/evaluation/serve.py` | `SUP/Qwen_38_27B/evaluation/serve.py` | 随该次实验冻结的模型服务启动/停止实现；需review后按哈希去重 | P1 / 保留；R8 |
| `SUP/Qwen_38_27B/evaluation/verify_results.py` | `SUP/Qwen_38_27B/evaluation/verify_results.py` | 冻结的逐题完整性核验实现，产生audit.json | P0 / 保留；R8 |
| `SUP/Qwen_38_27B/evaluation/runtime/service_history.json` | `data/logs/service-history.tar.gz#SUP/Qwen_38_27B/evaluation/runtime/service_history.json` | 历史服务启动/停止状态；已归档到统一日志目录 | P1 / 哈希核对后归档 |
| `SUP/Qwen_38_27B/training/complete.json` | `SUP/Qwen_38_27B/training/complete.json` | 训练完成记录：131步、训练参数、输入与产物信息 | P0 / 保留 |
| `SUP/arena_hard_v2/Qwen_35_9B/generation_config.json` | `SUP/arena_hard_v2/Qwen_35_9B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `SUP/arena_hard_v2/Qwen_35_9B/judge_0_config.json` | `SUP/arena_hard_v2/Qwen_35_9B/judge_0_config.json` | 该次benchmark裁判配置 | P1 / 保留 |
| `SUP/arena_hard_v2/Qwen_35_9B/judgments_0.jsonl` | `SUP/arena_hard_v2/Qwen_35_9B/judgments_0.jsonl` | 裁判逐题原始评分及判定依据 | P0 / 保留 |
| `SUP/arena_hard_v2/Qwen_35_9B/metrics.json` | `SUP/arena_hard_v2/Qwen_35_9B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `SUP/arena_hard_v2/Qwen_35_9B/responses.jsonl` | `SUP/arena_hard_v2/Qwen_35_9B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `SUP/arena_hard_v2/Qwen_38_27B/generation_config.json` | `SUP/arena_hard_v2/Qwen_38_27B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `SUP/arena_hard_v2/Qwen_38_27B/judge_0_config.json` | `SUP/arena_hard_v2/Qwen_38_27B/judge_0_config.json` | 该次benchmark裁判配置 | P1 / 保留 |
| `SUP/arena_hard_v2/Qwen_38_27B/judgments_0.jsonl` | `SUP/arena_hard_v2/Qwen_38_27B/judgments_0.jsonl` | 裁判逐题原始评分及判定依据 | P0 / 保留 |
| `SUP/arena_hard_v2/Qwen_38_27B/metrics.json` | `SUP/arena_hard_v2/Qwen_38_27B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `SUP/arena_hard_v2/Qwen_38_27B/responses.jsonl` | `SUP/arena_hard_v2/Qwen_38_27B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `SUP/eq_bench3/Qwen_35_9B/generation_config.json` | `SUP/eq_bench3/Qwen_35_9B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `SUP/eq_bench3/Qwen_35_9B/judge_0_config.json` | `SUP/eq_bench3/Qwen_35_9B/judge_0_config.json` | 该次benchmark裁判配置 | P1 / 保留 |
| `SUP/eq_bench3/Qwen_35_9B/judgments_0.jsonl` | `SUP/eq_bench3/Qwen_35_9B/judgments_0.jsonl` | 裁判逐题原始评分及判定依据 | P0 / 保留 |
| `SUP/eq_bench3/Qwen_35_9B/metrics.json` | `SUP/eq_bench3/Qwen_35_9B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `SUP/eq_bench3/Qwen_35_9B/responses.jsonl` | `SUP/eq_bench3/Qwen_35_9B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `SUP/eq_bench3/Qwen_38_27B/generation_config.json` | `SUP/eq_bench3/Qwen_38_27B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `SUP/eq_bench3/Qwen_38_27B/judge_0_config.json` | `SUP/eq_bench3/Qwen_38_27B/judge_0_config.json` | 该次benchmark裁判配置 | P1 / 保留 |
| `SUP/eq_bench3/Qwen_38_27B/judgments_0.jsonl` | `SUP/eq_bench3/Qwen_38_27B/judgments_0.jsonl` | 裁判逐题原始评分及判定依据 | P0 / 保留 |
| `SUP/eq_bench3/Qwen_38_27B/metrics.json` | `SUP/eq_bench3/Qwen_38_27B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `SUP/eq_bench3/Qwen_38_27B/responses.jsonl` | `SUP/eq_bench3/Qwen_38_27B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `SUP/ifeval/Qwen_35_9B/generation_config.json` | `SUP/ifeval/Qwen_35_9B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `SUP/ifeval/Qwen_35_9B/judgments.jsonl` | `SUP/ifeval/Qwen_35_9B/judgments.jsonl` | IFEval逐题规则检查结果 | P0 / 保留 |
| `SUP/ifeval/Qwen_35_9B/metrics.json` | `SUP/ifeval/Qwen_35_9B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `SUP/ifeval/Qwen_35_9B/responses.jsonl` | `SUP/ifeval/Qwen_35_9B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `SUP/ifeval/Qwen_38_27B/generation_config.json` | `SUP/ifeval/Qwen_38_27B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `SUP/ifeval/Qwen_38_27B/judgments.jsonl` | `SUP/ifeval/Qwen_38_27B/judgments.jsonl` | IFEval逐题规则检查结果 | P0 / 保留 |
| `SUP/ifeval/Qwen_38_27B/metrics.json` | `SUP/ifeval/Qwen_38_27B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `SUP/ifeval/Qwen_38_27B/responses.jsonl` | `SUP/ifeval/Qwen_38_27B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `SUP/multi_challenge/Qwen_35_9B/generation_config.json` | `SUP/multi_challenge/Qwen_35_9B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `SUP/multi_challenge/Qwen_35_9B/judge_0_config.json` | `SUP/multi_challenge/Qwen_35_9B/judge_0_config.json` | 该次benchmark裁判配置 | P1 / 保留 |
| `SUP/multi_challenge/Qwen_35_9B/judgments_0.jsonl` | `SUP/multi_challenge/Qwen_35_9B/judgments_0.jsonl` | 裁判逐题原始评分及判定依据 | P0 / 保留 |
| `SUP/multi_challenge/Qwen_35_9B/metrics.json` | `SUP/multi_challenge/Qwen_35_9B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `SUP/multi_challenge/Qwen_35_9B/responses.jsonl` | `SUP/multi_challenge/Qwen_35_9B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `SUP/multi_challenge/Qwen_38_27B/generation_config.json` | `SUP/multi_challenge/Qwen_38_27B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `SUP/multi_challenge/Qwen_38_27B/judge_0_config.json` | `SUP/multi_challenge/Qwen_38_27B/judge_0_config.json` | 该次benchmark裁判配置 | P1 / 保留 |
| `SUP/multi_challenge/Qwen_38_27B/judgments_0.jsonl` | `SUP/multi_challenge/Qwen_38_27B/judgments_0.jsonl` | 裁判逐题原始评分及判定依据 | P0 / 保留 |
| `SUP/multi_challenge/Qwen_38_27B/metrics.json` | `SUP/multi_challenge/Qwen_38_27B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `SUP/multi_challenge/Qwen_38_27B/responses.jsonl` | `SUP/multi_challenge/Qwen_38_27B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |

### AUG-SUP

| 原文件 | 现在放哪里 | 用途 | 优先级 / 处理 |
|---|---|---|---|
| `AUG-SUP/cot.jsonl` | `AUG-SUP/cot.jsonl` | 该方法生成的逆向CoT样本及生成来源 | P0 / 保留 |
| `AUG-SUP/data_manifest.json` | `AUG-SUP/data_manifest.json` | 该方法配对训练集的数量、选择规则与哈希 | P0 / 保留 |
| `AUG-SUP/train.jsonl` | `AUG-SUP/train.jsonl` | 实际用于两模型训练的4175条共同配对样本 | P0 / 保留 |
| `AUG-SUP/Qwen_35_9B/evaluation/audit.json` | `AUG-SUP/Qwen_35_9B/evaluation/audit.json` | 该模型四benchmark输出的完整性审计 | P0 / 保留 |
| `AUG-SUP/Qwen_35_9B/evaluation/config.json` | `AUG-SUP/Qwen_35_9B/evaluation/config.json` | 冻结评测配置：目标模型、裁判与benchmark协议 | P0 / 保留 |
| `AUG-SUP/Qwen_35_9B/evaluation/deployment.json` | `AUG-SUP/Qwen_35_9B/evaluation/deployment.json` | 该次模型服务路径、端口与运行环境 | P1 / 保留 |
| `AUG-SUP/Qwen_35_9B/evaluation/evaluate.py` | `AUG-SUP/Qwen_35_9B/evaluation/evaluate.py` | 随该次实验冻结的生成、评分与API传输实现；需review后按哈希去重 | P0 / 保留；R8 |
| `AUG-SUP/Qwen_35_9B/evaluation/evaluation_sources.sha256.json` | `configs/evaluation_sources.sha256.json` | 固定测试数据与评测源码SHA256；8份字节相同，集中保留1份 | P0 / 相同副本合并 |
| `AUG-SUP/Qwen_35_9B/evaluation/serve.py` | `AUG-SUP/Qwen_35_9B/evaluation/serve.py` | 随该次实验冻结的模型服务启动/停止实现；需review后按哈希去重 | P1 / 保留；R8 |
| `AUG-SUP/Qwen_35_9B/evaluation/verify_results.py` | `AUG-SUP/Qwen_35_9B/evaluation/verify_results.py` | 冻结的逐题完整性核验实现，产生audit.json | P0 / 保留；R8 |
| `AUG-SUP/Qwen_35_9B/evaluation/runtime/service_history.json` | `data/logs/service-history.tar.gz#AUG-SUP/Qwen_35_9B/evaluation/runtime/service_history.json` | 历史服务启动/停止状态；已归档到统一日志目录 | P1 / 哈希核对后归档 |
| `AUG-SUP/Qwen_35_9B/training/complete.json` | `AUG-SUP/Qwen_35_9B/training/complete.json` | 训练完成记录：131步、训练参数、输入与产物信息 | P0 / 保留 |
| `AUG-SUP/Qwen_38_27B/evaluation/audit.json` | `AUG-SUP/Qwen_38_27B/evaluation/audit.json` | 该模型四benchmark输出的完整性审计 | P0 / 保留 |
| `AUG-SUP/Qwen_38_27B/evaluation/config.json` | `AUG-SUP/Qwen_38_27B/evaluation/config.json` | 冻结评测配置：目标模型、裁判与benchmark协议 | P0 / 保留 |
| `AUG-SUP/Qwen_38_27B/evaluation/deployment.json` | `AUG-SUP/Qwen_38_27B/evaluation/deployment.json` | 该次模型服务路径、端口与运行环境 | P1 / 保留 |
| `AUG-SUP/Qwen_38_27B/evaluation/evaluate.py` | `AUG-SUP/Qwen_38_27B/evaluation/evaluate.py` | 随该次实验冻结的生成、评分与API传输实现；需review后按哈希去重 | P0 / 保留；R8 |
| `AUG-SUP/Qwen_38_27B/evaluation/evaluation_sources.sha256.json` | `configs/evaluation_sources.sha256.json` | 固定测试数据与评测源码SHA256；8份字节相同，集中保留1份 | P0 / 相同副本合并 |
| `AUG-SUP/Qwen_38_27B/evaluation/serve.py` | `AUG-SUP/Qwen_38_27B/evaluation/serve.py` | 随该次实验冻结的模型服务启动/停止实现；需review后按哈希去重 | P1 / 保留；R8 |
| `AUG-SUP/Qwen_38_27B/evaluation/verify_results.py` | `AUG-SUP/Qwen_38_27B/evaluation/verify_results.py` | 冻结的逐题完整性核验实现，产生audit.json | P0 / 保留；R8 |
| `AUG-SUP/Qwen_38_27B/evaluation/runtime/service_history.json` | `data/logs/service-history.tar.gz#AUG-SUP/Qwen_38_27B/evaluation/runtime/service_history.json` | 历史服务启动/停止状态；已归档到统一日志目录 | P1 / 哈希核对后归档 |
| `AUG-SUP/Qwen_38_27B/training/complete.json` | `AUG-SUP/Qwen_38_27B/training/complete.json` | 训练完成记录：131步、训练参数、输入与产物信息 | P0 / 保留 |
| `AUG-SUP/arena_hard_v2/Qwen_35_9B/generation_config.json` | `AUG-SUP/arena_hard_v2/Qwen_35_9B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `AUG-SUP/arena_hard_v2/Qwen_35_9B/judge_0_config.json` | `AUG-SUP/arena_hard_v2/Qwen_35_9B/judge_0_config.json` | 该次benchmark裁判配置 | P1 / 保留 |
| `AUG-SUP/arena_hard_v2/Qwen_35_9B/judgments_0.jsonl` | `AUG-SUP/arena_hard_v2/Qwen_35_9B/judgments_0.jsonl` | 裁判逐题原始评分及判定依据 | P0 / 保留 |
| `AUG-SUP/arena_hard_v2/Qwen_35_9B/metrics.json` | `AUG-SUP/arena_hard_v2/Qwen_35_9B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `AUG-SUP/arena_hard_v2/Qwen_35_9B/responses.jsonl` | `AUG-SUP/arena_hard_v2/Qwen_35_9B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `AUG-SUP/arena_hard_v2/Qwen_38_27B/generation_config.json` | `AUG-SUP/arena_hard_v2/Qwen_38_27B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `AUG-SUP/arena_hard_v2/Qwen_38_27B/judge_0_config.json` | `AUG-SUP/arena_hard_v2/Qwen_38_27B/judge_0_config.json` | 该次benchmark裁判配置 | P1 / 保留 |
| `AUG-SUP/arena_hard_v2/Qwen_38_27B/judgments_0.jsonl` | `AUG-SUP/arena_hard_v2/Qwen_38_27B/judgments_0.jsonl` | 裁判逐题原始评分及判定依据 | P0 / 保留 |
| `AUG-SUP/arena_hard_v2/Qwen_38_27B/metrics.json` | `AUG-SUP/arena_hard_v2/Qwen_38_27B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `AUG-SUP/arena_hard_v2/Qwen_38_27B/responses.jsonl` | `AUG-SUP/arena_hard_v2/Qwen_38_27B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `AUG-SUP/eq_bench3/Qwen_35_9B/generation_config.json` | `AUG-SUP/eq_bench3/Qwen_35_9B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `AUG-SUP/eq_bench3/Qwen_35_9B/judge_0_config.json` | `AUG-SUP/eq_bench3/Qwen_35_9B/judge_0_config.json` | 该次benchmark裁判配置 | P1 / 保留 |
| `AUG-SUP/eq_bench3/Qwen_35_9B/judgments_0.jsonl` | `AUG-SUP/eq_bench3/Qwen_35_9B/judgments_0.jsonl` | 裁判逐题原始评分及判定依据 | P0 / 保留 |
| `AUG-SUP/eq_bench3/Qwen_35_9B/metrics.json` | `AUG-SUP/eq_bench3/Qwen_35_9B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `AUG-SUP/eq_bench3/Qwen_35_9B/responses.jsonl` | `AUG-SUP/eq_bench3/Qwen_35_9B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `AUG-SUP/eq_bench3/Qwen_38_27B/generation_config.json` | `AUG-SUP/eq_bench3/Qwen_38_27B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `AUG-SUP/eq_bench3/Qwen_38_27B/judge_0_config.json` | `AUG-SUP/eq_bench3/Qwen_38_27B/judge_0_config.json` | 该次benchmark裁判配置 | P1 / 保留 |
| `AUG-SUP/eq_bench3/Qwen_38_27B/judgments_0.jsonl` | `AUG-SUP/eq_bench3/Qwen_38_27B/judgments_0.jsonl` | 裁判逐题原始评分及判定依据 | P0 / 保留 |
| `AUG-SUP/eq_bench3/Qwen_38_27B/metrics.json` | `AUG-SUP/eq_bench3/Qwen_38_27B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `AUG-SUP/eq_bench3/Qwen_38_27B/responses.jsonl` | `AUG-SUP/eq_bench3/Qwen_38_27B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `AUG-SUP/ifeval/Qwen_35_9B/generation_config.json` | `AUG-SUP/ifeval/Qwen_35_9B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `AUG-SUP/ifeval/Qwen_35_9B/judgments.jsonl` | `AUG-SUP/ifeval/Qwen_35_9B/judgments.jsonl` | IFEval逐题规则检查结果 | P0 / 保留 |
| `AUG-SUP/ifeval/Qwen_35_9B/metrics.json` | `AUG-SUP/ifeval/Qwen_35_9B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `AUG-SUP/ifeval/Qwen_35_9B/responses.jsonl` | `AUG-SUP/ifeval/Qwen_35_9B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `AUG-SUP/ifeval/Qwen_38_27B/generation_config.json` | `AUG-SUP/ifeval/Qwen_38_27B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `AUG-SUP/ifeval/Qwen_38_27B/judgments.jsonl` | `AUG-SUP/ifeval/Qwen_38_27B/judgments.jsonl` | IFEval逐题规则检查结果 | P0 / 保留 |
| `AUG-SUP/ifeval/Qwen_38_27B/metrics.json` | `AUG-SUP/ifeval/Qwen_38_27B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `AUG-SUP/ifeval/Qwen_38_27B/responses.jsonl` | `AUG-SUP/ifeval/Qwen_38_27B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `AUG-SUP/multi_challenge/Qwen_35_9B/generation_config.json` | `AUG-SUP/multi_challenge/Qwen_35_9B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `AUG-SUP/multi_challenge/Qwen_35_9B/judge_0_config.json` | `AUG-SUP/multi_challenge/Qwen_35_9B/judge_0_config.json` | 该次benchmark裁判配置 | P1 / 保留 |
| `AUG-SUP/multi_challenge/Qwen_35_9B/judgments_0.jsonl` | `AUG-SUP/multi_challenge/Qwen_35_9B/judgments_0.jsonl` | 裁判逐题原始评分及判定依据 | P0 / 保留 |
| `AUG-SUP/multi_challenge/Qwen_35_9B/metrics.json` | `AUG-SUP/multi_challenge/Qwen_35_9B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `AUG-SUP/multi_challenge/Qwen_35_9B/responses.jsonl` | `AUG-SUP/multi_challenge/Qwen_35_9B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `AUG-SUP/multi_challenge/Qwen_38_27B/generation_config.json` | `AUG-SUP/multi_challenge/Qwen_38_27B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `AUG-SUP/multi_challenge/Qwen_38_27B/judge_0_config.json` | `AUG-SUP/multi_challenge/Qwen_38_27B/judge_0_config.json` | 该次benchmark裁判配置 | P1 / 保留 |
| `AUG-SUP/multi_challenge/Qwen_38_27B/judgments_0.jsonl` | `AUG-SUP/multi_challenge/Qwen_38_27B/judgments_0.jsonl` | 裁判逐题原始评分及判定依据 | P0 / 保留 |
| `AUG-SUP/multi_challenge/Qwen_38_27B/metrics.json` | `AUG-SUP/multi_challenge/Qwen_38_27B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `AUG-SUP/multi_challenge/Qwen_38_27B/responses.jsonl` | `AUG-SUP/multi_challenge/Qwen_38_27B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |

### SSR

| 原文件 | 现在放哪里 | 用途 | 优先级 / 处理 |
|---|---|---|---|
| `SSR/cot.jsonl` | `SSR/cot.jsonl` | 该方法生成的逆向CoT样本及生成来源 | P0 / 保留 |
| `SSR/data_manifest.json` | `SSR/data_manifest.json` | 该方法配对训练集的数量、选择规则与哈希 | P0 / 保留 |
| `SSR/train.jsonl` | `SSR/train.jsonl` | 实际用于两模型训练的4175条共同配对样本 | P0 / 保留 |
| `SSR/Qwen_35_9B/evaluation/audit.json` | `SSR/Qwen_35_9B/evaluation/audit.json` | 该模型四benchmark输出的完整性审计 | P0 / 保留 |
| `SSR/Qwen_35_9B/evaluation/config.json` | `SSR/Qwen_35_9B/evaluation/config.json` | 冻结评测配置：目标模型、裁判与benchmark协议 | P0 / 保留 |
| `SSR/Qwen_35_9B/evaluation/deployment.json` | `SSR/Qwen_35_9B/evaluation/deployment.json` | 该次模型服务路径、端口与运行环境 | P1 / 保留 |
| `SSR/Qwen_35_9B/evaluation/evaluate.py` | `SSR/Qwen_35_9B/evaluation/evaluate.py` | 随该次实验冻结的生成、评分与API传输实现；需review后按哈希去重 | P0 / 保留；R8 |
| `SSR/Qwen_35_9B/evaluation/evaluation_sources.sha256.json` | `configs/evaluation_sources.sha256.json` | 固定测试数据与评测源码SHA256；8份字节相同，集中保留1份 | P0 / 相同副本合并 |
| `SSR/Qwen_35_9B/evaluation/serve.py` | `SSR/Qwen_35_9B/evaluation/serve.py` | 随该次实验冻结的模型服务启动/停止实现；需review后按哈希去重 | P1 / 保留；R8 |
| `SSR/Qwen_35_9B/evaluation/verify_results.py` | `SSR/Qwen_35_9B/evaluation/verify_results.py` | 冻结的逐题完整性核验实现，产生audit.json | P0 / 保留；R8 |
| `SSR/Qwen_35_9B/evaluation/runtime/service_history.json` | `data/logs/service-history.tar.gz#SSR/Qwen_35_9B/evaluation/runtime/service_history.json` | 历史服务启动/停止状态；已归档到统一日志目录 | P1 / 哈希核对后归档 |
| `SSR/Qwen_35_9B/training/complete.json` | `SSR/Qwen_35_9B/training/complete.json` | 训练完成记录：131步、训练参数、输入与产物信息 | P0 / 保留 |
| `SSR/Qwen_38_27B/evaluation/audit.json` | `SSR/Qwen_38_27B/evaluation/audit.json` | 该模型四benchmark输出的完整性审计 | P0 / 保留 |
| `SSR/Qwen_38_27B/evaluation/config.json` | `SSR/Qwen_38_27B/evaluation/config.json` | 冻结评测配置：目标模型、裁判与benchmark协议 | P0 / 保留 |
| `SSR/Qwen_38_27B/evaluation/deployment.json` | `SSR/Qwen_38_27B/evaluation/deployment.json` | 该次模型服务路径、端口与运行环境 | P1 / 保留 |
| `SSR/Qwen_38_27B/evaluation/evaluate.py` | `SSR/Qwen_38_27B/evaluation/evaluate.py` | 随该次实验冻结的生成、评分与API传输实现；需review后按哈希去重 | P0 / 保留；R8 |
| `SSR/Qwen_38_27B/evaluation/evaluation_sources.sha256.json` | `configs/evaluation_sources.sha256.json` | 固定测试数据与评测源码SHA256；8份字节相同，集中保留1份 | P0 / 相同副本合并 |
| `SSR/Qwen_38_27B/evaluation/serve.py` | `SSR/Qwen_38_27B/evaluation/serve.py` | 随该次实验冻结的模型服务启动/停止实现；需review后按哈希去重 | P1 / 保留；R8 |
| `SSR/Qwen_38_27B/evaluation/verify_results.py` | `SSR/Qwen_38_27B/evaluation/verify_results.py` | 冻结的逐题完整性核验实现，产生audit.json | P0 / 保留；R8 |
| `SSR/Qwen_38_27B/evaluation/runtime/service_history.json` | `data/logs/service-history.tar.gz#SSR/Qwen_38_27B/evaluation/runtime/service_history.json` | 历史服务启动/停止状态；已归档到统一日志目录 | P1 / 哈希核对后归档 |
| `SSR/Qwen_38_27B/training/complete.json` | `SSR/Qwen_38_27B/training/complete.json` | 训练完成记录：131步、训练参数、输入与产物信息 | P0 / 保留 |
| `SSR/arena_hard_v2/Qwen_35_9B/generation_config.json` | `SSR/arena_hard_v2/Qwen_35_9B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `SSR/arena_hard_v2/Qwen_35_9B/judge_0_config.json` | `SSR/arena_hard_v2/Qwen_35_9B/judge_0_config.json` | 该次benchmark裁判配置 | P1 / 保留 |
| `SSR/arena_hard_v2/Qwen_35_9B/judgments_0.jsonl` | `SSR/arena_hard_v2/Qwen_35_9B/judgments_0.jsonl` | 裁判逐题原始评分及判定依据 | P0 / 保留 |
| `SSR/arena_hard_v2/Qwen_35_9B/metrics.json` | `SSR/arena_hard_v2/Qwen_35_9B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `SSR/arena_hard_v2/Qwen_35_9B/responses.jsonl` | `SSR/arena_hard_v2/Qwen_35_9B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `SSR/arena_hard_v2/Qwen_38_27B/generation_config.json` | `SSR/arena_hard_v2/Qwen_38_27B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `SSR/arena_hard_v2/Qwen_38_27B/judge_0_config.json` | `SSR/arena_hard_v2/Qwen_38_27B/judge_0_config.json` | 该次benchmark裁判配置 | P1 / 保留 |
| `SSR/arena_hard_v2/Qwen_38_27B/judgments_0.jsonl` | `SSR/arena_hard_v2/Qwen_38_27B/judgments_0.jsonl` | 裁判逐题原始评分及判定依据 | P0 / 保留 |
| `SSR/arena_hard_v2/Qwen_38_27B/metrics.json` | `SSR/arena_hard_v2/Qwen_38_27B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `SSR/arena_hard_v2/Qwen_38_27B/responses.jsonl` | `SSR/arena_hard_v2/Qwen_38_27B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `SSR/eq_bench3/Qwen_35_9B/generation_config.json` | `SSR/eq_bench3/Qwen_35_9B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `SSR/eq_bench3/Qwen_35_9B/judge_0_config.json` | `SSR/eq_bench3/Qwen_35_9B/judge_0_config.json` | 该次benchmark裁判配置 | P1 / 保留 |
| `SSR/eq_bench3/Qwen_35_9B/judgments_0.jsonl` | `SSR/eq_bench3/Qwen_35_9B/judgments_0.jsonl` | 裁判逐题原始评分及判定依据 | P0 / 保留 |
| `SSR/eq_bench3/Qwen_35_9B/metrics.json` | `SSR/eq_bench3/Qwen_35_9B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `SSR/eq_bench3/Qwen_35_9B/responses.jsonl` | `SSR/eq_bench3/Qwen_35_9B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `SSR/eq_bench3/Qwen_38_27B/generation_config.json` | `SSR/eq_bench3/Qwen_38_27B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `SSR/eq_bench3/Qwen_38_27B/judge_0_config.json` | `SSR/eq_bench3/Qwen_38_27B/judge_0_config.json` | 该次benchmark裁判配置 | P1 / 保留 |
| `SSR/eq_bench3/Qwen_38_27B/judgments_0.jsonl` | `SSR/eq_bench3/Qwen_38_27B/judgments_0.jsonl` | 裁判逐题原始评分及判定依据 | P0 / 保留 |
| `SSR/eq_bench3/Qwen_38_27B/metrics.json` | `SSR/eq_bench3/Qwen_38_27B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `SSR/eq_bench3/Qwen_38_27B/responses.jsonl` | `SSR/eq_bench3/Qwen_38_27B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `SSR/ifeval/Qwen_35_9B/generation_config.json` | `SSR/ifeval/Qwen_35_9B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `SSR/ifeval/Qwen_35_9B/judgments.jsonl` | `SSR/ifeval/Qwen_35_9B/judgments.jsonl` | IFEval逐题规则检查结果 | P0 / 保留 |
| `SSR/ifeval/Qwen_35_9B/metrics.json` | `SSR/ifeval/Qwen_35_9B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `SSR/ifeval/Qwen_35_9B/responses.jsonl` | `SSR/ifeval/Qwen_35_9B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `SSR/ifeval/Qwen_38_27B/generation_config.json` | `SSR/ifeval/Qwen_38_27B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `SSR/ifeval/Qwen_38_27B/judgments.jsonl` | `SSR/ifeval/Qwen_38_27B/judgments.jsonl` | IFEval逐题规则检查结果 | P0 / 保留 |
| `SSR/ifeval/Qwen_38_27B/metrics.json` | `SSR/ifeval/Qwen_38_27B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `SSR/ifeval/Qwen_38_27B/responses.jsonl` | `SSR/ifeval/Qwen_38_27B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `SSR/multi_challenge/Qwen_35_9B/generation_config.json` | `SSR/multi_challenge/Qwen_35_9B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `SSR/multi_challenge/Qwen_35_9B/judge_0_config.json` | `SSR/multi_challenge/Qwen_35_9B/judge_0_config.json` | 该次benchmark裁判配置 | P1 / 保留 |
| `SSR/multi_challenge/Qwen_35_9B/judgments_0.jsonl` | `SSR/multi_challenge/Qwen_35_9B/judgments_0.jsonl` | 裁判逐题原始评分及判定依据 | P0 / 保留 |
| `SSR/multi_challenge/Qwen_35_9B/metrics.json` | `SSR/multi_challenge/Qwen_35_9B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `SSR/multi_challenge/Qwen_35_9B/responses.jsonl` | `SSR/multi_challenge/Qwen_35_9B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |
| `SSR/multi_challenge/Qwen_38_27B/generation_config.json` | `SSR/multi_challenge/Qwen_38_27B/generation_config.json` | 该次benchmark回答生成参数与目标模型 | P1 / 保留 |
| `SSR/multi_challenge/Qwen_38_27B/judge_0_config.json` | `SSR/multi_challenge/Qwen_38_27B/judge_0_config.json` | 该次benchmark裁判配置 | P1 / 保留 |
| `SSR/multi_challenge/Qwen_38_27B/judgments_0.jsonl` | `SSR/multi_challenge/Qwen_38_27B/judgments_0.jsonl` | 裁判逐题原始评分及判定依据 | P0 / 保留 |
| `SSR/multi_challenge/Qwen_38_27B/metrics.json` | `SSR/multi_challenge/Qwen_38_27B/metrics.json` | 该benchmark最终汇总指标与完成状态 | P0 / 保留 |
| `SSR/multi_challenge/Qwen_38_27B/responses.jsonl` | `SSR/multi_challenge/Qwen_38_27B/responses.jsonl` | 该方法/模型在该benchmark的逐题原始回答 | P0 / 保留 |

## 新增的维护文件

| 文件 | 用途 |
|---|---|
| docs/RESULTS.md | 由原三份结果说明合并的唯一结论入口 |
| docs/PLAN.md | 全工作区整理前盘点、保留顺序、目录方案、实际执行范围 |
| docs/REVIEW.md | 不确定能否永久删除的具体范围、影响和建议 |
| docs/FILES.md | 本逐文件用途清单 |
| docs/FULL_INVENTORY.tsv.gz | 全工作区每个原文件的路径、用途、优先级、当前处理和review编号 |
| configs/evaluation_sources.sha256.json | 合并8份相同评测来源清单后的集中版本 |
| data/logs/service-history.tar.gz | 8份方法服务记录与内部哈希清单 |

## 56个链接与工作区外的权重

每个“方法/模型”保留 adapter、model、training/checkpoints 三个链接，共24个；指向工作区外 CoT_Analyse_artifacts/baseline_method/。冻结 evaluation 下另有四个 benchmark 相对链接，共32个，仍用于冻结评测实现的路径解析。这56个链接不是额外实验，也不是重复结果副本。

| 产物 | 数量 | 大小（十进制GB） | 保留原因 / review |
|---|---:|---:|---|
| adapter | 8 | 2.721 | 最终LoRA，保留 |
| model | 8 | 290.608 | 直接复跑模型服务，R5 |
| checkpoints | 8 | 7.847 | optimizer/RNG续训状态，R4 |

## 其余工作区怎么阅读

| 区域 | 最先阅读 / 核心用途 | 整理决定 |
|---|---|---|
| 工作区 README.md | 总导航和已完成/未完成实验范围 | 已整理 |
| docs/data/ | 数据统计、说明、来源锁定清单 | 3个文件归到同一处 |
| test_scripts/ | Base_model_result_conclusion.md 与各模型原始指标 | 保留原协议和结果；91个历史日志压成7包 |
| test_data/、training_data/ | 固定测试数据、LMArena和训练样本索引 | 保留；不拆散上游冻结快照 |
| auto_research/pattern_20261004/ | README、RESULTS、各阶段结果和冻结实现 | 先完成全量用途分类；唯一内容的删除待R1–R3等review |

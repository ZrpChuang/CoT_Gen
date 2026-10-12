# 四种方法的 baseline 实验

**2026-10-12：本目录训练结果作为历史证据保留，训练权重已弃用。** 用户要求 GPT 同时生成答案与四方法说明，并从原始基座重新训练。当前执行入口为 [scaling](../auto_research/scaling/README.md)，新产物是 `CoT_Analyse_artifacts/gpt_fresh_100k_20261012/`；不要从本目录旧 checkpoint 续训。

先看 [总表与结论](docs/RESULTS.md)，再按需要进入方法目录或运行代码。四方法 × 两模型 × 四 benchmark，共8次训练、32项评测；每组训练131步。原始 Baseline 在总表第一行。

| 要做的事 | 入口 |
|---|---|
| 看成绩、实验设置和诊断 | [docs/RESULTS.md](docs/RESULTS.md) |
| 查看 GPT 全新 100K 重训范围与 TODO | [docs/SCALE_100K_TODO.md](docs/SCALE_100K_TODO.md) |
| 运行远端常驻 100K 队列 | [auto_research/scaling/README.md](../auto_research/scaling/README.md) |
| 理解每个文件的用途 | [docs/FILES.md](docs/FILES.md) |
| 看全工作区整理方案 | [docs/PLAN.md](docs/PLAN.md) |
| review 不确定能否删除的材料 | [docs/REVIEW.md](docs/REVIEW.md) |
| 查看历史流水线命令 | `python pipeline.py --help`；本轮训练使用 scaling 入口 |

```text
baseline_method/
├── README.md、pipeline.py
├── core/        数据构造、准备、训练、评测、汇总与共享模块
├── configs/     本机配置、实验配方、依赖、提示词与来源清单
├── docs/        核心结论、文件用途、整理方案与review
├── data/        共享参考答案、完成标记、logs/
├── NEU/
├── SUP/
├── AUG-SUP/
└── SSR/
```

每种方法保留 `cot.jsonl`、`train.jsonl`、`data_manifest.json`；四个 benchmark 下各有两个模型的指标、回答、裁判记录和配置。两个模型记录目录保留训练完成文件和冻结评测实现；adapter、合并权重及checkpoint链接到工作区外的 `CoT_Analyse_artifacts/baseline_method/`。

## 运行

在独立环境安装 `configs/requirements-training.txt`；参照 `configs/local_paths.example.json` 和 `configs/experiment.example.json` 配置本机路径与教师API。当前本机配置也放在 configs/ 并忽略Git。

```bash
python pipeline.py validate
python pipeline.py --dry-run all
python pipeline.py train --method NEU --model Qwen_35_9B
python pipeline.py evaluate --method NEU --model Qwen_35_9B
python pipeline.py report
```

`all` 按完成标记跳过已有阶段。生成/训练集标记位于 data/；训练标记位于各模型的 training/complete.json；评测标记位于 evaluation/audit.json。当前日志、状态和历史日志包都在 data/logs/；report 派生表也写入该目录。生成数据依赖固定LMArena索引和仓库内的冻结API传输代码；评测需要固定测试集与裁判配置。

代码检查：

```bash
PYTHONPATH=core python -m unittest discover -s core/tests -v
ruff check --config configs/pyproject.toml .
ruff format --check --config configs/pyproject.toml .
```

本轮是4,175条、BF16 LoRA、rank16、1epoch、8K上限的小规模实验。参考答案为模型生成，EQ/Arena采用本地评分口径。完整限制见总表与结论。

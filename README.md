# CoT_Gen

逆向构造 CoT 的实验目录。已完成五个原始模型在四项 benchmark 上的基座评测，保存数据来源、固定版本、下载工具、评测代码及逐题结果，尚未开展正式训练。

## 基座评测

完整成绩与评测口径见 [基座评测总结](test_scripts/Base_model_result_conclusion.md)。

- 模型：Qwen3.5-4B、Qwen3.5-9B、Qwen3.8-27B、GPT-6 Astra、Qwen3.8-Max。
- Benchmark：IFEval、MultiChallenge、EQ-Bench3、Arena-Hard-v2。
- 各模型代码和结果独立存放在 `test_scripts/` 下；`metrics.json` 为最终成绩，`responses.jsonl` 为逐题回答，`judgments*.jsonl` 为评分记录，`audit.json` 为完整性检查。
- 汇总表采用 Astra 裁判；早期三个模型的 Qwen Max 裁判记录保留用于追溯。
- Qwen3.5 使用官方通用思考采样，其余模型沿用各自已记录的配置；结果不代表严格等计算预算比较或 SSR 论文数值复现。

运行前按各模型 README 配置数据路径、模型路径和 API 凭证；配置中的绝对路径对应实验服务器。API 密钥从项目外读取，不包含在仓库中。原始模型权重不提交到 Git。

## 数据

| 目录 | 数据 | 数量 |
|---|---|---:|
| `training_data/lmarena_140k/` | LMArena 原始用户对话 | 135,634 |
| `test_data/arena_hard_v2/` | ArenaHard v2 | 750 题（500 hard + 250 creative） |
| `test_data/eq_bench3/` | EQ-Bench3 | 46 个标题，45 个有效场景（1 个为空） |
| `test_data/ifeval/` | IFEval | 541 题 |
| `test_data/multi_challenge/` | MultiChallenge | 273 段对话 |

原始数据及配套评测资源约 1.72 GB，通过清单恢复，不直接提交到 Git。数据及第三方代码的许可证以各上游来源为准；下载时保留上游提供的许可证或数据卡。

LMArena 当前仅作为训练原料，尚未抽取 100k 样本、生成新的参考答案或 CoT。测试数据不参与训练。论文未给出精确数据快照版本，这里记录的是实际下载的固定版本。

## 下载与校验

需要 Python 3.9+，下载工具仅依赖标准库。

```bash
python3 download_data.py --dry-run
python3 download_data.py                 # 下载全部，并核对 SHA256
python3 download_data.py --group test    # 只下载四个测试集
python3 download_data.py --verify-only   # 检查本地已有文件
```

默认将文件放在仓库根目录下，也可用 `--root /path/to/CoT_Analyse` 指定位置。
运行环境需能访问清单中的官方 GitHub/Hugging Face 文件地址；如果服务器直连失败，可在联网机器下载后同步目录。

## 文件说明

- `DATASETS.md`：数据入口及使用说明。
- `DATASET_SUMMARY.json`：数量、规模和来源版本。
- `DOWNLOAD_MANIFEST.json`：全部文件的固定下载地址、大小和 SHA256。
- `sources.lock.json`：下载前固定的源版本。
- 各数据目录的 `DATASET_MANIFEST.json`：分数据集的校验与统计结果。
- `Chain-of-Thought_Generation/experiment_design.md`：原目录中的空白实验设计占位文件。

当前快照不含 SSR 训练代码或已训练权重，也不代表完整复现 SSR-D。

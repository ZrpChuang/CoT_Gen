# CoT_Gen

逆向构造 CoT 的实验数据准备目录。当前保存数据来源、固定版本、校验清单和下载工具，尚未开展正式训练。

## 数据

| 目录 | 数据 | 数量 |
|---|---|---:|
| `training_data/lmarena_140k/` | LMArena 原始用户对话 | 135,634 |
| `test_data/arena_hard_v2/` | ArenaHard v2 | 750 题（500 hard + 250 creative） |
| `test_data/eq_bench3/` | EQ-Bench3 | 46 个场景 |
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

# 9B CoT 格式研究

先看 [成绩与结论](RESULTS.md)。本目录保留一套维护代码、固定训练集和逐题结果；历史过程与缓存在外部产物库中按用途压缩保存。

**当前结论：正文 NLL 选择格式的优势尚未确定；confirm200 为 53.25 分，95% 区间 [48.00, 58.25]。四 benchmark 终评完成 7/16 项，已停止。**

| 目录 / 文件 | 用途 |
|---|---|
| RESULTS.md | 开发集、确认集、部分终评成绩与解释边界 |
| pipeline.py | 统一运行入口 |
| core/ | LoRA训练、开发集评测、持久请求缓存、格式选择、汇总与归档读取 |
| configs/ | 各实验配方、冻结协议、本机路径、原始证据哈希、产物索引 |
| data/ | 10份最终训练集、固定screen100和confirm200；不散放逐条中间缓存 |
| results/ | 训练记录、开发集、确认集、四benchmark的逐题回答、评分和指标 |
| docs/ | [文件用途](docs/FILES.md)、[阶段历史](docs/HISTORY.md)、[整理记录](docs/CLEANUP.md)、[保留与删除记录](docs/REVIEW.md) |
| logs/ | 历史运行日志的统一读取入口 |

## 常用命令

```bash
python pipeline.py report
python pipeline.py validate
PYTHONPATH=core python -m unittest discover -s core/tests -v
ruff check --config configs/pyproject.toml .
ruff format --check --config configs/pyproject.toml .
```

新训练显式指定数据、输出目录和GPU；输出放在本研究目录外。本机环境配置在 configs/local_paths.json，模板为 configs/local_paths.example.json；依赖见 configs/requirements.txt。

```bash
python pipeline.py train --data data/train/body_format.jsonl \
  --output /path/to/new_run --model Qwen_35_9B --gpus 0,1,2,3 --dry-run
python pipeline.py dev -- --help
```

train 的 dry-run 只打印命令；正式训练移除该参数。开发评测对新实验先 freeze 协议，再 generate、judge、summarize，完整参数由 dev 帮助列出。正文格式选择与匹配随机的纯函数在 core/formats.py。

## 历史证据与模型

外部产物库：`/mnt/bn/vai-llm-hl2/ruipeng/CoT_Analyse_artifacts/auto_research/pattern_20261004/`。configs/artifacts.json 列出现存归档、保留adapter和已删权重状态；docs/ARCHIVE_INDEX.tsv.gz 可按原路径查找文件。

历史源码、阶段计划、巡检与逐条请求缓存都保存在带 MANIFEST.jsonl 的归档中。恢复某个归档到新的外部目录：

```bash
python pipeline.py restore --archive /path/to/bundle.tar.gz --output /path/to/restored_evidence
```

恢复工具还会校验归档内文件哈希；链接目标单独记录在 RESTORED_LINKS.json。旧manifest保留当时的路径和源码哈希，维护代码用于新实验，复核旧实验时使用归档中的原始源码快照。复跑不是自动续跑原终评；本次整理未新增模型请求或训练。

## 2026-10-23 存储清理

已删除停止研究的10组合并模型和10组训练断点，释放180.49 GiB；10组最终LoRA、原20个证据归档、数据及逐题结果保留，另保留1个配置/训练元数据归档。原状态续训不可恢复，复跑推理需使用基座和LoRA重新合并。详细清单见[docs/REVIEW.md](docs/REVIEW.md)，现存位置及删除状态见`configs/artifacts.json`和外部`WEIGHTS.json`。当前10K实验不在清理范围。

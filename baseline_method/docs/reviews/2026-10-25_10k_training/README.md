# 首批10K训练复盘备份

作者：张瑞鹏（ZrpChuang）。审计日期2026-10-25，Git备份日期2026-10-26。
训练日期2026-10-18，评测完成核验日期2026-10-24。

## 阅读顺序

1. [完整复盘](REVIEW.md)：测试总表、训练曲线、数据与代码问题、证据边界及后续建议。
2. [训练曲线](figures/training_curves.png)、[评测推理长度](figures/reasoning_length.png)、[训练数据分布](figures/data_distribution.png)。每图另存可编辑SVG。
3. [抽样评价](SAMPLES.md)：27组输入的人工审查摘要及失败类型，不含原文。
4. [训练记录](evidence/training/)与[统计检查](evidence/checks/)：支持报告的数值和检查结果。
5. [完整性与来源清单](manifest.json)：本备份SHA256及受审源码的固定提交链接。

## 目录用途

| 内容 | 用途 |
|---|---|
| `review.py` + `requirements.txt` | 从已保存统计重绘三张图 |
| `figures/` | 三张图，各含PNG/SVG |
| `evidence/training/metrics.json` | 四方法各63条记录，62个loss窗口加最终总结 |
| `evidence/training/training.json` | 实际完成配置；机器路径以占位符替换 |
| `evidence/training/versions.json` | 审计时已装训练库版本 |
| `evidence/training/final_verification.json` | 最终分数、分母、metric SHA256和完成范围 |
| `evidence/training/historical_lr_control.md` | 旧4,175条数据的学习率对照，不能与新10K混用 |
| `evidence/checks/` | 配对、长度、编码、tokenizer、SSR与约束变化的聚合检查 |

长期训练及数据代码已在仓库中，无需重复复制：[训练实现](../../../../auto_research/scaling/core/train.py)、[生成实现](../../../../auto_research/scaling/core/data.py)、[token序列构造](../../../../auto_research/scaling/core/common.py)。受审版本及SHA256见manifest中的`reviewed_sources`；报告里的行号链接固定到该提交。

## 重绘

建议Python 3.12及以上，在独立环境安装依赖。在本目录运行：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
MPLCONFIGDIR=.matplotlib-cache .venv/bin/python review.py
```

仅读取已保存的曲线与统计，不调用教师或裁判API，不训练、不加载9B模型。PNG/SVG的渲染元数据可能随重绘变化；清单记录的是备份时文件哈希。

本备份能重绘图表和追溯统计，不能脱离私有数据与权重重新执行全部审计。`source_sha256`指向原始未脱敏证据的指纹；它不表示原始证据已随Git发布。

## 备份边界

不包含密钥、SSH/机器配置、权重、checkpoint、原始对话/回答/裁判文本、缓存、日志包或论文全文。它们按既有约定保留于产物库，本次未删除或移动。

`training.json`使用`${BASE_MODEL}`与`${ARTIFACT_ROOT}`表示原物理位置；`data_statistics.json`移除了单条原文示例，只保留聚合统计与生成来源。`tokenizer_checks.json`中的模型位置也使用相同占位符。

本次是已有结果备份，没有启用新训练、生成、评测或定时任务。首批10K完成不等于100K或全研究完成。

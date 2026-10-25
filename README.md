# CoT_Analyse

先看成绩，再按实验进入代码和原始证据。

| 内容 | 阅读入口 | 运行入口 |
|---|---|---|
| 四种逆向 CoT 方法 × 两模型 × 四 benchmark | [总表与结论](baseline_method/docs/RESULTS.md) | [统一流水线](baseline_method/README.md) |
| GPT首批10K：4种9B训练与16项评测完成；后续暂缓 | [进度与 TODO](baseline_method/docs/SCALE_100K_TODO.md) | [远端队列与效率规则](auto_research/scaling/README.md) |
| 原始模型、27B effort 对照 | [基座结论](test_scripts/Base_model_result_conclusion.md) | 各模型目录中的冻结评测代码 |
| CoT 格式研究与尚未完成的终评 | [研究结论](auto_research/pattern_20261004/RESULTS.md) | 研究目录中的阶段说明 |
| 固定训练源、采样索引和四个测试集 | [数据说明](docs/data/DATASETS.md) | training_data/、test_data/ |
| 本次整理范围、逐文件用途和待删除项 | [整理方案](baseline_method/docs/PLAN.md)、[文件用途](baseline_method/docs/FILES.md)、[待 review](baseline_method/docs/REVIEW.md) | — |

## 当前结果范围

2026-10-25最新复核：首批10K四种9B均训练313步，16项评测与独立核验完成，三项benchmark明显退化。曲线、数据及脚本问题见[训练复核](baseline_method/docs/TRAINING_REVIEW.md)。以下2026-10-12方案与旧实验数量均为历史记录；后续100K、其他对照及27B未启动。

2026-10-12 用户纠正方案：GPT 生成答案与四方法说明，旧权重全部退出训练来源，15 项 9B 从原始基座重训，27B 的 4 项继续暂缓。旧 `scale_100k_20261011` 已退休，新产物为 `CoT_Analyse_artifacts/gpt_fresh_100k_20261012`。详见 [训练入口](auto_research/scaling/README.md) 和 [TODO](baseline_method/docs/SCALE_100K_TODO.md)；实际运行读取新产物 reports/STATUS.md 并核对进程。用户随后明确恢复持续监测与空闲保卡：Codex 每小时复核，远端队列每 15 秒检查，保卡自动让位于正式任务。

- 基座：五个模型、四个 benchmark，另有 27B effort 对照。
- 逆向 CoT：NEU、SUP、AUG-SUP、SSR，共 8 次训练、32 项评测，每次训练 131 步；总表首行保留未训练 Baseline。独立 9B NEU 低学习率对照为另 1 次训练、4 项评测。
- 格式研究：开发与 confirm200 已完成；冻结终评最后记录为 7/16，现有结果不能支持完整终评的胜者结论。

## 目录约定

baseline_method/ 顶层只保留阅读入口、流水线、core/、configs/、docs/、data/ 和四种方法目录。各方法下按 benchmark → 模型保存结果；CoT 和训练样本放在方法根目录。日志集中在 baseline_method/data/logs/，基座历史日志位于各 runtime/logs.tar.gz，归档内带逐文件 SHA256 清单。

固定数据和评测实现保留原始协议。数据来源清单在 [docs/data/sources.lock.json](docs/data/sources.lock.json)，各数据集另有 DATASET_MANIFEST.json；第三方数据与代码遵守各自许可和数据卡。公开快照不等同于已复现论文的全部数据版本；本项目的小规模训练设置和本地 EQ/Arena 指标限制见实验结论。

模型权重、checkpoint 和独立诊断产物位于同级 CoT_Analyse_artifacts/baseline_method/，实验目录通过链接引用。原始回答、评分和机器配置保留在本地，不进入 Git。本次整理的变更记录在同级 CoT_Analyse_cleanup_backup_20261010/layout_20261010/；此前一次性脚本的备份也在该备份库。待 review 的唯一实验资料当前保留。

CoT格式研究现有独立pipeline.py入口；维护代码、最终训练集和逐题结果已分开，历史缓存/巡检和权重位于外部产物库。详见 [研究目录](auto_research/pattern_20261004/README.md) 和其 docs/CLEANUP.md。

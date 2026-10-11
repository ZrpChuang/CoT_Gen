# 训练复核：为什么多数 benchmark 下降

复核日期：2026-10-11。范围为原始四方法 × 两模型，以及已完成的独立 NEU/9B 低学习率对照。此次仅做 CPU 读取与检查，没有新增训练、推理或裁判请求。

## 1. 原始成绩

每格为 **Qwen3.5-9B / Qwen3.8-27B**，分数越高越好。原始分组表和逐项 metrics 链接见 [RESULTS.md](RESULTS.md)。

| 方法 | IFEval | MultiChallenge | EQ-Bench3 | Arena-Hard-v2 |
|---|---:|---:|---:|---:|
| Baseline | 90.76 / 90.57 | 55.31 / 57.14 | 30.80 / 45.02 | 12.64 / 41.57 |
| NEU | 73.38 / 81.15 | 33.70 / 42.49 | 36.96 / 51.04 | 3.01 / 15.18 |
| SUP | 74.12 / 81.15 | 32.97 / 43.22 | 38.89 / 50.39 | 3.22 / 14.11 |
| AUG-SUP | 73.20 / 81.89 | 30.77 / 45.05 | 36.56 / 51.11 | 3.42 / 15.36 |
| SSR | 70.98 / 81.33 | 32.97 / 41.03 | 36.89 / 52.15 | 2.93 / 14.89 |

32 个训练后成绩中，24 个低于基座、8 个高于基座：全部 EQ 上升，其余三项全部下降。不同 benchmark 量纲和评分方式不同，不把它们简单平均成总分。

所有主实验：4,175 条共同配对样本，五类各 835 条，BF16 LoRA r16/alpha32/dropout0.05，学习率 1e-4，global batch32，4 个进程，1 epoch，131 步，最大序列 8,192。Baseline 为 0 步。

指标：IFEval 题级严格通过率；MultiChallenge Astra 裁判总体通过率；EQ 为本地 Astra rubric 0–100；Arena 为本地 500 道 hard prompt 加权胜率。EQ/Arena 不是官方排行榜分数。

## 2. 有对照证据的因素：更新幅度

| 9B 配置 | IFEval | MultiChallenge | EQ | Arena |
|---|---:|---:|---:|---:|
| Baseline | 90.76 | 55.31 | 30.80 | 12.64 |
| NEU，LR 1e-4 | 73.38 | 33.70 | 36.96 | 3.01 |
| NEU，LR 1e-5 | 81.15 | 39.93 | 31.76 | 11.72 |

同一训练数据、131 步和其他配方，仅降低学习率后，IF/MC/Arena 恢复，EQ 回落。已存对照报告给出的 IFEval 成对差值为 +7.76 个百分点，题级 bootstrap 95% 区间 [3.88, 11.83]；不包含训练、生成和评分随机性。IFEval 两道题的原评分存在随机处理，单次约可波动 0.37 个百分点。

**判断：更新幅度影响退化程度，证据较强；1e-5 也没有整体恢复到基座，学习率不是完整解释。** 单次对照不等于已找到最佳学习率，也不能证明 CoT 变短造成退化。

对照材料：`/mnt/bn/vai-llm-hl2/ruipeng/CoT_Analyse_artifacts/baseline_method/diagnostics/lr_control_20261004/CONTROL_RESULT.md`。

## 3. 本次发现的数据风险：答案点评被当作作答前的 CoT

实际编码为 `问题 -> <think>合成 CoT</think> -> 参考答案`，问题不参与 loss，CoT 与答案共同参与 loss。

抽查五类样本发现，不少合成 CoT 是对已有回复的说明、归纳或评价。例如首条代码类样本的 NEU 开头为 “The response provides a JSON template”，SSR 骨架包含 “Describe the supplied response’s structural organization”。这些文本依赖已给出的答案；学生输入只有问题，训练时却要先生成这类回顾性文本。

全量词面筛查（每方法 4,175 条）：

| 方法 | 明确提到已有答案/回复 | 明确说明不是内部推理的措辞 | high-level 解释/总结措辞 |
|---|---:|---:|---:|
| NEU | 1092 | 395 | 1476 |
| SUP | 650 | 623 | 1652 |
| AUG-SUP | 90 | 432 | 1412 |
| SSR | 2739 | 3164 | 3049 |

这些是预先列明正则的命中数，类别可重叠，**不是错误率、拒绝率或因果贡献率**。正常解释也可能命中；抽查样本用于说明具体风险，不能推断所有样本均不合格。SSR 4,175 条全部保留 skeleton，未发现骨架被删除。

此前全量分词审计显示平均 CoT token 为 NEU 237、SUP 160、AUG-SUP 152、SSR 304；共同答案平均 1,481 token。按两部分 token 数估计，答案占约 83%–91%；这不是实际 loss 数值贡献比例。因此，这轮训练同时包含大量普通答案蒸馏，不能把表现变化全部归因于 CoT 方法。

参考答案由 Qwen3.8-Max 生成，CoT 由 GPT-6 Astra 依据 Q/A 合成；已有门禁主要保证输出格式、配对和长度，没有证明答案及 CoT 的语义正确性。抽查中还存在 CoT 指出参考答案有未验证假设、但仍保留该原答案作监督的情况。训练仅取单轮问题，多轮约束保持的覆盖不足亦是待验证因素。

**判断：训练目标与使用场景存在明确风险；它是否以及多大程度造成退化，尚缺隔离对照。** EQ 上升可能包含表达风格变化，当前不能区分风格适配与真实能力提升。

## 4. 工程核查结果与范围

- 本次重新读取全部 16,700 条训练记录：四方法的 ID、顺序、问题、答案一致；4,175 个 ID 唯一；训练文件与 manifest、8 个完成记录的 SHA256 一致。
- 8 组均为 131 步、1 epoch、smoke=false；训练日志中的 loss、学习率、grad norm 均为有限值，最大已记录 grad norm 约 0.705；未见明显数值发散。
- 复用已有 33,400 条编码的全量审计：问题 mask 正确，CoT+答案+结束符参与 loss，最大长度 8,176，无截断。本次又对每组 3 条样本复查 mask、结束符和基座/合并模型 tokenizer 一致性，24/24 通过。
- 8 个 adapter 中，9B 各有 496 个 LoRA tensor、27B 各有 992 个，未见 vision/visual adapter；每组抽查 3 个 LoRA-B tensor 均非零且有限。保存代码从实际训练后的模型合并；8 个评测部署路径均指向对应合并产物。未逐 tensor 比较所有合并权重，因此不声称完成全量权重数值审计。
- 32 组方法评测的解码参数、数据 hash 与各自 Baseline 一致（排除模型服务名称和地址）；原审计均 verified，32 份 metrics hash 与原审计一致。本次从 40 份 metrics 重新读取总表。
- 27B tokenizer 加载时出现 regex pattern 警告；当前抽样一致，历史训练/服务环境也有 64 条跨环境一致记录。罕见文本是否受影响未全量跨环境验证，暂不能据此解释方法相对基座的退化。

| 方法 | 模型 | step 5 训练 loss | step 130 训练 loss | 最大已记录 grad norm |
|---|---|---:|---:|---:|
| NEU | Qwen_35_9B | 1.3103 | 0.9722 | 0.4710 |
| SUP | Qwen_35_9B | 1.3439 | 0.9907 | 0.7054 |
| AUG-SUP | Qwen_35_9B | 1.3607 | 1.0050 | 0.5431 |
| SSR | Qwen_35_9B | 1.5726 | 0.9959 | 0.5056 |
| NEU | Qwen_38_27B | 0.8597 | 0.7217 | 0.1896 |
| SUP | Qwen_38_27B | 0.8532 | 0.7146 | 0.1553 |
| AUG-SUP | Qwen_38_27B | 0.8526 | 0.7144 | 0.1558 |
| SSR | Qwen_38_27B | 1.0002 | 0.7538 | 0.1637 |

以上是不同 batch 的已记录训练 loss，不是固定样本上的学习曲线。8 组没有开发集 eval_loss，也没有依据独立开发集选择早期 checkpoint。因此不能仅凭训练 loss 下降定性过拟合、证明训练充分，或决定增加 epoch。

机器可读复核记录：`/mnt/bn/vai-llm-hl2/ruipeng/CoT_Analyse_artifacts/baseline_method/diagnostics/training_review_20261011.json`。词面筛查模式、抽查 ID、指标原值、训练配方和核查细项均在其中。此前全量格式审计在同目录 `training_format_audit.json`。

## 5. 论文比较和后续实验的边界

原论文 PDF 第 8 页 Table 3 中 NEU 也是训练后方法，没有提供四个核心 benchmark 的未训练基座行；SSR 优于 NEU 不等于 SSR 优于未训练基座。Table 4 则列出 w/o Train，且许多训练后 OOD 成绩低于基座。

论文第 29–30 页列出 100k、4 epoch、32,768 序列、batch256、LR4e-5；本轮数据规模、模型、教师、LoRA 和配方均不同。附录 D.2 描述 SSR-D 的骨架与 reasoning 双目标及专门训练的教师；本地原表没有实现 SSR-D，不应当作该流程的完整复现。主表 SSR 的精确学生序列化仍需官方训练代码才能最终核定。

论文来源：`/Users/bytedance/Desktop/file/CoT-prblom/01_SSR-D_2602.14469.pdf`。

后续 `auto_research/pattern_20261004` 已做过低 LR、空 CoT、格式变体等训练，不能再把 answer-only 当成尚未做过。该项目 screen100 的空 CoT 为49.0、NEU低LR为45.0，区间均覆盖中性50；空 CoT仍保留 thinking 控制符，不是关闭 thinking 的纯答案训练。四 benchmark 终评仅完成7/16，未证明新方法优于基座。见 [后续研究结果](../../auto_research/pattern_20261004/RESULTS.md)。

## 6. 建议的处理顺序

1. 先人工审计并制定训练目标门禁：题目约束是否满足、参考答案是否正确、解释是否可由问题支持、是否存在对已给答案的点评及与答案矛盾的内容。合格目标应为可核验的解题步骤或约束清单。
2. 保留已有 NEU 学习率对照作为更新幅度证据；对其余方法不外推单次对照的结论。
3. 后续如重训，在独立开发集上记录任务表现并保留早期 checkpoint；以预先固定的规则选择停止步数，最终测试集仅用于终评。
4. 把后续研究已完成、未完成的评测分开解释，先确定最关键的待验证假设，再安排额外实验。

本次结论：更新幅度已有对照证据，合成目标的回顾性解释与答案蒸馏占比是优先的数据风险；未发现可以直接解释全部退化的 mask、截断或部署路径错误。当前数据不支持整体性能提升，也不能给出单一根因。

## 7. W&B、数据划分与论文差距补充

### W&B 同步范围

截至本次核查，19组历史训练均有 `synced_complete` 上传记录：原始四方法×两模型8组、NEU/9B低LR对照1组、后续格式研究10组。项目为 [cot-analyse](https://wandb.ai/zrpchuang-south-china-university-of-technology/cot-analyse)。Baseline未训练，不计入19组。

确认上传的内容为 train/loss、grad_norm、learning_rate、epoch、global_step，以及训练配置、数据hash和完成信息。原Trainer每5步记录，最后曲线点为130，summary记录最终131步；不是缺了最后一步训练。同步上传时间也不是实际训练时间，应使用 train/global_step 横轴。低LR恢复训练的聚合loss/runtime已有范围修正，不能按完整epoch总量解读。

此次核验归档同步记录，当前无可用W&B登录凭据，未重新逐条核验云端history。不能把19个同步标记解释为全部数据集、模型权重及benchmark结果也已上传。当前核心训练代码仍 `report_to="none"`；历史同步依赖独立sidecar，整理后的主流水线尚未集成未来训练的自动W&B上报。

原始8组记录在 `CoT_Analyse_artifacts/baseline_method/wandb_sync/`；低LR记录在其 `diagnostics/lr_control_20261004/wandb_sync/`；后续10组状态保存在 `CoT_Analyse_artifacts/auto_research/pattern_20261004/archives/training/runs.tar.gz` 的各 `wandb_sync/state.json`。核查索引在 `CoT_Analyse_artifacts/baseline_method/diagnostics/tracking_data_review_20261011.json`。

### 数据配比与测试集关系

原始来源为 LMArena `arena-human-preference-140k`，固定revision `6322995ab34d7c2693e3f47dd13fa5caa0789a74`，本地135,634条记录。抽样限制为英语、纯文本、完整单轮、同一会话首个投票、A/B用户输入相同、问题不超过1024 token。

| 配额类别 | 候选训练 | 最终训练 | 最终占比 | 保留开发 |
|---|---:|---:|---:|---:|
| 代码 | 1,000 | 835 | 20% | 200 |
| 数学 | 1,000 | 835 | 20% | 200 |
| 指令遵循 | 1,000 | 835 | 20% | 200 |
| 创意写作 | 1,000 | 835 | 20% | 200 |
| 其他 | 1,000 | 835 | 20% | 200 |
| 总计 | 5,000 | 4,175 | 100% | 1,000 |

类别来自原标签的互斥配额分配；原标签可以重叠，other是所选四类未命中的剩余项。20%是本地人为均衡，未在论文中找到相同配比要求。

候选训练因序列超8K排除297条、参考或方法缺失14条、控制符5条，剩4,684条；再为保证每类835条去掉509条，得到4,175条。四方法与两模型共用这批问题和参考答案，变化的是CoT；不是每方法另采4,175个问题。

1,000条开发数据没有用于主实验训练或eval_loss。后续研究从中取互不重叠的screen100（每类20）和confirm200（每类40）。本次重新确认候选训练与开发的ID及prompt hash交集为0；最终训练是候选子集。

四个benchmark来自独立文件：IFEval541题、MultiChallenge273段对话、EQ45个场景、Arena750题（主表取其中500道hard）。抽样manifest记载：针对四个冻结测试集的所有用户轮次，排除规范化精确匹配，以及至少3个共同13词片段且覆盖较小片段集至少50%的近重复。该去污染方法不保证语义级完全无污染。

训练和测试是跨数据集评测，不是把四个benchmark按比例拆成train/test。能力维度相关，但任务分布不完全一致；尤其主训练严格单轮，MultiChallenge和EQ包含多轮场景。Arena训练源与测试源均涉及Arena生态，仍使用独立题目文件及上述去重策略。

来源与划分依据：服务器 `training_data/lmarena_140k/DATASET_MANIFEST.json`、`training_data/sample/manifest.json` 和 `baseline_method/data/prepared_status.json`。

### 相对论文的差距

| 项目 | 论文 | 本地主实验 |
|---|---|---|
| 数据量 | 100,000条LMArena问题 | 4,175条，约为论文4.18% |
| 配比 | 未找到五类各20%的要求 | 五类固定均衡、仅英语单轮 |
| 学生 | Qwen3-8B/32B、4B-Thinking、NBG4-3B | Qwen3.5-9B、Qwen3.8-27B |
| 参考答案 | Qwen3-Max | Qwen3.8-Max |
| 逆向解释生成器 | Qwen3-235B-Instruct-2507 | GPT-6 Astra |
| 训练框架 | OpenRLHF、ZeRO3，表20未列LoRA设置 | BF16 LoRA rank16 |
| 序列上限 | 32,768 | 8,192 |
| global batch | 256 | 32 |
| 学习率 | 4e-5 | 1e-4；独立对照1e-5 |
| epoch | Max Epochs为4 | 1 |
| warmup | 1% | 3% |
| sample packing | 开启 | 当前脚本未使用 |
| SSR-D | 专门训练SSR教师后再生成蒸馏数据 | 尚未实现 |
| 核心评分 | 论文指定协议 | 本地裁判与EQ/Arena口径不同 |
| OOD | GPQA-Diamond、AIME2025 | 原主表未覆盖 |

倍数仅描述配方数值，不代表不同训练方式下的等价更新强度。当前属于小规模方法变体验证，多个重要变量同时改变，无法把退化单独归因于论文方法，也不能声称完整复现。优先需要对齐的内容是生成目标的语义与质量、SSR/SSR-D流程、学生/教师选择及评分协议，然后再考虑扩展规模。

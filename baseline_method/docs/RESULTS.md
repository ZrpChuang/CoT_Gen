# 原始基座与四种方法的 benchmark 结果

**历史结果，非本轮重训成绩。** 2026-10-12 用户弃用旧训练权重，改为 GPT 生成共享答案和四方法说明，从原始基座重新训练。旧分数保留用于审计退化；新成绩仅写入 `CoT_Analyse_artifacts/gpt_fresh_100k_20261012/reports/RESULTS.md`，不与本表混填。

## 2026-10-11 训练复核

详见 [TRAINING_REVIEW.md](TRAINING_REVIEW.md)：重新核对40份指标、8组训练和32组评测配置；新增合成目标语义风险分析，区分已有学习率对照证据与待验证解释。后续 answer-only/格式变体实验见 auto_research，不能视为尚未进行。

首行是未经本项目训练的原始模型（Baseline）；其余每行是一种 CoT 构造方法。点击分数可打开对应模型的原始 metrics.json；分数越高越好。

**核心结论：四种方法在两个模型上都提高了本地 EQ 分数，但 IFEval、MultiChallenge、Arena 均低于原始基座，当前结果不支持整体性能提升。** 9B NEU 降低学习率后部分恢复，四项成绩仍未整体超过基座。

<table>
  <thead>
    <tr><th rowspan="2">方法</th>
      <th colspan="2">IFEval</th>
      <th colspan="2">MultiChallenge</th>
      <th colspan="2">EQ-Bench3</th>
      <th colspan="2">Arena-Hard-v2</th>
    </tr>
    <tr>
      <th>Qwen3.5-9B</th>
      <th>Qwen3.8-27B</th>
      <th>Qwen3.5-9B</th>
      <th>Qwen3.8-27B</th>
      <th>Qwen3.5-9B</th>
      <th>Qwen3.8-27B</th>
      <th>Qwen3.5-9B</th>
      <th>Qwen3.8-27B</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Baseline（原始模型）</td><td><a href="../../test_scripts/Qwen_35_9B/ifeval/metrics.json">90.76</a></td><td><a href="../../test_scripts/Qwen_38_27B/ifeval/metrics.json">90.57</a></td><td><a href="../../test_scripts/Qwen_35_9B/multi_challenge/metrics.json">55.31</a></td><td><a href="../../test_scripts/Qwen_38_27B/multi_challenge/metrics.json">57.14</a></td><td><a href="../../test_scripts/Qwen_35_9B/eq_bench3/metrics.json">30.80</a></td><td><a href="../../test_scripts/Qwen_38_27B/eq_bench3/metrics.json">45.02</a></td><td><a href="../../test_scripts/Qwen_35_9B/arena_hard_v2/metrics.json">12.64</a></td><td><a href="../../test_scripts/Qwen_38_27B/arena_hard_v2/metrics.json">41.57</a></td></tr>
    <tr><td>NEU</td><td><a href="../NEU/ifeval/Qwen_35_9B/metrics.json">73.38</a></td><td><a href="../NEU/ifeval/Qwen_38_27B/metrics.json">81.15</a></td><td><a href="../NEU/multi_challenge/Qwen_35_9B/metrics.json">33.70</a></td><td><a href="../NEU/multi_challenge/Qwen_38_27B/metrics.json">42.49</a></td><td><a href="../NEU/eq_bench3/Qwen_35_9B/metrics.json">36.96</a></td><td><a href="../NEU/eq_bench3/Qwen_38_27B/metrics.json">51.04</a></td><td><a href="../NEU/arena_hard_v2/Qwen_35_9B/metrics.json">3.01</a></td><td><a href="../NEU/arena_hard_v2/Qwen_38_27B/metrics.json">15.18</a></td></tr>
    <tr><td>SUP</td><td><a href="../SUP/ifeval/Qwen_35_9B/metrics.json">74.12</a></td><td><a href="../SUP/ifeval/Qwen_38_27B/metrics.json">81.15</a></td><td><a href="../SUP/multi_challenge/Qwen_35_9B/metrics.json">32.97</a></td><td><a href="../SUP/multi_challenge/Qwen_38_27B/metrics.json">43.22</a></td><td><a href="../SUP/eq_bench3/Qwen_35_9B/metrics.json">38.89</a></td><td><a href="../SUP/eq_bench3/Qwen_38_27B/metrics.json">50.39</a></td><td><a href="../SUP/arena_hard_v2/Qwen_35_9B/metrics.json">3.22</a></td><td><a href="../SUP/arena_hard_v2/Qwen_38_27B/metrics.json">14.11</a></td></tr>
    <tr><td>AUG-SUP</td><td><a href="../AUG-SUP/ifeval/Qwen_35_9B/metrics.json">73.20</a></td><td><a href="../AUG-SUP/ifeval/Qwen_38_27B/metrics.json">81.89</a></td><td><a href="../AUG-SUP/multi_challenge/Qwen_35_9B/metrics.json">30.77</a></td><td><a href="../AUG-SUP/multi_challenge/Qwen_38_27B/metrics.json">45.05</a></td><td><a href="../AUG-SUP/eq_bench3/Qwen_35_9B/metrics.json">36.56</a></td><td><a href="../AUG-SUP/eq_bench3/Qwen_38_27B/metrics.json">51.11</a></td><td><a href="../AUG-SUP/arena_hard_v2/Qwen_35_9B/metrics.json">3.42</a></td><td><a href="../AUG-SUP/arena_hard_v2/Qwen_38_27B/metrics.json">15.36</a></td></tr>
    <tr><td>SSR</td><td><a href="../SSR/ifeval/Qwen_35_9B/metrics.json">70.98</a></td><td><a href="../SSR/ifeval/Qwen_38_27B/metrics.json">81.33</a></td><td><a href="../SSR/multi_challenge/Qwen_35_9B/metrics.json">32.97</a></td><td><a href="../SSR/multi_challenge/Qwen_38_27B/metrics.json">41.03</a></td><td><a href="../SSR/eq_bench3/Qwen_35_9B/metrics.json">36.89</a></td><td><a href="../SSR/eq_bench3/Qwen_38_27B/metrics.json">52.15</a></td><td><a href="../SSR/arena_hard_v2/Qwen_35_9B/metrics.json">2.93</a></td><td><a href="../SSR/arena_hard_v2/Qwen_38_27B/metrics.json">14.89</a></td></tr>
  </tbody>
</table>

## 方法全称

- NEU：Neutral Prompting（中性提示）。
- SUP：Semantic Suppression（语义抑制）。
- AUG-SUP：Augmented Suppression（增强抑制）。
- SSR：Structural Skeleton-guided Reasoning（结构骨架引导推理）。

## 训练步数

Baseline 为原始模型，未做本项目训练（0 步）。NEU、SUP、AUG-SUP、SSR 各在 Qwen3.5-9B 和 Qwen3.8-27B 上训练 131 步：共 8 组，每组使用 4,175 条配对样本、1 个 epoch。独立的 Qwen3.5-9B NEU 低学习率对照也训练了 131 步，但不计入上表四方法。

## 指标与文件

IFEval 为题级严格通过率；MultiChallenge 为 GPT-6 Astra 裁判的总体通过率；EQ-Bench3 为 Astra rubric 0–100 分；Arena-Hard-v2 为 500 道 hard 题的加权胜率（完整评测另含 250 道创作题）。

各方法的 CoT 样本在 {方法}/cot.jsonl；逐题回答、评分记录、配置和指标在 {方法}/{benchmark}/{模型}/。Baseline 的原始逐题结果在 ../../test_scripts/{模型}/{benchmark}/。独立的 9B NEU 低学习率诊断结论见本文下方的实验说明，完整产物在工作区外的 `CoT_Analyse_artifacts/baseline_method/diagnostics/`。

---

## 实验设置与复现范围

5,000 条候选训练问题按五类均衡采样；开发集另留 1,000 条。最终训练只使用四方法共同有效、两模型均不截断的配对样本，并保持五类平衡。

本轮为小规模 BF16 LoRA 方法复现：1 epoch、rank=16、序列上限 8,192；不是论文 100k / 4 epoch / 32K 的精确复现。27B 测试 effort=medium；9B 保留原 thinking 设置。测试集及裁判沿用基座评测。

各方法的 `cot.jsonl` 为生成的 CoT，`train.jsonl` 为最终配对训练数据；模型子目录包含 adapter、合并权重、训练记录和 evaluation。

采样索引：`../../training_data/sample/`。现行流水线入口、日志和本地配置见 [README.md](../README.md)。

CoT 是依据已有答案新构造的训练文本，不是声称还原教师实际内部推理；参考答案由模型生成，并非验证过的标准答案。

最终共同训练集为五类各835条，共4,175条；使用固定候选索引、共同参考答案和8K配对过滤。候选生成并非全部完成，以 prepared_status.json 的冻结训练集标记为准。参考请求的长度失败曾允许一次32K重试，学生训练上限仍为8K。

## 退化诊断与单变量对照

原始8组与32项评测已完成；诊断结论见本文下方的退化诊断；逐题与格式核查材料保存在工作区外的产物库。
独立对照：NEU/9B 学习率 1e-4 → 1e-5，其余配方与评测不变；核心成绩见下表，完整审计记录保存在工作区外的产物库。
该对照不替换原四种方法成绩。

服务器迁移及评分复核记录保存在产物库中。IFEval两题存在原评分随机性，单次分数可波动至约0.37个百分点；原结果和评分代码保留不变。

## 9B NEU 低学习率对照（独立结果）

| 配置 | IFEval | MultiChallenge | EQ | Arena难题 | CoT token估计 |
|---|---:|---:|---:|---:|---:|
| BASE | 90.76 | 55.31 | 30.80 | 12.64 | 6,988,311 |
| NEU_lr1e-4 | 73.38 | 33.70 | 36.96 | 3.01 | 348,828 |
| NEU_lr1e-5 | 81.15 | 39.93 | 31.76 | 11.72 | 3,675,667 |

四项完整评测审计通过。降低LR后IF/MC/Arena恢复、EQ回落，尚未整体超过基座；不能把长度变化视为因果解释。
恢复段训练23.3分钟；评测至汇总51.4分钟（其中推理服务41.0分钟），不含停机和首段训练。
完整低学习率对照材料保存在产物库中。下一步建议仅答案蒸馏对照，当前不自动新增实验。

---

## 复核与诊断

8组训练及32项评测已完成并重新通过原评分审计；4175条训练数据的配对、顺序和哈希一致，四种方法与对应基座的解码配置一致。

## Token 与训练耗时

| 模型 | 方法 | 输出 token | CoT token（重分词估计） | 空 CoT 调用 | 截断调用 | 训练分钟 |
|---|---|---:|---:|---:|---:|---:|
| Qwen_38_27B | BASE | 4,627,108 | 2,601,276 | 0 | 4 | 0.0 |
| Qwen_38_27B | NEU | 2,633,212 | 431,308 | 0 | 5 | 77.7 |
| Qwen_38_27B | SUP | 2,375,874 | 236,637 | 0 | 5 | 74.4 |
| Qwen_38_27B | AUG-SUP | 2,486,075 | 268,481 | 0 | 5 | 73.6 |
| Qwen_38_27B | SSR | 2,655,844 | 497,744 | 0 | 4 | 78.0 |
| Qwen_35_9B | BASE | 8,572,277 | 6,988,311 | 0 | 19 | 0.0 |
| Qwen_35_9B | NEU | 2,672,143 | 348,828 | 0 | 8 | 27.7 |
| Qwen_35_9B | SUP | 2,449,656 | 232,460 | 0 | 11 | 26.9 |
| Qwen_35_9B | AUG-SUP | 2,500,957 | 217,759 | 0 | 7 | 26.1 |
| Qwen_35_9B | SSR | 2,832,799 | 516,519 | 0 | 10 | 27.5 |

输出 token 使用服务端 usage；CoT 使用各模型原 tokenizer 重分词估计，未含解析时去掉的控制符。EQ 按实际 turns＋debrief 调用统计，含 Arena 全部750题。请求 latency 的累加不是并发墙钟耗时；训练分钟为 Trainer 实测，不含模型加载和合并保存。

## 解释边界

- loss 下降只能说明训练拟合，当前没有开发集 loss，不能据此决定增加 epoch。
- 当前所有方法均呈 EQ 提升、其他三项下降；不能把这直接归因于逆向 CoT，也不能视为论文复现成功。
- 本轮为4175条、LoRA、1 epoch/8K；模型、教师与原论文不同。需先检查输出长度、错误类型、模板与 loss mask。
- 已重新确认 SSR 的4,175条训练样本全部含骨架；论文主表 SSR 的精确学生序列化仍需官方训练代码核定。

失败案例与逐题诊断记录已归档至工作区外的 `CoT_Analyse_artifacts/baseline_method/diagnostics/`。

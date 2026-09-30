# 基座模型评测总结

评测日期：2026-09-29～30。目的：了解蒸馏前的能力差距。本轮未训练模型，五个模型 × 四个 benchmark 均已完成，完整性检查通过。

## 1. 模型与设置

- **Qwen3.5-4B / 9B**：官方后训练权重，本地 BF16；每个模型各用 2 张 H20，两个单卡实例。
- **Qwen3.8-27B**：本地 BF16 部署，4 张 H20，两个 TP2 实例。
- **GPT-6 Astra**：API 模型 `gpt-6-astra`。
- **Qwen3.8-Max**：API 模型 `qwen3.8-max-0902`。

原有三个模型请求 `reasoning_effort=medium`；新增 Qwen3.5-4B/9B 开启思考，采用官方通用思考采样：`temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5`，不设置 medium 档位。

均请求输出上限 32,768 token，仅用最终回答评分；长度耗尽的结果保留计分。Max 实际返回曾超过该上限，因此并非严格等推理预算比较。“基座”指未经过本项目蒸馏的原始模型。

## 2. Benchmark 简介

| Benchmark | 本轮规模 | 主要测什么 | 本轮计分方式 |
|---|---|---|---|
| IFEval | 541 题、834 条约束 | 是否遵守字数、格式、关键词等明确指令 | 官方程序检查；一题全部约束严格通过才算通过，报告题级通过率 |
| MultiChallenge | 273 题 | 多轮对话中的记忆、指令保持、版本修改和前后一致性 | 给定历史、生成末轮回答；裁判按官方条件判断，通过数 / 273 |
| EQ-Bench3 | 45 个有效场景，118 次模型调用（含追问） | 情绪理解、共情和复杂人际互动 | 按官方评分标准打分，维度均分映射到 0–100，再对场景取平均；不是 ELO |
| Arena-Hard-v2 | 750 题：500 难题 + 250 创作题 | 开放式复杂问题与创作回答质量 | 与固定参考答案比较，交换 A/B 顺序评分；强胜负权重 3、平局计半，报告加权胜率 |

Arena 难题参考答案来自 o3-mini-2025-01-31，创作题来自 Gemini 2.0 Flash。下面主表只展示 **500 道难题**，完整结果另含创作题。

## 3. 基座成绩

分数均为 0–100，越高越好。IFEval 使用官方程序检查，其余三项仅展示 **GPT-6 Astra 裁判**的评分。

| 模型 | IFEval | MultiChallenge | EQ-Bench3 | Arena 难题 |
|---|---:|---:|---:|---:|
| Qwen3.5-4B | 88.91 | 50.55 | 26.43 | 10.52 |
| Qwen3.5-9B | 90.76 | 55.31 | 30.80 | 12.64 |
| Qwen3.8-27B | 90.57 | 57.14 | 45.02 | 41.57 |
| GPT-6 Astra | 95.93 | 72.53 | 82.78 | 98.91 |
| Qwen3.8-Max | 94.27 | 68.50 | 67.33 | 92.56 |

**初步判断：**9B 的 IFEval 比此前 27B 高 0.19 个百分点，MultiChallenge 低 1.83 个百分点；EQ 和 Arena 则有较大差距。相对 Astra，4B/9B 的 MultiChallenge 分别低 21.98/17.22 个百分点；能否通过 CoT 蒸馏转化为收益仍需训练验证。

**解释边界：**Qwen3.5 与 Qwen3.8 的代际和采样设置不同，不能把差异单独归因于参数规模。Astra 参与自身评分，可能存在自评偏差，不能仅据此确定模型的绝对排名。本轮不是 SSR 论文数值的精确复现，也不是官方排行榜成绩；Arena 未做风格控制。

## 4. 结果位置

服务器根目录：`/mnt/bn/vai-llm-hl2/ruipeng/CoT_Analyse/test_scripts/`。

- 模型目录：`Qwen_35_4B/`、`Qwen_35_9B/`、`Qwen_38_27B/`、`GPT/`、`Qwen_Max/`。
- 各自包含：`ifeval/`、`multi_challenge/`、`eq_bench3/`、`arena_hard_v2/`。
- 每个 benchmark 的 `metrics.json` 为最终成绩，`responses.jsonl` 为逐题回答，`judgments*.jsonl` 为评分记录；各模型根目录 `audit.json` 为完整性检查。

新增模型权重：`/mnt/bn/vai-llm-hl2/ruipeng/models/Qwen3.5-4B/`、`/mnt/bn/vai-llm-hl2/ruipeng/models/Qwen3.5-9B/`；均已校验下载文件 SHA256。

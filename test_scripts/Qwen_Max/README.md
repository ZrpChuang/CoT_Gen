# 四项基座评测

固定全量数据：Arena-Hard-v2 750 题（hard 500、creative 250）、IFEval 541 题、MultiChallenge 273 题、EQ-Bench3 45 个有效场景/92 个对话轮次（文件有 46 个标题，但 409 为空，官方解析器同样仅加载 45 个），非 analysis 场景另加官方 debrief。

## 结果

每个 benchmark 子目录只有一套当前结果：`responses.jsonl`（逐题有效返回）、`judgments*.jsonl`（逐题评分）、`metrics.json`（最终分数）、固定配置及 `progress.json`。断点续跑仅补缺失题；失败请求不写成回答，不删低分/拒绝/达到 token 上限的有效样本。发生未解决错误时不发布最终分数。重试成功后删除错误清单。代码、数据、权重、密钥分离。

## 协议

- 使用未做本项目蒸馏的原始模型，启用原生推理。每题一次，请求的输出上限为 32768 token，评测仅使用公开最终回答。各服务实际支持的参数记录在配置和每条响应中。
- IFEval：直接调用 Google 官方检查器，输出 prompt/instruction × strict/loose 四个指标，以 prompt strict 为主。
- Arena：官方分类裁判提示及固定参考回答，A/B 顺序互换评两次。strong win 权重 3，tie 0.5。分开报告 hard 与 creative 的加权胜率，当前不做 style control。不能冒充官方排行榜或论文精确复现。
- EQ：官方场景、模板、多轮与 debrief，调用官方 rubric 格式化/解析；逐场景按有效能力维度平均，再乘 5 映射到 0–100，最后场景等权平均。此处为 rubric 分数，不是 ELO。
- MultiChallenge：保留官方给定历史，只生成最后一轮；按官方 TARGET_QUESTION 的 YES/NO 裁判，匹配 PASS_CRITERIA。一次尝试，通过数/273，同时列各轴。
- 三个候选模型统一使用 GPT-6 Astra 与 Qwen3.8-Max 双裁判，分开报告两套评分。Astra/Max 对自身输出的自评可能有偏差，应结合两套结果判断，不将单一裁判结果当绝对排名。
- 完整性：每个数据 ID 均需存在，生成配置及输入哈希固定；API 错误不计为模型答错，也不从分母剔除。
- 不同模型 tokenizer 与内部推理不可完全等价；记录 native reasoning 与 token/延迟信息，判断蒸馏空间应结合逐题差异，不能仅看总分。

## 运行

`bash <benchmark>/run.sh --stage generate` 生成；`--stage score` 评分；省略 stage 连续执行。

Responses 接口使用 Azure 路由，EQ 同一场景的所有轮次与 debrief 使用固定 session_id 及固定凭证。Astra 返回的 reasoning summary 单独保存，不当作原始 CoT。所有模型 reasoning effort=medium；Astra 不传不支持的采样温度参数。

API 密钥仅从服务器项目外 `/home/tiger/.config/cot-baseline/credentials.json`（权限 600）读取，或由 COT_API_SECRETS 指定。不要把密钥写进代码/结果/Git。

## 实测预算差异

Qwen Max 接口返回的 completion_tokens 包含 reasoning_tokens，实测有样本超过请求的 max_tokens=32768（例如 140699，其中 reasoning_tokens=131072）。因此不能把该参数当作三个接口统一执行的总推理上限。本轮保留冻结请求参数和全部有效返回，不截断或重采样；结果表示这些原生服务配置下的能力，不是严格等计算预算比较。逐题 usage 和 settings 均保留。

Arena 裁判若未给出合法判定标签，会追加统一的裁判任务/引用数据隔离/输出格式提醒后重试一次，并在逐局记录 format_repair。有效判定不重评，不根据胜负选择结果。

Arena 可用 score_progressively.py 与生成并行：只评分已经完整落盘的回答，并记录逐题回答哈希；最终指标仍必须等待全部 750 题和两套评分齐全。

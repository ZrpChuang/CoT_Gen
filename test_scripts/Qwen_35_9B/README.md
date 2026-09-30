# Qwen3.5-9B 基座评测

采用官方 Qwen/Qwen3.5-9B 后训练权重，未做本项目蒸馏。ModelScope 官方镜像下载，逐文件 SHA256 校验；模型位于 deployment.json 指定目录。

- 两个 BF16 单 GPU 实例，开启思考；Qwen3.5 模板使用 enable_thinking，不声称支持 medium 强度档位。
- 与此前 27B 保持相同四套数据、评分代码和 32,768 输出上限；上下文上限 65,536。采样使用 Qwen3.5 官方通用思考配置：temperature=1.0、top_p=0.95、top_k=20、min_p=0、presence_penalty=1.5、repetition_penalty=1，全部场景统一使用，覆盖旧脚本中的 EQ 温度。
- 仅使用 GPT-6 Astra 裁判（medium）。IFEval 使用官方程序检查。
- IFEval 541 题；MultiChallenge 273 题；EQ-Bench3 45 个有效场景、118 次模型调用；Arena-Hard-v2 750 题，主指标为 500 道难题加权胜率。
- EQ 为 0–100 rubric 分数而非 ELO；Arena 不做风格控制；不等同官方榜单或 SSR 精确复现。
- 每题保留一次有效返回，拒绝与长度耗尽均保留；仅对传输或格式失败重试，不按分数筛选。

运行：先用 Python 执行 download_model.py，再用 /home/tiger/.venvs/cot-analyse-ssr/bin/python 执行 run_all.py。按 benchmark 分目录保存唯一一套 responses.jsonl、judgments*.jsonl 和 metrics.json；最终 audit.json 检查全量 ID 与分数。runtime/execution_events.jsonl 记录实际起止时间。模型服务结束时恢复暂停的 GPU keepalive。

初次沿用 27B 采样时观察到重复思考，在评分前整体中止并重置，未选择性保留回答。runtime/initial_sampling_correction.json 记录初始配置和样本计数；当前最终结果仅对应上述官方采样配置。

Arena 使用生成与评分并行流程，逐回答校验哈希，完整 750 题才发布分数。评分与结束检查通过；runtime/timing_summary.json 保存本次正式运行耗时。

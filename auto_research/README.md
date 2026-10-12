# GPT 重训与历史诊断

- [GPT 全新 100K 训练队列](scaling/README.md)：当前维护入口，15 项 9B 从原始基座新建 LoRA；27B 的 4 项暂缓。旧训练权重与生成缓存弃用。
- [范围、部署验证与 TODO](../baseline_method/docs/SCALE_100K_TODO.md)：当前执行状态、数据口径和比较限制。
- `pattern_20261004/`：历史格式选择与对照实验的冻结数据、结果和复用实现；保留证据。

新产物写入 `/mnt/bn/vai-llm-hl2/ruipeng/CoT_Analyse_artifacts/gpt_fresh_100k_20261012/`，`reports/PLAN.md` 为方案，`reports/STATUS.md` 为进度，`reports/RESULTS.md` 为新结果。旧 `scale_100k_20261011` 已退休，禁止恢复。Codex 定时维护保持关闭。

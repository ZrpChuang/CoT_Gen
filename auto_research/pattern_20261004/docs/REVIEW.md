# 需要review的永久删除项

归档与权重当前保留。散文件已归档清理，下面只讨论是否永久删除唯一产物。

| 内容 | 数量 / 大小（十进制GB） | 删除影响 | 建议 |
|---|---:|---|---|
| 训练checkpoint | 10组 / 5.399 | 失去optimizer/RNG与原状态续训 | 仅保留最终baseline时可删，待review |
| 合并服务模型 | 10组 / 188.398 | 直接复跑服务需要重新合并 | 先验证基座+adapter重建，再决定 |
| 最终LoRA adapter | 10组 / 1.932 | 失去训练后的核心小权重 | 保留 |
| 20个原始证据归档 | 0.971 GB | 含付费回答、裁判原始返回、失败尝试和历史源码 | 保留；不建议永久删除 |

产物库：`/mnt/bn/vai-llm-hl2/ruipeng/CoT_Analyse_artifacts/auto_research/pattern_20261004`。每组目录为 models/{方法}/{adapter,model,checkpoints}；详细原文件、大小和原inode见 WEIGHTS.json。

| 方法 | checkpoint GB | 合并模型 GB |
|---|---:|---:|
| answer_only | 0.540 | 18.840 |
| aug_sup_lr1e5 | 0.540 | 18.840 |
| body_format | 0.540 | 18.840 |
| body_matched_random | 0.540 | 18.840 |
| matched_random_format | 0.540 | 18.840 |
| paired_plain | 0.540 | 18.840 |
| paired_steps | 0.540 | 18.840 |
| ssr_lr1e5 | 0.540 | 18.840 |
| student_format | 0.540 | 18.840 |
| sup_lr1e5 | 0.540 | 18.840 |

# 日志

全部历史运行日志集中在外部产物库的 `archives/logs/runtime.tar.gz`，内部 MANIFEST.jsonl 列出原路径、字节数与SHA256。configs/artifacts.json记录绝对路径。

```bash
tar -tzf /mnt/bn/vai-llm-hl2/ruipeng/CoT_Analyse_artifacts/auto_research/pattern_20261004/archives/logs/runtime.tar.gz
```

用根目录 pipeline.py restore 可以恢复到新的外部目录。阶段状态和完整巡检快照按用途保存在history归档中。当前没有监督或后台训练入口。

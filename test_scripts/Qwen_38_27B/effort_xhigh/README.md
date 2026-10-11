# Qwen3.8-27B xhigh

Fixed effort sweep: same checkpoint, datasets, decoding and judge as medium; only reasoning_effort changes. Medium results remain in parent directory. No training.

## 本轮执行补充

本轮先使用 GPU 4–7（两组 TP2），low 完成后加入 GPU 0–3，成为四组 TP2。Arena 保留已保存的 158 条回答，取消未保存的在途请求后续跑；不更换有效答案。采样参数与 32K 输出上限不变。配置已同步为最终 8 卡部署；原始时间线见 `runtime/execution_events.jsonl`。扩容后并发 128，网络等待上限 3600 秒；token 表不含取消与重试开销，耗时不作等卡数对照。

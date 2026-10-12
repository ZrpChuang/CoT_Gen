# 9B 训练效率记录

更新时间：2026-10-12。正式配置保留每卡 batch=1；27B 暂缓。10:25 已恢复远端连接，队列与环境已重新核验，本次恢复没有新增 GPU 测量。以下为 2026-10-11 已完成的独立训练速度试验。

## 首轮实际测量

2 × H20，Qwen3.5-9B，历史 NEU checkpoint-131，同一份 128 条训练样本，BF16 LoRA、SDPA、已有 FLA、global batch=32、完整序列。每项临时训练 4 步，首步预热，统计后 3 步。临时 adapter 不保存到研究 checkpoint，16 个临时更新步不计入正式训练。

| 执行配置 | 双卡真实 token/s | token/s/GPU | 3 步耗时（秒） | 8K 前向/梯度峰值（GiB） | padding 比例 |
|---|---:|---:|---:|---:|---:|
| 每卡 batch=1，baseline A | 2,845 | 1,423 | 54.73 | 46.38 | 0% |
| 每卡 batch=1，baseline B | 2,926 | 1,463 | 53.22 | 46.38 | 0% |
| 每卡 batch=2，首次 | 1,483 | 741 | 105.01 | 67.45 | 35.45% |
| 每卡 batch=2，热缓存复测 | 1,892 | 946 | 82.30 | 67.45 | 35.45% |

每次被测部分均包含 96 条样本、155,726 个真实 sequence token。batch=1 的梯度累积为 16，batch=2 为 8；输入、学习率、global batch 和优化步数保持一致。

**决定：丢弃 batch=2。** 热缓存复测仍比较快 baseline 低 35.3% 吞吐，容量检查通过不能代表更快。batch=2 增加了 35.45% 的 padding，这是额外计算的来源之一；现有证据不足以把全部差距归因于 padding。正式训练继续采用每卡 batch=1。

这是历史 NEU 的短试验，不能直接外推新增多语言、多轮 100K 数据的训练时长，也不是模型能力评测。各新增阶段会重新进行固定训练样本的有预算测速。

## 后续优化顺序

1. 在隔离依赖目录研究 `causal_conv1d`。当前 FLA 已安装，深度卷积核缺失；先检查 loss/梯度等价，再测实际吞吐。
2. 研究 Liger 融合 loss 等减少 logits 中间张量的实现，核对当前 Qwen3.5 多模态类支持及版本。
3. 比较单卡与双卡的每 GPU 小时产出；DDP world size 和执行计划只能在新阶段开始前确定。

每次只改变一个变量，冻结训练输入、监督 mask、global batch、学习率、LoRA、8K 上限和评测。两次候选均超过较快 baseline 至少 10%，且通过完整长度容量、有限数值、梯度等价和恢复检查，才考虑正式采用。失败和丢弃记录保留。

### 内核试验准备（2026-10-12，尚未执行）

- 官方 `causal_conv1d` 提供 PyTorch reference 与 autograd 实现。恢复 GPU 后先固定源码版本，核对当前 Qwen3.5 实际调用的卷积参数、dtype、布局与激活，再检查输出和反向误差；不能以成功导入代替核验模型已使用该路径。[官方实现](https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py)
- Liger 当前源码的 Qwen3.5 patch 支持 `Qwen3_5ForConditionalGeneration`，但默认同时启用 fused loss、RMSNorm 和 SwiGLU。研究融合 loss 时必须显式关闭其他 patch，分别核验 PEFT 包装、监督 mask、loss 归一化和梯度；源码支持不代表与已装 Transformers 5.12.1 已验证兼容。[官方 patch](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/monkey_patch.py)
- 安装仅放在隔离目录，固定版本及源码哈希，不更换共享环境的 Torch、Transformers 或 NumPy。固定数据先 baseline，再单变量候选与复测；保存环境清单、实际启用路径、误差、吞吐、GPU 小时、显存与失败原因。只有门槛全部通过，才在尚未开始的阶段应用。

参考：[Transformers Qwen3.5](https://huggingface.co/docs/transformers/v5.12.0/en/model_doc/qwen3_5)、[causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)、[Liger Kernel](https://github.com/linkedin/Liger-Kernel)。

## 运行与维护

- 本轮启用 15 项 9B、60 项最终评测；27B 的 4 项训练、16 项评测保持暂缓。
- 正式训练每 5 步记录所有 rank 汇总的真实 sequence / supervised token/s、padding、峰值显存、GPU 小时和 ETA，W&B 侧车同步。
- 计算任务存活时，远端 supervisor 独立执行队列；整个计算任务结束后需新资源恢复。2026-10-12 按用户要求关闭 Codex 定时维护，远端队列保持待命；效率试验等待后续明确安排。当前恢复说明见 [扩训 TODO](SCALE_100K_TODO.md#更换计算任务后恢复)。
- Git 维护源码、配置示例和核心结论；权重、逐题回答、请求缓存、日志和主机配置留在产物库。原始回答等从 Git 索引移除，服务器文件保留。

## 证据位置

- 维护代码：`/mnt/bn/vai-llm-hl2/ruipeng/CoT_Analyse/auto_research/scaling/`。
- 本文远端位置：`/mnt/bn/vai-llm-hl2/ruipeng/CoT_Analyse/baseline_method/docs/TRAINING_EFFICIENCY.md`。
- 自动报告：`/mnt/bn/vai-llm-hl2/ruipeng/CoT_Analyse_artifacts/scale_100k_20261011/reports/EFFICIENCY.md`。
- 台账：上述产物目录下 `efficiency/ledger.tsv`。
- 本次试验：`efficiency/Qwen_35_9B/0/` 下四个试验目录，各自保留结果、逐步指标和日志。

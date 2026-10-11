# 统一扩训队列

从已有的 19 次训练继续到累计 100K 个不同的问题及上下文。主实验 4 方法 × 2 模型，加 1 个学习率对照和 10 个历史诊断；BASE 不训练。代码只负责这项固定实验，不根据测试成绩搜索或改写训练配方。

**本轮启用 9B 共 15 项，27B 的 4 项暂缓。** 27B 的历史 checkpoint、配方和数据约束保留，仅在用户明确恢复后入队。

## 使用

首次部署先复制 `config.example.json` 为 `config.json`，填写服务器路径和 W&B 项目；本地配置不入 Git。

```bash
cd /mnt/bn/vai-llm-hl2/ruipeng/CoT_Analyse/auto_research/scaling
/mnt/bn/vai-llm-hl/ruipeng/.venvs/gui-training-v2/bin/python pipeline.py start
/mnt/bn/vai-llm-hl/ruipeng/.venvs/gui-training-v2/bin/python pipeline.py status
/mnt/bn/vai-llm-hl/ruipeng/.venvs/gui-training-v2/bin/python pipeline.py stop
```

`start` 幂等启动独立会话中的 supervisor，SSH 断开不影响运行。controller 异常退出后自动重启，恢复持久化任务状态并接管仍在运行的子任务。每项任务失败最多重试 3 次，等待 API 凭证不消耗重试次数。`stop` 保留数据、训练 checkpoint 和任务进度，可再次 `start`。

运行依赖当前远端计算任务持续存活；服务器或整个计算任务被释放后，需在获得新计算资源时执行同一入口恢复。程序无法自行申请替代服务器。代码、配置、评测协议和数据哈希会保存在产物目录，支持检查实际运行版本。

更换计算任务时，先核验共享盘产物、模型及环境路径、进程身份、冻结源代码和 9B 操作范围，再启动一次队列。`status` 读取持久化状态，不能单独证明进程存活。完整步骤和最近的连接状态见 [扩训 TODO](../../baseline_method/docs/SCALE_100K_TODO.md#更换计算任务后恢复)。

## 自动执行顺序

1. 审计已下载的 LMArena 原始数据。保留旧 4,175 条，冻结 dev，排除测试文本重合；按固定随机顺序使用多语言、多轮原生上下文，取消没有论文依据的五类均分。优先使用两个 tokenizer 下输入均不超过 2,048 的候选，再使用已冻结的 2,049–4,096 reserve；完整训练序列上限 8,192，禁止截断。当前审计 120,304 个候选（117,114 primary，3,190 reserve），尚需经过生成质量筛选。
2. 自动构造共享答案与 NEU / SUP / AUG-SUP / SSR CoT，执行格式、长度、控制 token 和模型语义检查。语义检查是近似审计，不是正确性金标。候选不足以达到 100K 时明确报缺口，禁止重复样本充数。
3. 同内容 plain / steps 对按历史构造规则生成。原始未训练 9B 计算答案+EOS NLL 及共同正文 NLL；固定 1e-6 阈值和 20261004 随机种子，生成选择及匹配随机对照。随机配额在每个新增阶段的领域 × 长度五分位中保持一致，旧数据的历史选择保持不变。
4. 10K → 25K → 50K → 100K。每阶段只训练新增输入一遍，继承上阶段 adapter，重新初始化 AdamW 和该阶段的学习率调度；阶段内恢复完整 optimizer / scheduler / RNG / 数据位置。保留各训练臂历史学习率、LoRA 和 global batch=32。
5. 本轮 9B 每组 2 卡，最多并行 4 组；27B 暂缓。GPU 编码检查和候选 NLL 每组 1 卡。队列结合自己持有的 GPU 租约、实际空闲显存和利用率分配，尽量填满 8 卡，不杀外部任务，不用空算维持占用。
6. 达到 100K 后自动运行固定四 benchmark，本轮 9B 共 60 项评测，27B 的 16 项暂缓。生成与裁判评分并行，生成结束立即释放 GPU。独立程序检查完整分母、输入哈希和评分算术。W&B 在 CPU 侧同步训练曲线，最终核验云端摘要。

部署验证包含 19 个历史 adapter 的有限数值检查、两个模型完整 8K 序列前向/梯度检查、双卡 checkpoint 主动中断及恢复、冻结 BASE 的 541 条 IFEval 重评分、NLL 数值及正文映射检查。原 IFEval 的未设种子 `letter_frequency` 检查存在两题波动，依赖检查只放行已知两题的该指令位差异，其他指令逐位一致。验证使用独立产物，合成样本不计入 19 项研究训练。

任务启动采用落盘意图、进程身份和确认握手，controller 在启动窗口退出也能接管任务。训练/生成超时按最后一次持久化进度计算，正常长任务不会仅因运行多日被终止。超过重试上限或数据质量门槛失败会保留失败信息并停止对应依赖链，其他独立训练继续执行。

扩大后的数据分布与历史英语单轮 4K 有差异；这是分阶段的数据扩展实验，不能单独归因于样本量，也不能称为论文严格复现。不同 CoT 的 token 量会记录，不以相同步数代替相同数据规模。

## 文件职责

| 文件 | 职责 |
|---|---|
| `pipeline.py` | 启动、状态、停止和 worker 命令 |
| `config.example.json` / 本地 `config.json` | 目标、阶段、资源、环境和重试规则；不保存密钥 |
| `core/common.py` | 原子落盘、进程身份、配置、原生序列化 |
| `core/data.py` | 原始数据审计、恢复生成、共享配对数据 |
| `core/variants.py` | 历史格式对、精确正文映射、NLL 和随机对照 |
| `core/train.py` | 内存映射 token 数据、LoRA 续训、checkpoint、W&B |
| `core/efficiency.py` | 固定训练样本吞吐对照、容量检查、阶段执行计划和效率报告 |
| `core/evaluate.py` | 冻结评测、vLLM 生命周期和自动核验 |
| `core/queue.py` | 依赖队列、资源分配、重试、监督和进度报告 |
| `tests/` | 数据隔离、格式选择和队列恢复的关键验证 |

产物统一位于 `/mnt/bn/vai-llm-hl2/ruipeng/CoT_Analyse_artifacts/scale_100k_20261011/`：

- `manifest.json`：19 项历史来源和扩训目标。
- `config/`、`source/`：冻结输入、代码和部署配置。
- `data/`：候选审计、持久化请求缓存、配对数据、各阶段 token 缓存。
- `runs/实验/规模/`：训练、W&B 和最终评测。
- `state/`：任务状态、进程身份、完成标记。
- `logs/`：统一事件日志及分类任务日志。
- `reports/STATUS.md`：自动更新的 19 项进度；最终结果只有完整评测才标记完成。
- `reports/RESULTS.md`：BASE 首行，四方法 × 四 benchmark × 两模型的历史/100K 总表，以及诊断实验表。
- `reports/ledger.tsv`：训练规模、checkpoint、运行成本和经核验的最终指标。
- `reports/EFFICIENCY.md`、`efficiency/ledger.tsv`：各阶段速度和容量的对照记录。

教师和裁判凭证由 `COT_API_SECRETS` 或 `config.json` 的 `api_credentials` 指定，格式 `{"keys":{"GPT":["..."],"Qwen_Max":["..."]}}`，文件权限建议 600。凭证缺失时等待，恢复文件后自动继续。W&B 读取现有独立凭证文件。密钥不写入源码、结果或 Git。

## 运行范围与代码更新

```bash
python pipeline.py stop
python pipeline.py models --enable Qwen_35_9B
# 修改维护代码后归档旧部署、迁移兼容任务并启动。
python pipeline.py upgrade
```

操作范围保存在产物 `config/operations.json`，与冻结的实验配置分开；暂缓任务不算完成或失败。`upgrade` 要求所有 owned worker 已停止，保存旧源码、部署和任务状态，保留已完成任务。更换科学配置必须另开实验。

## 效率优化规则

正式训练每 5 步记录所有 rank 汇总的真实 sequence / supervised token/s、examples/s、padding、峰值显存和 ETA；W&B 同步这些指标。按每 GPU 吞吐判断效率。

每阶段对 NEU 的固定 128 条训练输入测量：batch=1 baseline、batch=2 候选、候选复测、batch=1 复测；各 4 步，首步预热。global batch=32、样本顺序、8K 上限和原配方固定。候选先通过完整 8K 前向/梯度与 10% 显存余量检查；两次候选均超过较快 baseline 10% 才记为速度胜出，改变 micro batch 还需梯度数值等价验证。当前正式默认仍为 batch=1，胜出记为待验证，失败记录保留。临时优化不写回研究 checkpoint，不计入正式规模。

当前环境已经安装 FLA、未安装 causal_conv1d；后者影响 Qwen3.5 的深度卷积路径。后续依次研究隔离安装的 causal_conv1d、Liger 融合 loss、单/双卡 GPU 小时效率。每次只改变一个变量，检查 loss/梯度、完整序列容量及断点恢复，再考虑正式采用。禁止因测速截断样本、减少 CoT 或改评测。

参考：[Transformers Qwen3.5](https://huggingface.co/docs/transformers/v5.12.0/en/model_doc/qwen3_5)、[causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)、[Liger Kernel](https://github.com/linkedin/Liger-Kernel)。

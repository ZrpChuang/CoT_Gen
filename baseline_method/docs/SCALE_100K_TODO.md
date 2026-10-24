# 100K 训练：当前方案与历史部署

## 2026-10-24：首批10K四方法9B评测完成

最新执行范围为同一批10,000个GPT输入/共享答案，NEU、SUP、AUG-SUP、SSR四种9B从基座新LoRA，2026-10-18各训练313步。2026-10-23续跑评分，2026-10-24完成16/16项评分、4/4独立核验及4/4 W&B同步。`state/active_complete.json`已确认四个9B实验完成，`study_complete=false`；100K后续、其余对照和27B保持暂缓，未启动新训练或数据生成。以下旧部署记录均按日期理解，不表示当前执行范围。

| 方法 | IFEval | MultiChallenge | EQ-Bench 3 | ArenaHard |
|---|---:|---:|---:|---:|
| BASE | 90.76 | 55.31 | 30.80 | 12.64 |
| NEU | 67.65 | 27.11 | 30.87 | 3.41 |
| SUP | 66.54 | 27.11 | 33.02 | 4.72 |
| AUG-SUP | 67.10 | 30.04 | 32.48 | 4.14 |
| SSR | 64.70 | 24.18 | 31.89 | 3.19 |

- 分母：IFEval541、MultiChallenge273、EQ45个场景、Arena750。Arena表中为hard_prompt主指标（500题）；EQ为rubric0–100，不是榜单Elo。全部为本地冻结协议结果。
- 训练后IFEval、MultiChallenge与Arena均明显低于BASE；EQ略有变化，不能据此宣称整体改善。尚未定位退化原因，不能归因于数据规模不足，也不构成论文严格复现。
- NEU和AUG-SUP各有一题HTTP400曾使自动评分达到3次失败上限；按原协议补齐两局后才生成完整指标，失败记录和原有效结果均保留，未换裁判、删题、改提示词或按测试分数挑checkpoint。
- 产物：`CoT_Analyse_artifacts/gpt_fresh_100k_20261012/`；总表`reports/RESULTS.md`、阶段总结`reports/TRAIN10K_PROGRESS.md`。
- 日期入口：`evaluations/2026-10-23_1514_qwen35-9b_10k/`，独立核验和文件SHA256证据在其`logs/records/final_verification.json`，开始/完成日期均记录于`manifest.json`。


## 2026-10-12 用户纠正：旧方案退休，从基座重训

以下旧续训记录仅作历史证据，已经停止执行。当前方案为 GPT 同时生成共享答案及四方法解题说明；100K 全新数据，15 项 9B 从原始基座新建 LoRA，27B 继续暂缓。旧 checkpoint-131、旧 Qwen 答案、旧 GPT 说明及旧生成缓存均不用于新训练。用户随后明确恢复持续监测与空闲保卡，并将 Codex 巡检改为每小时一次；远端队列仍每 15 秒检查，保卡持续保持至正式 GPU 任务接管。GPT 生成限流为 8 并发、每密钥最多 30 次/分钟、至少间隔 2 秒，详见训练入口说明。

当前产物：`CoT_Analyse_artifacts/gpt_fresh_100k_20261012/`；入口仍为 `auto_research/scaling/pipeline.py`。详细方案见 [训练入口说明](../../auto_research/scaling/README.md) 和新产物 `reports/PLAN.md`。进度读取新产物 `reports/STATUS.md` 并核对进程，不能由下文旧目录推断本次状态。

- [x] 停止旧 supervisor/controller/owned worker，写入永久退休标记。
- [x] 实现 GPT-only 数据策略、零历史 parent、基座新建 LoRA 与来源保护。
- [x] 19 项离线检查通过，包含旧权重拒绝、阶段来源隔离、GPT-only 凭证与退休队列禁止重启。
- [x] 新建 LoRA 前向/梯度通过，248 个 B 矩阵初始全零；完整 8K 序列容量通过，峰值分配约 46.38 GiB。
- [x] 新基座双卡诊断：64 条合成样本，step 1 主动中断后恢复至 step 2，参数确认更新，parent=null；诊断不计入正式规模。证据为新产物 `state/checks/fresh_ddp_resume.json`。
- [x] 重新审计 135,634 条源数据，120,304 个候选通过（117,114 primary / 3,190 reserve）；GPT 新答案与四方法说明生成已启动。
- [x] 首批 30 条 GPT 配对样本共同通过检查，缓存的真实返回模型均为 GPT；已记录每条答案和四方法说明的实际 deployment。首阶段仍需 10K 合格输入才启动正式训练。
- [x] 用户最新授权恢复持续监测与空闲保卡：9b 每 10 分钟复核，旧 cot 仍暂停，远端每 15 秒检查。24 项离线检查通过。
- [x] 8 张 H20 保卡实测通过；容器/宿主机 PID 映射已修复。controller 主动终止后约 16.3 秒恢复，同一 8 个保卡和 GPT 生成 worker 接管，无重复启动。
- [x] GPU 0 外部 CUDA 诊断触发自动让卡，约 16.0 秒（含诊断 CUDA 初始化）；外部进程正常结束，随后重新保卡，其余 7 卡身份不变。保卡和诊断均不计研究训练，失败与验证证据位于新产物 `state/checks/gpu_reservation.json`。
- [x] 整个计算任务停止后在新主机恢复同一实验：共享盘 1,011 条样本和 8,775 条请求缓存保留，118 项冻结来源与维护源码校验通过；恢复后新样本和请求继续增长，8 卡自有保卡恢复。未重建实验、未重复已完成检查、未启动旧权重训练。证据为新产物 `state/checks/resource_recovery.json`。
- [x] 按用户最新纠正保卡交接规则：外部 CUDA 出现时保留自有进程、上下文和显存，只暂停保卡计算；本队列实际 GPU 任务就绪后才释放所需卡。26 项离线检查通过，包含旧提前释放路径的回归检查；旧外部诊断触发退出的测试结论已被本规则替代。
- [x] 2026-10-13 修正无输出巡检：每小时明确汇报有效增量、速度、等待原因与行动；增加请求阶段、慢请求时长和待按序入库遥测。单批预取保持限流及原始批次/重试/入库顺序，32 项检查通过；1,024 条缓存耗时回放仅为诊断，线上 0/1/0/1 有预算重复测速另记台账，未通过门槛自动回退。详见训练入口说明“生成推进与可见报告”。
- [ ] 100K 数据、15 项 9B 阶段训练、W&B 和 60 项最终评测完成。

---

## 已退休的旧续训方案（历史记录）

更新时间：2026-10-12 10:33（北京时间）。本轮范围仍为 15 项 9B，27B 的 4 项暂缓。按用户要求，Codex 定时维护已关闭；远端 supervisor/controller 保持待命。

**连接与队列已恢复。** 04:24 SSH 代理返回 `job run ... is not running`；用户重新提供入口后，10:25 已成功连接。原 supervisor/controller 已退出，核验共享盘后只启动了一份新 supervisor。常驻程序能处理进程退出，不能在整个计算任务结束后自行申请新资源。

本次核验：118 份冻结文件哈希一致，19 个历史 checkpoint-131 的必要文件齐全，训练环境与模型路径可用，8 × H20 空闲。新增有效训练数据为 0，正式扩训尚未开始，队列仍按既有规则等待教师/裁判凭证；已完成的四次独立速度试验保留，没有重复执行。

## 目标与范围

从已有 19 个 checkpoint-131 继续到累计 100,000 个不同的问题及上下文：旧 4,175 条保留，新增 95,825 条。四种 CoT、两个模型和重复曝光不增加独立输入数。BASE 不训练，保留在总表第一行。

| 组别 | 实验 | 模型 | 数量 | 历史学习率 |
|---|---|---|---:|---:|
| 主实验 | NEU、SUP、AUG-SUP、SSR | 9B、27B | 8 | 1e-4 |
| 学习率对照 | NEU 低学习率 | 9B | 1 | 1e-5 |
| 历史诊断 | answer_only、aug_sup_lr1e5、body_format、body_matched_random、matched_random_format、paired_plain、paired_steps、ssr_lr1e5、student_format、sup_lr1e5 | 9B | 10 | 1e-5 |

19 项历史训练均为 4,175 条、1 epoch、131 步；数据哈希、完整 checkpoint 及 adapter 有限数值均已核验。部署验证的合成样本与两次优化步单独保存，不属于这 19 项实验。

27B 不因 GPU 空闲自动恢复。当前先完成 15 项 9B 的累计 100K 和 60 项最终评测。

## 已完成的部署工作

- [x] 原始 135,634 条完成审计：旧训练/dev 会话与输入隔离、四测试集文本重合过滤、上下文去重、两 tokenizer 的真实长度检查。
- [x] 冻结 120,304 个候选，其中 117,114 个 primary、3,190 个 reserve。优先输入均不超过 2,048 token，再使用 2,049–4,096 token reserve；完整序列不超过 8,192，禁止截断。候选尚不等于最终合格训练样本。
- [x] 新分布显式采用多语言、多轮原生上下文，不沿用无论文依据的五类均分。排除旧 4,175 条、保留 dev 及测试重合；同一训练会话的不同上下文可能相关。
- [x] 共享答案、四方法 CoT、canonical plain/steps 的恢复生成、逐请求持久化缓存、格式/长度/控制 token 和模型语义审计已实现。所有训练臂使用共同通过的数据；模型审计不等于正确性金标。
- [x] 原始未训练 9B 的 NLL 选择、精确共同正文映射及固定随机对照已实现。数值、映射和覆盖率验证通过；质量不达标时停止对应链，不挑选高覆盖子集。
- [x] 10K → 25K → 50K → 100K 的阶段续训已实现。各阶段只训练新增输入一遍，继承 adapter，重新初始化 AdamW 和阶段 scheduler；阶段内完整恢复 optimizer、scheduler、RNG 与数据位置。
- [x] 保留各训练臂历史学习率、LoRA 和 global batch=32。记录独立输入、阶段/累计步数、监督 token、DDP padding 曝光、parent、数据哈希及运行成本。
- [x] 常驻 supervisor、持久化任务依赖、GPU 租约、启动握手、孤儿进程接管、失败重试和分类日志已部署。无持久化进度超时才终止长任务，正常多日运行可继续。
- [x] W&B 独立 CPU 同步及云端最终摘要核验已实现并实测通过。
- [x] 19 项最终四 benchmark 共 76 项的生成、评分、独立核验和总表自动汇总已排队。冻结 118 个配置、tokenizer、数据、提示词和评测源文件。

## 已完成的实际验证

| 检查 | 实际结果 |
|---|---|
| 19 个历史来源 | 数据哈希、完整 checkpoint、adapter 有限数值通过 |
| 9B / 27B 原生序列 | 前向、梯度检查通过；未更新历史权重 |
| 完整 8K 序列容量 | 两模型通过，峰值分配显存约 46.38 / 82.26 GiB；0 optimizer step |
| 双卡训练与恢复 | 独立合成 64 条；step 1 保存后主动中断，恢复到 step 2，累计 133，adapter 确认更新 |
| W&B | 上述验证训练同步完成，云端核验累计 step 133 与数据哈希一致 |
| NLL 与格式对 | 4 条独立 fixture；full/chunk 数值通过，共同正文覆盖率 97.6%，选择与随机对照完成 |
| 冻结评测 | BASE 541 条 IFEval 重评分与复制后评分通过，EQ 依赖导入通过 |
| 常驻队列恢复 | 主动终止 controller，supervisor 自动重启；已执行任务未重复启动 |
| 代码契约 | 10 项数据隔离、长度计数、选择规则、依赖及启动恢复检查通过 |

IFEval 原始未设种子的 `letter_frequency` 检查存在已知两题波动。依赖检查仅放行这两题的该指令位变化，其余指令逐位一致；没有修改冻结评测器。

## 仍待完成

- [x] 恢复 GPU 任务连接，核验共享盘产物、环境和进程身份，恢复同一队列；证据为产物 `state/checks/resource_recovery_20261012.json`。
- [ ] 恢复教师/裁判凭证。默认路径 `/home/tiger/.config/cot-baseline/credentials.json`，需要 `keys.GPT` 和 `keys.Qwen_Max` 数组；不用把密钥发到聊天或写入仓库。
- [ ] 生成并审计新增 95,825 条共享配对数据。候选筛选损耗过大导致不足时，记录缺口并停止，禁止重复采样充数或降低目标。
- [ ] 本轮 15 项 9B 完成四个阶段，累计 100K 独立输入；核验实际数据、曝光、token 和步数。27B 暂缓。
- [ ] 所有 W&B 阶段完成并核验云端摘要。
- [ ] 完成 9B 的 60 项最终评测并更新总表；27B 另 16 项暂缓。缺失显示“未完成”，不填 0。
- [ ] 根据结果分析规模、学习率、数据质量、分布与训练目标的影响。无论改善或退化都保留结果，不按测试成绩筛选 checkpoint。

保留 dev 已冻结并完全排除训练；当前队列不额外运行阶段行为评测，也不根据 dev/测试分数自动改配方。阶段结束固定保存 checkpoint，最终统一四 benchmark 评测。

## 资源与操作

本次恢复已确认 8 × H20。9B 每组 2 卡、最多并行 4 组；27B 暂缓，恢复后每组 4 卡。NLL 每组 1 卡。按实际空闲显存、利用率、任务依赖和本队列租约尽量填满 GPU。没有可执行数据时允许空闲，不运行 GPU 占用脚本，不影响外部任务。

```bash
cd /mnt/bn/vai-llm-hl2/ruipeng/CoT_Analyse/auto_research/scaling
/mnt/bn/vai-llm-hl/ruipeng/.venvs/gui-training-v2/bin/python pipeline.py status
/mnt/bn/vai-llm-hl/ruipeng/.venvs/gui-training-v2/bin/python pipeline.py stop
/mnt/bn/vai-llm-hl/ruipeng/.venvs/gui-training-v2/bin/python pipeline.py start
```

SSH 断开或 Codex 关闭不影响程序。远端计算任务必须持续存活；整个计算任务被释放后，需在新资源上恢复路径并执行同一入口。程序不能自行申请替代服务器。任务失败最多重试 3 次，凭证/API 暂时不可用采用等待与断路，不消耗普通任务失败次数。

### 更换计算任务后恢复

1. 确认新资源挂载同一共享盘，项目、产物、训练 Python、模型和 CUDA 兼容库路径可用。读取 `reports/STATUS.md`、`reports/EFFICIENCY.md`、`state/status.json` 与 `state/jobs.json`，记录文件更新时间；状态文件本身不证明程序仍在运行。
2. 核对 supervisor、controller、worker 的真实进程身份和 GPU 使用情况，确认没有同一队列在其他资源运行。保留任务状态、请求缓存、已完成检查与 checkpoint，不清空状态或重跑整个实验。
3. 核对 `manifest.json` 仍保留 19 项历史来源，`config/operations.json` 仅启用 `Qwen_35_9B`；27B 保持暂缓。维护源码和本地配置须与产物 `source/` 一致；Git 文档更新不需要迁移训练源码。路径或环境发生变化时先记录差异，不能通过覆盖冻结配置绕过校验。
4. 检查教师/裁判及 W&B 凭证是否可读，只记录可用状态，不输出内容。缺失的教师凭证继续按既有等待条件处理。
5. 执行一次 `pipeline.py start`，核对新 supervisor/controller 及状态更新时间。若发现存活的同一队列，使用其状态，不启动第二份。已启动训练阶段保留绑定的执行计划，已有 checkpoint 按原 optimizer/scheduler/RNG 和数据位置恢复。
6. 在新资源上核验 GPU 型号、数量与依赖版本，再做新的固定数据 baseline。内核优化仍须先通过正确性、完整 8K 容量、重复测速和断点恢复门槛。

## 文件位置

- 维护代码与操作说明：`/mnt/bn/vai-llm-hl2/ruipeng/CoT_Analyse/auto_research/scaling/README.md`。
- 本文：`/mnt/bn/vai-llm-hl2/ruipeng/CoT_Analyse/baseline_method/docs/SCALE_100K_TODO.md`。
- 统一产物：`/mnt/bn/vai-llm-hl2/ruipeng/CoT_Analyse_artifacts/scale_100k_20261011/`。
- 实时进度：上述产物下 `reports/STATUS.md`，机器状态 `state/status.json`。
- 自动总表：`reports/RESULTS.md`，BASE 第一行，四方法 × 四 benchmark × 两模型，另附历史诊断表。
- 机器清单：`manifest.json`；冻结配置/源代码：`config/`、`source/`。
- 数据与请求缓存：`data/`；训练与评测：`runs/实验/规模/`；统一日志：`logs/`；验证证据：`state/checks/`、`state/implementation_validation/`。

## 结论限制

统一到论文的 100K 有研究价值，但目前不能将退化归因于数据少。新数据分布、阶段间优化器/调度器重置及原有 LoRA 配方均有比较限制，完成后仍不是论文严格复现。

本轮累计每个独立输入一遍，global batch=32，约 3,128 累计更新步，另记录 DDP 尾部 padding 的重复曝光。论文 Max Epochs=4、batch=256、学习率 4e-5、32K 长度与本轮不同；不把数据量一致写成训练预算或配方一致。

## 训练效率与代码维护

- [x] 9B 启用 / 27B 暂缓的操作范围持久化，保留 19 项历史来源。
- [x] 12 项关键队列与数据契约通过；baseline 3 项、格式研究 15 项离线检查通过。
- [x] 每阶段固定训练数据的速度对照加入 DAG；先 baseline，再 batch=2 和复测，记录 GPU 小时与显存。
- [x] 正式训练增加所有 rank 汇总的 token/s、padding、峰值显存、GPU 小时和 ETA；阶段执行计划与断点绑定。
- [x] 2026-10-12 按用户要求关闭 Codex 定时维护；9B 心跳与旧 CoT 心跳均为暂停。远端队列独立待命，27B 不自动恢复。
- [ ] 研究隔离环境 causal_conv1d、Liger 融合 loss、单/双卡效率。FLA 当前已启用，causal_conv1d 未安装。
- [x] 维护代码和核心结论已同步 Git，排除主机配置、权重、原始回答、缓存和日志包。
- [x] 已核验提交 `bd476e0` 的 [GitHub Actions](https://github.com/ZrpChuang/CoT_Gen/actions/runs/38147740181) 全部通过；这是离线代码检查，不代表正式扩训完成。
- [x] 首轮效率结论：batch=1 双卡 2,845–2,926 token/s；batch=2 热缓存复测仍低 35.3%，丢弃该候选。详见 [效率记录](TRAINING_EFFICIENCY.md)。

效率报告：产物 `reports/EFFICIENCY.md`；逐次台账：`efficiency/ledger.tsv`。测速只使用训练数据，临时优化不写回历史或正式 checkpoint，也不计入训练规模。

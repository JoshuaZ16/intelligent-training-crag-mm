# VEGA-RAG 真实多轮性能对比实验设计

日期：2026-07-16
项目：CRAG-MM 第 20 组 Task 2
状态：实验规范已锁定，本文不包含任何未运行的结果

## 1. 目标

在现有 Task 2 真实 30 样本基线之上，实现并真实运行 VEGA-RAG 的三个独立模块与完整组合，回答以下问题：

1. 实体一致性重排能否减少图像实体错认？
2. 只对中等置信样例追加检索，能否在可控延迟下补齐关键事实？
3. 对低置信或证据冲突样例拒答，能否降低错误但自信的回答并提高 Truthfulness？
4. 三个模块组合后，质量、拒答率、延迟和显存之间的整体取舍是什么？

实验必须产生真实运行日志、逐样本答案、检索证据、GPU 遥测和退出状态。未运行的版本不填写性能数字；崩溃、无提升或负提升均作为真实结果保留。

## 2. 已有事实与研究动机

现有 B0 基线已经在双 A800 服务器上完成固定 30 条 validation 样本的 Task 2 运行：进程退出状态为 0，30 条 trace 均为 `ok`，29/30 同时包含 Image KG 与 Web 证据。非官方单人快速复核得到正确 7、部分覆盖 6、不正确 17，其中实体错认 11，是当前最大的质量瓶颈。

这些数字只用于确定研究问题，不作为新版本的提升结论。新旧版本的性能比较必须使用本设计规定的成对实跑和统一复核口径。

## 3. 不做的事情

- 不把精确字符串匹配 0/30 当成真实语义准确率。
- 不把单人复核包装成 AIcrowd 官方 leaderboard 分数。
- 不更换样本、seed 或模型来挑选有利结果。
- 不在不同时间窗口的单卡运行之间直接声称因果提升。
- 不使用未落盘的终端输出、手工编造的 GPU 数字或估算的延迟。
- 不在观察 20 条主评测集结果后重新调整阈值。

## 4. 固定实验条件

所有正式版本使用以下共同条件：

| 项目 | 固定值 |
|---|---|
| 数据 | CRAG-MM single-turn validation，revision `v0.1.2` |
| 清单 | 现有 `sample_manifest.json` 中的固定 30 条 |
| seed | `20260712` |
| 生成模型 | `unsloth/Llama-3.2-11B-Vision-Instruct` |
| 后端 | vLLM 0.7.3，BF16，temperature 0 |
| 最大回答 | 75 token |
| 基础检索 | Image KG top 3；Web top 3 |
| 硬件 | 两张 NVIDIA A800-SXM4-40GB |
| 并行方式 | GPU0 跑当轮 B0，GPU1 同时跑创新版本 |
| 样本顺序 | manifest 顺序；评价时按 query 稳定 ID 对齐 |

正式运行前允许 1 条和 5 条 smoke test，但 smoke 结果不进入性能表。任何正式失败运行均保留独立 run ID 和日志，不覆盖后续成功重跑。

## 5. 五个对比版本

### 5.1 B0：当前 Task 2 基线

保持当前行为不变：图像检索得到 Image KG，使用前两名实体名与原问题构造一次 Web 查询，过滤低分/重复证据后，将 Image KG 与 Web 证据分区送入 11B 模型。B0 增加 trace v3 观测字段，但不得改变证据顺序、Prompt 文本或最终答案。

### 5.2 A1：实体一致性重排

A1 在 B0 检索完成后对候选图像实体进行确定性重排，不追加搜索、不直接拒答。

对图像候选实体 `e_i` 计算：

```text
image_norm_i = clamp(image_score_i, 0, 1)
web_support_i = max_j token_f1(entity_name_i, web_title_j + web_snippet_j)
attribute_support_i = question_terms 与实体属性字段/取值的 token_f1
entity_score_i = 0.50 * image_norm_i
               + 0.35 * web_support_i
               + 0.15 * attribute_support_i
```

现有真实 trace 中图像检索分数位于 0.546 到 0.929，已经处于 0 到 1 范围，因此只做边界裁剪，不对每个样本单独 min-max；后者会把很接近的候选强行拉成 1 和 0，阻止 Web 证据纠错。按 `entity_score_i` 重排 Image KG，并记录 top-1、top-2、margin、Web 支持和最终实体顺序。所有归一化、分词与并列排序规则必须确定性实现；并列时保持原检索顺序。

A1 的唯一自变量是实体一致性重排，因此它不使用动态 K，也不使用置信拒答。

### 5.3 A2：A1 + 自适应查询重写与动态 K

A2 先执行 A1。若实体 margin 或 Web 支持落入中置信区间，则追加一次 Web 检索：

```text
rewritten_query = top_entity_name + question_key_terms + disputed_attribute
k = 5
```

新旧 Web 结果按 URL 与规范化标题去重，再根据 query relevance 与实体支持排序。高置信样例不追加检索；低置信样例不在 A2 中拒答，仍交给生成模型，以便单独测量查询重写的贡献。

### 5.4 A3：A1 + 校准拒答

A3 先执行 A1，但不追加检索。若实体一致性低于 `tau_low`，或 Image KG 与 Web 对问题核心属性给出可检测的冲突值，则直接返回标准化 `I don't know`，并在 trace 中记录拒答原因、冲突实体和证据 ID。

A3 的唯一新增行为是低置信/冲突拒答，目的是用 Missing 换取更低 Hallucination，而不是提高表面回答率。

### 5.5 FULL：完整 VEGA-RAG

FULL 使用三段式门控：

```text
score >= tau_high        -> 直接使用 A1 重排证据回答
tau_low < score < high   -> 执行 A2 查询重写和动态 K 后回答
score <= tau_low         -> 执行 A3 校准拒答
```

FULL 必须记录门控前后分数、触发动作、追加证据数量、延迟增量与拒答原因，使每个样本的决策可解释、可复跑。

## 6. 阈值校准与防止数据泄漏

30 条固定样本按 manifest 顺序预注册为：

- 校准集：前 10 条，只用于选择 `tau_low`、`tau_high` 与冲突检测阈值。
- 主评测集：后 20 条，用于主要性能结论。
- 全 30 条：作为补充描述，明确包含校准样本，不作为无偏主结论。

校准阶段在前 10 条上真实并行运行两侧：GPU0 为只增加观测字段、不改变行为的 `B0-cal`；GPU1 为对所有非低置信样例执行一次追加检索的 `A2-enrich-all`。评价时对每个阈值组合使用真实落盘答案进行组合：低于 `tau_low` 使用标准拒答，中置信区间使用 `A2-enrich-all` 答案，高于 `tau_high` 使用 B0 答案。这样无需在正式 20 条上试阈值，也不需要模拟生成结果。

候选阈值只在预先固定的网格上选择：`0.20, 0.30, 0.40, 0.50, 0.60, 0.70`，且要求 `tau_low < tau_high`。在查看任何后 20 条主评测结果前，前 10 条真实校准揭示了一个选择性预测退化：若只最大化 Truthfulness，全量拒答会优于当前负 Truthfulness 的回答系统。该无约束结果原样保存在 `vega_calibration_unconstrained.json`。经用户在正式轮开始前确认，正式阈值增加回答覆盖率 `1 - Missing >= 0.70` 的可行性约束，再按以下顺序优化：

1. 最大化校准集 Truthfulness；
2. 若并列，选择 Accuracy 更高者；
3. 若仍并列，选择拒答率更低者；
4. 若仍并列，依次选择更低的 `tau_low`、`tau_high`，保留更多回答并使网格选择确定化。

阈值、最小覆盖率、可行与被拒候选数、网格搜索输入、选择理由和 SHA-256 写入 `vega_calibration.json`。本次真实校准冻结结果为 `tau_low=0.20`、`tau_high=0.30`，校准覆盖率为 `0.80`；阈值冻结后，不因主评测集结果改变。

## 7. 四轮双卡成对实验

| 轮次 | GPU0 | GPU1 | 核心问题 |
|---|---|---|---|
| R1 | B0-R1 | A1 | 实体重排是否减少实体错误 |
| R2 | B0-R2 | A2 | 追加检索是否改善中置信样例 |
| R3 | B0-R3 | A3 | 拒答是否降低 Hallucination |
| R4 | B0-R4 | FULL | 完整方案的质量/成本取舍 |

每一对进程须在 30 秒内启动，使用独立输出目录、独立 trace 和各自 GPU 遥测。若一侧崩溃、样本数不是 30、样本集合不一致或启动时间差超过 30 秒，则该轮不进入成对性能结论，保留失败证据后以新 run ID 重跑。

四次 B0 重复运行用于测量稳定性。若 B0 的逐样本答案或主要指标发生变化，报告必须给出变动率，性能比较优先使用同轮 B0，而不是任意挑选一个基线。

## 8. 评价指标

### 8.1 官方公式兼容指标

对每条回答按固定二元语义口径得到：正确 `C`、标准拒答 `M`、错误非拒答 `H`。总数为 `N`，且 `N = C + M + H`。

```text
Accuracy          = C / N
Missing           = M / N
Hallucination     = H / N
Truthfulness      = (2*C + M) / N - 1
                  = Accuracy - Hallucination
```

语义正确必须包含 ground truth 的核心事实且不能包含矛盾事实。部分覆盖、错误实体、错误数值和无依据扩写在官方公式兼容二元口径中均计为 `H`；同时在诊断表中保留更细的 partial 标签。

由于当前没有 AIcrowd 官方语义裁判，本实验的上述指标标记为“固定量表的非官方复核指标”，不能写成 leaderboard 成绩。

### 8.2 盲化复核流程

所有版本答案按稳定样本 ID 汇总后隐藏版本名、轮次和门控动作，使用固定随机种子打乱。复核表只显示问题、ground truth 和答案。

第一遍标注：`correct / partial / incorrect / missing`。
第二遍核查：对所有 `partial`、所有版本间标签发生变化的样本及所有拒答样本重新检查。
最终二元映射：`correct -> C`，`missing -> M`，`partial/incorrect -> H`。

复核文件保存原始标签、根因、说明和复核时间。报告将复核者数量明确写为 1，不声称双人一致率。

### 8.3 诊断指标

- 实体错认数与实体错认率。
- 部分证据覆盖数。
- 推理或事实错误数。
- 跑题/截断数。
- 不恰当拒答数。
- top-1/top-2 entity margin。
- Image KG/Web evidence agreement。
- Web/KG 冲突率。
- 追加检索触发率与新增有效证据数。
- 拒答精度：拒答样本中原 B0 确实错误的比例。
- 回答覆盖率：`1 - Missing`。

### 8.4 系统性能指标

- run completion、30/30 完成率和 trace error 数。
- Image、Web、追加检索、Generation、Total 的 P50/P95/最大延迟。
- 每样本 token 数、证据数和搜索调用数。
- 墙钟耗时、GPU 峰值显存、利用率、功耗和温度。
- FULL 相对同轮 B0 的 P95 延迟增量。

## 9. 统计比较

主要结论基于 20 条主评测集的同样本成对比较：

- 报告 Accuracy、Missing、Hallucination、Truthfulness 的绝对值与相对同轮 B0 的差值。
- 对指标差值执行固定 seed 的 10,000 次 paired bootstrap，报告 95% 置信区间。
- 对二元正确性执行 McNemar 精确检验；样本较小时同时报告变化样本数，不只报告 p 值。
- 输出逐样本迁移矩阵：B0 错误变正确、正确变错误、错误变拒答、拒答变正确。
- 不以“置信区间跨 0”掩盖方向和实际样本数。

## 10. 证据目录与文件约定

正式结果统一保存到：

```text
artifacts/week2/vega_real_ablation/
  calibration/
  round_1/{b0,a1}/
  round_2/{b0,a2}/
  round_3/{b0,a3}/
  round_4/{b0,full}/
  comparison/
```

每个正式运行目录必须包含：

- `run_config.json`
- `sample_manifest.json`
- `agent_trace_v3.jsonl`
- `turn_evaluation_results_all.csv`
- `scores_dictionary.json`
- `stdout_stderr.log`
- `exit_status.txt`
- `gpu_telemetry.csv`
- `image_prefetch.json` 或其不可变引用
- 代码版本指纹与环境版本快照

`comparison/` 包含盲化复核文件、最终标签、逐样本配对表、指标汇总、bootstrap 结果、阈值配置和图表数据。失败运行使用新目录保留，不覆盖、不删除。

## 11. 实现边界与测试

新增逻辑按独立、可测试组件实现：

1. `entity_agreement.py`：实体分词、支持度、重排和 margin。
2. `adaptive_retrieval.py`：中置信检测、查询重写、动态 K、结果合并去重。
3. `calibrated_abstention.py`：阈值门控、冲突检测和拒答原因。
4. `vega_config.py`：版本开关、冻结阈值和 trace schema。
5. 现有 `CourseRAGAgentV2` 只负责按配置编排这些组件。

必须先写测试，再修改实现。最低测试覆盖：

- A1 重排在 Web 明确支持第二候选时能够改变顺序。
- 无 Web 支持时保持原始顺序。
- A2 只在中置信区间触发一次追加检索，并正确去重。
- A3 只在低置信/冲突时标准拒答。
- FULL 三个分支均可由确定性输入触发。
- B0 模式的 Prompt、搜索次数和答案行为不回归。
- trace v3 字段完整、可 JSON 序列化。
- 五个版本的 1 条和 5 条 smoke test 均能落盘。

## 12. 真实运行顺序与停止条件

1. 实现 trace v3 与 B0 行为等价测试。
2. 在前 10 条上并行运行 `B0-cal` 与 `A2-enrich-all`，完成盲化复核并冻结阈值文件。
3. 实现并测试 A1；完成 1 条、5 条 smoke。
4. 双卡运行 R1。
5. 实现并测试 A2；完成 smoke；双卡运行 R2。
6. 实现并测试 A3；完成 smoke；双卡运行 R3。
7. 组合 FULL；完成 smoke；双卡运行 R4。
8. 汇总、盲化、复核、统计分析。
9. 生成 PDF，逐页渲染验收。

若发现 B0 行为被观测代码改变，立即停止正式对比并修复。若远端实例不可用、磁盘不足或模型无法加载，记录外部阻塞，不生成替代数字。若某创新版本真实结果下降，继续完成剩余版本，以便得到完整消融结论。

## 13. PDF 交付设计

最终交付文件名：

```text
output/pdf/VEGA-RAG-真实多轮性能对比实验报告.tex
output/pdf/VEGA-RAG-真实多轮性能对比实验报告.pdf
```

PDF 只从同步后的 JSON/CSV/日志生成，至少包含：

1. 实验问题与真实性声明。
2. 五版本结构和四轮双卡时间线。
3. Accuracy、Hallucination、Missing、Truthfulness 总对比。
4. 主评测 20 条与全 30 条补充结果。
5. 实体错误、拒答质量与证据一致性对比。
6. 延迟、吞吐、显存和墙钟成本。
7. paired bootstrap、McNemar 与逐样本迁移图。
8. 成功案例、负面案例和无提升模块。
9. 可复现命令、运行目录、代码指纹和限制。

所有图表必须标明样本量、版本、轮次和评价口径。未通过官方语义裁判的指标统一标注“非官方固定量表复核”，不出现“官方提升”或“leaderboard 提升”措辞。

## 14. 验收标准

- B0、A1、A2、A3、FULL 均有真实 30 条正式运行证据；B0 有四次同轮重复。
- 四轮每侧样本集合相同，正式进程退出状态和异常均有记录。
- 所有性能数字可从落盘数据重新计算。
- Accuracy、Missing、Hallucination、Truthfulness 严格遵循第 8 节公式。
- 主结论来自冻结阈值后的 20 条主评测集；全 30 条只作补充。
- 报告同时展示质量收益和延迟/GPU 成本。
- 任何负提升、失败运行和复核限制均保留。
- 完整测试通过，`git diff --check` 通过。
- 最终 PDF 编译成功，逐页检查无缺字、裁切、重叠或表格溢出。
- PDF、TeX、原始日志、结果 CSV/JSON 和复核表全部返回给用户。

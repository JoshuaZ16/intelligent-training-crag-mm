# CRAG-MM 智能实训项目交接文档

> 最后更新：2026-07-15  
> 小组：第 20 组  
> 成员：张家豪、俞凡  
> 本次原定范围：完成第 1、2 周进度的重做与验收，不包含第 3、4 周实施

## 1. 一句话结论

项目已完成前两周的**工程底座、真实数据准备和汇报材料**，但没有完成前两周原计划中的**真实检索效果验收**。目前可准确表述为：

> Vision、Task1、Task2 的程序架构已建立，本地数据和轻量模型链路已验证；完整 Image KG/Web 索引与官方 11B Vision 模型下的真实 30+30 对照实验尚未运行。

因此，不应对外宣称“前两周全部完成”或“Task1/Task2 已取得真实性能提升”。

## 2. 项目在做什么

本项目要实现一个“看图片、查资料、根据证据答题”的多模态 RAG 系统，按三个任务递进：

1. **Vision Baseline**：只使用原图和问题，不检索。
2. **Task1**：使用图像搜索得到候选实体与 Image KG 属性，不允许网页搜索。
3. **Task2**：在 Task1 基础上加入查询改写、Web Search、证据去重和多源融合。
4. **Task3**：处理多轮对话历史、指代和上下文一致性。这是第 3、4 周工作，本次没有要求实施。

## 3. 当前进度总览

| 阶段 | 当前状态 | 可以说已完成的内容 | 不能说已完成的内容 |
| --- | --- | --- | --- |
| 第 1 周 | 部分完成 | 课题理解、运行环境、三任务路线、Agent 入口、本地轻量 VLM 调用 | 完整 Vision baseline 30 例、真实 Task1 端到端检索原型 |
| 第 2 周 | 部分完成 | Task1/Task2 程序逻辑、证据结构、实验脚本、固定真实样例、周报与中期 PPT | 真实 Task1/Task2 30+30 运行、性能对比、完整案例分析、效果门槛验收 |
| 第 3 周 | 未开始（不在本次范围） | 仅有计划文档 | Task2 优化、Task3 多轮原型、真实错误表、第 3 周周报 |
| 第 4 周 | 未开始（不在本次范围） | 仅有计划和报告提纲 | Task3 完善、最终实验、实训报告、最终 PPT、演示与提交 |

## 4. 已完成的具体工作

### 4.1 旧交付审计与验收口径

- 审计了旧 Agent、旧进度文档和周报中的完成声明。
- 确认旧实现没有真实检索、真实模型、实验 CSV/JSONL 和案例证据支持。
- 旧进度说明已标记为旧草稿，正式状态以 [docs/course/week1-2-rework-acceptance.md](docs/course/week1-2-rework-acceptance.md) 为准。

### 4.2 本机环境与后端抽象

- 建立了 Python 3.11 独立环境和 Apple Silicon 依赖清单，见 [requirements-macos.txt](requirements-macos.txt)。
- 确认 `torch`、`torchvision`、`datasets`、`transformers`、`cragmm-search-pipeline`、`mlx`、`mlx-vlm` 可导入，MPS 可用。
- 将模型调用封装为统一生成后端：macOS 使用 MLX-VLM，官方 Linux/CUDA 环境使用 vLLM。
- 完成了一次真实 validation 图片的本地模型调用。轻量 SmolVLM 在 3.113 秒内将 Kadhi Chawal 误答为 `Chicken curry`；该结果只证明本地推理链路可用，不证明效果合格。

### 4.3 Vision / Task1 / Task2 统一 Agent

核心代码位于 [agents/course_agent_v2.py](agents/course_agent_v2.py)，正式入口位于 [agents/user_config.py](agents/user_config.py)。已实现：

- `vision`：只使用原图和问题，不调用检索。
- `task1`：调用 Image Search/KG，在代码层面禁止 Web Search。
- `task2`：使用 Image KG 候选实体生成查询，加入 Web Search 与多源证据。
- 将实体名称、检索分数、KG 属性、网页标题、摘要和 URL 解析为结构化对象。
- 根据问题对 KG 属性排序，对 Web 结果执行去重、空摘要过滤、分数阈值和长度控制。
- Prompt 区分“当前图片可见事实”、“相似图片 KG”和“网页证据”。
- 区分无检索结果、证据不足、证据冲突、检索错误和空生成等内部原因，对外统一规范拒答。
- 增加批次输入长度、后端返回数量、答案空值和 75-token 上限检查。
- 为每个样例保存证据、耗时、拒答原因和异常的 JSONL trace。

### 4.4 自动化验证与实验框架

- [tests/test_course_agent_v2.py](tests/test_course_agent_v2.py) 和 [tests/test_course_week2_agent.py](tests/test_course_week2_agent.py) 共包含 19 项测试，覆盖任务切换、证据解析、网页去重、批次顺序、错误 trace、拒答和 token 截断。
- [scripts/offline_week2_acceptance.py](scripts/offline_week2_acceptance.py) 可用可控的假检索器/假模型运行 120 次流程，验证 Task1 不调用 Web、Task2 走 Web 分支。
- 上述 120 次是**工程契约测试**，不是 CRAG-MM 真实成绩。
- [scripts/week2_experiment.py](scripts/week2_experiment.py) 支持固定抽样、真实模型/检索、原始结果与运行配置保存。
- [scripts/summarize_week2_experiments.py](scripts/summarize_week2_experiments.py) 支持汇总准确率、缺失率、拒答率、失败率和耗时。

### 4.5 真实数据准备

- 已从 CRAG-MM single-turn validation 真实 parquet 分片读取 387 条数据。
- 已使用固定种子 `20260712` 抽取 30 条样例，覆盖 5 类问题和 5 个领域。
- 后续 Vision、Task1、Task2 必须使用同一批 interaction ID，不能分别抽题。
- 下载的 parquet、模型、图片和生成的 `artifacts/` 默认不上传 GitHub；见本文档第 8 节。

### 4.6 文档与汇报材料

- 前两周运行手册：[docs/course/week1-2-runbook.md](docs/course/week1-2-runbook.md)
- 前两周正式验收状态：[docs/course/week1-2-rework-acceptance.md](docs/course/week1-2-rework-acceptance.md)
- 错误分析与云端记录模板：[docs/course/week2-error-analysis.md](docs/course/week2-error-analysis.md)
- 张家豪第 1、2 周周报：`deliverables/week1-2-weekly-reports/`
- 俞凡第 1、2 周周报：同上，文件名标记“待本人确认”。
- 7 页中期汇报草稿：`deliverables/中期汇报/CRAG-MM-第20组-中期汇报草稿.pptx`
- 四份周报和 PPT 均已在本地渲染检查中文字体、分页和溢出。

## 5. 前两周原计划中仍未完成的工作

本节是最重要的交接内容。以下项目都属于用户一开始要求完成的第 1、2 周范围，不是后续新增要求。

### 5.1 第 1 周遗留项

1. **完整 starter kit 端到端流程尚未在真实资源下运行**  
   当前可确认环境、Agent 入口和评测脚本可加载，但没有在完整 Image Search 索引 + 真实 VLM 下运行完整 `local_evaluation.py`/Task1 流程。

2. **Vision Baseline 固定 30 样例未运行**  
   目前只完成了一张真实图片的轻量模型回答，不足以构成 baseline 结果。需使用固定 30 样例保存完整回答、耗时和原始记录。

3. **Task1 真实最小可运行版未完成**  
   Task1 的代码路径、结构化 KG 解析和答案生成已完成，但还没有在真实 Image Search 索引上完成“图片 → 候选实体 → KG 属性 → 最终答案”的端到端运行。

4. **真实 baseline 错误样例尚未整理**  
   现有一个轻量模型的视觉识别错误，但还没有“答非所问、幻觉、超长、错误拒答、检索信息使用不足”等真实 30 样例分类。

### 5.2 第 2 周遗留项

1. **Task1 尚未达到“真实稳定可展示”**  
   需在真实 Image Search 索引上完成固定 30 样例，且无崩溃、无丢样例、无批次错位。

2. **Task2 尚未完成真实可运行原型**  
   Task2 的查询生成、Web 证据解析、去重、编号与冲突拒答逻辑已完成，但尚未在官方 Web validation 索引上运行。

3. **真实 30+30 对照实验未运行**  
   仍需完成同一批样例上的 Vision、Task1、Task2 原始输出。其中 Task1 与 Vision 比较，Task2 与 Task1 比较。

4. **真实指标表未产生**  
   尚无基于真实输出的 Accuracy、Missing rate、Hallucination rate、Truthfulness score、平均/P95 耗时、空检索率、拒答率和失败率。未配置语义评测时，exact-match 与人工复核结果必须分列。

5. **真实案例包未完成**  
   尚需 Task1 至少 3 个成功、2 个失败、1 个合理拒答；Task2 至少 2 个 Web 增益、2 个 Web 噪声和 1 个跨源冲突案例。

6. **前两周的效果门槛未达成**  
   必须真实验证 Task1 的真实性得分高于 Vision Baseline，并且 Task2 相比 Task1 整体不退化。如果未达到，需继续调整字段筛选、网页阈值、查询改写和拒答策略。

7. **官方时延约束未验证**  
   尚需在官方 11B Vision + vLLM + 完整检索下检查平均和 P95 总耗时，确认单轮尽量满足 10 秒限制。

8. **俞凡周报尚需本人确认**  
   Word 文档已生成并渲染检查，但个人实际参与内容必须由俞凡本人确认后才能正式提交。

## 6. 为什么剩余前两周工作没有在 Mac 上做完

并非所有剩余工作都绝对不能在 Mac 上做，但完整验收不适合继续在当前本机环境硬跑：

- 本机是 Apple M4/24 GB 统一内存，可用 MPS/MLX，没有 NVIDIA CUDA。
- 官方目标后端是 Linux + CUDA + vLLM + Llama 3.2 11B Vision，不能在 Apple Silicon 上按原配置运行。
- Web validation 检索索引至少约 9.5GB，另有 Image Search 索引、检索模型和视觉模型。
- 当时 Hugging Face 官方端点多次出现 TLS EOF/连接中断，镜像可用但大文件下载速率不足以在当日闭环。
- 轻量本地模型已证明接口可用，但真实图片回答错误，不适合代替官方 11B 模型进行效果结论。

建议剩余前两周验收转移到实验室 NVIDIA 服务器或云端 GPU。租用付费 GPU 前必须由张家豪确认，当前尚未产生租卡费用。

## 7. 第 3、4 周工作：本次原本就没有要求实施

以下项目没有完成，但它们不应与第 1、2 周遗留项混在一起。它们属于后两周计划，不是本次交付欠项。

### 7.1 第 3 周计划

- 根据真实 Task2 结果优化查询改写、Web 阈值、证据排序和答案后处理。
- 对 Task2 不同版本进行对比实验。
- 建立真实错误分析表。
- 接入 `message_histories`，实现 Task3 多轮上下文原型。
- 设计指代恢复、历史使用、是否重新检索和当前问题分类策略。
- 运行 Task3 小规模真实样例。
- 撰写两人第 3 周周报。

### 7.2 第 4 周计划

- 完善 Task3 多轮一致性和错误传播控制。
- 统一三个任务的最终 Agent 配置与提交入口。
- 整理 Vision、Task1、Task2、Task3 最终实验表和案例。
- 撰写 6–8 页最终实训报告。
- 制作 10–14 页最终汇报 PPT，准备 10 分钟汇报和 5 分钟问答。
- 准备现场演示、最终源码、运行说明和提交材料。
- 撰写两人第 4 周周报。

## 8. 下一位接手者应如何继续

### 8.1 优先级 0：先补完前两周，不要直接跳到 Task3

下一步不是开发多轮对话，而是在完整资源环境下证明 Task1 和 Task2 真的有用。

### 8.2 建议的执行顺序

1. 获得 Linux + NVIDIA CUDA 主机，确认磁盘、显存和网络条件。
2. 按官方 `Dockerfile`/`requirements.txt` 安装 vLLM、Llama 3.2 11B Vision 和 CRAG-MM 检索依赖。
3. 下载并校验 Image Search validation 索引与 Web Search validation 索引。
4. 分别完成一次真实 dataset、image search、web search 和 11B VLM 调用，保存字段结构和耗时。
5. 用种子 `20260712` 的同一批样例运行 Vision 30、Task1 30、Task2 30。
6. 汇总 exact-match、拒答、失败与耗时，再按 ground truth 进行人工复核。
7. 从原始 trace 整理 Task1/Task2 成功、失败、拒答、噪声和冲突案例。
8. 如果 Task1 没有优于 Vision，调整 KG 字段排序、Prompt 和拒答策略后重跑。
9. 如果 Task2 相比 Task1 退化，调整 Web 阈值、查询改写或加入“KG 不足时才使用 Web”的门控后重跑。
10. 达到门槛并更新周报/PPT 后，才将前两周标记为完成；之后再进入第 3 周 Task3。

### 8.3 真实实验命令

环境变量：

```bash
export CRAG_TASK_MODE=vision   # 或 task1 / task2
export CRAG_BACKEND=vllm
export CRAG_MODEL=meta-llama/Llama-3.2-11B-Vision-Instruct
```

固定样例实验：

```bash
.venv/bin/python scripts/week2_experiment.py \
  --mode vision --backend vllm \
  --num-samples 30 --seed 20260712 \
  --output-dir artifacts/week2/real_vision_30

.venv/bin/python scripts/week2_experiment.py \
  --mode task1 --backend vllm \
  --num-samples 30 --seed 20260712 \
  --output-dir artifacts/week2/real_task1_30

.venv/bin/python scripts/week2_experiment.py \
  --mode task2 --backend vllm \
  --num-samples 30 --seed 20260712 \
  --output-dir artifacts/week2/real_task2_30
```

汇总：

```bash
.venv/bin/python scripts/summarize_week2_experiments.py \
  artifacts/week2/real_vision_30 \
  artifacts/week2/real_task1_30 \
  artifacts/week2/real_task2_30 \
  --output artifacts/week2/experiment_summary.csv
```

## 9. 本地复现与基础检查

macOS 环境：

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools wheel
.venv/bin/pip install -r requirements-macos.txt
.venv/bin/python scripts/week2_environment_check.py \
  --output artifacts/week2/environment.json
.venv/bin/python -m unittest discover -s tests -v
```

只验证工程分支，不产生真实 benchmark 结论：

```bash
.venv/bin/python scripts/offline_week2_acceptance.py
```

## 10. GitHub 中不包含的本地大文件与证据

下列目录在 `.gitignore` 中，不会随 GitHub 仓库下载：

- `.venv/`：本地 Python 环境。
- `local_models/`：已下载的 MLX 模型。
- `local_data/`：已下载的 validation parquet 分片。
- `artifacts/`：环境快照、固定样例清单、原始 JSONL/CSV、本地模型回答、文档/PPT 渲染预览。

因此，新环境需要重新下载数据、模型和检索索引，并按 [docs/course/week1-2-runbook.md](docs/course/week1-2-runbook.md) 重新生成证据。GitHub 中的 Word、PPT、源码、测试和说明文档是已提交的正式文件。

## 11. 完成门槛

前两周只能在以下条件同时满足后改为“完成”：

- Vision、Task1、Task2 均在同一批 30 样例上完成，无崩溃、丢样例和错位。
- 所有实验都有配置、种子、原始 JSONL/CSV 和汇总表。
- Task1 相比 Vision 的真实性得分正向提升。
- Task2 相比 Task1 整体不退化，且有可解释的 Web 增益案例。
- 成功、失败、拒答、噪声和冲突案例均能追溯到原始 trace。
- 汇报与周报中的结论已更新为真实实验结果。
- 俞凡已确认两份个人周报的真实参与内容。

## 12. 关键文件索引

| 用途 | 文件 |
| --- | --- |
| 四周计划 | [docs/course/4-week-execution-plan.md](docs/course/4-week-execution-plan.md) |
| 前两周验收状态 | [docs/course/week1-2-rework-acceptance.md](docs/course/week1-2-rework-acceptance.md) |
| 环境与实验命令 | [docs/course/week1-2-runbook.md](docs/course/week1-2-runbook.md) |
| 错误分析 | [docs/course/week2-error-analysis.md](docs/course/week2-error-analysis.md) |
| 正式 Agent | [agents/course_agent_v2.py](agents/course_agent_v2.py) |
| AIcrowd 入口 | [agents/user_config.py](agents/user_config.py) |
| 真实实验脚本 | [scripts/week2_experiment.py](scripts/week2_experiment.py) |
| 实验汇总 | [scripts/summarize_week2_experiments.py](scripts/summarize_week2_experiments.py) |
| 合成分支验证 | [scripts/offline_week2_acceptance.py](scripts/offline_week2_acceptance.py) |
| 自动化测试 | [tests/test_course_agent_v2.py](tests/test_course_agent_v2.py) |
| 前两周周报 | `deliverables/week1-2-weekly-reports/` |
| 中期汇报草稿 | `deliverables/中期汇报/CRAG-MM-第20组-中期汇报草稿.pptx` |

---

交接时最重要的原则：**合成流程验证不等于真实效果，程序能运行不等于任务已验收。**

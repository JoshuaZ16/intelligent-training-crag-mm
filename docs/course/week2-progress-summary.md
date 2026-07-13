# 第二周进度完成说明（旧原型，已停止作为验收依据）

> 状态：本文件是 2026-07-12 早期原型说明，缺少端到端评测证据。正式状态与证据请以 `week1-2-rework-acceptance.md` 为准。

## 1. 已阅读材料

本阶段已阅读并对照以下材料：

- 根目录课程安排：`智能技术实训2026课程教学安排.docx`
- 课程 PPT：`2026智能技术实训.pptx`、`实训课PPT .pptx`
- Starter kit 文档：`README.md`、`agents/README.md`、`docs/dataset.md`、`docs/search_api.md`、`docs/submission.md`
- 课程计划文档：`docs/course/4-week-execution-plan.md`、`weekly-reports-personal.md`、`teammate-weekly-report-guide.md`、`presentation-and-report-outline.md`

课程要求明确：第 2 周需要完成任务1稳定版本、任务2初始版本和中期汇报准备；个人周报共 4 次，最终还需提交源码、6-8 页实训报告和 10+5 分钟 PPT。

## 2. 第二周代码进度

已新增课程第二周版本 Agent：

- 文件：`agents/course_week2_agent.py`
- 配置：`agents/user_config.py` 已切换到 `CourseWeek2RAGAgent`
- 测试：`tests/test_course_week2_agent.py`

该 Agent 对应第 2 周目标，主要实现以下结构：

1. 任务1：使用图片调用 image search，提取相似图片中的结构化 KG 实体和属性。
2. 任务2：使用“当前问题 + 图片摘要 + 最近对话信息”构造 web search 查询。
3. 多源证据组织：将 Image KG evidence 和 Web evidence 分块放入 Prompt，减少证据来源混杂。
4. 生成约束：要求模型短答案、事实优先、证据不足时回答 `I don't know`。
5. 本地可测辅助逻辑：证据格式化、查询改写和 Agent 配置均有单元测试覆盖。

说明：当前本机环境未安装 `torch`、`vllm` 和 `cragmm-search-pipeline`，因此未运行完整 GPU 模型评测；已完成不依赖 GPU 的本地单元测试。

## 3. 第二周汇报口径

中期汇报可按以下逻辑叙述：

1. 赛题背景：CRAG-MM 面向多模态 RAG 和智能眼镜问答，核心挑战是事实性、检索噪声和幻觉控制。
2. 三任务递进：任务1单源图像 KG，任务2图像 KG + 网页检索，任务3多轮上下文。
3. 已完成工作：完成文档阅读、任务拆解、Agent 接口理解、任务1证据组织和任务2多源证据原型。
4. 当前改进：将图像 KG 和网页 snippet 分块组织，Prompt 中加入证据优先、短答案和拒答策略。
5. 当前问题：网页检索可能引入噪声，模型仍可能在证据不足时猜测；后续需要做小规模评测和错误案例表。
6. 后两周计划：第 3 周优化任务2并启动任务3，第 4 周整合系统、完善报告和最终 PPT。

## 4. 两人分工说明

成员 A：模型生成与 Agent 策略负责人。

- 负责 Prompt、答案生成、证据融合、拒答策略和中期汇报方法部分。
- 本周重点完成任务1 Prompt 优化和任务2多源融合 Prompt 设计。

成员 B：检索、评测与实验负责人。

- 负责环境、检索 API、评测脚本、实验记录和错误案例整理。
- 本周重点完成 web search 字段整理、任务1/任务2样例评测和中期汇报实验部分。

## 5. 后续待做

- 安装完整依赖后运行：

```bash
python local_evaluation.py \
  --dataset-type single-turn \
  --split validation \
  --num-conversations 10 \
  --display-conversations 3 \
  --eval-model None
```

- 任务1评测时可增加 `--suppress-web-search-api`，验证 Agent 在 web search 不可用时仍能使用 image KG 证据。
- 整理 3-5 个成功/失败案例，补入第 2 周中期汇报 PPT。
- 第 3 周继续优化任务2的查询改写、证据筛选和错误分析表。

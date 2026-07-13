# 前两周重做验收报告

更新日期：2026-07-12

## 1. 结论

前两周的工程底座、可切换 Agent、自动化测试、实验追踪、真实数据固定样例和合成契约验收已经完成。当前不能把“真实 Task1/Task2效果达标”标为完成，因为完整 CRAG-MM 检索索引尚未在本机就绪，真实 30+30 对比实验没有运行。

本报告采用三种状态：

- **已真实验证**：有本机命令和输出证据。
- **已合成验证**：使用可控检索器/模型验证工程契约，不代表 benchmark 性能。
- **待云端验证**：需要完整索引与 CUDA/vLLM 环境。

## 2. 验收矩阵

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| Python 3.11 独立环境 | 已真实验证 | `artifacts/week2/environment.json` |
| torch/datasets/transformers/cragmm/mlx 导入 | 已真实验证 | 环境快照全部为 `ok`，MPS 可用 |
| Vision/Task1/Task2 显式切换 | 已合成验证 | 19 项单元/集成测试通过 |
| Task1 禁止网页检索 | 已合成验证 | 60 条 task1 trace 中网页证据为 0 |
| Task2 image→web 调用 | 已合成验证 | 30 条 task2 trace 均含网页证据 |
| 批次无错位、无崩溃 | 已合成验证 | 120 条调用状态全部为 `ok` |
| 回答不超过 75 token | 已合成验证 | 合成验收最大为 3；正式后端另有 tokenizer 截断 |
| 真实 validation 数据读取 | 已真实验证 | 77.1MB parquet、387 条记录 |
| 真实固定 30 样例 | 已真实验证 | 固定种子 20260712，覆盖 5 类问题和 5 个领域 |
| 真实图片下载 | 已真实验证 | `sample-69.jpg`，1280×960 |
| 本机视觉模型真实回答 | 已真实验证，答案错误 | SmolVLM-500M-4bit 对真实图片回答 `Chicken curry`，耗时 3.113 秒；原图是 Kadhi Chawal |
| Task1 真实 30 样例与 Vision 对比 | 待云端验证 | image search 索引尚未就绪 |
| Task2 真实 30 样例与 Task1 对比 | 待云端验证 | web validation 索引约 9.5GB，尚未就绪 |
| 人工错误案例 | 已有 1 个模型识别错误 | `artifacts/week2/real_model_smoke/smolvlm-500m.json`；30+30 结果级分析仍待云端 |

## 3. 代码交付

- `agents/course_agent_v2.py`：后端无关核心 Agent、结构化证据、问题相关排序、模式隔离、拒答、追踪和答案截断。
- `agents/user_config.py`：正式 AIcrowd 入口，默认 task1，可用环境变量切换。
- `scripts/week2_experiment.py`：固定抽样、真实实验、原始输出与配置保存。
- `scripts/week2_model_smoke.py`：真实图片的 MLX-VLM 冒烟测试，保存答案、耗时与异常堆栈。
- `scripts/summarize_week2_experiments.py`：指标、耗时、失败和拒答汇总。
- `scripts/offline_week2_acceptance.py`：明确标注为非 benchmark 的 30+30 契约测试。
- `tests/test_course_agent_v2.py`：15 项新测试；连同旧原型兼容测试共 19 项。

## 4. 本机可行性结论

本机硬件和 Python 依赖已完成 Apple Silicon 真实图片推理，但完整 CRAG-MM 实验还受下载规模约束：

- 287MB 的 SmolVLM-500M 量化模型已下载并完成真实回答；该回答错误，仅证明推理链路可用，不证明模型效果。
- 单个真实 validation parquet 分片约 80MB，已成功下载。
- web validation 检索索引至少约 9.5GB，另有图像索引和检索模型。
- 当前 Hugging Face 官方端点多次出现 TLS EOF/连接超时，镜像可用但大文件速率不足。

因此本机适合代码开发、真实数据检查、轻量视觉冒烟和契约验收；完整 Task1/Task2 对比应转到网络和磁盘条件更好的 NVIDIA 云主机或实验室服务器。

## 5. 完成判定

在下列证据补齐前，项目状态写为“第二周工程与汇报材料已完成，真实检索效果验证待云端”，而不是“第二周全部完成”：

1. Vision、Task1、Task2 各 30 条相同固定样例原始输出。
2. exact-match 与人工复核分列的指标表。
3. Task1 至少 3 成功、2 失败、1 合理拒答案例。
4. Task2 至少 2 网页增益、2 网页噪声、1 跨源冲突案例。
5. Task1 相较 Vision 正向提升，Task2相较 Task1不退化；否则继续迭代并保留失败记录。

# TurnVEGA Task 2 / Task 3 实验交接与续跑计划

> 状态基线：2026-07-22（Asia/Shanghai）
>
> Git 分支：`codex/turnvega-single-gpu`
>
> 适用范围：CRAG-MM Task 2 / Task 3 首次回答准确率实验

## 1. 当前结论

本分支提交的是 TurnVEGA 实验基础设施、配置、审计、评分代码和单元测试，不是最终实验结果。

原 GPU 实例已经确认不可恢复。原服务器上的模型缓存、索引、日志及两个真实 smoke 的原始产物均视为丢失，必须在新实例上按冻结配置复现。历史记录表明真实 11B 多模态推理链路曾经跑通，但当前仍未完成 Task 2 / Task 3 创新变体的正式 dev/test 实验，也没有最终“后 40 条”对比表。

当前 runner 真正实现并通过真实推理的变体只有：

- Task 2：`t2_b0`
- Task 3：`t3_no_history`

其余变体虽已进入枚举和 18 项实验矩阵，但 support gate 会拒绝运行。不得只解除 gate 而让多个版本复用同一基线逻辑。

## 2. 本次代码范围

主要模块：

- `agents/evidence_schema.py`：证据与 Trace v5 数据结构。
- `agents/turnvega_config.py`、`agents/vega_config.py`：实验配置、枚举和冻结字段。
- `agents/course_agent_v2.py`：基线链路的配置接入与 trace 扩展。
- `configs/turnvega_core_experiments.json`：18 项核心实验矩阵。
- `scripts/build_turnvega_manifests.py`：可审计 manifest 构建。
- `scripts/run_turnvega_triplet.py`：单组三版本运行编排。
- `scripts/run_turnvega_suite.py`：实验矩阵编排与恢复保护。
- `scripts/turnvega_experiment.py`：Task 2 / Task 3 实验入口及实现边界。
- `scripts/audit_turnvega_run.py`：终态、行数、身份和配置审计。
- `scripts/score_turnvega.py`：C/P/I/M 指标及部分 Task 3 指标。
- `tests/test_*.py`：上述能力的回归测试。

提交前在 Python 3.10+ 运行完整测试：`Ran 219 tests`，结果 `OK`。macOS 系统自带 Python 3.9 会在收集两个使用 `X | None` 类型语法的既有测试模块时失败，因此接手环境必须使用 Python 3.10 或更高版本；这不是模型实验或准确率结果。

## 3. 历史真实 smoke 记录

以下数据仅是上一实例的历史记录；原始远端产物已丢失，不能作为当前可审计交付，必须重跑。

### 3.1 Task 2 B0

- 变体：`t2_b0`
- run ID：`smoke-t2-b0-v2-20260721`
- manifest SHA-256：`c0c818dc6a036e853a5ee94fac9112c45d25e7dce7e185c489cd6e2ac466efb8`
- source index：`69`
- 历史状态：`completed`，Trace/结果各 1 行
- 历史指标：Accuracy `0.0`，Missing `0.0`，Hallucination `1.0`
- 历史显存峰值：约 `36,756 MiB`

第一次 smoke 必须作为失败审计记录保留：

- 旧 manifest SHA-256：`008b1e91f0dd02de56d89935b2dbb82b07bbb9312d086d2c065ad96a9ee14e55`
- 错误：`selected row 0 does not match manifest order`
- 原因：局部 parquet 索引顺序与完整 validation 顺序不一致。

### 3.2 Task 3 no-history

- 变体：`t3_no_history`
- run ID：`smoke-t3-no-history-v1-20260721`
- manifest SHA-256：`a9c8fa486da3b846c84ada974014b44475d97563f0f0f6d6a5b94d9817338218`
- source index：`0`
- 历史规模：1 conversation / 5 turns
- 历史状态：`completed`，Trace/结果各 5 行
- 历史标签：0 correct、4 missing、1 hallucination
- 历史指标：Accuracy `0.0`，Missing `0.8`，Hallucination `0.2`
- 历史显存峰值：约 `36,756 MiB`

两个历史进程都在结果落盘后卡在 vLLM/检索线程清理阶段，最后通过 TERM/KILL 释放 GPU。正式批量实验前必须修复或封装该问题。

## 4. 冻结资源

### 4.1 生成模型

- 模型：`unsloth/Llama-3.2-11B-Vision-Instruct`
- revision：`677b0c1b7008230a0fb88708c5550748e72b9a83`

| 分片 | LFS SHA-256 |
| --- | --- |
| `model-00001-of-00005.safetensors` | `d183d799f0476061adc580ccc0a24abba5ab274d610f02e73aff7576e581118c` |
| `model-00002-of-00005.safetensors` | `51a7afac7fbc248a8b038709ec3fb5bae4e5588ea84130056c8ed4e5a5144b40` |
| `model-00003-of-00005.safetensors` | `24149e6a1922551067d64d78d8a41ac8393e502d6b64f551a4f73bbacbb5bf90` |
| `model-00004-of-00005.safetensors` | `caf17798db8a5de475bc7580706956032533cd7a9dfb588ad8488229c745bbad` |
| `model-00005-of-00005.safetensors` | `656566a9aeee1115ca174c16ea3c51a2839b3a7f14cc349805aa0ef5c7433e7b` |

`model.safetensors.index.json` 的 `metadata.total_size` 是张量字节数，不是文件物理大小之和。完整性检查必须逐分片核对 LFS SHA-256。

### 4.2 数据与检索资源

| 资源 | revision / SHA-256 |
| --- | --- |
| `crag-mm-single-turn-public` | `711dd84fa2f1611975d476261afcb07292151923` |
| `crag-mm-multi-turn-public` | `d7dd32e948c7eebc9a77eca081e8519c9af1c4f5` |
| `image-search-index-validation` | `19b5f4dca7218b0231b59e2c3da74da73b6acad7` |
| `web-search-index-validation` | `ad1614b964d62575637babb7469f8c3086adb402` |
| CLIP 权重 | `c6032c2e0caae3dc2d4fba35535fa6307dbb49df59c7e182b1bc4b3329b81801` |
| BGE 权重 | `45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7` |

下载 CLIP/BGE 时使用必需文件白名单，不要同时下载 TensorFlow、Flax、ONNX 或 OpenVINO 备用权重。

## 5. 新实例续跑计划

### P0：恢复可复现实验环境

1. 创建新的单 GPU 实例；显存应能容纳历史约 `36.8 GiB` 峰值并保留安全余量。
2. 记录 GPU 型号、驱动、CUDA、Python、Torch、Transformers、vLLM、磁盘和内存状态。
3. 拉取本分支，安装依赖并执行 `pip check`。
4. 下载冻结 revision，逐分片/逐权重校验 SHA-256。
5. 不在仓库中保存 API key、SSH 私钥或服务器密码。

### P1：补齐基础设施

1. 修复检索 executor/thread pool 与 vLLM engine 的退出清理。
2. 以“终态文件完成写入并 fsync + 子进程可正常回收”作为共同成功条件。
3. 完成 Task 7 的 CSV、置信区间、McNemar 与版本盲化复核流水线。
4. 实现 `scripts/export_turnvega_main40.py`，加入顺序、ID、哈希和公式拒绝测试。
5. 运行完整单元测试，保存实际测试数与日志。
6. 复现 legacy30 B0，确认 30/30 Trace 和结果顺序与冻结 manifest 一致。
7. 生成并冻结六个互斥 manifest、两个 main40 派生 manifest 及 SHA-256。

### P2：复现历史 smoke

1. 先运行 `t2_b0` 单条，保留首次身份审计失败的独立记录，再确认 v2 样本身份和输出行数。
2. 再运行 `t3_no_history` 一整段五轮会话，检查无跨会话泄漏和 turn 错序。
3. 每次运行保存：命令、环境、配置、manifest/hash、日志、Trace v5、结果、评分、GPU 峰值和退出状态。
4. 若 watchdog 终止清理卡住的进程，必须先验证 `status=completed`、Trace/结果行数和文件哈希。

### P3：实现并筛选 Task 2

按组件逐项实现并测试：`t2_budget_b0`、`t2_candidate_grid`、`t2_relation_grid`、`t2_circularity`、`t2_answerability`、`t2_evidence_card`、`t2_typed_repair`、`t2_core_full`。

每个变体必须经过：fake backend 单测 → 1 条真实 smoke → dev80。dev80 只用于冻结核心组件和阈值，不得查看 test120 后继续调参。

### P4：实现并筛选 Task 3

以 `t3_last_turn` 为强基线，逐项实现 `t3_full_history`、`t3_user_only`、`t3_structured_state`、`t3_verified_state`、`t3_state_gated`、`t3_core_full` 及 equal-token history-summary control。

每个 history mode 必须经过：状态/泄漏单测 → 一整段真实会话 smoke → dev40。状态声明需区分 verified、provisional、quarantined，并审计 history/image gate。

### P5：confirmatory 与后 40 表

1. 冻结 `t2_core_full`、`t3_core_full`、state schema 和配置哈希。
2. Task 2 在 `t2_test120` 只运行 `t2_b0`、`t2_budget_b0`、`t2_core_full`。
3. Task 3 在 `t3_test60` 只运行 `t3_last_turn`、equal-token summary control、`t3_core_full`。
4. 从 confirmatory 既有结果按冻结顺序切片最后 40；不得重复推理、按结果选样本或二次调参。
5. 生成隐藏版本名、门控、分数和 history mode 的盲审表，人工标注 C/P/I/M。
6. 导出 CSV、JSON、Markdown、40-ID manifest/hash、公式审计和转移表。

## 6. 最终表定义

固定列：`实验、版本、C/P/I/M、严格 Acc.、部分 Acc.、覆盖率、Missing、Halluc.、Truth.`

- `StrictAcc = C / N`
- `PartialAcc = (C + 0.5P) / N`
- `Coverage = (C + P) / N`
- `Missing = M / N`
- `Hallucination = (P + I) / N`
- `Truthfulness = (2C + M) / N - 1 = StrictAcc - Hallucination`

Task 3 另报告总 turn 数、EPC、Recovery@1、History Harm、会话全正确率，并按 conversation 聚类 bootstrap。

当前没有三版本 confirmatory 输出和完整盲审标签，因此最终后 40 表尚未完成，严禁填入预计数字或把 smoke 指标当作正式结果。

## 7. 完成判据

只有同时满足以下条件，才能汇报“正式对比完成”：

- Task 2 / Task 3 所有 confirmatory 版本均有完整配置、Trace v5、CSV、日志与哈希。
- 每行人工盲审 `C + P + I + M = 40`，且复核表不泄漏版本信息。
- Task 3 指标按 conversation 聚类，所有分母可审计。
- 最终表可以从冻结产物一键重建，篡改顺序、ID、标签或版本名会被拒绝。
- 同时报告绝对 W→C、C→W 和净改对数；置信区间跨 0 时只表述为趋势或负结果。

## 8. 明确禁止

- 不得把单元测试通过数当作模型准确率。
- 不得把 1 条/1 会话 smoke 当作正式实验。
- 不得把 prepare-only manifest 当作真实推理结果。
- 不得把 `t3_no_history` 写成 `t3_last_turn`。
- 不得解除 support gate 后复用同一逻辑制造伪对比。
- 不得按结果重选 main40 或用 exact match 替代人工 C/P/I/M 盲审。
- 不得在仓库、日志或交接文档中提交密钥、密码或私钥。

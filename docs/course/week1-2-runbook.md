# 前两周环境与实验运行手册

## 1. 已验证本机环境

- 设备：Apple M4、24 GB 统一内存、macOS 26.5.1、arm64。
- Python：Homebrew Python 3.11.15，虚拟环境位于项目根目录 `.venv`。
- 关键版本：`torch 2.13.0`、`datasets 3.6.0`、`transformers 5.12.1`、`cragmm-search-pipeline 0.5.1`、`mlx-vlm 0.6.4`。
- PyTorch MPS 可用，CUDA 不可用。完整机器快照见 `artifacts/week2/environment.json`。

## 2. 从干净终端复现

```bash
cd /Users/jiahao/Code/智能实训/meta-comprehensive-rag-benchmark-starter-kit-main
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools wheel
.venv/bin/pip install -r requirements-macos.txt
.venv/bin/python scripts/week2_environment_check.py \
  --output artifacts/week2/environment.json
.venv/bin/python -m unittest discover -s tests
```

官方 Linux/CUDA 环境继续使用根目录 `Dockerfile` 与 `requirements.txt`。不要把 `requirements-macos.txt` 用作 AIcrowd 镜像依赖。

## 3. Agent 模式

正式入口为 `agents/user_config.py`，核心实现为 `agents/course_agent_v2.py`。

```bash
export CRAG_TASK_MODE=vision   # 纯视觉 baseline，不调用检索
export CRAG_TASK_MODE=task1    # image KG，代码路径禁止 web search
export CRAG_TASK_MODE=task2    # image KG + web search

export CRAG_BACKEND=mlx        # Apple Silicon
export CRAG_BACKEND=vllm       # 官方 Linux/CUDA
export CRAG_MODEL=/absolute/local/model/path
```

每次实验设置独立的 `CRAG_TRACE_PATH`，保存检索证据、耗时、回答 token 数、拒答原因和异常。实验脚本会自动配置该路径。

## 4. 固定样例与实验命令

真实 validation 分片：

```text
local_data/crag-mm-single-turn-public/validation/validation-00004-of-00005.parquet
```

该分片来自公开数据集提交 `711dd84fa2f1611975d476261afcb07292151923`，含 387 条数据。固定种子 `20260712` 选出 30 条，清单见：

```text
artifacts/week2/real_validation_shard_manifest/sample_manifest.json
```

仅准备样例、不加载模型：

```bash
.venv/bin/python scripts/week2_experiment.py \
  --mode vision \
  --num-samples 30 \
  --seed 20260712 \
  --dataset-path local_data/crag-mm-single-turn-public/validation/validation-00004-of-00005.parquet \
  --output-dir artifacts/week2/real_validation_shard_manifest \
  --prepare-only
```

真实 Vision Baseline：

```bash
.venv/bin/python scripts/week2_experiment.py \
  --mode vision \
  --backend mlx \
  --model /absolute/path/to/mlx-vlm-model \
  --num-samples 30 \
  --seed 20260712 \
  --dataset-path local_data/crag-mm-single-turn-public/validation/validation-00004-of-00005.parquet \
  --output-dir artifacts/week2/real_vision_30
```

完整任务1/任务2需要官方图像与网页索引。云端索引就绪后运行：

```bash
.venv/bin/python scripts/week2_experiment.py --mode task1 --backend vllm \
  --num-samples 30 --seed 20260712 --output-dir artifacts/week2/real_task1_30

.venv/bin/python scripts/week2_experiment.py --mode task2 --backend vllm \
  --num-samples 30 --seed 20260712 --output-dir artifacts/week2/real_task2_30
```

汇总真实实验：

```bash
.venv/bin/python scripts/summarize_week2_experiments.py \
  artifacts/week2/real_vision_30 \
  artifacts/week2/real_task1_30 \
  artifacts/week2/real_task2_30 \
  --output artifacts/week2/experiment_summary.csv
```

## 5. 合成契约验收

```bash
.venv/bin/python scripts/offline_week2_acceptance.py
```

该脚本运行 4 组、每组 30 条的可控样例，验证模式切换、批处理顺序、任务1禁用网页、任务2网页证据、拒答、追踪和指标汇总。结果位于 `artifacts/week2/offline_acceptance/`。

这些结果不是 CRAG-MM benchmark，不得写成官方或真实模型性能。

## 6. 指标口径

- 未配置语义评测模型时，程序输出的是 exact-match 指标，不等于官方排行榜成绩。
- `I don't know` 计为 Missing，不计为正确。
- 人工复核需要保留 `interaction_id`、标准答案、模型答案和复核理由。
- PPT、周报和报告中的数字必须能追溯到 CSV/JSONL，禁止手工编造或把合成测试数字当成真实效果。

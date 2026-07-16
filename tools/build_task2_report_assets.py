#!/usr/bin/env python3
"""Build reproducible data assets for the Task 2 LaTeX report."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "artifacts/week2/real_task2_30"
LOG_DIR = ROOT / "artifacts/week2/cloud_task2_run/logs"
ASSET_DIR = ROOT / "output/pdf/assets"


MANUAL_REVIEW = {
    "Is it possible to fit a 90 by 60-inch table in the back seat of the 2025 model of this car?": ("incorrect", "inappropriate_refusal", "未回答尺寸比较，错误触发人物识别拒答"),
    "How many states are in the country whose national bird is this animal?": ("partial", "partial_coverage", "识别美国与白头海雕，但没有回答 50 个州"),
    "How long did it take to build this skyscraper?": ("incorrect", "reasoning_or_fact", "回答约 4 年，参考答案约 3 年"),
    "What are the fermentation effects of this bread?": ("partial", "partial_coverage", "覆盖降糖与消化影响，未覆盖全部参考要点"),
    "Where has this breed worked as a sled dog in history?": ("partial", "partial_coverage", "回答 Alaska，遗漏 Antarctica 与 Northern Asia"),
    "Where can this plant typically be found?": ("incorrect", "entity_error", "将 Trithuria inconspicua 识别为其他植物分布"),
    "What is the oldest automobile company in the country where the manufacturer of this car belongs?": ("correct", "correct", "核心答案 Kia Motors、1944 与参考一致"),
    "Who is the Prime Minister of the country where this food originated?": ("incorrect", "entity_error", "将意大利食物误判为瑞士来源"),
    "What are the popular places to visit in this neighborhood as of 2024?": ("partial", "partial_coverage", "覆盖 Ipanema、Copacabana、Arpoador，列表不完整"),
    "Prior to Heinz, why were these sauces dangerous?": ("partial", "partial_coverage", "覆盖保存与卫生风险，但关键原因表述不完全一致"),
    "What are the ingredients for its filling?": ("incorrect", "entity_error", "将 grape pie 误识别为 blueberry pie"),
    "What award did the designer of this car win in 1959?": ("correct", "correct", "Compasso d'Oro 与参考一致"),
    "Which state does the city belong to, where this dish was originally made?": ("correct", "correct", "Rajasthan 与参考一致"),
    "What is the typical native elevation range of this tree species?": ("incorrect", "reasoning_or_fact", "单位换算后的范围上界明显偏高"),
    "what century did this food originate in?": ("correct", "correct", "Beef Stroganoff、19 世纪与参考一致"),
    "What is the capital of the country this plant is native to?": ("incorrect", "entity_error", "将澳大利亚植物误识别为巴西植物"),
    "What is the the gross domestic product of the country where this museum is located?": ("incorrect", "entity_error", "将台湾博物馆误判为美国博物馆"),
    "In what years has this car changed its look since its first year?": ("incorrect", "entity_error", "车型识别与参考 Honda Accord 不一致"),
    "What is the difference between these and pierogi?": ("incorrect", "off_target_or_truncated", "回答在描述图片处截断，未形成比较"),
    "What is the average lifespan of this animal in the wild compared to rabbit in the wild?": ("incorrect", "entity_error", "将 Lepus 问题误解为 dog 与 rabbit"),
    "Which building lasted longer in the Assynt lands, this house or Ardvreck castle?": ("correct", "correct", "Ardvreck Castle 更久，与参考时间线一致"),
    "What is the typical color of the fruit of this plant?": ("incorrect", "entity_error", "Calamus moti 果实颜色误判"),
    "what is the typical habitat of this creature?": ("incorrect", "entity_error", "将绿色毛虫栖息地误答为淡水湿地"),
    "What type of engine is in this vehicle?": ("incorrect", "reasoning_or_fact", "参考 V8，回答 2.7L 直列四缸"),
    "who constructed this historic fortress?": ("correct", "correct", "Gosamaru 与参考一致"),
    "Which one has a more powerful base engine, this vehicle or the Toyota 4Runner?": ("incorrect", "reasoning_or_fact", "比较结论与参考相反"),
    "What is the nickname of the state in Australia where this plant is endemic?": ("incorrect", "entity_error", "Queensland/Sunshine State 误判为 Western Australia"),
    "How do Ammophila compare to the length of this plant?": ("incorrect", "entity_error", "实体与量纲均发生错配"),
    "What was the primary source of food for this large prehistoric fish?": ("correct", "correct", "核心回答 smaller fish/other fish 与参考一致"),
    "When does the flowering period of this plant occur?": ("partial", "partial_coverage", "回答 May，参考范围 February-June"),
}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    run = json.loads((RESULT_DIR / "run_config.json").read_text(encoding="utf-8"))
    traces = [
        json.loads(line)
        for line in (RESULT_DIR / "agent_trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with (RESULT_DIR / "turn_evaluation_results_all.csv").open(encoding="utf-8") as handle:
        evaluation_rows = list(csv.DictReader(handle))

    latency_rows = []
    for index, trace in enumerate(traces, 1):
        latency_rows.append({
            "sample": index,
            "image_ms": round(trace["image_search_ms"], 3),
            "web_ms": round(trace["web_search_ms"], 3),
            "generation_ms": round(trace["generation_ms"], 3),
            "total_ms": round(trace["total_ms"], 3),
            "answer_tokens": trace["answer_token_count"],
            "image_evidence": len(trace["image_evidence"]),
            "web_evidence": len(trace["web_evidence"]),
        })
    write_csv(ASSET_DIR / "latency.csv", list(latency_rows[0]), latency_rows)

    review_rows = []
    for row in evaluation_rows:
        label, cause, note = MANUAL_REVIEW[row["query"]]
        review_rows.append({
            "query": row["query"],
            "ground_truth": row["ground_truth"],
            "answer": row["agent_response"],
            "label": label,
            "cause": cause,
            "note": note,
        })
    write_csv(RESULT_DIR / "manual_review.csv", list(review_rows[0]), review_rows)

    started = datetime.fromisoformat(
        (LOG_DIR / "task2-full30.started").read_text(encoding="utf-8").strip()
    )
    ended = datetime.fromisoformat(run["captured_at_utc"])
    telemetry_rows = []
    with (LOG_DIR / "gpu-telemetry.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, skipinitialspace=True):
            timestamp = datetime.strptime(
                row["timestamp"].strip(), "%Y/%m/%d %H:%M:%S.%f"
            ).replace(tzinfo=started.tzinfo)
            if started <= timestamp <= ended:
                telemetry_rows.append({
                    "seconds": round((timestamp - started).total_seconds(), 3),
                    "index": int(row["index"].strip()),
                    "memory_mib": int(row["memory.used [MiB]"].split()[0]),
                    "utilization": int(row["utilization.gpu [%]"].split()[0]),
                    "power_w": float(row["power.draw [W]"].split()[0]),
                    "temperature_c": int(row["temperature.gpu"].strip()),
                })
    for gpu_index in (0, 1):
        gpu_rows = [row for row in telemetry_rows if row["index"] == gpu_index]
        write_csv(
            ASSET_DIR / f"gpu{gpu_index}.csv",
            ["seconds", "memory_mib", "utilization", "power_w", "temperature_c"],
            [{key: value for key, value in row.items() if key != "index"} for row in gpu_rows],
        )

    latency_summary = {}
    for key in ("image_search_ms", "web_search_ms", "generation_ms", "total_ms"):
        values = [float(trace[key]) for trace in traces]
        latency_summary[key] = {
            "min": min(values),
            "median": statistics.median(values),
            "p95": percentile(values, 0.95),
            "max": max(values),
        }

    gpu_summary = {}
    for gpu_index in (0, 1):
        gpu_rows = [row for row in telemetry_rows if row["index"] == gpu_index]
        gpu_summary[str(gpu_index)] = {
            "samples": len(gpu_rows),
            "max_memory_mib": max(row["memory_mib"] for row in gpu_rows),
            "max_utilization": max(row["utilization"] for row in gpu_rows),
            "mean_utilization": statistics.mean(row["utilization"] for row in gpu_rows),
            "max_power_w": max(row["power_w"] for row in gpu_rows),
            "max_temperature_c": max(row["temperature_c"] for row in gpu_rows),
        }

    summary = {
        "run": run,
        "counts": {
            "manifest": len(json.loads((RESULT_DIR / "sample_manifest.json").read_text(encoding="utf-8"))),
            "traces": len(traces),
            "evaluation_rows": len(evaluation_rows),
            "status_ok": sum(trace["status"] == "ok" for trace in traces),
            "trace_errors": sum(bool(trace["errors"]) for trace in traces),
            "ego": sum(row["is_ego"] == "True" for row in evaluation_rows),
            "external_cached": sum(item["status"] == "cached" for item in json.loads((RESULT_DIR / "image_prefetch.json").read_text(encoding="utf-8"))),
        },
        "latency_ms": latency_summary,
        "answer_tokens": {
            "min": min(trace["answer_token_count"] for trace in traces),
            "median": statistics.median(trace["answer_token_count"] for trace in traces),
            "max": max(trace["answer_token_count"] for trace in traces),
            "at_or_over_75": sum(trace["answer_token_count"] >= 75 for trace in traces),
        },
        "evidence": {
            "image_median": statistics.median(len(trace["image_evidence"]) for trace in traces),
            "web_median": statistics.median(len(trace["web_evidence"]) for trace in traces),
            "both_sources": sum(bool(trace["image_evidence"]) and bool(trace["web_evidence"]) for trace in traces),
        },
        "manual_review_non_official": {
            "labels": dict(Counter(row["label"] for row in review_rows)),
            "causes": dict(Counter(row["cause"] for row in review_rows)),
            "reviewer_count": 1,
        },
        "gpu": gpu_summary,
    }
    (ASSET_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

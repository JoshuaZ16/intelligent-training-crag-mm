# VEGA-RAG Real Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement B0/A1/A2/A3/FULL as deterministic CRAG-MM Task 2 variants, run every comparison on the same 30 real samples in paired dual-A800 rounds, and generate a visually verified PDF whose numbers come only from synchronized logs and result files.

**Architecture:** Keep `CourseRAGAgentV2` as the orchestration boundary and add three focused, dependency-light VEGA modules for entity agreement, adaptive retrieval, and calibrated abstention. Extend trace/config types without changing B0 behavior, add reproducible calibration/pair-run/scoring scripts, then use the existing vLLM and `CRAGEvaluator` path for all real runs. Generate all tables and charts from machine-readable artifacts and clearly label the one-reviewer semantic metrics as non-official.

**Tech Stack:** Python 3.10/3.12, dataclasses, vLLM 0.7.3, CRAG-MM search pipeline, unittest, pandas/numpy, XeLaTeX/TikZ/pgfplots, tmux, nvidia-smi.

**Execution constraint:** The user has not authorized commits or pushes. Replace every normal commit checkpoint with `git diff --check` and a progress log entry; do not run `git commit`, `git push`, or create a PR.

---

## File map

**Create**

- `agents/vega_config.py`: variant enum, frozen thresholds, prepared-turn and trace-compatible dataclasses.
- `agents/entity_agreement.py`: deterministic tokenization, entity support scores, reordering, margin, numeric conflict detection.
- `agents/adaptive_retrieval.py`: query rewrite, trigger decision, merge/deduplicate Web evidence.
- `agents/calibrated_abstention.py`: low-confidence/conflict gate and stable refusal reasons.
- `tests/test_vega_config.py`: config parsing and threshold validation.
- `tests/test_entity_agreement.py`: A1 scoring/reordering/conflict tests.
- `tests/test_adaptive_retrieval.py`: A2 trigger/query/merge tests.
- `tests/test_calibrated_abstention.py`: A3/FULL action tests.
- `tests/test_vega_agent_integration.py`: B0 non-regression and five-version orchestration tests.
- `scripts/calibrate_vega_thresholds.py`: fixed-grid calibration from real B0-cal and A2-enrich-all answers.
- `scripts/run_vega_pair.py`: simultaneous two-GPU process launch, telemetry, exit capture, pair validity metadata.
- `scripts/build_vega_blind_review.py`: deterministic version-blinded review sheet.
- `scripts/score_vega_comparison.py`: metric formulas, paired bootstrap, exact McNemar and transition tables.
- `tools/build_vega_report_assets.py`: verified CSV/JSON assets for LaTeX.
- `output/pdf/VEGA-RAG-真实多轮性能对比实验报告.tex`: final report source after real runs.

**Modify**

- `agents/course_agent_v2.py`: trace v3 fields, variant-aware preparation, forced refusal without generation, optional additional Web search.
- `scripts/week2_experiment.py`: variant, frozen-threshold, fixed-manifest and output metadata arguments.
- `tests/test_course_agent_v2.py`: preserve existing B0 behavior assertions.

**Generated artifacts**

- `artifacts/week2/vega_real_ablation/calibration/`
- `artifacts/week2/vega_real_ablation/round_1/{b0,a1}/`
- `artifacts/week2/vega_real_ablation/round_2/{b0,a2}/`
- `artifacts/week2/vega_real_ablation/round_3/{b0,a3}/`
- `artifacts/week2/vega_real_ablation/round_4/{b0,full}/`
- `artifacts/week2/vega_real_ablation/comparison/`

### Task 1: Variant configuration and frozen thresholds

**Files:**
- Create: `agents/vega_config.py`
- Create: `tests/test_vega_config.py`

- [ ] **Step 1: Write failing config tests**

```python
import json
import tempfile
import unittest
from pathlib import Path

from agents.vega_config import ExperimentVariant, VegaThresholds


class VegaConfigTest(unittest.TestCase):
    def test_variants_are_stable_cli_values(self):
        self.assertEqual(
            [item.value for item in ExperimentVariant],
            ["b0", "a1", "a2", "a3", "full", "a2_enrich_all"],
        )

    def test_thresholds_require_ordered_unit_interval(self):
        with self.assertRaises(ValueError):
            VegaThresholds(tau_low=0.6, tau_high=0.3)
        with self.assertRaises(ValueError):
            VegaThresholds(tau_low=-0.1, tau_high=0.6)

    def test_thresholds_load_frozen_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vega_calibration.json"
            path.write_text(json.dumps({"tau_low": 0.3, "tau_high": 0.6}), encoding="utf-8")
            value = VegaThresholds.from_json(path)
        self.assertEqual((value.tau_low, value.tau_high), (0.3, 0.6))
```

- [ ] **Step 2: Run the tests and verify the import failure**

Run:

```bash
python3 -m unittest tests.test_vega_config -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agents.vega_config'`.

- [ ] **Step 3: Implement the minimal frozen config**

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ExperimentVariant(str, Enum):
    B0 = "b0"
    A1 = "a1"
    A2 = "a2"
    A3 = "a3"
    FULL = "full"
    A2_ENRICH_ALL = "a2_enrich_all"


@dataclass(frozen=True)
class VegaThresholds:
    tau_low: float = 0.3
    tau_high: float = 0.6

    def __post_init__(self) -> None:
        if not (0.0 <= self.tau_low < self.tau_high <= 1.0):
            raise ValueError("thresholds must satisfy 0 <= tau_low < tau_high <= 1")

    @classmethod
    def from_json(cls, path: str | Path) -> "VegaThresholds":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(float(payload["tau_low"]), float(payload["tau_high"]))
```

- [ ] **Step 4: Run config tests**

Run: `python3 -m unittest tests.test_vega_config -v`
Expected: 3 tests, `OK`.

- [ ] **Step 5: Check scope without committing**

Run: `git diff --check -- agents/vega_config.py tests/test_vega_config.py`
Expected: exit 0 and no output.

### Task 2: A1 entity agreement and deterministic reordering

**Files:**
- Create: `agents/entity_agreement.py`
- Create: `tests/test_entity_agreement.py`

- [ ] **Step 1: Write failing A1 tests**

```python
import unittest
from dataclasses import dataclass, field

from agents.entity_agreement import score_and_rerank, detect_numeric_conflict


@dataclass(frozen=True)
class Item:
    evidence_id: str
    source: str
    text: str
    score: float | None = None
    metadata: dict = field(default_factory=dict)


class EntityAgreementTest(unittest.TestCase):
    def test_web_support_can_promote_second_image_candidate(self):
        image = [
            Item("KG1", "image_kg", "Museum of Texas | location: Austin", 0.82),
            Item("KG2", "image_kg", "National Taiwan Museum | location: Taipei", 0.79),
        ]
        web = [Item("WEB1", "web", "National Taiwan Museum is located in Taipei", 0.9)]
        result = score_and_rerank(image, web, "Where is this museum located?")
        self.assertEqual([x.evidence_id for x in result.items], ["KG2", "KG1"])
        self.assertGreater(result.top_score, result.second_score)

    def test_no_support_preserves_original_tie_order(self):
        image = [Item("KG1", "image_kg", "Alpha", 0.8), Item("KG2", "image_kg", "Beta", 0.8)]
        result = score_and_rerank(image, [], "Who built it?")
        self.assertEqual([x.evidence_id for x in result.items], ["KG1", "KG2"])

    def test_numeric_conflict_requires_disjoint_numbers(self):
        self.assertTrue(detect_numeric_conflict(["engine: 2.0 L"], ["base engine is 2.7 L"]))
        self.assertFalse(detect_numeric_conflict(["built in 1959"], ["award received in 1959"]))
```

- [ ] **Step 2: Verify the tests fail before implementation**

Run: `python3 -m unittest tests.test_entity_agreement -v`
Expected: FAIL importing `agents.entity_agreement`.

- [ ] **Step 3: Implement token F1, score clipping and stable ordering**

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

TOKEN_RE = re.compile(r"[a-z0-9]+")
NUMBER_RE = re.compile(r"(?<![a-z])\d+(?:\.\d+)?")


def tokens(value: str) -> list[str]:
    return TOKEN_RE.findall((value or "").lower())


def token_f1(left: str, right: str) -> float:
    a, b = set(tokens(left)), set(tokens(right))
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(a), overlap / len(b)
    return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class AgreementResult:
    items: list[Any]
    candidate_scores: list[dict[str, float | str]]
    top_score: float
    second_score: float
    margin: float


def score_and_rerank(image_items: Sequence[Any], web_items: Sequence[Any], query: str) -> AgreementResult:
    web_text = " ".join(item.text for item in web_items)
    scored = []
    for index, item in enumerate(image_items):
        name, _, attributes = item.text.partition(" | ")
        image_norm = min(1.0, max(0.0, float(item.score or 0.0)))
        web_support = token_f1(name, web_text)
        attribute_support = token_f1(query, attributes)
        total = 0.50 * image_norm + 0.35 * web_support + 0.15 * attribute_support
        scored.append((index, item, total, web_support, attribute_support))
    scored.sort(key=lambda row: (-row[2], row[0]))
    values = [row[2] for row in scored]
    top, second = (values + [0.0, 0.0])[:2]
    return AgreementResult(
        items=[row[1] for row in scored],
        candidate_scores=[{"evidence_id": row[1].evidence_id, "entity_score": row[2], "web_support": row[3], "attribute_support": row[4]} for row in scored],
        top_score=top,
        second_score=second,
        margin=max(0.0, top - second),
    )


def detect_numeric_conflict(image_texts: Sequence[str], web_texts: Sequence[str]) -> bool:
    image_numbers = set(NUMBER_RE.findall(" ".join(image_texts).lower()))
    web_numbers = set(NUMBER_RE.findall(" ".join(web_texts).lower()))
    return bool(image_numbers and web_numbers and image_numbers.isdisjoint(web_numbers))
```

- [ ] **Step 4: Run A1 tests and the existing agent suite**

Run:

```bash
python3 -m unittest tests.test_entity_agreement tests.test_course_agent_v2 -v
```

Expected: all tests `OK`.

### Task 3: A2 adaptive retrieval

**Files:**
- Create: `agents/adaptive_retrieval.py`
- Create: `tests/test_adaptive_retrieval.py`

- [ ] **Step 1: Write failing trigger/query/merge tests**

```python
import unittest
from dataclasses import dataclass

from agents.adaptive_retrieval import should_expand, rewrite_query, merge_web_results


@dataclass(frozen=True)
class Web:
    evidence_id: str
    text: str
    metadata: dict
    score: float = 0.5


class AdaptiveRetrievalTest(unittest.TestCase):
    def test_only_mid_confidence_expands(self):
        self.assertFalse(should_expand(0.2, 0.3, 0.6, False))
        self.assertTrue(should_expand(0.4, 0.3, 0.6, False))
        self.assertFalse(should_expand(0.8, 0.3, 0.6, False))
        self.assertTrue(should_expand(0.8, 0.3, 0.6, True))

    def test_rewrite_begins_with_selected_entity(self):
        self.assertEqual(
            rewrite_query("National Taiwan Museum", "Where is this museum located?"),
            "National Taiwan Museum Where is this museum located?",
        )

    def test_merge_deduplicates_url_and_renumbers(self):
        old = [Web("WEB1", "old", {"url": "u1"})]
        new = [Web("WEBX", "duplicate", {"url": "u1"}), Web("WEBY", "new", {"url": "u2"})]
        merged = merge_web_results(old, new)
        self.assertEqual([x.evidence_id for x in merged], ["WEB1", "WEB2"])
        self.assertEqual([x.metadata["url"] for x in merged], ["u1", "u2"])
```

- [ ] **Step 2: Verify import failure**

Run: `python3 -m unittest tests.test_adaptive_retrieval -v`
Expected: FAIL importing `agents.adaptive_retrieval`.

- [ ] **Step 3: Implement deterministic A2 helpers**

```python
from dataclasses import replace
from typing import Any, Sequence

from agents.entity_agreement import tokens


def should_expand(score: float, tau_low: float, tau_high: float, force_all: bool = False) -> bool:
    return force_all or (tau_low < score < tau_high)


def rewrite_query(entity_name: str, query: str, max_chars: int = 240) -> str:
    value = " ".join([entity_name.strip(), query.strip()]).strip()
    return value[:max_chars]


def merge_web_results(existing: Sequence[Any], additional: Sequence[Any]) -> list[Any]:
    seen, merged = set(), []
    for item in [*existing, *additional]:
        key = (item.metadata.get("url") or "", " ".join(tokens(item.text)))
        if key in seen:
            continue
        seen.add(key)
        merged.append(replace(item, evidence_id=f"WEB{len(merged) + 1}"))
    return merged
```

- [ ] **Step 4: Run A2 tests**

Run: `python3 -m unittest tests.test_adaptive_retrieval -v`
Expected: 3 tests, `OK`.

### Task 4: A3 calibrated abstention

**Files:**
- Create: `agents/calibrated_abstention.py`
- Create: `tests/test_calibrated_abstention.py`

- [ ] **Step 1: Write failing gate tests**

```python
import unittest

from agents.calibrated_abstention import GateAction, choose_action
from agents.vega_config import ExperimentVariant, VegaThresholds


class AbstentionTest(unittest.TestCase):
    def setUp(self):
        self.thresholds = VegaThresholds(0.3, 0.6)

    def test_a3_refuses_only_low_or_conflict(self):
        self.assertEqual(choose_action(ExperimentVariant.A3, 0.2, False, self.thresholds), GateAction.ABSTAIN)
        self.assertEqual(choose_action(ExperimentVariant.A3, 0.8, True, self.thresholds), GateAction.ABSTAIN)
        self.assertEqual(choose_action(ExperimentVariant.A3, 0.5, False, self.thresholds), GateAction.ANSWER)

    def test_full_has_three_branches(self):
        self.assertEqual(choose_action(ExperimentVariant.FULL, 0.2, False, self.thresholds), GateAction.ABSTAIN)
        self.assertEqual(choose_action(ExperimentVariant.FULL, 0.4, False, self.thresholds), GateAction.EXPAND)
        self.assertEqual(choose_action(ExperimentVariant.FULL, 0.8, False, self.thresholds), GateAction.ANSWER)
```

- [ ] **Step 2: Verify tests fail**

Run: `python3 -m unittest tests.test_calibrated_abstention -v`
Expected: FAIL importing the module.

- [ ] **Step 3: Implement the gate without model calls**

```python
from enum import Enum

from agents.vega_config import ExperimentVariant, VegaThresholds


class GateAction(str, Enum):
    ANSWER = "answer"
    EXPAND = "expand"
    ABSTAIN = "abstain"


def choose_action(variant: ExperimentVariant, score: float, conflict: bool, thresholds: VegaThresholds) -> GateAction:
    if variant is ExperimentVariant.A3:
        return GateAction.ABSTAIN if conflict or score <= thresholds.tau_low else GateAction.ANSWER
    if variant is ExperimentVariant.FULL:
        if conflict or score <= thresholds.tau_low:
            return GateAction.ABSTAIN
        if score < thresholds.tau_high:
            return GateAction.EXPAND
    if variant is ExperimentVariant.A2_ENRICH_ALL:
        return GateAction.EXPAND if score > thresholds.tau_low else GateAction.ABSTAIN
    return GateAction.ANSWER
```

- [ ] **Step 4: Run A3 tests**

Run: `python3 -m unittest tests.test_calibrated_abstention -v`
Expected: 2 tests, `OK`.

### Task 5: Integrate five variants while proving B0 non-regression

**Files:**
- Modify: `agents/course_agent_v2.py:53-117, 369-500`
- Create: `tests/test_vega_agent_integration.py`
- Modify: `tests/test_course_agent_v2.py:104-168`

- [ ] **Step 1: Add failing integration tests**

Create fake search fixtures where the first image candidate is visually higher but the second is Web-supported. Assert:

```python
def test_b0_keeps_original_prompt_and_one_image_one_web_call():
    agent, search, backend = build_agent("b0")
    agent.batch_generate_response(["Where?"], [image], [[]])
    assert [call.kind for call in search.calls] == ["image", "web"]
    assert backend.prompts[0].index("KG1") < backend.prompts[0].index("KG2")

def test_a1_reorders_without_extra_search_or_refusal():
    agent, search, backend = build_agent("a1")
    agent.batch_generate_response(["Where?"], [image], [[]])
    assert [call.kind for call in search.calls] == ["image", "web"]
    assert backend.prompts[0].index("KG2") < backend.prompts[0].index("KG1")

def test_a2_adds_exactly_one_web_call_in_mid_band():
    agent, search, backend = build_agent("a2", score_fixture="mid")
    agent.batch_generate_response(["Where?"], [image], [[]])
    assert [call.kind for call in search.calls] == ["image", "web", "web"]

def test_a3_low_score_returns_idk_without_generation():
    agent, search, backend = build_agent("a3", score_fixture="low")
    assert agent.batch_generate_response(["Where?"], [image], [[]]) == ["I don't know"]
    assert backend.prompts == []

def test_full_writes_trace_v3_action_and_scores():
    agent, search, backend, trace_path = build_traced_agent("full", score_fixture="mid")
    agent.batch_generate_response(["Where?"], [image], [[]])
    trace = json.loads(Path(trace_path).read_text())
    assert trace["trace_schema"] == "v3"
    assert trace["experiment_variant"] == "full"
    assert trace["gate_action"] == "expand"
    assert len(trace["entity_candidates"]) == 2
```

- [ ] **Step 2: Run integration tests and verify failures**

Run: `python3 -m unittest tests.test_vega_agent_integration -v`
Expected: failures for missing variant config and trace fields.

- [ ] **Step 3: Extend config and trace with backward-compatible defaults**

Add to `AgentConfig`:

```python
variant: ExperimentVariant = ExperimentVariant.B0
thresholds: VegaThresholds = field(default_factory=VegaThresholds)
adaptive_k: int = 5
```

Add to `TurnTrace`:

```python
trace_schema: str = "v3"
experiment_variant: str = "b0"
entity_candidates: List[Dict[str, Any]] = field(default_factory=list)
entity_agreement: float = 0.0
entity_margin: float = 0.0
evidence_conflict: bool = False
gate_action: str = "answer"
additional_search_query: str = ""
additional_web_evidence: List[Dict[str, Any]] = field(default_factory=list)
additional_web_search_ms: float = 0.0
search_call_count: int = 0
```

- [ ] **Step 4: Introduce a prepared-turn object and forced-answer path**

```python
@dataclass
class PreparedTurn:
    prompt: str
    trace: TurnTrace
    forced_answer: str | None = None
```

Modify `batch_generate_response` to send only non-forced indices to the backend, map generated outputs back to their original positions, and write every trace. When all items are forced refusals, do not invoke `answer_batch`.

- [ ] **Step 5: Add A1/A2/A3/FULL orchestration in `_prepare_turn`**

The exact order is:

```python
raw_image -> parse_image_evidence
baseline_query -> raw_web -> parse_web_evidence
agreement = score_and_rerank(...)
action = choose_action(...)
if action == EXPAND:
    additional_raw_web = _search(rewrite_query(...), adaptive_k, "web_extra", trace)
    web_evidence = merge_web_results(web_evidence, parse_web_evidence(additional_raw_web, ...))
if action == ABSTAIN:
    forced_answer = IDK_RESPONSE
prompt_image_evidence = original image evidence for B0; agreement.items for A1/A2/A3/FULL
```

- [ ] **Step 6: Run all focused and legacy tests**

Run:

```bash
python3 -m unittest tests.test_vega_config tests.test_entity_agreement tests.test_adaptive_retrieval tests.test_calibrated_abstention tests.test_vega_agent_integration tests.test_course_agent_v2 -v
```

Expected: all tests `OK`; B0 test observes exactly the old search/prompt behavior.

### Task 6: Reproducible CLI, fixed manifest and run metadata

**Files:**
- Modify: `scripts/week2_experiment.py:43-180`
- Create: `tests/test_week2_experiment.py`

- [ ] **Step 1: Write failing CLI/manifest tests**

Test that:

```python
args = parse_args(["--variant", "a1", "--manifest-path", "sample_manifest.json"])
assert args.variant == "a1"
assert args.manifest_path.endswith("sample_manifest.json")

indices = load_manifest_indices(path)
assert indices == [69, 111, 176]
```

Also assert the run config contains `variant`, `thresholds_sha256`, `manifest_sha256`, `git_diff_sha256`, physical GPU label, start/end UTC, and exit-completion status.

- [ ] **Step 2: Run test and verify missing arguments/functions**

Run: `python3 -m unittest tests.test_week2_experiment -v`
Expected: FAIL for missing variant/manifest support.

- [ ] **Step 3: Add exact CLI arguments**

```python
parser.add_argument("--variant", choices=[v.value for v in ExperimentVariant], default="b0")
parser.add_argument("--manifest-path")
parser.add_argument("--thresholds-path")
parser.add_argument("--adaptive-k", type=int, default=5)
parser.add_argument("--run-id", required=True)
```

When `--manifest-path` is present, load `source_index` values and select exactly those rows instead of resampling. Compute SHA-256 for manifest, thresholds and the textual `git diff` using `hashlib.sha256`.

- [ ] **Step 4: Wire the variant config**

```python
thresholds = VegaThresholds.from_json(args.thresholds_path) if args.thresholds_path else VegaThresholds()
config = AgentConfig(
    task_mode=mode,
    trace_path=str(trace_path),
    variant=ExperimentVariant(args.variant),
    thresholds=thresholds,
    adaptive_k=args.adaptive_k,
)
```

- [ ] **Step 5: Run CLI tests and prepare-only verification**

Run:

```bash
python3 -m unittest tests.test_week2_experiment -v
python3 scripts/week2_experiment.py --mode task2 --backend vllm --variant b0 --run-id local-prepare --manifest-path artifacts/week2/real_task2_30/sample_manifest.json --dataset-path local_data/.../validation-00004-of-00005.parquet --output-dir /tmp/vega-prepare --prepare-only
```

Expected: tests `OK`; prepare-only prints exactly 30 source indices matching the manifest.

### Task 7: Real calibration pipeline

**Files:**
- Create: `scripts/calibrate_vega_thresholds.py`
- Create: `tests/test_calibrate_vega_thresholds.py`

**Pre-formal amendment from real calibration:** The unconstrained Truthfulness-first grid selected an all-missing solution. Before any primary-20 run, the user approved a minimum answer coverage of `0.70`. Preserve the unconstrained JSON, exclude candidates below this coverage, and apply the original metric ordering only to feasible candidates.

- [ ] **Step 1: Write failing fixed-grid selection tests**

Use four synthetic calibration rows with real labels and two real-answer columns. Assert `select_thresholds` evaluates only pairs from `[.2, .3, .4, .5, .6, .7]` with `low < high`, maximizes Truthfulness, then Accuracy, then lower Missing.

```python
chosen = select_thresholds(rows, grid=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
assert chosen["tau_low"] == 0.3
assert chosen["tau_high"] == 0.6
assert chosen["selection_order"] == ["truthfulness", "accuracy", "lower_missing", "simpler_threshold"]
```

- [ ] **Step 2: Verify test failure, then implement formulas exactly**

```python
def metrics(labels: list[str]) -> dict[str, float]:
    n = len(labels)
    correct = sum(x == "correct" for x in labels)
    missing = sum(x == "missing" for x in labels)
    hallucination = n - correct - missing
    return {
        "accuracy": correct / n,
        "missing": missing / n,
        "hallucination": hallucination / n,
        "truthfulness": ((2 * correct + missing) / n) - 1,
    }
```

The calibration output includes both real input file SHA-256 hashes, every grid candidate score, chosen thresholds and a canonical JSON SHA-256.

- [ ] **Step 3: Run calibration unit tests**

Run: `python3 -m unittest tests.test_calibrate_vega_thresholds -v`
Expected: all tests `OK`.

- [ ] **Step 4: Verify the coverage-constrained real freeze**

Run the real reviewed input with `--min-coverage 0.7`. Expected: the unconstrained `.5/.6` result remains preserved separately; the formal `vega_calibration.json` selects `.2/.3`, reports coverage `.8`, 12 feasible candidates and 3 rejected candidates.

### Task 8: Paired dual-GPU runner and failure preservation

**Files:**
- Create: `scripts/run_vega_pair.py`
- Create: `tests/test_run_vega_pair.py`

- [ ] **Step 1: Write process-orchestration tests with fake commands**

Inject command arrays rather than shell strings. Assert the runner:

- gives left process `CUDA_VISIBLE_DEVICES=0` and right process `CUDA_VISIBLE_DEVICES=1`;
- records both start UTC values and `start_delta_seconds <= 30`;
- captures each exit status separately;
- marks `pair_valid=false` if either exit is nonzero or result count is not 30;
- never deletes a failed directory.

- [ ] **Step 2: Implement subprocess and telemetry lifecycle**

Use `subprocess.Popen([...], env=...)` for both agents and a third `nvidia-smi --query-gpu=... --loop-ms=5000` process. Terminate telemetry only after both sides exit. Write `pair_config.json`, `left/exit_status.txt`, `right/exit_status.txt` and a top-level `pair_summary.json` atomically via temporary files followed by `Path.replace`.

- [ ] **Step 3: Run orchestration tests**

Run: `python3 -m unittest tests.test_run_vega_pair -v`
Expected: all tests `OK`.

### Task 9: Full local verification and remote synchronization

**Files:**
- All implementation and tests above

- [ ] **Step 1: Run the complete local suite with Python 3.12**

Run:

```bash
/Users/yufan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest discover -s tests -v
```

Expected: all existing 32 tests plus new VEGA tests pass; zero failures/errors.

- [ ] **Step 2: Run `git diff --check` and secret scan**

Run:

```bash
git diff --check
rg -n 'BEGIN .*PRIVATE KEY|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|sshpass' agents scripts tests
```

Expected: diff check exit 0; no secret matches.

- [ ] **Step 3: Synchronize only required code to the server**

Use rsync exclusions for `.git`, output artifacts, caches and credentials. Record a local file-list SHA-256 and a remote file-list SHA-256; they must match before experiments.

- [ ] **Step 4: Run remote unit tests**

Run inside the existing `cragmm` environment and save to `artifacts/week2/vega_real_ablation/logs/remote-unit-tests-pre-run.log`.

Expected: the same test count as local and final `OK`.

### Task 10: Smoke tests and real calibration

**Files:**
- Generated remote artifact directories

- [ ] **Step 1: Run 1-sample smoke for all six executable variants**

Variants: `b0`, `a1`, `a2`, `a3`, `full`, `a2_enrich_all`. Each must exit 0, write one trace v3 row and one result row.

- [ ] **Step 2: Run 5-sample smoke for A1/A2/A3/FULL**

Each must exit 0; B0 uses two search calls per sample, A2/FULL mid-band samples use one additional call, and forced A3/FULL refusals do not invoke generation for those rows.

- [ ] **Step 3: Run paired 10-sample calibration**

GPU0: `B0-cal`; GPU1: `A2-enrich-all`. Use the first 10 manifest rows and save complete logs/telemetry.

- [ ] **Step 4: Build and complete the blinded 20-answer calibration review**

Label each real answer using `correct/partial/incorrect/missing`, add a cause and concise rationale, then run the calibration script. Freeze `vega_calibration.json`; never modify it after starting R1.

### Task 11: Four real paired 30-sample rounds

**Files:**
- Generated remote and synchronized artifact directories

- [ ] **Step 1: Run R1 `B0-R1 vs A1`**

Accept only if both exit 0, both contain 30 identical query IDs, start delta is at most 30 seconds and both trace files have 30 rows.

- [ ] **Step 2: Run R2 `B0-R2 vs A2`**

Use the immutable threshold file from Task 10 and the same acceptance gate.

- [ ] **Step 3: Run R3 `B0-R3 vs A3`**

Verify every forced refusal has a nonempty reason and no generation record for that sample.

- [ ] **Step 4: Run R4 `B0-R4 vs FULL`**

Verify all three gate actions appear if dictated by the frozen scores; if a branch has zero naturally occurring samples, report zero rather than changing thresholds.

- [ ] **Step 5: Synchronize all successes and failures locally**

Use rsync without deleting existing local artifacts. Compare remote/local counts and SHA-256 for every JSON, JSONL and CSV.

### Task 12: Blinded review and statistical scoring

**Files:**
- Create: `scripts/build_vega_blind_review.py`
- Create: `scripts/score_vega_comparison.py`
- Create: corresponding unit tests

- [ ] **Step 1: Build a deterministic blinded sheet**

Input every real version answer, replace version/round with opaque `response_id`, shuffle with seed `20260716`, and retain the reversible mapping in a separate JSON not shown during labeling.

- [ ] **Step 2: Complete first-pass and targeted second-pass labels**

Review all rows. Second-pass scope is all partial labels, all refusals and every query whose versions disagree. Preserve both passes and final adjudication.

- [ ] **Step 3: Implement and test paired statistics**

Use 10,000 paired bootstrap samples with seed `20260716`. Implement exact McNemar from discordant counts `b,c` using `math.comb`; report `p=1.0` when `b+c=0`.

- [ ] **Step 4: Generate comparison artifacts**

Required outputs:

```text
metrics_primary_20.csv
metrics_all_30.csv
paired_deltas_primary_20.csv
bootstrap_ci.csv
mcnemar.csv
transition_matrix.csv
error_causes.csv
system_metrics.csv
review_final.csv
```

### Task 13: Generate the real-results PDF

**Files:**
- Create: `tools/build_vega_report_assets.py`
- Create: `output/pdf/VEGA-RAG-真实多轮性能对比实验报告.tex`
- Generate: `output/pdf/VEGA-RAG-真实多轮性能对比实验报告.pdf`

- [ ] **Step 1: Validate every source before asset generation**

The asset script exits nonzero unless every accepted run has exit 0, 30 trace rows, 30 result rows, matching query sets and a valid threshold SHA-256. It must include failed-run evidence separately without mixing it into performance metrics.

- [ ] **Step 2: Generate only data-driven tables/charts**

Build grouped charts for Accuracy/Missing/Hallucination/Truthfulness, primary-20 paired deltas, error causes, gate actions, latency P50/P95, wall time and GPU peak. Every chart data file is CSV under `output/pdf/vega_assets/`.

- [ ] **Step 3: Compile the Chinese LaTeX PDF**

Run:

```bash
/Library/TeX/texbin/latexmk -xelatex -interaction=nonstopmode -halt-on-error VEGA-RAG-真实多轮性能对比实验报告.tex
```

Expected: exit 0 and no missing-character, overfull-box or undefined-reference warnings.

- [ ] **Step 4: Render and inspect every page**

Run:

```bash
pdftoppm -png -r 150 output/pdf/VEGA-RAG-真实多轮性能对比实验报告.pdf tmp/pdfs/vega-report/page
```

Inspect every page at original size for glyphs, chart labels, clipping, tables, headers, footers and page numbers. Fix and repeat until no visual defects remain, then remove temporary PNGs.

### Task 14: Final requirement audit

**Files:**
- Design spec, implementation plan, code, artifacts, report

- [ ] **Step 1: Run fresh full tests and `git diff --check`**

- [ ] **Step 2: Recompute all metrics independently from `review_final.csv`**

Assert `N=C+M+H`, `Accuracy=C/N`, `Missing=M/N`, `Hallucination=H/N`, and `Truthfulness=Accuracy-Hallucination` for every version/scope.

- [ ] **Step 3: Scan PDF text and repository outputs for credentials**

Expected: zero private key, HF/OpenAI/AWS token, Bearer credential or `sshpass` matches.

- [ ] **Step 4: Report exactly what happened**

Return links to PDF, TeX, raw artifacts, comparison CSVs, review sheet and logs. State official-vs-non-official boundaries, failed runs, negative results, unexecuted branches and the fact that no commit/push occurred.

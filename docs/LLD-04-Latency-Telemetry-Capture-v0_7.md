# LLD-04 · Latency Telemetry Capture

> Codex-Bench · Low-Level Design  
> Derived from HLD Spec v2.3 · April 2026  
> Sprint: Design S1 → Implement S1 (with LLD-03)  
> Status: SIGNED OFF v0.7

---

## Changelog

| Version | Change |
|---|---|
| v0.7 | **Post-sign-off cleanups (non-blocking).** Cleanup 1: `snapshot_after()` docstring now correctly attributes the orphaned-before case to SnapshotManager state loss across process boundaries, not `snapshot_before()` network failure — consistent with §6.1 and §12.3. Cleanup 2: `harness='codex'` scoping added to §16 validation checklist, §17 LLD-12 connection entry, and §19 open questions, matching the authoritative requirement in §10.2's `aggregate_by_model()` docstring. |
| v0.6 | One P0. Recovery path corrected to out-of-band operator DB surgery — `invalidate_stale_runs()` does not apply. |
| v0.5 | One P0. Telemetry-gap recovery downgraded from automated re-dispatch to operator workflow. |
| v0.4 | One P0, two P1 cleanups. `reportable_runs` join key corrected to `(task_id, model_id, seed, attempt)`. `harness` scoped to `codex` at LLD-12 query time. Stale `seed`-only wording and `error_path` docstring fixed. |
| v0.3 | Two P0 fixes, one P1, two cleanups. Resolved-only aggregation via `reportable_runs` inner join. `attempt` added to run identity. Pool-filtering mismatch fixed. `error_path` anomaly removed from defaults. `orphaned_before` causal explanation corrected. |
| v0.2 | Three P0 fixes, three P1 fixes. Interface aligned with LLD-03 call sites. Default filters fixed. `prompt_tokens` fallback removed. Turn extraction deferred to post-hoc. `orphaned_before` emitted. Completeness validation added. |
| v0.1 | Initial draft. |

---

## 1. Purpose & Scope

This document specifies the Latency Telemetry Capture component — the subsystem responsible for extracting the four Contribution A latency metrics (TTFT, prefill throughput, decode throughput, prefix cache hit rate) from vLLM's `/metrics` endpoint, attributing them to individual tasks, and storing them for downstream consumption by LLD-12 (Results & Artifact Generator).

**Responsibilities:**

- Implement the `LatencyCapture` interface that LLD-03 calls before and after each task
- Parse Prometheus exposition format text from vLLM's `GET /metrics` endpoint
- Resolve metric name schema variants at server startup (LLD-01 §9.2 resolver pattern)
- Compute per-task metric deltas using the `_sum` delta procedure defined in LLD-01 §9.2
- Derive the four Contribution A metrics per task: TTFT (ms), prefill throughput (tok/s), decode throughput (tok/s), cache hit rate (%)
- Store raw deltas and derived metrics in a per-campaign JSONL file for LLD-12 consumption
- Provide a post-hoc turn extraction utility that LLD-12 calls with trajectory paths from LLD-02 run records
- Detect and flag anomalous telemetry (orphaned snapshots, zero-delta tasks, negative deltas)
- Validate telemetry completeness against LLD-02 run records before aggregation (fail-closed)
- Provide an aggregation API that LLD-12 calls to produce Contribution A latency anatomy tables

**Out of scope:** Model serving and metric exposure (LLD-01), task execution and snapshot timing coordination (LLD-03), campaign orchestration (LLD-07), trajectory parsing for SFT training (LLD-06), result packaging and bootstrap CI computation (LLD-12), SWE-Agent telemetry (LLD-09 — mini-SWE-Agent does not route through Codex and does not produce the same `/metrics` footprint).

**Track-agnostic design:** The capture mechanism is identical for SWE-bench and Codex-Long runs. Both task types produce inference requests through Codex CLI → vLLM, and vLLM's `/metrics` counters accumulate the same way regardless of task type. LLD-04 does not need to distinguish between tracks for metric computation. The `track` field is set at campaign construction and passed through to storage for LLD-12 filtering.

---

## 2. Architecture Overview

```
                        LLD-01 · vLLM
                        ┌─────────────────┐
                        │  GET /metrics    │
                        │  (Prometheus     │
                        │   exposition)    │
                        └────────┬────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
              │    LLD-04 · LATENCY TELEMETRY        │
              │                                      │
              │  ┌────────────┐  ┌────────────────┐  │
              │  │ Schema     │  │ Prometheus     │  │
              │  │ Resolver   │  │ Parser         │  │
              │  │ (§4)       │  │ (§5)           │  │
              │  └────────────┘  └────────────────┘  │
              │                                      │
              │  ┌────────────┐  ┌────────────────┐  │
              │  │ Snapshot   │  │ Delta          │  │
              │  │ Manager    │  │ Computer       │  │
              │  │ (§6)       │  │ (§7)           │  │
              │  └────────────┘  └────────────────┘  │
              │                                      │
              │  ┌────────────┐  ┌────────────────┐  │
              │  │ JSONL      │  │ Completeness   │  │
              │  │ Writer     │  │ Validator      │  │
              │  │ (§9)       │  │ (§10.3)        │  │
              │  └────────────┘  └────────────────┘  │
              │                                      │
              │  ┌────────────────────────────────┐  │
              │  │ Aggregation API (§10)           │  │
              │  │ → LLD-12 Contribution A tables  │  │
              │  └────────────────────────────────┘  │
              └──────────────────────────────────────┘
                       │              │           │
                       ▼              ▼           ▼
                  LLD-03          LLD-12       LLD-02
                  snapshot_       Latency      run records
                  before/after   anatomy      (completeness
                                 tables       check + turn
                                              trajectory paths)
```

---

## 3. Interface Contract — `LatencyCapture`

This is the interface that LLD-03 consumes. LLD-03's `execute_task()` calls `snapshot_before()` after `claim_run()` succeeds and `snapshot_after()` in the `finally` block (see LLD-03 §10.1).

**Critical alignment note:** The method signatures below match the signed-off LLD-03 v1.1 call sites exactly, with one same-sprint amendment (§14.1): `seed` and `attempt` are added to `snapshot_before()`. No other changes to LLD-03 are required.

```python
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class TaskMetrics:
    """Per-task derived metrics — the four Contribution A values plus raw deltas."""
    task_id: str
    model_id: str                     # campaign-level, from construction
    track: str                        # campaign-level, derived from pool_or_split
    pool_or_split: str                # campaign-level, from construction
    seed: int                         # per-task, from snapshot_before (§14.1 amendment)
    attempt: int                      # per-task, from snapshot_before (§14.1 amendment)

    # Derived Contribution A metrics (None if insufficient data)
    ttft_ms: Optional[float]          # Mean TTFT across turns
    prefill_throughput_tps: Optional[float]  # Newly computed tokens / prefill GPU time
    decode_throughput_tps: Optional[float]   # Generated tokens / decode GPU time
    cache_hit_rate_pct: Optional[float]      # Cache hits / cache queries × 100

    # Raw deltas for LLD-12 drill-down
    prompt_tokens: float              # Total prompt tokens incl. cache hits (context size tracking)
    kv_computed_tokens: float         # Newly computed tokens (prefill throughput numerator)
    gen_tokens: float                 # Generated tokens (decode throughput numerator)
    prefill_sum_s: float              # Total prefill GPU time across all turns
    decode_sum_s: float               # Total decode GPU time across all turns
    ttft_sum_s: float                 # Sum of per-turn TTFT
    ttft_count: int                   # Number of turns (TTFT samples)
    cache_queries: float              # Prefix cache queries
    cache_hits: float                 # Prefix cache hits

    # Wall-clock context
    snapshot_before_ts: float         # time.monotonic() at pre-task snapshot
    snapshot_after_ts: float          # time.monotonic() at post-task snapshot
    wall_clock_s: float               # snapshot_after_ts - snapshot_before_ts

    # Quality flags
    anomalies: list[str] = field(default_factory=list)
    #   Possible values:
    #   - "zero_gen_tokens": no decode tokens observed (task produced no output)
    #   - "zero_prefill_tokens": no prefill tokens observed (task issued no requests)
    #   - "negative_delta:<metric>": counter went backwards (server restart mid-task)
    #   - "orphaned_before": snapshot_after called without matching snapshot_before


class LatencyCapture:
    """
    Captures per-task latency telemetry from vLLM /metrics.

    Lifecycle:
      1. Construct once per campaign (model × pool), passing vLLM host/port
         and output path. track is derived from pool_or_split.
      2. resolve_schema() — called once after vLLM is confirmed healthy.
      3. Per-task cycle: snapshot_before(task_id, seed, attempt) → [task runs] → snapshot_after(task_id)
      4. After campaign: LLD-12 calls load_telemetry() + aggregate_by_model().

    Thread safety: NOT thread-safe. LLD-03 executes tasks sequentially on a
    single async event loop. If LLD-07 ever parallelizes tasks on the same
    vLLM instance (which the HLD does not plan), this design would need a
    per-task snapshot isolation mechanism.
    """

    def __init__(
        self,
        vllm_host: str,
        vllm_port: int,
        output_dir: str,
        model_id: str,
        pool_or_split: str,
    ):
        ...

    async def resolve_schema(self) -> None:
        """
        Probe /metrics once at startup. Resolve counter name variants
        (LLD-01 §9.2 resolver pattern). Must be called before any
        snapshot_before/after cycle. Raises RuntimeError if any required
        metric is missing — including request_prefill_kv_computed_tokens.
        No fallback to prompt_tokens.
        """
        ...

    async def snapshot_before(self, task_id: str, seed: int, attempt: int) -> None:
        """
        Capture the pre-task /metrics snapshot.

        Called by LLD-03 execute_task() after claim_run() succeeds
        (LLD-03 §4.3, §10.1 step 4). Must complete before container
        setup begins.

        task_id: scenario_id from TaskSpec (instance_id for SWE-bench,
                 "family_id/variant_id" for Codex-Long).
        seed:    integer seed from TaskSpec (§14.1 same-sprint amendment).
        attempt: attempt number from TaskSpec (§14.1 same-sprint amendment).
                 1 for fresh runs, 2 for retries. Required for aligning
                 telemetry records with LLD-02's latest_runs view.

        Stores the snapshot keyed by task_id. Raises if a snapshot_before
        for a different task_id is already pending (indicates a bug in
        LLD-03 — snapshot_after was never called for the previous task).
        """
        ...

    async def snapshot_after(self, task_id: str) -> TaskMetrics:
        """
        Capture the post-task /metrics snapshot, compute deltas, write
        to JSONL.

        Called by LLD-03 execute_task() in the finally block
        (LLD-03 §10.1). This is the ONLY call site — there is no
        separate happy-path call. The finally block runs on both
        success and error paths.

        If no pending snapshot_before exists for this task_id (e.g.,
        SnapshotManager state lost across a process restart), a synthetic
        empty before-snapshot is used and the "orphaned_before" anomaly
        flag is set. Note: LLD-03 only calls snapshot_after() when
        telemetry_started=True, so this is not triggered by a
        snapshot_before() network failure within a single process lifetime.

        LLD-04 does not distinguish success from error at this interface.
        LLD-12 filters on LLD-02 run outcome when computing Contribution A
        aggregation — only outcome="resolved" tasks are included.

        Returns the computed TaskMetrics.
        """
        ...
```

**Timing contract (from LLD-01 §9.2 and LLD-03 §4.3):**

```
1. LLD-03   claim_run() + verification succeed
2. LLD-04 → snapshot_before(task_id, seed, attempt) ← GET /metrics
3. LLD-03   container setup + codex exec      ← task runs
4. LLD-03   stdout EOF / timeout              ← task complete
5. LLD-04 → snapshot_after(task_id)           ← GET /metrics + compute deltas (finally block)
6. LLD-03   POST /reset_prefix_cache          ← AFTER step 5
```

The cache flush in step 6 must come after the snapshot in step 5. The flush clears KV blocks but issues no inference requests and does not move any `/metrics` counters. This ordering is enforced by LLD-03's `execute_task()` control flow — LLD-04 does not call the flush.

---

## 4. Schema Resolution

vLLM counter names vary across releases (see LLD-01 §9.2 for the full explanation). LLD-04 must resolve the canonical form at startup and fail hard if any required metric is absent.

### 4.1 Metric Registry

All metrics consumed by LLD-04, with their expected types and usage:

| Logical name | Prometheus candidates | Type | Used for |
|---|---|---|---|
| `prompt_tokens` | `vllm:prompt_tokens_total`, `vllm:prompt_tokens` | Counter | Context size tracking (not throughput numerator) |
| `generation_tokens` | `vllm:generation_tokens_total`, `vllm:generation_tokens` | Counter | Decode token count — throughput numerator |
| `kv_computed_tokens` | `vllm:request_prefill_kv_computed_tokens` | Histogram | Newly computed prefill tokens — throughput numerator |
| `cache_queries` | `vllm:prefix_cache_queries` | Counter | Cache hit rate denominator |
| `cache_hits` | `vllm:prefix_cache_hits` | Counter | Cache hit rate numerator |
| `ttft` | `vllm:time_to_first_token_seconds` | Histogram | TTFT per turn |
| `prefill_time` | `vllm:request_prefill_time_seconds` | Histogram | GPU prefill time — throughput denominator |
| `decode_time` | `vllm:request_decode_time_seconds` | Histogram | GPU decode time — throughput denominator |
| `itl` | `vllm:inter_token_latency_seconds` | Histogram | ITL / TPOT (stored but not a primary Contribution A metric) |

Histogram metrics produce three sub-keys in Prometheus exposition format: `_sum`, `_count`, and `_bucket`. LLD-04 reads `_sum` and `_count` only. Buckets are ignored.

**All metrics in this table are required.** If any metric is absent from the live `/metrics` output after vLLM version pin, `resolve_metric_schema()` raises `RuntimeError` and the campaign cannot start. There is no fallback or soft degradation. This matches the LLD-01 §9.2 contract: "Raises RuntimeError if neither variant is present — never silently defaults to 0."

### 4.2 Resolver Implementation

```python
# ── Variant table (extend as new vLLM releases are tested) ──────────
_METRIC_VARIANTS: dict[str, list[str]] = {
    "prompt_tokens":      ["vllm:prompt_tokens_total",     "vllm:prompt_tokens"],
    "generation_tokens":  ["vllm:generation_tokens_total", "vllm:generation_tokens"],
    # Histograms have no known naming variants yet. If future vLLM releases
    # rename them, add variants here before updating the pinned version.
    "kv_computed_tokens":  ["vllm:request_prefill_kv_computed_tokens"],
    "cache_queries":       ["vllm:prefix_cache_queries"],
    "cache_hits":          ["vllm:prefix_cache_hits"],
    "ttft":                ["vllm:time_to_first_token_seconds"],
    "prefill_time":        ["vllm:request_prefill_time_seconds"],
    "decode_time":         ["vllm:request_decode_time_seconds"],
    "itl":                 ["vllm:inter_token_latency_seconds"],
}

# Histograms — LLD-04 reads _sum and _count sub-keys
_HISTOGRAM_METRICS = {
    "kv_computed_tokens", "ttft", "prefill_time", "decode_time", "itl",
}


def resolve_metric_schema(metrics_snapshot: dict[str, float]) -> dict[str, str]:
    """
    Probe the live /metrics output and return the canonical Prometheus name
    for each logical metric.

    For counters: looks for the counter name directly.
    For histograms: looks for the _sum sub-key (if _sum is present,
    _count is also present).

    Raises RuntimeError if ANY required metric is absent. This is a
    startup-time hard failure — never silently default to 0. No fallback
    to alternative metrics (e.g., prompt_tokens for kv_computed_tokens).

    Called once after vLLM is confirmed healthy (GET /health returns 200).
    The resolved schema is cached for the lifetime of the campaign.
    """
    resolved = {}
    for logical_name, candidates in _METRIC_VARIANTS.items():
        for candidate in candidates:
            # For histograms, check for the _sum sub-key
            probe_key = f"{candidate}_sum" if logical_name in _HISTOGRAM_METRICS else candidate
            if probe_key in metrics_snapshot:
                resolved[logical_name] = candidate
                break
        else:
            raise RuntimeError(
                f"vLLM /metrics does not expose any of {candidates} "
                f"(probed keys: {[c + ('_sum' if logical_name in _HISTOGRAM_METRICS else '') for c in candidates]}). "
                f"This is a hard startup failure per LLD-01 §9.2. "
                f"Update _METRIC_VARIANTS for the pinned vLLM version, "
                f"or upgrade vLLM to a version that exposes the required metric."
            )
    return resolved
```

### 4.3 Sprint 0 Validation Item

After pinning the vLLM version and confirming it boots on the DGX Spark:

1. Run `curl -s http://127.0.0.1:$VLLM_PORT/metrics | grep -E "prompt_tokens|generation_tokens|prefill_kv|prefix_cache|time_to_first|prefill_time|decode_time|inter_token"` and record which name forms appear.
2. Verify that `resolve_metric_schema()` succeeds against the live output.
3. Log the resolved schema alongside the vLLM git hash in the launch log header.
4. If `request_prefill_kv_computed_tokens` is absent, this is a **Sprint 0 blocker**. The pinned vLLM version must expose this metric. If it does not, either upgrade vLLM or file an upstream issue. There is no fallback — the prefill throughput formula requires it.

---

## 5. Prometheus Text Parser

vLLM's `/metrics` endpoint returns Prometheus exposition format (text/plain). LLD-04 needs a parser that extracts counter values and histogram `_sum` / `_count` values into a flat `dict[str, float]`.

### 5.1 Parser Requirements

- Parse `# TYPE` and `# HELP` lines (skip them).
- Parse sample lines: `metric_name{labels} value timestamp?`.
- Strip labels — LLD-04 operates in single-model, single-request-stream mode. Label filtering is unnecessary because `--max-num-seqs 4` ensures the only concurrent requests are from the single Codex session. If label filtering is ever needed (multi-model serving), it can be added by matching `model_name` labels.
- Handle histogram sub-keys: `_sum`, `_count`, `_bucket`. Store `_sum` and `_count`; ignore `_bucket` lines.
- Return `dict[str, float]` mapping full metric names (including suffixes) to values.

### 5.2 Implementation

```python
import re
from typing import Optional

_SAMPLE_RE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)'   # metric name (with : for vllm: prefix)
    r'(?:\{[^}]*\})?'                  # optional labels — stripped
    r'\s+'
    r'([0-9eE.+\-]+|NaN|Inf|\+Inf|-Inf)'  # value
    r'(?:\s+\d+)?$'                    # optional timestamp — stripped
)

# Ignore histogram bucket lines — only _sum and _count are needed
_SKIP_SUFFIXES = ("_bucket",)


def parse_prometheus_text(raw: str) -> dict[str, float]:
    """
    Parse Prometheus exposition format into a flat {name: value} dict.

    Only counter and histogram _sum/_count values are retained.
    Bucket lines are skipped. Labels are stripped (single-stream assumption).

    If a metric name appears multiple times (e.g., with different label sets),
    the values are summed. This handles vLLM metrics that split by model_name
    label when multiple models are served — but in Codex-Bench's single-model
    serving mode, there is at most one label set per metric.
    """
    result: dict[str, float] = {}

    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if any(line.split("{")[0].split()[0].endswith(s) for s in _SKIP_SUFFIXES):
            continue

        m = _SAMPLE_RE.match(line)
        if m is None:
            continue

        name = m.group(1)
        try:
            value = float(m.group(2))
        except ValueError:
            continue

        # Sum across label sets (no-op in single-model mode)
        result[name] = result.get(name, 0.0) + value

    return result


async def fetch_metrics(host: str, port: int, timeout: float = 5.0) -> dict[str, float]:
    """
    GET /metrics from vLLM and return parsed result.

    Uses aiohttp for async compatibility with LLD-03's event loop.
    Timeout is aggressive (5s) — /metrics should respond in <100ms
    under normal conditions. A timeout here suggests vLLM is under
    severe load or crashed.
    """
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"http://{host}:{port}/metrics",
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            resp.raise_for_status()
            raw = await resp.text()
    return parse_prometheus_text(raw)
```

### 5.3 Parser Validation

Sprint 1 implementation must include a test that:

1. Captures a real `/metrics` response from the pinned vLLM version.
2. Verifies all resolved metric names are present in the parsed output.
3. Verifies that `_sum` and `_count` values are non-negative.
4. Verifies that histogram `_bucket` lines are not present in the output.

This test is run once during Sprint 1 setup and the captured response is committed to the repo as a test fixture.

---

## 6. Snapshot Manager

The snapshot manager holds pending `before` snapshots in memory and pairs them with `after` snapshots when the task completes.

### 6.1 State Machine

Each task_id progresses through exactly two states:

```
  (no entry)
      │
      ▼  snapshot_before(task_id, seed, attempt)
  PENDING  ─── stores before_snapshot + seed + attempt
      │
      ▼  snapshot_after(task_id)
  COMPLETED ─── computes deltas, writes JSONL, removes from pending
```

**Invariants:**

- At most one `PENDING` snapshot exists at any time (sequential task execution per LLD-03).
- `snapshot_before()` raises `RuntimeError` if a `PENDING` entry already exists for a different task_id (indicates LLD-03 failed to call `snapshot_after` for the previous task).
- `snapshot_after()` for a task_id with no `PENDING` entry emits the `"orphaned_before"` anomaly, captures a snapshot anyway, and uses a synthetic empty before-snapshot. This can happen if the `SnapshotManager` instance is replaced between `snapshot_before()` and `snapshot_after()` (e.g., process restart, LLD-07 recreating the `LatencyCapture` instance mid-campaign). Note: LLD-03 only calls `snapshot_after()` when `telemetry_started=True`, which requires `snapshot_before()` to have succeeded — so this is not triggered by a `snapshot_before()` network failure within a single process lifetime.

### 6.2 Implementation

```python
@dataclass
class PendingSnapshot:
    task_id: str
    seed: int
    attempt: int
    snapshot: dict[str, float]
    timestamp: float                  # time.monotonic()


class SnapshotManager:
    """
    Manages the before/after snapshot lifecycle for one vLLM instance.

    Campaign-level metadata (model_id, track, pool_or_split) is set at
    construction and attached to every record. Per-task metadata (task_id,
    seed) comes from snapshot_before().
    """

    def __init__(
        self,
        vllm_host: str,
        vllm_port: int,
        schema: dict[str, str],       # resolved metric names
        model_id: str,
        track: str,
        pool_or_split: str,
    ):
        self._host = vllm_host
        self._port = vllm_port
        self._schema = schema
        self._model_id = model_id
        self._track = track
        self._pool_or_split = pool_or_split
        self._pending: Optional[PendingSnapshot] = None

    async def capture_before(self, task_id: str, seed: int, attempt: int) -> None:
        if self._pending is not None and self._pending.task_id != task_id:
            raise RuntimeError(
                f"Pending snapshot exists for {self._pending.task_id} "
                f"but snapshot_before called for {task_id}. "
                f"LLD-03 must call snapshot_after() before starting a new task."
            )
        snapshot = await fetch_metrics(self._host, self._port)
        self._pending = PendingSnapshot(
            task_id=task_id,
            seed=seed,
            attempt=attempt,
            snapshot=snapshot,
            timestamp=time.monotonic(),
        )

    async def capture_after(
        self, task_id: str
    ) -> tuple[PendingSnapshot, dict[str, float], float, list[str]]:
        """
        Returns (pending_snapshot, after_snapshot, after_timestamp, anomalies).

        If no pending snapshot exists for task_id, returns a synthetic
        empty before-snapshot and includes "orphaned_before" in anomalies.
        """
        after_snapshot = await fetch_metrics(self._host, self._port)
        after_ts = time.monotonic()
        anomalies: list[str] = []

        if self._pending is None or self._pending.task_id != task_id:
            # ── Orphaned: no matching before-snapshot ──
            # LLD-03 only calls snapshot_after() when telemetry_started=True
            # (i.e., snapshot_before() succeeded). This path is triggered if
            # the SnapshotManager instance was replaced between calls (e.g.,
            # process restart). Emit the anomaly flag for identification.
            anomalies.append("orphaned_before")
            synthetic_before = PendingSnapshot(
                task_id=task_id,
                seed=0,
                attempt=0,
                snapshot={k: 0.0 for k in after_snapshot},
                timestamp=after_ts,  # wall_clock_s = 0
            )
            return synthetic_before, after_snapshot, after_ts, anomalies

        pending = self._pending
        self._pending = None
        return pending, after_snapshot, after_ts, anomalies
```

---

## 7. Delta Computation

This section implements the per-task metric formulas defined in LLD-01 §9.2. The formulas are reproduced here for LLD-04 implementor reference; LLD-01 §9.2 is the authoritative source.

### 7.1 Core Delta Function

```python
def _delta(
    before: dict[str, float],
    after: dict[str, float],
    key: str,
) -> tuple[float, Optional[str]]:
    """
    Compute after[key] - before[key].

    Returns (delta_value, anomaly_or_none).
    - If key is missing from after: raises RuntimeError (schema resolver
      should have caught this at startup).
    - If delta is negative: returns the delta but flags "negative_delta:<key>".
      Negative deltas indicate a vLLM restart occurred between snapshots.
    """
    if key not in after:
        raise RuntimeError(
            f"Expected metric '{key}' not found in /metrics snapshot. "
            f"Did the vLLM server restart with a different version?"
        )
    before_val = before.get(key, 0.0)
    delta = after[key] - before_val
    anomaly = f"negative_delta:{key}" if delta < 0 else None
    return delta, anomaly
```

### 7.2 Task Metric Computation

```python
def compute_task_metrics(
    pending: PendingSnapshot,
    after: dict[str, float],
    after_ts: float,
    schema: dict[str, str],
    model_id: str,
    track: str,
    pool_or_split: str,
    initial_anomalies: Optional[list[str]] = None,
) -> TaskMetrics:
    """
    Compute per-task Contribution A metrics from before/after snapshots.

    All throughput denominators use histogram _sum deltas (total GPU time
    across all N turns of the task), NOT per-request mean times.
    See LLD-01 §9.2 for the rationale.

    schema: resolved metric names from resolve_metric_schema().
    initial_anomalies: anomalies from the snapshot manager (e.g., "orphaned_before").
    """
    before = pending.snapshot
    anomalies: list[str] = list(initial_anomalies or [])

    def safe_delta(key: str) -> float:
        val, anomaly = _delta(before, after, key)
        if anomaly:
            anomalies.append(anomaly)
        return val

    # ── Token counts ──
    prompt_tokens      = safe_delta(schema["prompt_tokens"])
    kv_computed_tokens = safe_delta(f"{schema['kv_computed_tokens']}_sum")
    gen_tokens         = safe_delta(schema["generation_tokens"])
    cache_queries      = safe_delta(schema["cache_queries"])
    cache_hits         = safe_delta(schema["cache_hits"])

    # ── Histogram _sum deltas (total GPU time across all task turns) ──
    ttft_sum   = safe_delta(f"{schema['ttft']}_sum")
    ttft_count = safe_delta(f"{schema['ttft']}_count")
    prefill_sum_s = safe_delta(f"{schema['prefill_time']}_sum")
    decode_sum_s  = safe_delta(f"{schema['decode_time']}_sum")

    # ── Quality flags ──
    if gen_tokens == 0 and "orphaned_before" not in anomalies:
        anomalies.append("zero_gen_tokens")
    if kv_computed_tokens == 0 and prompt_tokens == 0 and "orphaned_before" not in anomalies:
        anomalies.append("zero_prefill_tokens")

    # ── Derived Contribution A metrics ──
    # TTFT: mean per turn (one TTFT per Codex API call = one per turn)
    ttft_ms = (ttft_sum / ttft_count * 1000) if ttft_count > 0 else None

    # Prefill throughput: newly computed tokens / total prefill GPU time
    # Uses kv_computed_tokens (excludes cache hits) per LLD-01 §9.2.
    # No fallback to prompt_tokens — kv_computed_tokens is a required metric.
    prefill_tps = (
        kv_computed_tokens / prefill_sum_s
        if prefill_sum_s > 0 else None
    )

    # Decode throughput: generated tokens / total decode GPU time
    decode_tps = gen_tokens / decode_sum_s if decode_sum_s > 0 else None

    # Cache hit rate: hits / queries × 100
    cache_pct = (
        cache_hits / cache_queries * 100
        if cache_queries > 0 else None
    )

    wall_clock_s = after_ts - pending.timestamp

    return TaskMetrics(
        task_id=pending.task_id,
        model_id=model_id,
        track=track,
        pool_or_split=pool_or_split,
        seed=pending.seed,
        attempt=pending.attempt,
        ttft_ms=ttft_ms,
        prefill_throughput_tps=prefill_tps,
        decode_throughput_tps=decode_tps,
        cache_hit_rate_pct=cache_pct,
        prompt_tokens=prompt_tokens,
        kv_computed_tokens=kv_computed_tokens,
        gen_tokens=gen_tokens,
        prefill_sum_s=prefill_sum_s,
        decode_sum_s=decode_sum_s,
        ttft_sum_s=ttft_sum,
        ttft_count=int(ttft_count),
        cache_queries=cache_queries,
        cache_hits=cache_hits,
        snapshot_before_ts=pending.timestamp,
        snapshot_after_ts=after_ts,
        wall_clock_s=wall_clock_s,
        anomalies=anomalies,
    )
```

### 7.3 Correctness Invariants

These invariants must hold for every non-anomalous task and should be checked in Sprint 1 validation:

1. **`ttft_count` equals the number of Codex API calls (turns).** Each turn produces one request; each request contributes one TTFT sample to the histogram. Cross-check against the trajectory JSONL turn count.
2. **`gen_tokens` is approximately equal to the total output tokens across all turns** (as counted from the trajectory JSONL). Discrepancies > 5% should be investigated.
3. **`kv_computed_tokens ≤ prompt_tokens`** always — computed tokens are a subset of total prompt tokens (the rest are cache hits).
4. **`cache_hits ≤ cache_queries`** always.
5. **`prefill_sum_s + decode_sum_s < wall_clock_s`** — GPU compute time is a fraction of wall-clock time (the remainder is scheduling, network, I/O). If GPU time exceeds wall-clock, the metrics are suspect.

---

## 8. Per-Turn Attribution

Per-task metrics (§7) are the primary Contribution A deliverable. Per-turn attribution provides additional diagnostic granularity — it shows how latency evolves over the course of a multi-turn Codex session (e.g., TTFT increasing as context grows, cache hit rate stabilizing after turn 3).

### 8.1 Design Decision: Post-Hoc, Not Intermediate Snapshots

Per-turn intermediate `/metrics` snapshots during the task would require either (a) LLD-04 monitoring the trajectory JSONL in real time, or (b) LLD-03 emitting turn-boundary callbacks to LLD-04. Both add complexity and risk to the critical path. Instead, LLD-04 provides a **post-hoc** turn extraction utility.

The per-task `_sum`/`_count` deltas give total GPU time and turn count. Per-turn TTFT is reported as `ttft_sum_delta / ttft_count_delta` (mean across turns — this is the Contribution A definition). If per-turn distributions are needed for a future extension, intermediate snapshotting can be added as a Sprint 3 enhancement.

### 8.2 Turn Extraction — Separate Utility, Not Part of `snapshot_after()`

Turn extraction is **not** performed inside `snapshot_after()`. It is a separate post-hoc function called by LLD-12 at aggregation time, when LLD-12 has the trajectory path from LLD-02 run records. This separation avoids the trajectory path discovery problem: LLD-03 constructs trajectory filenames using `scenario_id.replace('/', '_')` (LLD-03 §9.2) and records the full path in LLD-02 via `finish_run(trajectory_path=...)`. LLD-04 should not re-derive this path — it should receive it from LLD-12, which gets it from LLD-02.

```python
import json
from dataclasses import dataclass


@dataclass
class TurnInfo:
    """Extracted turn-level data from trajectory JSONL."""
    turn_index: int
    start_timestamp: Optional[str]    # ISO format from Codex event
    end_timestamp: Optional[str]
    output_tokens_approx: int         # character count / 4 as rough estimate
    tool_calls: int                   # number of tool calls in this turn


def extract_turns(trajectory_path: str) -> list[TurnInfo]:
    """
    Parse trajectory JSONL to extract turn boundaries.

    trajectory_path: full path to the trajectory JSONL file, as stored
    in LLD-02 run records (field: trajectory_path). LLD-12 passes this
    directly — LLD-04 does not discover or construct the path.

    The exact event types depend on the Codex CLI --json output format.
    This parser looks for response-level delimiters (message events with
    role="assistant") and tool-call/tool-result pairs.

    This is a best-effort parser — the Codex event stream format is not
    formally specified in the HLD. Sprint 1 must validate the actual event
    types against a captured trajectory from Gate 1 smoke tests.
    """
    turns: list[TurnInfo] = []
    current_turn_index = 0
    current_output_chars = 0
    current_tool_calls = 0
    current_start = None

    with open(trajectory_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")

            # Turn start: a new assistant message or response
            if event_type in ("message", "response.created", "response_created"):
                if current_start is not None and current_output_chars > 0:
                    # Close previous turn
                    turns.append(TurnInfo(
                        turn_index=current_turn_index,
                        start_timestamp=current_start,
                        end_timestamp=event.get("timestamp"),
                        output_tokens_approx=current_output_chars // 4,
                        tool_calls=current_tool_calls,
                    ))
                    current_turn_index += 1
                current_start = event.get("timestamp")
                current_output_chars = 0
                current_tool_calls = 0

            # Accumulate output
            elif event_type in ("message_delta", "response.output_text.delta",
                                "content_block_delta"):
                text = event.get("delta", {}).get("text", "")
                current_output_chars += len(text)

            # Count tool calls
            elif event_type in ("tool_call", "response.function_call_arguments.done",
                                "function_call"):
                current_tool_calls += 1

    # Close final turn
    if current_start is not None:
        turns.append(TurnInfo(
            turn_index=current_turn_index,
            start_timestamp=current_start,
            end_timestamp=None,
            output_tokens_approx=current_output_chars // 4,
            tool_calls=current_tool_calls,
        ))

    return turns
```

**Sprint 1 action:** After running the first 10 SWE-bench + 10 Codex-Long tasks through the pipeline (LLD-03 §15 end-to-end validation), capture the actual event types and update `extract_turns()` to match. The event type names above are provisional.

### 8.3 Cross-Validation

LLD-12 can cross-validate telemetry and turn data after calling both `load_telemetry()` and `extract_turns()`:

- `task_metrics.ttft_count == len(turns)` — if these differ by more than 1, the turn parser or the histogram delta is wrong.
- `sum(t.output_tokens_approx for t in turns)` ≈ `task_metrics.gen_tokens` — rough cross-check (character-based approximation will be noisy).

Log warnings for discrepancies > 20%. Do not fail — the character-based approximation is inherently imprecise.

---

## 9. Storage Format

### 9.1 Per-Campaign JSONL Output

Each campaign (a model × pool combination managed by LLD-07) produces a single JSONL file containing one line per completed task.

**File path convention:**

```
output/
  telemetry/
    latency_qwen3.5-27b_dev-bench.jsonl
    latency_qwen3.5-27b_train-long.jsonl
    latency_qwen3.5-35b-a3b_dev-bench.jsonl
    ...
```

The file is named `latency_{model_id}_{pool_or_split}.jsonl`. LLD-07 creates a new `LatencyCapture` instance for each model × pool campaign, so each file is written by a single instance.

### 9.2 Record Schema

Each JSONL line is a JSON object. Turn data is **not** included — turn extraction is a separate post-hoc step (§8.2).

```json
{
  "task_id": "django__django-11099",
  "model_id": "qwen3.5-27b",
  "track": "swe_bench",
  "pool_or_split": "dev_bench",
  "seed": 1,
  "attempt": 1,

  "ttft_ms": 1245.3,
  "prefill_throughput_tps": 4523.1,
  "decode_throughput_tps": 10.2,
  "cache_hit_rate_pct": 72.4,

  "prompt_tokens": 185000.0,
  "kv_computed_tokens": 51000.0,
  "gen_tokens": 58000.0,
  "prefill_sum_s": 11.3,
  "decode_sum_s": 5686.3,
  "ttft_sum_s": 37.4,
  "ttft_count": 30,
  "cache_queries": 30.0,
  "cache_hits": 21.7,

  "snapshot_before_ts": 12345.678,
  "snapshot_after_ts": 19031.991,
  "wall_clock_s": 6686.313,

  "anomalies": []
}
```

**Field notes:**

- `track` and `pool_or_split` are campaign-level values set at `LatencyCapture` construction. They are not derived from the call site.
- `seed` and `attempt` come from the `snapshot_before()` call (§14.1 same-sprint amendment). Together with `task_id` and `model_id`, they form the join key against LLD-02's `latest_runs` view.
- `snapshot_before_ts` and `snapshot_after_ts` are `time.monotonic()` values — relative to process start, not wall-clock. They are useful for computing `wall_clock_s` and detecting ordering anomalies, but not for correlating with external timestamps.
- `anomalies` is a list of strings (possibly empty). Records with anomalies are excluded from Contribution A aggregation by `load_telemetry()` default filters (§10.1). All records, including anomalous ones, are available for diagnostic appendix tables.

### 9.3 Writer Implementation

```python
import json
import os
import fcntl


class TelemetryWriter:
    """
    Append-only JSONL writer for telemetry records.

    Uses file-level locking (fcntl.flock) as a safety net against
    concurrent writes. In normal operation, only one LatencyCapture
    instance writes to a given file — the lock guards against
    unexpected process duplication.
    """

    def __init__(self, output_path: str):
        self._path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    @property
    def path(self) -> str:
        return self._path

    def write_record(self, metrics: TaskMetrics) -> None:
        record = {
            "task_id": metrics.task_id,
            "model_id": metrics.model_id,
            "track": metrics.track,
            "pool_or_split": metrics.pool_or_split,
            "seed": metrics.seed,
            "attempt": metrics.attempt,

            "ttft_ms": metrics.ttft_ms,
            "prefill_throughput_tps": metrics.prefill_throughput_tps,
            "decode_throughput_tps": metrics.decode_throughput_tps,
            "cache_hit_rate_pct": metrics.cache_hit_rate_pct,

            "prompt_tokens": metrics.prompt_tokens,
            "kv_computed_tokens": metrics.kv_computed_tokens,
            "gen_tokens": metrics.gen_tokens,
            "prefill_sum_s": metrics.prefill_sum_s,
            "decode_sum_s": metrics.decode_sum_s,
            "ttft_sum_s": metrics.ttft_sum_s,
            "ttft_count": metrics.ttft_count,
            "cache_queries": metrics.cache_queries,
            "cache_hits": metrics.cache_hits,

            "snapshot_before_ts": metrics.snapshot_before_ts,
            "snapshot_after_ts": metrics.snapshot_after_ts,
            "wall_clock_s": metrics.wall_clock_s,

            "anomalies": metrics.anomalies,
        }

        with open(self._path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
```

### 9.4 Why JSONL, Not SQLite

JSONL is chosen over SQLite because:

1. **Append-only workload.** Telemetry records are written once and never updated. JSONL is the simplest append-only format.
2. **LLD-12 reads the entire file.** The aggregation API (§10) loads all records into memory. There is no random-access query pattern that would benefit from SQLite indexing.
3. **Consistency with LLD-03/LLD-06 trajectory files.** The trajectory pipeline already uses JSONL throughout. Adding a SQLite dependency for telemetry alone increases operational complexity.
4. **Debuggability.** JSONL can be inspected with `cat`, `jq`, `head`, `tail`, `grep`. SQLite requires a client tool.

Expected file sizes: ~300 tasks × ~400 bytes/record = ~120 KB per campaign file. Even the largest campaign (1,100 B2 runs) produces ~440 KB. In-memory loading is not a concern.

---

## 10. Aggregation API — LLD-12 Interface

LLD-12 consumes telemetry data to produce Contribution A latency anatomy tables. This section defines the aggregation API that LLD-12 calls.

### 10.1 Data Loading

```python
from dataclasses import dataclass
from typing import Optional
import json
import glob


@dataclass
class LatencyRecord:
    """Flat record for aggregation. Subset of JSONL fields."""
    task_id: str
    model_id: str
    track: str
    pool_or_split: str
    seed: int
    attempt: int
    ttft_ms: Optional[float]
    prefill_throughput_tps: Optional[float]
    decode_throughput_tps: Optional[float]
    cache_hit_rate_pct: Optional[float]
    gen_tokens: float
    kv_computed_tokens: float
    prompt_tokens: float
    wall_clock_s: float
    ttft_count: int
    anomalies: list[str]


def _matches_exclusion(anomaly: str, exclusions: set[str]) -> bool:
    """Check if an anomaly matches any exclusion pattern.

    Supports exact match and prefix match with trailing '*'.
    E.g., "negative_delta:*" matches "negative_delta:vllm:generation_tokens_total".
    """
    for excl in exclusions:
        if excl.endswith("*"):
            if anomaly.startswith(excl[:-1]):
                return True
        elif anomaly == excl:
            return True
    return False


# ── Default exclusion set ──────────────────────────────────────────────
# Excludes restart-corrupted records (negative deltas from vLLM restarts
# mid-task) and orphaned-before records (no matching pre-task snapshot).
# Crash/timeout/failed tasks are excluded separately by the reportable_runs
# inner join in aggregate_by_model() — the anomaly filter handles data
# corruption, not task-level outcome filtering.
DEFAULT_EXCLUDE_ANOMALIES = {"orphaned_before", "negative_delta:*"}


def load_telemetry(
    telemetry_dir: str,
    exclude_anomalies: Optional[set[str]] = None,
) -> list[LatencyRecord]:
    """
    Load all telemetry JSONL files from the output directory.

    exclude_anomalies: set of anomaly patterns to filter out.
      Default: DEFAULT_EXCLUDE_ANOMALIES — excludes orphaned-before
      and restart-corrupted records.
    """
    if exclude_anomalies is None:
        exclude_anomalies = DEFAULT_EXCLUDE_ANOMALIES

    records = []
    for path in sorted(glob.glob(f"{telemetry_dir}/latency_*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)

                # Filter by anomalies
                task_anomalies = data.get("anomalies", [])
                if any(_matches_exclusion(a, exclude_anomalies) for a in task_anomalies):
                    continue

                records.append(LatencyRecord(
                    task_id=data["task_id"],
                    model_id=data["model_id"],
                    track=data["track"],
                    pool_or_split=data["pool_or_split"],
                    seed=data["seed"],
                    attempt=data["attempt"],
                    ttft_ms=data.get("ttft_ms"),
                    prefill_throughput_tps=data.get("prefill_throughput_tps"),
                    decode_throughput_tps=data.get("decode_throughput_tps"),
                    cache_hit_rate_pct=data.get("cache_hit_rate_pct"),
                    gen_tokens=data["gen_tokens"],
                    kv_computed_tokens=data["kv_computed_tokens"],
                    prompt_tokens=data["prompt_tokens"],
                    wall_clock_s=data["wall_clock_s"],
                    ttft_count=data["ttft_count"],
                    anomalies=task_anomalies,
                ))
    return records
```

### 10.2 Contribution A Aggregation

LLD-12 calls these functions to produce the Dev-Bench leaderboard tables.

```python
from collections import defaultdict


@dataclass
class ModelLatencySummary:
    """Per-model aggregated latency summary for Contribution A tables."""
    model_id: str
    pool_or_split: str
    n_tasks: int

    # Central tendency (median — robust to outliers)
    ttft_ms_median: Optional[float]
    prefill_throughput_tps_median: Optional[float]
    decode_throughput_tps_median: Optional[float]
    cache_hit_rate_pct_median: Optional[float]

    # Spread (IQR — p25/p75)
    ttft_ms_p25: Optional[float]
    ttft_ms_p75: Optional[float]
    prefill_throughput_tps_p25: Optional[float]
    prefill_throughput_tps_p75: Optional[float]
    decode_throughput_tps_p25: Optional[float]
    decode_throughput_tps_p75: Optional[float]
    cache_hit_rate_pct_p25: Optional[float]
    cache_hit_rate_pct_p75: Optional[float]

    # Volume
    total_gen_tokens: float
    total_prompt_tokens: float
    total_wall_clock_s: float
    total_turns: int


def _percentile(values: list[float], p: float) -> Optional[float]:
    """Compute the p-th percentile (0–100) of a sorted list."""
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    floor_k = int(k)
    ceil_k = min(floor_k + 1, len(values) - 1)
    frac = k - floor_k
    return values[floor_k] + frac * (values[ceil_k] - values[floor_k])


class TelemetryGapError(Exception):
    """Raised when reportable runs are missing telemetry records."""
    def __init__(self, missing_keys: set[tuple[str, str, int, int]]):
        self.missing_keys = missing_keys
        super().__init__(
            f"{len(missing_keys)} reportable runs have no telemetry record. "
            f"First 5: {sorted(missing_keys)[:5]}. "
            f"This blocks Contribution A aggregation — telemetry must be "
            f"complete for all published results. See §12.5 for the "
            f"operator recovery workflow."
        )


def aggregate_by_model(
    records: list[LatencyRecord],
    reportable_runs: set[tuple[str, str, int, int]],
) -> list[ModelLatencySummary]:
    """
    Aggregate telemetry records into per-model summaries.

    reportable_runs: set of (task_id, model_id, seed, attempt) tuples
    identifying the runs that should contribute to Contribution A.
    LLD-12 constructs this set from LLD-02's latest_runs view filtered to:
        outcome = 'resolved' AND is_current = 1 AND harness = 'codex'
    and pre-filtered to the target pool_or_split.

    The key includes model_id because Dev-Bench runs the same 50 tasks
    across multiple models — without model_id, a telemetry record from
    model A could satisfy a completeness check for model B on the same
    task/seed/attempt. harness is scoped by LLD-12 at query time (only
    harness='codex' runs produce /metrics telemetry; SWE-Agent runs
    through LLD-09 do not route through Codex CLI and are not captured
    by LLD-04).

    This set serves two purposes:
      1. ROW SELECTION — only records matching a key in reportable_runs
         contribute to medians and percentiles. Crash/timeout/failed runs
         are excluded because they are not in the set. Superseded attempts
         (retries where a newer attempt exists) are excluded because
         latest_runs collapses to MAX(attempt).
      2. COMPLETENESS VALIDATION — every key in reportable_runs must have
         a matching telemetry record. If any are missing, TelemetryGapError
         is raised and aggregation is blocked.

    LLD-12 is responsible for pool-scoping and harness-scoping: it queries
    LLD-02 for the target pool with harness='codex' and passes only those
    runs. aggregate_by_model() does not filter by pool_or_split or
    harness — reportable_runs is the sole authority on which records to
    include.
    """
    # ── Row selection: inner join with reportable_runs ──
    selected = [
        r for r in records
        if (r.task_id, r.model_id, r.seed, r.attempt) in reportable_runs
    ]

    # ── Completeness check ──
    present_keys = {(r.task_id, r.model_id, r.seed, r.attempt) for r in selected}
    missing = reportable_runs - present_keys
    if missing:
        raise TelemetryGapError(missing)

    # ── Aggregate by model ──
    by_model: dict[str, list[LatencyRecord]] = defaultdict(list)
    for r in selected:
        by_model[r.model_id].append(r)

    summaries = []
    for model_id, model_records in sorted(by_model.items()):
        ttfts = [r.ttft_ms for r in model_records if r.ttft_ms is not None]
        prefills = [r.prefill_throughput_tps for r in model_records
                    if r.prefill_throughput_tps is not None]
        decodes = [r.decode_throughput_tps for r in model_records
                   if r.decode_throughput_tps is not None]
        caches = [r.cache_hit_rate_pct for r in model_records
                  if r.cache_hit_rate_pct is not None]

        summaries.append(ModelLatencySummary(
            model_id=model_id,
            pool_or_split=model_records[0].pool_or_split,
            n_tasks=len(model_records),
            ttft_ms_median=_percentile(ttfts, 50),
            prefill_throughput_tps_median=_percentile(prefills, 50),
            decode_throughput_tps_median=_percentile(decodes, 50),
            cache_hit_rate_pct_median=_percentile(caches, 50),
            ttft_ms_p25=_percentile(ttfts, 25),
            ttft_ms_p75=_percentile(ttfts, 75),
            prefill_throughput_tps_p25=_percentile(prefills, 25),
            prefill_throughput_tps_p75=_percentile(prefills, 75),
            decode_throughput_tps_p25=_percentile(decodes, 25),
            decode_throughput_tps_p75=_percentile(decodes, 75),
            cache_hit_rate_pct_p25=_percentile(caches, 25),
            cache_hit_rate_pct_p75=_percentile(caches, 75),
            total_gen_tokens=sum(r.gen_tokens for r in model_records),
            total_prompt_tokens=sum(r.prompt_tokens for r in model_records),
            total_wall_clock_s=sum(r.wall_clock_s for r in model_records),
            total_turns=sum(r.ttft_count for r in model_records),
        ))
    return summaries
```

### 10.3 Contribution A Table Format

The aggregation above produces the data for the following Contribution A table structure (rendered by LLD-12):

```
Model                    | TTFT (ms)      | Prefill (tok/s) | Decode (tok/s) | Cache Hit (%)   | N
                         | med [p25–p75]  | med [p25–p75]   | med [p25–p75]  | med [p25–p75]   |
─────────────────────────┼────────────────┼─────────────────┼────────────────┼─────────────────┼────
Qwen3.5-35B-A3B          | 890 [720–1100] | 5200 [4800–5600]| 35.1 [33–37]   | 74.2 [68–79]    | 50
Qwen3-Coder-Next-80B-A3B| 950 [780–1200] | 4800 [4400–5200]| 33.0 [31–35]   | 72.1 [65–77]    | 50
...
```

All values are illustrative. Actual numbers come from Sprint 2 data collection.

---

## 11. Full LatencyCapture Implementation

Assembling the components from §4–§10 into the complete class:

```python
class LatencyCapture:
    """
    Full implementation of the LatencyCapture interface consumed by LLD-03.

    Campaign-level metadata (model_id, pool_or_split, track) is set at
    construction by LLD-07. Per-task metadata (task_id, seed) comes from
    snapshot_before(). The interface matches LLD-03 v1.1 call sites with
    the §14.1 same-sprint amendment (seed and attempt kwargs on snapshot_before).

    Lifecycle:
      1. Construct with vLLM connection params, output directory, and
         campaign metadata (model_id, pool_or_split).
      2. Call resolve_schema() once after vLLM is healthy.
      3. Per-task: snapshot_before(task_id, seed, attempt) → [task] → snapshot_after(task_id)
      4. After campaign: LLD-12 calls load_telemetry() + aggregate_by_model().
    """

    # ── Track derivation ──
    _SWE_BENCH_POOLS = {"dev_bench", "bench_control", "final_test"}
    _CODEX_LONG_SPLITS = {"train_long", "val_long", "test_long", "public_dev"}

    @staticmethod
    def _derive_track(pool_or_split: str) -> str:
        if pool_or_split in LatencyCapture._SWE_BENCH_POOLS:
            return "swe_bench"
        if pool_or_split in LatencyCapture._CODEX_LONG_SPLITS:
            return "codex_long"
        raise ValueError(
            f"Cannot derive track from pool_or_split='{pool_or_split}'. "
            f"Expected one of {LatencyCapture._SWE_BENCH_POOLS | LatencyCapture._CODEX_LONG_SPLITS}."
        )

    def __init__(
        self,
        vllm_host: str,
        vllm_port: int,
        output_dir: str,
        model_id: str,
        pool_or_split: str,
    ):
        self._host = vllm_host
        self._port = vllm_port
        self._model_id = model_id
        self._pool_or_split = pool_or_split
        self._track = self._derive_track(pool_or_split)
        self._schema: Optional[dict[str, str]] = None
        self._snapshot_mgr: Optional[SnapshotManager] = None
        self._writer = TelemetryWriter(
            os.path.join(output_dir, "telemetry",
                         f"latency_{model_id}_{pool_or_split}.jsonl")
        )

    async def resolve_schema(self) -> None:
        snapshot = await fetch_metrics(self._host, self._port)
        self._schema = resolve_metric_schema(snapshot)
        self._snapshot_mgr = SnapshotManager(
            self._host, self._port, self._schema,
            self._model_id, self._track, self._pool_or_split,
        )
        import logging
        logger = logging.getLogger("codex_bench.telemetry")
        logger.info(f"Metric schema resolved: {self._schema}")

    async def snapshot_before(self, task_id: str, seed: int, attempt: int) -> None:
        """Match LLD-03 call site: snapshot_before(task_id=..., seed=..., attempt=...)"""
        if self._snapshot_mgr is None:
            raise RuntimeError(
                "resolve_schema() must be called before snapshot_before()"
            )
        await self._snapshot_mgr.capture_before(task_id, seed, attempt)

    async def snapshot_after(self, task_id: str) -> TaskMetrics:
        """Match LLD-03 call site: snapshot_after(task_id=...)"""
        if self._snapshot_mgr is None:
            raise RuntimeError(
                "resolve_schema() must be called before snapshot_after()"
            )
        pending, after, after_ts, snapshot_anomalies = (
            await self._snapshot_mgr.capture_after(task_id)
        )

        metrics = compute_task_metrics(
            pending, after, after_ts, self._schema,
            model_id=self._model_id,
            track=self._track,
            pool_or_split=self._pool_or_split,
            initial_anomalies=snapshot_anomalies,
        )

        self._writer.write_record(metrics)
        return metrics
```

---

## 12. Error Handling

### 12.1 `/metrics` Fetch Failure

If `fetch_metrics()` fails (network error, timeout, vLLM crashed):

- **In `snapshot_before()`:** The exception propagates to LLD-03's `execute_task()`. LLD-03's `try` block catches it, calls `finish_run(outcome="crash")`, and re-raises. The task is retried by LLD-07 with `attempt + 1`. This is correct — if `/metrics` is unreachable, vLLM is likely unhealthy and the task would fail anyway.
- **In `snapshot_after()`:** Called from LLD-03's `finally` block, which wraps it in a `try/except` that silently swallows the exception (telemetry failure must not mask the real error). The task proceeds with no telemetry record written. The `aggregate_by_model()` completeness check (§10.2) catches this: the missing record causes `TelemetryGapError` when LLD-12 attempts aggregation, blocking Contribution A publication until the gap is resolved.

### 12.2 vLLM Restart Between Snapshots

If vLLM restarts between `snapshot_before()` and `snapshot_after()` (e.g., OOM crash), all cumulative counters reset to zero. The delta computation produces large negative values. These are caught by the `_delta()` function (§7.1) and flagged as `"negative_delta:<metric>"` anomalies.

The record is written (for diagnostic value) but **excluded from Contribution A aggregation** by `load_telemetry()`'s default anomaly filters, which include `"negative_delta:*"`. This is consistent with the `reportable_runs` completeness check: the excluded record is absent from the loaded records, so if the run's (task_id, model_id, seed, attempt) is in `reportable_runs`, `aggregate_by_model()` raises `TelemetryGapError`. Recovery follows the operator workflow in §12.5.

**Detection heuristic:** If any delta is negative, log a warning identifying the affected metric and the task_id. LLD-07 should correlate this with any vLLM restart events in its own logs.

### 12.3 Orphaned Snapshots

If LLD-03 calls `snapshot_before()` but never calls `snapshot_after()` (process crash, unhandled exception that bypasses `finally`), the pending snapshot remains in memory. On the next `snapshot_before()` call for a different task_id, `SnapshotManager` raises `RuntimeError` to flag the programming error.

If `snapshot_after()` is called for a task_id that has no pending `snapshot_before()` (e.g., `SnapshotManager` instance was replaced between the two calls due to a process restart), the `SnapshotManager` emits the `"orphaned_before"` anomaly (§6.2). The record is written with a synthetic zero before-snapshot and excluded from Contribution A aggregation by the default anomaly filters.

Note: LLD-03 only calls `snapshot_after()` when `telemetry_started=True`, which requires `snapshot_before()` to have succeeded within the current process. The orphan path is therefore a guard against `SnapshotManager` state loss across process boundaries, not against `snapshot_before()` network failures.

### 12.4 Zero-Delta Tasks

A task that produces no inference requests (e.g., immediate crash before the first Codex API call) will have zero deltas for all metrics. The derived metrics are all `None` (guarded by `> 0` divisor checks). The `"zero_gen_tokens"` and/or `"zero_prefill_tokens"` anomaly flags are set. The record is written and passes default anomaly filters (zero-delta is not an exclusion condition — but LLD-12 should filter these tasks by LLD-02 outcome, since they will have `outcome="crash"` or `outcome="timeout"`, not `outcome="resolved"`).

### 12.5 Telemetry Completeness Enforcement

Missing telemetry on reportable runs is a **fail-closed** condition for Contribution A. The `aggregate_by_model()` function (§10.2) enforces this via the `reportable_runs` inner join:

1. LLD-12 queries LLD-02's `latest_runs` view for `outcome='resolved' AND is_current=1 AND harness='codex'` in the target pool.
2. LLD-12 constructs `reportable_runs = {(r.scenario_id, r.model_id, r.seed, r.attempt) for r in resolved_latest_runs}`.
3. LLD-12 calls `aggregate_by_model(loaded_records, reportable_runs)`.
4. `aggregate_by_model()` selects only records matching `reportable_runs`, then checks that every key in `reportable_runs` has a matching record.
5. If any reportable run is missing → `TelemetryGapError`. LLD-12 must not publish partial results.

This design handles three correctness cases:

- **Crash/timeout/failed tasks:** Not in `reportable_runs` (outcome ≠ "resolved"), so their telemetry rows are ignored even though `snapshot_after()` was called in LLD-03's `finally` block.
- **Superseded attempts (retries):** Not in `reportable_runs` (`latest_runs` collapses to `MAX(attempt)`). An attempt-1 crash followed by an attempt-2 success means only attempt 2 is in `reportable_runs`.
- **Anomaly-excluded records:** Removed by `load_telemetry()` before reaching `aggregate_by_model()`. If a reportable run's only record was excluded (e.g., negative delta), it shows up as missing here.

**Recovery path for missing telemetry — out-of-band operator procedure:**

There is no way to reconstruct telemetry after the fact — the per-task before/after `/metrics` snapshot window is gone. The affected run must be re-executed to produce clean telemetry.

Signed-off LLD-02's dispatch state machine has no telemetry-aware invalidation path. `check_dispatch_eligible()` returns `SKIP` for a current finished run — a resolved run with missing telemetry is still a valid resolved run from LLD-02's perspective. LLD-02's `invalidate_stale_runs()` API is scoped to Codex-Long manifest-artifact changes (`family_id`, `affected_artifact`, `new_manifest_version`) and does not apply to arbitrary run-level telemetry repair, nor to SWE-bench runs at all.

Recovery therefore requires **out-of-band operator DB surgery** — a direct SQL update on the LLD-02 `runs` table, outside the normal API surface:

1. Operator identifies the missing runs from the `TelemetryGapError` (which lists the affected `(task_id, model_id, seed, attempt)` keys).
2. Operator executes a direct SQL update to mark the affected runs non-current:
   ```sql
   UPDATE runs
   SET is_current = 0
   WHERE scenario_id = :task_id
     AND model_id = :model_id
     AND seed = :seed
     AND attempt = :attempt;
   ```
3. Operator logs the justification (telemetry repair) outside the manifest changelog — this is not a manifest-driven invalidation.
4. On the next dispatch cycle, `check_dispatch_eligible()` returns `RERUN_NEEDED` for the invalidated run (the standard non-current → rerun path).
5. LLD-07 re-dispatches the run. The new attempt produces fresh telemetry.
6. LLD-12 re-queries `latest_runs` and re-runs `aggregate_by_model()`.

**This is not an automated recovery and not covered by any signed-off LLD-02 API.** It is explicit DB surgery for a rare edge case. No amendments to LLD-02's state machine or API surface are proposed — the frequency does not justify a new invalidation path.

**Expected frequency:** This path is exercised only when a task resolves successfully but `/metrics` is unreachable at the exact moment `snapshot_after()` is called, or when vLLM restarts mid-task but the task still produces a correct result. Both are rare — across a typical 300-run Dev-Bench campaign, zero occurrences are expected under normal operating conditions.

---

## 13. Configuration

### 13.1 Config Schema

```python
@dataclass
class TelemetryConfig:
    """Configuration for the LatencyCapture component."""

    # vLLM connection
    vllm_host: str = "127.0.0.1"
    vllm_port: int = 8000

    # Output
    output_dir: str = "output"

    # Fetch settings
    metrics_fetch_timeout_s: float = 5.0

    # Aggregation defaults — must match DEFAULT_EXCLUDE_ANOMALIES
    default_exclude_anomalies: set[str] = field(
        default_factory=lambda: {"orphaned_before", "negative_delta:*"}
    )
```

### 13.2 Deployment Notes

- LLD-04 runs in the same process as LLD-03 (the orchestrator). No separate service or deployment.
- `/metrics` is accessed on `127.0.0.1:8000` (the same loopback address used by LLD-03 for `/health` and `/reset_prefix_cache`). It is NOT proxied through the inference proxy (§5A in LLD-03) — the proxy forwards only inference endpoints.
- The output directory is the same as LLD-03's output directory. Telemetry files are written to a `telemetry/` subdirectory.

---

## 14. Cross-LLD Amendments

### 14.1 Same-Sprint Amendment to LLD-03 — `seed` and `attempt` in `snapshot_before()`

LLD-03 v1.1 calls `snapshot_before(task_id=task.scenario_id)` with only `task_id`. LLD-04 requires `seed` and `attempt` to produce unambiguous telemetry records that align with LLD-02's run identity model:

- `seed` distinguishes multi-seed runs of the same task (a task at seed 1 vs seed 2 must produce separate records).
- `attempt` distinguishes retries (a crash on attempt 1 followed by a resolved attempt 2 must produce separate records). Without `attempt`, LLD-12 cannot join telemetry with LLD-02's `latest_runs` view, which collapses retries by `MAX(attempt)`.

**Amendment (same-sprint — both LLD-03 and LLD-04 are Sprint 1):**

LLD-03 §4.3 and §10.1 call site change from:

```python
await latency_capture.snapshot_before(task_id=task.scenario_id)
```

to:

```python
await latency_capture.snapshot_before(
    task_id=task.scenario_id, seed=task.seed, attempt=task.attempt
)
```

This is a two-kwarg addition. The `snapshot_after()` call site is unchanged:

```python
await latency_capture.snapshot_after(task_id=task.scenario_id)
```

No other LLD-03 sections are affected. The `LatencyCapture` type hint in `execute_task()` remains unchanged.

---

## 15. Sprint 0 Checklist — LLD-04 Items

These items must complete during Sprint 0 alongside the vLLM version pin and Gate 1 validation:

- [ ] **Metric name resolution:** After pinning vLLM version, run `resolve_metric_schema()` against the live endpoint. Record the resolved schema in the launch log.
- [ ] **`request_prefill_kv_computed_tokens` presence:** Verify this histogram exists in the `/metrics` output. If absent, this is a **Sprint 0 blocker** — upgrade vLLM or file upstream issue. No fallback.
- [ ] **Capture a reference `/metrics` snapshot:** Save the raw Prometheus text output as a test fixture for the parser unit tests.
- [ ] **Validate delta-sampling on a 5-task run:** Run 5 SWE-bench tasks from the Gate 1 smoke test through the full before/after snapshot cycle. Verify that deltas are non-negative and that `ttft_count` matches the observed turn count in the trajectory JSONL.

---

## 16. Sprint 1 Validation Checklist

These items are part of the LLD-03 §15 end-to-end validation (10 SWE-bench + 10 Codex-Long tasks):

- [ ] Telemetry JSONL files are created in `output/telemetry/`.
- [ ] Every successfully completed task (`outcome="resolved"` AND `is_current=1` AND `harness="codex"` in LLD-02 `latest_runs`) has a corresponding telemetry record — `aggregate_by_model(records, reportable_runs)` completes without `TelemetryGapError`.
- [ ] No `negative_delta:*` anomalies (no vLLM restarts during the validation run).
- [ ] No `orphaned_before` anomalies (every `snapshot_after` has a matching `snapshot_before`).
- [ ] `kv_computed_tokens ≤ prompt_tokens` for every task.
- [ ] `cache_hit_rate_pct` is > 0% for tasks with > 1 turn (prefix cache is ON).
- [ ] `decode_throughput_tps` is in a plausible range for the model under test (e.g., 8–15 tok/s for 27B dense at FP8).
- [ ] `aggregate_by_model()` produces a valid `ModelLatencySummary` with no `None` medians for a pool with ≥ 5 completed tasks.
- [ ] Turn cross-validation (§8.3): `ttft_count == len(extract_turns(trajectory_path))` for ≥80% of tasks — run separately by LLD-12 using trajectory paths from LLD-02 run records.
- [ ] `seed` and `attempt` fields are correctly populated in all telemetry records (verifies §14.1 amendment is implemented).

---

## 17. Connections to Other LLDs

| LLD | Relationship |
|---|---|
| **LLD-01** vLLM Serving Layer | Consumes `GET /metrics` endpoint. §9.2 is the authoritative source for metric formulas, delta-sampling protocol, schema resolver pattern, and the `_sum` delta procedure. LLD-04 implements §9.2 — it does not define new formulas. All metrics in §9.2 are required with no fallback. |
| **LLD-03** Task Orchestrator | Provides the `LatencyCapture` interface that LLD-03 calls. LLD-03 owns snapshot timing (before/after each task in the `finally` block). §14.1 defines a same-sprint amendment adding `seed` and `attempt` to `snapshot_before()`. Runs in the same process as LLD-03. |
| **LLD-07** Benchmark Runner | LLD-07 instantiates `LatencyCapture` for each campaign (model × pool), passing `model_id` and `pool_or_split` at construction. LLD-07 passes the `LatencyCapture` instance to LLD-03's `execute_task()`. LLD-07 does not read telemetry directly — it delegates to LLD-12. |
| **LLD-12** Results & Artifact Generator | Terminal consumer. Calls `load_telemetry()` and `aggregate_by_model()` to produce Contribution A latency anatomy tables. Constructs `reportable_runs` from LLD-02's `latest_runs` view (`outcome='resolved' AND is_current=1 AND harness='codex'`) for both row selection and completeness validation. Calls `extract_turns()` with trajectory paths from LLD-02 for per-turn diagnostic analysis. LLD-12 owns bootstrap CI computation on the per-task metric distributions. |

No amendments to signed-off LLDs are required beyond the §14.1 same-sprint change to LLD-03 (two kwargs added to `snapshot_before()`). The telemetry-gap recovery path (§12.5) is an out-of-band operator procedure (direct SQL on the LLD-02 `runs` table) for a rare edge case — it does not use any signed-off LLD-02 API and does not require changes to the dispatch state machine.

---

## 18. Known Issues & Risks

| Issue | Severity | Mitigation |
|---|---|---|
| **`request_prefill_kv_computed_tokens` may be absent in older vLLM** | HIGH | Sprint 0 blocker. `resolve_metric_schema()` hard-fails if absent. Must upgrade vLLM or file upstream issue. No fallback to `prompt_tokens` — that would change the metric definition and violate LLD-01 §9.2. |
| **Codex event stream format not formally specified** | MEDIUM | Turn extraction (§8) is best-effort and separate from the primary telemetry pipeline. Sprint 1 validates against actual Codex output. If event types differ from provisional names, update the parser. Per-turn attribution is a diagnostic extra, not a primary deliverable. |
| **vLLM metric name schema may change with version updates** | LOW | Schema resolver (§4.2) probes both known variants and fails hard on unknown names. When pinning a new vLLM version, run the resolver first and add new variants if needed. |
| **`/metrics` fetch adds ~10–50ms per task** | LOW | Two fetches per task (before + after) add negligible overhead relative to 40–110 min task durations. Not on the critical path. |
| **Prometheus text parser is a simplified implementation** | LOW | Does not handle all edge cases of the Prometheus exposition format (e.g., multi-line HELP strings, escaped characters in labels). This is acceptable because vLLM's `/metrics` output uses a consistent subset. Validate against the captured test fixture. |
| **Turn count cross-validation may fail for complex Codex sessions** | MEDIUM | Multi-tool-call turns may produce event patterns that the turn parser misinterprets. The 20% tolerance in §8.3 accommodates this. If cross-validation fails > 20% of tasks, refine the parser using actual event stream examples. |
| **Single-stream assumption** | LOW | LLD-04 assumes one Codex session runs at a time (matching HLD `--max-num-seqs 4` for single-stream serving). If this changes, per-task metric attribution via global counter deltas would be inaccurate. This is not planned to change. |
| **Missing telemetry requires out-of-band operator DB surgery** | MEDIUM | There is no way to reconstruct telemetry after the fact — the before/after snapshot window is gone. If `aggregate_by_model()` raises `TelemetryGapError`, recovery requires direct SQL on the LLD-02 `runs` table to mark the affected run non-current (§12.5). This is not covered by any signed-off LLD-02 API — it is explicit DB surgery for a rare edge case. Expected to be rare: requires a task to resolve successfully while `/metrics` is simultaneously unreachable. |
| **JSONL may accumulate stale records from retries** | LOW | A crashed attempt 1 followed by a resolved attempt 2 leaves two telemetry records in the JSONL. The stale attempt-1 record is not in `reportable_runs` and is excluded by the inner join. It occupies disk space (~400 bytes) but does not affect correctness. No compaction is implemented — the file sizes are negligible. |

---

## 19. Open Questions — Status

| Question | Status |
|---|---|
| Exact Prometheus metric names for the pinned vLLM version | **OPEN — Sprint 0 validation. §4.3 defines the procedure. Resolver handles both known variants.** |
| Codex `--json` event type names for turn boundary detection | **OPEN — Sprint 1 validation. §8.2 provides provisional parser. Updated after first real trajectory capture.** |
| Whether `request_prefill_kv_computed_tokens` is present in the target vLLM build | **OPEN — Sprint 0 blocker. Hard-fail if absent, no fallback (§4.1, §15).** |
| Per-turn intermediate snapshot capture (vs post-hoc estimation) | **Resolved (§8.1) — post-hoc estimation chosen. Mean-per-turn is the Contribution A reporting level. Intermediate snapshots deferred to Sprint 3 enhancement if needed.** |
| Storage format (JSONL vs SQLite) | **Resolved (§9.4) — JSONL chosen. Append-only workload, small files, consistency with trajectory pipeline.** |
| Bootstrap CI computation on latency distributions | **Resolved — owned by LLD-12, not LLD-04. LLD-04 provides raw per-task values; LLD-12 computes CIs.** |
| How LLD-04 gets track/pool_or_split/seed/attempt metadata | **Resolved (§3, §14.1) — track and pool_or_split are campaign-level (construction). seed and attempt are per-task via same-sprint LLD-03 amendment.** |
| How LLD-04 finds trajectory files for turn extraction | **Resolved (§8.2) — LLD-04 does not find them. Turn extraction is a post-hoc utility. LLD-12 passes trajectory_path from LLD-02 run records.** |
| Telemetry completeness enforcement | **Resolved (§10.2, §12.5) — fail-closed. aggregate_by_model() inner-joins with reportable_runs from LLD-02 latest_runs view, then validates completeness. Missing runs raise TelemetryGapError.** |
| How crash/timeout/retry rows are excluded from Contribution A | **Resolved (§10.2, §12.5) — reportable_runs inner join. Only outcome="resolved" AND is_current=1 AND harness="codex" runs from LLD-02's latest_runs view contribute. Crash/timeout rows are excluded by row selection, not anomaly filtering. Superseded attempts are excluded because latest_runs collapses to MAX(attempt).** |

---

*LLD-04 · Latency Telemetry Capture · Signed Off v0.7 · April 2026*

from __future__ import annotations

import asyncio
import fcntl
import glob
import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

_METRIC_VARIANTS: dict[str, list[str]] = {
    "prompt_tokens": ["vllm:prompt_tokens_total", "vllm:prompt_tokens"],
    "generation_tokens": ["vllm:generation_tokens_total", "vllm:generation_tokens"],
    "kv_computed_tokens": ["vllm:request_prefill_kv_computed_tokens"],
    "cache_queries": ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"],
    "cache_hits": ["vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"],
    "ttft": ["vllm:time_to_first_token_seconds"],
    "prefill_time": ["vllm:request_prefill_time_seconds"],
    "decode_time": ["vllm:request_decode_time_seconds"],
    "itl": ["vllm:inter_token_latency_seconds"],
}

REQUIRED_METRIC_VARIANTS = {
    "prompt_tokens": _METRIC_VARIANTS["prompt_tokens"],
    "generation_tokens": _METRIC_VARIANTS["generation_tokens"],
    "prefix_cache_queries": _METRIC_VARIANTS["cache_queries"],
    "prefix_cache_hits": _METRIC_VARIANTS["cache_hits"],
    "kv_computed_tokens_sum": _METRIC_VARIANTS["kv_computed_tokens"],
    "ttft_seconds_sum": _METRIC_VARIANTS["ttft"],
    "ttft_seconds_count": _METRIC_VARIANTS["ttft"],
    "prefill_seconds_sum": _METRIC_VARIANTS["prefill_time"],
    "decode_seconds_sum": _METRIC_VARIANTS["decode_time"],
}

_HISTOGRAM_METRICS = {"kv_computed_tokens", "ttft", "prefill_time", "decode_time", "itl"}
_SKIP_SUFFIXES = ("_bucket",)
_SAMPLE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{[^}]*\})?"
    r"\s+"
    r"([0-9eE.+\-]+|NaN|Inf|\+Inf|-Inf)"
    r"(?:\s+\d+)?$"
)


def _normalize_metric_key(key: str) -> str:
    return key.split("{", 1)[0]


def parse_prometheus_text(raw: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        metric_name = stripped.split("{", 1)[0].split()[0]
        if any(metric_name.endswith(suffix) for suffix in _SKIP_SUFFIXES):
            continue

        match = _SAMPLE_RE.match(stripped)
        if match is None:
            continue

        try:
            value = float(match.group(2))
        except ValueError:
            continue

        key = _normalize_metric_key(match.group(1))
        result[key] = result.get(key, 0.0) + value
    return result


def _resolve_metric_candidate(logical_name: str, candidates: list[str], metrics_snapshot: dict[str, float]) -> str:
    for candidate in candidates:
        probe_key = f"{candidate}_sum" if logical_name in _HISTOGRAM_METRICS else candidate
        if probe_key in metrics_snapshot:
            return candidate
    if logical_name in _HISTOGRAM_METRICS:
        probed = [f"{candidate}_sum" for candidate in candidates]
    else:
        probed = list(candidates)
    raise RuntimeError(
        f"vLLM /metrics does not expose any of {candidates}. "
        f"Probed keys: {probed}. Update metric constants for the pinned vLLM version."
    )


def resolve_metric_schema(metrics_snapshot: dict[str, float]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for logical_name, candidates in _METRIC_VARIANTS.items():
        resolved[logical_name] = _resolve_metric_candidate(logical_name, candidates, metrics_snapshot)
    return resolved


def _schema_value(schema: dict[str, str], logical_name: str, *aliases: str) -> str:
    candidate_keys = [logical_name, *aliases]
    for key in candidate_keys:
        value = schema.get(key)
        if isinstance(value, str):
            return value
        total_value = schema.get(f"{key}_total")
        if isinstance(total_value, str):
            return total_value
    raise KeyError(f"Schema is missing metric mapping for {logical_name!r}")


def _schema_histogram_base(schema: dict[str, str], logical_name: str, *aliases: str) -> str:
    candidate_keys = [logical_name, *aliases]
    for key in candidate_keys:
        value = schema.get(key)
        if isinstance(value, str):
            if value.endswith("_sum"):
                return value[: -len("_sum")]
            if value.endswith("_count"):
                return value[: -len("_count")]
            return value
        for suffix in ("_sum", "_count"):
            suffixed_value = schema.get(f"{key}{suffix}")
            if isinstance(suffixed_value, str):
                return suffixed_value[: -len(suffix)]
    raise KeyError(f"Schema is missing histogram metric mapping for {logical_name!r}")


def _required_delta(before: dict[str, float], after: dict[str, float], key: str) -> float:
    if key not in before or key not in after:
        raise RuntimeError(f"Expected metric '{key}' not found in both /metrics snapshots")
    delta = after[key] - before[key]
    if delta < 0:
        raise RuntimeError(
            f"Metric '{key}' decreased between /metrics snapshots; counters likely reset or the server restarted mid-task."
        )
    return delta


def _delta(before: dict[str, float], after: dict[str, float], key: str) -> tuple[float, Optional[str]]:
    if key not in after:
        raise RuntimeError(
            f"Expected metric '{key}' not found in /metrics snapshot. Did the vLLM server restart with a different version?"
        )
    before_val = before.get(key, 0.0)
    delta = after[key] - before_val
    anomaly = f"negative_delta:{key}" if delta < 0 else None
    return delta, anomaly


def _task_metrics_from_snapshots(
    before: dict[str, float],
    after: dict[str, float],
    schema: dict[str, str],
) -> dict[str, float | None]:
    prompt_tokens = _required_delta(before, after, _schema_value(schema, "prompt_tokens"))
    kv_key = _schema_histogram_base(schema, "kv_computed_tokens", "kv_computed_tokens_sum")
    gen_key = _schema_value(schema, "generation_tokens")
    cache_queries_key = _schema_value(schema, "cache_queries", "prefix_cache_queries")
    cache_hits_key = _schema_value(schema, "cache_hits", "prefix_cache_hits")
    ttft_key = _schema_histogram_base(schema, "ttft", "ttft_seconds")
    prefill_key = _schema_histogram_base(schema, "prefill_time", "prefill_seconds")
    decode_key = _schema_histogram_base(schema, "decode_time", "decode_seconds")

    kv_computed_tokens = _required_delta(before, after, f"{kv_key}_sum")
    gen_tokens = _required_delta(before, after, gen_key)
    cache_queries = _required_delta(before, after, cache_queries_key)
    cache_hits = _required_delta(before, after, cache_hits_key)
    ttft_sum_s = _required_delta(before, after, f"{ttft_key}_sum")
    ttft_count = _required_delta(before, after, f"{ttft_key}_count")
    prefill_sum_s = _required_delta(before, after, f"{prefill_key}_sum")
    decode_sum_s = _required_delta(before, after, f"{decode_key}_sum")

    return {
        "ttft_ms": (ttft_sum_s / ttft_count * 1000) if ttft_count > 0 else None,
        "prefill_throughput_tps": (kv_computed_tokens / prefill_sum_s) if prefill_sum_s > 0 else None,
        "decode_throughput_tps": (gen_tokens / decode_sum_s) if decode_sum_s > 0 else None,
        "cache_hit_rate_pct": (cache_hits / cache_queries * 100) if cache_queries > 0 else None,
        "prompt_tokens": prompt_tokens,
        "kv_computed_tokens": kv_computed_tokens,
        "gen_tokens": gen_tokens,
        "prefill_sum_s": prefill_sum_s,
        "decode_sum_s": decode_sum_s,
        "ttft_sum_s": ttft_sum_s,
        "ttft_count": ttft_count,
        "cache_queries": cache_queries,
        "cache_hits": cache_hits,
    }


@dataclass
class TaskMetrics:
    """Per-task derived metrics plus raw deltas and timing context."""

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
    prompt_tokens: float
    kv_computed_tokens: float
    gen_tokens: float
    prefill_sum_s: float
    decode_sum_s: float
    ttft_sum_s: float
    ttft_count: int
    cache_queries: float
    cache_hits: float
    snapshot_before_ts: float
    snapshot_after_ts: float
    wall_clock_s: float
    anomalies: list[str] = field(default_factory=list)


@dataclass
class PendingSnapshot:
    task_id: str
    seed: int
    attempt: int
    snapshot: dict[str, float]
    timestamp: float


@dataclass
class TurnInfo:
    """Extracted turn-level data from trajectory JSONL."""

    turn_index: int
    start_timestamp: Optional[str]
    end_timestamp: Optional[str]
    output_tokens_approx: int
    tool_calls: int


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


@dataclass
class ModelLatencySummary:
    """Per-model aggregated latency summary for Contribution A tables."""

    model_id: str
    pool_or_split: str
    n_tasks: int
    ttft_ms_median: Optional[float]
    prefill_throughput_tps_median: Optional[float]
    decode_throughput_tps_median: Optional[float]
    cache_hit_rate_pct_median: Optional[float]
    ttft_ms_p25: Optional[float]
    ttft_ms_p75: Optional[float]
    prefill_throughput_tps_p25: Optional[float]
    prefill_throughput_tps_p75: Optional[float]
    decode_throughput_tps_p25: Optional[float]
    decode_throughput_tps_p75: Optional[float]
    cache_hit_rate_pct_p25: Optional[float]
    cache_hit_rate_pct_p75: Optional[float]
    total_gen_tokens: float
    total_prompt_tokens: float
    total_wall_clock_s: float
    total_turns: int


@dataclass
class TelemetryConfig:
    """Configuration for the LatencyCapture component."""

    vllm_host: str = "127.0.0.1"
    vllm_port: int = 8000
    output_dir: str = "output"
    metrics_fetch_timeout_s: float = 5.0
    default_exclude_anomalies: set[str] = field(default_factory=lambda: {"orphaned_before", "negative_delta:*"})


class TelemetryGapError(Exception):
    """Raised when reportable runs are missing telemetry records."""

    def __init__(self, missing_keys: set[tuple[str, str, int, int]]):
        self.missing_keys = missing_keys
        super().__init__(
            f"{len(missing_keys)} reportable runs have no telemetry record. "
            f"First 5: {sorted(missing_keys)[:5]}. "
            "This blocks Contribution A aggregation."
        )


def _matches_exclusion(anomaly: str, exclusions: set[str]) -> bool:
    """Check if an anomaly matches any exclusion pattern."""

    for exclusion in exclusions:
        if exclusion.endswith("*"):
            if anomaly.startswith(exclusion[:-1]):
                return True
        elif anomaly == exclusion:
            return True
    return False


DEFAULT_EXCLUDE_ANOMALIES = {"orphaned_before", "negative_delta:*"}


class TelemetryWriter:
    """Append-only JSONL writer for telemetry records."""

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

        with open(self._path, "a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


async def fetch_metrics(host: str, port: int, timeout: float = 5.0, headers: dict[str, str] | None = None) -> dict[str, float]:
    """Fetch and parse vLLM /metrics in a background thread."""

    def _request() -> str:
        response = requests.get(f"http://{host}:{port}/metrics", headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text

    raw = await asyncio.to_thread(_request)
    return parse_prometheus_text(raw)


def _snapshot_zero_dict(snapshot: dict[str, float]) -> dict[str, float]:
    return {key: 0.0 for key in snapshot}


class SnapshotManager:
    """Manages the before/after snapshot lifecycle for one vLLM instance."""

    def __init__(
        self,
        vllm_host: str,
        vllm_port: int,
        schema: dict[str, str],
        model_id: str,
        track: str,
        pool_or_split: str,
        *,
        metrics_fetch_timeout_s: float = 5.0,
    ) -> None:
        self._host = vllm_host
        self._port = vllm_port
        self._schema = schema
        self._model_id = model_id
        self._track = track
        self._pool_or_split = pool_or_split
        self._metrics_fetch_timeout_s = metrics_fetch_timeout_s
        self._pending: PendingSnapshot | None = None

    async def capture_before(self, task_id: str, seed: int, attempt: int) -> None:
        if self._pending is not None:
            raise RuntimeError(
                f"Pending snapshot exists for {self._pending.task_id} but snapshot_before called for {task_id}. "
                "LLD-03 must call snapshot_after() before starting a new task."
            )
        snapshot = await fetch_metrics(self._host, self._port, timeout=self._metrics_fetch_timeout_s)
        self._pending = PendingSnapshot(
            task_id=task_id,
            seed=seed,
            attempt=attempt,
            snapshot=snapshot,
            timestamp=time.monotonic(),
        )

    async def capture_after(self, task_id: str) -> tuple[PendingSnapshot, dict[str, float], float, list[str]]:
        after_snapshot = await fetch_metrics(self._host, self._port, timeout=self._metrics_fetch_timeout_s)
        after_ts = time.monotonic()
        anomalies: list[str] = []

        if self._pending is None or self._pending.task_id != task_id:
            anomalies.append("orphaned_before")
            synthetic_before = PendingSnapshot(
                task_id=task_id,
                seed=0,
                attempt=0,
                snapshot=_snapshot_zero_dict(after_snapshot),
                timestamp=after_ts,
            )
            return synthetic_before, after_snapshot, after_ts, anomalies

        pending = self._pending
        self._pending = None
        return pending, after_snapshot, after_ts, anomalies


def _build_task_metrics(
    pending: PendingSnapshot,
    after: dict[str, float],
    after_ts: float,
    schema: dict[str, str],
    model_id: str,
    track: str,
    pool_or_split: str,
    *,
    initial_anomalies: Optional[list[str]] = None,
) -> TaskMetrics:
    before = pending.snapshot
    anomalies = list(initial_anomalies or [])

    def safe_delta(key: str) -> float:
        delta, anomaly = _delta(before, after, key)
        if anomaly is not None:
            anomalies.append(anomaly)
        return delta

    prompt_tokens = safe_delta(_schema_value(schema, "prompt_tokens"))
    kv_computed_tokens = safe_delta(f"{_schema_value(schema, 'kv_computed_tokens', 'kv_computed_tokens_sum')}_sum")
    gen_tokens = safe_delta(_schema_value(schema, "generation_tokens"))
    cache_queries = safe_delta(_schema_value(schema, "cache_queries", "prefix_cache_queries"))
    cache_hits = safe_delta(_schema_value(schema, "cache_hits", "prefix_cache_hits"))
    ttft_key = _schema_value(schema, "ttft", "ttft_seconds")
    prefill_key = _schema_value(schema, "prefill_time", "prefill_seconds")
    decode_key = _schema_value(schema, "decode_time", "decode_seconds")

    ttft_sum_s = safe_delta(f"{ttft_key}_sum")
    ttft_count = safe_delta(f"{ttft_key}_count")
    prefill_sum_s = safe_delta(f"{prefill_key}_sum")
    decode_sum_s = safe_delta(f"{decode_key}_sum")

    if gen_tokens == 0 and "orphaned_before" not in anomalies:
        anomalies.append("zero_gen_tokens")
    if kv_computed_tokens == 0 and prompt_tokens == 0 and "orphaned_before" not in anomalies:
        anomalies.append("zero_prefill_tokens")

    ttft_ms = (ttft_sum_s / ttft_count * 1000) if ttft_count > 0 else None
    prefill_tps = kv_computed_tokens / prefill_sum_s if prefill_sum_s > 0 else None
    decode_tps = gen_tokens / decode_sum_s if decode_sum_s > 0 else None
    cache_pct = cache_hits / cache_queries * 100 if cache_queries > 0 else None
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
        ttft_sum_s=ttft_sum_s,
        ttft_count=int(ttft_count),
        cache_queries=cache_queries,
        cache_hits=cache_hits,
        snapshot_before_ts=pending.timestamp,
        snapshot_after_ts=after_ts,
        wall_clock_s=wall_clock_s,
        anomalies=anomalies,
    )


def compute_task_metrics(*args: Any, **kwargs: Any) -> Any:
    """Compute per-task telemetry from either legacy or LLD-04 call styles."""

    if "before" in kwargs or {"before", "after", "schema"} <= kwargs.keys():
        before = kwargs.pop("before")
        after = kwargs.pop("after")
        schema = kwargs.pop("schema")
        if not isinstance(before, dict) or not isinstance(after, dict) or not isinstance(schema, dict):
            raise TypeError("compute_task_metrics(before=..., after=..., schema=...) expects dict inputs")
        return _task_metrics_from_snapshots(before, after, schema)

    if len(args) == 3 and not kwargs:
        before, after, schema = args
        if not isinstance(before, dict) or not isinstance(after, dict) or not isinstance(schema, dict):
            raise TypeError("compute_task_metrics(before, after, schema) expects dict inputs")
        return _task_metrics_from_snapshots(before, after, schema)

    if args and isinstance(args[0], PendingSnapshot):
        pending = args[0]
        try:
            after = args[1]
            after_ts = args[2]
            schema = args[3]
        except IndexError as exc:
            raise TypeError("LLD-04 compute_task_metrics call requires pending, after, after_ts, schema") from exc
        if len(args) >= 7:
            model_id = args[4]
            track = args[5]
            pool_or_split = args[6]
            initial_anomalies = args[7] if len(args) >= 8 else kwargs.pop("initial_anomalies", None)
        else:
            model_id = kwargs.pop("model_id")
            track = kwargs.pop("track")
            pool_or_split = kwargs.pop("pool_or_split")
            initial_anomalies = kwargs.pop("initial_anomalies", None)
        if not isinstance(after, dict) or not isinstance(schema, dict):
            raise TypeError("LLD-04 compute_task_metrics call expects after and schema to be dicts")
        if not isinstance(after_ts, (int, float)):
            raise TypeError("after_ts must be a number")
        return _build_task_metrics(
            pending,
            after,
            float(after_ts),
            schema,
            model_id,
            track,
            pool_or_split,
            initial_anomalies=initial_anomalies,
        )

    if "pending" in kwargs:
        pending = kwargs.pop("pending")
        after = kwargs.pop("after")
        after_ts = kwargs.pop("after_ts")
        schema = kwargs.pop("schema")
        model_id = kwargs.pop("model_id")
        track = kwargs.pop("track")
        pool_or_split = kwargs.pop("pool_or_split")
        initial_anomalies = kwargs.pop("initial_anomalies", None)
        return _build_task_metrics(
            pending,
            after,
            float(after_ts),
            schema,
            model_id,
            track,
            pool_or_split,
            initial_anomalies=initial_anomalies,
        )

    raise TypeError("Unsupported compute_task_metrics call signature")


def extract_turns(trajectory_path: str) -> list[TurnInfo]:
    """Parse trajectory JSONL to extract coarse turn boundaries."""

    turns: list[TurnInfo] = []
    current_turn_index = 0
    current_output_chars = 0
    current_tool_calls = 0
    current_start: Optional[str] = None

    def _event_type(event: dict[str, Any]) -> str:
        event_type = event.get("type")
        return event_type if isinstance(event_type, str) else ""

    def _event_timestamp(event: dict[str, Any]) -> Optional[str]:
        timestamp = event.get("timestamp")
        return timestamp if isinstance(timestamp, str) else None

    def _is_assistant_event(event: dict[str, Any]) -> bool:
        role = event.get("role")
        return not isinstance(role, str) or role == "assistant"

    def _extract_text(event: dict[str, Any]) -> str:
        for key in ("text", "message", "content"):
            value = event.get(key)
            if isinstance(value, str):
                return value
        delta = event.get("delta")
        if isinstance(delta, dict):
            for key in ("text", "content", "message"):
                value = delta.get(key)
                if isinstance(value, str):
                    return value
        if isinstance(event.get("content"), list):
            pieces: list[str] = []
            for item in event["content"]:
                if isinstance(item, dict):
                    value = item.get("text")
                    if isinstance(value, str):
                        pieces.append(value)
            return "".join(pieces)
        return ""

    start_types = {
        "message",
        "response.created",
        "response_created",
        "assistant_message",
        "response.output_text.start",
        "message_start",
    }
    output_types = {
        "message_delta",
        "response.output_text.delta",
        "content_block_delta",
        "assistant_message",
        "response.output_text.done",
    }
    tool_types = {
        "tool_call",
        "response.function_call_arguments.done",
        "function_call",
        "tool_calls",
    }

    with open(trajectory_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue

            event_type = _event_type(event)
            timestamp = _event_timestamp(event)

            if event_type in start_types and _is_assistant_event(event):
                if current_start is not None and current_output_chars > 0:
                    turns.append(
                        TurnInfo(
                            turn_index=current_turn_index,
                            start_timestamp=current_start,
                            end_timestamp=timestamp,
                            output_tokens_approx=current_output_chars // 4,
                            tool_calls=current_tool_calls,
                        )
                    )
                    current_turn_index += 1
                if timestamp is not None:
                    current_start = timestamp
                current_output_chars = 0
                current_tool_calls = 0

            if event_type in output_types and _is_assistant_event(event):
                current_output_chars += len(_extract_text(event))
            elif (
                any(key in event for key in ("text", "message", "content"))
                and event_type not in tool_types
                and _is_assistant_event(event)
            ):
                if current_start is None:
                    current_start = timestamp
                current_output_chars += len(_extract_text(event))

            if event_type in tool_types:
                current_tool_calls += 1
            elif isinstance(event.get("tool_calls"), list):
                current_tool_calls += len(event["tool_calls"])
            elif isinstance(event.get("tool_call"), dict):
                current_tool_calls += 1

    if current_start is not None:
        turns.append(
            TurnInfo(
                turn_index=current_turn_index,
                start_timestamp=current_start,
                end_timestamp=None,
                output_tokens_approx=current_output_chars // 4,
                tool_calls=current_tool_calls,
            )
        )

    return turns


def load_telemetry(
    telemetry_dir: str,
    exclude_anomalies: Optional[set[str]] = None,
) -> list[LatencyRecord]:
    if exclude_anomalies is None:
        exclude_anomalies = DEFAULT_EXCLUDE_ANOMALIES

    records: list[LatencyRecord] = []
    pattern = os.path.join(telemetry_dir, "latency_*.jsonl")
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                task_anomalies = data.get("anomalies") or []
                if not isinstance(task_anomalies, list):
                    task_anomalies = [str(task_anomalies)]
                if any(_matches_exclusion(anomaly, exclude_anomalies) for anomaly in task_anomalies):
                    continue
                records.append(
                    LatencyRecord(
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
                    )
                )
    return records


def _percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    floor_k = int(k)
    ceil_k = min(floor_k + 1, len(values) - 1)
    frac = k - floor_k
    return values[floor_k] + frac * (values[ceil_k] - values[floor_k])


def aggregate_by_model(
    records: list[LatencyRecord],
    reportable_runs: set[tuple[str, str, int, int]],
) -> list[ModelLatencySummary]:
    selected = [record for record in records if (record.task_id, record.model_id, record.seed, record.attempt) in reportable_runs]
    present_keys = {(record.task_id, record.model_id, record.seed, record.attempt) for record in selected}
    missing = reportable_runs - present_keys
    if missing:
        raise TelemetryGapError(missing)

    by_model: dict[str, list[LatencyRecord]] = defaultdict(list)
    for record in selected:
        by_model[record.model_id].append(record)

    summaries: list[ModelLatencySummary] = []
    for model_id, model_records in sorted(by_model.items()):
        ttfts = [record.ttft_ms for record in model_records if record.ttft_ms is not None]
        prefills = [record.prefill_throughput_tps for record in model_records if record.prefill_throughput_tps is not None]
        decodes = [record.decode_throughput_tps for record in model_records if record.decode_throughput_tps is not None]
        caches = [record.cache_hit_rate_pct for record in model_records if record.cache_hit_rate_pct is not None]

        summaries.append(
            ModelLatencySummary(
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
                total_gen_tokens=sum(record.gen_tokens for record in model_records),
                total_prompt_tokens=sum(record.prompt_tokens for record in model_records),
                total_wall_clock_s=sum(record.wall_clock_s for record in model_records),
                total_turns=sum(record.ttft_count for record in model_records),
            )
        )
    return summaries


class LatencyCapture:
    """Captures per-task latency telemetry from vLLM /metrics."""

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
        *,
        metrics_fetch_timeout_s: float = 5.0,
    ) -> None:
        self._host = vllm_host
        self._port = vllm_port
        self._model_id = model_id
        self._pool_or_split = pool_or_split
        self._track = self._derive_track(pool_or_split)
        self._metrics_fetch_timeout_s = metrics_fetch_timeout_s
        self._schema: dict[str, str] | None = None
        self._snapshot_mgr: SnapshotManager | None = None
        self._writer = TelemetryWriter(os.path.join(output_dir, "telemetry", f"latency_{model_id}_{pool_or_split}.jsonl"))

    @property
    def writer_path(self) -> str:
        return self._writer.path

    @property
    def resolved_schema(self) -> dict[str, str]:
        if self._schema is None:
            raise RuntimeError("resolve_schema() must be called before accessing resolved_schema")
        return dict(self._schema)

    async def resolve_schema(self) -> None:
        snapshot = await fetch_metrics(self._host, self._port, timeout=self._metrics_fetch_timeout_s)
        self._schema = resolve_metric_schema(snapshot)
        self._snapshot_mgr = SnapshotManager(
            self._host,
            self._port,
            self._schema,
            self._model_id,
            self._track,
            self._pool_or_split,
            metrics_fetch_timeout_s=self._metrics_fetch_timeout_s,
        )
        logger.info("Metric schema resolved: %s", self._schema)

    async def snapshot_before(self, task_id: str, seed: int, attempt: int) -> None:
        if self._snapshot_mgr is None:
            raise RuntimeError("resolve_schema() must be called before snapshot_before()")
        await self._snapshot_mgr.capture_before(task_id, seed, attempt)

    async def snapshot_after(self, task_id: str) -> TaskMetrics:
        if self._schema is None or self._snapshot_mgr is None:
            raise RuntimeError("resolve_schema() must be called before snapshot_after()")
        pending, after, after_ts, snapshot_anomalies = await self._snapshot_mgr.capture_after(task_id)
        metrics = compute_task_metrics(
            pending,
            after,
            after_ts,
            self._schema,
            model_id=self._model_id,
            track=self._track,
            pool_or_split=self._pool_or_split,
            initial_anomalies=snapshot_anomalies,
        )
        self._writer.write_record(metrics)
        return metrics


__all__ = [
    "DEFAULT_EXCLUDE_ANOMALIES",
    "LatencyCapture",
    "LatencyRecord",
    "ModelLatencySummary",
    "PendingSnapshot",
    "REQUIRED_METRIC_VARIANTS",
    "SnapshotManager",
    "TaskMetrics",
    "TelemetryConfig",
    "TelemetryGapError",
    "TelemetryWriter",
    "TurnInfo",
    "aggregate_by_model",
    "compute_task_metrics",
    "extract_turns",
    "fetch_metrics",
    "load_telemetry",
    "parse_prometheus_text",
    "resolve_metric_schema",
]

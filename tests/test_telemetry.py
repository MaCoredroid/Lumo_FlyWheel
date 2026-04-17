from __future__ import annotations

import asyncio
import json
from pathlib import Path

from lumo_flywheel_serving.metrics import (
    LatencyCapture,
    LatencyRecord,
    TaskMetrics,
    TelemetryGapError,
    TelemetryWriter,
    aggregate_by_model,
    extract_turns,
    load_telemetry,
)


def _task_metrics(**overrides):
    values = {
        "task_id": "family-a/v1",
        "model_id": "qwen3.5-27b",
        "track": "codex_long",
        "pool_or_split": "train_long",
        "seed": 7,
        "attempt": 2,
        "ttft_ms": 500.0,
        "prefill_throughput_tps": 12.0,
        "decode_throughput_tps": 18.0,
        "cache_hit_rate_pct": 83.3333333333,
        "prompt_tokens": 60.0,
        "kv_computed_tokens": 24.0,
        "gen_tokens": 18.0,
        "prefill_sum_s": 2.0,
        "decode_sum_s": 1.0,
        "ttft_sum_s": 1.5,
        "ttft_count": 3,
        "cache_queries": 6.0,
        "cache_hits": 5.0,
        "snapshot_before_ts": 10.0,
        "snapshot_after_ts": 14.5,
        "wall_clock_s": 4.5,
        "anomalies": [],
    }
    values.update(overrides)
    return TaskMetrics(**values)


def test_telemetry_writer_and_loader_filter_anomalies(tmp_path: Path) -> None:
    writer = TelemetryWriter(str(tmp_path / "telemetry" / "latency_qwen3.5-27b_train-long.jsonl"))
    writer.write_record(_task_metrics())
    writer.write_record(_task_metrics(task_id="family-a/v2", anomalies=["orphaned_before"]))

    records = load_telemetry(str(tmp_path / "telemetry"))

    assert len(records) == 1
    assert records[0].task_id == "family-a/v1"
    assert records[0].ttft_count == 3


def test_aggregate_by_model_summarizes_records() -> None:
    records = [
        LatencyRecord(
            task_id="family-a/v1",
            model_id="qwen3.5-27b",
            track="codex_long",
            pool_or_split="train_long",
            seed=7,
            attempt=2,
            ttft_ms=500.0,
            prefill_throughput_tps=12.0,
            decode_throughput_tps=18.0,
            cache_hit_rate_pct=80.0,
            gen_tokens=18.0,
            kv_computed_tokens=24.0,
            prompt_tokens=60.0,
            wall_clock_s=4.5,
            ttft_count=3,
            anomalies=[],
        ),
        LatencyRecord(
            task_id="family-a/v2",
            model_id="qwen3.5-27b",
            track="codex_long",
            pool_or_split="train_long",
            seed=8,
            attempt=1,
            ttft_ms=700.0,
            prefill_throughput_tps=10.0,
            decode_throughput_tps=15.0,
            cache_hit_rate_pct=90.0,
            gen_tokens=20.0,
            kv_computed_tokens=30.0,
            prompt_tokens=70.0,
            wall_clock_s=5.5,
            ttft_count=4,
            anomalies=[],
        ),
    ]

    summaries = aggregate_by_model(records, {("family-a/v1", "qwen3.5-27b", 7, 2), ("family-a/v2", "qwen3.5-27b", 8, 1)})

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.n_tasks == 2
    assert summary.model_id == "qwen3.5-27b"
    assert summary.total_gen_tokens == 38.0
    assert summary.total_turns == 7
    assert summary.ttft_ms_median == 600.0


def test_aggregate_by_model_raises_on_missing_telemetry() -> None:
    records = []
    reportable_runs = {("family-a/v1", "qwen3.5-27b", 7, 2)}

    try:
        aggregate_by_model(records, reportable_runs)
    except TelemetryGapError as exc:
        assert exc.missing_keys == reportable_runs
    else:
        raise AssertionError("Expected aggregate_by_model() to raise TelemetryGapError")


def test_extract_turns_parses_codex_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"type": "run_metadata", "timestamp": "2026-04-17T00:00:00Z"}),
                json.dumps({"type": "response.created", "timestamp": "2026-04-17T00:00:01Z"}),
                json.dumps({"type": "message_delta", "delta": {"text": "hello world"}}),
                json.dumps({"type": "tool_call"}),
                json.dumps({"type": "response.created", "timestamp": "2026-04-17T00:00:02Z"}),
                json.dumps({"type": "assistant_message", "message": "second turn"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    turns = extract_turns(str(path))

    assert len(turns) == 2
    assert turns[0].tool_calls == 1
    assert turns[0].output_tokens_approx > 0
    assert turns[1].output_tokens_approx > 0


def test_extract_turns_ignores_non_assistant_message_events(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"type": "message", "role": "user", "timestamp": "2026-04-17T00:00:00Z"}),
                json.dumps({"type": "response.created", "timestamp": "2026-04-17T00:00:01Z"}),
                json.dumps({"type": "message_delta", "delta": {"text": "assistant reply"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    turns = extract_turns(str(path))

    assert len(turns) == 1
    assert turns[0].tool_calls == 0
    assert turns[0].output_tokens_approx > 0


def test_latency_capture_lifecycle_writes_jsonl(tmp_path: Path, monkeypatch) -> None:
    snapshots = iter(
        [
            {
                "vllm:prompt_tokens_total": 100.0,
                "vllm:generation_tokens_total": 40.0,
                "vllm:request_prefill_kv_computed_tokens_sum": 80.0,
                "vllm:prefix_cache_queries_total": 10.0,
                "vllm:prefix_cache_hits_total": 5.0,
                "vllm:time_to_first_token_seconds_sum": 2.0,
                "vllm:time_to_first_token_seconds_count": 4.0,
                "vllm:request_prefill_time_seconds_sum": 3.0,
                "vllm:request_decode_time_seconds_sum": 2.0,
                "vllm:inter_token_latency_seconds_sum": 1.0,
                "vllm:inter_token_latency_seconds_count": 1.0,
            },
            {
                "vllm:prompt_tokens_total": 100.0,
                "vllm:generation_tokens_total": 40.0,
                "vllm:request_prefill_kv_computed_tokens_sum": 80.0,
                "vllm:prefix_cache_queries_total": 10.0,
                "vllm:prefix_cache_hits_total": 5.0,
                "vllm:time_to_first_token_seconds_sum": 2.0,
                "vllm:time_to_first_token_seconds_count": 4.0,
                "vllm:request_prefill_time_seconds_sum": 3.0,
                "vllm:request_decode_time_seconds_sum": 2.0,
                "vllm:inter_token_latency_seconds_sum": 1.0,
                "vllm:inter_token_latency_seconds_count": 1.0,
            },
            {
                "vllm:prompt_tokens_total": 160.0,
                "vllm:generation_tokens_total": 58.0,
                "vllm:request_prefill_kv_computed_tokens_sum": 104.0,
                "vllm:prefix_cache_queries_total": 16.0,
                "vllm:prefix_cache_hits_total": 10.0,
                "vllm:time_to_first_token_seconds_sum": 3.5,
                "vllm:time_to_first_token_seconds_count": 7.0,
                "vllm:request_prefill_time_seconds_sum": 5.0,
                "vllm:request_decode_time_seconds_sum": 3.0,
                "vllm:inter_token_latency_seconds_sum": 1.5,
                "vllm:inter_token_latency_seconds_count": 2.0,
            },
        ]
    )

    async def fake_fetch_metrics(*args, **kwargs):
        return next(snapshots)

    monkeypatch.setattr("lumo_flywheel_serving.metrics.fetch_metrics", fake_fetch_metrics)

    capture = LatencyCapture(
        "127.0.0.1",
        8000,
        str(tmp_path / "output"),
        "qwen3.5-27b",
        "train_long",
    )

    asyncio.run(capture.resolve_schema())
    asyncio.run(capture.snapshot_before("family-a/v1", seed=7, attempt=2))
    metrics = asyncio.run(capture.snapshot_after("family-a/v1"))

    assert metrics.task_id == "family-a/v1"
    assert metrics.seed == 7
    assert metrics.attempt == 2
    assert metrics.ttft_ms == 500.0

    contents = (tmp_path / "output" / "telemetry" / "latency_qwen3.5-27b_train_long.jsonl").read_text(encoding="utf-8").strip()
    record = json.loads(contents)
    assert record["seed"] == 7
    assert record["attempt"] == 2
    assert record["ttft_ms"] == 500.0

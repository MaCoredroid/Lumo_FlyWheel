from __future__ import annotations

import json
import sys
from argparse import Namespace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_track_b_e2e_summary import (  # noqa: E402
    SAMPLE_HASH,
    TRACK_B_E2E_TASKS,
    _dcgm_profile_field_status,
    _load_vllm_request_metrics,
    _validate_runtime_config_hash,
    _vllm_jsonl_by_request,
    build_round_summary,
    build_task_summary,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _dcgm_sample(ts: str) -> dict[str, object]:
    return {
        "ts": ts,
        "gpu": 0,
        "runtime_config_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "profile_fields_available": True,
        "dram_active_pct": 0.54,
        "sm_active_pct": 0.31,
        "sm_occupancy_pct": 0.27,
        "pipe_tensor_active_pct": 0.18,
        "pipe_fp16_active_pct": 0.04,
    }


def test_summary_rejects_unstamped_runtime_hash() -> None:
    _validate_runtime_config_hash("sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    with pytest.raises(ValueError, match="runtime-config-hash"):
        _validate_runtime_config_hash("not-a-runtime-hash")
    with pytest.raises(ValueError, match="runtime-config-hash"):
        _validate_runtime_config_hash("sha256:")


def test_dcgm_profile_status_requires_available_flag() -> None:
    sample = _dcgm_sample("2026-05-07T18:00:00.000Z")
    sample["profile_fields_available"] = False

    status = _dcgm_profile_field_status([sample])

    assert status["ok"] is False
    assert status["missing_profile_fields"] == []
    assert status["profile_fields_available_sample_count"] == 0
    assert status["profile_fields_available_missing"] is True


def test_task_summary_requires_and_records_truthful_attestation(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    base = datetime(2026, 5, 7, 18, 0, 0, tzinfo=UTC)

    def ts(offset_s: float) -> str:
        return (base + timedelta(seconds=offset_s)).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    _write_jsonl(
        task_dir / "codex_trace.jsonl",
        [
            {"event": "task_start", "ts": ts(0.0), "task_id": "transcript-merge-regression/v1-clean-baseline"},
            {"event": "turn_start", "turn": 0, "regime": "prefill", "ts": ts(0.1), "vllm_request_id": "req-0"},
            {"event": "turn_end", "turn": 0, "ts": ts(0.5), "prompt_tokens": 50, "completion_tokens": 0},
            {"event": "turn_start", "turn": 1, "regime": "plan", "ts": ts(0.5), "vllm_request_id": "req-1"},
            {"event": "turn_end", "turn": 1, "ts": ts(1.5), "prompt_tokens": 50, "completion_tokens": 12, "max_tokens": 128},
            {
                "event": "tool_call",
                "turn": 2,
                "name": "read_file",
                "ts_codex_emit_start": ts(1.5),
                "ts_codex_emit_end": ts(1.6),
                "ts_tool_exec_end": ts(1.7),
            },
            {"event": "task_end", "ts": ts(2.0), "exit_code": 0, "task_score": 0.74, "wallclock_s": 2.0},
        ],
    )
    (task_dir / "vllm_per_turn.json").write_text(
        json.dumps(
            {
                "requests": {
                    "req-0": {"prompt_tokens": 50, "completion_tokens": 0, "prefill_sum_s": 0.4},
                    "req-1": {
                        "prompt_tokens": 50,
                        "completion_tokens": 12,
                        "decode_sum_s": 1.0,
                        "decode_tps": 12.0,
                        "spec_decode_num_accepted_tokens": 3,
                        "spec_decode_num_draft_tokens": 12,
                        "accepted_per_draft_token": 0.25,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    samples = [_dcgm_sample(ts(index / 100)) for index in range(200)]
    _write_jsonl(task_dir / "dcgm_samples.jsonl", samples)

    summary = build_task_summary(
        Namespace(
            round=0,
            task_dir=str(task_dir),
            family="transcript-merge-regression",
            variant="v1-clean-baseline",
            runtime_config_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            baseline_workspace_hash=None,
            median_of_n_runs=3,
            run_wallclocks_json="[1.9, 2.0, 2.1]",
            clock_skew_ms_p99=8,
            trace_emitter_correctness_verified_at="2026-05-07T14:00:00Z",
            dcgm_interval_s=0.01,
            cold_completion_discarded=True,
            cache_reset_verified=True,
            protocol_hash_match=True,
            generation_volume_within_band=True,
            sample_hash_match=True,
            write_untrusted_diagnostic=False,
        )
    )

    assert summary["trusted_measurement"] is True
    assert summary["sample_hash"] == SAMPLE_HASH
    assert summary["wallclock_s"] == 2.0
    assert summary["bottleneck_regime"] == "plan"
    assert summary["truthful_measurement_attestation"]["rule_12_spec_decode_metrics_present"] is True
    assert summary["truthful_measurement_attestation"]["rule_6_dcgm_missing_profile_fields"] == []
    assert (task_dir / "summary.json").is_file()


def test_task_summary_single_run_is_diagnostic_only(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    base = datetime(2026, 5, 7, 18, 0, 0, tzinfo=UTC)

    def ts(offset_s: float) -> str:
        return (base + timedelta(seconds=offset_s)).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    _write_jsonl(
        task_dir / "codex_trace.jsonl",
        [
            {"event": "task_start", "ts": ts(0.0), "task_id": "transcript-merge-regression/v1-clean-baseline"},
            {"event": "turn_start", "turn": 0, "regime": "plan", "ts": ts(0.1), "vllm_request_id": "req-1"},
            {"event": "turn_end", "turn": 0, "ts": ts(1.0), "prompt_tokens": 50, "completion_tokens": 12},
            {"event": "task_end", "ts": ts(2.0), "exit_code": 0, "task_score": 0.74, "wallclock_s": 2.0},
        ],
    )
    (task_dir / "vllm_per_turn.json").write_text(
        json.dumps(
            {
                "requests": {
                    "req-1": {
                        "completion_tokens": 12,
                        "decode_sum_s": 0.9,
                        "decode_tps": 13.33,
                        "spec_decode_num_accepted_tokens": 3,
                        "spec_decode_num_draft_tokens": 12,
                        "accepted_per_draft_token": 0.25,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        task_dir / "dcgm_samples.jsonl",
        [_dcgm_sample(ts(index / 100)) for index in range(200)],
    )

    summary = build_task_summary(
        Namespace(
            round=0,
            task_dir=str(task_dir),
            family="transcript-merge-regression",
            variant="v1-clean-baseline",
            runtime_config_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            baseline_workspace_hash=None,
            run_wallclocks_json="",
            clock_skew_ms_p99=8,
            trace_emitter_correctness_verified_at="2026-05-07T14:00:00Z",
            dcgm_interval_s=0.01,
            cold_completion_discarded=True,
            cache_reset_verified=True,
            protocol_hash_match=True,
            generation_volume_within_band=True,
            sample_hash_match=True,
            write_untrusted_diagnostic=True,
        )
    )

    assert summary["trusted_measurement"] is False
    assert summary["truthful_measurement_attestation"]["rule_3_median_of_n_runs"] == 1


def test_task_summary_rejects_trace_task_id_mismatch(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    base = datetime(2026, 5, 7, 18, 0, 0, tzinfo=UTC)

    def ts(offset_s: float) -> str:
        return (base + timedelta(seconds=offset_s)).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    _write_jsonl(
        task_dir / "codex_trace.jsonl",
        [
            {"event": "task_start", "ts": ts(0.0), "task_id": "transcript-merge-regression/v1-clean-baseline"},
            {"event": "turn_start", "turn": 1, "regime": "plan", "ts": ts(0.5), "vllm_request_id": "req-1"},
            {"event": "turn_end", "turn": 1, "ts": ts(1.5), "completion_tokens": 12},
            {
                "event": "task_end",
                "ts": ts(2.0),
                "task_id": "transcript-merge-regression/v1-clean-baseline",
                "exit_code": 0,
                "task_score": 0.74,
                "wallclock_s": 2.0,
            },
        ],
    )
    (task_dir / "vllm_per_turn.json").write_text(
        json.dumps(
            {
                "requests": {
                    "req-1": {
                        "prompt_tokens": 50,
                        "completion_tokens": 12,
                        "decode_sum_s": 1.0,
                        "spec_decode_num_accepted_tokens": 3,
                        "spec_decode_num_draft_tokens": 12,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="task_id"):
        build_task_summary(
            Namespace(
                round=0,
                task_dir=str(task_dir),
                family="dead-flag-reachability-audit",
                variant="v1-clean-baseline",
                runtime_config_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                baseline_workspace_hash=None,
                run_wallclocks_json="[1.9, 2.0, 2.1]",
                clock_skew_ms_p99=8,
                trace_emitter_correctness_verified_at="2026-05-07T14:00:00Z",
                dcgm_interval_s=0.01,
                cold_completion_discarded=True,
                cache_reset_verified=True,
                protocol_hash_match=True,
                generation_volume_within_band=True,
                sample_hash_match=True,
                write_untrusted_diagnostic=False,
            )
        )


def test_task_summary_rejects_runtime_config_hash_mismatch(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    base = datetime(2026, 5, 7, 18, 0, 0, tzinfo=UTC)

    def ts(offset_s: float) -> str:
        return (base + timedelta(seconds=offset_s)).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    _write_jsonl(
        task_dir / "codex_trace.jsonl",
        [
            {"event": "task_start", "ts": ts(0.0), "task_id": "transcript-merge-regression/v1-clean-baseline"},
            {"event": "turn_start", "turn": 1, "regime": "plan", "ts": ts(0.5), "vllm_request_id": "req-1"},
            {"event": "turn_end", "turn": 1, "ts": ts(1.5), "completion_tokens": 12},
            {
                "event": "task_end",
                "ts": ts(2.0),
                "task_id": "transcript-merge-regression/v1-clean-baseline",
                "exit_code": 0,
                "task_score": 0.74,
                "wallclock_s": 2.0,
            },
        ],
    )
    (task_dir / "vllm_per_turn.json").write_text(
        json.dumps(
            {
                "runtime_config_hash": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "requests": {
                    "req-1": {
                        "prompt_tokens": 50,
                        "completion_tokens": 12,
                        "decode_sum_s": 1.0,
                        "spec_decode_num_accepted_tokens": 3,
                        "spec_decode_num_draft_tokens": 12,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="runtime_config_hash mismatch"):
        build_task_summary(
            Namespace(
                round=0,
                task_dir=str(task_dir),
                family="transcript-merge-regression",
                variant="v1-clean-baseline",
                runtime_config_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                baseline_workspace_hash=None,
                run_wallclocks_json="[1.9, 2.0, 2.1]",
                clock_skew_ms_p99=8,
                trace_emitter_correctness_verified_at="2026-05-07T14:00:00Z",
                dcgm_interval_s=0.01,
                cold_completion_discarded=True,
                cache_reset_verified=True,
                protocol_hash_match=True,
                generation_volume_within_band=True,
                sample_hash_match=True,
                write_untrusted_diagnostic=False,
            )
        )


def test_task_summary_rejects_dcgm_runtime_config_hash_mismatch(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    base = datetime(2026, 5, 7, 18, 0, 0, tzinfo=UTC)

    def ts(offset_s: float) -> str:
        return (base + timedelta(seconds=offset_s)).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    _write_jsonl(
        task_dir / "codex_trace.jsonl",
        [
            {"event": "task_start", "ts": ts(0.0), "task_id": "transcript-merge-regression/v1-clean-baseline"},
            {"event": "turn_start", "turn": 1, "regime": "plan", "ts": ts(0.5), "vllm_request_id": "req-1"},
            {"event": "turn_end", "turn": 1, "ts": ts(1.5), "completion_tokens": 12},
            {
                "event": "task_end",
                "ts": ts(2.0),
                "task_id": "transcript-merge-regression/v1-clean-baseline",
                "exit_code": 0,
                "task_score": 0.74,
                "wallclock_s": 2.0,
            },
        ],
    )
    (task_dir / "vllm_per_turn.json").write_text(
        json.dumps(
            {
                "runtime_config_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "requests": {
                    "req-1": {
                        "prompt_tokens": 50,
                        "completion_tokens": 12,
                        "decode_sum_s": 1.0,
                        "spec_decode_num_accepted_tokens": 3,
                        "spec_decode_num_draft_tokens": 12,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    sample = _dcgm_sample(ts(1.0))
    sample["runtime_config_hash"] = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    _write_jsonl(task_dir / "dcgm_samples.jsonl", [sample])

    with pytest.raises(RuntimeError, match="dcgm_samples.jsonl"):
        build_task_summary(
            Namespace(
                round=0,
                task_dir=str(task_dir),
                family="transcript-merge-regression",
                variant="v1-clean-baseline",
                runtime_config_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                baseline_workspace_hash=None,
                run_wallclocks_json="[1.9, 2.0, 2.1]",
                clock_skew_ms_p99=8,
                trace_emitter_correctness_verified_at="2026-05-07T14:00:00Z",
                dcgm_interval_s=0.01,
                cold_completion_discarded=True,
                cache_reset_verified=True,
                protocol_hash_match=True,
                generation_volume_within_band=True,
                sample_hash_match=True,
                write_untrusted_diagnostic=False,
            )
        )


def test_task_summary_rejects_trace_runtime_config_hash_mismatch(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    base = datetime(2026, 5, 7, 18, 0, 0, tzinfo=UTC)

    def ts(offset_s: float) -> str:
        return (base + timedelta(seconds=offset_s)).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    _write_jsonl(
        task_dir / "codex_trace.jsonl",
        [
            {
                "event": "task_start",
                "ts": ts(0.0),
                "task_id": "transcript-merge-regression/v1-clean-baseline",
                "runtime_config_hash": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            },
            {"event": "turn_start", "turn": 1, "regime": "plan", "ts": ts(0.5), "vllm_request_id": "req-1"},
            {"event": "turn_end", "turn": 1, "ts": ts(1.5), "completion_tokens": 12},
            {
                "event": "task_end",
                "ts": ts(2.0),
                "task_id": "transcript-merge-regression/v1-clean-baseline",
                "exit_code": 0,
                "task_score": 0.74,
                "wallclock_s": 2.0,
            },
        ],
    )
    (task_dir / "vllm_per_turn.json").write_text(
        json.dumps(
            {
                "runtime_config_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "requests": {
                    "req-1": {
                        "prompt_tokens": 50,
                        "completion_tokens": 12,
                        "decode_sum_s": 1.0,
                        "spec_decode_num_accepted_tokens": 3,
                        "spec_decode_num_draft_tokens": 12,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="codex_trace.jsonl"):
        build_task_summary(
            Namespace(
                round=0,
                task_dir=str(task_dir),
                family="transcript-merge-regression",
                variant="v1-clean-baseline",
                runtime_config_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                baseline_workspace_hash=None,
                run_wallclocks_json="[1.9, 2.0, 2.1]",
                clock_skew_ms_p99=8,
                trace_emitter_correctness_verified_at="2026-05-07T14:00:00Z",
                dcgm_interval_s=0.01,
                cold_completion_discarded=True,
                cache_reset_verified=True,
                protocol_hash_match=True,
                generation_volume_within_band=True,
                sample_hash_match=True,
                write_untrusted_diagnostic=False,
            )
        )


def test_task_summary_accepts_vllm_request_metrics_jsonl_side_channel(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    base = datetime(2026, 5, 7, 18, 0, 0, tzinfo=UTC)

    def ts(offset_s: float) -> str:
        return (base + timedelta(seconds=offset_s)).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    _write_jsonl(
        task_dir / "codex_trace.jsonl",
        [
            {"event": "task_start", "ts": ts(0.0), "task_id": "transcript-merge-regression/v1-clean-baseline"},
            {"event": "turn_start", "turn": 0, "regime": "prefill", "ts": ts(0.1), "vllm_request_id": "req-0"},
            {"event": "turn_end", "turn": 0, "ts": ts(0.5), "prompt_tokens": 50, "completion_tokens": 0},
            {"event": "turn_start", "turn": 1, "regime": "plan", "ts": ts(0.5), "vllm_request_id": "req-1"},
            {"event": "turn_end", "turn": 1, "ts": ts(1.5), "prompt_tokens": 50, "completion_tokens": 12},
            {"event": "task_end", "ts": ts(2.0), "exit_code": 0, "task_score": 0.74, "wallclock_s": 2.0},
        ],
    )
    _write_jsonl(
        task_dir / "vllm_request_metrics.jsonl",
        [
            {
                "request_id": "req-0",
                "runtime_config_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "prompt_tokens": 50,
                "generation_tokens": 0,
                "prefill_s": 0.4,
                "decode_s": 0.0,
                "spec_decode_num_accepted_tokens": 0,
                "spec_decode_num_draft_tokens": 0,
            },
            {
                "request_id": "req-1",
                "runtime_config_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "prompt_tokens": 50,
                "generation_tokens": 12,
                "prefill_s": 0.2,
                "decode_s": 1.0,
                "spec_decode_num_accepted_tokens": 3,
                "spec_decode_num_draft_tokens": 12,
            },
        ],
    )
    _write_jsonl(
        task_dir / "dcgm_samples.jsonl",
        [_dcgm_sample(ts(index / 100)) for index in range(200)],
    )

    summary = build_task_summary(
        Namespace(
            round=0,
            task_dir=str(task_dir),
            family="transcript-merge-regression",
            variant="v1-clean-baseline",
            runtime_config_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            baseline_workspace_hash=None,
            run_wallclocks_json="[1.9, 2.0, 2.1]",
            clock_skew_ms_p99=8,
            trace_emitter_correctness_verified_at="2026-05-07T14:00:00Z",
            dcgm_interval_s=0.01,
            cold_completion_discarded=True,
            cache_reset_verified=True,
            protocol_hash_match=True,
            generation_volume_within_band=True,
            sample_hash_match=True,
            write_untrusted_diagnostic=False,
        )
    )

    plan_turn = summary["turns"][1]
    assert summary["trusted_measurement"] is True
    assert plan_turn["decode_tps"] == 12.0
    assert plan_turn["accepted_per_draft"] == 0.25
    assert summary["runtime_config_hash_artifacts"]["vllm_request_metrics.jsonl"] == "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def test_task_summary_rejects_raw_vllm_jsonl_runtime_config_hash_mismatch(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    base = datetime(2026, 5, 7, 18, 0, 0, tzinfo=UTC)

    def ts(offset_s: float) -> str:
        return (base + timedelta(seconds=offset_s)).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    _write_jsonl(
        task_dir / "codex_trace.jsonl",
        [
            {"event": "task_start", "ts": ts(0.0), "task_id": "transcript-merge-regression/v1-clean-baseline"},
            {"event": "turn_start", "turn": 0, "regime": "plan", "ts": ts(0.5), "vllm_request_id": "req-1"},
            {"event": "turn_end", "turn": 0, "ts": ts(1.5), "completion_tokens": 12},
            {"event": "task_end", "ts": ts(2.0), "exit_code": 0, "task_score": 0.74, "wallclock_s": 2.0},
        ],
    )
    _write_jsonl(
        task_dir / "vllm_request_metrics.jsonl",
        [
            {
                "request_id": "req-1",
                "runtime_config_hash": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "prompt_tokens": 50,
                "generation_tokens": 12,
                "decode_s": 1.0,
                "spec_decode_num_accepted_tokens": 3,
                "spec_decode_num_draft_tokens": 12,
            },
        ],
    )

    with pytest.raises(RuntimeError, match="vllm_request_metrics.jsonl"):
        build_task_summary(
            Namespace(
                round=0,
                task_dir=str(task_dir),
                family="transcript-merge-regression",
                variant="v1-clean-baseline",
                runtime_config_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                baseline_workspace_hash=None,
                run_wallclocks_json="[1.9, 2.0, 2.1]",
                clock_skew_ms_p99=8,
                trace_emitter_correctness_verified_at="2026-05-07T14:00:00Z",
                dcgm_interval_s=0.01,
                cold_completion_discarded=True,
                cache_reset_verified=True,
                protocol_hash_match=True,
                generation_volume_within_band=True,
                sample_hash_match=True,
                write_untrusted_diagnostic=False,
            )
        )


def test_task_summary_rejects_incomplete_vllm_request_metrics_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "vllm_request_metrics.jsonl"
    _write_jsonl(path, [{"request_id": "req-1", "prompt_tokens": 50}])

    with pytest.raises(RuntimeError, match="missing numeric fields"):
        _vllm_jsonl_by_request(path)


def test_task_summary_rejects_empty_vllm_per_turn_json(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "vllm_per_turn.json").write_text('{"requests": {}}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="No request-keyed vLLM rows"):
        _load_vllm_request_metrics(task_dir)


def test_task_summary_rejects_missing_dcgm_profile_fields(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    base = datetime(2026, 5, 7, 18, 0, 0, tzinfo=UTC)

    def ts(offset_s: float) -> str:
        return (base + timedelta(seconds=offset_s)).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    _write_jsonl(
        task_dir / "codex_trace.jsonl",
        [
            {"event": "task_start", "ts": ts(0.0), "task_id": "transcript-merge-regression/v1-clean-baseline"},
            {"event": "turn_start", "turn": 0, "regime": "plan", "ts": ts(0.1), "vllm_request_id": "req-1"},
            {"event": "turn_end", "turn": 0, "ts": ts(1.0), "prompt_tokens": 50, "completion_tokens": 12},
            {"event": "task_end", "ts": ts(2.0), "exit_code": 0, "task_score": 0.74, "wallclock_s": 2.0},
        ],
    )
    (task_dir / "vllm_per_turn.json").write_text(
        json.dumps(
            {
                "requests": {
                    "req-1": {
                        "completion_tokens": 12,
                        "decode_sum_s": 0.9,
                        "decode_tps": 13.33,
                        "spec_decode_num_accepted_tokens": 3,
                        "spec_decode_num_draft_tokens": 12,
                        "accepted_per_draft_token": 0.25,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        task_dir / "dcgm_samples.jsonl",
        [
            {
                "ts": ts(index / 100),
                "gpu": 0,
                "runtime_config_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "dram_active_pct": 0.54,
                "sm_active_pct": 0.31,
            }
            for index in range(200)
        ],
    )

    summary = build_task_summary(
        Namespace(
            round=0,
            task_dir=str(task_dir),
            family="transcript-merge-regression",
            variant="v1-clean-baseline",
            runtime_config_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            baseline_workspace_hash=None,
            run_wallclocks_json="[1.9, 2.0, 2.1]",
            clock_skew_ms_p99=8,
            trace_emitter_correctness_verified_at="2026-05-07T14:00:00Z",
            dcgm_interval_s=0.01,
            cold_completion_discarded=True,
            cache_reset_verified=True,
            protocol_hash_match=True,
            generation_volume_within_band=True,
            sample_hash_match=True,
            write_untrusted_diagnostic=True,
        )
    )

    assert summary["trusted_measurement"] is False
    assert summary["truthful_measurement_attestation"]["rule_6_dcgm_profile_fields_present"] is False
    assert summary["truthful_measurement_attestation"]["rule_6_dcgm_missing_profile_fields"] == [
        "sm_occupancy_pct",
        "pipe_tensor_active_pct",
        "pipe_fp16_active_pct",
    ]


def _write_task_summary(
    round_dir: Path,
    index: int,
    task_id: str,
    *,
    runtime_config_hash: str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    sample_hash: str = SAMPLE_HASH,
    task_completed: bool = True,
    task_score: float | None = 0.8,
) -> None:
    task_dir = round_dir / f"task_{index:02d}"
    task_dir.mkdir(parents=True)
    (task_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema": "lumo.track_b.e2e_task_summary.v1",
                "round": 0,
                "task_id": task_id,
                "runtime_config_hash": runtime_config_hash,
                "trusted_measurement": True,
                "wallclock_s": float(100 + index),
                "task_completed": task_completed,
                "task_score": task_score,
                "regime_share": {"plan": 1.0},
                "bottleneck_diagnosis": "memory-bw-headroom",
                "sample_hash": sample_hash,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_nested_task_summary(round_dir: Path, index: int, task_id: str) -> None:
    task_dir = round_dir / f"task_{index:02d}" / "run_01"
    task_dir.mkdir(parents=True)
    (task_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema": "lumo.track_b.e2e_task_summary.v1",
                "round": 0,
                "task_id": task_id,
                "runtime_config_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "trusted_measurement": True,
                "wallclock_s": float(100 + index),
                "task_completed": True,
                "task_score": 0.8,
                "regime_share": {"plan": 1.0},
                "bottleneck_diagnosis": "memory-bw-headroom",
                "sample_hash": SAMPLE_HASH,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_round_summary_requires_unique_fixed_sample_tasks(tmp_path: Path) -> None:
    round_dir = tmp_path / "round_0"
    round_dir.mkdir()
    for index, task_id in enumerate(TRACK_B_E2E_TASKS[:12]):
        _write_task_summary(round_dir, index, task_id)

    summary = build_round_summary(
        Namespace(
            round=0,
            round_dir=str(round_dir),
            runtime_config_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            config_delta_vs_prior_round="",
            hypothesis="baseline",
            wallclock_delta_vs_prior_round_s=None,
            auto_research_agent_recommendation="",
            next_round_proposal="",
            write_untrusted_diagnostic=False,
        )
    )

    assert summary["trusted_task_count"] == 12
    assert summary["trusted_unique_task_count"] == 12
    assert summary["duplicate_trusted_task_ids"] == []
    assert summary["runtime_config_hash_mismatch_count"] == 0
    assert summary["task_summary_schema_mismatch_count"] == 0
    assert summary["task_summary_round_mismatch_count"] == 0
    assert (round_dir / "round_summary.json").is_file()


def test_round_summary_accepts_runner_nested_attempt_summaries(tmp_path: Path) -> None:
    round_dir = tmp_path / "round_0"
    round_dir.mkdir()
    for index, task_id in enumerate(TRACK_B_E2E_TASKS[:12]):
        _write_nested_task_summary(round_dir, index, task_id)

    summary = build_round_summary(
        Namespace(
            round=0,
            round_dir=str(round_dir),
            runtime_config_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            config_delta_vs_prior_round="",
            hypothesis="baseline",
            wallclock_delta_vs_prior_round_s=None,
            auto_research_agent_recommendation="",
            next_round_proposal="",
            write_untrusted_diagnostic=False,
        )
    )

    assert summary["trusted_task_count"] == 12
    assert summary["trusted_unique_task_count"] == 12
    assert summary["duplicate_trusted_task_ids"] == []


def test_round_summary_rejects_duplicate_or_mismatched_sample_tasks(tmp_path: Path) -> None:
    round_dir = tmp_path / "round_0"
    round_dir.mkdir()
    for index in range(12):
        _write_task_summary(
            round_dir,
            index,
            TRACK_B_E2E_TASKS[0],
            sample_hash="sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" if index == 0 else SAMPLE_HASH,
        )

    with pytest.raises(RuntimeError, match="unique trusted sample tasks"):
        build_round_summary(
            Namespace(
                round=0,
                round_dir=str(round_dir),
                runtime_config_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                config_delta_vs_prior_round="",
                hypothesis="baseline",
                wallclock_delta_vs_prior_round_s=None,
                auto_research_agent_recommendation="",
                next_round_proposal="",
                write_untrusted_diagnostic=False,
            )
        )


def test_round_summary_rejects_incomplete_trusted_tasks(tmp_path: Path) -> None:
    round_dir = tmp_path / "round_0"
    round_dir.mkdir()
    for index, task_id in enumerate(TRACK_B_E2E_TASKS[:12]):
        _write_task_summary(
            round_dir,
            index,
            task_id,
            task_completed=index != 0,
            task_score=None if index == 1 else 0.8,
        )

    with pytest.raises(RuntimeError, match="trusted task summaries completed"):
        build_round_summary(
            Namespace(
                round=0,
                round_dir=str(round_dir),
                runtime_config_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                config_delta_vs_prior_round="",
                hypothesis="baseline",
                wallclock_delta_vs_prior_round_s=None,
                auto_research_agent_recommendation="",
                next_round_proposal="",
                write_untrusted_diagnostic=False,
            )
        )


def test_round_summary_rejects_runtime_config_hash_mismatch(tmp_path: Path) -> None:
    round_dir = tmp_path / "round_0"
    round_dir.mkdir()
    for index, task_id in enumerate(TRACK_B_E2E_TASKS[:12]):
        _write_task_summary(
            round_dir,
            index,
            task_id,
            runtime_config_hash="sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" if index == 0 else "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )

    with pytest.raises(RuntimeError, match="runtime_config_hash"):
        build_round_summary(
            Namespace(
                round=0,
                round_dir=str(round_dir),
                runtime_config_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                config_delta_vs_prior_round="",
                hypothesis="baseline",
                wallclock_delta_vs_prior_round_s=None,
                auto_research_agent_recommendation="",
                next_round_proposal="",
                write_untrusted_diagnostic=False,
            )
        )


def test_round_summary_rejects_task_summary_schema_or_round_mismatch(tmp_path: Path) -> None:
    round_dir = tmp_path / "round_0"
    round_dir.mkdir()
    for index, task_id in enumerate(TRACK_B_E2E_TASKS[:12]):
        _write_task_summary(round_dir, index, task_id)
    first_summary = round_dir / "task_00" / "summary.json"
    payload = json.loads(first_summary.read_text(encoding="utf-8"))
    payload["schema"] = "not-track-b"
    first_summary.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    second_summary = round_dir / "task_01" / "summary.json"
    payload = json.loads(second_summary.read_text(encoding="utf-8"))
    payload["round"] = 99
    second_summary.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="mismatched schema"):
        build_round_summary(
            Namespace(
                round=0,
                round_dir=str(round_dir),
                runtime_config_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                config_delta_vs_prior_round="",
                hypothesis="baseline",
                wallclock_delta_vs_prior_round_s=None,
                auto_research_agent_recommendation="",
                next_round_proposal="",
                write_untrusted_diagnostic=False,
            )
        )

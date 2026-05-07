from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import preflight_track_b_e2e  # noqa: E402


def test_preflight_blocks_without_trace_out_request_labels_or_pynvml(monkeypatch) -> None:
    def fake_command(command: list[str], timeout_s: float = 10.0) -> dict[str, object]:
        if command == ["codex", "--version"]:
            return {"ok": True, "stdout": "codex-cli 0.128.0\n"}
        if command == ["codex", "exec", "--help"]:
            return {"ok": True, "stdout": "Usage: codex exec [OPTIONS]\n      --json\n"}
        return {"ok": True, "stdout": ""}

    def fake_get(url: str, timeout_s: float = 5.0) -> dict[str, object]:
        if url.endswith("/health"):
            return {"ok": True, "status_code": 200, "text": ""}
        return {
            "ok": True,
            "status_code": 200,
            "text": "\n".join(
                [
                    "unrelated_metric_total{request_id=\"req-1\"} 1",
                    "vllm:prompt_tokens_total{engine=\"0\"} 128",
                    "vllm:generation_tokens_total{engine=\"0\"} 32",
                    "vllm:spec_decode_num_drafts_total{engine=\"0\"} 1",
                    "vllm:spec_decode_num_draft_tokens_total{engine=\"0\"} 12",
                    "vllm:spec_decode_num_accepted_tokens_total{engine=\"0\"} 4",
                ]
            ),
        }

    class FakeCompleted:
        returncode = 1
        stdout = ""
        stderr = ""

    monkeypatch.setattr(preflight_track_b_e2e, "_command", fake_command)
    monkeypatch.setattr(preflight_track_b_e2e, "_get", fake_get)
    monkeypatch.setattr(
        preflight_track_b_e2e,
        "_sampler_smoke",
        lambda python, duration_s: {
            "ok": False,
            "sample_count": 0,
            "profile_fields_present": False,
            "observed_numeric_profile_fields": [],
            "missing_profile_fields": list(preflight_track_b_e2e.DCGM_PROFILE_FIELDS),
            "telemetry_sources": [],
            "stderr": "",
        },
    )
    monkeypatch.setattr(preflight_track_b_e2e.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(preflight_track_b_e2e.subprocess, "run", lambda *args, **kwargs: FakeCompleted())

    payload = preflight_track_b_e2e.audit(
        Namespace(
            health_url="http://127.0.0.1:9950/health",
            metrics_url="http://127.0.0.1:9950/metrics",
            python=sys.executable,
            sampler_smoke_duration_s=0.05,
            vllm_request_metrics_jsonl="",
            required_checks=[
                "vllm_health",
                "spec_decode_metrics_exposed",
                "vllm_request_metrics_join_available",
                "codex_trace_out_supported",
                "pynvml_available",
            ],
        )
    )

    assert payload["round0_may_run"] is False
    assert payload["blocking_reasons"] == [
        "vllm_request_metrics_join_available",
        "codex_trace_out_supported",
        "pynvml_available",
    ]


def test_preflight_accepts_required_e2e_instrumentation(monkeypatch) -> None:
    def fake_command(command: list[str], timeout_s: float = 10.0) -> dict[str, object]:
        if command == ["codex", "--version"]:
            return {"ok": True, "stdout": "codex-cli patched\n"}
        if command == ["codex", "exec", "--help"]:
            return {"ok": True, "stdout": "Usage: codex exec [OPTIONS]\n      --json\n      --trace-out <PATH>\n"}
        return {"ok": True, "stdout": ""}

    def fake_get(url: str, timeout_s: float = 5.0) -> dict[str, object]:
        if url.endswith("/health"):
            return {"ok": True, "status_code": 200, "text": ""}
        return {
            "ok": True,
            "status_code": 200,
            "text": "\n".join(
                [
                    "vllm:prompt_tokens_total{engine=\"0\",request_id=\"req-1\"} 128",
                    "vllm:generation_tokens_total{engine=\"0\",request_id=\"req-1\"} 32",
                    "vllm:spec_decode_num_drafts_total{engine=\"0\",request_id=\"req-1\"} 1",
                    "vllm:spec_decode_num_draft_tokens_total{engine=\"0\",request_id=\"req-1\"} 12",
                    "vllm:spec_decode_num_accepted_tokens_total{engine=\"0\",request_id=\"req-1\"} 4",
                ]
            ),
        }

    class FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(preflight_track_b_e2e, "_command", fake_command)
    monkeypatch.setattr(preflight_track_b_e2e, "_get", fake_get)
    monkeypatch.setattr(
        preflight_track_b_e2e,
        "_sampler_smoke",
        lambda python, duration_s: {
            "ok": True,
            "sample_count": 5,
            "profile_fields_present": True,
            "observed_numeric_profile_fields": list(preflight_track_b_e2e.DCGM_PROFILE_FIELDS),
            "missing_profile_fields": [],
            "telemetry_sources": ["dcgm"],
            "stderr": "",
        },
    )
    monkeypatch.setattr(preflight_track_b_e2e.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(preflight_track_b_e2e.subprocess, "run", lambda *args, **kwargs: FakeCompleted())

    payload = preflight_track_b_e2e.audit(
        Namespace(
            health_url="http://127.0.0.1:9950/health",
            metrics_url="http://127.0.0.1:9950/metrics",
            python=sys.executable,
            sampler_smoke_duration_s=0.05,
            vllm_request_metrics_jsonl="",
            required_checks=[
                "vllm_health",
                "spec_decode_metrics_exposed",
                "vllm_request_metrics_join_available",
                "codex_trace_out_supported",
                "pynvml_available",
            ],
        )
    )

    assert payload["round0_may_run"] is True
    assert payload["blocking_reasons"] == []


def test_preflight_accepts_request_metrics_side_channel(tmp_path: Path, monkeypatch) -> None:
    metrics_jsonl = tmp_path / "vllm_request_metrics.jsonl"
    metrics_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "request_id": "req-1",
                        "prompt_tokens": 128,
                        "generation_tokens": 32,
                        "spec_decode_num_draft_tokens": 12,
                        "spec_decode_num_accepted_tokens": 4,
                    }
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_command(command: list[str], timeout_s: float = 10.0) -> dict[str, object]:
        if command == ["codex", "--version"]:
            return {"ok": True, "stdout": "codex-cli patched\n"}
        if command == ["codex", "exec", "--help"]:
            return {"ok": True, "stdout": "Usage: codex exec [OPTIONS]\n      --trace-out <PATH>\n"}
        return {"ok": True, "stdout": ""}

    def fake_get(url: str, timeout_s: float = 5.0) -> dict[str, object]:
        if url.endswith("/health"):
            return {"ok": True, "status_code": 200, "text": ""}
        return {
            "ok": True,
            "status_code": 200,
            "text": "\n".join(
                [
                    "vllm:prompt_tokens_total{engine=\"0\"} 128",
                    "vllm:generation_tokens_total{engine=\"0\"} 32",
                    "vllm:spec_decode_num_drafts_total{engine=\"0\"} 1",
                    "vllm:spec_decode_num_draft_tokens_total{engine=\"0\"} 12",
                    "vllm:spec_decode_num_accepted_tokens_total{engine=\"0\"} 4",
                ]
            ),
        }

    class FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(preflight_track_b_e2e, "_command", fake_command)
    monkeypatch.setattr(preflight_track_b_e2e, "_get", fake_get)
    monkeypatch.setattr(
        preflight_track_b_e2e,
        "_sampler_smoke",
        lambda python, duration_s: {
            "ok": True,
            "sample_count": 5,
            "profile_fields_present": True,
            "observed_numeric_profile_fields": list(preflight_track_b_e2e.DCGM_PROFILE_FIELDS),
            "missing_profile_fields": [],
            "telemetry_sources": ["dcgm"],
            "stderr": "",
        },
    )
    monkeypatch.setattr(preflight_track_b_e2e.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(preflight_track_b_e2e.subprocess, "run", lambda *args, **kwargs: FakeCompleted())

    payload = preflight_track_b_e2e.audit(
        Namespace(
            health_url="http://127.0.0.1:9950/health",
            metrics_url="http://127.0.0.1:9950/metrics",
            python=sys.executable,
            sampler_smoke_duration_s=0.05,
            vllm_request_metrics_jsonl=str(metrics_jsonl),
            required_checks=[
                "vllm_health",
                "spec_decode_metrics_exposed",
                "vllm_request_metrics_join_available",
                "codex_trace_out_supported",
                "pynvml_available",
                "dcgm_sampler_runs",
                "dcgm_profile_fields_available",
            ],
        )
    )

    assert payload["round0_may_run"] is True
    assert payload["checks"]["vllm_request_id_labels_exposed"]["ok"] is False
    assert payload["checks"]["vllm_request_metrics_side_channel"]["ok"] is True
    assert payload["checks"]["vllm_request_metrics_side_channel"]["valid_request_metric_row_count"] == 1
    assert payload["checks"]["vllm_request_metrics_join_available"]["ok"] is True


def test_request_metrics_side_channel_requires_complete_rows(tmp_path: Path) -> None:
    metrics_jsonl = tmp_path / "vllm_request_metrics.jsonl"
    metrics_jsonl.write_text(
        "\n".join(
            [
                json.dumps({"request_id": "req-1"}),
                json.dumps(
                    {
                        "prompt_tokens": 128,
                        "generation_tokens": 32,
                        "spec_decode_num_draft_tokens": 12,
                        "spec_decode_num_accepted_tokens": 4,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    coverage = preflight_track_b_e2e._request_metrics_jsonl_coverage(str(metrics_jsonl))

    assert coverage["ok"] is False
    assert coverage["request_id_seen"] is True
    assert all(coverage["required_field_coverage"].values())
    assert coverage["valid_request_metric_row_count"] == 0
    assert coverage["invalid_request_metric_row_count"] == 2


def test_request_metrics_side_channel_accepts_completion_tokens_alias(tmp_path: Path) -> None:
    metrics_jsonl = tmp_path / "vllm_request_metrics.jsonl"
    metrics_jsonl.write_text(
        json.dumps(
            {
                "request_id": "req-1",
                "prompt_tokens": 128,
                "completion_tokens": 32,
                "spec_decode_num_draft_tokens": 12,
                "spec_decode_num_accepted_tokens": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    coverage = preflight_track_b_e2e._request_metrics_jsonl_coverage(str(metrics_jsonl))

    assert coverage["ok"] is True
    assert coverage["valid_request_metric_row_count"] == 1
    assert coverage["required_field_coverage"]["generation_tokens"] is True


def test_request_id_gate_requires_labels_on_join_metrics() -> None:
    metrics = "\n".join(
        [
            "some_debug_metric_total{request_id=\"req-1\"} 1",
            "vllm:prompt_tokens_total{engine=\"0\"} 128",
            "vllm:generation_tokens_total{engine=\"0\",request_id=\"req-1\"} 32",
            "vllm:spec_decode_num_draft_tokens_total{engine=\"0\",request_id=\"req-1\"} 12",
            "vllm:spec_decode_num_accepted_tokens_total{engine=\"0\",request_id=\"req-1\"} 4",
        ]
    )

    coverage = preflight_track_b_e2e._request_labeled_metric_coverage(metrics)

    assert coverage == {
        "vllm:prompt_tokens_total": False,
        "vllm:generation_tokens_total": True,
        "vllm:spec_decode_num_draft_tokens_total": True,
        "vllm:spec_decode_num_accepted_tokens_total": True,
    }


def test_dcgm_profile_gate_requires_all_profile_fields(monkeypatch, tmp_path: Path) -> None:
    sample_path = tmp_path / "samples.jsonl"
    sample_path.write_text(
        "\n".join(
            [
                '{"dram_active_pct":0.4,"sm_active_pct":0.5,"telemetry_source":"dcgm"}',
                '{"dram_active_pct":0.4,"sm_active_pct":0.5,"telemetry_source":"dcgm"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeTemp:
        name = str(sample_path)

        def __enter__(self) -> "FakeTemp":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(
        preflight_track_b_e2e.tempfile,
        "NamedTemporaryFile",
        lambda **kwargs: FakeTemp(),
    )
    monkeypatch.setattr(preflight_track_b_e2e.subprocess, "run", lambda *args, **kwargs: FakeCompleted())

    payload = preflight_track_b_e2e._sampler_smoke(Path(sys.executable), 0.05)

    assert payload["profile_fields_present"] is False
    assert payload["observed_numeric_profile_fields"] == ["dram_active_pct", "sm_active_pct"]
    assert payload["missing_profile_fields"] == [
        "sm_occupancy_pct",
        "pipe_tensor_active_pct",
        "pipe_fp16_active_pct",
    ]


def test_dcgm_profile_gate_requires_available_flag(monkeypatch, tmp_path: Path) -> None:
    sample_path = tmp_path / "samples.jsonl"
    sample_path.write_text(
        json.dumps(
            {
                "profile_fields_available": False,
                "dram_active_pct": 0.4,
                "sm_active_pct": 0.5,
                "sm_occupancy_pct": 0.6,
                "pipe_tensor_active_pct": 0.7,
                "pipe_fp16_active_pct": 0.8,
                "telemetry_source": "nvml",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeTemp:
        name = str(sample_path)

        def __enter__(self) -> "FakeTemp":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(
        preflight_track_b_e2e.tempfile,
        "NamedTemporaryFile",
        lambda **kwargs: FakeTemp(),
    )
    monkeypatch.setattr(preflight_track_b_e2e.subprocess, "run", lambda *args, **kwargs: FakeCompleted())

    payload = preflight_track_b_e2e._sampler_smoke(Path(sys.executable), 0.05)

    assert payload["profile_fields_present"] is False
    assert payload["profile_fields_available_sample_count"] == 0
    assert payload["observed_numeric_profile_fields"] == sorted(preflight_track_b_e2e.DCGM_PROFILE_FIELDS)
    assert payload["missing_profile_fields"] == []

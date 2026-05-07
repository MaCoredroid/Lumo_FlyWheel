from __future__ import annotations

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
        lambda python, duration_s: {"ok": False, "sample_count": 0, "profile_fields_present": False, "stderr": ""},
    )
    monkeypatch.setattr(preflight_track_b_e2e.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(preflight_track_b_e2e.subprocess, "run", lambda *args, **kwargs: FakeCompleted())

    payload = preflight_track_b_e2e.audit(
        Namespace(
            health_url="http://127.0.0.1:9950/health",
            metrics_url="http://127.0.0.1:9950/metrics",
            python=sys.executable,
            sampler_smoke_duration_s=0.05,
            required_checks=[
                "vllm_health",
                "spec_decode_metrics_exposed",
                "vllm_request_id_labels_exposed",
                "codex_trace_out_supported",
                "pynvml_available",
            ],
        )
    )

    assert payload["round0_may_run"] is False
    assert payload["blocking_reasons"] == [
        "vllm_request_id_labels_exposed",
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
        lambda python, duration_s: {"ok": True, "sample_count": 5, "profile_fields_present": True, "stderr": ""},
    )
    monkeypatch.setattr(preflight_track_b_e2e.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(preflight_track_b_e2e.subprocess, "run", lambda *args, **kwargs: FakeCompleted())

    payload = preflight_track_b_e2e.audit(
        Namespace(
            health_url="http://127.0.0.1:9950/health",
            metrics_url="http://127.0.0.1:9950/metrics",
            python=sys.executable,
            sampler_smoke_duration_s=0.05,
            required_checks=[
                "vllm_health",
                "spec_decode_metrics_exposed",
                "vllm_request_id_labels_exposed",
                "codex_trace_out_supported",
                "pynvml_available",
            ],
        )
    )

    assert payload["round0_may_run"] is True
    assert payload["blocking_reasons"] == []


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

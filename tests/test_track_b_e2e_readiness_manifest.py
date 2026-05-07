from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_track_b_e2e_readiness_manifest as readiness  # noqa: E402


def test_readiness_manifest_reports_round0_blocked(tmp_path: Path) -> None:
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(
        json.dumps(
            {
                "round0_may_run": False,
                "blocking_reasons": [
                    "vllm_request_id_labels_exposed",
                    "codex_trace_out_supported",
                    "dcgm_profile_fields_available",
                ],
                "checks": {
                    "codex_trace_out_supported": {"ok": False},
                    "dcgm_sampler_runs": {"ok": True},
                    "dcgm_profile_fields_available": {"ok": False},
                    "pynvml_available": {"ok": True},
                    "vllm_request_id_labels_exposed": {"ok": False},
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = readiness.build_manifest(Namespace(preflight_json=str(preflight_path), out=""))

    assert manifest["round0_ready"] is False
    assert manifest["decision"] == "round0_blocked"
    statuses = {step["step"]: step["status"] for step in manifest["implementation_steps"]}
    assert statuses["A"] == "blocked"
    assert statuses["B"] == "blocked"
    assert statuses["C"] == "complete"
    assert statuses["D"] == "blocked"
    assert statuses["E"] == "complete"
    assert statuses["F"] == "complete"
    assert statuses["G"] == "blocked"
    assert manifest["hard_gates"]["round0_summary_exists"] is False


def test_readiness_manifest_requires_round0_artifacts_even_if_preflight_passes(tmp_path: Path) -> None:
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(
        json.dumps(
            {
                "round0_may_run": True,
                "blocking_reasons": [],
                "checks": {
                    "codex_trace_out_supported": {"ok": True},
                    "dcgm_sampler_runs": {"ok": True},
                    "dcgm_profile_fields_available": {"ok": True},
                    "pynvml_available": {"ok": True},
                    "vllm_request_id_labels_exposed": {"ok": True},
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = readiness.build_manifest(Namespace(preflight_json=str(preflight_path), out=""))

    assert manifest["hard_gates"]["preflight_round0_may_run"] is True
    assert manifest["hard_gates"]["round0_summary_exists"] is False
    assert manifest["round0_ready"] is False

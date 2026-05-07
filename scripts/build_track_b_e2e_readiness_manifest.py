#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _git(command: list[str]) -> str:
    result = subprocess.run(
        ["git", *command],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _exists(rel: str) -> bool:
    return (REPO_ROOT / rel).exists()


def _contains(rel: str, needle: str) -> bool:
    path = REPO_ROOT / rel
    if not path.is_file():
        return False
    return needle in path.read_text(encoding="utf-8", errors="replace")


def _status(ok: bool, *, blocked: bool = False) -> str:
    if ok:
        return "complete"
    if blocked:
        return "blocked"
    return "missing"


def _step(step: str, requirement: str, evidence: dict[str, Any], ok: bool, *, blocked: bool = False) -> dict[str, Any]:
    return {
        "step": step,
        "requirement": requirement,
        "status": _status(ok, blocked=blocked),
        "evidence": evidence,
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    preflight = _load_json(Path(args.preflight_json)) if args.preflight_json else {}
    checks = preflight.get("checks") if isinstance(preflight.get("checks"), dict) else {}
    blockers = preflight.get("blocking_reasons") if isinstance(preflight.get("blocking_reasons"), list) else []

    trace_patch_exists = any(
        path.exists()
        for path in [
            REPO_ROOT / "vendor" / "codex-cli" / "patches" / "trace_emitter.patch",
            REPO_ROOT / "patches" / "codex" / "trace_emitter.patch",
        ]
    )
    trace_correctness_path = REPO_ROOT / "output" / "track_b_e2e" / "codex_trace_emitter_correctness.json"
    round0_summary_path = REPO_ROOT / "output" / "track_b_e2e" / "round_0" / "round_summary.json"
    ncu_outputs = sorted((REPO_ROOT / "output" / "track_b_e2e").glob("ncu_*.csv"))

    steps = [
        _step(
            "A",
            "Patch Codex CLI with --trace-out and verify trace-emitter correctness.",
            {
                "trace_patch_exists": trace_patch_exists,
                "trace_correctness_artifact": str(trace_correctness_path.relative_to(REPO_ROOT)),
                "trace_correctness_exists": trace_correctness_path.is_file(),
                "codex_trace_out_supported": checks.get("codex_trace_out_supported", {}).get("ok"),
            },
            trace_patch_exists and trace_correctness_path.is_file() and checks.get("codex_trace_out_supported", {}).get("ok") is True,
            blocked="codex_trace_out_supported" in blockers,
        ),
        _step(
            "B",
            "DCGM/NVML 100 Hz sampler emits required profile fields.",
            {
                "sampler_script": "scripts/sample_dcgm_during_task.py",
                "sampler_script_exists": _exists("scripts/sample_dcgm_during_task.py"),
                "pynvml_available": checks.get("pynvml_available", {}).get("ok"),
                "dcgm_sampler_runs": checks.get("dcgm_sampler_runs", {}).get("ok"),
                "dcgm_profile_fields_available": checks.get("dcgm_profile_fields_available", {}).get("ok"),
            },
            _exists("scripts/sample_dcgm_during_task.py")
            and checks.get("dcgm_sampler_runs", {}).get("ok") is True
            and checks.get("dcgm_profile_fields_available", {}).get("ok") is True,
            blocked="dcgm_profile_fields_available" in blockers,
        ),
        _step(
            "C",
            "E2E task runner exists.",
            {"runner_script": "scripts/run_track_b_e2e_task.py", "exists": _exists("scripts/run_track_b_e2e_task.py")},
            _exists("scripts/run_track_b_e2e_task.py"),
        ),
        _step(
            "D",
            "Per-turn vLLM metric extension captures spec_decode accepted/draft metrics keyed by request id.",
            {
                "metrics_module": "src/lumo_flywheel_serving/metrics.py",
                "compute_vllm_per_request_metrics_exists": _contains(
                    "src/lumo_flywheel_serving/metrics.py",
                    "def compute_vllm_per_request_metrics",
                ),
                "spec_decode_accepted_metric_named": _contains(
                    "src/lumo_flywheel_serving/metrics.py",
                    "spec_decode_num_accepted_tokens",
                ),
                "vllm_request_id_labels_exposed": checks.get("vllm_request_id_labels_exposed", {}).get("ok"),
            },
            _contains("src/lumo_flywheel_serving/metrics.py", "def compute_vllm_per_request_metrics")
            and checks.get("vllm_request_id_labels_exposed", {}).get("ok") is True,
            blocked="vllm_request_id_labels_exposed" in blockers,
        ),
        _step(
            "E",
            "Summary join and deterministic diagnosis rule exists.",
            {
                "summary_script": "scripts/build_track_b_e2e_summary.py",
                "exists": _exists("scripts/build_track_b_e2e_summary.py"),
                "truthful_attestation_fields": _contains(
                    "scripts/build_track_b_e2e_summary.py",
                    "truthful_measurement_attestation",
                ),
            },
            _exists("scripts/build_track_b_e2e_summary.py")
            and _contains("scripts/build_track_b_e2e_summary.py", "truthful_measurement_attestation"),
        ),
        _step(
            "F",
            "Auto research agent round proposal prompt template exists.",
            {"prompt": "prompts/track_b_e2e_round_proposal.md", "exists": _exists("prompts/track_b_e2e_round_proposal.md")},
            _exists("prompts/track_b_e2e_round_proposal.md"),
        ),
        _step(
            "G",
            "Round 0 dry run populated and validated.",
            {
                "round0_summary": str(round0_summary_path.relative_to(REPO_ROOT)),
                "round0_summary_exists": round0_summary_path.is_file(),
                "ncu_profile_count": len(ncu_outputs),
                "expected_ncu_profile_count": 5,
            },
            round0_summary_path.is_file() and len(ncu_outputs) >= 5,
            blocked=bool(blockers),
        ),
    ]

    hard_gates = {
        "preflight_round0_may_run": preflight.get("round0_may_run") is True,
        "round0_summary_exists": round0_summary_path.is_file(),
        "trace_correctness_exists": trace_correctness_path.is_file(),
        "all_implementation_steps_complete": all(step["status"] == "complete" for step in steps),
    }
    ready = all(hard_gates.values())
    return {
        "schema": "lumo.track_b.e2e_readiness_manifest.v1",
        "generated_at": _now(),
        "git_head": _git(["rev-parse", "--short", "HEAD"]),
        "git_status_short": _git(["status", "--short", "--branch"]),
        "plan": "docs/reports/auto_research/track-b-e2e-agentic-saturation-plan-20260507.md",
        "preflight_json": args.preflight_json,
        "blocking_reasons": blockers,
        "implementation_steps": steps,
        "hard_gates": hard_gates,
        "round0_ready": ready,
        "decision": "round0_may_run" if ready else "round0_blocked",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Track B E2E readiness manifest from real artifacts.")
    parser.add_argument("--preflight-json", default="output/track_b_e2e/preflight_20260507.json")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    payload = build_manifest(args)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if payload["round0_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"YAML must be a mapping: {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _gate_pass(round_dir: Path, tier: str) -> bool:
    candidates = sorted(round_dir.glob(f"candidates/*/{tier}_result.json"))
    return any(bool((_load_json(path) or {}).get("pass")) for path in candidates)


def _candidate_gate_pass(round_dir: Path, candidate_id: str, tier: str) -> bool:
    return bool((_load_json(round_dir / "candidates" / candidate_id / f"{tier}_result.json") or {}).get("pass"))


def _candidate_quality_equivalence_passed(round_dir: Path, candidate_id: str) -> bool:
    result = _load_json(round_dir / "candidates" / candidate_id / "controller_result.json") or {}
    if result.get("status") in {"accepted_candidate", "accepted_final"}:
        return True
    return (
        _candidate_gate_pass(round_dir, candidate_id, "b1")
        and _candidate_gate_pass(round_dir, candidate_id, "b2")
        and _candidate_gate_pass(round_dir, candidate_id, "b3")
    )


def audit(round_dir: Path) -> dict[str, Any]:
    spec_path = round_dir / "round_spec.yaml"
    if not spec_path.is_file():
        raise RuntimeError(f"round_spec.yaml missing: {spec_path}")
    spec = _load_yaml(spec_path)
    target_tps = float(spec["success_criteria"]["decode_speed_at_least_tps"])
    baseline_tps = float(spec.get("baseline_decode_tps", 7.5))
    incremental_multiplier = float(
        spec.get("success_criteria", {}).get("candidate_acceptance_incremental_speedup_at_least", 1.2)
    )
    throughput_results: list[dict[str, Any]] = []
    for path in sorted(round_dir.glob("candidates/*/throughput.json")):
        payload = _load_json(path)
        if not payload:
            continue
        decode_tps = payload.get("decode_tps")
        if decode_tps is None:
            continue
        throughput_results.append(
            {
                "candidate_id": path.parent.name,
                "decode_tps": float(decode_tps),
                "schema": payload.get("schema"),
                "completions_per_task": payload.get("completions_per_task"),
                "cold_completions_discarded": payload.get("cold_completions_discarded"),
                "warm_completions_measured": payload.get("warm_completions_measured"),
                "task_count": payload.get("task_count"),
                "path": str(path),
            }
        )
    best = max((row["decode_tps"] for row in throughput_results), default=None)
    incumbent_tps = max(
        (
            row["decode_tps"]
            for row in throughput_results
            if _candidate_quality_equivalence_passed(round_dir, str(row["candidate_id"]))
        ),
        default=baseline_tps,
    )
    candidate_accept_tps = incumbent_tps * incremental_multiplier
    incremental_candidates = [
        row
        for row in throughput_results
        if float(row["decode_tps"]) >= candidate_accept_tps
        and row.get("schema") == "lumo.track_b.real_workload_first_five.v1"
    ]
    promoted_candidates = []
    for row in throughput_results:
        candidate_id = str(row["candidate_id"])
        if (
            float(row["decode_tps"]) >= target_tps
            and row.get("schema") == "lumo.track_b.real_workload_first_five.v1"
            and int(row.get("completions_per_task") or 0) == 5
            and int(row.get("cold_completions_discarded") or 0) == 1
            and int(row.get("warm_completions_measured") or 0) == 4
            and _candidate_gate_pass(round_dir, candidate_id, "b1")
            and _candidate_gate_pass(round_dir, candidate_id, "b2")
            and _candidate_gate_pass(round_dir, candidate_id, "b3")
        ):
            promoted_candidates.append(row)
    checklist = [
        {
            "requirement": "Track B round spec exists and targets quality-bounded mutation",
            "pass": spec.get("round_type") == "track_b_quality_bounded_mutation",
            "evidence": str(spec_path),
        },
        {
            "requirement": "Prior CUTLASS auto-research memory is inherited",
            "pass": (round_dir / "prior_cutlass_memory.json").is_file()
            and int(spec.get("prior_cutlass_memory", {}).get("round_count_indexed", 0)) > 0,
            "evidence": str(round_dir / "prior_cutlass_memory.json"),
        },
        {
            "requirement": f"Real workload warm decode throughput reaches {target_tps:.2f} tok/s",
            "pass": best is not None and best >= target_tps,
            "evidence": throughput_results,
        },
        {
            "requirement": (
                f"At least one candidate clears the incremental acceptance preflight "
                f"({incremental_multiplier:.2f}x over quality-equivalent incumbent "
                f"{incumbent_tps:.2f} tok/s = {candidate_accept_tps:.2f} tok/s)"
            ),
            "pass": bool(incremental_candidates),
            "evidence": incremental_candidates,
        },
        {
            "requirement": "Speed gate uses the real vLLM first-five workload window",
            "pass": any(
                row.get("schema") == "lumo.track_b.real_workload_first_five.v1"
                and int(row.get("completions_per_task") or 0) == 5
                and int(row.get("cold_completions_discarded") or 0) == 1
                and int(row.get("warm_completions_measured") or 0) == 4
                for row in throughput_results
            ),
            "evidence": throughput_results,
        },
        {
            "requirement": "B-1 strong-equivalence/distributional gate passed",
            "pass": _gate_pass(round_dir, "b1"),
            "evidence": "candidates/*/b1_result.json",
        },
        {
            "requirement": "B-2 behavioral gate passed",
            "pass": _gate_pass(round_dir, "b2"),
            "evidence": "candidates/*/b2_result.json",
        },
        {
            "requirement": "B-3 full quality gate passed before promotion",
            "pass": _gate_pass(round_dir, "b3"),
            "evidence": "candidates/*/b3_result.json",
        },
        {
            "requirement": "One candidate satisfies speed plus B-1/B-2/B-3",
            "pass": bool(promoted_candidates),
            "evidence": promoted_candidates,
        },
    ]
    return {
        "round_id": spec.get("round_id"),
        "target_decode_tps": target_tps,
        "incumbent_decode_tps": incumbent_tps,
        "candidate_accept_decode_tps": candidate_accept_tps,
        "best_decode_tps": best,
        "incremental_candidates": incremental_candidates,
        "promoted_candidates": promoted_candidates,
        "complete": all(item["pass"] for item in checklist),
        "checklist": checklist,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Track B completion against round artifacts.")
    parser.add_argument("round_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = audit(args.round_dir.resolve())
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

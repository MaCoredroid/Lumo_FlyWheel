#!/usr/bin/env python3
"""FR10 serving wiring-op gate.

Kernel tests prove the tree math on fixed tensors. This gate consumes evidence
from a real ``tree_mtp`` request, but only fixed-input op replay is eligible for
byte-exact claims: capture serving op inputs once, replay the serving wiring op
and the reference on those same inputs, then compare. End-to-end sampled LLM
output is intentionally outside this gate and must be judged statistically.
Missing replay evidence is a failure, not a skip.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DIAG = {
    "conv_path0_vs_linear_ref_max": 0,
    "conv_flat_path0_vs_linear_ref_max": 1,
    "conv_path0_nonzero_count": 2,
    "conv_flat_path0_nonzero_count": 3,
    "conv_diag_events": 4,
    "tree_node_count": 5,
    "conv_all_nodes_vs_serial_max": 6,
    "conv_state_rows_vs_native_max": 7,
    "sibling_leaf_path0_conv_max": 8,
    "flat_sibling_leaf_path0_conv_max": 9,
    "sibling_leaf_path0_nonzero_count": 10,
    "flat_sibling_leaf_path0_nonzero_count": 11,
    "staged_ssm_state_vs_tree_state_max": 12,
    "staged_ssm_state_nonzero_count": 13,
    "conv_tree_eligible_spec_rows": 14,
    "conv_tree_branch_spec_rows": 15,
    "conv_flat_fallback_spec_rows": 16,
    "scan_tree_eligible_spec_rows": 17,
    "scan_tree_branch_spec_rows": 18,
    "scan_flat_fallback_spec_rows": 19,
    "conv_num_prefills_sum": 20,
    "conv_num_decodes_sum": 21,
    "scan_num_spec_decodes_sum": 22,
}


@dataclass(frozen=True)
class MatrixRow:
    gate: str
    case: str
    regime: str
    passed: bool
    detail: str
    metrics: dict[str, Any]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _counter_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in rows if isinstance(row.get("conv_diag"), list)]


def _best_counter_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    live = [
        row
        for row in rows
        if row.get("has_sibling") and float(row.get("count") or 0.0) > 0.0
    ]
    if not live:
        live = [row for row in rows if float(row.get("count") or 0.0) > 0.0]
    if not live:
        return None
    return max(live, key=lambda row: float(row.get("count") or 0.0))


def _diag(row: dict[str, Any] | None, name: str) -> float | None:
    if row is None:
        return None
    values = row.get("conv_diag")
    idx = DIAG[name]
    if not isinstance(values, list) or idx >= len(values):
        return None
    return float(values[idx])


def _bool_row(
    *,
    gate: str,
    case: str,
    regime: str,
    passed: bool,
    detail: str,
    metrics: dict[str, Any],
) -> MatrixRow:
    return MatrixRow(
        gate=gate,
        case=case,
        regime=regime,
        passed=passed,
        detail=detail,
        metrics=metrics,
    )


def evaluate_wiring(
    *,
    counters_path: Path,
    commit_log_path: Path | None = None,
    commit_state_capture_path: Path | None = None,
    commit_native_handoff_path: Path | None = None,
    src_native_handoff_path: Path | None = None,
    scan_replay_path: Path | None = None,
    atol: float = 0.0,
) -> tuple[list[MatrixRow], dict[str, Any]]:
    rows = _counter_rows(counters_path)
    best = _best_counter_row(rows)
    commit_rows = _load_jsonl(commit_log_path) if commit_log_path else []
    scan_replay = (
        json.loads(scan_replay_path.read_text(encoding="utf-8"))
        if scan_replay_path and scan_replay_path.exists()
        else None
    )
    commit_state_rows = (
        _load_jsonl(commit_state_capture_path) if commit_state_capture_path else []
    )
    handoff_rows = (
        _load_jsonl(commit_native_handoff_path) if commit_native_handoff_path else []
    )
    src_native_handoff = (
        json.loads(src_native_handoff_path.read_text(encoding="utf-8"))
        if src_native_handoff_path and src_native_handoff_path.exists()
        else None
    )
    matrix: list[MatrixRow] = []

    diag_events = _diag(best, "conv_diag_events")
    has_live = best is not None and diag_events is not None and diag_events > 0
    matrix.append(
        _bool_row(
            gate="nonvacuity",
            case="wiring",
            regime="serving_evidence",
            passed=bool(has_live),
            detail="live branched tree_mtp GDN counter row is present",
            metrics={
                "counter_rows": len(rows),
                "selected_count": None if best is None else best.get("count"),
                "diag_events": diag_events,
                "shape": None if best is None else best.get("shape"),
                "has_sibling": None if best is None else best.get("has_sibling"),
            },
        )
    )

    conv_all = _diag(best, "conv_all_nodes_vs_serial_max")
    flat_path0 = _diag(best, "conv_flat_path0_vs_linear_ref_max")
    conv_eligible = _diag(best, "conv_tree_eligible_spec_rows")
    conv_branch = _diag(best, "conv_tree_branch_spec_rows")
    conv_fallback = _diag(best, "conv_flat_fallback_spec_rows")
    scan_eligible = _diag(best, "scan_tree_eligible_spec_rows")
    scan_branch = _diag(best, "scan_tree_branch_spec_rows")
    scan_fallback = _diag(best, "scan_flat_fallback_spec_rows")
    matrix.append(
        _bool_row(
            gate="tree_spec_branch_engagement",
            case="wiring",
            regime="structural_hard_zero_violation",
            passed=(
                conv_eligible is not None
                and conv_eligible > 0.0
                and conv_branch == conv_eligible
                and (conv_fallback or 0.0) == 0.0
                and scan_eligible is not None
                and scan_eligible > 0.0
                and scan_branch == scan_eligible
                and (scan_fallback or 0.0) == 0.0
            ),
            detail=(
                "every eligible tree_mtp spec row must enter the tree-aware conv "
                "and scan branches; flat fallback on eligible spec rows is a hard failure"
            ),
            metrics={
                "conv_tree_eligible_spec_rows": conv_eligible,
                "conv_tree_branch_spec_rows": conv_branch,
                "conv_flat_fallback_spec_rows": conv_fallback,
                "scan_tree_eligible_spec_rows": scan_eligible,
                "scan_tree_branch_spec_rows": scan_branch,
                "scan_flat_fallback_spec_rows": scan_fallback,
                "conv_num_prefills_sum": _diag(best, "conv_num_prefills_sum"),
                "conv_num_decodes_sum": _diag(best, "conv_num_decodes_sum"),
                "scan_num_spec_decodes_sum": _diag(best, "scan_num_spec_decodes_sum"),
            },
        )
    )
    matrix.append(
        _bool_row(
            gate="conv_gather",
            case="wiring_op",
            regime="deterministic_captured_replay_byte_exact",
            passed=conv_all is not None and conv_all <= atol and (flat_path0 or 0.0) > atol,
            detail=(
                "captured serving conv inputs replay through the serving gather equal "
                "serial ancestry; flat-wired replay negative control diverges"
            ),
            metrics={
                "conv_all_nodes_vs_serial_max": conv_all,
                "flat_path0_negative_control_max": flat_path0,
                "atol": atol,
            },
        )
    )

    conv_rows = _diag(best, "conv_state_rows_vs_native_max")
    matrix.append(
        _bool_row(
            gate="conv_commit_state",
            case="wiring_op",
            regime="deterministic_captured_replay_byte_exact",
            passed=conv_rows is not None and conv_rows <= atol,
            detail=(
                "captured serving conv-state inputs replay to native temporal rows"
            ),
            metrics={"conv_state_rows_vs_native_max": conv_rows, "atol": atol},
        )
    )

    sibling = _diag(best, "sibling_leaf_path0_conv_max")
    flat_sibling = _diag(best, "flat_sibling_leaf_path0_conv_max")
    matrix.append(
        _bool_row(
            gate="sibling_leaf_no_trunk_mutation",
            case="wiring_op",
            regime="deterministic_captured_replay_byte_exact",
            passed=sibling is not None and sibling <= atol and (flat_sibling or 0.0) > atol,
            detail=(
                "captured-input replay with a perturbed leaf leaves path0 op output "
                "unchanged; flat-wired replay negative control changes it"
            ),
            metrics={
                "sibling_leaf_path0_conv_max": sibling,
                "flat_sibling_leaf_path0_negative_control_max": flat_sibling,
                "atol": atol,
            },
        )
    )

    staged_ssm = _diag(best, "staged_ssm_state_vs_tree_state_max")
    matrix.append(
        _bool_row(
            gate="scan_state_staging",
            case="wiring_op",
            regime="deterministic_captured_replay_byte_exact",
            passed=staged_ssm is not None and staged_ssm <= atol,
            detail=(
                "captured serving scan-state staging rows equal tree_state rows "
                "immediately after index_copy"
            ),
            metrics={"staged_ssm_state_vs_tree_state_max": staged_ssm, "atol": atol},
        )
    )

    scan_serial_has_wiring = bool(
        scan_replay
        and scan_replay.get("serving_replay_pass") is True
        and scan_replay.get("serial_parity_pass") is True
    )
    matrix.append(
        _bool_row(
            gate="scan_output_serial_parity",
            case="wiring_op",
            regime="deterministic_captured_replay_byte_exact",
            passed=scan_serial_has_wiring,
            detail=(
                "captured serving scan inputs replay through serving scan and "
                "serial-per-path reference"
                if scan_serial_has_wiring
                else (
                    "missing_or_failed_wiring_evidence: serving scan inputs are "
                    "not yet captured and replayed through serving scan plus "
                    "serial-per-path reference"
                )
            ),
            metrics={} if scan_replay is None else scan_replay,
        )
    )

    redirected = [
        row
        for row in commit_rows
        if row.get("event")
        in {"mamba_commit_copy_bias", "mamba_preprocess_commit_copy_bias"}
        and int(row.get("tree_accepted_row", 0) or 0) > 0
    ]
    matrix.append(
        _bool_row(
            gate="postprocess_commit_redirect",
            case="wiring",
            regime="structural_hard_zero_violation",
            passed=bool(redirected),
            detail="preprocess/postprocess copied from accepted tree row at least once",
            metrics={
                "commit_log_rows": len(commit_rows),
                "redirected_rows": len(redirected),
                "redirect_events": dict(
                    sorted(
                        {
                            str(row.get("event")): sum(
                                1 for other in redirected if other.get("event") == row.get("event")
                            )
                            for row in redirected
                        }.items()
                    )
                ),
            },
        )
    )

    native_handoff = [
        row for row in handoff_rows if row.get("event") == "commit_native_handoff"
    ]
    native_handoff_accept = [
        row for row in native_handoff if int(row.get("prev_accepted_len", 0) or 0) > 0
    ]
    native_handoff_no_accept = [
        row for row in native_handoff if int(row.get("prev_accepted_len", 0) or 0) == 0
    ]
    native_handoff_cache_match_rows = [
        row
        for row in native_handoff
        if float(row.get("ssm_bank_vs_cached_tree_state_max_abs", 1.0) or 0.0) <= atol
        and float(row.get("conv_cache_vs_cache_max_abs", 1.0) or 0.0) <= atol
    ]
    commit_state_pass = bool(
        native_handoff
        and native_handoff_accept
        and src_native_handoff
        and src_native_handoff.get("src_native_pass") is True
        and src_native_handoff.get("negative_powered") is True
        and scan_serial_has_wiring
    )
    matrix.append(
        _bool_row(
            gate="committed_state_native_replay",
            case="wiring_op",
            regime="deterministic_captured_replay_byte_exact",
            passed=commit_state_pass,
            detail=(
                "next-event starting ssm_state and conv_state equal serial native "
                "decode over only the previous event's accepted path"
                if commit_state_pass
                else (
                    "missing_or_failed_wiring_evidence: capture the cross-step "
                    "handoff with FR10_TREE_GDN_COMMIT_HANDOFF_LOG and "
                    "FR10_TREE_GDN_SRC_NATIVE_PAYLOAD, then require src_native_pass"
                )
            ),
            metrics={
                "commit_state_capture_rows": len(commit_state_rows),
                "commit_native_handoff_rows": len(native_handoff),
                "commit_native_handoff_cache_match_rows": len(
                    native_handoff_cache_match_rows
                ),
                "commit_native_handoff_accept_rows": len(native_handoff_accept),
                "commit_native_handoff_no_accept_rows": len(native_handoff_no_accept),
                "commit_native_handoff_sample": native_handoff[:8],
                "src_native_handoff": src_native_handoff,
                "scan_serial_replay_required": True,
                "scan_serial_replay_pass": scan_serial_has_wiring,
                "atol": atol,
            },
        )
    )

    summary = {
        "schema": "fr10.serving_wiring_gate.v3",
        "passed": all(row.passed for row in matrix),
        "counter_dump": str(counters_path),
        "commit_log": str(commit_log_path) if commit_log_path else None,
        "commit_state_capture": (
            str(commit_state_capture_path) if commit_state_capture_path else None
        ),
        "commit_native_handoff": (
            str(commit_native_handoff_path) if commit_native_handoff_path else None
        ),
        "src_native_handoff": (
            str(src_native_handoff_path) if src_native_handoff_path else None
        ),
        "scan_replay": str(scan_replay_path) if scan_replay_path else None,
        "selected_counter": best,
        "matrix": [row.__dict__ for row in matrix],
    }
    return matrix, summary


def _engine_pids(container: str) -> list[int]:
    cmd = "ps -eo pid=,comm=,args= | awk '$2 ~ /EngineCor/ || $3 == \"VLLM::EngineCore\" {print $1}'"
    result = subprocess.run(
        ["docker", "exec", container, "bash", "-lc", cmd],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    return [int(pid) for pid in result.stdout.split() if pid.isdigit()]


def dump_live_counters(container: str, wait_s: float) -> None:
    pids = _engine_pids(container)
    if not pids:
        raise RuntimeError(f"no EngineCore pid found in {container}")
    for pid in pids:
        subprocess.run(
            ["docker", "exec", container, "kill", "-USR1", str(pid)],
            check=True,
        )
    time.sleep(wait_s)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counter-dump", type=Path, required=True)
    parser.add_argument("--commit-log", type=Path)
    parser.add_argument("--commit-state-capture", type=Path)
    parser.add_argument("--commit-native-handoff", type=Path)
    parser.add_argument("--src-native-handoff", type=Path)
    parser.add_argument("--scan-replay", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--container", default=None)
    parser.add_argument("--dump-live", action="store_true")
    parser.add_argument("--dump-wait-s", type=float, default=2.0)
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()

    if args.dump_live:
        if not args.container:
            raise SystemExit("--container is required with --dump-live")
        dump_live_counters(args.container, args.dump_wait_s)

    _, summary = evaluate_wiring(
        counters_path=args.counter_dump,
        commit_log_path=args.commit_log,
        commit_state_capture_path=args.commit_state_capture,
        commit_native_handoff_path=args.commit_native_handoff,
        src_native_handoff_path=args.src_native_handoff,
        scan_replay_path=args.scan_replay,
        atol=args.atol,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

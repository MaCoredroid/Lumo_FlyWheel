#!/usr/bin/env python3
"""Score the FR10 B4 temp/top-p three-way serving distribution gate."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lumo_flywheel_serving.fr10_equivalence_gate import (  # noqa: E402
    GateThresholds,
    StateParityRow,
    default_fr10_negative_controls,
    evaluate_negative_controls,
    evaluate_state_parity,
    load_sampling_artifact,
    sampling_distribution_distance,
    summarize_sampling_distance,
)


def _distance(left: str, right: str, *, positions: int, top: int) -> dict[str, Any]:
    rows = sampling_distribution_distance(
        load_sampling_artifact(left),
        load_sampling_artifact(right),
        positions=positions,
        per_prompt=True,
        top=top,
    )
    return {
        "left": left,
        "right": right,
        "summary": summarize_sampling_distance(rows),
        "rows": [row.__dict__ for row in rows],
    }


def _aggregate_by_position(distance: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(row["position"]): row
        for row in distance["rows"]
        if row.get("prompt_id") is None
    }


def _floor_relative_rows(
    *,
    cross: dict[str, Any],
    same: dict[str, Any],
    baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    cross_rows = _aggregate_by_position(cross)
    same_rows = _aggregate_by_position(same)
    baseline_rows = _aggregate_by_position(baseline)
    out: list[dict[str, Any]] = []
    for pos in sorted(set(cross_rows) | set(same_rows) | set(baseline_rows)):
        cross_tv = float(cross_rows.get(pos, {}).get("tv", 0.0))
        same_tv = float(same_rows.get(pos, {}).get("tv", 0.0))
        baseline_tv = float(baseline_rows.get(pos, {}).get("tv", 0.0))
        out.append(
            {
                "position": pos,
                "tree_vs_non_mtp_tv": cross_tv,
                "non_mtp_run1_vs_run2_tv": same_tv,
                "naive_mtp_vs_non_mtp_tv": baseline_tv,
                "tree_minus_same_tv": cross_tv - same_tv,
                "tree_minus_naive_tv": cross_tv - baseline_tv,
                "tree_top": cross_rows.get(pos, {}).get("left_top", ()),
                "non_mtp_top": cross_rows.get(pos, {}).get("right_top", ()),
            }
        )
    return out


def _lossy_artifact(src: str, out: str, *, token_id: int = 42424242) -> str:
    data = json.loads(Path(src).read_text(encoding="utf-8"))
    lossy = copy.deepcopy(data)
    lossy["arm"] = f"{data.get('arm', 'unknown')}_forced_lossy"
    lossy["fr10_known_lossy_control"] = {
        "kind": "force-first-token",
        "token_id": token_id,
        "source": src,
    }
    for row in lossy.get("records", []):
        token_ids = list(row.get("token_ids") or [])
        if token_ids:
            token_ids[0] = token_id
        else:
            token_ids = [token_id]
        row["token_ids"] = token_ids
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lossy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def _state_parity_report(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [
        StateParityRow(
            layer=0,
            node=2,
            max_state_abs=float(data["tree_kernel_vs_native_decode_update_path_state_abs"]),
            max_output_abs=float(data["tree_kernel_vs_native_decode_update_path_out_abs"]),
            tag="real_tensor_tree_vs_native_decode_update",
        )
    ]
    report = evaluate_state_parity(rows, thresholds=GateThresholds())
    return {
        "source": path,
        "passed": report.passed,
        "violations": report.violations,
        "metrics": report.metrics,
        "native_linear_tree_contamination_abs": float(
            data.get("native_linear_vs_tree_ref_non_linear_nodes_abs", 0.0)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--non-mtp-run1", required=True)
    parser.add_argument("--non-mtp-run2", required=True)
    parser.add_argument("--naive-mtp", required=True)
    parser.add_argument("--tree-mtp", required=True)
    parser.add_argument("--real-tensor-validation", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--positions", type=int, default=32)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument(
        "--lossy-out",
        default="output/fr10_three_way_sampling_gate/known_lossy_forced_first_token.json",
    )
    args = parser.parse_args()

    same = _distance(args.non_mtp_run1, args.non_mtp_run2, positions=args.positions, top=args.top)
    naive = _distance(args.naive_mtp, args.non_mtp_run1, positions=args.positions, top=args.top)
    tree = _distance(args.tree_mtp, args.non_mtp_run1, positions=args.positions, top=args.top)
    lossy_path = _lossy_artifact(args.tree_mtp, args.lossy_out)
    lossy = _distance(lossy_path, args.non_mtp_run1, positions=args.positions, top=args.top)

    same_tv = float(same["summary"]["max_aggregate_tv"])
    naive_tv = float(naive["summary"]["max_aggregate_tv"])
    tree_tv = float(tree["summary"]["max_aggregate_tv"])
    lossy_tv = float(lossy["summary"]["max_aggregate_tv"])
    floor_tv = max(same_tv, naive_tv)

    floor_rows = _floor_relative_rows(cross=tree, same=same, baseline=naive)
    max_tree_minus_same = max((row["tree_minus_same_tv"] for row in floor_rows), default=0.0)
    max_tree_minus_naive = max((row["tree_minus_naive_tv"] for row in floor_rows), default=0.0)
    state = _state_parity_report(args.real_tensor_validation)
    negative = evaluate_negative_controls(default_fr10_negative_controls())

    lossy_detected = lossy_tv > floor_tv
    passed = bool(
        tree_tv <= floor_tv
        and max_tree_minus_same <= max(0.0, naive_tv - same_tv)
        and state["passed"]
        and lossy_detected
    )
    result = {
        "schema": "fr10.three_way_sampling_gate.v1",
        "inputs": {
            "non_mtp_run1": args.non_mtp_run1,
            "non_mtp_run2": args.non_mtp_run2,
            "naive_mtp": args.naive_mtp,
            "tree_mtp": args.tree_mtp,
            "real_tensor_validation": args.real_tensor_validation,
            "known_lossy": lossy_path,
        },
        "positions": args.positions,
        "top": args.top,
        "pairwise": {
            "non_mtp_run1_vs_run2": same["summary"],
            "naive_mtp_vs_non_mtp": naive["summary"],
            "tree_mtp_vs_non_mtp": tree["summary"],
            "known_lossy_vs_non_mtp": lossy["summary"],
        },
        "floor_relative": {
            "same_regime_max_aggregate_tv": same_tv,
            "naive_baseline_max_aggregate_tv": naive_tv,
            "serving_floor_max_aggregate_tv": floor_tv,
            "tree_max_aggregate_tv": tree_tv,
            "tree_minus_same_max_aggregate_tv": tree_tv - same_tv,
            "tree_minus_naive_max_aggregate_tv": tree_tv - naive_tv,
            "max_position_tree_minus_same_tv": max_tree_minus_same,
            "max_position_tree_minus_naive_tv": max_tree_minus_naive,
            "rows": floor_rows,
        },
        "state_parity_l1": state,
        "known_lossy_power": {
            "passed": lossy_detected,
            "known_lossy_max_aggregate_tv": lossy_tv,
            "serving_floor_max_aggregate_tv": floor_tv,
            "margin_over_floor": lossy_tv - floor_tv,
        },
        "negative_controls_l1": {
            "passed": negative.passed,
            "violations": negative.violations,
            "metrics": negative.metrics,
        },
        "passed": passed,
        "violations": [],
    }
    if tree_tv > floor_tv:
        result["violations"].append(
            f"tree max aggregate TV {tree_tv:.6g} exceeds serving floor {floor_tv:.6g}"
        )
    if max_tree_minus_same > max(0.0, naive_tv - same_tv):
        result["violations"].append(
            "tree per-position TV exceeds same-regime floor by more than naive baseline"
        )
    if not state["passed"]:
        result["violations"].extend(state["violations"])
    if not lossy_detected:
        result["violations"].append("known-lossy control was not above serving floor")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["pairwise"], indent=2, sort_keys=True))
    print(json.dumps({"passed": passed, "violations": result["violations"]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

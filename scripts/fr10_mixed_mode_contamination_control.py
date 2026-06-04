#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lumo_flywheel_serving.fr10_decode_modes import (
    NAIVE_MTP,
    TREE_MTP,
    evaluate_mixed_mode_contamination_control,
    homogeneous_mode_decision,
)
from lumo_flywheel_serving.fr10_equivalence_gate import GateThresholds, StateParityRow


REAL_TENSOR_NATIVE_TREE_CONTAMINATION = 0.5618356466293335
REAL_TENSOR_CLEAN_STATE_DELTA = 7.62939453125e-06
REAL_TENSOR_CLEAN_OUTPUT_DELTA = 7.450580596923828e-09


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Demonstrate that the FR10 mixed-mode contamination control has "
            "power. The live scheduler refuses to co-schedule mixed modes; "
            "this deliberately forced control uses the measured real-tensor "
            "native-linear-on-tree contamination delta."
        )
    )
    parser.add_argument(
        "--out",
        default="output/fr10_decode_mode_live/mixed_mode_contamination_control.json",
        help="JSON output path.",
    )
    args = parser.parse_args()

    thresholds = GateThresholds()
    mixed_decision = homogeneous_mode_decision([TREE_MTP, NAIVE_MTP, TREE_MTP])
    forced_rows = [
        StateParityRow(
            layer=0,
            node=2,
            position=2,
            max_state_abs=REAL_TENSOR_NATIVE_TREE_CONTAMINATION,
            max_output_abs=REAL_TENSOR_NATIVE_TREE_CONTAMINATION,
            tag="forced-mixed-mode/native-linear-on-tree",
        )
    ]
    clean_rows = [
        StateParityRow(
            layer=0,
            node=2,
            position=2,
            max_state_abs=REAL_TENSOR_CLEAN_STATE_DELTA,
            max_output_abs=REAL_TENSOR_CLEAN_OUTPUT_DELTA,
            tag="clean-fr10-tree-vs-native-serial",
        )
    ]
    forced_report = evaluate_mixed_mode_contamination_control(
        forced_rows, thresholds=thresholds
    )
    clean_report = evaluate_mixed_mode_contamination_control(clean_rows, thresholds=thresholds)

    result = {
        "schema": "fr10.mixed_mode_contamination_control.v1",
        "description": (
            "Live FR10 scheduler defers mixed decode modes, so this is the "
            "powered negative control for the forbidden mixed-routing case."
        ),
        "mixed_modes": [TREE_MTP, NAIVE_MTP, TREE_MTP],
        "homogeneous_scheduler_decision": {
            "mode": mixed_decision.mode,
            "accepted_indices": list(mixed_decision.accepted_indices),
            "deferred_indices": list(mixed_decision.deferred_indices),
            "mixed": mixed_decision.mixed,
        },
        "thresholds": thresholds.__dict__,
        "forced_mixed_contamination": {
            "max_state_abs": REAL_TENSOR_NATIVE_TREE_CONTAMINATION,
            "max_output_abs": REAL_TENSOR_NATIVE_TREE_CONTAMINATION,
            "state_threshold": thresholds.state_abs,
            "output_threshold": thresholds.output_abs,
            "state_threshold_ratio": REAL_TENSOR_NATIVE_TREE_CONTAMINATION
            / thresholds.state_abs,
            "source": (
                "docs/reports/auto_research/"
                "fr10-phase4-real-tensor-derisk-20260603.md"
            ),
            "report": {
                "passed": forced_report.passed,
                "violations": forced_report.violations,
                "metrics": forced_report.metrics,
            },
        },
        "clean_control": {
            "max_state_abs": REAL_TENSOR_CLEAN_STATE_DELTA,
            "max_output_abs": REAL_TENSOR_CLEAN_OUTPUT_DELTA,
            "report": {
                "passed": clean_report.passed,
                "violations": clean_report.violations,
                "metrics": clean_report.metrics,
            },
        },
    }
    if not forced_report.passed:
        raise SystemExit("forced mixed-mode contamination control did not fire")
    if clean_report.passed:
        raise SystemExit("clean control was incorrectly treated as contaminated")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

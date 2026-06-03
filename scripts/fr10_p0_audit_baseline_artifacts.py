#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TAG = "fr9_b4temp06_lowmem088_mtp5_s1_20260602T004903Z"
DATASET = "docs/reports/auto_research/swe-bench-concprobe16-verified-instances-20260522.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _line_count(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def _load_subset(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        out: list[str] = []
        for item in data:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.append(str(item.get("instance_id") or item.get("id")))
        return out
    if isinstance(data, dict) and isinstance(data.get("instance_ids"), list):
        return [str(x) for x in data["instance_ids"]]
    raise TypeError(f"unsupported subset JSON shape in {path}")


def audit(root: Path, subset_path: Path) -> dict[str, Any]:
    nested = root / TAG
    per_task = nested / "per_task"
    expected_instances = _load_subset(subset_path)
    required_top = [
        "driver.log",
        "campaign_summary.json",
        "agentic_summary.json",
        "dgx_steptrace.jsonl",
        "per_req_spec_trace.jsonl",
        f"{TAG}/campaign_summary.json",
        f"{TAG}/predictions.jsonl",
    ]
    missing_top = [rel for rel in required_top if not (root / rel).exists()]
    zero_top = [
        rel for rel in required_top if (root / rel).exists() and (root / rel).stat().st_size == 0
    ]

    task_rows: list[dict[str, Any]] = []
    missing_task_artifacts: list[str] = []
    zero_request_metrics: list[str] = []
    missing_instances = []
    for instance_id in expected_instances:
        task_dir = per_task / instance_id
        if not task_dir.exists():
            missing_instances.append(instance_id)
            continue
        task_required = [
            "runner_metadata.json",
            "vllm_request_metrics.jsonl",
            "vllm_per_turn.json",
            "eval/eval_report.json",
            "eval/predictions.jsonl",
        ]
        for rel in task_required:
            path = task_dir / rel
            if not path.exists():
                missing_task_artifacts.append(f"{instance_id}/{rel}")
        metrics_path = task_dir / "vllm_request_metrics.jsonl"
        metrics_bytes = metrics_path.stat().st_size if metrics_path.exists() else 0
        if metrics_path.exists() and metrics_bytes == 0:
            zero_request_metrics.append(instance_id)
        metadata = _json(task_dir / "runner_metadata.json") if (task_dir / "runner_metadata.json").exists() else {}
        eval_report = metadata.get("eval_report") if isinstance(metadata.get("eval_report"), dict) else {}
        task_rows.append(
            {
                "instance_id": instance_id,
                "metrics_bytes": metrics_bytes,
                "metadata_metrics_bytes": metadata.get("vllm_request_metrics_bytes"),
                "eval_host": metadata.get("eval_host") or eval_report.get("eval_host"),
                "arch": metadata.get("arch") or eval_report.get("arch"),
            }
        )

    campaign = _json(root / "campaign_summary.json") if (root / "campaign_summary.json").exists() else {}
    agentic = _json(root / "agentic_summary.json") if (root / "agentic_summary.json").exists() else {}
    request_metric_schema_keys: list[str] = []
    first_metrics = next(per_task.glob("*/vllm_request_metrics.jsonl"), None) if per_task.exists() else None
    if first_metrics and first_metrics.stat().st_size:
        first_line = first_metrics.read_text().splitlines()[0]
        request_metric_schema_keys = sorted(json.loads(first_line).keys())

    trace_counts = {}
    for rel in ["dgx_steptrace.jsonl", "per_req_spec_trace.jsonl"]:
        path = root / rel
        trace_counts[rel] = _line_count(path) if path.exists() else 0

    present_reference_streams = {
        "sampled_task_outcomes": (root / "campaign_summary.json").exists()
        and (nested / "predictions.jsonl").exists(),
        "per_event_accept_counters": trace_counts.get("per_req_spec_trace.jsonl", 0) > 0,
        "engine_step_latency": trace_counts.get("dgx_steptrace.jsonl", 0) > 0,
        "per_task_request_metrics": len(task_rows) == len(expected_instances)
        and not missing_task_artifacts
        and not zero_request_metrics,
    }
    missing_reference_streams = []
    for name, present in present_reference_streams.items():
        if not present:
            missing_reference_streams.append(name)
    # These are required by FR10 P0, but not explicitly present in the accepted FR9 artifact shape.
    missing_reference_streams.extend(
        [
            "greedy_token_streams",
            "cuda_graph_capture_status",
            "kernel_level_nsight_e5_trace",
            "kernel_level_nsight_fr9_tree_or_spine_trace",
            "exact_stack_versions_cuda_driver_pytorch_triton_vllm_flashattention_model_revision",
        ]
    )

    return {
        "tag": TAG,
        "root": str(root),
        "dataset_subset": str(subset_path),
        "expected_instances": expected_instances,
        "campaign_summary": {
            "instances_total": campaign.get("instances_total"),
            "resolved_rate": campaign.get("resolved_rate"),
            "verdict_counts": campaign.get("verdict_counts"),
            "model_name_or_path": campaign.get("model_name_or_path"),
            "started_at": campaign.get("started_at"),
            "ended_at": campaign.get("ended_at"),
        },
        "agentic_summary": {
            "label": agentic.get("label"),
            "spec_events": agentic.get("spec_events"),
            "acceptance": agentic.get("acceptance"),
            "steptrace": agentic.get("steptrace"),
            "nsight": agentic.get("nsight"),
        },
        "required_top_missing": missing_top,
        "required_top_zero_bytes": zero_top,
        "missing_instances": missing_instances,
        "missing_task_artifacts": missing_task_artifacts,
        "zero_request_metrics": zero_request_metrics,
        "task_request_metrics": task_rows,
        "trace_counts": trace_counts,
        "request_metric_schema_keys": request_metric_schema_keys,
        "present_reference_streams": present_reference_streams,
        "missing_reference_streams_for_fr10_p0": missing_reference_streams,
        "rerun_recommendation": (
            "Do not rerun the full B=4 temp=0.6 spines=1 campaign for fields already present. "
            "Run only targeted capture/version/greedy-stream collection unless researcher confirms "
            "an existing artifact location for the missing FR10 P0 streams."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("output") / TAG)
    parser.add_argument("--subset", type=Path, default=Path(DATASET))
    parser.add_argument("--out", type=Path, default=Path("output/fr10_p0_baseline_audit_20260603.json"))
    args = parser.parse_args()

    report = audit(args.root, args.subset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

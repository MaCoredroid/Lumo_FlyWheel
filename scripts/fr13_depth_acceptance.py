#!/usr/bin/env python3
"""Reduce speculative acceptance by depth from real SWE task brackets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from fr13_fixed32_work_census import CensusError as WorkCensusError
from fr13_fixed32_work_census import CONV_PREGATHER_BLOCK
from fr13_fixed32_work_census import CONV_PREGATHER_LAYERS
from fr13_fixed32_work_census import CONV_PREGATHER_ROW_ELEMS
from fr13_fixed32_work_census import FIXED_WORK_SCOPE
from fr13_fixed32_work_census import MODE_SEMANTICS as WORK_CENSUS_MODE_SEMANTICS
from fr13_fixed32_work_census import REPORT_SCHEMA as WORK_CENSUS_REPORT_SCHEMA
from fr13_fixed32_work_census import SUPPORTED_BATCH_SIZES
from fr13_fixed32_work_census import forward_graph_structural_signature
from fr13_fixed32_work_census import load_jsonl as load_work_census_jsonl
from fr13_fixed32_work_census import reference_event
from fr13_fixed32_work_census import reference_terminal_summary
from fr13_fixed32_work_census import validate_campaign as validate_work_census_campaign


DRAFTS_METRIC = "vllm:spec_decode_num_drafts_total"
DRAFT_TOKENS_METRIC = "vllm:spec_decode_num_draft_tokens_total"
ACCEPTED_METRIC = "vllm:spec_decode_num_accepted_tokens_total"
POSITION_METRIC = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
SAMPLE_RE = re.compile(
    r"^(?P<name>[^\s{]+)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+0-9.eE]+)$"
)
POSITION_RE = re.compile(r'(?:^|,)position="(?P<position>\d+)"(?=,|$)')
CAMPAIGN_RE = re.compile(
    r"^=== .* dataset=.* n=(?P<task_count>\d+) "
    r"concurrency=(?P<concurrency>\d+) ===$"
)
CANONICAL_TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
    "astropy__astropy-13453",
    "astropy__astropy-13579",
    "astropy__astropy-13977",
    "astropy__astropy-14096",
    "astropy__astropy-14182",
    "astropy__astropy-14309",
    "astropy__astropy-14365",
    "astropy__astropy-14369",
    "astropy__astropy-14508",
    "astropy__astropy-14539",
    "astropy__astropy-14598",
    "astropy__astropy-14995",
)
CANONICAL_TASK_IDS_BY_COUNT = {
    4: CANONICAL_TASK_IDS[:4],
    16: CANONICAL_TASK_IDS,
}
FIXED32_RAW_ACCEPTANCE_POSITIONS = tuple(range(31))
FIXED32_FLOOR_GATE_SCHEMA = "fr13.canonical_swe_verified_fixed32_floor_gate.v8"
FIXED32_SLO_GATES = frozenset(
    {
        "tail6_fixed32_legacy_slo",
        "hydra27_fixed32_legacy_slo",
    }
)
FIXED32_REQUIRED_EVIDENCE_GATES = frozenset(
    {
        "source_runtime_fingerprint_equal",
        "external_artifact_fingerprint_equal",
        "arm_runtime_attestations_equal",
        "running_container_image_identity_exact",
        "task_metric_bracket_bytes_bound",
        "fixed32_pretask_zero_positive_traffic",
        "all_canonical_tasks_have_real_model_traffic",
        "all_positive_usage_proxy_pairs_task_bound",
        "all_task_agents_completed_cleanly",
        "all_tasks_have_terminal_swe_verdicts",
        "canonical_exact_4_or_16_task_binding",
        "canonical_completed_task_set",
        "canonical_subset_hash",
        "uncapped_sidecars",
        "sidecar_coverage_eq_1_0",
        "sidecar_counter_reconciliation",
        "fixed32_work_census_exact",
        "fixed32_per_batch_physical_work_equal",
        "fixed32_drafter_graph_lifecycle_exact_and_matched",
        "fixed32_forward_graph_pregather_exact",
        "fixed32_scope_limitations_explicit",
        "canonical_task_forward_union_covers_complete_stream",
        "fixed32_flush_generation_chain_exact",
        "fixed32_task_boundaries_exact",
        "b4_occupancy_exposure",
    }
)
FIXED32_FLOOR_GATE_TOP_KEYS = frozenset(
    {
        "schema",
        "analysis_valid",
        "gate_verdict",
        "repo",
        "runroot",
        "tag",
        "task_count",
        "inferred_concurrency",
        "source_runtime_fingerprint",
        "external_artifact_fingerprint",
        "matched_runtime_attestation",
        "fixed32_work_census",
        "slo_definition",
        "uncertainty_model",
        "evidence_requirements",
        "arms",
        "comparison",
        "gates",
    }
)
FIXED32_FLOOR_GATE_ARM_KEYS = frozenset(
    {
        "arm",
        "artifact_dir",
        "inferred_concurrency",
        "expected_draft_tokens_per_event",
        "active_logical_drafts_per_event",
        "valid_mask",
        "canonical_task_ids",
        "provenance",
        "sidecar",
        "flush_chain",
        "work_census_expected",
        "statistics",
    }
)
FIXED32_WORK_CENSUS_KEYS = frozenset(
    {
        "report",
        "physical_work_comparison",
        "drafter_graph_lifecycle",
        "forward_graph_pregather_lifecycle",
        "scope",
        "scope_interpretation",
        "files",
        "complete_terminal_stream_reconciled_to_sfwd_sidecar",
        "canonical_task_forward_counter_union_selected_posthoc",
        "canonical_task_forward_union_covers_complete_stream",
        "b4_occupancy_gate",
    }
)
WORK_CENSUS_REPORT_KEYS = frozenset(
    {
        "schema",
        "status",
        "required_batch_sizes",
        "event_counts",
        "batch_size_sequences",
        "forward_step_indices",
        "event_ids",
        "producer_pids",
        "terminal_summaries",
        "drafter_graph_registries",
        "forward_graph_registries",
        "conv_pregather_auxiliary",
        "physical_work_histograms",
        "scope",
        "semantic_modes",
        "normalized_work_signature",
        "normalized_work_signature_sha256",
    }
)


def bound_artifact_bytes(
    path: Path,
    expected_identity: dict,
    *,
    label: str,
) -> bytes:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"{path}: cannot read {label}: {error}") from error
    if not isinstance(expected_identity, dict):
        raise ValueError(f"{path}: {label} binding is not an object")
    if set(expected_identity) != {"path", "sha256", "bytes"}:
        raise ValueError(f"{path}: {label} binding fields are not exact")
    recorded_path = expected_identity.get("path")
    recorded_hash = expected_identity.get("sha256")
    recorded_bytes = expected_identity.get("bytes")
    if not isinstance(recorded_path, str) or recorded_path != str(path.resolve()):
        raise ValueError(f"{path}: bound {label} path does not match")
    if (
        not isinstance(recorded_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", recorded_hash) is None
    ):
        raise ValueError(f"{path}: bound {label} SHA-256 is invalid")
    if (
        isinstance(recorded_bytes, bool)
        or not isinstance(recorded_bytes, int)
        or recorded_bytes < 0
    ):
        raise ValueError(f"{path}: bound {label} byte count is invalid")
    if len(raw) != recorded_bytes:
        raise ValueError(
            f"{path}: current {label} byte count does not match floor gate"
        )
    if hashlib.sha256(raw).hexdigest() != recorded_hash:
        raise ValueError(
            f"{path}: current {label} SHA-256 does not match floor gate"
        )
    return raw


def metric_artifact_text(path: Path, expected_identity: dict | None = None) -> str:
    if expected_identity is None:
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise ValueError(
                f"{path}: cannot read metric artifact: {error}"
            ) from error
    else:
        raw = bound_artifact_bytes(
            path,
            expected_identity,
            label="metric artifact",
        )
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: metric artifact is not strict UTF-8") from error


def parse_metrics(
    path: Path,
    expected_identity: dict | None = None,
    *,
    expected_positions: tuple[int, ...] | None = None,
) -> tuple[dict[str, float], dict[int, float], dict[str, str]]:
    wanted_scalars = (DRAFTS_METRIC, DRAFT_TOKENS_METRIC, ACCEPTED_METRIC)
    scalars: dict[str, float] = {}
    positions: dict[int, float] = {}
    series_labels: dict[str, str] = {}
    for line in metric_artifact_text(path, expected_identity).splitlines():
        match = SAMPLE_RE.match(line)
        if match is None:
            if line.startswith((*wanted_scalars, POSITION_METRIC)):
                raise ValueError(f"{path}: malformed required metric line {line!r}")
            continue
        name = match.group("name")
        value = float(match.group("value"))
        labels = match.group("labels") or ""
        if name in wanted_scalars:
            if not math.isfinite(value) or value < 0 or not value.is_integer():
                raise ValueError(f"{path}: invalid counter value for {name}: {value}")
            if name in scalars:
                raise ValueError(f"{path}: duplicate series for {name}")
            scalars[name] = value
            series_labels[name] = labels
        elif name == POSITION_METRIC:
            position_matches = list(POSITION_RE.finditer(labels))
            if len(position_matches) != 1:
                raise ValueError(
                    f"{path}: {POSITION_METRIC} must have exactly one "
                    "integer position label"
                )
            if not math.isfinite(value) or value < 0 or not value.is_integer():
                raise ValueError(
                    f"{path}: invalid counter value for {name}: {value}"
                )
            index = int(position_matches[0].group("position"))
            if index in positions:
                raise ValueError(
                    f"{path}: duplicate {POSITION_METRIC} position {index}"
                )
            positions[index] = value
            series_labels[f"{POSITION_METRIC}:{index}"] = labels
    missing = [name for name in wanted_scalars if name not in scalars]
    if missing:
        raise ValueError(f"{path}: missing required metrics {missing!r}")
    if not positions:
        raise ValueError(f"{path}: missing {POSITION_METRIC}")
    contiguous_positions = tuple(range(max(positions) + 1))
    actual_positions = tuple(sorted(positions))
    if actual_positions != contiguous_positions:
        raise ValueError(f"{path}: acceptance positions are not contiguous")
    if expected_positions is not None and actual_positions != expected_positions:
        raise ValueError(
            f"{path}: raw acceptance positions must be exactly "
            f"{expected_positions[0]}..{expected_positions[-1]}; "
            f"found {actual_positions!r}"
        )
    return scalars, positions, series_labels


def reduce_window(
    pre_path: Path,
    post_path: Path,
    label: str,
    *,
    expected_pre_identity: dict | None = None,
    expected_post_identity: dict | None = None,
    expected_positions: tuple[int, ...] | None = None,
) -> dict:
    pre_scalars, pre_positions, pre_labels = parse_metrics(
        pre_path,
        expected_pre_identity,
        expected_positions=expected_positions,
    )
    post_scalars, post_positions, post_labels = parse_metrics(
        post_path,
        expected_post_identity,
        expected_positions=expected_positions,
    )
    if pre_labels != post_labels:
        raise ValueError(f"{label}: pre/post required metric labels do not match")
    if set(pre_positions) != set(post_positions):
        raise ValueError(f"{label}: pre/post acceptance depth sets do not match")
    scalar_deltas = {
        name: post_scalars[name] - pre_scalars[name] for name in pre_scalars
    }
    drafts = scalar_deltas[DRAFTS_METRIC]
    if drafts <= 0:
        raise ValueError(f"{label}: non-positive draft delta {drafts}")
    draft_tokens = scalar_deltas[DRAFT_TOKENS_METRIC]
    accepted_scalar = scalar_deltas[ACCEPTED_METRIC]
    if draft_tokens <= 0:
        raise ValueError(f"{label}: non-positive draft-token delta {draft_tokens}")
    if accepted_scalar < 0:
        raise ValueError(f"{label}: negative accepted-token delta {accepted_scalar}")

    max_position = max(set(pre_positions) | set(post_positions))
    counts = [
        post_positions.get(position, 0.0) - pre_positions.get(position, 0.0)
        for position in range(max_position + 1)
    ]
    if any(count < 0 for count in counts):
        raise ValueError(f"{label}: negative accepted-position delta")
    if counts and counts[0] > drafts:
        raise ValueError(
            f"{label}: depth-1 accepted count {counts[0]} exceeds drafts {drafts}"
        )
    if any(left < right for left, right in zip(counts, counts[1:])):
        raise ValueError(f"{label}: accepted-position deltas are not monotone")
    if not math.isclose(sum(counts), accepted_scalar, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"{label}: position sum {sum(counts)} != accepted delta {accepted_scalar}"
        )
    if accepted_scalar > draft_tokens:
        raise ValueError(
            f"{label}: accepted delta {accepted_scalar} exceeds "
            f"draft-token delta {draft_tokens}"
        )

    position_slots = len(counts)
    if expected_positions is None:
        while counts and counts[-1] == 0:
            counts.pop()
    return {
        "label": label,
        "drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted": sum(counts),
        "accept_per_event": sum(counts) / drafts,
        "position_slots": position_slots,
        "position_counts": counts,
    }


def reduce_task(
    task_dir: Path,
    metric_binding: dict | None = None,
    *,
    expected_positions: tuple[int, ...] | None = None,
) -> dict:
    expected_pre = metric_binding.get("pre") if metric_binding is not None else None
    expected_post = metric_binding.get("post") if metric_binding is not None else None
    task = reduce_window(
        task_dir / "vllm_metrics_pre.txt",
        task_dir / "vllm_metrics_post.txt",
        task_dir.name,
        expected_pre_identity=expected_pre,
        expected_post_identity=expected_post,
        expected_positions=expected_positions,
    )
    task["instance_id"] = task.pop("label")
    return task


def assert_topology_shape(
    window: dict,
    expected_tokens_per_draft: int,
    *,
    require_exact_position_slots: bool = False,
) -> None:
    expected = window["drafts"] * expected_tokens_per_draft
    if not math.isclose(window["draft_tokens"], expected, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"{window.get('label', window.get('instance_id', 'window'))}: "
            f"draft tokens/event is not exactly {expected_tokens_per_draft}; "
            f"draft_tokens={window['draft_tokens']} drafts={window['drafts']}"
        )
    if (
        require_exact_position_slots
        and window["position_slots"] != expected_tokens_per_draft
    ):
        raise ValueError(
            f"{window.get('label', window.get('instance_id', 'window'))}: "
            f"acceptance position slots must be exactly {expected_tokens_per_draft}; "
            f"found {window['position_slots']}"
        )


def retain_fixed32_position_counts(window: dict, expected_slots: int) -> None:
    counts = window["position_counts"]
    if len(counts) != expected_slots:
        raise ValueError(
            f"{window.get('label', window.get('instance_id', 'window'))}: "
            f"accepted depth {len(counts)} does not retain all "
            f"{expected_slots} fixed32 slots"
        )


def canonical_task_ids(required_task_count: int) -> tuple[str, ...]:
    task_ids = CANONICAL_TASK_IDS_BY_COUNT.get(required_task_count)
    if task_ids is None:
        raise ValueError(
            f"required task count must be exactly 4 or 16; got {required_task_count}"
        )
    return task_ids


def depth_rows(drafts: float, counts: list[float]) -> list[dict]:
    previous = drafts
    rows = []
    for position, count in enumerate(counts):
        rows.append(
            {
                "depth": position + 1,
                "accepted_count": count,
                "survival": count / drafts,
                "conditional": count / previous if previous else 0.0,
            }
        )
        previous = count
    return rows


def assert_arm_provenance(root: Path, expected_hydra: bool) -> None:
    env_path = root / "container_env.txt"
    if not env_path.is_file():
        raise ValueError(f"{root}: missing arm provenance {env_path}")
    values = [
        line.removeprefix("FR13_HYDRA23=")
        for line in env_path.read_text().splitlines()
        if line.startswith("FR13_HYDRA23=")
    ]
    expected = "1" if expected_hydra else "0"
    if values != [expected]:
        raise ValueError(f"{root}: expected FR13_HYDRA23={expected}, found {values!r}")


def fixed32_arm_spec(expected_hydra: bool) -> dict:
    import fr13_fixed32_topology as topology

    topology.validate_contract()
    shape = (
        topology.PHYSICAL_DRAFTS,
        topology.TAIL6_ACTIVE_DRAFTS,
        topology.HYDRA27_ACTIVE_DRAFTS,
    )
    if shape != (31, 21, 27):
        raise ValueError(
            "fixed32 depth reducer requires physical/active drafts=(31, 21, 27), "
            f"got {shape!r}"
        )
    if expected_hydra:
        return {
            "arm": "hydra27_fixed32",
            "mode": "hydra27_fixed32",
            "valid_mask": topology.HYDRA27_VALID_MASK,
            "active_drafts": topology.HYDRA27_ACTIVE_DRAFTS,
            "physical_drafts": topology.PHYSICAL_DRAFTS,
        }
    return {
        "arm": "tail21_fixed32",
        "mode": "tail6_fixed32",
        "valid_mask": topology.TAIL6_VALID_MASK,
        "active_drafts": topology.TAIL6_ACTIVE_DRAFTS,
        "physical_drafts": topology.PHYSICAL_DRAFTS,
    }


def assert_fixed32_arm_provenance(root: Path, expected_hydra: bool) -> dict:
    spec = fixed32_arm_spec(expected_hydra)
    env_path = root / "container_env.txt"
    if not env_path.is_file():
        raise ValueError(f"{root}: missing arm provenance {env_path}")
    expected = {
        "FR13_HYDRA23": "0",
        "FR13_FIXED32_MODE": spec["mode"],
        "FR13_FIXED32_VALID_MASK": f"{spec['valid_mask']:#010x}",
        "FR13_FIXED32_ACTIVE_NODES": str(spec["active_drafts"]),
        "FR13_FIXED32_PHYSICAL_DRAFTS": str(spec["physical_drafts"]),
    }
    lines = env_path.read_text().splitlines()
    for key, value in expected.items():
        values = [
            line.removeprefix(f"{key}=") for line in lines if line.startswith(f"{key}=")
        ]
        if values != [value]:
            raise ValueError(
                f"{root}: expected exactly {key}={value}, found {values!r}"
            )
    return spec


def assert_campaign_provenance(
    root: Path, required_task_count: int, concurrency: int
) -> None:
    log_path = root / "swe_orchestrator.log"
    if not log_path.is_file():
        raise ValueError(f"{root}: missing campaign provenance {log_path}")
    matches = [
        CAMPAIGN_RE.match(line)
        for line in log_path.read_text().splitlines()
        if CAMPAIGN_RE.match(line) is not None
    ]
    if len(matches) != 1:
        raise ValueError(f"{log_path}: expected exactly one campaign header")
    match = matches[0]
    assert match is not None
    recorded_task_count = int(match.group("task_count"))
    recorded_concurrency = int(match.group("concurrency"))
    if recorded_task_count != required_task_count:
        raise ValueError(
            f"{root}: requested {required_task_count} tasks but artifact records "
            f"{recorded_task_count}"
        )
    if recorded_concurrency != concurrency:
        raise ValueError(
            f"{root}: requested concurrency {concurrency} but artifact records "
            f"{recorded_concurrency}"
        )


def bracketed_task_dirs(root: Path, required_task_count: int) -> list[Path]:
    task_root = root / "swe_out" / "verified" / "per_task"
    if not task_root.is_dir():
        raise ValueError(f"{root}: missing task directory {task_root}")
    task_dirs = sorted(path for path in task_root.iterdir() if path.is_dir())
    if len(task_dirs) != required_task_count:
        raise ValueError(
            f"{root}: found {len(task_dirs)} task directories; "
            f"required exactly {required_task_count}"
        )
    for task_dir in task_dirs:
        pre = task_dir / "vllm_metrics_pre.txt"
        post = task_dir / "vllm_metrics_post.txt"
        if not pre.is_file() or not post.is_file():
            raise ValueError(f"{task_dir}: incomplete metrics bracket")
    expected_ids = sorted(canonical_task_ids(required_task_count))
    actual_ids = [task_dir.name for task_dir in task_dirs]
    if actual_ids != expected_ids:
        raise ValueError(
            f"{root}: task set does not match the canonical "
            f"{required_task_count}-task set; actual={actual_ids!r}"
        )
    return task_dirs


def reduce_arm(
    root: Path,
    required_task_count: int,
    concurrency: int,
    *,
    expected_hydra: bool,
    fixed32: bool = False,
    metric_bindings: dict | None = None,
) -> dict:
    canonical_ids = canonical_task_ids(required_task_count)
    if concurrency not in (1, 4):
        raise ValueError(f"unsupported concurrency {concurrency}; expected B1 or B4")
    fixed32_spec = None
    if fixed32:
        fixed32_spec = assert_fixed32_arm_provenance(root, expected_hydra)
        expected_tokens_per_draft = fixed32_spec["physical_drafts"]
        arm = fixed32_spec["arm"]
    else:
        assert_arm_provenance(root, expected_hydra)
        expected_tokens_per_draft = 23 if expected_hydra else 21
        arm = "hydra23" if expected_hydra else "tail6"
    assert_campaign_provenance(root, required_task_count, concurrency)
    task_dirs = bracketed_task_dirs(root, required_task_count)
    expected_task_ids = {task_dir.name for task_dir in task_dirs}
    if expected_task_ids != set(canonical_ids):
        raise ValueError(f"{root}: canonical task identity set drifted")
    if metric_bindings is not None:
        if (
            not isinstance(metric_bindings, dict)
            or set(metric_bindings) != expected_task_ids
        ):
            raise ValueError(
                f"{root}: bound metric task set does not match canonical brackets"
            )
        for task_id, bracket in metric_bindings.items():
            if not isinstance(bracket, dict) or set(bracket) != {"pre", "post"}:
                raise ValueError(
                    f"{root}: bound metric bracket for {task_id} is not exact"
                )
    common = {
        "root": str(root),
        "arm": arm,
        "evidence_set": f"canonical_swe_verified_{required_task_count}",
        "concurrency": concurrency,
        "expected_tokens_per_draft": expected_tokens_per_draft,
        "task_count": len(task_dirs),
        "instance_ids": [task_dir.name for task_dir in task_dirs],
    }
    if fixed32_spec is not None:
        common["fixed32_contract"] = {
            "mode": fixed32_spec["mode"],
            "valid_mask": f"{fixed32_spec['valid_mask']:#010x}",
            "active_drafts": fixed32_spec["active_drafts"],
            "physical_drafts": fixed32_spec["physical_drafts"],
        }
        common["raw_acceptance_positions"] = list(
            FIXED32_RAW_ACCEPTANCE_POSITIONS
        )

    expected_positions = FIXED32_RAW_ACCEPTANCE_POSITIONS if fixed32 else None
    if fixed32:
        for task_dir in task_dirs:
            bracket = (
                metric_bindings[task_dir.name]
                if metric_bindings is not None
                else {}
            )
            for snapshot in ("pre", "post"):
                parse_metrics(
                    task_dir / f"vllm_metrics_{snapshot}.txt",
                    bracket.get(snapshot),
                    expected_positions=expected_positions,
                )
    if concurrency == 1:
        tasks = [
            reduce_task(
                task_dir,
                metric_bindings[task_dir.name] if metric_bindings is not None else None,
                expected_positions=expected_positions,
            )
            for task_dir in task_dirs
        ]
        for task in tasks:
            assert_topology_shape(
                task,
                expected_tokens_per_draft,
                require_exact_position_slots=fixed32,
            )
            if fixed32:
                retain_fixed32_position_counts(task, expected_tokens_per_draft)
        drafts = sum(task["drafts"] for task in tasks)
        draft_tokens = sum(task["draft_tokens"] for task in tasks)
        max_depth = max(len(task["position_counts"]) for task in tasks)
        counts = [
            sum(
                task["position_counts"][position]
                if position < len(task["position_counts"])
                else 0.0
                for task in tasks
            )
            for position in range(max_depth)
        ]
        return {
            **common,
            "bracket_mode": "nonoverlapping_task_sum",
            "per_task_available": True,
            "drafts": drafts,
            "draft_tokens": draft_tokens,
            "accepted": sum(counts),
            "accept_per_event": sum(counts) / drafts,
            "depths": depth_rows(drafts, counts),
            "per_task": [
                {
                    **{
                        key: value
                        for key, value in task.items()
                        if key != "position_counts"
                    },
                    "depths": depth_rows(task["drafts"], task["position_counts"]),
                }
                for task in tasks
            ],
        }

    # Concurrent task brackets expose overlapping global Prometheus counters.
    # Reduce one union window; summing the per-task deltas double-counts time.
    earliest_pre = min(
        (task_dir / "vllm_metrics_pre.txt" for task_dir in task_dirs),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    latest_post = max(
        (task_dir / "vllm_metrics_post.txt" for task_dir in task_dirs),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    earliest_binding = (
        metric_bindings[earliest_pre.parent.name] if metric_bindings is not None else {}
    )
    latest_binding = (
        metric_bindings[latest_post.parent.name] if metric_bindings is not None else {}
    )
    window = reduce_window(
        earliest_pre,
        latest_post,
        "concurrent_union",
        expected_pre_identity=earliest_binding.get("pre"),
        expected_post_identity=latest_binding.get("post"),
        expected_positions=expected_positions,
    )
    assert_topology_shape(
        window,
        expected_tokens_per_draft,
        require_exact_position_slots=fixed32,
    )
    if fixed32:
        retain_fixed32_position_counts(window, expected_tokens_per_draft)
    counts = window.pop("position_counts")
    window.pop("label")
    return {
        **common,
        "bracket_mode": "union_earliest_pre_latest_post",
        "per_task_available": False,
        **window,
        "depths": depth_rows(window["drafts"], counts),
        "per_task": [],
        "union_window": {
            "pre": str(earliest_pre),
            "post": str(latest_post),
            "wall_seconds_by_metric_mtime": (
                latest_post.stat().st_mtime - earliest_pre.stat().st_mtime
            ),
        },
    }


def compare(tail: dict, hydra: dict) -> dict:
    legacy_pair = tail["arm"] == "tail6" and hydra["arm"] == "hydra23"
    pair_name = (
        "Tail6 and Hydra23" if legacy_pair else f"{tail['arm']} and {hydra['arm']}"
    )
    if tail["instance_ids"] != hydra["instance_ids"]:
        raise ValueError(f"{pair_name} task sets or order do not match")
    if len(tail["depths"]) != len(hydra["depths"]):
        raise ValueError(f"{pair_name} maximum acceptance depths do not match")
    tail_conditionals = [row["conditional"] for row in tail["depths"]]
    hydra_conditionals = [row["conditional"] for row in hydra["depths"]]

    def expected_accept(conditionals: list[float]) -> float:
        survival = 1.0
        accepted = 0.0
        for conditional in conditionals:
            survival *= conditional
            accepted += survival
        return accepted

    accept_gap = tail["accept_per_event"] - hydra["accept_per_event"]
    counterfactuals = []
    for position in range(len(tail_conditionals)):
        replaced = list(hydra_conditionals)
        replaced[position] = tail_conditionals[position]
        recovery = expected_accept(replaced) - expected_accept(hydra_conditionals)
        counterfactuals.append(
            {
                "depth": position + 1,
                "hydra_recovery_if_tail_conditional": recovery,
                "fraction_of_accept_gap": (
                    recovery / accept_gap if accept_gap > 0 else None
                ),
            }
        )

    task_cluster_depths = []
    if tail["per_task_available"] and hydra["per_task_available"]:
        for position in range(len(tail["depths"])):
            deltas = []
            for tail_task, hydra_task in zip(
                tail["per_task"], hydra["per_task"], strict=True
            ):
                if position < len(tail_task["depths"]) and position < len(
                    hydra_task["depths"]
                ):
                    deltas.append(
                        hydra_task["depths"][position]["conditional"]
                        - tail_task["depths"][position]["conditional"]
                    )
            negative = sum(delta < 0 for delta in deltas)
            positive = sum(delta > 0 for delta in deltas)
            non_ties = negative + positive
            sign_p = (
                sum(
                    math.comb(non_ties, count)
                    for count in range(negative, non_ties + 1)
                )
                / (2**non_ties)
                if non_ties
                else None
            )
            task_cluster_depths.append(
                {
                    "depth": position + 1,
                    "task_count_with_depth": len(deltas),
                    "hydra_lower_count": negative,
                    "hydra_higher_count": positive,
                    "tie_count": len(deltas) - non_ties,
                    "one_sided_sign_p_hydra_lower": sign_p,
                }
            )

    return {
        "hydra_minus_tail_accept_per_event": (
            hydra["accept_per_event"] - tail["accept_per_event"]
        ),
        "depths": [
            {
                "depth": tail_row["depth"],
                "survival_delta": hydra_row["survival"] - tail_row["survival"],
                "conditional_delta": (
                    hydra_row["conditional"] - tail_row["conditional"]
                ),
            }
            for tail_row, hydra_row in zip(tail["depths"], hydra["depths"], strict=True)
        ],
        "one_depth_at_a_time_counterfactuals": counterfactuals,
        "task_cluster_depths": task_cluster_depths,
        "interpretation": (
            "Depth substitutions are descriptive, non-additive counterfactuals, "
            "not causal attribution. Sign tests use task clusters when B1 "
            "per-task brackets are non-overlapping."
        ),
    }


def exact_json(path: Path) -> tuple[dict, bytes]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{path}: non-finite JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    if not path.is_file():
        raise ValueError(f"missing fixed32 floor-gate report: {path}")
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        payload = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid fixed32 floor-gate JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: fixed32 floor-gate report is not an object")
    return payload, raw


def locate_floor_gate_report(
    explicit_path: Path | None,
    tail_root: Path,
    hydra_root: Path,
) -> Path:
    tail_parent = tail_root.resolve().parent
    hydra_parent = hydra_root.resolve().parent
    if tail_parent != hydra_parent:
        raise ValueError(
            "fixed32 arm roots do not share a runroot; "
            f"tail_parent={tail_parent} hydra_parent={hydra_parent}"
        )
    if explicit_path is not None:
        return explicit_path.resolve()
    return tail_parent / "fixed32_floor_gate.json"


def require_exact_keys(value: object, expected: frozenset[str], label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected an object")
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{label}: fields are not exact; "
            f"missing={sorted(expected - actual)!r} "
            f"unknown={sorted(actual - expected)!r}"
        )
    return value


def canonical_json_sha256(value: object) -> str:
    try:
        canonical = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ValueError(f"value is not canonical JSON: {error}") from error
    return hashlib.sha256(canonical).hexdigest()


def strict_json_text(text: str, *, label: str) -> object:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{label}: non-finite JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label}: invalid strict JSON: {error}") from error


def positive_token_usage(value: object) -> bool:
    if isinstance(value, list):
        return any(positive_token_usage(item) for item in value)
    if not isinstance(value, dict):
        return False
    usage = value.get("usage")
    if isinstance(usage, dict):
        for key in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "prompt_tokens",
            "completion_tokens",
            "cache_read_input_tokens",
        ):
            count = usage.get(key)
            if type(count) is int and count > 0:
                return True
    return any(positive_token_usage(item) for item in value.values())


def trace_has_assistant_output(event: object) -> bool:
    if not isinstance(event, dict):
        return False
    event_type = event.get("type")
    if event_type == "assistant":
        message = event.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            return False
        content = message.get("content")
    elif event_type == "message" and event.get("role") == "assistant":
        content = event.get("content")
    elif event_type == "item.completed":
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") != "agent_message":
            return False
        content = item.get("text", item.get("content"))
    else:
        return False
    if isinstance(content, str):
        return bool(content.strip())
    if not isinstance(content, list):
        return False
    for item in content:
        if isinstance(item, str) and item.strip():
            return True
        if not isinstance(item, dict):
            continue
        if any(
            isinstance(item.get(key), str) and item[key].strip()
            for key in ("text", "thinking", "output_text")
        ):
            return True
        if item.get("type") in {"tool_use", "function_call"} and str(
            item.get("name", "")
        ).strip():
            return True
    return False


def validate_work_census_v5_report(
    report: object,
    *,
    required_batch: int,
) -> dict[str, Any]:
    report = require_exact_keys(
        report,
        WORK_CENSUS_REPORT_KEYS,
        "fixed32 work-census v5 report",
    )
    if (
        report["schema"] != WORK_CENSUS_REPORT_SCHEMA
        or report["status"] != "PASS"
        or report["required_batch_sizes"] != [required_batch]
        or report["scope"] != FIXED_WORK_SCOPE
        or report["semantic_modes"] != WORK_CENSUS_MODE_SEMANTICS
        or not isinstance(report["normalized_work_signature"], dict)
        or not isinstance(report["normalized_work_signature_sha256"], str)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            report["normalized_work_signature_sha256"],
        )
        is None
        or canonical_json_sha256(report["normalized_work_signature"])
        != report["normalized_work_signature_sha256"]
    ):
        raise ValueError("fixed32 work-census v5 report contract mismatch")

    modes = ("tail6_fixed32", "hydra27_fixed32")
    mode_keys = frozenset(modes)
    for field in (
        "event_counts",
        "batch_size_sequences",
        "forward_step_indices",
        "event_ids",
        "producer_pids",
        "terminal_summaries",
        "drafter_graph_registries",
        "forward_graph_registries",
        "conv_pregather_auxiliary",
        "physical_work_histograms",
    ):
        require_exact_keys(report[field], mode_keys, f"work census {field}")

    histograms = report["physical_work_histograms"]
    event_counts = report["event_counts"]
    batch_keys = frozenset(str(batch) for batch in SUPPORTED_BATCH_SIZES)
    observed_by_mode: dict[str, set[int]] = {}
    signatures_by_mode: dict[str, dict[int, str]] = {}
    for mode in modes:
        counts = event_counts[mode]
        if not isinstance(counts, dict) or any(
            key not in batch_keys
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for key, value in counts.items()
        ):
            raise ValueError(f"{mode}: work-census event counts are malformed")
        sequence = report["batch_size_sequences"][mode]
        forward_steps = report["forward_step_indices"][mode]
        event_ids = report["event_ids"][mode]
        event_total = sum(counts.values())
        if (
            not isinstance(sequence, list)
            or len(sequence) != event_total
            or any(
                isinstance(batch, bool)
                or not isinstance(batch, int)
                or batch not in SUPPORTED_BATCH_SIZES
                for batch in sequence
            )
            or {
                str(batch): sequence.count(batch)
                for batch in SUPPORTED_BATCH_SIZES
                if batch in sequence
            }
            != counts
            or not isinstance(forward_steps, list)
            or len(forward_steps) != event_total
            or any(
                isinstance(step, bool) or not isinstance(step, int) or step < 0
                for step in forward_steps
            )
            or forward_steps != sorted(set(forward_steps))
            or not isinstance(event_ids, list)
            or len(event_ids) != event_total
            or len(set(event_ids)) != event_total
            or not all(isinstance(event_id, str) and event_id for event_id in event_ids)
            or isinstance(report["producer_pids"][mode], bool)
            or not isinstance(report["producer_pids"][mode], int)
            or report["producer_pids"][mode] <= 0
        ):
            raise ValueError(f"{mode}: work-census event stream is malformed")

        mode_histogram = require_exact_keys(
            histograms[mode],
            batch_keys,
            f"physical_work_histograms.{mode}",
        )
        observed_by_mode[mode] = set()
        signatures_by_mode[mode] = {}
        for batch in SUPPORTED_BATCH_SIZES:
            batch_key = str(batch)
            entry = require_exact_keys(
                mode_histogram[batch_key],
                frozenset({"event_count", "normalized_event_signatures"}),
                f"physical_work_histograms.{mode}.{batch_key}",
            )
            event_count = entry["event_count"]
            signatures = entry["normalized_event_signatures"]
            expected_count = counts.get(batch_key, 0)
            if (
                isinstance(event_count, bool)
                or not isinstance(event_count, int)
                or event_count < 0
                or event_count != expected_count
                or not isinstance(signatures, dict)
                or any(
                    not isinstance(signature, str)
                    or re.fullmatch(r"[0-9a-f]{64}", signature) is None
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or count <= 0
                    for signature, count in signatures.items()
                )
                or sum(signatures.values()) != event_count
                or (event_count == 0 and bool(signatures))
                or (event_count > 0 and len(signatures) != 1)
            ):
                raise ValueError(
                    f"{mode}: B{batch} physical-work histogram is malformed"
                )
            if event_count:
                observed_by_mode[mode].add(batch)
                signatures_by_mode[mode][batch] = next(iter(signatures))

    tail_mode, hydra_mode = modes
    if observed_by_mode[tail_mode] != observed_by_mode[hydra_mode]:
        raise ValueError("Tail/Hydra occupied work-census batch sets differ")
    observed_batches = sorted(observed_by_mode[tail_mode])
    if required_batch not in observed_batches:
        raise ValueError(f"fixed32 work census lacks required B{required_batch}")
    physical_per_batch = {}
    for batch in observed_batches:
        tail_signature = signatures_by_mode[tail_mode][batch]
        hydra_signature = signatures_by_mode[hydra_mode][batch]
        if tail_signature != hydra_signature:
            raise ValueError(
                f"B{batch}: Tail/Hydra normalized physical-work SHA differs"
            )
        physical_per_batch[str(batch)] = {
            "normalized_event_signature_sha256": tail_signature,
            "tail_event_count": histograms[tail_mode][str(batch)]["event_count"],
            "hydra_event_count": histograms[hydra_mode][str(batch)]["event_count"],
        }

    registry_keys = frozenset(
        {
            "batch_size",
            "graph_signature",
            "captures",
            "capture_origin",
            "measured_replays",
            "unmeasured_replays",
        }
    )
    registries_by_mode: dict[str, dict[int, dict[str, Any]]] = {}
    for mode in modes:
        rows = report["drafter_graph_registries"][mode]
        terminal = report["terminal_summaries"][mode]
        if (
            not isinstance(rows, list)
            or not isinstance(terminal, dict)
            or terminal.get("drafter_graph_registry") != rows
            or terminal.get("scope") != FIXED_WORK_SCOPE
            or terminal.get("producer_pid") != report["producer_pids"][mode]
            or terminal.get("event_count") != sum(event_counts[mode].values())
        ):
            raise ValueError(f"{mode}: terminal drafter registry/scope mismatch")
        registries_by_mode[mode] = {}
        ordered_batches = []
        for index, raw_row in enumerate(rows):
            row = require_exact_keys(
                raw_row,
                registry_keys,
                f"drafter_graph_registries.{mode}[{index}]",
            )
            batch = row["batch_size"]
            if (
                isinstance(batch, bool)
                or not isinstance(batch, int)
                or batch not in SUPPORTED_BATCH_SIZES
                or batch in registries_by_mode[mode]
                or not isinstance(row["graph_signature"], str)
                or re.fullmatch(r"[0-9a-f]{64}", row["graph_signature"]) is None
                or type(row["captures"]) is not int
                or row["captures"] != 1
                or row["capture_origin"] not in {"measured", "unmeasured"}
                or any(
                    isinstance(row[key], bool)
                    or not isinstance(row[key], int)
                    or row[key] < 0
                    for key in ("measured_replays", "unmeasured_replays")
                )
                or row["measured_replays"]
                != histograms[mode][str(batch)]["event_count"]
            ):
                raise ValueError(f"{mode}: drafter registry row {index} is invalid")
            ordered_batches.append(batch)
            registries_by_mode[mode][batch] = row
        if ordered_batches != sorted(ordered_batches):
            raise ValueError(f"{mode}: drafter registry rows are not sorted")

    if set(registries_by_mode[tail_mode]) != set(registries_by_mode[hydra_mode]):
        raise ValueError("Tail/Hydra drafter graph registry batch sets differ")
    lifecycle_per_batch = {}
    for batch in sorted(registries_by_mode[tail_mode]):
        tail_row = registries_by_mode[tail_mode][batch]
        hydra_row = registries_by_mode[hydra_mode][batch]
        if (
            tail_row["graph_signature"] != hydra_row["graph_signature"]
            or tail_row["capture_origin"] != hydra_row["capture_origin"]
        ):
            raise ValueError(f"B{batch}: Tail/Hydra drafter graph lifecycle differs")
        lifecycle_per_batch[str(batch)] = {
            "graph_signature": tail_row["graph_signature"],
            "captures_per_arm": 1,
            "capture_origin": tail_row["capture_origin"],
            "tail_measured_replays": tail_row["measured_replays"],
            "hydra_measured_replays": hydra_row["measured_replays"],
            "tail_unmeasured_replays": tail_row["unmeasured_replays"],
            "hydra_unmeasured_replays": hydra_row["unmeasured_replays"],
        }

    forward_registry_keys = frozenset(
        {
            "batch_size",
            "graph_signature",
            "conv_layout_sha256",
            "captures",
            "capture_origin",
            "stage_calls",
            "stage_before_all_consumes",
            "layers",
            "requests",
            "row_elems",
            "programs",
            "ssi_pointer_entries",
            "ssi_groups",
            "source_validations",
            "staged_rows",
            "consume_calls",
            "consume_hits",
            "consume_fallbacks",
            "freshness_matches",
            "measured_replays",
        }
    )
    auxiliary_keys = frozenset(
        {
            "profile_capture_stages",
            "aux_capture_stages",
            "host_actual_stages",
            "host_actual_stages_by_batch",
        }
    )
    zero_by_batch = {
        str(batch): 0 for batch in SUPPORTED_BATCH_SIZES
    }
    forward_by_mode: dict[str, dict[int, dict[str, Any]]] = {}
    nonpure_dispatch_by_mode: dict[str, dict[str, int]] = {}
    nonpure_committer_replays_by_mode: dict[str, dict[str, int]] = {}
    nonpure_dispatch_keys = frozenset(
        {
            "guarded_steps",
            "piecewise_steps",
            "none_steps",
            "forbidden_full_steps",
        }
    )
    for mode in modes:
        rows = report["forward_graph_registries"][mode]
        terminal = report["terminal_summaries"][mode]
        auxiliary = report["conv_pregather_auxiliary"][mode]
        if (
            not isinstance(rows, list)
            or not rows
            or not isinstance(terminal, dict)
            or terminal.get("forward_graph_registry") != rows
            or terminal.get("conv_pregather_auxiliary") != auxiliary
        ):
            raise ValueError(
                f"{mode}: terminal forward graph pregather proof mismatch"
            )
        nonpure_dispatch = require_exact_keys(
            terminal.get("nonpure_dispatch"),
            nonpure_dispatch_keys,
            f"terminal_summaries.{mode}.nonpure_dispatch",
        )
        if (
            any(
                type(nonpure_dispatch[key]) is not int
                or nonpure_dispatch[key] < 0
                for key in nonpure_dispatch_keys
            )
            or nonpure_dispatch["guarded_steps"]
            != (
                nonpure_dispatch["piecewise_steps"]
                + nonpure_dispatch["none_steps"]
                + nonpure_dispatch["forbidden_full_steps"]
            )
            or nonpure_dispatch["forbidden_full_steps"] != 0
        ):
            raise ValueError(
                f"{mode}: terminal nonpure dispatch counts do not reconcile"
            )
        nonpure_dispatch_by_mode[mode] = dict(nonpure_dispatch)
        nonpure_committer = require_exact_keys(
            terminal.get("nonpure_committer_replays_by_batch"),
            batch_keys,
            (
                f"terminal_summaries.{mode}."
                "nonpure_committer_replays_by_batch"
            ),
        )
        if (
            any(
                type(nonpure_committer[key]) is not int
                or nonpure_committer[key] < 0
                for key in batch_keys
            )
            or sum(nonpure_committer.values())
            > nonpure_dispatch["guarded_steps"]
        ):
            raise ValueError(
                f"{mode}: terminal nonpure committer counts are invalid"
            )
        nonpure_committer_replays_by_mode[mode] = dict(nonpure_committer)
        auxiliary = require_exact_keys(
            auxiliary,
            auxiliary_keys,
            f"conv_pregather_auxiliary.{mode}",
        )
        if (
            any(
                type(auxiliary[key]) is not int or auxiliary[key] != 0
                for key in (
                    "profile_capture_stages",
                    "aux_capture_stages",
                    "host_actual_stages",
                )
            )
            or auxiliary["host_actual_stages_by_batch"] != zero_by_batch
            or not isinstance(
                auxiliary["host_actual_stages_by_batch"], dict
            )
            or any(
                type(value) is not int
                for value in auxiliary[
                    "host_actual_stages_by_batch"
                ].values()
            )
        ):
            raise ValueError(
                f"{mode}: pregather auxiliary/host stage counts are not zero"
            )
        forward_by_mode[mode] = {}
        ordered_batches: list[int] = []
        signatures: set[str] = set()
        layout_signatures: set[str] = set()
        for index, raw_row in enumerate(rows):
            row = require_exact_keys(
                raw_row,
                forward_registry_keys,
                f"forward_graph_registries.{mode}[{index}]",
            )
            batch = row["batch_size"]
            signature = row["graph_signature"]
            layout_signature = row["conv_layout_sha256"]
            if (
                type(batch) is not int
                or batch not in SUPPORTED_BATCH_SIZES
                or batch in forward_by_mode[mode]
                or not isinstance(signature, str)
                or re.fullmatch(r"[0-9a-f]{64}", signature) is None
                or signature != forward_graph_structural_signature(batch)
                or signature in signatures
                or not isinstance(layout_signature, str)
                or re.fullmatch(r"[0-9a-f]{64}", layout_signature) is None
                or layout_signature in layout_signatures
            ):
                raise ValueError(
                    f"{mode}: forward graph registry row {index} identity is invalid"
                )
            expected_programs = (
                CONV_PREGATHER_LAYERS
                * batch
                * (
                    (
                        CONV_PREGATHER_ROW_ELEMS
                        + CONV_PREGATHER_BLOCK
                        - 1
                    )
                    // CONV_PREGATHER_BLOCK
                )
            )
            expected_row = {
                "batch_size": batch,
                "captures": 1,
                "capture_origin": "final_full",
                "stage_calls": 1,
                "stage_before_all_consumes": True,
                "layers": CONV_PREGATHER_LAYERS,
                "requests": batch,
                "row_elems": CONV_PREGATHER_ROW_ELEMS,
                "programs": expected_programs,
                "ssi_pointer_entries": CONV_PREGATHER_LAYERS,
                "ssi_groups": 3,
                "source_validations": CONV_PREGATHER_LAYERS,
                "staged_rows": CONV_PREGATHER_LAYERS * batch,
                "consume_calls": CONV_PREGATHER_LAYERS,
                "consume_hits": CONV_PREGATHER_LAYERS,
                "consume_fallbacks": 0,
                "freshness_matches": CONV_PREGATHER_LAYERS,
                "measured_replays": histograms[mode][str(batch)][
                    "event_count"
                ],
            }
            if any(
                row[key] != expected
                or (
                    isinstance(expected, int)
                    and not isinstance(expected, bool)
                    and type(row[key]) is not int
                )
                or (
                    isinstance(expected, bool)
                    and type(row[key]) is not bool
                )
                for key, expected in expected_row.items()
            ):
                raise ValueError(
                    f"{mode}: forward graph registry row {index} "
                    "does not prove one ordered final-FULL pregather capture"
                )
            ordered_batches.append(batch)
            signatures.add(signature)
            layout_signatures.add(layout_signature)
            forward_by_mode[mode][batch] = row
        if ordered_batches != list(range(1, required_batch + 1)):
            raise ValueError(
                f"{mode}: forward graph registry must be exact B1.."
                f"B{required_batch}"
            )
        if not observed_by_mode[mode].issubset(forward_by_mode[mode]):
            raise ValueError(
                f"{mode}: forward graph registry does not cover occupied batches"
            )

    forward_per_batch = {}
    for batch in range(1, required_batch + 1):
        tail_row = forward_by_mode[tail_mode][batch]
        hydra_row = forward_by_mode[hydra_mode][batch]
        if (
            tail_row["graph_signature"] != hydra_row["graph_signature"]
            or tail_row["conv_layout_sha256"]
            != hydra_row["conv_layout_sha256"]
        ):
            raise ValueError(
                f"B{batch}: Tail/Hydra final-FULL forward graph/layout "
                "signatures differ"
            )
        forward_per_batch[str(batch)] = {
            "graph_signature": tail_row["graph_signature"],
            "conv_layout_sha256": tail_row["conv_layout_sha256"],
            "captures_per_arm": 1,
            "capture_origin": "final_full",
            "stage_calls_per_capture": 1,
            "stage_before_all_consumes": True,
            "layers": CONV_PREGATHER_LAYERS,
            "requests": batch,
            "row_elems": CONV_PREGATHER_ROW_ELEMS,
            "programs": tail_row["programs"],
            "ssi_pointer_entries": CONV_PREGATHER_LAYERS,
            "ssi_groups": 3,
            "source_validations": CONV_PREGATHER_LAYERS,
            "staged_rows": CONV_PREGATHER_LAYERS * batch,
            "consume_calls": CONV_PREGATHER_LAYERS,
            "consume_hits": CONV_PREGATHER_LAYERS,
            "consume_fallbacks": 0,
            "freshness_matches": CONV_PREGATHER_LAYERS,
            "tail_measured_replays": tail_row["measured_replays"],
            "hydra_measured_replays": hydra_row["measured_replays"],
        }

    return {
        "physical_work_comparison": {
            "observed_batch_sizes": observed_batches,
            "per_batch": physical_per_batch,
            "event_counts_compared": False,
            "one_normalized_signature_per_occupied_batch": True,
            "signature_keys_equal_across_arms": True,
        },
        "drafter_graph_lifecycle": {
            "registry_batch_sizes": sorted(registries_by_mode[tail_mode]),
            "per_batch": lifecycle_per_batch,
            "graph_signature_and_capture_origin_equal_across_arms": True,
            "replay_counts_may_differ": True,
        },
        "forward_graph_pregather_lifecycle": {
            "registry_batch_sizes": list(range(1, required_batch + 1)),
            "per_batch": forward_per_batch,
            "one_final_full_capture_per_batch_per_arm": True,
            "graph_signatures_unique_within_each_arm": True,
            "conv_layout_signatures_unique_within_each_arm": True,
            "graph_signatures_equal_across_arms_per_batch": True,
            "conv_layout_signatures_equal_across_arms_per_batch": True,
            "measured_replays_match_event_histograms": True,
            "stage_precedes_all_layer_consumes": True,
            "profile_auxiliary_and_host_stage_counts_zero": True,
            "nonpure_dispatch_by_mode": nonpure_dispatch_by_mode,
            "nonpure_committer_replays_by_mode": (
                nonpure_committer_replays_by_mode
            ),
            "forbidden_mixed_full_dispatches_zero": True,
        },
        "scope": json.loads(json.dumps(FIXED_WORK_SCOPE)),
    }


def validate_real_task_artifacts(
    root: Path,
    real_tasks: object,
    expected_ids: list[str],
) -> int:
    if not isinstance(real_tasks, dict):
        raise ValueError(f"{root}: floor-gate real-task provenance is missing")
    tasks = real_tasks.get("tasks")
    if not isinstance(tasks, dict) or list(tasks) != expected_ids:
        raise ValueError(f"{root}: real-task IDs are not the exact canonical order")
    for key in (
        "all_canonical_tasks_have_real_model_traffic",
        "all_positive_usage_proxy_pairs_task_bound",
        "all_agents_completed_cleanly",
        "all_tasks_have_terminal_eval_verdicts",
    ):
        if real_tasks.get(key) is not True:
            raise ValueError(f"{root}: real-task evidence {key!r} is not true")

    proxy_paths: set[Path] = set()
    artifact_count = 0
    task_keys = frozenset(
        {
            "trace_path",
            "trace_sha256",
            "trace_bytes",
            "trace_events",
            "assistant_output_events",
            "positive_token_usage",
            "agent_terminal",
            "eval_verdict",
            "proxy_pairs",
        }
    )
    proxy_keys = frozenset(
        {"path", "sha256", "bytes", "response_positive_token_usage"}
    )
    for task_id in expected_ids:
        task = require_exact_keys(
            tasks[task_id],
            task_keys,
            f"{root}: real task {task_id}",
        )
        trace_path = (
            root
            / "swe_out"
            / "verified"
            / "per_task"
            / task_id
            / "qwen_trace.jsonl"
        ).resolve()
        raw_trace = bound_artifact_bytes(
            trace_path,
            {
                "path": task["trace_path"],
                "sha256": task["trace_sha256"],
                "bytes": task["trace_bytes"],
            },
            label="task trace",
        )
        try:
            trace_text = raw_trace.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{trace_path}: task trace is not strict UTF-8") from error
        trace_events = []
        for line_number, line in enumerate(trace_text.splitlines(), start=1):
            if not line.strip():
                raise ValueError(f"{trace_path}:{line_number}: blank JSONL record")
            trace_events.append(
                strict_json_text(
                    line,
                    label=f"{trace_path}:{line_number}",
                )
            )
        assistant_output_events = sum(
            trace_has_assistant_output(event) for event in trace_events
        )
        if (
            type(task["trace_events"]) is not int
            or task["trace_events"] <= 0
            or task["trace_events"] != len(trace_events)
            or type(task["assistant_output_events"]) is not int
            or task["assistant_output_events"] <= 0
            or task["assistant_output_events"] != assistant_output_events
            or task["assistant_output_events"] > task["trace_events"]
            or task["positive_token_usage"] is not True
            or not positive_token_usage(trace_events)
            or not isinstance(task["agent_terminal"], dict)
            or task["eval_verdict"] not in {"resolved", "failed"}
            or not isinstance(task["proxy_pairs"], list)
            or not task["proxy_pairs"]
        ):
            raise ValueError(f"{root}: real task {task_id} evidence is incomplete")
        artifact_count += 1
        for index, raw_proxy in enumerate(task["proxy_pairs"]):
            proxy = require_exact_keys(
                raw_proxy,
                proxy_keys,
                f"{root}: real task {task_id} proxy {index}",
            )
            if proxy["response_positive_token_usage"] is not True:
                raise ValueError(
                    f"{root}: real task {task_id} proxy {index} lacks traffic"
                )
            if not isinstance(proxy["path"], str):
                raise ValueError(
                    f"{root}: real task {task_id} proxy {index} path is invalid"
                )
            proxy_path = Path(proxy["path"]).resolve()
            if proxy_path.parent != (root / "proxy_pair_dumps").resolve():
                raise ValueError(
                    f"{root}: real task {task_id} proxy path is outside its arm"
                )
            if proxy_path in proxy_paths:
                raise ValueError(f"{root}: proxy artifact is bound more than once")
            proxy_paths.add(proxy_path)
            raw_proxy = bound_artifact_bytes(
                proxy_path,
                {
                    "path": proxy["path"],
                    "sha256": proxy["sha256"],
                    "bytes": proxy["bytes"],
                },
                label="proxy artifact",
            )
            try:
                proxy_text = raw_proxy.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"{proxy_path}: proxy artifact is not strict UTF-8"
                ) from error
            proxy_payload = strict_json_text(proxy_text, label=str(proxy_path))
            request = (
                proxy_payload.get("request")
                if isinstance(proxy_payload, dict)
                else None
            )
            response = (
                proxy_payload.get("response")
                if isinstance(proxy_payload, dict)
                else None
            )
            if (
                not isinstance(proxy_payload, dict)
                or proxy_payload.get("schema") != "lumo.fr13.proxy_pair_dump.v1"
                or not isinstance(request, dict)
                or not isinstance(response, dict)
                or not positive_token_usage(response)
            ):
                raise ValueError(
                    f"{proxy_path}: proxy artifact does not prove model traffic"
                )
            request_text = json.dumps(
                request,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            matched_tasks = [
                candidate
                for candidate in expected_ids
                if f"# SWE-Bench task: {candidate}" in request_text
            ]
            if matched_tasks != [task_id]:
                raise ValueError(
                    f"{proxy_path}: proxy artifact is not bound to exactly "
                    f"task {task_id}"
                )
            artifact_count += 1
    return artifact_count


def validate_floor_gate_binding(
    path: Path,
    tail_root: Path,
    hydra_root: Path,
    *,
    required_task_count: int,
    concurrency: int,
) -> tuple[dict, dict[str, dict]]:
    payload, raw = exact_json(path)
    require_exact_keys(
        payload,
        FIXED32_FLOOR_GATE_TOP_KEYS,
        f"{path}: floor-gate v8 report",
    )
    if payload.get("schema") != FIXED32_FLOOR_GATE_SCHEMA:
        raise ValueError(
            f"{path}: expected floor-gate schema {FIXED32_FLOOR_GATE_SCHEMA!r}"
        )
    if payload.get("analysis_valid") is not True:
        raise ValueError(f"{path}: fixed32 floor-gate analysis is not valid")

    runroot = tail_root.resolve().parent
    recorded_runroot = payload.get("runroot")
    if (
        not isinstance(recorded_runroot, str)
        or Path(recorded_runroot).resolve() != runroot
    ):
        raise ValueError(f"{path}: floor-gate runroot does not match arm roots")
    if (
        payload.get("task_count") != required_task_count
        or type(payload.get("task_count")) is not int
    ):
        raise ValueError(f"{path}: floor-gate task count does not match reducer")
    if (
        payload.get("inferred_concurrency") != concurrency
        or type(payload.get("inferred_concurrency")) is not int
    ):
        raise ValueError(f"{path}: floor-gate concurrency does not match reducer")

    arms = require_exact_keys(
        payload.get("arms"),
        frozenset({"tail6_fixed32", "hydra27_fixed32"}),
        f"{path}: floor-gate arms",
    )
    expected_ids = list(canonical_task_ids(required_task_count))
    arm_bindings = (
        ("tail6_fixed32", tail_root, fixed32_arm_spec(False)),
        ("hydra27_fixed32", hydra_root, fixed32_arm_spec(True)),
    )
    metric_bindings: dict[str, dict] = {}
    arm_artifact_counts: dict[str, int] = {}
    arm_slo_pass: dict[str, bool] = {}
    for mode, root, spec in arm_bindings:
        arm = require_exact_keys(
            arms.get(mode),
            FIXED32_FLOOR_GATE_ARM_KEYS,
            f"{path}: floor-gate arm {mode!r}",
        )
        artifact_dir = arm.get("artifact_dir")
        if (
            not isinstance(artifact_dir, str)
            or Path(artifact_dir).resolve() != root.resolve()
        ):
            raise ValueError(f"{path}: floor-gate arm {mode!r} root does not match")
        expected_arm = {
            "arm": spec["mode"],
            "inferred_concurrency": concurrency,
            "expected_draft_tokens_per_event": spec["physical_drafts"],
            "active_logical_drafts_per_event": spec["active_drafts"],
            "valid_mask": f"{spec['valid_mask']:#010x}",
            "canonical_task_ids": expected_ids,
        }
        for key, expected in expected_arm.items():
            if arm.get(key) != expected or type(arm.get(key)) is not type(expected):
                raise ValueError(
                    f"{path}: floor-gate arm {mode!r} field {key!r} does not match"
                )
        provenance = arm.get("provenance")
        if (
            not isinstance(provenance, dict)
            or provenance.get("metric_hashes_derived_from_parsed_bytes") is not True
        ):
            raise ValueError(
                f"{path}: floor-gate arm {mode!r} metric-byte provenance is missing"
            )
        task_brackets = provenance.get("task_metric_brackets")
        if not isinstance(task_brackets, dict) or list(task_brackets) != expected_ids:
            raise ValueError(
                f"{path}: floor-gate arm {mode!r} metric task IDs/order do not match"
            )
        normalized_brackets = {}
        artifact_count = 0
        for task_id in expected_ids:
            bracket = task_brackets[task_id]
            if not isinstance(bracket, dict) or set(bracket) != {"pre", "post"}:
                raise ValueError(
                    f"{path}: floor-gate arm {mode!r} task {task_id!r} "
                    "metric bracket is not exact"
                )
            normalized_brackets[task_id] = {}
            for snapshot in ("pre", "post"):
                metric_path = (
                    root.resolve()
                    / "swe_out"
                    / "verified"
                    / "per_task"
                    / task_id
                    / f"vllm_metrics_{snapshot}.txt"
                )
                identity = bracket[snapshot]
                parse_metrics(
                    metric_path,
                    identity,
                    expected_positions=FIXED32_RAW_ACCEPTANCE_POSITIONS,
                )
                normalized_brackets[task_id][snapshot] = dict(identity)
                artifact_count += 1
        artifact_count += validate_real_task_artifacts(
            root.resolve(),
            provenance.get("real_tasks"),
            expected_ids,
        )
        statistics = arm.get("statistics")
        gate = statistics.get("gate") if isinstance(statistics, dict) else None
        gate_pass = gate.get("pass") if isinstance(gate, dict) else None
        if type(gate_pass) is not bool:
            raise ValueError(
                f"{path}: floor-gate arm {mode!r} legacy SLO pass is not boolean"
            )
        arm_slo_pass[mode] = gate_pass
        arm_artifact_counts[mode] = artifact_count
        metric_bindings[mode] = normalized_brackets

    required_gates = FIXED32_REQUIRED_EVIDENCE_GATES | FIXED32_SLO_GATES
    gates = require_exact_keys(
        payload.get("gates"),
        required_gates,
        f"{path}: floor-gate gates",
    )
    if any(type(value) is not bool for value in gates.values()):
        raise ValueError(f"{path}: fixed32 gates are not exact booleans")
    failed_evidence = sorted(
        key
        for key, value in gates.items()
        if key not in FIXED32_SLO_GATES and value is not True
    )
    if failed_evidence:
        raise ValueError(
            f"{path}: fixed32 non-SLO evidence gates failed {failed_evidence!r}"
        )
    for mode in ("tail6_fixed32", "hydra27_fixed32"):
        slo_key = f"{mode}_legacy_slo"
        if gates[slo_key] is not arm_slo_pass[mode]:
            raise ValueError(
                f"{path}: {mode} arm/gate legacy SLO verdicts differ"
            )
    slo_definition = payload.get("slo_definition")
    if (
        not isinstance(slo_definition, dict)
        or slo_definition.get("name") != "legacy_aggressive_weight_stream_slo"
    ):
        raise ValueError(f"{path}: floor-gate legacy SLO definition is missing")
    expected_verdict = (
        "PASS" if all(value is True for value in gates.values()) else "FAIL"
    )
    if payload.get("gate_verdict") != expected_verdict:
        raise ValueError(f"{path}: floor-gate verdict is inconsistent with gates")

    fixed32_work = require_exact_keys(
        payload.get("fixed32_work_census"),
        FIXED32_WORK_CENSUS_KEYS,
        f"{path}: fixed32_work_census",
    )
    derived_work = validate_work_census_v5_report(
        fixed32_work["report"],
        required_batch=concurrency,
    )
    if (
        canonical_json_sha256(fixed32_work["physical_work_comparison"])
        != canonical_json_sha256(derived_work["physical_work_comparison"])
        or canonical_json_sha256(fixed32_work["drafter_graph_lifecycle"])
        != canonical_json_sha256(derived_work["drafter_graph_lifecycle"])
        or canonical_json_sha256(
            fixed32_work["forward_graph_pregather_lifecycle"]
        )
        != canonical_json_sha256(
            derived_work["forward_graph_pregather_lifecycle"]
        )
        or canonical_json_sha256(fixed32_work["scope"])
        != canonical_json_sha256(derived_work["scope"])
        or not isinstance(fixed32_work["scope_interpretation"], str)
        or not fixed32_work["scope_interpretation"]
        or fixed32_work[
            "complete_terminal_stream_reconciled_to_sfwd_sidecar"
        ]
        is not True
        or fixed32_work[
            "canonical_task_forward_counter_union_selected_posthoc"
        ]
        is not True
        or fixed32_work[
            "canonical_task_forward_union_covers_complete_stream"
        ]
        is not True
        or (
            concurrency == 1
            and fixed32_work["b4_occupancy_gate"] != "not_applicable_b1"
        )
        or (
            concurrency == 4
            and not isinstance(fixed32_work["b4_occupancy_gate"], dict)
        )
    ):
        raise ValueError(f"{path}: fixed32 work-census v5 summary mismatch")

    census_files = require_exact_keys(
        fixed32_work["files"],
        frozenset({"tail6_fixed32", "hydra27_fixed32"}),
        f"{path}: fixed32 work-census files",
    )
    census_paths = {}
    for mode, root, _spec in arm_bindings:
        census_path = (
            root.resolve() / "logs" / "fr13_fixed32_work_census.jsonl"
        )
        identity = census_files[mode]
        bound_artifact_bytes(
            census_path,
            identity,
            label="work-census artifact",
        )
        census_paths[mode] = census_path
        arm_artifact_counts[mode] += 1
    try:
        recomputed_work_report = validate_work_census_campaign(
            load_work_census_jsonl(census_paths["tail6_fixed32"]),
            load_work_census_jsonl(census_paths["hydra27_fixed32"]),
            required_batches=(concurrency,),
        )
    except WorkCensusError as error:
        raise ValueError(f"{path}: bound work-census files failed: {error}") from error
    if canonical_json_sha256(recomputed_work_report) != canonical_json_sha256(
        fixed32_work["report"]
    ):
        raise ValueError(
            f"{path}: current work-census files do not reproduce the bound report"
        )

    return (
        {
            "bound": True,
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "schema": FIXED32_FLOOR_GATE_SCHEMA,
            "analysis_valid": True,
            "gate_verdict": expected_verdict,
            "runroot": str(runroot),
            "task_count": required_task_count,
            "concurrency": concurrency,
            "non_slo_evidence_gates": "all_true",
            "metric_bracket_count": 2 * required_task_count * 2,
            "metric_brackets_current_bytes_match": True,
            "arm_artifact_counts": arm_artifact_counts,
            "work_census_report_schema": WORK_CENSUS_REPORT_SCHEMA,
            "work_census_normalized_signature_sha256": fixed32_work["report"][
                "normalized_work_signature_sha256"
            ],
            "work_census_files": {
                mode: dict(census_files[mode])
                for mode in ("tail6_fixed32", "hydra27_fixed32")
            },
            "raw_acceptance_positions": list(
                FIXED32_RAW_ACCEPTANCE_POSITIONS
            ),
            "slo_gates": {key: gates[key] for key in sorted(FIXED32_SLO_GATES)},
        },
        metric_bindings,
    )


def build_report(
    tail: dict,
    hydra: dict,
    comparison: dict,
    *,
    required_task_count: int,
    concurrency: int,
    fixed32: bool,
    floor_gate_binding: dict | None = None,
    self_test_fixture: bool = False,
) -> dict:
    common = {
        "required_task_count": required_task_count,
        "concurrency": concurrency,
        "comparison": comparison,
    }
    if fixed32:
        if floor_gate_binding is None:
            if not self_test_fixture:
                raise ValueError(
                    "fixed32 report requires a validated formal floor-gate binding"
                )
            floor_gate_binding = {
                "bound": False,
                "scope": "self_test_fixture_only",
            }
        return {
            "schema": "fr13.depth_acceptance.fixed32.v1",
            **common,
            "floor_gate_binding": floor_gate_binding,
            "tail21_fixed32": tail,
            "hydra27_fixed32": hydra,
        }
    return {
        "schema": "fr13.depth_acceptance.v2",
        **common,
        "tail6": tail,
        "hydra23": hydra,
    }


def print_table(tail: dict, hydra: dict, comparison: dict) -> None:
    if tail["arm"] == "tail6" and hydra["arm"] == "hydra23":
        tail_label = "tail"
        hydra_label = "hydra"
    else:
        tail_label = tail["arm"]
        hydra_label = hydra["arm"]
    print(
        f"tasks={tail['task_count']} B={tail['concurrency']} "
        f"mode={tail['bracket_mode']} "
        f"{tail_label}_accept={tail['accept_per_event']:.6f} "
        f"{hydra_label}_accept={hydra['accept_per_event']:.6f} "
        f"delta={comparison['hydra_minus_tail_accept_per_event']:+.6f}"
    )
    print(
        f"depth {tail_label}_surv {hydra_label}_surv "
        f"{tail_label}_cond {hydra_label}_cond cond_delta"
    )
    for tail_row, hydra_row, delta in zip(
        tail["depths"],
        hydra["depths"],
        comparison["depths"],
        strict=True,
    ):
        print(
            f"{tail_row['depth']:>5} "
            f"{tail_row['survival']:.6f} {hydra_row['survival']:.6f} "
            f"{tail_row['conditional']:.6f} {hydra_row['conditional']:.6f} "
            f"{delta['conditional_delta']:+.6f}"
        )
    strongest = max(
        comparison["one_depth_at_a_time_counterfactuals"],
        key=lambda row: row["hydra_recovery_if_tail_conditional"],
    )
    print(
        "largest_single_depth_recovery "
        f"depth={strongest['depth']} "
        f"accept={strongest['hydra_recovery_if_tail_conditional']:.6f} "
        "gap_fraction="
        + (
            f"{strongest['fraction_of_accept_gap']:.6f}"
            if strongest["fraction_of_accept_gap"] is not None
            else "n/a"
        )
    )


def write_fixture(
    root: Path,
    task: str,
    drafts: float,
    counts: list[float],
    *,
    tokens_per_draft: int,
    mtime: float | None = None,
) -> None:
    task_dir = root / "swe_out" / "verified" / "per_task" / task
    task_dir.mkdir(parents=True)
    labels = 'engine="0",model_name="fixture"'
    pre = [
        f"{DRAFTS_METRIC}{{{labels}}} 10",
        f"{DRAFT_TOKENS_METRIC}{{{labels}}} 100",
        f"{ACCEPTED_METRIC}{{{labels}}} 20",
    ]
    post = [
        f"{DRAFTS_METRIC}{{{labels}}} {10 + drafts}",
        f"{DRAFT_TOKENS_METRIC}{{{labels}}} {100 + tokens_per_draft * drafts}",
        f"{ACCEPTED_METRIC}{{{labels}}} {20 + sum(counts)}",
    ]
    for position, count in enumerate(counts):
        position_labels = f'{labels},position="{position}"'
        pre.append(f"{POSITION_METRIC}{{{position_labels}}} 3")
        post.append(f"{POSITION_METRIC}{{{position_labels}}} {3 + count}")
    (task_dir / "vllm_metrics_pre.txt").write_text("\n".join(pre) + "\n")
    (task_dir / "vllm_metrics_post.txt").write_text("\n".join(post) + "\n")
    if mtime is not None:
        os.utime(task_dir / "vllm_metrics_pre.txt", (mtime, mtime))
        os.utime(task_dir / "vllm_metrics_post.txt", (mtime + 1, mtime + 1))


def write_campaign_fixture(
    root: Path,
    concurrency: int,
    *,
    task_count: int = 4,
) -> None:
    (root / "swe_orchestrator.log").write_text(
        "=== [fixture] dataset=SWE-bench_Verified "
        f"tag=verified n={task_count} concurrency={concurrency} ===\n"
    )


def write_fixed32_provenance_fixture(root: Path, expected_hydra: bool) -> None:
    spec = fixed32_arm_spec(expected_hydra)
    values = {
        "FR13_HYDRA23": "0",
        "FR13_FIXED32_MODE": spec["mode"],
        "FR13_FIXED32_VALID_MASK": f"{spec['valid_mask']:#010x}",
        "FR13_FIXED32_ACTIVE_NODES": str(spec["active_drafts"]),
        "FR13_FIXED32_PHYSICAL_DRAFTS": str(spec["physical_drafts"]),
    }
    (root / "container_env.txt").write_text(
        "".join(f"{key}={value}\n" for key, value in values.items())
    )


def write_floor_gate_fixture(
    path: Path,
    runroot: Path,
    tail_root: Path,
    hydra_root: Path,
    *,
    task_count: int,
    concurrency: int,
) -> None:
    gates = {
        key: True for key in sorted(FIXED32_REQUIRED_EVIDENCE_GATES | FIXED32_SLO_GATES)
    }
    gates["tail6_fixed32_legacy_slo"] = False
    gates["hydra27_fixed32_legacy_slo"] = False

    census_paths: dict[str, Path] = {}
    for mode, root in (
        ("tail6_fixed32", tail_root),
        ("hydra27_fixed32", hydra_root),
    ):
        event = reference_event(
            mode,
            concurrency,
            f"{mode}:fixture:B{concurrency}:0",
            event_index=0,
            forward_step_index=0,
        )
        records = [
            event,
            reference_terminal_summary(
                [event],
                fixture_synthetic_runtime_proof=True,
            ),
        ]
        census_path = root / "logs" / "fr13_fixed32_work_census.jsonl"
        census_path.parent.mkdir(parents=True, exist_ok=True)
        census_path.write_text(
            "".join(
                json.dumps(record, sort_keys=True) + "\n" for record in records
            )
        )
        census_paths[mode] = census_path.resolve()
    work_report = validate_work_census_campaign(
        load_work_census_jsonl(census_paths["tail6_fixed32"]),
        load_work_census_jsonl(census_paths["hydra27_fixed32"]),
        required_batches=(concurrency,),
    )
    derived_work = validate_work_census_v5_report(
        work_report,
        required_batch=concurrency,
    )

    def artifact_identity(artifact_path: Path) -> dict:
        artifact_path = artifact_path.resolve()
        raw = artifact_path.read_bytes()
        return {
            "path": str(artifact_path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }

    def arm(root: Path, expected_hydra: bool) -> dict:
        spec = fixed32_arm_spec(expected_hydra)
        task_metric_brackets = {}
        real_task_records = {}
        proxy_root = root / "proxy_pair_dumps"
        proxy_root.mkdir(parents=True, exist_ok=True)
        for task_index, task_id in enumerate(canonical_task_ids(task_count)):
            task_dir = (
                root
                / "swe_out"
                / "verified"
                / "per_task"
                / task_id
            )
            task_metric_brackets[task_id] = {}
            for snapshot in ("pre", "post"):
                metric_path = (
                    task_dir
                    / f"vllm_metrics_{snapshot}.txt"
                )
                task_metric_brackets[task_id][snapshot] = artifact_identity(
                    metric_path
                )
            trace_path = task_dir / "qwen_trace.jsonl"
            trace_path.write_text(
                json.dumps(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": "fixture result",
                        "usage": {"total_tokens": task_index + 1},
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            proxy_path = proxy_root / f"{task_index:02d}-{task_id}.json"
            proxy_path.write_text(
                json.dumps(
                    {
                        "schema": "lumo.fr13.proxy_pair_dump.v1",
                        "request": {
                            "prompt": f"# SWE-Bench task: {task_id}\nfixture"
                        },
                        "response": {"usage": {"total_tokens": task_index + 1}},
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            trace_identity = artifact_identity(trace_path)
            real_task_records[task_id] = {
                "trace_path": trace_identity["path"],
                "trace_sha256": trace_identity["sha256"],
                "trace_bytes": trace_identity["bytes"],
                "trace_events": 1,
                "assistant_output_events": 1,
                "positive_token_usage": True,
                "agent_terminal": {
                    "exit_code": 0,
                    "timed_out": False,
                    "offloaded": True,
                    "network_drop": False,
                },
                "eval_verdict": "resolved",
                "proxy_pairs": [
                    {
                        **artifact_identity(proxy_path),
                        "response_positive_token_usage": True,
                    }
                ],
            }
        return {
            "arm": spec["mode"],
            "artifact_dir": str(root.resolve()),
            "inferred_concurrency": concurrency,
            "expected_draft_tokens_per_event": spec["physical_drafts"],
            "active_logical_drafts_per_event": spec["active_drafts"],
            "valid_mask": f"{spec['valid_mask']:#010x}",
            "canonical_task_ids": list(canonical_task_ids(task_count)),
            "provenance": {
                "task_metric_brackets": task_metric_brackets,
                "metric_hashes_derived_from_parsed_bytes": True,
                "real_tasks": {
                    "all_canonical_tasks_have_real_model_traffic": True,
                    "all_positive_usage_proxy_pairs_task_bound": True,
                    "all_agents_completed_cleanly": True,
                    "all_tasks_have_terminal_eval_verdicts": True,
                    "tasks": real_task_records,
                },
            },
            "sidecar": {},
            "flush_chain": {},
            "work_census_expected": {},
            "statistics": {"gate": {"pass": False}},
        }

    work_census = {
        "report": work_report,
        "physical_work_comparison": derived_work["physical_work_comparison"],
        "drafter_graph_lifecycle": derived_work["drafter_graph_lifecycle"],
        "forward_graph_pregather_lifecycle": derived_work[
            "forward_graph_pregather_lifecycle"
        ],
        "scope": derived_work["scope"],
        "scope_interpretation": (
            "Fixture evidence is limited to the explicit fixed-work census scope."
        ),
        "files": {
            mode: artifact_identity(census_path)
            for mode, census_path in census_paths.items()
        },
        "complete_terminal_stream_reconciled_to_sfwd_sidecar": True,
        "canonical_task_forward_counter_union_selected_posthoc": True,
        "canonical_task_forward_union_covers_complete_stream": True,
        "b4_occupancy_gate": (
            {"fixture_exact_b4": True}
            if concurrency == 4
            else "not_applicable_b1"
        ),
    }
    path.write_text(
        json.dumps(
            {
                "schema": FIXED32_FLOOR_GATE_SCHEMA,
                "analysis_valid": True,
                "gate_verdict": "FAIL",
                "repo": str(runroot.resolve()),
                "runroot": str(runroot.resolve()),
                "tag": "fixture",
                "task_count": task_count,
                "inferred_concurrency": concurrency,
                "source_runtime_fingerprint": {},
                "external_artifact_fingerprint": {},
                "matched_runtime_attestation": {},
                "fixed32_work_census": work_census,
                "slo_definition": {
                    "name": "legacy_aggressive_weight_stream_slo",
                },
                "uncertainty_model": "fixture",
                "evidence_requirements": {},
                "arms": {
                    "tail6_fixed32": arm(tail_root, False),
                    "hydra27_fixed32": arm(hydra_root, True),
                },
                "comparison": {},
                "gates": gates,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def write_overlapping_fixtures(
    root: Path, counts_per_ten_drafts: list[float], tokens_per_draft: int
) -> None:
    labels = 'engine="0",model_name="fixture"'
    for index, task_id in enumerate(CANONICAL_TASK_IDS[:4]):
        task_dir = root / "swe_out" / "verified" / "per_task" / task_id
        pre_multiple = index
        post_multiple = index + 4

        def snapshot(multiple: int) -> str:
            drafts = 10 * multiple
            counts = [count * multiple for count in counts_per_ten_drafts]
            lines = [
                f"{DRAFTS_METRIC}{{{labels}}} {drafts}",
                f"{DRAFT_TOKENS_METRIC}{{{labels}}} {tokens_per_draft * drafts}",
                f"{ACCEPTED_METRIC}{{{labels}}} {sum(counts)}",
            ]
            for position, count in enumerate(counts):
                position_labels = f'{labels},position="{position}"'
                lines.append(f"{POSITION_METRIC}{{{position_labels}}} {count}")
            return "\n".join(lines) + "\n"

        pre_path = task_dir / "vllm_metrics_pre.txt"
        post_path = task_dir / "vllm_metrics_post.txt"
        pre_path.write_text(snapshot(pre_multiple))
        post_path.write_text(snapshot(post_multiple))
        os.utime(pre_path, (100 + index, 100 + index))
        os.utime(post_path, (104 + index, 104 + index))


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        tail_root = base / "tail"
        hydra_root = base / "hydra"
        tail_root.mkdir()
        hydra_root.mkdir()
        (tail_root / "container_env.txt").write_text("FR13_HYDRA23=0\n")
        (hydra_root / "container_env.txt").write_text("FR13_HYDRA23=1\n")
        write_campaign_fixture(tail_root, 1)
        write_campaign_fixture(hydra_root, 1)
        for index, task_id in enumerate(CANONICAL_TASK_IDS[:4]):
            write_fixture(
                tail_root,
                task_id,
                10,
                [8, 4, 2],
                tokens_per_draft=21,
                mtime=100 + index,
            )
            write_fixture(
                hydra_root,
                task_id,
                10,
                [8, 3, 1],
                tokens_per_draft=23,
                mtime=100 + index,
            )
        tail = reduce_arm(tail_root, 4, 1, expected_hydra=False)
        hydra = reduce_arm(hydra_root, 4, 1, expected_hydra=True)
        result = compare(tail, hydra)
        legacy_report = build_report(
            tail,
            hydra,
            result,
            required_task_count=4,
            concurrency=1,
            fixed32=False,
        )
        assert legacy_report["schema"] == "fr13.depth_acceptance.v2"
        assert legacy_report["tail6"] is tail
        assert legacy_report["hydra23"] is hydra
        assert "tail21_fixed32" not in legacy_report
        assert tail["bracket_mode"] == "nonoverlapping_task_sum"
        assert tail["drafts"] == 40
        assert tail["accept_per_event"] == 1.4
        assert hydra["accept_per_event"] == 1.2
        assert abs(result["hydra_minus_tail_accept_per_event"] + 0.2) < 1e-12
        assert result["depths"][1]["conditional_delta"] == -0.125
        assert (
            abs(
                result["one_depth_at_a_time_counterfactuals"][1][
                    "hydra_recovery_if_tail_conditional"
                ]
                - (2.0 / 15.0)
            )
            < 1e-12
        )
        write_campaign_fixture(tail_root, 4)
        write_campaign_fixture(hydra_root, 4)
        write_overlapping_fixtures(tail_root, [8, 4, 2], 21)
        write_overlapping_fixtures(hydra_root, [8, 3, 1], 23)
        tail_union = reduce_arm(tail_root, 4, 4, expected_hydra=False)
        hydra_union = reduce_arm(hydra_root, 4, 4, expected_hydra=True)
        assert tail_union["bracket_mode"] == "union_earliest_pre_latest_post"
        assert tail_union["per_task_available"] is False
        assert tail_union["per_task"] == []
        assert tail_union["drafts"] == 70
        assert tail_union["accept_per_event"] == 1.4
        assert hydra_union["accept_per_event"] == 1.2
        assert tail_union["union_window"]["pre"].endswith(
            f"{CANONICAL_TASK_IDS[0]}/vllm_metrics_pre.txt"
        )
        assert tail_union["union_window"]["post"].endswith(
            f"{CANONICAL_TASK_IDS[3]}/vllm_metrics_post.txt"
        )
        try:
            reduce_arm(tail_root, 4, 1, expected_hydra=False)
        except ValueError as error:
            assert "artifact records 4" in str(error)
        else:
            raise AssertionError("mislabeled concurrency did not fail closed")
        write_campaign_fixture(tail_root, 1)
        try:
            reduce_arm(tail_root, 16, 1, expected_hydra=False)
        except ValueError as error:
            assert "requested 16 tasks" in str(error)
        else:
            raise AssertionError("undersized evidence set did not fail closed")

        fixed_tail_root = base / "tail_fixed32"
        fixed_hydra_root = base / "hydra_fixed32"
        fixed_tail_root.mkdir()
        fixed_hydra_root.mkdir()
        write_fixed32_provenance_fixture(fixed_tail_root, False)
        write_fixed32_provenance_fixture(fixed_hydra_root, True)
        write_campaign_fixture(fixed_tail_root, 1)
        write_campaign_fixture(fixed_hydra_root, 1)
        for index, task_id in enumerate(CANONICAL_TASK_IDS[:4]):
            write_fixture(
                fixed_tail_root,
                task_id,
                10,
                [8, 4, 2] + [0] * 28,
                tokens_per_draft=31,
                mtime=200 + index,
            )
            write_fixture(
                fixed_hydra_root,
                task_id,
                10,
                [8, 3, 0] + [0] * 28,
                tokens_per_draft=31,
                mtime=200 + index,
            )
        fixed_tail = reduce_arm(
            fixed_tail_root,
            4,
            1,
            expected_hydra=False,
            fixed32=True,
        )
        fixed_hydra = reduce_arm(
            fixed_hydra_root,
            4,
            1,
            expected_hydra=True,
            fixed32=True,
        )
        fixed_result = compare(fixed_tail, fixed_hydra)
        fixed_report = build_report(
            fixed_tail,
            fixed_hydra,
            fixed_result,
            required_task_count=4,
            concurrency=1,
            fixed32=True,
            self_test_fixture=True,
        )
        assert fixed_report["schema"] == "fr13.depth_acceptance.fixed32.v1"
        assert fixed_report["tail21_fixed32"] is fixed_tail
        assert fixed_report["hydra27_fixed32"] is fixed_hydra
        assert "tail6" not in fixed_report
        assert fixed_tail["arm"] == "tail21_fixed32"
        assert fixed_hydra["arm"] == "hydra27_fixed32"
        assert fixed_tail["expected_tokens_per_draft"] == 31
        assert fixed_hydra["expected_tokens_per_draft"] == 31
        assert fixed_tail["fixed32_contract"]["active_drafts"] == 21
        assert fixed_hydra["fixed32_contract"]["active_drafts"] == 27
        assert fixed_tail["bracket_mode"] == "nonoverlapping_task_sum"
        assert fixed_tail["drafts"] == 40
        assert fixed_tail["accept_per_event"] == 1.4
        assert fixed_hydra["accept_per_event"] == 1.1
        assert fixed_tail["raw_acceptance_positions"] == list(range(31))
        assert fixed_hydra["raw_acceptance_positions"] == list(range(31))
        assert len(fixed_tail["depths"]) == 31
        assert len(fixed_hydra["depths"]) == 31
        assert all(len(task["depths"]) == 31 for task in fixed_tail["per_task"])
        assert all(len(task["depths"]) == 31 for task in fixed_hydra["per_task"])
        assert fixed_tail["depths"][-1]["accepted_count"] == 0
        assert fixed_hydra["depths"][2]["accepted_count"] == 0
        assert abs(fixed_result["hydra_minus_tail_accept_per_event"] + 0.3) < 1e-12
        assert fixed_report["floor_gate_binding"] == {
            "bound": False,
            "scope": "self_test_fixture_only",
        }

        write_campaign_fixture(fixed_tail_root, 4)
        write_campaign_fixture(fixed_hydra_root, 4)
        write_overlapping_fixtures(
            fixed_tail_root,
            [8, 4, 2] + [0] * 28,
            31,
        )
        write_overlapping_fixtures(
            fixed_hydra_root,
            [8, 3, 0] + [0] * 28,
            31,
        )
        fixed_tail_union = reduce_arm(
            fixed_tail_root,
            4,
            4,
            expected_hydra=False,
            fixed32=True,
        )
        fixed_hydra_union = reduce_arm(
            fixed_hydra_root,
            4,
            4,
            expected_hydra=True,
            fixed32=True,
        )
        assert fixed_tail_union["bracket_mode"] == "union_earliest_pre_latest_post"
        assert fixed_tail_union["per_task_available"] is False
        assert fixed_tail_union["per_task"] == []
        assert fixed_tail_union["drafts"] == 70
        assert fixed_tail_union["draft_tokens"] == 31 * 70
        assert fixed_hydra_union["draft_tokens"] == 31 * 70
        assert len(fixed_tail_union["depths"]) == 31
        assert len(fixed_hydra_union["depths"]) == 31

        gate_path = base / "fixed32_floor_gate.json"
        write_floor_gate_fixture(
            gate_path,
            base,
            fixed_tail_root,
            fixed_hydra_root,
            task_count=4,
            concurrency=4,
        )
        assert (
            locate_floor_gate_report(
                None,
                fixed_tail_root,
                fixed_hydra_root,
            )
            == gate_path
        )
        binding, metric_bindings = validate_floor_gate_binding(
            gate_path,
            fixed_tail_root,
            fixed_hydra_root,
            required_task_count=4,
            concurrency=4,
        )
        assert binding["bound"] is True
        assert binding["analysis_valid"] is True
        assert binding["gate_verdict"] == "FAIL"
        assert binding["non_slo_evidence_gates"] == "all_true"
        assert binding["metric_bracket_count"] == 16
        assert binding["metric_brackets_current_bytes_match"] is True
        bound_tail_union = reduce_arm(
            fixed_tail_root,
            4,
            4,
            expected_hydra=False,
            fixed32=True,
            metric_bindings=metric_bindings["tail6_fixed32"],
        )
        bound_hydra_union = reduce_arm(
            fixed_hydra_root,
            4,
            4,
            expected_hydra=True,
            fixed32=True,
            metric_bindings=metric_bindings["hydra27_fixed32"],
        )
        bound_report = build_report(
            bound_tail_union,
            bound_hydra_union,
            compare(bound_tail_union, bound_hydra_union),
            required_task_count=4,
            concurrency=4,
            fixed32=True,
            floor_gate_binding=binding,
        )
        assert bound_report["floor_gate_binding"] is binding

        bound_post = (
            fixed_tail_root
            / "swe_out"
            / "verified"
            / "per_task"
            / CANONICAL_TASK_IDS[3]
            / "vllm_metrics_post.txt"
        )
        good_bound_post = bound_post.read_bytes()
        original_bound_hash = metric_bindings["tail6_fixed32"][CANONICAL_TASK_IDS[3]][
            "post"
        ]["sha256"]
        bound_post.write_bytes(good_bound_post + b"\n")
        try:
            validate_floor_gate_binding(
                gate_path,
                fixed_tail_root,
                fixed_hydra_root,
                required_task_count=4,
                concurrency=4,
            )
        except ValueError as error:
            assert (
                "current metric artifact byte count does not match floor gate"
                in str(error)
            )
        else:
            raise AssertionError("post-gate metric mutation did not fail binding")
        try:
            reduce_arm(
                fixed_tail_root,
                4,
                4,
                expected_hydra=False,
                fixed32=True,
                metric_bindings=metric_bindings["tail6_fixed32"],
            )
        except ValueError as error:
            assert (
                "current metric artifact byte count does not match floor gate"
                in str(error)
            )
        else:
            raise AssertionError("bound reducer consumed mutated metric bytes")

        write_floor_gate_fixture(
            gate_path,
            base,
            fixed_tail_root,
            fixed_hydra_root,
            task_count=4,
            concurrency=4,
        )
        rebound_binding, rebound_metrics = validate_floor_gate_binding(
            gate_path,
            fixed_tail_root,
            fixed_hydra_root,
            required_task_count=4,
            concurrency=4,
        )
        rebound_hash = rebound_metrics["tail6_fixed32"][CANONICAL_TASK_IDS[3]]["post"][
            "sha256"
        ]
        assert rebound_hash != original_bound_hash
        assert rebound_binding["metric_brackets_current_bytes_match"] is True
        reduce_arm(
            fixed_tail_root,
            4,
            4,
            expected_hydra=False,
            fixed32=True,
            metric_bindings=rebound_metrics["tail6_fixed32"],
        )
        bound_post.write_bytes(good_bound_post)
        write_floor_gate_fixture(
            gate_path,
            base,
            fixed_tail_root,
            fixed_hydra_root,
            task_count=4,
            concurrency=4,
        )

        gate_payload = json.loads(gate_path.read_text())
        gate_payload["gates"]["canonical_subset_hash"] = False
        gate_path.write_text(json.dumps(gate_payload) + "\n")
        try:
            validate_floor_gate_binding(
                gate_path,
                fixed_tail_root,
                fixed_hydra_root,
                required_task_count=4,
                concurrency=4,
            )
        except ValueError as error:
            assert "non-SLO evidence gates failed" in str(error)
        else:
            raise AssertionError("failed non-SLO evidence gate did not fail closed")
        write_floor_gate_fixture(
            gate_path,
            base,
            fixed_tail_root,
            fixed_hydra_root,
            task_count=4,
            concurrency=4,
        )
        original_gate = json.loads(gate_path.read_text())
        tail_metric_path = (
            "arms",
            "tail6_fixed32",
            "provenance",
            "task_metric_brackets",
            CANONICAL_TASK_IDS[0],
            "pre",
        )
        original_metric_bytes = original_gate
        for key in tail_metric_path:
            original_metric_bytes = original_metric_bytes[key]
        tail_trace_path = (
            "arms",
            "tail6_fixed32",
            "provenance",
            "real_tasks",
            "tasks",
            CANONICAL_TASK_IDS[0],
        )
        original_tail_trace = original_gate
        for key in tail_trace_path:
            original_tail_trace = original_tail_trace[key]
        hydra_proxy_path = (
            "arms",
            "hydra27_fixed32",
            "provenance",
            "real_tasks",
            "tasks",
            CANONICAL_TASK_IDS[0],
            "proxy_pairs",
            0,
        )
        original_hydra_proxy = original_gate
        for key in hydra_proxy_path:
            original_hydra_proxy = original_hydra_proxy[key]
        tail_census_path = (
            "fixed32_work_census",
            "files",
            "tail6_fixed32",
        )
        original_tail_census = original_gate
        for key in tail_census_path:
            original_tail_census = original_tail_census[key]
        gate_tamper_cases = (
            (("schema",), "wrong-schema", "expected floor-gate schema"),
            (("analysis_valid",), False, "analysis is not valid"),
            (("runroot",), "/wrong/runroot", "runroot does not match"),
            (("task_count",), 16, "task count does not match"),
            (("inferred_concurrency",), 1, "concurrency does not match"),
            (
                ("arms", "tail6_fixed32", "artifact_dir"),
                "/wrong/tail",
                "arm 'tail6_fixed32' root does not match",
            ),
            (
                ("arms", "hydra27_fixed32", "artifact_dir"),
                "/wrong/hydra",
                "arm 'hydra27_fixed32' root does not match",
            ),
            (
                (
                    "arms",
                    "tail6_fixed32",
                    "provenance",
                    "metric_hashes_derived_from_parsed_bytes",
                ),
                False,
                "metric-byte provenance is missing",
            ),
            (
                tail_metric_path[:-1],
                {},
                "metric bracket is not exact",
            ),
            (
                tail_metric_path + ("path",),
                "/wrong/metrics.txt",
                "bound metric artifact path does not match",
            ),
            (
                tail_metric_path + ("sha256",),
                "0" * 64,
                "current metric artifact SHA-256 does not match floor gate",
            ),
            (
                tail_metric_path + ("bytes",),
                original_metric_bytes["bytes"] + 1,
                "current metric artifact byte count does not match floor gate",
            ),
            (
                tail_trace_path + ("trace_path",),
                "/wrong/trace.jsonl",
                "bound task trace path does not match",
            ),
            (
                tail_trace_path + ("trace_sha256",),
                "0" * 64,
                "current task trace SHA-256 does not match floor gate",
            ),
            (
                tail_trace_path + ("trace_bytes",),
                original_tail_trace["trace_bytes"] + 1,
                "current task trace byte count does not match floor gate",
            ),
            (
                hydra_proxy_path + ("path",),
                "/wrong/proxy.json",
                "proxy path is outside its arm",
            ),
            (
                hydra_proxy_path + ("sha256",),
                "0" * 64,
                "current proxy artifact SHA-256 does not match floor gate",
            ),
            (
                hydra_proxy_path + ("bytes",),
                original_hydra_proxy["bytes"] + 1,
                "current proxy artifact byte count does not match floor gate",
            ),
            (
                tail_census_path + ("path",),
                "/wrong/census.jsonl",
                "bound work-census artifact path does not match",
            ),
            (
                tail_census_path + ("sha256",),
                "0" * 64,
                "current work-census artifact SHA-256 does not match floor gate",
            ),
            (
                tail_census_path + ("bytes",),
                original_tail_census["bytes"] + 1,
                "current work-census artifact byte count does not match floor gate",
            ),
            (
                (
                    "fixed32_work_census",
                    "report",
                    "normalized_work_signature_sha256",
                ),
                "0" * 64,
                "work-census v5 report contract mismatch",
            ),
        )
        for keys, tampered, expected_error in gate_tamper_cases:
            gate_payload = json.loads(json.dumps(original_gate))
            cursor = gate_payload
            for key in keys[:-1]:
                cursor = cursor[key]
            cursor[keys[-1]] = tampered
            gate_path.write_text(json.dumps(gate_payload) + "\n")
            try:
                validate_floor_gate_binding(
                    gate_path,
                    fixed_tail_root,
                    fixed_hydra_root,
                    required_task_count=4,
                    concurrency=4,
                )
            except ValueError as error:
                assert expected_error in str(error)
            else:
                raise AssertionError(
                    f"floor-gate binding tamper {keys!r} did not fail closed"
                )
        gate_path.write_text(json.dumps(original_gate) + "\n")

        def expect_gate_payload_failure(
            payload: dict,
            label: str,
            expected_error: str,
        ) -> None:
            gate_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
            try:
                validate_floor_gate_binding(
                    gate_path,
                    fixed_tail_root,
                    fixed_hydra_root,
                    required_task_count=4,
                    concurrency=4,
                )
            except ValueError as error:
                assert expected_error in str(error)
            else:
                raise AssertionError(f"{label} did not fail closed")

        extra_top = json.loads(json.dumps(original_gate))
        extra_top["unexpected"] = True
        expect_gate_payload_failure(
            extra_top,
            "unknown v8 top-level field",
            "floor-gate v8 report: fields are not exact",
        )
        extra_arm = json.loads(json.dumps(original_gate))
        extra_arm["arms"]["tail6_fixed32"]["unexpected"] = True
        expect_gate_payload_failure(
            extra_arm,
            "unknown v8 arm field",
            "floor-gate arm 'tail6_fixed32': fields are not exact",
        )
        extra_gate = json.loads(json.dumps(original_gate))
        extra_gate["gates"]["unexpected"] = True
        expect_gate_payload_failure(
            extra_gate,
            "unknown v8 gate",
            "floor-gate gates: fields are not exact",
        )
        nonboolean_slo = json.loads(json.dumps(original_gate))
        nonboolean_slo["gates"]["tail6_fixed32_legacy_slo"] = 0
        expect_gate_payload_failure(
            nonboolean_slo,
            "non-boolean legacy SLO",
            "fixed32 gates are not exact booleans",
        )
        mismatched_slo = json.loads(json.dumps(original_gate))
        mismatched_slo["arms"]["tail6_fixed32"]["statistics"]["gate"]["pass"] = True
        expect_gate_payload_failure(
            mismatched_slo,
            "arm/gate legacy SLO disagreement",
            "arm/gate legacy SLO verdicts differ",
        )
        inconsistent_verdict = json.loads(json.dumps(original_gate))
        inconsistent_verdict["gate_verdict"] = "PASS"
        expect_gate_payload_failure(
            inconsistent_verdict,
            "inconsistent floor-gate verdict",
            "floor-gate verdict is inconsistent with gates",
        )
        malformed_work_summary = json.loads(json.dumps(original_gate))
        malformed_work_summary["fixed32_work_census"][
            "physical_work_comparison"
        ]["one_normalized_signature_per_occupied_batch"] = 1
        expect_gate_payload_failure(
            malformed_work_summary,
            "non-exact v5 physical-work summary type",
            "fixed32 work-census v5 summary mismatch",
        )
        malformed_forward_summary = json.loads(json.dumps(original_gate))
        malformed_forward_summary["fixed32_work_census"][
            "forward_graph_pregather_lifecycle"
        ]["stage_precedes_all_layer_consumes"] = 1
        expect_gate_payload_failure(
            malformed_forward_summary,
            "non-exact v5 forward pregather summary type",
            "fixed32 work-census v5 summary mismatch",
        )
        malformed_forward_report = json.loads(json.dumps(original_gate))
        malformed_forward_report["fixed32_work_census"]["report"][
            "forward_graph_registries"
        ]["tail6_fixed32"][0]["measured_replays"] += 1
        expect_gate_payload_failure(
            malformed_forward_report,
            "forward pregather replay mismatch",
            "terminal forward graph pregather proof mismatch",
        )
        malformed_auxiliary_report = json.loads(json.dumps(original_gate))
        for parent in (
            malformed_auxiliary_report["fixed32_work_census"]["report"][
                "conv_pregather_auxiliary"
            ]["tail6_fixed32"],
            malformed_auxiliary_report["fixed32_work_census"]["report"][
                "terminal_summaries"
            ]["tail6_fixed32"]["conv_pregather_auxiliary"],
        ):
            parent["profile_capture_stages"] = 1
        expect_gate_payload_failure(
            malformed_auxiliary_report,
            "profile pregather stage mismatch",
            "pregather auxiliary/host stage counts are not zero",
        )

        for tail_slo, hydra_slo in ((True, False), (False, True), (True, True)):
            allowed_slo = json.loads(json.dumps(original_gate))
            for mode, passed in (
                ("tail6_fixed32", tail_slo),
                ("hydra27_fixed32", hydra_slo),
            ):
                allowed_slo["arms"][mode]["statistics"]["gate"]["pass"] = passed
                allowed_slo["gates"][f"{mode}_legacy_slo"] = passed
            allowed_slo["gate_verdict"] = (
                "PASS" if tail_slo and hydra_slo else "FAIL"
            )
            gate_path.write_text(json.dumps(allowed_slo, sort_keys=True) + "\n")
            allowed_binding, _ = validate_floor_gate_binding(
                gate_path,
                fixed_tail_root,
                fixed_hydra_root,
                required_task_count=4,
                concurrency=4,
            )
            assert allowed_binding["gate_verdict"] == allowed_slo["gate_verdict"]
            assert allowed_binding["slo_gates"] == {
                "hydra27_fixed32_legacy_slo": hydra_slo,
                "tail6_fixed32_legacy_slo": tail_slo,
            }
        gate_path.write_text(json.dumps(original_gate, sort_keys=True) + "\n")

        trace_artifact = Path(original_tail_trace["trace_path"])
        original_trace_bytes = trace_artifact.read_bytes()
        bad_trace_bytes = (
            json.dumps(
                {
                    "type": "message",
                    "role": "user",
                    "content": "not an assistant output",
                    "usage": {"total_tokens": 1},
                },
                sort_keys=True,
            )
            + "\n"
        ).encode()
        trace_artifact.write_bytes(bad_trace_bytes)
        bad_trace_gate = json.loads(json.dumps(original_gate))
        bad_trace_record = bad_trace_gate
        for key in tail_trace_path:
            bad_trace_record = bad_trace_record[key]
        bad_trace_record["trace_sha256"] = hashlib.sha256(
            bad_trace_bytes
        ).hexdigest()
        bad_trace_record["trace_bytes"] = len(bad_trace_bytes)
        expect_gate_payload_failure(
            bad_trace_gate,
            "self-consistent non-assistant task trace",
            "real task astropy__astropy-12907 evidence is incomplete",
        )
        trace_artifact.write_bytes(original_trace_bytes)

        proxy_artifact = Path(original_hydra_proxy["path"])
        original_proxy_bytes = proxy_artifact.read_bytes()
        bad_proxy_payload = strict_json_text(
            original_proxy_bytes.decode(),
            label=str(proxy_artifact),
        )
        assert isinstance(bad_proxy_payload, dict)
        bad_proxy_payload["request"] = {
            "prompt": f"# SWE-Bench task: {CANONICAL_TASK_IDS[1]}\nwrong task"
        }
        bad_proxy_bytes = (
            json.dumps(bad_proxy_payload, sort_keys=True) + "\n"
        ).encode()
        proxy_artifact.write_bytes(bad_proxy_bytes)
        bad_proxy_gate = json.loads(json.dumps(original_gate))
        bad_proxy_record = bad_proxy_gate
        for key in hydra_proxy_path:
            bad_proxy_record = bad_proxy_record[key]
        bad_proxy_record["sha256"] = hashlib.sha256(bad_proxy_bytes).hexdigest()
        bad_proxy_record["bytes"] = len(bad_proxy_bytes)
        expect_gate_payload_failure(
            bad_proxy_gate,
            "self-consistent cross-task proxy pair",
            "proxy artifact is not bound to exactly task",
        )
        proxy_artifact.write_bytes(original_proxy_bytes)
        gate_path.write_text(json.dumps(original_gate, sort_keys=True) + "\n")

        last_post = (
            fixed_tail_root
            / "swe_out"
            / "verified"
            / "per_task"
            / CANONICAL_TASK_IDS[3]
            / "vllm_metrics_post.txt"
        )
        original_post = last_post.read_text()
        token_line = (
            f'{DRAFT_TOKENS_METRIC}{{engine="0",model_name="fixture"}} {31 * 70}\n'
        )
        assert token_line in original_post
        last_post.write_text(
            original_post.replace(
                token_line,
                f'{DRAFT_TOKENS_METRIC}{{engine="0",model_name="fixture"}} '
                f"{31 * 70 + 1}\n",
            )
        )
        try:
            reduce_arm(
                fixed_tail_root,
                4,
                4,
                expected_hydra=False,
                fixed32=True,
            )
        except ValueError as error:
            assert "draft tokens/event is not exactly 31" in str(error)
        else:
            raise AssertionError("non-31 draft-token ratio did not fail closed")
        finally:
            last_post.write_text(original_post)

        first_pre = (
            fixed_tail_root
            / "swe_out"
            / "verified"
            / "per_task"
            / CANONICAL_TASK_IDS[0]
            / "vllm_metrics_pre.txt"
        )
        original_pre = first_pre.read_text()
        first_pre_stat = first_pre.stat()
        last_post_stat = last_post.stat()
        extra_position = (
            f'{POSITION_METRIC}{{engine="0",model_name="fixture",position="31"}} 0\n'
        )
        first_pre.write_text(original_pre + extra_position)
        last_post.write_text(original_post + extra_position)
        os.utime(
            first_pre,
            ns=(first_pre_stat.st_atime_ns, first_pre_stat.st_mtime_ns),
        )
        os.utime(
            last_post,
            ns=(last_post_stat.st_atime_ns, last_post_stat.st_mtime_ns),
        )
        try:
            reduce_arm(
                fixed_tail_root,
                4,
                4,
                expected_hydra=False,
                fixed32=True,
            )
        except ValueError as error:
            assert "raw acceptance positions must be exactly 0..30" in str(error)
        else:
            raise AssertionError("32nd acceptance position did not fail closed")
        finally:
            first_pre.write_text(original_pre)
            last_post.write_text(original_post)

        nonselected_pre = (
            fixed_tail_root
            / "swe_out"
            / "verified"
            / "per_task"
            / CANONICAL_TASK_IDS[1]
            / "vllm_metrics_pre.txt"
        )
        original_nonselected_pre = nonselected_pre.read_text()
        position_30_lines = [
            line
            for line in original_nonselected_pre.splitlines()
            if 'position="30"' in line
        ]
        assert len(position_30_lines) == 1
        raw_position_tampers = (
            (
                original_nonselected_pre.replace(
                    position_30_lines[0] + "\n",
                    "",
                ),
                "raw acceptance positions must be exactly 0..30",
                "missing position in non-selected metric",
            ),
            (
                original_nonselected_pre + position_30_lines[0] + "\n",
                "duplicate vllm:spec_decode_num_accepted_tokens_per_pos_total",
                "duplicate position in non-selected metric",
            ),
            (
                original_nonselected_pre.replace(
                    'position="30"',
                    'position="not-an-integer"',
                ),
                "must have exactly one integer position label",
                "malformed position in non-selected metric",
            ),
        )
        for tampered, expected_error, label in raw_position_tampers:
            nonselected_pre.write_text(tampered)
            try:
                reduce_arm(
                    fixed_tail_root,
                    4,
                    4,
                    expected_hydra=False,
                    fixed32=True,
                )
            except ValueError as error:
                assert expected_error in str(error)
            else:
                raise AssertionError(f"{label} did not fail closed")
            finally:
                nonselected_pre.write_text(original_nonselected_pre)

        env_path = fixed_tail_root / "container_env.txt"
        original_env = env_path.read_text()
        fixed_tail_spec = fixed32_arm_spec(False)
        tamper_cases = (
            (
                "FR13_FIXED32_MODE",
                fixed_tail_spec["mode"],
                "hydra27_fixed32",
            ),
            (
                "FR13_FIXED32_VALID_MASK",
                f"{fixed_tail_spec['valid_mask']:#010x}",
                "0x00000000",
            ),
            (
                "FR13_FIXED32_ACTIVE_NODES",
                str(fixed_tail_spec["active_drafts"]),
                "20",
            ),
            (
                "FR13_FIXED32_PHYSICAL_DRAFTS",
                str(fixed_tail_spec["physical_drafts"]),
                "30",
            ),
        )
        for key, original, tampered in tamper_cases:
            env_path.write_text(
                original_env.replace(
                    f"{key}={original}\n",
                    f"{key}={tampered}\n",
                )
            )
            try:
                reduce_arm(
                    fixed_tail_root,
                    4,
                    4,
                    expected_hydra=False,
                    fixed32=True,
                )
            except ValueError as error:
                assert f"expected exactly {key}={original}" in str(error)
            else:
                raise AssertionError(f"{key} tamper did not fail closed")
            finally:
                env_path.write_text(original_env)

        sixteen_runroot = base / "sixteen"
        sixteen_tail_root = sixteen_runroot / "tail_fixed32"
        sixteen_hydra_root = sixteen_runroot / "hydra_fixed32"
        sixteen_tail_root.mkdir(parents=True)
        sixteen_hydra_root.mkdir(parents=True)
        write_fixed32_provenance_fixture(sixteen_tail_root, False)
        write_fixed32_provenance_fixture(sixteen_hydra_root, True)
        write_campaign_fixture(
            sixteen_tail_root,
            1,
            task_count=16,
        )
        write_campaign_fixture(
            sixteen_hydra_root,
            1,
            task_count=16,
        )
        for index, task_id in enumerate(CANONICAL_TASK_IDS):
            write_fixture(
                sixteen_tail_root,
                task_id,
                10,
                [8, 4, 2] + [0] * 28,
                tokens_per_draft=31,
                mtime=400 + index,
            )
            write_fixture(
                sixteen_hydra_root,
                task_id,
                10,
                [8, 3, 0] + [0] * 28,
                tokens_per_draft=31,
                mtime=400 + index,
            )
        sixteen_gate_path = sixteen_runroot / "fixed32_floor_gate.json"
        write_floor_gate_fixture(
            sixteen_gate_path,
            sixteen_runroot,
            sixteen_tail_root,
            sixteen_hydra_root,
            task_count=16,
            concurrency=1,
        )
        sixteen_binding, sixteen_metrics = validate_floor_gate_binding(
            sixteen_gate_path,
            sixteen_tail_root,
            sixteen_hydra_root,
            required_task_count=16,
            concurrency=1,
        )
        assert sixteen_binding["task_count"] == 16
        assert sixteen_binding["metric_bracket_count"] == 64
        assert sixteen_binding["arm_artifact_counts"] == {
            "tail6_fixed32": 65,
            "hydra27_fixed32": 65,
        }
        sixteen_tail = reduce_arm(
            sixteen_tail_root,
            16,
            1,
            expected_hydra=False,
            fixed32=True,
            metric_bindings=sixteen_metrics["tail6_fixed32"],
        )
        sixteen_hydra = reduce_arm(
            sixteen_hydra_root,
            16,
            1,
            expected_hydra=True,
            fixed32=True,
            metric_bindings=sixteen_metrics["hydra27_fixed32"],
        )
        assert sixteen_tail["instance_ids"] == list(CANONICAL_TASK_IDS)
        assert sixteen_hydra["instance_ids"] == list(CANONICAL_TASK_IDS)
        assert len(sixteen_tail["per_task"]) == 16
        assert len(sixteen_hydra["per_task"]) == 16
        assert all(len(task["depths"]) == 31 for task in sixteen_tail["per_task"])
        assert all(len(task["depths"]) == 31 for task in sixteen_hydra["per_task"])

        reordered_sixteen_gate = json.loads(sixteen_gate_path.read_text())
        reordered_ids = list(CANONICAL_TASK_IDS)
        reordered_ids[0], reordered_ids[1] = reordered_ids[1], reordered_ids[0]
        reordered_sixteen_gate["arms"]["tail6_fixed32"][
            "canonical_task_ids"
        ] = reordered_ids
        sixteen_gate_path.write_text(
            json.dumps(reordered_sixteen_gate, sort_keys=True) + "\n"
        )
        try:
            validate_floor_gate_binding(
                sixteen_gate_path,
                sixteen_tail_root,
                sixteen_hydra_root,
                required_task_count=16,
                concurrency=1,
            )
        except ValueError as error:
            assert "field 'canonical_task_ids' does not match" in str(error)
        else:
            raise AssertionError("reordered canonical 16-task IDs did not fail closed")

        try:
            reduce_arm(
                fixed_tail_root,
                5,
                4,
                expected_hydra=False,
                fixed32=True,
            )
        except ValueError as error:
            assert "exactly 4 or 16" in str(error)
        else:
            raise AssertionError("noncanonical task count did not fail closed")
    print("self-test OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail-root", type=Path)
    parser.add_argument("--hydra-root", type=Path)
    parser.add_argument("--required-task-count", type=int, choices=(4, 16), default=4)
    parser.add_argument("--concurrency", type=int, choices=(1, 4))
    parser.add_argument(
        "--fixed32",
        action="store_true",
        help=(
            "reduce Tail21/Hydra27 fixed-work arms and require 31 physical "
            "draft tokens per event"
        ),
    )
    parser.add_argument(
        "--floor-gate",
        type=Path,
        help=(
            "fixed32 formal evidence report; defaults to the arm roots' common "
            "parent/fixed32_floor_gate.json"
        ),
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.tail_root is None or args.hydra_root is None:
        parser.error("--tail-root and --hydra-root are required")
    if args.concurrency is None:
        parser.error("--concurrency is required to select overlap-safe reduction")
    if args.floor_gate is not None and not args.fixed32:
        parser.error("--floor-gate is only valid with --fixed32")

    floor_gate_binding = None
    metric_bindings = None
    if args.fixed32:
        gate_path = locate_floor_gate_report(
            args.floor_gate,
            args.tail_root,
            args.hydra_root,
        )
        floor_gate_binding, metric_bindings = validate_floor_gate_binding(
            gate_path,
            args.tail_root,
            args.hydra_root,
            required_task_count=args.required_task_count,
            concurrency=args.concurrency,
        )

    tail = reduce_arm(
        args.tail_root,
        args.required_task_count,
        args.concurrency,
        expected_hydra=False,
        fixed32=args.fixed32,
        metric_bindings=(
            metric_bindings["tail6_fixed32"] if metric_bindings is not None else None
        ),
    )
    hydra = reduce_arm(
        args.hydra_root,
        args.required_task_count,
        args.concurrency,
        expected_hydra=True,
        fixed32=args.fixed32,
        metric_bindings=(
            metric_bindings["hydra27_fixed32"] if metric_bindings is not None else None
        ),
    )
    comparison = compare(tail, hydra)
    report = build_report(
        tail,
        hydra,
        comparison,
        required_task_count=args.required_task_count,
        concurrency=args.concurrency,
        fixed32=args.fixed32,
        floor_gate_binding=floor_gate_binding,
    )
    print_table(tail, hydra, comparison)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

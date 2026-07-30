#!/usr/bin/env python3
"""Read-only floor-SLO reducer for canonical real SWE-Verified campaigns.

The reducer deliberately uses different uncertainty models for B=1 and B=4.
At B=1, a whole SWE task is the sampling cluster. At B=4, task brackets
overlap global counters, so the reducer selects their counter-index union once
and reports only time-series-conditional moving-block sensitivity. Fixed-work
census files bind the complete SFWD stream before that task selection is made.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import re
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fr13_fixed32_contract as fixed32_contract  # noqa: E402
from fr13_fixed32_contract import (  # noqa: E402
    CONTAINER_FA2_DESTINATION,
    ContractError as Fixed32ContractError,
    expected_pid1_argv,
    fixed32_tree_text,
    validate_external_manifest,
    validate_runtime_attestation,
)
from fr13_fixed32_topology import (  # noqa: E402
    HYDRA27_ACTIVE_DRAFTS,
    HYDRA27_VALID_MASK,
    PHYSICAL_DRAFTS,
    TAIL6_ACTIVE_DRAFTS,
    TAIL6_VALID_MASK,
)
from fr13_fixed32_flush_protocol import (  # noqa: E402
    ACK_KEYS as FLUSH_ACK_KEYS,
    ACK_SCHEMA as FLUSH_ACK_SCHEMA,
    READY_NONCE as FLUSH_READY_NONCE,
    REQUEST_KEYS as FLUSH_REQUEST_KEYS,
    REQUEST_SCHEMA as FLUSH_REQUEST_SCHEMA,
    RESULT_SCHEMA as FLUSH_RESULT_SCHEMA,
)
from fr13_fixed32_work_census import (  # noqa: E402
    CensusError as WorkCensusError,
)
from fr13_fixed32_work_census import CONV_PREGATHER_BLOCK  # noqa: E402
from fr13_fixed32_work_census import CONV_PREGATHER_LAYERS  # noqa: E402
from fr13_fixed32_work_census import CONV_PREGATHER_ROW_ELEMS  # noqa: E402
from fr13_fixed32_work_census import FIXED_WORK_SCOPE  # noqa: E402
from fr13_fixed32_work_census import MODE_SEMANTICS as WORK_CENSUS_MODE_SEMANTICS  # noqa: E402
from fr13_fixed32_work_census import REPORT_SCHEMA as WORK_CENSUS_REPORT_SCHEMA  # noqa: E402
from fr13_fixed32_work_census import SUPPORTED_BATCH_SIZES  # noqa: E402
from fr13_fixed32_work_census import load_jsonl as load_work_census_jsonl  # noqa: E402
from fr13_fixed32_work_census import reference_event as work_census_fixture  # noqa: E402
from fr13_fixed32_work_census import (  # noqa: E402
    reference_terminal_summary as work_census_terminal_fixture,
)
from fr13_fixed32_work_census import (  # noqa: E402
    validate_campaign as validate_work_census_campaign,
)
from fr13_fixed32_work_census import (  # noqa: E402
    forward_graph_structural_signature,
)
from fr13_runtime_manifest import (  # noqa: E402
    ManifestError as RuntimeManifestError,
)
from fr13_runtime_manifest import build_manifest as build_runtime_manifest  # noqa: E402


class GateError(RuntimeError):
    """An input artifact failed a fail-closed gate."""


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
PINNED_SWE_VERIFIED_PARQUET_RELATIVE = (
    ".cache/huggingface/hub/"
    "datasets--princeton-nlp--SWE-bench_Verified/blobs/"
    "a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd"
)
EVIDENCE_SETS = {
    4: {
        "relative_path": "output/fr13_b1_gold_swe/subset_b4_four.json",
        "sha256": ("0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"),
        "task_ids": CANONICAL_TASK_IDS[:4],
    },
    16: {
        "relative_path": "output/fr13_b1_gold_swe/subset_b4_sixteen.json",
        "sha256": ("47b0a3c9be49e2cb5f7e7217ae03c267a05359f269f3e3b038942f57d7dc0b5c"),
        "task_ids": CANONICAL_TASK_IDS,
    },
}

METRICS = {
    "fwd_s": "vllm:fr13_decode_forward_gpu_seconds_total",
    "fwd_steps": "vllm:fr13_decode_forward_gpu_steps_total",
    "fwd_drafts": "vllm:fr13_decode_forward_gpu_drafts_total",
    "wall_s": "vllm:fr13_decode_step_wall_seconds_total",
    "wall_drafts": "vllm:fr13_decode_step_wall_drafts_total",
    "wall_steps": "vllm:fr13_decode_step_wall_steps_total",
    "wall_attempts": "vllm:fr13_decode_step_wall_attempts_total",
    "wall_rejected": "vllm:fr13_decode_step_wall_rejected_total",
    "spec_drafts": "vllm:spec_decode_num_drafts_total",
    "spec_tokens": "vllm:spec_decode_num_draft_tokens_total",
}
INTEGRAL_METRICS = {
    "fwd_steps",
    "fwd_drafts",
    "wall_drafts",
    "wall_steps",
    "wall_attempts",
    "wall_rejected",
    "spec_drafts",
    "spec_tokens",
}
EXPECTED_METRIC_LABELS = {
    key: (
        'engine="0",model_name="qwen3.6-27b"'
        if key in {"spec_drafts", "spec_tokens"}
        else ""
    )
    for key in METRICS
}
PRETASK_REQUIRED_METRICS = frozenset({"spec_drafts", "spec_tokens"})
SAMPLE_RE = re.compile(
    r"^(?P<name>[^\s{]+)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+0-9.eE]+)$"
)
ORCHESTRATOR_HEADER_RE = re.compile(
    r"^=== \[[^\]]+\] dataset=(?P<dataset>\S+) tag=\S+ "
    r"n=(?P<tasks>\d+) concurrency=(?P<concurrency>\d+) ===$"
)
ORCHESTRATOR_DONE_RE = re.compile(r"^=== \[[^\]]+\] DONE n=(?P<tasks>\d+) .+ ===$")
TASK_START_RE = re.compile(r"^\[[^\]]+\] -> (?P<task>\S+)$")
TASK_END_RE = re.compile(r"^\[[^\]]+\] <- (?P<task>\S+) .+$")
ARM_HEADER_RE = re.compile(
    r"^=== BIGDENOM-VARIANT SWEServe ARM (?P<arm>\S+) "
    r"kind=(?P<kind>\S+) .* expect=(?P<tokens>\d+) .* "
    r"subset=(?P<subset>\S+) ===$"
)
ENGINE_CORE_PID_RE = re.compile(r"^PID (?P<pid>\d+) cmd=\[VLLM::EngineCore(?:\s|\])")
FIXED32_TREE = fixed32_tree_text()
FIXED32_PRESEED = (
    "[FR13_SUBTREE_PARALLEL] preseeded: n=32 schedule=fixed32 "
    "levels=[1, 11] lens=[5, 7] critical=12 (monolith 32) "
    "route_armed=1 selfcheck_armed=0"
)
FIXED32_ENGAGED = (
    "[FR13_SUBTREE_PARALLEL ENGAGED] n_actual=32 schedule=fixed32 critical=12"
)
FIXED32_WORK_ENGAGED = (
    "[FR13_FIXED32_WORK] engaged: physical_drafts=31 rows=32 "
    "gdn_launches=2 gdn_programs=12 gdn_slots=82 taw_walk=12 "
    "taw_buffer=32 output_slots=32 path_slots=16 reqkey_slots=16 "
    "kv_slots=16 conv_layers=48 committer_slots=16"
)
TAIL6_TOPOLOGY = (
    "[FR13_FIXED32] topology engaged: mode=tail6_fixed32 active_drafts=21 "
    "valid_mask=0x7a9ce73f"
)
HYDRA27_TOPOLOGY = (
    "[FR13_FIXED32] topology engaged: mode=hydra27_fixed32 active_drafts=27 "
    "valid_mask=0x7abdffff"
)
FIXED32_MODE_SPECS = {
    "tail6_fixed32": {
        "active_drafts": TAIL6_ACTIVE_DRAFTS,
        "valid_mask": TAIL6_VALID_MASK,
        "topology_needle": TAIL6_TOPOLOGY,
    },
    "hydra27_fixed32": {
        "active_drafts": HYDRA27_ACTIVE_DRAFTS,
        "valid_mask": HYDRA27_VALID_MASK,
        "topology_needle": HYDRA27_TOPOLOGY,
    },
}

WEIGHT_STREAM_LOWER_BOUND_MS = 98.6
COMPUTE_MS_PER_ROW = 0.54
SLO_MULTIPLIER = 1.15
REQUIRED_COVERAGE = 1.0
MIN_FULL_GRAPH_FRACTION = 0.99
MIN_TASK_COUNTER_STEPS = 64
MIN_B4_EXACT_EVENTS = 512
MIN_B4_GE3_FRACTION = 0.65
MIN_B4_MEAN_OCCUPANCY = 2.9
MAX_B4_MEAN_OCCUPANCY_GAP = 0.25
BLOCK_SENSITIVITY = (64, 128, 256, 512)
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20260729
RUNTIME_MANIFEST_PROFILE = "fixed32"
RUNTIME_MANIFEST_SEQUENCE = "scripts/fr13_fixed32_floor_timers_seq.sh"
FIXED32_BOUNDARY_SCHEMA = "fr13-fixed32-task-boundary-v1"
FIXED32_RUNTIME_SNAPSHOT_SCHEMA = "fr13-fixed32-boundary-snapshot-v2"
FIXED32_COUNTER_KEYS = frozenset(
    {
        "pure_decode_forward_steps",
        "complete_work_census_events",
        "work_census_first_forward_step",
        "work_census_last_forward_step",
        "sfwd_pending",
        "dfwd_pending",
        "cfwd_pending",
    }
)
FIXED32_STEP_METRIC = "vllm:fr13_fixed32_pure_decode_forward_steps_total"
FIXED32_CENSUS_METRIC = "vllm:fr13_fixed32_complete_work_census_events_total"
T95_ONE_SIDED = {
    3: 2.3533634348018264,
    15: 1.7530503556925547,
}


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise GateError(f"missing required artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@functools.lru_cache(maxsize=2)
def pinned_dataset_record_digests(repo_text: str) -> dict[str, str]:
    path = Path(repo_text) / PINNED_SWE_VERIFIED_PARQUET_RELATIVE
    if not path.is_file():
        raise GateError(f"pinned SWE-Verified Parquet is missing: {path}")
    try:
        import pyarrow.parquet as pq

        rows = pq.read_table(path).to_pylist()
    except Exception as error:
        raise GateError(
            f"cannot read pinned SWE-Verified Parquet {path}: {error}"
        ) from error
    digests: dict[str, str] = {}
    for row in rows:
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or instance_id in digests:
            raise GateError(f"{path}: invalid or duplicate instance_id")
        digests[instance_id] = hashlib.sha256(
            json.dumps(
                row,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    missing = sorted(set(CANONICAL_TASK_IDS) - set(digests))
    if missing:
        raise GateError(f"{path}: canonical task records are missing: {missing}")
    return digests


def read_text(path: Path) -> str:
    if not path.is_file():
        raise GateError(f"missing required artifact: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def strict_utf8_artifact(path: Path, *, label: str) -> tuple[bytes, str]:
    if not path.is_file():
        raise GateError(f"missing required artifact: {path}")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise GateError(f"{label}: artifact is not strict UTF-8: {error}") from error
    return raw, text


def exact_json_text(text: str, *, label: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise GateError(f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise GateError(f"{label}: non-finite JSON constant {value!r}")

    try:
        payload = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise GateError(f"{label}: invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise GateError(f"{label}: JSON root must be an object")
    return payload


def exact_json(path: Path, *, label: str) -> dict[str, Any]:
    _raw, text = strict_utf8_artifact(path, label=label)
    return exact_json_text(text, label=label)


def exact_keys(payload: dict[str, Any], keys: set[str] | frozenset[str], label: str) -> None:
    if set(payload) != set(keys):
        raise GateError(
            f"{label}: keys mismatch missing={sorted(set(keys) - set(payload))} "
            f"extra={sorted(set(payload) - set(keys))}"
        )


def fixed32_metric_text(text: str, *, label: str, metric: str) -> int:
    values = []
    for line in text.splitlines():
        match = SAMPLE_RE.match(line)
        if match is not None and match.group("name") == metric:
            if match.group("labels"):
                raise GateError(f"{label}: fixed32 flush metric must be unlabelled")
            values.append(float(match.group("value")))
    if len(values) != 1:
        raise GateError(f"{label}: expected exactly one {metric} series")
    return integral(values[0], f"{label}:{metric}")


def validate_fixed32_counters(payload: object, *, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise GateError(f"{label}: counters must be an object")
    exact_keys(payload, FIXED32_COUNTER_KEYS, f"{label}.counters")
    for key in (
        "pure_decode_forward_steps",
        "complete_work_census_events",
        "sfwd_pending",
        "dfwd_pending",
        "cfwd_pending",
    ):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GateError(f"{label}.{key}: expected nonnegative integer")
    for key in ("work_census_first_forward_step", "work_census_last_forward_step"):
        value = payload[key]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise GateError(f"{label}.{key}: expected null or nonnegative integer")
    if any(payload[key] != 0 for key in ("sfwd_pending", "dfwd_pending", "cfwd_pending")):
        raise GateError(f"{label}: flush acknowledged pending timer samples")
    steps = payload["pure_decode_forward_steps"]
    events = payload["complete_work_census_events"]
    first = payload["work_census_first_forward_step"]
    last = payload["work_census_last_forward_step"]
    if events > steps:
        raise GateError(f"{label}: complete census events exceed pure-decode steps")
    if events == 0:
        if first is not None or last is not None:
            raise GateError(f"{label}: empty census requires null first/last")
    elif first is None or last is None or not 0 <= first <= last < steps:
        raise GateError(f"{label}: census first/last range is invalid")
    return payload


def validate_fixed32_ack(
    payload: object,
    *,
    label: str,
    mode: str,
    producer_pid: int,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise GateError(f"{label}: ack must be an object")
    exact_keys(payload, FLUSH_ACK_KEYS, label)
    if (
        payload["schema"] != FLUSH_ACK_SCHEMA
        or payload["mode"] != mode
        or payload["producer_pid"] != producer_pid
        or payload["status"] != "ok"
    ):
        raise GateError(f"{label}: ack identity/status mismatch")
    generation = payload["generation"]
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise GateError(f"{label}: invalid generation")
    nonce = payload["nonce"]
    if (
        not isinstance(nonce, str)
        or re.fullmatch(r"[0-9a-f]{64}", nonce) is None
    ):
        raise GateError(f"{label}: invalid nonce")
    if not isinstance(payload["action"], str):
        raise GateError(f"{label}: action must be a string")
    validate_fixed32_counters(payload["counters"], label=label)
    return payload


def integral(value: float, label: str) -> int:
    rounded = round(value)
    if not math.isfinite(value) or abs(value - rounded) > 1e-6:
        raise GateError(f"{label} is not an integral counter: {value}")
    return int(rounded)


def metric_snapshot_text(
    text: str,
    *,
    label: str,
) -> tuple[dict[str, float], dict[str, str]]:
    wanted = {metric: key for key, metric in METRICS.items()}
    found: dict[str, float] = {}
    labels: dict[str, str] = {}
    for line in text.splitlines():
        match = SAMPLE_RE.match(line)
        if match is None:
            if line.startswith(tuple(wanted)):
                raise GateError(f"{label}: malformed required metric line {line!r}")
            continue
        key = wanted.get(match.group("name"))
        if key is None:
            continue
        if key in found:
            raise GateError(f"{label}: duplicate metric series for {METRICS[key]}")
        value = float(match.group("value"))
        if not math.isfinite(value) or value < 0:
            raise GateError(f"{label}: invalid metric {METRICS[key]}={value}")
        if key in INTEGRAL_METRICS:
            integral(value, f"{label}:{key}")
        found[key] = value
        labels[key] = match.group("labels") or ""
    missing = sorted(set(METRICS) - set(found))
    if missing:
        raise GateError(f"{label}: missing required metrics {missing}")
    return found, labels


def pretask_metric_snapshot_text(
    text: str,
    *,
    label: str,
) -> tuple[dict[str, float], dict[str, str]]:
    """Parse a raw generation-zero API scrape without requiring lazy worker series."""

    wanted = {metric: key for key, metric in METRICS.items()}
    found: dict[str, float] = {}
    labels: dict[str, str] = {}
    for line in text.splitlines():
        match = SAMPLE_RE.match(line)
        if match is None:
            if line.startswith(tuple(wanted)):
                raise GateError(f"{label}: malformed pretask metric line {line!r}")
            continue
        key = wanted.get(match.group("name"))
        if key is None:
            continue
        if key in found:
            raise GateError(f"{label}: duplicate metric series for {METRICS[key]}")
        value = float(match.group("value"))
        if not math.isfinite(value) or value < 0:
            raise GateError(f"{label}: invalid metric {METRICS[key]}={value}")
        if key in INTEGRAL_METRICS:
            integral(value, f"{label}:{key}")
        found[key] = value
        labels[key] = match.group("labels") or ""
    missing = sorted(PRETASK_REQUIRED_METRICS - set(found))
    if missing:
        raise GateError(f"{label}: missing required pretask metrics {missing}")
    if any(value != 0.0 for value in found.values()):
        raise GateError(f"{label}: pretask decode metrics are not exact zero")
    if any(labels[key] != EXPECTED_METRIC_LABELS[key] for key in found):
        raise GateError(f"{label}: pretask metric labels differ from the contract")
    return found, labels


def metric_snapshot(path: Path) -> tuple[dict[str, float], dict[str, str]]:
    _raw, text = strict_utf8_artifact(path, label=str(path))
    return metric_snapshot_text(text, label=str(path))


def load_metric_artifact(path: Path) -> dict[str, Any]:
    raw, text = strict_utf8_artifact(path, label=str(path))
    values, labels = metric_snapshot_text(text, label=str(path))
    return {
        "values": values,
        "labels": labels,
        "identity": {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        },
        "fixed32": {
            "pure_decode_forward_steps": fixed32_metric_text(
                text,
                label=str(path),
                metric=FIXED32_STEP_METRIC,
            ),
            "complete_work_census_events": fixed32_metric_text(
                text,
                label=str(path),
                metric=FIXED32_CENSUS_METRIC,
            ),
        },
    }


def validate_runtime_boundary_snapshot(
    path: Path,
    *,
    ack: dict[str, Any],
    server_capacity: int,
    metrics_path: Path | None,
    metric_values: dict[str, float] | None,
    reference: object,
    census_path: Path,
) -> dict[str, Any]:
    payload = exact_json(path, label=str(path))
    exact_keys(
        payload,
        {
            "schema",
            "mode",
            "producer_pid",
            "generation",
            "nonce",
            "action",
            "counters",
            "metrics",
        },
        str(path),
    )
    if (
        payload["schema"] != FIXED32_RUNTIME_SNAPSHOT_SCHEMA
        or payload["mode"] != ack["mode"]
        or payload["producer_pid"] != ack["producer_pid"]
        or payload["generation"] != ack["generation"]
        or payload["nonce"] != ack["nonce"]
        or payload["action"] != ack["action"]
        or payload["counters"] != ack["counters"]
    ):
        raise GateError(f"{path}: runtime snapshot does not bind to flush ack")
    expected_reference = {
        "schema": FIXED32_RUNTIME_SNAPSHOT_SCHEMA,
        "generation": ack["generation"],
        "path": str(path),
        "sha256": sha256_file(path),
    }
    if reference is not None and reference != expected_reference:
        raise GateError(f"{path}: task boundary runtime snapshot reference mismatch")

    metrics = payload["metrics"]
    exact_keys(
        metrics,
        {"fixed32", "sfwd", "dfwd", "cfwd", "committer", "conv_pregather"},
        f"{path}:metrics",
    )
    fixed = metrics["fixed32"]
    exact_keys(
        fixed,
        {
            "pure_decode_forward_steps",
            "complete_work_census_events",
            "complete_spec_rows",
            "spec_drafts",
            "spec_tokens",
            "batch_histogram",
            "first_forward_step",
            "last_forward_step",
            "events_sha256",
        },
        f"{path}:fixed32",
    )
    steps = integral(
        float(fixed["pure_decode_forward_steps"]), f"{path}:fixed32 steps"
    )
    events = integral(
        float(fixed["complete_work_census_events"]), f"{path}:fixed32 events"
    )
    spec_drafts = integral(float(fixed["spec_drafts"]), f"{path}:spec drafts")
    if (
        steps != ack["counters"]["pure_decode_forward_steps"]
        or events != ack["counters"]["complete_work_census_events"]
        or fixed["first_forward_step"]
        != ack["counters"]["work_census_first_forward_step"]
        or fixed["last_forward_step"]
        != ack["counters"]["work_census_last_forward_step"]
        or fixed["complete_spec_rows"] != spec_drafts
        or fixed["spec_tokens"] != 31 * spec_drafts
    ):
        raise GateError(f"{path}: fixed counters do not reconcile")
    histogram = fixed["batch_histogram"]
    if not isinstance(histogram, dict) or set(histogram) != {"1", "2", "3", "4"}:
        raise GateError(f"{path}: batch histogram keys mismatch")
    batch_counts = {
        int(batch): integral(float(count), f"{path}:batch {batch}")
        for batch, count in histogram.items()
    }
    if (
        any(count < 0 for count in batch_counts.values())
        or sum(batch_counts.values()) != events
        or sum(batch * count for batch, count in batch_counts.items()) != spec_drafts
        or any(
            batch > server_capacity and count
            for batch, count in batch_counts.items()
        )
    ):
        raise GateError(f"{path}: batch histogram does not reconcile")

    sfwd = metrics["sfwd"]
    exact_keys(
        sfwd,
        {
            "gpu_seconds",
            "steps",
            "drafts",
            "wall_seconds",
            "wall_drafts",
            "wall_steps",
            "wall_rejected",
        },
        f"{path}:sfwd",
    )
    sfwd_values = {
        key: float(value)
        for key, value in sfwd.items()
    }
    if any(not math.isfinite(value) or value < 0 for value in sfwd_values.values()):
        raise GateError(f"{path}: SFWD values must be finite and nonnegative")
    for key in ("steps", "drafts", "wall_drafts", "wall_steps", "wall_rejected"):
        integral(sfwd_values[key], f"{path}:sfwd {key}")
    if (
        integral(sfwd_values["steps"], f"{path}:sfwd steps") != steps
        or integral(sfwd_values["drafts"], f"{path}:sfwd drafts") != spec_drafts
    ):
        raise GateError(f"{path}: SFWD counters do not reconcile")
    for label in ("dfwd", "cfwd"):
        span = metrics[label]
        exact_keys(span, {"gpu_seconds", "spans"}, f"{path}:{label}")
        seconds = float(span["gpu_seconds"])
        if not math.isfinite(seconds) or seconds < 0:
            raise GateError(f"{path}: {label} seconds are invalid")
        if integral(float(span["spans"]), f"{path}:{label} spans") != events:
            raise GateError(f"{path}: {label} spans do not reconcile")

    committer = metrics["committer"]
    pregather = metrics["conv_pregather"]
    expected_by_batch = {
        str(batch): batch_counts[batch] for batch in range(1, 5)
    }
    expected_capture_by_batch = {
        str(batch): int(batch <= server_capacity) for batch in range(1, 5)
    }
    zero_by_batch = {str(batch): 0 for batch in range(1, 5)}
    pregather_keys = {
        "preseeded",
        "pointer_entries",
        "preseeded_batches",
        "max_batch_size",
        "graph_capture_stages",
        "graph_capture_stages_by_batch",
        "profile_capture_stages",
        "aux_capture_stages",
        "actual_stages",
        "actual_stages_by_batch",
        "graph_replay_stages",
        "graph_replay_stages_by_batch",
    }
    if isinstance(pregather, dict):
        exact_keys(pregather, pregather_keys, f"{path}:conv_pregather")
    pregather_integer_keys = (
        "pointer_entries",
        "max_batch_size",
        "graph_capture_stages",
        "profile_capture_stages",
        "aux_capture_stages",
        "actual_stages",
        "graph_replay_stages",
    )
    pregather_integer_types_ok = isinstance(pregather, dict) and all(
        isinstance(pregather.get(key), int)
        and not isinstance(pregather.get(key), bool)
        for key in pregather_integer_keys
    )
    pregather_maps_ok = isinstance(pregather, dict) and all(
        isinstance(pregather.get(key), dict)
        and set(pregather[key]) == {"1", "2", "3", "4"}
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in pregather[key].values()
        )
        for key in (
            "graph_capture_stages_by_batch",
            "actual_stages_by_batch",
            "graph_replay_stages_by_batch",
        )
    )
    preseeded_batches = pregather.get("preseeded_batches") if isinstance(
        pregather, dict
    ) else None
    preseeded_batches_ok = (
        isinstance(preseeded_batches, list)
        and all(
            isinstance(batch, int) and not isinstance(batch, bool)
            for batch in preseeded_batches
        )
        and preseeded_batches == list(range(1, server_capacity + 1))
    )
    if (
        not isinstance(committer, dict)
        or committer.get("all_batches_ready") is not True
        or committer.get("required_capacity") != server_capacity
        or committer.get("captures") != server_capacity
        or committer.get("preseeded_graphs") != server_capacity
        or committer.get("actual_replays_enqueued") != events
        or committer.get("actual_replays_by_batch") != expected_by_batch
        or not isinstance(pregather, dict)
        or not pregather_integer_types_ok
        or not pregather_maps_ok
        or not preseeded_batches_ok
        or pregather.get("preseeded") is not True
        or pregather.get("pointer_entries") != 48
        or pregather.get("max_batch_size") != server_capacity
        or pregather.get("graph_capture_stages") != server_capacity
        or pregather.get("graph_capture_stages_by_batch")
        != expected_capture_by_batch
        or pregather.get("profile_capture_stages") != 0
        or pregather.get("aux_capture_stages") != 0
        or pregather.get("actual_stages") != 0
        or pregather.get("actual_stages_by_batch") != zero_by_batch
        or pregather.get("graph_replay_stages") != events
        or pregather.get("graph_replay_stages_by_batch") != expected_by_batch
    ):
        raise GateError(
            f"{path}: committer/in-graph pregather counters do not reconcile"
        )

    census_lines = read_text(census_path).splitlines()
    if len(census_lines) < events + 1:
        raise GateError(f"{path}: census stream is shorter than snapshot prefix")
    try:
        census_prefix = [json.loads(line) for line in census_lines[:events]]
    except json.JSONDecodeError as error:
        raise GateError(f"{census_path}: invalid JSONL: {error}") from error
    expected_events_hash = hashlib.sha256(
        json.dumps(
            census_prefix,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if fixed["events_sha256"] != expected_events_hash:
        raise GateError(f"{path}: census prefix digest mismatch")

    if metrics_path is not None:
        if metric_values is None:
            raise GateError(f"{metrics_path}: cached parsed metrics are missing")
        observed = metric_values
        expected_metrics = {
            "fwd_s": sfwd_values["gpu_seconds"],
            "fwd_steps": float(steps),
            "fwd_drafts": float(spec_drafts),
            "wall_s": sfwd_values["wall_seconds"],
            "wall_drafts": sfwd_values["wall_drafts"],
            "wall_steps": sfwd_values["wall_steps"],
            "wall_attempts": (
                sfwd_values["wall_steps"] + sfwd_values["wall_rejected"]
            ),
            "wall_rejected": sfwd_values["wall_rejected"],
            "spec_drafts": float(spec_drafts),
            "spec_tokens": float(31 * spec_drafts),
        }
        for key, expected in expected_metrics.items():
            if not math.isclose(
                observed[key], expected, rel_tol=0.0, abs_tol=1e-9
            ):
                raise GateError(
                    f"{metrics_path}: metric {key} differs from frozen "
                    "runtime snapshot"
                )
    return {
        "path": str(path),
        "sha256": expected_reference["sha256"],
        "generation": ack["generation"],
        "events_sha256": expected_events_hash,
    }


def validate_subset(path: Path, task_count: int) -> dict[str, Any]:
    expected = EVIDENCE_SETS[task_count]
    actual_hash = sha256_file(path)
    if actual_hash != expected["sha256"]:
        raise GateError(
            f"{path}: canonical subset sha256 mismatch; "
            f"expected {expected['sha256']}, got {actual_hash}"
        )
    try:
        payload = json.loads(read_text(path))
    except json.JSONDecodeError as error:
        raise GateError(f"{path}: invalid subset JSON: {error}") from error
    if payload.get("dataset_name") != "princeton-nlp/SWE-bench_Verified":
        raise GateError(f"{path}: subset is not SWE-bench_Verified")
    actual_ids = payload.get("instance_ids")
    if actual_ids != list(expected["task_ids"]):
        raise GateError(
            f"{path}: subset IDs do not exactly match canonical {task_count}-task set"
        )
    return {
        "path": str(path),
        "sha256": actual_hash,
        "task_ids": list(expected["task_ids"]),
    }


def parse_orchestrator(arm_dir: Path, task_count: int) -> dict[str, Any]:
    path = arm_dir / "swe_orchestrator.log"
    lines = read_text(path).splitlines()
    headers = [match for line in lines if (match := ORCHESTRATOR_HEADER_RE.match(line))]
    done = [match for line in lines if (match := ORCHESTRATOR_DONE_RE.match(line))]
    if len(headers) != 1 or len(done) != 1:
        raise GateError(
            f"{path}: expected exactly one campaign header and one DONE footer"
        )
    header = headers[0]
    recorded_tasks = int(header.group("tasks"))
    concurrency = int(header.group("concurrency"))
    if header.group("dataset") != "princeton-nlp/SWE-bench_Verified":
        raise GateError(f"{path}: campaign is not SWE-bench_Verified")
    if recorded_tasks != task_count or int(done[0].group("tasks")) != task_count:
        raise GateError(
            f"{path}: requested {task_count} tasks but header/footer do not match"
        )
    if concurrency not in (1, 4):
        raise GateError(f"{path}: unsupported inferred concurrency {concurrency}")
    expected_ids = sorted(EVIDENCE_SETS[task_count]["task_ids"])
    starts = sorted(
        match.group("task") for line in lines if (match := TASK_START_RE.match(line))
    )
    ends = sorted(
        match.group("task") for line in lines if (match := TASK_END_RE.match(line))
    )
    if starts != expected_ids or ends != expected_ids:
        raise GateError(
            f"{path}: start/completion records are not the canonical completed set"
        )
    return {
        "path": str(path),
        "inferred_concurrency": concurrency,
        "completed_task_ids": expected_ids,
    }


def task_directories(arm_dir: Path, task_count: int) -> list[Path]:
    root = arm_dir / "swe_out" / "verified" / "per_task"
    if not root.is_dir():
        raise GateError(f"missing task artifact directory: {root}")
    directories = sorted(path for path in root.iterdir() if path.is_dir())
    expected_ids = sorted(EVIDENCE_SETS[task_count]["task_ids"])
    if [path.name for path in directories] != expected_ids:
        raise GateError(
            f"{root}: task directories are not the exact canonical completed set"
        )
    for task_dir in directories:
        metadata_path = task_dir / "runner_metadata.json"
        metadata = exact_json(metadata_path, label=str(metadata_path))
        if metadata.get("instance_id") != task_dir.name or not metadata.get("ended_at"):
            raise GateError(f"{metadata_path}: task is not recorded as completed")
        for name in ("vllm_metrics_pre.txt", "vllm_metrics_post.txt"):
            if not (task_dir / name).is_file():
                raise GateError(f"{task_dir}: incomplete metrics bracket")
    return directories


def _positive_token_usage(value: Any) -> bool:
    if isinstance(value, list):
        return any(_positive_token_usage(item) for item in value)
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
            token_count = usage.get(key)
            if (
                isinstance(token_count, int)
                and not isinstance(token_count, bool)
                and token_count > 0
            ):
                return True
    return any(_positive_token_usage(item) for item in value.values())


def _trace_has_assistant_output(event: dict[str, Any]) -> bool:
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


def validate_real_task_provenance(
    arm_dir: Path,
    task_dirs: list[Path],
    *,
    mode: str,
) -> dict[str, Any]:
    task_records: dict[str, dict[str, Any]] = {}
    for task_dir in task_dirs:
        metadata_path = task_dir / "runner_metadata.json"
        metadata = exact_json(metadata_path, label=str(metadata_path))
        agent = metadata.get("agent")
        codex = metadata.get("codex")
        if (
            not isinstance(agent, dict)
            or not isinstance(codex, dict)
            or canonical_json_sha256(codex) != canonical_json_sha256(agent)
        ):
            raise GateError(
                f"{metadata_path}: fixed32 agent/codex terminal metadata differs"
            )
        exit_code = agent.get("exit_code")
        if (
            isinstance(exit_code, bool)
            or not isinstance(exit_code, int)
            or exit_code != 0
            or agent.get("timed_out") is not False
            or agent.get("offloaded") is not True
            or agent.get("network_drop") is not False
            or (
                agent.get("stall_killed") is not None
                and agent.get("stall_killed") is not False
            )
        ):
            raise GateError(
                f"{metadata_path}: fixed32 task agent did not complete cleanly"
            )
        eval_report = metadata.get("eval_report")
        verdict = eval_report.get("verdict") if isinstance(eval_report, dict) else None
        if verdict not in {"resolved", "failed"}:
            raise GateError(
                f"{metadata_path}: fixed32 task has no terminal SWE verdict"
            )
        provenance = metadata.get("fixed32_real_task_provenance")
        assistant_output_event_count = (
            provenance.get("assistant_output_event_count")
            if isinstance(provenance, dict)
            else None
        )
        if (
            not isinstance(provenance, dict)
            or provenance.get("schema")
            != "fr13-fixed32-real-task-provenance-v1"
            or provenance.get("instance_id") != task_dir.name
            or provenance.get("positive_token_usage") is not True
            or isinstance(assistant_output_event_count, bool)
            or not isinstance(assistant_output_event_count, int)
            or assistant_output_event_count <= 0
        ):
            raise GateError(
                f"{metadata_path}: fixed32 real-task provenance is incomplete"
            )
        trace_path = task_dir / "qwen_trace.jsonl"
        raw_trace = trace_path.read_bytes()
        if (
            not raw_trace
            or provenance.get("trace_path") != str(trace_path.resolve())
            or provenance.get("trace_sha256")
            != hashlib.sha256(raw_trace).hexdigest()
            or provenance.get("trace_bytes") != len(raw_trace)
        ):
            raise GateError(
                f"{metadata_path}: fixed32 trace identity does not match provenance"
            )
        events = []
        try:
            trace_text = raw_trace.decode("utf-8")
            for line_number, line in enumerate(trace_text.splitlines(), start=1):
                if not line.strip():
                    raise GateError(f"{trace_path}: blank JSONL record")
                event = exact_json_text(
                    line,
                    label=f"{trace_path}:{line_number}",
                )
                events.append(event)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GateError(f"{trace_path}: invalid strict JSONL: {error}") from error
        event_count = provenance.get("event_count")
        observed_assistant_output_events = sum(
            _trace_has_assistant_output(event) for event in events
        )
        if (
            isinstance(event_count, bool)
            or not isinstance(event_count, int)
            or len(events) != event_count
            or observed_assistant_output_events != assistant_output_event_count
            or not _positive_token_usage(events)
        ):
            raise GateError(
                f"{trace_path}: trace content does not prove real model traffic"
            )
        terminal = provenance.get("agent_terminal")
        if not isinstance(terminal, dict) or any(
            terminal.get(key) != agent.get(key)
            or type(terminal.get(key)) is not type(agent.get(key))
            for key in ("exit_code", "timed_out", "offloaded", "network_drop")
        ):
            raise GateError(
                f"{metadata_path}: provenance terminal fields differ from agent metadata"
            )
        task_records[task_dir.name] = {
            "trace_path": str(trace_path),
            "trace_sha256": provenance["trace_sha256"],
            "trace_bytes": len(raw_trace),
            "trace_events": len(events),
            "assistant_output_events": provenance[
                "assistant_output_event_count"
            ],
            "positive_token_usage": True,
            "agent_terminal": terminal,
            "eval_verdict": verdict,
        }

    fetch_status = arm_dir / "offload_fetch_status.txt"
    if read_text(fetch_status) != "ok\n":
        raise GateError(f"{fetch_status}: offload pair-dump fetch was not successful")
    pair_root = arm_dir / "proxy_pair_dumps"
    if not pair_root.is_dir():
        raise GateError(f"{pair_root}: missing proxy pair-dump directory")
    pair_paths = sorted(path for path in pair_root.iterdir() if path.is_file())
    if not pair_paths:
        raise GateError(f"{pair_root}: no proxy pair dumps")
    pair_bindings = {task_dir.name: [] for task_dir in task_dirs}
    task_ids = set(pair_bindings)
    zero_usage_pair_count = 0
    for pair_path in pair_paths:
        pair = exact_json(pair_path, label=str(pair_path))
        if pair.get("schema") != "lumo.fr13.proxy_pair_dump.v1":
            raise GateError(f"{pair_path}: wrong proxy pair schema")
        request = pair.get("request")
        response = pair.get("response")
        if not isinstance(request, dict) or not isinstance(response, dict):
            raise GateError(f"{pair_path}: proxy pair request/response is incomplete")
        request_text = json.dumps(
            request, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        )
        matched = sorted(
            task_id
            for task_id in task_ids
            if f"# SWE-Bench task: {task_id}" in request_text
        )
        if len(matched) > 1:
            raise GateError(f"{pair_path}: proxy request names multiple canonical tasks")
        if not matched:
            if _positive_token_usage(response):
                raise GateError(
                    f"{pair_path}: unmatched proxy response has positive token usage"
                )
            zero_usage_pair_count += 1
            continue
        if not _positive_token_usage(response):
            raise GateError(
                f"{pair_path}: task-bound proxy response has no positive token usage"
            )
        pair_bindings[matched[0]].append(
            {
                "path": str(pair_path),
                "sha256": sha256_file(pair_path),
                "bytes": pair_path.stat().st_size,
                "response_positive_token_usage": True,
            }
        )
    missing = sorted(task_id for task_id, rows in pair_bindings.items() if not rows)
    if missing:
        raise GateError(
            f"{pair_root}: no task-bound positive-usage proxy pair for {missing}"
        )
    for task_id, rows in pair_bindings.items():
        task_records[task_id]["proxy_pairs"] = rows
    positive_pair_count = sum(len(rows) for rows in pair_bindings.values())
    pair_counts = {
        task_id: len(pair_bindings[task_id]) for task_id in sorted(task_ids)
    }
    audit_path = arm_dir / "fixed32_positive_traffic_audit.json"
    audit = exact_json(audit_path, label=str(audit_path))
    exact_keys(
        audit,
        {
            "schema",
            "mode",
            "all_positive_usage_pairs_bound_to_one_canonical_task",
            "positive_pair_count",
            "zero_usage_pair_count",
            "task_positive_pair_counts",
        },
        str(audit_path),
    )
    audit_pair_counts = audit["task_positive_pair_counts"]
    if (
        audit["schema"] != "fr13-fixed32-positive-traffic-audit-v1"
        or audit["mode"] != mode
        or audit["all_positive_usage_pairs_bound_to_one_canonical_task"] is not True
        or isinstance(audit["positive_pair_count"], bool)
        or not isinstance(audit["positive_pair_count"], int)
        or audit["positive_pair_count"] != positive_pair_count
        or isinstance(audit["zero_usage_pair_count"], bool)
        or not isinstance(audit["zero_usage_pair_count"], int)
        or audit["zero_usage_pair_count"] != zero_usage_pair_count
        or not isinstance(audit_pair_counts, dict)
        or audit_pair_counts != pair_counts
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in audit_pair_counts.values()
        )
    ):
        raise GateError(
            f"{audit_path}: positive-traffic audit does not match proxy pair bytes"
        )
    return {
        "all_canonical_tasks_have_real_model_traffic": True,
        "all_positive_usage_proxy_pairs_task_bound": True,
        "all_agents_completed_cleanly": True,
        "all_tasks_have_terminal_eval_verdicts": True,
        "offload_fetch_status": {
            "path": str(fetch_status),
            "sha256": sha256_file(fetch_status),
        },
        "proxy_pair_dump_count": len(pair_paths),
        "zero_usage_control_pair_count": zero_usage_pair_count,
        "positive_traffic_audit": {
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
            "positive_pair_count": positive_pair_count,
            "zero_usage_pair_count": zero_usage_pair_count,
            "task_positive_pair_counts": pair_counts,
        },
        "tasks": task_records,
    }


def resolve_subset_from_runlog(
    repo: Path,
    runroot: Path,
    arm: str,
    expected_kind: str,
    expected_tokens: int,
    task_count: int,
    concurrency: int,
) -> dict[str, Any]:
    path = runroot / f"{arm}.runlog"
    text = read_text(path)
    matches = [
        match for line in text.splitlines() if (match := ARM_HEADER_RE.match(line))
    ]
    if len(matches) != 1:
        raise GateError(f"{path}: expected exactly one BIGDENOM arm header")
    match = matches[0]
    if (
        match.group("arm") != arm
        or match.group("kind") != expected_kind
        or int(match.group("tokens")) != expected_tokens
    ):
        raise GateError(f"{path}: arm header does not match the requested arm")
    subset_path = Path(match.group("subset"))
    if not subset_path.is_absolute():
        subset_path = repo / subset_path
    subset = validate_subset(subset_path.resolve(), task_count)
    if expected_kind not in FIXED32_MODE_SPECS:
        raise GateError(f"{path}: unsupported fixed-32 mode {expected_kind!r}")
    process_path = runroot / arm / "fixed32_process_identity.json"
    process_identity = exact_json(process_path, label=str(process_path))
    if process_identity.get("schema") != "fr13-fixed32-process-identity-v1":
        raise GateError(f"{process_path}: wrong process identity schema")
    pid1 = process_identity.get("pid1")
    engine_core = process_identity.get("engine_core")
    if (
        not isinstance(pid1, dict)
        or isinstance(pid1.get("pid"), bool)
        or pid1.get("pid") != 1
        or not isinstance(pid1.get("environ"), list)
        or not isinstance(engine_core, dict)
        or isinstance(engine_core.get("pid"), bool)
        or not isinstance(engine_core.get("pid"), int)
        or not isinstance(engine_core.get("environ"), list)
    ):
        raise GateError(f"{process_path}: incomplete PID1/EngineCore identity")
    argv = pid1.get("argv")
    expected_argv = expected_pid1_argv(concurrency)
    if argv != expected_argv:
        raise GateError(
            f"{process_path}: PID1 argv differs from the exact fixed32 contract"
        )
    fa2_path = str(CONTAINER_FA2_DESTINATION)
    if not any(
        fa2_path in line for line in engine_core.get("forked_fa2_maps", [])
    ):
        raise GateError(
            f"{process_path}: EngineCore did not map the pinned forked FA2 binary"
        )
    ratio_needle = f"draft_tokens/drafts={float(expected_tokens):.1f}"
    if text.count(ratio_needle) != 1:
        raise GateError(f"{path}: exact warmup draft-token ratio needle is absent")
    engine_core_pids = [
        int(pid_match.group("pid"))
        for line in text.splitlines()
        if (pid_match := ENGINE_CORE_PID_RE.match(line))
    ]
    if len(engine_core_pids) != 1:
        raise GateError(
            f"{path}: expected exactly one recorded VLLM::EngineCore PID, "
            f"got {engine_core_pids}"
        )
    if engine_core.get("pid") != engine_core_pids[0]:
        raise GateError(
            f"{process_path}: EngineCore PID differs from the runlog producer PID"
        )
    return {
        "runlog": str(path),
        "subset": subset,
        "pid1_argv": argv,
        "pid1_exact_contract": True,
        "process_identity": {
            "path": str(process_path),
            "sha256": sha256_file(process_path),
        },
        "engine_core_pid": engine_core_pids[0],
    }


def fixed32_required_env(arm_dir: Path, *, mode: str) -> dict[str, str]:
    try:
        mode_spec = FIXED32_MODE_SPECS[mode]
    except KeyError as error:
        raise GateError(f"{arm_dir}: unsupported fixed-32 mode {mode!r}") from error
    required_env = {
        "FR13_HYDRA23": "0",
        "FR13_TAIL_MODE": "1",
        "FR13_DRAFT_SOURCE": "merged",
        "FR13_TREE_GDN_GEOM_OVERRIDE": "BV=8",
        "FR13_FIXED32_MODE": mode,
        "FR13_FIXED32_VALID_MASK": f"{mode_spec['valid_mask']:#010x}",
        "FR13_FIXED32_ACTIVE_NODES": str(mode_spec["active_drafts"]),
        "FR13_FIXED32_PHYSICAL_DRAFTS": "31",
        "FR13_FIXED32_ENGINE_PID_FILE": "/logs/fr13_fixed32_engine_pid",
        "FR13_FIXED32_FLUSH_REQUEST_PATH": (
            "/logs/fr13_fixed32_flush_request.json"
        ),
        "FR13_FIXED32_FLUSH_ACK_PATH": "/logs/fr13_fixed32_flush_ack.json",
        "FR13_FIXED32_WORK_CENSUS_PATH": (
            "/logs/fr13_fixed32_work_census.jsonl"
        ),
        "FR13_FIXED32_BOUNDARY_SNAPSHOT_PATH": (
            "/logs/fr13_fixed32_boundary_snapshot"
        ),
        "FR13_FIXED32_WORK_CENSUS": "1",
        "FR13_FIXED32_DEVICE_PUBLISH": "1",
        "FR13_FIXED32_ACCEPT_PACK": "1",
        "FR13_FIXED32_REQKEY_DEVICE": "1",
        "FR13_FIXED32_KV_REMAP16": "1",
        "FR13_FIXED32_COMMIT_DEVICE_FILL": "1",
        "FR13_FIXED32_TAW_WALK_CAP": "12",
        "FR13_DEVICE_MULTIDRAFT": "1",
        "FR13_DRAFTER_GRAPH": "1",
        "FR13_DRAFTER_SINGLE_LOGITS": "1",
        "FR13_DM_DEPTHSYNC": "1",
        "FR13_TAW": "1",
        "FR13_PARENT_GATHER": "1",
        "FR13_EAGER_PACK": "1",
        "FR13_COMMIT_BATCH_OUTPUT": "1",
        "FR13_COMMITTER_NATIVE": "1",
        "FR13_COMMITTER_BATCHED": "1",
        "FR13_COMMITTER_GRAPH": "1",
        "FR13_COMMIT_OVERLAP": "0",
        "FR13_RING_EXPORT": "1",
        "FR13_REPLAY_ROUTE": "1",
        "FR13_ATTN_KV_REMAP": "1",
        "FR13_SLOT_REORDER": "1",
        "FR13_KV_REMAP_SYNCFREE": "1",
        "FR13_FA2_TREE_BIAS": "1",
        "FR13_TREE_CONV_FUSED": "1",
        "FR13_CONV_WB_FUSED": "1",
        "FR13_CONV_WB_BATCHED": "1",
        "FR13_CONV_PREGATHER": "1",
        "FR13_CONV_COMMITTED_PATH": "1",
        "FR13_APC_COMMIT_TO_RUNNING_ROW": "1",
        "FR13_TREE_RUNROW_INIT": "1",
        "FR13_FLAGS_INKERNEL": "1",
        "FR13_SFWD_GPU_TIMER": "1",
        "FR13_SFWD_GPU_TIMER_JSON": (
            f"/workspace/output/fr13_sfwd_sidecar/{arm_dir.name}.json"
        ),
        "FR13_TIMER_EXPLICIT_FLUSH": "1",
        "FR13_STEP_WALL_CAP_S": "1.5",
        "FR13_STEP_GRAPH": "0",
        "FR13_SUBTREE_PARALLEL": "1",
        "FR13_SUBTREE_PARALLEL_SELFCHECK": "0",
    }
    required_env.update(
        {
            "ATTENTION_BACKEND": "TREE_ATTN",
            "FR10_DECODE_MODE_DEFAULT": "tree_mtp",
            "FR10_METRICS": "0",
            "VLLM_BATCH_INVARIANT": "0",
            "LUMO_BATCH_INVARIANT_VLLM": "0",
            "LUMO_FB_KERNEL_ROWS": "1",
            "LUMO_FB_PROJ_PAD_ROWS": "16",
            "FR13_ENABLE_APC": "1",
            "FR13_APC_CONFIG_ONLY": "0",
            "FR13_INPUTPREP_GUARD": "1",
            "FR13_DRAFT_VOCAB_K": "65536",
            "FR13_DRAFT_VOCAB_BLOCKS": (
                "/workspace/scripts/fr13_dvk_subset_blocks.json"
            ),
            "FR13_DEVICE_MULTIDRAFT": "1",
            "FR13_DFWD_GPU_TIMER": "1",
            "FR13_DFWD_GPU_TIMER_JSON": (
                f"/workspace/output/fr13_sfwd_sidecar/{arm_dir.name}_dfwd.json"
            ),
            "FR13_CFWD_GPU_TIMER": "1",
            "FR13_CFWD_GPU_TIMER_JSON": (
                f"/workspace/output/fr13_sfwd_sidecar/{arm_dir.name}_cfwd.json"
            ),
            "FR13_SFWD_GPU_TIMER_MAXPENDING": "256",
            "FR13_SFWD_GPU_TIMER_SAMPLES_MAX": "200000",
            "FR13_SFWD_GPU_TIMER_DUMP_S": "1",
            "FR13_SFWD_SAMPLES_DUMP_S": "30",
            "FR13_SPAN_GPU_TIMER_DUMP_S": "1",
            "FR13_WEIGHT_FLOOR_MS": "98.6",
            "FR13_COMPUTE_MS_PER_ROW": "0.54",
            "FR13_APC_CONV_FIX": "1",
            "FR13_APC_CONV_SNAPSHOT": "1",
            "FR13_APC_ZERO_MAMBA_ON_ALLOC": "1",
            "FR13_APC_COPY_SRC_FIX": "1",
            "FR13_FREE_TREE_POSGLOBALS": "0",
            "FR13_APC_BLOCK_ALIGN_45477": "1",
            "FR13_FULL_ATTN_KV_FP8": "0",
            "FR13_SERVE_BATCH_FLAGS": "",
            "FR13_MULTIDRAFT_GPU_TIMER": "0",
            "FR13_REPLAY_GPU_TIMER": "0",
            "FR13_COMMIT_FULL_GPU_TIMER": "0",
            "FR13_COMMITTER_SG_TIMER": "0",
            "FR13_REPLAY_ONLY_GPU_TIMER": "0",
            "FR13_GRAPH_TIMER": "0",
            "FR13_KVREMAP_TIMER": "0",
            "FR13_STATEREMAP_TIMER": "0",
            "FR13_DFWD_SPLIT_NEEDLE": "0",
            "FR13_REPLAY_MULTISTREAM": "0",
            "FR13_BRANCH_ACCEPT_DIAG": "0",
            "FR13_FORCE_SPINE_COMMIT": "0",
            "FR13_FIX1_SELFCHECK": "0",
            "FR13_COMMIT_ARGMAX_GATE": "0",
            "FR13_FORK_MARGIN_DUMP": "0",
            "FR13_CHASE_DIAG": "0",
            "FR13_REPLAY_BOUNDARY_LOG": "0",
            "FR13_GDN_SUBOP_MAB": "0",
            "FR13_CONV_SUBOP_MAB": "0",
            "FR13_FA2_MAB": "0",
            "FR13_REPLAY_DURABLE_AB": "0",
            "FR13_TREE_POSREAD_PROBE": "0",
            "FR13_LEAK_PROBE": "0",
            "FR13_SERVE_LOG": "0",
            "FR13_TORCH_DET_WARN": "0",
            "FR13_TCF_DIAG_OVERRIDE": "0",
            "FR13_TCF_SELFCHECK": "0",
            "FR13_PARENT_GATHER_SELFCHECK": "0",
            "FR13_TORCHPROF": "0",
            "FR13_TORCH_PROF": "",
            "FR13_DVK_DRAFTID_DUMP": "",
            "LUMO_NSYS_WRAP_VLLM": "0",
            "LUMO_FA_ACTIVATION_REPLAY_BATCH4_DIAG": "0",
            "LUMO_FA_REPLAY_COMMIT_BATCH4_RUNNER_DIAG": "0",
            "LUMO_IR_DIAGNOSTIC_UNISOLATED": "0",
            "LUMO_IR_ALLOW_UNVERIFIED_SPINES2_MEASUREMENT": "0",
            "FR10_TREE_GDN_CAPTURE_PAYLOAD": "",
            "FR10_TREE_GDN_COMMIT_HANDOFF_LOG": "",
            "FR10_TREE_GDN_SRC_NATIVE_PAYLOAD": "",
            "FR10_ROOT_HIDDEN_CAPTURE": "",
            "FR10_ROOT_LOGIT_CAPTURE": "",
            "FR10_LAYER_HIDDEN_CAPTURE": "",
            "FR12_FULL_ATTN_CAPTURE": "",
            "FR12_SUBKERNEL_CAPTURE": "",
            "FR13_TREE_ATTN_OP_CAPTURE": "",
            "FR13_FLASH_ATTN_OP_CAPTURE": "",
            "FR13_PREPROCESS_INPUT_CAPTURE": "",
            "FR13_PREFILL_GDN_CAPTURE": "",
            "FR13_DECODE_GDN_CAPTURE": "",
            "FR10_SPINE_LOGIT_CAPTURE": "",
            "FR13_FINAL_LOGIT_CAPTURE": "",
            "FR13_HIDDEN_SUBSTITUTE": "",
            "LUMO_MTP_DRAFT_TRACE_FILE": "",
            "LUMO_TREE_SAMPLER_DEBUG_LOG": "",
            "LUMO_TREE_PATH_LCP_LOG": "",
        }
    )
    return required_env


def validate_fixed32_pretask_zero_traffic(
    arm_dir: Path,
    *,
    mode: str,
) -> dict[str, Any]:
    marker_path = arm_dir / "fixed32_pretask_zero_traffic.json"
    marker = exact_json(marker_path, label=str(marker_path))
    exact_keys(
        marker,
        {
            "schema",
            "mode",
            "no_positive_probe",
            "generation_probe_commands_executed",
            "metrics",
            "work_census",
            "ready_ack",
        },
        str(marker_path),
    )
    if (
        marker["schema"] != "fr13-fixed32-pretask-zero-traffic-v1"
        or marker["mode"] != mode
        or marker["no_positive_probe"] is not True
        or marker["generation_probe_commands_executed"] != 0
        or isinstance(marker["generation_probe_commands_executed"], bool)
    ):
        raise GateError(f"{marker_path}: fixed32 pretask traffic claim is invalid")

    metrics_path = arm_dir / "metrics_before_swe.txt"
    metrics = marker["metrics"]
    if not isinstance(metrics, dict):
        raise GateError(f"{marker_path}: metrics identity is missing")
    exact_keys(
        metrics,
        {"path", "sha256", "spec_drafts", "spec_tokens"},
        f"{marker_path}:metrics",
    )
    _metrics_raw, metrics_text = strict_utf8_artifact(
        metrics_path,
        label=str(metrics_path),
    )
    metric_values, _metric_labels = pretask_metric_snapshot_text(
        metrics_text,
        label=str(metrics_path),
    )
    if (
        metrics["path"] != str(metrics_path.resolve())
        or metrics["sha256"] != sha256_file(metrics_path)
        or metrics["spec_drafts"] != 0
        or metrics["spec_tokens"] != 0
        or isinstance(metrics["spec_drafts"], bool)
        or isinstance(metrics["spec_tokens"], bool)
        or integral(metric_values["spec_drafts"], f"{metrics_path}:spec drafts") != 0
        or integral(metric_values["spec_tokens"], f"{metrics_path}:spec tokens") != 0
    ):
        raise GateError(f"{marker_path}: pretask decode metrics are not exact zero")

    census_path = arm_dir / "logs" / "fr13_fixed32_work_census.jsonl"
    census = marker["work_census"]
    if not isinstance(census, dict):
        raise GateError(f"{marker_path}: work-census baseline is missing")
    exact_keys(
        census,
        {"path", "exists", "bytes", "sha256"},
        f"{marker_path}:work_census",
    )
    empty_sha256 = hashlib.sha256(b"").hexdigest()
    if (
        census["path"] != str(census_path.resolve())
        or not isinstance(census["exists"], bool)
        or census["bytes"] != 0
        or isinstance(census["bytes"], bool)
        or census["sha256"] != empty_sha256
    ):
        raise GateError(f"{marker_path}: pretask work census was not empty")

    ready_path = arm_dir / "fixed32_ready_ack.json"
    ready = marker["ready_ack"]
    if not isinstance(ready, dict):
        raise GateError(f"{marker_path}: ready-ack identity is missing")
    exact_keys(
        ready,
        {"path", "sha256", "generation"},
        f"{marker_path}:ready_ack",
    )
    if (
        ready["path"] != str(ready_path.resolve())
        or ready["sha256"] != sha256_file(ready_path)
        or ready["generation"] != 0
        or isinstance(ready["generation"], bool)
    ):
        raise GateError(f"{marker_path}: zero-generation ready ack is not bound")

    forbidden = (
        "warmup_probe.json",
        "warmup_request_metrics.jsonl",
        "warmup_probe_stdout.log",
        "metrics_before_warmup.txt",
        "metrics_after_warmup.txt",
        "docker_after_warmup.log",
        "reset_prefix_cache.txt",
    )
    present = [name for name in forbidden if (arm_dir / name).exists()]
    if present:
        raise GateError(
            f"{arm_dir}: fixed32 forbidden pretask probe artifacts exist: {present}"
        )
    return {
        "path": str(marker_path),
        "sha256": sha256_file(marker_path),
        "metrics": dict(metrics),
        "work_census": dict(census),
        "ready_ack": dict(ready),
        "forbidden_probe_artifacts_absent": True,
    }


def validate_runtime_needles(
    arm_dir: Path, *, mode: str, expected_tokens: int
) -> dict[str, Any]:
    env_path = arm_dir / "container_env.txt"
    env_lines = read_text(env_path).splitlines()
    try:
        mode_spec = FIXED32_MODE_SPECS[mode]
    except KeyError as error:
        raise GateError(f"{arm_dir}: unsupported fixed-32 mode {mode!r}") from error
    required_env = fixed32_required_env(arm_dir, mode=mode)
    for key, expected in required_env.items():
        values = [
            line.removeprefix(f"{key}=")
            for line in env_lines
            if line.startswith(f"{key}=")
        ]
        if values != [expected]:
            raise GateError(
                f"{env_path}: expected exactly {key}={expected}, got {values}"
            )
    process_path = arm_dir / "fixed32_process_identity.json"
    process_identity = exact_json(process_path, label=str(process_path))
    pid1_entries = (process_identity.get("pid1") or {}).get("environ")
    if not isinstance(pid1_entries, list):
        raise GateError(f"{process_path}: PID1 environment is missing")
    pid1_env: dict[str, str] = {}
    for entry in pid1_entries:
        if not isinstance(entry, str) or "=" not in entry:
            raise GateError(f"{process_path}: malformed PID1 environment entry")
        key, value = entry.split("=", 1)
        if key in pid1_env:
            raise GateError(f"{process_path}: duplicate PID1 environment key {key}")
        pid1_env[key] = value
    for key, expected in required_env.items():
        if pid1_env.get(key) != expected:
            raise GateError(
                f"{process_path}: PID1 expected {key}={expected}, "
                f"got {pid1_env.get(key)!r}"
            )
    runtime_attestation_path = (
        arm_dir / "logs" / "fr13_fixed32_runtime_attestation.json"
    )
    try:
        runtime_attestation = validate_runtime_attestation(
            exact_json(
                runtime_attestation_path,
                label=str(runtime_attestation_path),
            )
        )
    except Fixed32ContractError as error:
        raise GateError(
            f"{runtime_attestation_path}: invalid runtime attestation: {error}"
        ) from error
    runtime_fa2 = runtime_attestation["forked_fa2"]
    if (
        runtime_fa2["source"].get("path")
        != str(fixed32_contract.CONTAINER_FA2_SOURCE)
        or runtime_fa2["destination"].get("path")
        != str(CONTAINER_FA2_DESTINATION)
    ):
        raise GateError(
            f"{runtime_attestation_path}: runtime FA2 paths differ from the "
            "fixed32 contract"
        )
    container_identity_path = arm_dir / "fixed32_container_identity.json"
    container_identity = exact_json(
        container_identity_path,
        label=str(container_identity_path),
    )
    expected_container_identity = {
        "schema": "fr13-fixed32-container-identity-v1",
        "name": f"/fr13-bigdenom-{arm_dir.name}",
        "image_id": fixed32_contract.IMAGE_ID,
        "configured_image": fixed32_contract.IMAGE_REFERENCE,
        "platform": fixed32_contract.IMAGE_OS,
        "running": True,
    }
    if container_identity != expected_container_identity:
        raise GateError(
            f"{container_identity_path}: running container identity differs "
            "from the fixed32 external contract"
        )
    eval_preflight = arm_dir / "eval_offload_preflight.txt"
    if read_text(eval_preflight) != "eval offload: alienware reachable\n":
        raise GateError(f"{eval_preflight}: fixed32 evaluator was not offloaded")
    pretask_zero_traffic = validate_fixed32_pretask_zero_traffic(
        arm_dir,
        mode=mode,
    )
    log_path = arm_dir / "docker_full.log"
    log = read_text(log_path)
    needles = (
        FIXED32_PRESEED,
        FIXED32_ENGAGED,
        FIXED32_WORK_ENGAGED,
        mode_spec["topology_needle"],
    )
    for needle in needles:
        if log.count(needle) != 1:
            raise GateError(
                f"{log_path}: expected exactly one current runtime needle {needle!r}"
            )
    other_mode = "hydra27_fixed32" if mode == "tail6_fixed32" else "tail6_fixed32"
    other_needle = FIXED32_MODE_SPECS[other_mode]["topology_needle"]
    if other_needle in log:
        raise GateError(f"{log_path}: emitted both fixed-32 mode topology needles")
    return {
        "container_env": str(env_path),
        "docker_after_canonical_tasks": str(log_path),
        "fixed32_mode": mode,
        "active_drafts": mode_spec["active_drafts"],
        "valid_mask": f"{mode_spec['valid_mask']:#010x}",
        "draft_tokens_per_event": expected_tokens,
        "required_container_env": required_env,
        "pid1_required_env_exact": True,
        "pretask_zero_traffic": pretask_zero_traffic,
        "runtime_attestation": {
            "path": str(runtime_attestation_path),
            "sha256": sha256_file(runtime_attestation_path),
            "canonical_sha256": runtime_attestation[
                "overall_canonical_sha256"
            ],
            "vllm": runtime_attestation["vllm"],
            "forked_fa2": runtime_attestation["forked_fa2"],
            "arctic": {
                "name": runtime_attestation["arctic"].get("name"),
                "version": runtime_attestation["arctic"]["version"],
                "canonical_sha256": runtime_attestation["arctic"][
                    "canonical_sha256"
                ],
                "pinned_source_url": runtime_attestation["arctic"][
                    "pinned_source_url"
                ],
                "pinned_source_sha256": runtime_attestation["arctic"][
                    "pinned_source_sha256"
                ],
                "cache_class_module": runtime_attestation["arctic"][
                    "cache_class_module"
                ],
                "cache_class_qualname": runtime_attestation["arctic"][
                    "cache_class_qualname"
                ],
            },
        },
        "container_identity": {
            "path": str(container_identity_path),
            "sha256": sha256_file(container_identity_path),
            **container_identity,
        },
        "runtime_needles": list(needles),
        "eval_offload_preflight": {
            "path": str(eval_preflight),
            "sha256": sha256_file(eval_preflight),
        },
    }


def canonical_json_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_runtime_manifest(path: Path) -> dict[str, Any]:
    payload = exact_json(path, label=str(path))
    digest = payload.get("overall_canonical_sha256")
    unsigned = {
        key: value
        for key, value in payload.items()
        if key != "overall_canonical_sha256"
    }
    if digest != canonical_json_sha256(unsigned):
        raise GateError(f"{path}: runtime manifest canonical digest mismatch")
    if payload.get("schema") != "fr13-runtime-manifest-v1":
        raise GateError(f"{path}: wrong runtime manifest schema")
    if payload.get("profile") != RUNTIME_MANIFEST_PROFILE:
        raise GateError(f"{path}: wrong runtime manifest profile")
    if payload.get("sequence") != RUNTIME_MANIFEST_SEQUENCE:
        raise GateError(f"{path}: wrong runtime manifest sequence")
    return payload


def load_external_manifest(path: Path) -> dict[str, Any]:
    try:
        return validate_external_manifest(exact_json(path, label=str(path)))
    except Fixed32ContractError as error:
        raise GateError(f"{path}: invalid external manifest: {error}") from error


def validate_external_fingerprint(runroot: Path) -> dict[str, Any]:
    at_launch = runroot / "external_manifest.at_launch.json"
    at_end = runroot / "external_manifest.at_end.json"
    launch = load_external_manifest(at_launch)
    end = load_external_manifest(at_end)
    if launch != end or at_launch.read_bytes() != at_end.read_bytes():
        raise GateError(f"{runroot}: external manifest changed during the campaign")
    return {
        "at_launch": str(at_launch),
        "at_end": str(at_end),
        "byte_equal": True,
        "schema": launch["schema"],
        "image": launch["image"],
        "forked_fa2": launch["forked_fa2"],
        "model": {
            "root": launch["model"]["root"],
            "file_count": launch["model"]["file_count"],
            "canonical_sha256": launch["model"]["canonical_sha256"],
        },
        "arctic_source": launch["arctic_source"],
        "overall_canonical_sha256": launch["overall_canonical_sha256"],
        "file_sha256": sha256_file(at_launch),
    }


def validate_source_fingerprint(repo: Path, runroot: Path) -> dict[str, Any]:
    at_launch = runroot / "runtime_manifest.at_launch.json"
    at_end = runroot / "runtime_manifest.at_end.json"
    launch = load_runtime_manifest(at_launch)
    end = load_runtime_manifest(at_end)
    if launch != end:
        raise GateError(f"{runroot}: runtime manifest changed during the campaign")
    try:
        current = build_runtime_manifest(
            repo,
            profile=RUNTIME_MANIFEST_PROFILE,
            sequence=RUNTIME_MANIFEST_SEQUENCE,
        )
    except RuntimeManifestError as error:
        raise GateError(f"cannot rebuild current runtime manifest: {error}") from error
    if current != end:
        raise GateError(
            f"{runroot}: current runtime closure does not match the campaign manifest"
        )
    summary = launch.get("summary")
    if (
        not isinstance(summary, dict)
        or summary.get("file_count") != 61
        or summary.get("python_package_file_count") != 25
    ):
        raise GateError(f"{at_launch}: runtime closure cardinality is not pinned")
    return {
        "at_launch": str(at_launch),
        "at_end": str(at_end),
        "byte_equal": True,
        "matches_current_runtime_closure": True,
        "profile": RUNTIME_MANIFEST_PROFILE,
        "sequence": RUNTIME_MANIFEST_SEQUENCE,
        "file_count": summary["file_count"],
        "python_package_file_count": summary["python_package_file_count"],
        "overall_canonical_sha256": launch["overall_canonical_sha256"],
    }


def load_windows(
    arm_dir: Path,
    task_dirs: list[Path],
    expected_tokens: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    windows = []
    for task_dir in task_dirs:
        pre_artifact = load_metric_artifact(task_dir / "vllm_metrics_pre.txt")
        post_artifact = load_metric_artifact(task_dir / "vllm_metrics_post.txt")
        pre = pre_artifact["values"]
        post = post_artifact["values"]
        pre_labels = pre_artifact["labels"]
        post_labels = post_artifact["labels"]
        if pre_labels != post_labels:
            raise GateError(f"{task_dir.name}: pre/post required metric labels differ")
        if pre_labels != EXPECTED_METRIC_LABELS:
            raise GateError(
                f"{task_dir.name}: required metric labels do not match the "
                "pinned qwen3.6-27b series"
            )
        for snapshot_name, snapshot in (("pre", pre), ("post", post)):
            snapshot_steps = integral(
                snapshot["wall_steps"],
                f"{task_dir.name}:{snapshot_name} wall steps",
            )
            snapshot_attempts = integral(
                snapshot["wall_attempts"],
                f"{task_dir.name}:{snapshot_name} wall attempts",
            )
            snapshot_rejected = integral(
                snapshot["wall_rejected"],
                f"{task_dir.name}:{snapshot_name} wall rejected",
            )
            if snapshot_attempts != snapshot_steps + snapshot_rejected:
                raise GateError(
                    f"{task_dir.name}: {snapshot_name} wall attempts != retained "
                    f"+ rejected: {snapshot_attempts} != {snapshot_steps} + "
                    f"{snapshot_rejected}"
                )
        delta: dict[str, float] = {}
        for key in METRICS:
            value = post[key] - pre[key]
            if value < -1e-9:
                raise GateError(f"{task_dir.name}: counter {key} regressed")
            delta[key] = max(0.0, value)
        for key in set(METRICS) - {"wall_rejected"}:
            if delta[key] <= 0:
                raise GateError(f"{task_dir.name}: non-positive {key} delta")
        wall_steps = integral(delta["wall_steps"], f"{task_dir.name}:wall steps")
        wall_attempts = integral(
            delta["wall_attempts"], f"{task_dir.name}:wall attempts"
        )
        wall_rejected = integral(
            delta["wall_rejected"], f"{task_dir.name}:wall rejected"
        )
        if wall_attempts != wall_steps + wall_rejected:
            raise GateError(
                f"{task_dir.name}: wall attempts != retained + rejected: "
                f"{wall_attempts} != {wall_steps} + {wall_rejected}"
            )
        if wall_rejected != 0:
            raise GateError(
                f"{task_dir.name}: censored wall intervals in task window: "
                f"{wall_rejected}"
            )
        for family in ("fwd", "wall"):
            steps = integral(delta[f"{family}_steps"], f"{task_dir.name}:{family}")
            drafts = integral(
                delta[f"{family}_drafts"], f"{task_dir.name}:{family} drafts"
            )
            if steps < MIN_TASK_COUNTER_STEPS:
                raise GateError(
                    f"{task_dir.name}: {family} exposure below "
                    f"{MIN_TASK_COUNTER_STEPS} retained steps: {steps}"
                )
            if not steps <= drafts <= concurrency * steps:
                raise GateError(
                    f"{task_dir.name}: {family} drafts/step is outside "
                    f"[1, {concurrency}]: drafts={drafts}, steps={steps}"
                )
        expected_draft_tokens = delta["spec_drafts"] * expected_tokens
        if not math.isclose(
            delta["spec_tokens"], expected_draft_tokens, rel_tol=0, abs_tol=1e-6
        ):
            raise GateError(
                f"{task_dir.name}: draft-token ratio is not exactly {expected_tokens}"
            )
        windows.append(
            {
                "task_id": task_dir.name,
                "pre": pre,
                "post": post,
                "delta": delta,
                "metric_labels": pre_labels,
                "metric_artifacts": {
                    "pre": pre_artifact["identity"],
                    "post": post_artifact["identity"],
                },
                "fixed32_metrics": {
                    "pre": pre_artifact["fixed32"],
                    "post": post_artifact["fixed32"],
                },
                "fwd_span": (
                    integral(pre["fwd_steps"], f"{task_dir.name}:fwd pre"),
                    integral(post["fwd_steps"], f"{task_dir.name}:fwd post"),
                ),
                "wall_span": (
                    integral(pre["wall_steps"], f"{task_dir.name}:wall pre"),
                    integral(post["wall_steps"], f"{task_dir.name}:wall post"),
                ),
            }
        )
    windows = sorted(windows, key=lambda item: item["task_id"])
    first_labels = windows[0]["metric_labels"]
    if any(window["metric_labels"] != first_labels for window in windows[1:]):
        raise GateError(f"{arm_dir}: required metric labels differ across tasks")
    return windows


def validate_flush_chain(
    arm_dir: Path,
    task_dirs: list[Path],
    windows: list[dict[str, Any]],
    *,
    mode: str,
    producer_pid: int,
    complete_steps: int,
    server_capacity: int,
    dataset_record_digests: dict[str, str],
) -> dict[str, Any]:
    pid_path = arm_dir / "logs" / "fr13_fixed32_engine_pid"
    if read_text(pid_path) != f"{producer_pid}\n":
        raise GateError(f"{pid_path}: EngineCore PID file is not exact")
    mode_path = arm_dir / "logs" / "fr13_fixed32_mode.flag"
    if read_text(mode_path) != f"{mode}\n":
        raise GateError(f"{mode_path}: fixed32 mode sidecar is not exact")

    ready_path = arm_dir / "fixed32_ready_ack.json"
    ready = validate_fixed32_ack(
        exact_json(ready_path, label=str(ready_path)),
        label=str(ready_path),
        mode=mode,
        producer_pid=producer_pid,
    )
    if (
        ready["generation"] != 0
        or ready["nonce"] != FLUSH_READY_NONCE
        or ready["action"] != "ready"
        or ready["counters"]
        != {
            "pure_decode_forward_steps": 0,
            "complete_work_census_events": 0,
            "work_census_first_forward_step": None,
            "work_census_last_forward_step": None,
            "sfwd_pending": 0,
            "dfwd_pending": 0,
            "cfwd_pending": 0,
        }
    ):
        raise GateError(f"{ready_path}: generation-zero ready ack is not pristine")

    windows_by_task = {window["task_id"]: window for window in windows}
    task_acks: list[dict[str, Any]] = []
    task_reports: dict[str, Any] = {}
    for task_dir in task_dirs:
        boundary_path = task_dir / "fixed32_task_boundary.json"
        boundary = exact_json(boundary_path, label=str(boundary_path))
        exact_keys(
            boundary,
            {
                "schema",
                "instance_id",
                "mode",
                "producer_pid",
                "pre",
                "post",
                "pre_runtime_snapshot",
                "post_runtime_snapshot",
                "forward_step_interval",
            },
            str(boundary_path),
        )
        if (
            boundary["schema"] != FIXED32_BOUNDARY_SCHEMA
            or boundary["instance_id"] != task_dir.name
            or boundary["mode"] != mode
            or boundary["producer_pid"] != producer_pid
        ):
            raise GateError(f"{boundary_path}: task boundary identity mismatch")
        pre = validate_fixed32_ack(
            boundary["pre"],
            label=f"{boundary_path}:pre",
            mode=mode,
            producer_pid=producer_pid,
        )
        post = validate_fixed32_ack(
            boundary["post"],
            label=f"{boundary_path}:post",
            mode=mode,
            producer_pid=producer_pid,
        )
        if pre["action"] != "snapshot" or post["action"] != "snapshot":
            raise GateError(f"{boundary_path}: task boundaries must be snapshot acks")
        if pre["generation"] >= post["generation"]:
            raise GateError(f"{boundary_path}: pre generation is not before post")
        start = pre["counters"]["pure_decode_forward_steps"]
        end = post["counters"]["pure_decode_forward_steps"]
        event_delta = (
            post["counters"]["complete_work_census_events"]
            - pre["counters"]["complete_work_census_events"]
        )
        expected_interval = {
            "start_forward_step": start,
            "end_forward_step": end,
            "expected_complete_events": event_delta,
        }
        if boundary["forward_step_interval"] != expected_interval:
            raise GateError(f"{boundary_path}: derived forward-step interval mismatch")
        if end <= start or event_delta != end - start:
            raise GateError(f"{boundary_path}: incomplete task census interval")

        window = windows_by_task[task_dir.name]
        if tuple(window["fwd_span"]) != (start, end):
            raise GateError(
                f"{boundary_path}: flush interval does not match metrics fwd span"
            )
        runtime_reports: dict[str, Any] = {}
        for snapshot, ack in (("pre", pre), ("post", post)):
            metrics_path = task_dir / f"vllm_metrics_{snapshot}.txt"
            fixed32_values = window["fixed32_metrics"][snapshot]
            if (
                fixed32_values["pure_decode_forward_steps"]
                != ack["counters"]["pure_decode_forward_steps"]
            ):
                raise GateError(f"{metrics_path}: fixed32 step metric/ack mismatch")
            if (
                fixed32_values["complete_work_census_events"]
                != ack["counters"]["complete_work_census_events"]
            ):
                raise GateError(f"{metrics_path}: fixed32 census metric/ack mismatch")
            runtime_path = (
                arm_dir
                / "logs"
                / f"fr13_fixed32_boundary_snapshot.{ack['generation']}.json"
            )
            runtime_reports[snapshot] = validate_runtime_boundary_snapshot(
                runtime_path,
                ack=ack,
                server_capacity=server_capacity,
                metrics_path=metrics_path,
                metric_values=window[snapshot],
                reference=boundary[f"{snapshot}_runtime_snapshot"],
                census_path=(
                    arm_dir / "logs" / "fr13_fixed32_work_census.jsonl"
                ),
            )

        metadata_path = task_dir / "runner_metadata.json"
        metadata = exact_json(metadata_path, label=str(metadata_path))
        if metadata.get("fixed32_task_boundary") != boundary:
            raise GateError(f"{metadata_path}: embedded task boundary differs from artifact")
        if (
            metadata.get("fixed32_dataset_record_sha256")
            != dataset_record_digests[task_dir.name]
        ):
            raise GateError(
                f"{metadata_path}: fixed32 dataset record digest mismatch"
            )
        task_acks.extend((pre, post))
        task_reports[task_dir.name] = {
            "path": str(boundary_path),
            "sha256": sha256_file(boundary_path),
            "pre_generation": pre["generation"],
            "post_generation": post["generation"],
            "forward_step_interval": [start, end],
            "runtime_snapshots": runtime_reports,
        }

    result_path = arm_dir / "fixed32_final_flush.json"
    result = exact_json(result_path, label=str(result_path))
    exact_keys(result, {"schema", "ack"}, str(result_path))
    if result["schema"] != FLUSH_RESULT_SCHEMA:
        raise GateError(f"{result_path}: wrong flush client result schema")
    final_ack = validate_fixed32_ack(
        result["ack"],
        label=f"{result_path}:ack",
        mode=mode,
        producer_pid=producer_pid,
    )
    if final_ack["action"] != "final":
        raise GateError(f"{result_path}: terminal ack action is not final")
    final_runtime_path = (
        arm_dir
        / "logs"
        / f"fr13_fixed32_boundary_snapshot.{final_ack['generation']}.json"
    )
    final_runtime = validate_runtime_boundary_snapshot(
        final_runtime_path,
        ack=final_ack,
        server_capacity=server_capacity,
        metrics_path=None,
        metric_values=None,
        reference=None,
        census_path=arm_dir / "logs" / "fr13_fixed32_work_census.jsonl",
    )
    stderr_path = arm_dir / "fixed32_final_flush.stderr"
    if read_text(stderr_path) != "":
        raise GateError(f"{stderr_path}: terminal flush wrote stderr")

    ack_path = arm_dir / "logs" / "fr13_fixed32_flush_ack.json"
    current_ack = validate_fixed32_ack(
        exact_json(ack_path, label=str(ack_path)),
        label=str(ack_path),
        mode=mode,
        producer_pid=producer_pid,
    )
    if current_ack != final_ack:
        raise GateError(f"{ack_path}: current ack differs from terminal result")

    request_path = arm_dir / "logs" / "fr13_fixed32_flush_request.json"
    request = exact_json(request_path, label=str(request_path))
    exact_keys(request, FLUSH_REQUEST_KEYS, str(request_path))
    if (
        request["schema"] != FLUSH_REQUEST_SCHEMA
        or request["mode"] != mode
        or request["producer_pid"] != producer_pid
        or request["action"] != "final"
        or request["generation"] != final_ack["generation"]
        or request["prev_generation"] != final_ack["generation"] - 1
        or request["nonce"] != final_ack["nonce"]
    ):
        raise GateError(f"{request_path}: terminal request/ack binding mismatch")
    logs_dir = arm_dir / "logs"
    temp_residue = []
    for pattern in (
        ".fr13_fixed32_flush_request.json.*.tmp",
        "fr13_fixed32_flush_request.json.tmp.*",
        "fr13_fixed32_flush_ack.json.tmp.*",
        "fr13_fixed32_work_census.jsonl.tmp.*",
        "fr13_fixed32_boundary_snapshot.*.json.tmp.*",
    ):
        temp_residue.extend(logs_dir.glob(pattern))
    temp_residue.extend(
        task_dir / "fixed32_task_boundary.json.tmp"
        for task_dir in task_dirs
        if (task_dir / "fixed32_task_boundary.json.tmp").exists()
    )
    if temp_residue:
        raise GateError(
            f"{arm_dir}: stale atomic-write temporary files: "
            f"{sorted(str(path) for path in temp_residue)}"
        )

    generations = [ready, *task_acks, final_ack]
    ordered = sorted(generations, key=lambda ack: ack["generation"])
    expected_generations = list(range(len(ordered)))
    actual_generations = [ack["generation"] for ack in ordered]
    if actual_generations != expected_generations:
        raise GateError(
            f"{arm_dir}: flush generation chain is not exact: {actual_generations}"
        )
    if [ack["action"] for ack in ordered] != [
        "ready",
        *(["snapshot"] * (2 * len(task_dirs))),
        "final",
    ]:
        raise GateError(f"{arm_dir}: flush action chain is not exact")
    nonces = [ack["nonce"] for ack in ordered]
    if len(nonces) != len(set(nonces)) or nonces[0] != FLUSH_READY_NONCE:
        raise GateError(f"{arm_dir}: flush nonce chain is not unique")
    if any(
        current["counters"]["pure_decode_forward_steps"]
        < previous["counters"]["pure_decode_forward_steps"]
        or current["counters"]["complete_work_census_events"]
        < previous["counters"]["complete_work_census_events"]
        for previous, current in zip(ordered, ordered[1:], strict=False)
    ):
        raise GateError(f"{arm_dir}: flush counters regress across generations")

    final_counters = final_ack["counters"]
    if (
        final_counters["pure_decode_forward_steps"] != complete_steps
        or final_counters["complete_work_census_events"] != complete_steps
        or final_counters["work_census_first_forward_step"] != 0
        or final_counters["work_census_last_forward_step"] != complete_steps - 1
    ):
        raise GateError(f"{arm_dir}: final flush counters do not close complete stream")
    expected_runtime_paths = {
        arm_dir
        / "logs"
        / f"fr13_fixed32_boundary_snapshot.{ack['generation']}.json"
        for ack in [*task_acks, final_ack]
    }
    actual_runtime_paths = set(
        (arm_dir / "logs").glob("fr13_fixed32_boundary_snapshot.*.json")
    )
    if actual_runtime_paths != expected_runtime_paths:
        raise GateError(
            f"{arm_dir}: runtime boundary snapshot generation set is not exact"
        )
    return {
        "ready": {"path": str(ready_path), "sha256": sha256_file(ready_path)},
        "tasks": task_reports,
        "final": {
            "result_path": str(result_path),
            "result_sha256": sha256_file(result_path),
            "request_path": str(request_path),
            "request_sha256": sha256_file(request_path),
            "ack_path": str(ack_path),
            "ack_sha256": sha256_file(ack_path),
            "generation": final_ack["generation"],
            "counters": final_counters,
            "runtime_snapshot": final_runtime,
        },
        "generation_chain": actual_generations,
        "all_pending_counts_zero": True,
        "task_intervals_bound_to_metrics": True,
    }


def unique_sidecar(
    sidecar_dir: Path,
    arm: str,
    concurrency: int,
    expected_pid: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    if not sidecar_dir.is_dir():
        raise GateError(f"missing per-step sidecar directory: {sidecar_dir}")
    prefix = f"{arm}.json.samples."
    paths = sorted(
        path
        for path in sidecar_dir.iterdir()
        if path.is_file() and path.name.startswith(prefix)
    )
    if len(paths) != 1:
        raise GateError(
            f"{sidecar_dir}: expected one per-step sidecar for {arm}, found {paths}"
        )
    path = paths[0]
    try:
        payload = json.loads(read_text(path))
    except json.JSONDecodeError as error:
        raise GateError(f"{path}: invalid JSON: {error}") from error
    if payload.get("schema") != "fr13.sfwd_per_step_samples.v1":
        raise GateError(f"{path}: wrong sidecar schema")
    if payload.get("final") is not True:
        raise GateError(f"{path}: per-step sidecar lacks an explicit final flush")
    try:
        suffix_pid = int(path.name.removeprefix(prefix))
    except ValueError as error:
        raise GateError(f"{path}: sidecar filename has no integral PID") from error
    if payload.get("pid") != suffix_pid:
        raise GateError(f"{path}: sidecar payload PID does not match filename")
    if suffix_pid != expected_pid:
        raise GateError(
            f"{path}: sidecar PID {suffix_pid} does not match recorded "
            f"VLLM::EngineCore PID {expected_pid}"
        )
    if payload.get("samples_capped") is not False:
        raise GateError(f"{path}: sidecar samples are capped or cap state is absent")
    fwd_fields = (
        "fwd_drafts",
        "fwd_ms",
        "fwd_cg",
        "fwd_host_ms",
        "fwd_exec_ms",
        "fwd_cpu_tail_ms",
    )
    wall_fields = ("wall_drafts", "wall_ms")
    lengths = {key: len(payload.get(key, [])) for key in (*fwd_fields, *wall_fields)}
    if len({lengths[key] for key in fwd_fields}) != 1:
        raise GateError(f"{path}: forward sidecar array lengths differ: {lengths}")
    if len({lengths[key] for key in wall_fields}) != 1:
        raise GateError(f"{path}: wall sidecar array lengths differ: {lengths}")
    arrays = {
        "fwd_drafts": np.asarray(payload["fwd_drafts"], dtype=np.float64),
        "fwd_ms": np.asarray(payload["fwd_ms"], dtype=np.float64),
        "fwd_full": np.asarray(
            [value == "FULL" for value in payload["fwd_cg"]], dtype=np.bool_
        ),
        "wall_drafts": np.asarray(payload["wall_drafts"], dtype=np.float64),
        "wall_ms": np.asarray(payload["wall_ms"], dtype=np.float64),
    }
    for key in ("fwd_drafts", "fwd_ms", "wall_drafts", "wall_ms"):
        values = arrays[key]
        if not np.all(np.isfinite(values)):
            raise GateError(f"{path}: {key} contains non-finite values")
    if np.any(arrays["fwd_ms"] <= 0) or np.any(arrays["wall_ms"] <= 0):
        raise GateError(f"{path}: timing arrays contain non-positive values")
    for key in ("fwd_drafts", "wall_drafts"):
        values = arrays[key]
        if (
            np.any(values < 1)
            or np.any(values > concurrency)
            or np.any(values != np.rint(values))
        ):
            raise GateError(
                f"{path}: {key} is inconsistent with inferred concurrency {concurrency}"
            )
    return (
        payload,
        arrays,
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "pid": suffix_pid,
            "pid_bound_to_runlog": True,
            "final": True,
            "samples_capped": False,
            "array_lengths": lengths,
        },
    )


def unique_main_sidecar(
    sidecar_dir: Path,
    arm: str,
    expected_pid: int,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    prefix = f"{arm}.json."
    paths = sorted(
        path
        for path in sidecar_dir.iterdir()
        if path.is_file()
        and path.name.startswith(prefix)
        and not path.name.startswith(f"{arm}.json.samples.")
    )
    if len(paths) != 1:
        raise GateError(
            f"{sidecar_dir}: expected one main timer sidecar for {arm}, found {paths}"
        )
    path = paths[0]
    try:
        payload = json.loads(read_text(path))
    except json.JSONDecodeError as error:
        raise GateError(f"{path}: invalid JSON: {error}") from error
    if payload.get("schema") != "fr13.sfwd_gpu_timer.v1":
        raise GateError(f"{path}: wrong main sidecar schema")
    try:
        suffix_pid = int(path.name.removeprefix(prefix))
    except ValueError as error:
        raise GateError(f"{path}: main sidecar filename has no integral PID") from error
    if payload.get("pid") != suffix_pid or suffix_pid != expected_pid:
        raise GateError(
            f"{path}: main sidecar PID is not bound to the recorded EngineCore"
        )
    if payload.get("final") is not True:
        raise GateError(f"{path}: main sidecar lacks an explicit final flush")
    wall_cap = float(payload.get("wall_cap_s", float("nan")))
    if not math.isclose(wall_cap, 1.5, rel_tol=0, abs_tol=0):
        raise GateError(f"{path}: main sidecar wall cap is not exactly 1.5 seconds")
    counters = {
        "fwd_steps": integral(
            float(payload.get("n_pure_decode_steps_timed", float("nan"))),
            f"{path}:fwd steps",
        ),
        "fwd_drafts": integral(
            float(payload.get("n_drafts_in_timed_steps", float("nan"))),
            f"{path}:fwd drafts",
        ),
        "wall_steps": integral(
            float(payload.get("n_wall_steps", float("nan"))),
            f"{path}:wall steps",
        ),
        "wall_drafts": integral(
            float(payload.get("n_drafts_in_wall_steps", float("nan"))),
            f"{path}:wall drafts",
        ),
        "wall_rejected": integral(
            float(payload.get("n_wall_rejected", float("nan"))),
            f"{path}:wall rejected",
        ),
    }
    if counters["wall_rejected"] < 0:
        raise GateError(f"{path}: negative wall rejected counter")
    expected_lengths = {
        "fwd": counters["fwd_steps"],
        "wall": counters["wall_steps"],
    }
    actual_lengths = {
        "fwd": len(arrays["fwd_ms"]),
        "wall": len(arrays["wall_ms"]),
    }
    if actual_lengths != expected_lengths:
        raise GateError(
            f"{path}: final sample lengths do not match main counters: "
            f"{actual_lengths} != {expected_lengths}"
        )
    reconciliation = {
        "forward": reconcile_counter_interval(
            arrays,
            "fwd",
            (0, counters["fwd_steps"]),
            float(payload.get("decode_forward_gpu_seconds", float("nan"))),
            counters["fwd_drafts"],
        ),
        "wall": reconcile_counter_interval(
            arrays,
            "wall",
            (0, counters["wall_steps"]),
            float(payload.get("decode_step_wall_seconds", float("nan"))),
            counters["wall_drafts"],
        ),
    }
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "pid": suffix_pid,
        "pid_bound_to_runlog": True,
        "final": True,
        "wall_cap_s": wall_cap,
        "counters": counters,
        "wall_attempts": counters["wall_steps"] + counters["wall_rejected"],
        "full_array_reconciliation": reconciliation,
    }


def merged_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if any(end <= start for start, end in spans):
        raise GateError(f"empty or reversed counter interval: {spans}")
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def selected_counter_indices(
    spans: list[tuple[int, int]],
    *,
    available_steps: int,
    label: str,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Return the ordered, de-duplicated counter-index union for task spans."""
    union = merged_spans(spans)
    for start, end in union:
        if start < 0 or end > available_steps:
            raise GateError(
                f"{label}: task counter union {(start, end)} is outside "
                f"available sidecar steps [0, {available_steps})"
            )
    indices = [index for start, end in union for index in range(start, end)]
    if not indices:
        raise GateError(f"{label}: task counter union selected no steps")
    return union, indices


def assert_nonoverlap(spans: list[tuple[int, int]], label: str) -> None:
    ordered = sorted(spans)
    for left, right in zip(ordered, ordered[1:], strict=False):
        if right[0] < left[1]:
            raise GateError(
                f"{label}: B=1 task counter intervals overlap: {left}, {right}"
            )


def select_span(values: np.ndarray, span: tuple[int, int]) -> np.ndarray:
    start, end = span
    available_end = min(end, len(values))
    if available_end <= start:
        return values[:0]
    return values[start:available_end]


def coverage_for_span(length: int, span: tuple[int, int]) -> dict[str, Any]:
    start, end = span
    expected = end - start
    selected = max(0, min(end, length) - start)
    return {
        "counter_interval": [start, end],
        "expected_steps": expected,
        "selected_steps": selected,
        "fraction": selected / expected,
    }


def reconcile_counter_interval(
    arrays: dict[str, np.ndarray],
    family: str,
    span: tuple[int, int],
    counter_seconds: float,
    counter_drafts: float,
) -> dict[str, Any]:
    start, end = span
    expected_steps = end - start
    ms_values = select_span(arrays[f"{family}_ms"], span)
    draft_values = select_span(arrays[f"{family}_drafts"], span)
    if len(ms_values) != expected_steps or len(draft_values) != expected_steps:
        raise GateError(
            f"{family}: sidecar does not completely cover counter interval "
            f"{span}; selected ms/drafts={len(ms_values)}/{len(draft_values)}"
        )
    sidecar_drafts = integral(
        float(draft_values.sum()), f"{family}: sidecar interval drafts"
    )
    expected_drafts = integral(counter_drafts, f"{family}: counter interval drafts")
    if sidecar_drafts != expected_drafts:
        raise GateError(
            f"{family}: sidecar/counter draft mismatch over {span}: "
            f"{sidecar_drafts} != {expected_drafts}"
        )
    sidecar_ms = math.fsum(float(value) for value in ms_values)
    counter_ms = 1000.0 * counter_seconds
    # Samples are serialized after rounding each observation to 4 decimal ms.
    rounding_bound_ms = expected_steps * 0.000051 + 1e-6
    error_ms = sidecar_ms - counter_ms
    if abs(error_ms) > rounding_bound_ms:
        raise GateError(
            f"{family}: sidecar/counter timing mismatch over {span}: "
            f"sidecar={sidecar_ms} ms counter={counter_ms} ms "
            f"error={error_ms} ms bound={rounding_bound_ms} ms"
        )
    return {
        "counter_interval": [start, end],
        "steps": expected_steps,
        "counter_drafts": expected_drafts,
        "sidecar_drafts": sidecar_drafts,
        "counter_ms": counter_ms,
        "sidecar_ms": sidecar_ms,
        "timing_error_ms": error_ms,
        "rounding_bound_ms": rounding_bound_ms,
        "exact_drafts_and_steps": True,
        "timing_within_per_sample_rounding_bound": True,
    }


def legacy_slo(rows_per_step: float) -> tuple[float, float]:
    reference = max(
        WEIGHT_STREAM_LOWER_BOUND_MS,
        COMPUTE_MS_PER_ROW * rows_per_step,
    )
    return reference, SLO_MULTIPLIER * reference


def cluster_summary(values: list[float]) -> dict[str, Any]:
    count = len(values)
    df = count - 1
    critical = T95_ONE_SIDED.get(df)
    if critical is None:
        raise GateError(f"no pinned one-sided t critical for df={df}")
    point = statistics.fmean(values)
    sample_sd = statistics.stdev(values)
    standard_error = sample_sd / math.sqrt(count)
    return {
        "cluster_count": count,
        "df": df,
        "point_estimate": point,
        "sample_sd_across_task_means": sample_sd,
        "standard_error": standard_error,
        "t_0_95_one_sided": critical,
        "u95": point + critical * standard_error,
    }


def b1_arm_statistics(
    windows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    expected_tokens: int,
) -> dict[str, Any]:
    assert_nonoverlap([window["fwd_span"] for window in windows], "forward")
    assert_nonoverlap([window["wall_span"] for window in windows], "wall")
    task_rows = []
    coverage_rows = []
    selected_full_graph = []
    for window in windows:
        delta = window["delta"]
        fwd_steps = delta["fwd_steps"]
        wall_steps = delta["wall_steps"]
        verify_ms = 1000.0 * delta["fwd_s"] / fwd_steps
        wall_ms = 1000.0 * delta["wall_s"] / wall_steps
        rows = (delta["wall_drafts"] / wall_steps) * (expected_tokens + 1)
        slo_reference, slo_limit = legacy_slo(rows)
        fwd_coverage = coverage_for_span(len(arrays["fwd_ms"]), window["fwd_span"])
        wall_coverage = coverage_for_span(len(arrays["wall_ms"]), window["wall_span"])
        if (
            fwd_coverage["fraction"] < REQUIRED_COVERAGE
            or wall_coverage["fraction"] < REQUIRED_COVERAGE
        ):
            raise GateError(
                f"{window['task_id']}: per-step sidecar coverage is not 100%"
            )
        reconciliation = {
            "forward": reconcile_counter_interval(
                arrays,
                "fwd",
                window["fwd_span"],
                delta["fwd_s"],
                delta["fwd_drafts"],
            ),
            "wall": reconcile_counter_interval(
                arrays,
                "wall",
                window["wall_span"],
                delta["wall_s"],
                delta["wall_drafts"],
            ),
        }
        fwd_sample = select_span(arrays["fwd_ms"], window["fwd_span"])
        full_sample = select_span(arrays["fwd_full"], window["fwd_span"])
        wall_sample = select_span(arrays["wall_ms"], window["wall_span"])
        selected_full_graph.append(full_sample)
        task_rows.append(
            {
                "task_id": window["task_id"],
                "verify_ms_per_step": verify_ms,
                "wall_ms_per_step": wall_ms,
                "rows_per_step": rows,
                "legacy_slo_reference_ms": slo_reference,
                "legacy_slo_limit_ms": slo_limit,
                "legacy_slo_excess_ms": wall_ms - slo_limit,
                "selected_sample_verify_ms_per_step": float(fwd_sample.mean()),
                "selected_sample_wall_ms_per_step": float(wall_sample.mean()),
            }
        )
        coverage_rows.append(
            {
                "task_id": window["task_id"],
                "forward": fwd_coverage,
                "wall": wall_coverage,
                "counter_reconciliation": reconciliation,
            }
        )
    full_graph_fraction = float(np.concatenate(selected_full_graph).mean())
    if full_graph_fraction < MIN_FULL_GRAPH_FRACTION:
        raise GateError("B=1 selected FULL graph fraction is below 99%")
    equal_task = {
        "wall_ms_per_step": cluster_summary(
            [row["wall_ms_per_step"] for row in task_rows]
        ),
        "verify_ms_per_step": cluster_summary(
            [row["verify_ms_per_step"] for row in task_rows]
        ),
        "rows_per_step": cluster_summary([row["rows_per_step"] for row in task_rows]),
        "legacy_slo_excess_ms": cluster_summary(
            [row["legacy_slo_excess_ms"] for row in task_rows]
        ),
    }
    total_fwd_steps = sum(row["delta"]["fwd_steps"] for row in windows)
    total_wall_steps = sum(row["delta"]["wall_steps"] for row in windows)
    weighted_verify = (
        1000.0 * sum(row["delta"]["fwd_s"] for row in windows) / total_fwd_steps
    )
    weighted_wall = (
        1000.0 * sum(row["delta"]["wall_s"] for row in windows) / total_wall_steps
    )
    weighted_rows = (
        sum(row["delta"]["wall_drafts"] for row in windows)
        / total_wall_steps
        * (expected_tokens + 1)
    )
    weighted_reference, weighted_limit = legacy_slo(weighted_rows)
    return {
        "inference_scope": (
            "equal-weight SWE task clusters; the t interval treats each whole "
            "task as one cluster and makes no within-task independence assumption"
        ),
        "bracket_mode": "nonoverlapping_task_clusters",
        "task_cluster_equal_weight": equal_task,
        "step_weighted_counter_point": {
            "verify_ms_per_step": weighted_verify,
            "wall_ms_per_step": weighted_wall,
            "rows_per_step": weighted_rows,
            "legacy_slo_reference_ms": weighted_reference,
            "legacy_slo_limit_ms": weighted_limit,
            "legacy_slo_excess_ms": weighted_wall - weighted_limit,
        },
        "per_task": task_rows,
        "sidecar_coverage_by_task": coverage_rows,
        "selected_full_graph_fraction": full_graph_fraction,
        "gate": {
            "statistic": "equal_task_legacy_slo_excess_ms_u95_le_0",
            "pass": equal_task["legacy_slo_excess_ms"]["u95"] <= 0,
        },
    }


def outer_counter_point(
    windows: list[dict[str, Any]],
    family: str,
    expected_tokens: int,
    concurrency: int,
) -> dict[str, Any]:
    step_key = f"{family}_steps"
    seconds_key = f"{family}_s"
    drafts_key = f"{family}_drafts"
    span_key = f"{family}_span"
    spans = [window[span_key] for window in windows]
    union = merged_spans(spans)
    if len(union) != 1:
        raise GateError(
            f"B=4 {family} task brackets do not form one contiguous union: {union}"
        )
    start, end = union[0]
    start_window = min(windows, key=lambda row: (row["pre"][step_key], row["task_id"]))
    end_window = max(windows, key=lambda row: (row["post"][step_key], row["task_id"]))
    if (
        integral(start_window["pre"][step_key], f"{family}:union pre") != start
        or integral(end_window["post"][step_key], f"{family}:union post") != end
    ):
        raise GateError(f"{family}: union endpoint snapshot mismatch")
    steps = end - start
    seconds = end_window["post"][seconds_key] - start_window["pre"][seconds_key]
    drafts = end_window["post"][drafts_key] - start_window["pre"][drafts_key]
    if seconds <= 0 or drafts <= 0:
        raise GateError(f"{family}: non-positive outer counter delta")
    if not steps <= drafts <= concurrency * steps:
        raise GateError(
            f"{family}: outer drafts/step is outside [1, {concurrency}]: "
            f"drafts={drafts}, steps={steps}"
        )
    point = {
        "counter_interval": [start, end],
        "steps": steps,
        "seconds": seconds,
        "drafts": drafts,
        "ms_per_step": 1000.0 * seconds / steps,
        "drafts_per_step": drafts / steps,
        "rows_per_step": drafts / steps * (expected_tokens + 1),
    }
    return point


def moving_block_means(
    arrays: tuple[np.ndarray, ...],
    reps: int,
    block: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    lengths = {len(values) for values in arrays}
    if len(lengths) != 1:
        raise GateError("moving-block arrays have different lengths")
    sample_count = lengths.pop()
    if sample_count < block:
        raise GateError(
            f"need at least {block} samples for requested block sensitivity; "
            f"got {sample_count}"
        )
    blocks_per_rep = math.ceil(sample_count / block)
    offsets = np.arange(block, dtype=np.int64)
    rng = np.random.default_rng(seed)
    outputs = tuple(np.empty(reps, dtype=np.float64) for _ in arrays)
    for lower in range(0, reps, 32):
        upper = min(reps, lower + 32)
        starts = rng.integers(
            0,
            sample_count - block + 1,
            size=(upper - lower, blocks_per_rep),
        )
        indices = (starts[:, :, None] + offsets).reshape(upper - lower, -1)
        indices = indices[:, :sample_count]
        for values, output in zip(arrays, outputs, strict=True):
            output[lower:upper] = values[indices].mean(axis=1)
    return outputs


def b4_arm_statistics(
    windows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    expected_tokens: int,
    concurrency: int,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    fwd_point = outer_counter_point(windows, "fwd", expected_tokens, concurrency)
    wall_point = outer_counter_point(windows, "wall", expected_tokens, concurrency)
    fwd_span = tuple(fwd_point["counter_interval"])
    wall_span = tuple(wall_point["counter_interval"])
    fwd_coverage = coverage_for_span(len(arrays["fwd_ms"]), fwd_span)
    wall_coverage = coverage_for_span(len(arrays["wall_ms"]), wall_span)
    if (
        fwd_coverage["fraction"] < REQUIRED_COVERAGE
        or wall_coverage["fraction"] < REQUIRED_COVERAGE
    ):
        raise GateError("B=4 union sidecar coverage is not 100%")
    reconciliation = {
        "forward": reconcile_counter_interval(
            arrays,
            "fwd",
            fwd_span,
            fwd_point["seconds"],
            fwd_point["drafts"],
        ),
        "wall": reconcile_counter_interval(
            arrays,
            "wall",
            wall_span,
            wall_point["seconds"],
            wall_point["drafts"],
        ),
    }
    fwd_ms = select_span(arrays["fwd_ms"], fwd_span)
    fwd_full = select_span(arrays["fwd_full"], fwd_span)
    fwd_drafts = select_span(arrays["fwd_drafts"], fwd_span)
    wall_ms = select_span(arrays["wall_ms"], wall_span)
    wall_drafts = select_span(arrays["wall_drafts"], wall_span)
    if len(fwd_drafts) != len(wall_drafts) or not np.array_equal(
        fwd_drafts, wall_drafts
    ):
        raise GateError(
            "B=4 canonical forward and wall occupancy sequences are not "
            "elementwise identical"
        )
    wall_rows = wall_drafts * (expected_tokens + 1)
    full_graph_fraction = float(fwd_full.mean())
    if full_graph_fraction < MIN_FULL_GRAPH_FRACTION:
        raise GateError("B=4 selected FULL graph fraction is below 99%")
    slo_reference, slo_limit = legacy_slo(wall_point["rows_per_step"])
    max_possible_compute_reference = COMPUTE_MS_PER_ROW * 4 * (expected_tokens + 1)
    if max_possible_compute_reference >= WEIGHT_STREAM_LOWER_BOUND_MS:
        raise GateError("B=4 campaign violates the pinned weight-bound dominance")
    point_excess = wall_point["ms_per_step"] - slo_limit
    sensitivity = []
    for block in BLOCK_SENSITIVITY:
        fwd_boot = moving_block_means((fwd_ms,), reps, block, seed + 1000 + block)[0]
        wall_boot, rows_boot = moving_block_means(
            (wall_ms, wall_rows), reps, block, seed + 2000 + block
        )
        bootstrap_reference = np.maximum(
            WEIGHT_STREAM_LOWER_BOUND_MS,
            COMPUTE_MS_PER_ROW * rows_boot,
        )
        bootstrap_excess = wall_boot - SLO_MULTIPLIER * bootstrap_reference
        sample_excess = float(wall_ms.mean()) - SLO_MULTIPLIER * max(
            WEIGHT_STREAM_LOWER_BOUND_MS,
            COMPUTE_MS_PER_ROW * float(wall_rows.mean()),
        )
        sensitivity.append(
            {
                "block_steps": block,
                "verify_ms_per_step_u95": (
                    fwd_point["ms_per_step"]
                    + float(np.quantile(fwd_boot - fwd_ms.mean(), 0.95))
                ),
                "wall_ms_per_step_u95": (
                    wall_point["ms_per_step"]
                    + float(np.quantile(wall_boot - wall_ms.mean(), 0.95))
                ),
                "legacy_slo_excess_ms_u95": (
                    point_excess
                    + float(np.quantile(bootstrap_excess - sample_excess, 0.95))
                ),
            }
        )
    worst = {
        "verify_ms_per_step_u95": max(
            row["verify_ms_per_step_u95"] for row in sensitivity
        ),
        "wall_ms_per_step_u95": max(row["wall_ms_per_step_u95"] for row in sensitivity),
        "legacy_slo_excess_ms_u95": max(
            row["legacy_slo_excess_ms_u95"] for row in sensitivity
        ),
    }
    exact_b4_selected = wall_drafts == concurrency
    exact_b4_count = int(np.count_nonzero(exact_b4_selected))
    if exact_b4_count < MIN_B4_EXACT_EVENTS:
        raise GateError(
            "B=4 exact-occupancy wall stratum has insufficient evidence: "
            f"{exact_b4_count} < {MIN_B4_EXACT_EVENTS}"
        )
    exact_b4_wall_ms = wall_ms[exact_b4_selected]
    exact_b4_fwd_ms = fwd_ms[exact_b4_selected]
    exact_b4_rows = concurrency * (expected_tokens + 1)
    exact_b4_reference, exact_b4_limit = legacy_slo(exact_b4_rows)
    exact_b4_wall_point = float(exact_b4_wall_ms.mean())
    exact_b4_fwd_point = float(exact_b4_fwd_ms.mean())
    exact_b4_sensitivity = []
    for block in BLOCK_SENSITIVITY:
        exact_fwd_boot, exact_wall_boot = moving_block_means(
            (exact_b4_fwd_ms, exact_b4_wall_ms),
            reps,
            block,
            seed + 3000 + block,
        )
        exact_b4_sensitivity.append(
            {
                "block_steps": block,
                "verify_ms_per_step_u95": (
                    exact_b4_fwd_point
                    + float(
                        np.quantile(
                            exact_fwd_boot - exact_b4_fwd_point,
                            0.95,
                        )
                    )
                ),
                "wall_ms_per_step_u95": (
                    exact_b4_wall_point
                    + float(
                        np.quantile(
                            exact_wall_boot - exact_b4_wall_point,
                            0.95,
                        )
                    )
                ),
            }
        )
    exact_b4_worst_wall_u95 = max(
        row["wall_ms_per_step_u95"] for row in exact_b4_sensitivity
    )
    exact_b4 = {
        "wall_drafts": concurrency,
        "selected_steps": exact_b4_count,
        "rows_per_step": exact_b4_rows,
        "verify_ms_per_step": exact_b4_fwd_point,
        "wall_ms_per_step": exact_b4_wall_point,
        "legacy_slo_reference_ms": exact_b4_reference,
        "legacy_slo_limit_ms": exact_b4_limit,
        "moving_block_u95_sensitivity": {
            "reps": reps,
            "blocks": exact_b4_sensitivity,
            "worst_wall_ms_per_step_u95": exact_b4_worst_wall_u95,
        },
        "gate": {
            "statistic": "worst_block_exact_b4_wall_ms_per_step_u95_le_slo",
            "pass": exact_b4_worst_wall_u95 <= exact_b4_limit,
        },
    }
    occupancy = []
    for drafts in range(1, 5):
        selected = wall_drafts == drafts
        if not np.any(selected):
            continue
        rows = drafts * (expected_tokens + 1)
        reference, limit = legacy_slo(rows)
        occupancy.append(
            {
                "wall_drafts": drafts,
                "selected_steps": int(np.count_nonzero(selected)),
                "wall_ms_per_step": float(wall_ms[selected].mean()),
                "rows_per_step": rows,
                "legacy_slo_reference_ms": reference,
                "legacy_slo_limit_ms": limit,
                "legacy_slo_excess_ms": (float(wall_ms[selected].mean()) - limit),
            }
        )
    return {
        "inference_scope": (
            "conditional on this one campaign time series; moving-block "
            "sensitivity is explicitly NOT a task-general uncertainty claim"
        ),
        "bracket_mode": (
            "overlap-safe_single_counter_index_union; task-sum is forbidden"
        ),
        "union_counter_point": {
            "verify_ms_per_step": fwd_point["ms_per_step"],
            "wall_ms_per_step": wall_point["ms_per_step"],
            "rows_per_step": wall_point["rows_per_step"],
            "legacy_slo_reference_ms": slo_reference,
            "legacy_slo_limit_ms": slo_limit,
            "legacy_slo_excess_ms": point_excess,
        },
        "union_intervals": {
            "forward": fwd_point["counter_interval"],
            "wall": wall_point["counter_interval"],
        },
        "sidecar_coverage": {
            "forward": fwd_coverage,
            "wall": wall_coverage,
        },
        "sidecar_counter_reconciliation": reconciliation,
        "selected_full_graph_fraction": full_graph_fraction,
        "forward_wall_occupancy_sequence_equal": True,
        "wall_occupancy_strata": occupancy,
        "exact_b4_stratum": exact_b4,
        "moving_block_u95_sensitivity": {
            "reps": reps,
            "centered_on_complete_union_counter_point": True,
            "blocks": sensitivity,
            "worst_across_requested_blocks": worst,
        },
        "gate": {
            "statistic": (
                "union_worst_block_legacy_slo_excess_ms_u95_le_0_and_"
                "exact_b4_worst_block_wall_u95_le_slo"
            ),
            "union_pass": worst["legacy_slo_excess_ms_u95"] <= 0,
            "exact_b4_pass": exact_b4["gate"]["pass"],
            "pass": (
                worst["legacy_slo_excess_ms_u95"] <= 0
                and exact_b4["gate"]["pass"]
            ),
        },
    }


def reduce_arm(
    repo: Path,
    runroot: Path,
    sidecar_dir: Path,
    arm: str,
    *,
    mode: str,
    task_count: int,
    expected_concurrency: int | None,
    reps: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    arm_dir = runroot / arm
    try:
        mode_spec = FIXED32_MODE_SPECS[mode]
    except KeyError as error:
        raise GateError(f"{arm}: unsupported fixed-32 mode {mode!r}") from error
    expected_tokens = PHYSICAL_DRAFTS
    expected_kind = mode
    orchestrator = parse_orchestrator(arm_dir, task_count)
    concurrency = orchestrator["inferred_concurrency"]
    if expected_concurrency is not None and concurrency != expected_concurrency:
        raise GateError(
            f"{arm}: inferred concurrency {concurrency} != "
            f"expected {expected_concurrency}"
        )
    task_dirs = task_directories(arm_dir, task_count)
    real_task_provenance = validate_real_task_provenance(
        arm_dir,
        task_dirs,
        mode=mode,
    )
    runtime = validate_runtime_needles(
        arm_dir, mode=mode, expected_tokens=expected_tokens
    )
    launch = resolve_subset_from_runlog(
        repo,
        runroot,
        arm,
        expected_kind,
        expected_tokens,
        task_count,
        concurrency,
    )
    windows = load_windows(arm_dir, task_dirs, expected_tokens, concurrency)
    _, arrays, sidecar = unique_sidecar(
        sidecar_dir,
        arm,
        concurrency,
        launch["engine_core_pid"],
    )
    main_sidecar = unique_main_sidecar(
        sidecar_dir,
        arm,
        launch["engine_core_pid"],
        arrays,
    )
    flush_chain = validate_flush_chain(
        arm_dir,
        task_dirs,
        windows,
        mode=mode,
        producer_pid=launch["engine_core_pid"],
        complete_steps=len(arrays["fwd_drafts"]),
        server_capacity=concurrency,
        dataset_record_digests=pinned_dataset_record_digests(str(repo)),
    )
    census_intervals, census_indices = selected_counter_indices(
        [tuple(window["fwd_span"]) for window in windows],
        available_steps=len(arrays["fwd_drafts"]),
        label=f"{arm}: work census",
    )
    complete_census_indices = list(range(len(arrays["fwd_drafts"])))
    complete_census_batch_sequence = [
        integral(value, f"{arm}: complete fwd_drafts") for value in arrays["fwd_drafts"]
    ]
    selected_census_batch_sequence = [
        integral(arrays["fwd_drafts"][index], f"{arm}: selected fwd_drafts")
        for index in census_indices
    ]
    complete_occupied_batch_histogram = {
        str(batch_size): complete_census_batch_sequence.count(batch_size)
        for batch_size in range(1, concurrency + 1)
        if batch_size in complete_census_batch_sequence
    }
    selected_occupied_batch_histogram = {
        str(batch_size): selected_census_batch_sequence.count(batch_size)
        for batch_size in range(1, concurrency + 1)
        if batch_size in selected_census_batch_sequence
    }
    work_census_expected = {
        "path": str(arm_dir / "logs" / "fr13_fixed32_work_census.jsonl"),
        "producer_pid": launch["engine_core_pid"],
        "binding": (
            "complete_pure_decode_sfwd_stream_then_posthoc_"
            "canonical_task_forward_counter_union"
        ),
        "complete_stream": {
            "forward_step_indices": complete_census_indices,
            "event_count": len(complete_census_batch_sequence),
            "batch_size_sequence": complete_census_batch_sequence,
            "occupied_batch_histogram": complete_occupied_batch_histogram,
        },
        "canonical_task_selection": {
            "counter_intervals": [list(span) for span in census_intervals],
            "forward_step_indices": census_indices,
            "event_count": len(selected_census_batch_sequence),
            "batch_size_sequence": selected_census_batch_sequence,
            "occupied_batch_histogram": selected_occupied_batch_histogram,
        },
    }
    endpoint_counters = {
        "fwd_steps": max(
            integral(window["post"]["fwd_steps"], f"{arm}:post fwd steps")
            for window in windows
        ),
        "wall_steps": max(
            integral(window["post"]["wall_steps"], f"{arm}:post wall steps")
            for window in windows
        ),
        "wall_rejected": max(
            integral(window["post"]["wall_rejected"], f"{arm}:post wall rejected")
            for window in windows
        ),
        "wall_attempts": max(
            integral(window["post"]["wall_attempts"], f"{arm}:post wall attempts")
            for window in windows
        ),
    }
    main_endpoints = {
        "fwd_steps": main_sidecar["counters"]["fwd_steps"],
        "wall_steps": main_sidecar["counters"]["wall_steps"],
        "wall_rejected": main_sidecar["counters"]["wall_rejected"],
        "wall_attempts": main_sidecar["wall_attempts"],
    }
    if endpoint_counters != main_endpoints:
        raise GateError(
            f"{arm}: final task bracket counters do not match the explicitly "
            f"flushed main sidecar: {endpoint_counters} != {main_endpoints}"
        )
    main_sidecar["final_task_bracket_endpoints"] = endpoint_counters
    statistics_out = (
        b1_arm_statistics(windows, arrays, expected_tokens)
        if concurrency == 1
        else b4_arm_statistics(
            windows,
            arrays,
            expected_tokens,
            concurrency,
            reps,
            seed,
        )
    )
    task_points = {
        window["task_id"]: {
            "wall_ms": 1000.0
            * window["delta"]["wall_s"]
            / window["delta"]["wall_steps"],
            "verify_ms": 1000.0
            * window["delta"]["fwd_s"]
            / window["delta"]["fwd_steps"],
            "rows": (
                window["delta"]["wall_drafts"]
                / window["delta"]["wall_steps"]
                * (expected_tokens + 1)
            ),
        }
        for window in windows
    }
    return (
        {
            "arm": expected_kind,
            "artifact_dir": str(arm_dir),
            "inferred_concurrency": concurrency,
            "expected_draft_tokens_per_event": expected_tokens,
            "active_logical_drafts_per_event": mode_spec["active_drafts"],
            "valid_mask": f"{mode_spec['valid_mask']:#010x}",
            "canonical_task_ids": [window["task_id"] for window in windows],
            "provenance": {
                "orchestrator": orchestrator,
                "launch": launch,
                "runtime": runtime,
                "real_tasks": real_task_provenance,
                "metric_labels": windows[0]["metric_labels"],
                "task_metric_brackets": {
                    window["task_id"]: window["metric_artifacts"]
                    for window in windows
                },
                "metric_hashes_derived_from_parsed_bytes": True,
                "all_required_provenance_valid": True,
            },
            "sidecar": {
                "per_step": sidecar,
                "main": main_sidecar,
            },
            "flush_chain": flush_chain,
            "work_census_expected": work_census_expected,
            "statistics": statistics_out,
        },
        task_points,
    )


def b1_comparison(
    tail_points: dict[str, dict[str, float]],
    hydra_points: dict[str, dict[str, float]],
) -> dict[str, Any]:
    if list(tail_points) != list(hydra_points):
        raise GateError("Tail6-fixed32/Hydra27-fixed32 task order differs")
    wall_deltas = []
    verify_deltas = []
    excess_deltas = []
    per_task = []
    for task_id in tail_points:
        tail = tail_points[task_id]
        hydra = hydra_points[task_id]
        tail_limit = legacy_slo(tail["rows"])[1]
        hydra_limit = legacy_slo(hydra["rows"])[1]
        wall_delta = hydra["wall_ms"] - tail["wall_ms"]
        verify_delta = hydra["verify_ms"] - tail["verify_ms"]
        excess_delta = (hydra["wall_ms"] - hydra_limit) - (tail["wall_ms"] - tail_limit)
        wall_deltas.append(wall_delta)
        verify_deltas.append(verify_delta)
        excess_deltas.append(excess_delta)
        per_task.append(
            {
                "task_id": task_id,
                "hydra_minus_tail_wall_ms_per_step": wall_delta,
                "hydra_minus_tail_verify_ms_per_step": verify_delta,
                "hydra_minus_tail_legacy_slo_excess_ms": excess_delta,
            }
        )
    return {
        "scope": (
            "paired equal-task cluster diagnostic; Hydra27-fixed32 minus Tail6-fixed32"
        ),
        "wall_ms_per_step_delta": cluster_summary(wall_deltas),
        "verify_ms_per_step_delta": cluster_summary(verify_deltas),
        "legacy_slo_excess_ms_delta": cluster_summary(excess_deltas),
        "per_task": per_task,
    }


def validate_work_census_v5_report(
    report: dict[str, Any],
    *,
    required_batch: int,
) -> dict[str, Any]:
    expected_modes = ("tail6_fixed32", "hydra27_fixed32")
    expected_batches = tuple(SUPPORTED_BATCH_SIZES)
    expected_batch_keys = {str(batch) for batch in expected_batches}
    exact_keys(
        report,
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
        },
        "fixed32 work-census v5 report",
    )
    if (
        report["schema"] != WORK_CENSUS_REPORT_SCHEMA
        or report["status"] != "PASS"
        or report["required_batch_sizes"] != [required_batch]
        or report["scope"] != FIXED_WORK_SCOPE
        or report["semantic_modes"] != WORK_CENSUS_MODE_SEMANTICS
        or not isinstance(report["normalized_work_signature"], dict)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(report["normalized_work_signature_sha256"]),
        )
        is None
        or canonical_json_sha256(report["normalized_work_signature"])
        != report["normalized_work_signature_sha256"]
    ):
        raise GateError("fixed32 work-census v5 report contract mismatch")

    histograms = report["physical_work_histograms"]
    event_counts = report["event_counts"]
    if not isinstance(histograms, dict) or not isinstance(event_counts, dict):
        raise GateError("fixed32 physical-work histograms are malformed")
    exact_keys(histograms, set(expected_modes), "physical_work_histograms")
    exact_keys(event_counts, set(expected_modes), "event_counts")
    observed_by_mode: dict[str, set[int]] = {}
    signatures_by_mode: dict[str, dict[int, str]] = {}
    for mode in expected_modes:
        mode_histogram = histograms[mode]
        mode_event_counts = event_counts[mode]
        if not isinstance(mode_histogram, dict) or not isinstance(
            mode_event_counts, dict
        ):
            raise GateError(f"{mode}: physical-work histogram is malformed")
        exact_keys(
            mode_histogram,
            expected_batch_keys,
            f"physical_work_histograms.{mode}",
        )
        observed_by_mode[mode] = set()
        signatures_by_mode[mode] = {}
        for batch in expected_batches:
            batch_key = str(batch)
            entry = mode_histogram[batch_key]
            if not isinstance(entry, dict):
                raise GateError(f"{mode}: B{batch} histogram entry is malformed")
            exact_keys(
                entry,
                {"event_count", "normalized_event_signatures"},
                f"physical_work_histograms.{mode}.{batch_key}",
            )
            event_count = entry["event_count"]
            signatures = entry["normalized_event_signatures"]
            expected_count = mode_event_counts.get(batch_key, 0)
            if (
                isinstance(event_count, bool)
                or not isinstance(event_count, int)
                or event_count < 0
                or event_count != expected_count
                or not isinstance(signatures, dict)
            ):
                raise GateError(
                    f"{mode}: B{batch} physical-work event count is inconsistent"
                )
            for signature, count in signatures.items():
                if (
                    re.fullmatch(r"[0-9a-f]{64}", str(signature)) is None
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or count <= 0
                ):
                    raise GateError(
                        f"{mode}: B{batch} physical-work signature is malformed"
                    )
            if sum(signatures.values()) != event_count:
                raise GateError(
                    f"{mode}: B{batch} signature counts do not reconcile"
                )
            if event_count == 0:
                if signatures:
                    raise GateError(
                        f"{mode}: B{batch} unobserved histogram is nonempty"
                    )
                continue
            if len(signatures) != 1:
                raise GateError(
                    f"{mode}: B{batch} does not have one physical-work signature"
                )
            observed_by_mode[mode].add(batch)
            signatures_by_mode[mode][batch] = next(iter(signatures))
        expected_count_keys = {
            str(batch) for batch in observed_by_mode[mode]
        }
        if set(mode_event_counts) != expected_count_keys:
            raise GateError(f"{mode}: event-count histogram keys are inconsistent")

    tail_mode, hydra_mode = expected_modes
    if observed_by_mode[tail_mode] != observed_by_mode[hydra_mode]:
        raise GateError(
            "Tail/Hydra occupied batch sets differ, so per-B physical work "
            "cannot be compared"
        )
    observed_batches = sorted(observed_by_mode[tail_mode])
    if required_batch not in observed_batches:
        raise GateError(f"fixed32 work census lacks required B{required_batch}")
    physical_per_batch: dict[str, dict[str, Any]] = {}
    for batch in observed_batches:
        tail_signature = signatures_by_mode[tail_mode][batch]
        hydra_signature = signatures_by_mode[hydra_mode][batch]
        if tail_signature != hydra_signature:
            raise GateError(
                f"B{batch}: Tail/Hydra normalized physical-work SHA differs"
            )
        physical_per_batch[str(batch)] = {
            "normalized_event_signature_sha256": tail_signature,
            "tail_event_count": histograms[tail_mode][str(batch)]["event_count"],
            "hydra_event_count": histograms[hydra_mode][str(batch)]["event_count"],
        }

    registries = report["drafter_graph_registries"]
    terminals = report["terminal_summaries"]
    if not isinstance(registries, dict) or not isinstance(terminals, dict):
        raise GateError("fixed32 drafter graph registries are malformed")
    exact_keys(registries, set(expected_modes), "drafter_graph_registries")
    exact_keys(terminals, set(expected_modes), "terminal_summaries")
    registry_by_mode: dict[str, dict[int, dict[str, Any]]] = {}
    registry_keys = {
        "batch_size",
        "graph_signature",
        "captures",
        "capture_origin",
        "measured_replays",
        "unmeasured_replays",
    }
    for mode in expected_modes:
        rows = registries[mode]
        terminal = terminals[mode]
        if (
            not isinstance(rows, list)
            or not rows
            or not isinstance(terminal, dict)
            or terminal.get("drafter_graph_registry") != rows
            or terminal.get("scope") != FIXED_WORK_SCOPE
        ):
            raise GateError(f"{mode}: terminal drafter registry/scope mismatch")
        registry_by_mode[mode] = {}
        ordered_batches = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise GateError(f"{mode}: drafter registry row {index} is malformed")
            exact_keys(
                row,
                registry_keys,
                f"drafter_graph_registries.{mode}[{index}]",
            )
            batch = row["batch_size"]
            if (
                isinstance(batch, bool)
                or not isinstance(batch, int)
                or batch not in expected_batches
                or batch in registry_by_mode[mode]
                or not isinstance(row["graph_signature"], str)
                or re.fullmatch(r"[0-9a-f]{64}", row["graph_signature"])
                is None
                or row["captures"] != 1
                or isinstance(row["captures"], bool)
                or not isinstance(row["capture_origin"], str)
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
                raise GateError(f"{mode}: drafter registry row {index} is invalid")
            ordered_batches.append(batch)
            registry_by_mode[mode][batch] = row
        if ordered_batches != sorted(ordered_batches):
            raise GateError(f"{mode}: drafter registry rows are not sorted")
        if not observed_by_mode[mode].issubset(registry_by_mode[mode]):
            raise GateError(
                f"{mode}: drafter registry does not cover occupied batches"
            )

    if set(registry_by_mode[tail_mode]) != set(registry_by_mode[hydra_mode]):
        raise GateError("Tail/Hydra drafter graph registry batch sets differ")
    lifecycle_per_batch: dict[str, dict[str, Any]] = {}
    for batch in sorted(registry_by_mode[tail_mode]):
        tail_row = registry_by_mode[tail_mode][batch]
        hydra_row = registry_by_mode[hydra_mode][batch]
        if (
            tail_row["graph_signature"] != hydra_row["graph_signature"]
            or tail_row["capture_origin"] != hydra_row["capture_origin"]
        ):
            raise GateError(
                f"B{batch}: Tail/Hydra drafter graph lifecycle differs"
            )
        lifecycle_per_batch[str(batch)] = {
            "graph_signature": tail_row["graph_signature"],
            "captures_per_arm": 1,
            "capture_origin": tail_row["capture_origin"],
            "tail_measured_replays": tail_row["measured_replays"],
            "hydra_measured_replays": hydra_row["measured_replays"],
            "tail_unmeasured_replays": tail_row["unmeasured_replays"],
            "hydra_unmeasured_replays": hydra_row["unmeasured_replays"],
        }

    forward_registries = report["forward_graph_registries"]
    auxiliary_by_mode = report["conv_pregather_auxiliary"]
    if not isinstance(forward_registries, dict) or not isinstance(
        auxiliary_by_mode, dict
    ):
        raise GateError("fixed32 forward graph pregather proof is malformed")
    exact_keys(
        forward_registries,
        set(expected_modes),
        "forward_graph_registries",
    )
    exact_keys(
        auxiliary_by_mode,
        set(expected_modes),
        "conv_pregather_auxiliary",
    )
    forward_registry_keys = {
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
    auxiliary_keys = {
        "profile_capture_stages",
        "aux_capture_stages",
        "host_actual_stages",
        "host_actual_stages_by_batch",
    }
    expected_zero_by_batch = {
        str(batch): 0 for batch in expected_batches
    }
    forward_by_mode: dict[str, dict[int, dict[str, Any]]] = {}
    nonpure_dispatch_by_mode: dict[str, dict[str, int]] = {}
    nonpure_committer_replays_by_mode: dict[str, dict[str, int]] = {}
    nonpure_dispatch_keys = {
        "guarded_steps",
        "piecewise_steps",
        "none_steps",
        "forbidden_full_steps",
    }
    for mode in expected_modes:
        rows = forward_registries[mode]
        terminal = terminals[mode]
        auxiliary = auxiliary_by_mode[mode]
        if (
            not isinstance(rows, list)
            or not rows
            or not isinstance(terminal, dict)
            or terminal.get("forward_graph_registry") != rows
            or terminal.get("conv_pregather_auxiliary") != auxiliary
        ):
            raise GateError(
                f"{mode}: terminal forward graph pregather proof mismatch"
            )
        nonpure_dispatch = terminal.get("nonpure_dispatch")
        if not isinstance(nonpure_dispatch, dict):
            raise GateError(
                f"{mode}: terminal nonpure dispatch proof is malformed"
            )
        exact_keys(
            nonpure_dispatch,
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
            raise GateError(
                f"{mode}: terminal nonpure dispatch counts do not reconcile"
            )
        nonpure_dispatch_by_mode[mode] = dict(nonpure_dispatch)
        nonpure_committer = terminal.get(
            "nonpure_committer_replays_by_batch"
        )
        if not isinstance(nonpure_committer, dict):
            raise GateError(
                f"{mode}: terminal nonpure committer proof is malformed"
            )
        exact_keys(
            nonpure_committer,
            expected_batch_keys,
            (
                f"terminal_summaries.{mode}."
                "nonpure_committer_replays_by_batch"
            ),
        )
        if (
            any(
                type(nonpure_committer[key]) is not int
                or nonpure_committer[key] < 0
                for key in expected_batch_keys
            )
            or sum(nonpure_committer.values())
            > nonpure_dispatch["guarded_steps"]
        ):
            raise GateError(
                f"{mode}: terminal nonpure committer counts are invalid"
            )
        nonpure_committer_replays_by_mode[mode] = dict(nonpure_committer)
        if not isinstance(auxiliary, dict):
            raise GateError(f"{mode}: pregather auxiliary proof is malformed")
        exact_keys(
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
            or auxiliary["host_actual_stages_by_batch"]
            != expected_zero_by_batch
            or any(
                type(value) is not int
                for value in auxiliary["host_actual_stages_by_batch"].values()
            )
        ):
            raise GateError(
                f"{mode}: pregather auxiliary/host stage counts are not zero"
            )
        forward_by_mode[mode] = {}
        ordered_batches: list[int] = []
        signatures: set[str] = set()
        layout_signatures: set[str] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise GateError(
                    f"{mode}: forward graph registry row {index} is malformed"
                )
            exact_keys(
                row,
                forward_registry_keys,
                f"forward_graph_registries.{mode}[{index}]",
            )
            batch = row["batch_size"]
            signature = row["graph_signature"]
            layout_signature = row["conv_layout_sha256"]
            if (
                type(batch) is not int
                or batch not in expected_batches
                or batch in forward_by_mode[mode]
                or not isinstance(signature, str)
                or re.fullmatch(r"[0-9a-f]{64}", signature) is None
                or signature != forward_graph_structural_signature(batch)
                or signature in signatures
                or not isinstance(layout_signature, str)
                or re.fullmatch(r"[0-9a-f]{64}", layout_signature) is None
                or layout_signature in layout_signatures
            ):
                raise GateError(
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
                raise GateError(
                    f"{mode}: forward graph registry row {index} "
                    "does not prove one ordered final-FULL pregather capture"
                )
            ordered_batches.append(batch)
            signatures.add(signature)
            layout_signatures.add(layout_signature)
            forward_by_mode[mode][batch] = row
        expected_registry_batches = list(range(1, required_batch + 1))
        if ordered_batches != expected_registry_batches:
            raise GateError(
                f"{mode}: forward graph registry must be exact B1.."
                f"B{required_batch}"
            )
        if not observed_by_mode[mode].issubset(forward_by_mode[mode]):
            raise GateError(
                f"{mode}: forward graph registry does not cover occupied batches"
            )

    forward_per_batch: dict[str, dict[str, Any]] = {}
    for batch in range(1, required_batch + 1):
        tail_row = forward_by_mode[tail_mode][batch]
        hydra_row = forward_by_mode[hydra_mode][batch]
        if (
            tail_row["graph_signature"] != hydra_row["graph_signature"]
            or tail_row["conv_layout_sha256"]
            != hydra_row["conv_layout_sha256"]
        ):
            raise GateError(
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
            "registry_batch_sizes": sorted(registry_by_mode[tail_mode]),
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


def reduce_campaign(
    repo: Path,
    runroot: Path,
    tag: str,
    task_count: int,
    expected_concurrency: int | None,
    sidecar_dir: Path,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    if task_count not in EVIDENCE_SETS:
        raise GateError("task count must be exactly 4 or 16")
    if reps != BOOTSTRAP_REPS or seed != BOOTSTRAP_SEED:
        raise GateError(
            "formal moving-block parameters are pinned to "
            f"reps={BOOTSTRAP_REPS}, seed={BOOTSTRAP_SEED}"
        )
    source_fingerprint = validate_source_fingerprint(repo, runroot)
    external_fingerprint = validate_external_fingerprint(runroot)
    tail_name = f"tail6_fixed32_{tag}"
    hydra_name = f"hydra27_fixed32_{tag}"
    tail, tail_points = reduce_arm(
        repo,
        runroot,
        sidecar_dir,
        tail_name,
        mode="tail6_fixed32",
        task_count=task_count,
        expected_concurrency=expected_concurrency,
        reps=reps,
        seed=seed + 100_000,
    )
    hydra, hydra_points = reduce_arm(
        repo,
        runroot,
        sidecar_dir,
        hydra_name,
        mode="hydra27_fixed32",
        task_count=task_count,
        expected_concurrency=expected_concurrency,
        reps=reps,
        seed=seed + 200_000,
    )
    if tail["inferred_concurrency"] != hydra["inferred_concurrency"]:
        raise GateError("Tail6-fixed32/Hydra27-fixed32 inferred concurrency differs")
    tail_subset = tail["provenance"]["launch"]["subset"]
    hydra_subset = hydra["provenance"]["launch"]["subset"]
    if tail_subset != hydra_subset:
        raise GateError(
            "Tail6-fixed32/Hydra27-fixed32 canonical subset provenance differs"
        )
    tail_attestation = tail["provenance"]["runtime"]["runtime_attestation"]
    hydra_attestation = hydra["provenance"]["runtime"]["runtime_attestation"]
    tail_attested_identity = {
        key: value
        for key, value in tail_attestation.items()
        if key not in {"path"}
    }
    hydra_attested_identity = {
        key: value
        for key, value in hydra_attestation.items()
        if key not in {"path"}
    }
    if tail_attested_identity != hydra_attested_identity:
        raise GateError(
            "Tail6-fixed32/Hydra27-fixed32 runtime attestations differ"
        )
    runtime_attestation_match = {
        "byte_equal": tail_attestation["sha256"] == hydra_attestation["sha256"],
        "canonical_sha256": tail_attestation["canonical_sha256"],
        "file_sha256": tail_attestation["sha256"],
        "vllm": tail_attestation["vllm"],
        "forked_fa2": tail_attestation["forked_fa2"],
        "arctic": tail_attestation["arctic"],
    }
    concurrency = tail["inferred_concurrency"]
    census_paths = {
        "tail6_fixed32": Path(tail["work_census_expected"]["path"]),
        "hydra27_fixed32": Path(hydra["work_census_expected"]["path"]),
    }
    for path in census_paths.values():
        if Path(f"{path}.tmp").exists():
            raise GateError(f"{path}: stale work-census temporary file is present")
    try:
        work_census_report = validate_work_census_campaign(
            load_work_census_jsonl(census_paths["tail6_fixed32"]),
            load_work_census_jsonl(census_paths["hydra27_fixed32"]),
            required_batches=(concurrency,),
        )
    except WorkCensusError as error:
        raise GateError(f"fixed32 work census failed: {error}") from error
    work_census_v5 = validate_work_census_v5_report(
        work_census_report,
        required_batch=concurrency,
    )
    b4_occupancy: dict[str, dict[str, Any]] = {}
    for mode, arm_result in (
        ("tail6_fixed32", tail),
        ("hydra27_fixed32", hydra),
    ):
        expected_census = arm_result["work_census_expected"]
        complete_stream = expected_census["complete_stream"]
        if (
            work_census_report["forward_step_indices"][mode]
            != complete_stream["forward_step_indices"]
        ):
            raise GateError(
                f"{mode}: work-census global forward-step indices do not "
                "exactly match the complete SFWD stream"
            )
        actual_histogram = work_census_report["event_counts"][mode]
        if actual_histogram != complete_stream["occupied_batch_histogram"]:
            raise GateError(
                f"{mode}: work-census occupancy does not match the complete "
                f"SFWD stream: {actual_histogram} != "
                f"{complete_stream['occupied_batch_histogram']}"
            )
        if sum(actual_histogram.values()) != complete_stream["event_count"]:
            raise GateError(
                f"{mode}: work-census event count does not match the complete "
                "SFWD stream"
            )
        actual_batch_sequence = work_census_report["batch_size_sequences"][mode]
        if actual_batch_sequence != complete_stream["batch_size_sequence"]:
            raise GateError(
                f"{mode}: work-census batch sequence does not match the "
                "complete SFWD stream"
            )
        if work_census_report["producer_pids"][mode] != expected_census["producer_pid"]:
            raise GateError(
                f"{mode}: work-census PID does not match recorded EngineCore PID"
            )
        terminal = work_census_report["terminal_summaries"][mode]
        final_counters = arm_result["flush_chain"]["final"]["counters"]
        if (
            terminal["producer_pid"] != expected_census["producer_pid"]
            or terminal["event_count"]
            != final_counters["complete_work_census_events"]
            or terminal["last_forward_step_index"]
            != final_counters["work_census_last_forward_step"]
        ):
            raise GateError(
                f"{mode}: final flush counters do not match terminal census summary"
            )
        selection = expected_census["canonical_task_selection"]
        if (
            selection["forward_step_indices"]
            != complete_stream["forward_step_indices"]
        ):
            raise GateError(
                f"{mode}: canonical-task forward union does not cover the "
                "complete post-ready decode stream"
            )
        selected_batches = [
            actual_batch_sequence[index] for index in selection["forward_step_indices"]
        ]
        if selected_batches != selection["batch_size_sequence"]:
            raise GateError(
                f"{mode}: post-hoc work-census task selection does not match "
                "the canonical task forward union"
            )
        selected_histogram = {
            str(batch_size): selected_batches.count(batch_size)
            for batch_size in range(1, concurrency + 1)
            if batch_size in selected_batches
        }
        if (
            len(selected_batches) != selection["event_count"]
            or selected_histogram != selection["occupied_batch_histogram"]
        ):
            raise GateError(
                f"{mode}: post-hoc work-census task selection cardinality "
                "or occupancy differs from the canonical task forward union"
            )
        if concurrency == 4:
            selected_count = len(selected_batches)
            exact_b4_events = selected_batches.count(4)
            ge3_events = sum(batch >= 3 for batch in selected_batches)
            mean_occupancy = sum(selected_batches) / selected_count
            ge3_fraction = ge3_events / selected_count
            if (
                exact_b4_events < MIN_B4_EXACT_EVENTS
                or ge3_fraction < MIN_B4_GE3_FRACTION
                or mean_occupancy < MIN_B4_MEAN_OCCUPANCY
            ):
                raise GateError(
                    f"{mode}: B4 canonical-task exposure is under-occupied: "
                    f"exact_b4={exact_b4_events} ge3_fraction={ge3_fraction:.6f} "
                    f"mean={mean_occupancy:.6f}"
                )
            b4_occupancy[mode] = {
                "selected_events": selected_count,
                "exact_b4_events": exact_b4_events,
                "at_least_b3_events": ge3_events,
                "at_least_b3_fraction": ge3_fraction,
                "mean_occupancy": mean_occupancy,
            }
    if concurrency == 4:
        mean_gap = abs(
            b4_occupancy["tail6_fixed32"]["mean_occupancy"]
            - b4_occupancy["hydra27_fixed32"]["mean_occupancy"]
        )
        if mean_gap > MAX_B4_MEAN_OCCUPANCY_GAP:
            raise GateError(
                "fixed32 B4 arm mean occupancies are not matched: "
                f"gap={mean_gap:.6f}"
            )
        b4_occupancy["matched_arm_mean_gap"] = mean_gap
    work_census = {
        "report": work_census_report,
        "physical_work_comparison": work_census_v5[
            "physical_work_comparison"
        ],
        "drafter_graph_lifecycle": work_census_v5[
            "drafter_graph_lifecycle"
        ],
        "forward_graph_pregather_lifecycle": work_census_v5[
            "forward_graph_pregather_lifecycle"
        ],
        "scope": work_census_v5["scope"],
        "scope_interpretation": (
            "Exact equality is limited to direct observations and "
            "contract-derived work listed in scope. The explicit "
            "data_dependent_unproven entries prevent a total memory-traffic "
            "or hardware-cycle claim."
        ),
        "files": {
            mode: {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for mode, path in census_paths.items()
        },
        "complete_terminal_stream_reconciled_to_sfwd_sidecar": True,
        "canonical_task_forward_counter_union_selected_posthoc": True,
        "canonical_task_forward_union_covers_complete_stream": True,
        "b4_occupancy_gate": (
            b4_occupancy if concurrency == 4 else "not_applicable_b1"
        ),
    }
    comparison = (
        b1_comparison(tail_points, hydra_points)
        if concurrency == 1
        else {
            "scope": (
                "descriptive union-counter point differences only; no paired "
                "task-general B=4 inference"
            ),
            "hydra_minus_tail_wall_ms_per_step": (
                hydra["statistics"]["union_counter_point"]["wall_ms_per_step"]
                - tail["statistics"]["union_counter_point"]["wall_ms_per_step"]
            ),
            "hydra_minus_tail_verify_ms_per_step": (
                hydra["statistics"]["union_counter_point"]["verify_ms_per_step"]
                - tail["statistics"]["union_counter_point"]["verify_ms_per_step"]
            ),
        }
    )
    gates = {
        "source_runtime_fingerprint_equal": True,
        "external_artifact_fingerprint_equal": True,
        "arm_runtime_attestations_equal": True,
        "running_container_image_identity_exact": True,
        "task_metric_bracket_bytes_bound": True,
        "fixed32_pretask_zero_positive_traffic": all(
            arm["provenance"]["runtime"]["pretask_zero_traffic"][
                "forbidden_probe_artifacts_absent"
            ]
            for arm in (tail, hydra)
        ),
        "all_canonical_tasks_have_real_model_traffic": all(
            arm["provenance"]["real_tasks"][
                "all_canonical_tasks_have_real_model_traffic"
            ]
            for arm in (tail, hydra)
        ),
        "all_positive_usage_proxy_pairs_task_bound": all(
            arm["provenance"]["real_tasks"][
                "all_positive_usage_proxy_pairs_task_bound"
            ]
            for arm in (tail, hydra)
        ),
        "all_task_agents_completed_cleanly": all(
            arm["provenance"]["real_tasks"]["all_agents_completed_cleanly"]
            for arm in (tail, hydra)
        ),
        "all_tasks_have_terminal_swe_verdicts": all(
            arm["provenance"]["real_tasks"][
                "all_tasks_have_terminal_eval_verdicts"
            ]
            for arm in (tail, hydra)
        ),
        "canonical_exact_4_or_16_task_binding": task_count in EVIDENCE_SETS,
        "canonical_completed_task_set": True,
        "canonical_subset_hash": True,
        "uncapped_sidecars": True,
        "sidecar_coverage_eq_1_0": True,
        "sidecar_counter_reconciliation": True,
        "fixed32_work_census_exact": True,
        "fixed32_per_batch_physical_work_equal": True,
        "fixed32_drafter_graph_lifecycle_exact_and_matched": True,
        "fixed32_forward_graph_pregather_exact": True,
        "fixed32_scope_limitations_explicit": bool(
            work_census_v5["scope"]["data_dependent_unproven"]
        ),
        "canonical_task_forward_union_covers_complete_stream": True,
        "fixed32_flush_generation_chain_exact": True,
        "fixed32_task_boundaries_exact": True,
        "b4_occupancy_exposure": True,
        "tail6_fixed32_legacy_slo": tail["statistics"]["gate"]["pass"],
        "hydra27_fixed32_legacy_slo": hydra["statistics"]["gate"]["pass"],
    }
    return {
        "schema": "fr13.canonical_swe_verified_fixed32_floor_gate.v8",
        "analysis_valid": True,
        "gate_verdict": "PASS" if all(gates.values()) else "FAIL",
        "repo": str(repo),
        "runroot": str(runroot),
        "tag": tag,
        "task_count": task_count,
        "inferred_concurrency": concurrency,
        "source_runtime_fingerprint": source_fingerprint,
        "external_artifact_fingerprint": external_fingerprint,
        "matched_runtime_attestation": runtime_attestation_match,
        "fixed32_work_census": work_census,
        "slo_definition": {
            "name": "legacy_aggressive_weight_stream_slo",
            "formula": ("wall_ms_per_step <= 1.15 * max(98.6, 0.54 * rows_per_step)"),
            "weight_stream_lower_bound_ms": WEIGHT_STREAM_LOWER_BOUND_MS,
            "compute_ms_per_row": COMPUTE_MS_PER_ROW,
            "multiplier": SLO_MULTIPLIER,
            "interpretation": (
                "98.6 ms is a weight-stream lower bound used by an aggressive "
                "legacy SLO; it is not a measured full physical hardware floor"
            ),
        },
        "uncertainty_model": (
            "B=1 uses equal-weight whole-task clusters and one-sided t U95; "
            "B=4 uses one overlap-safe physical-step union plus a separately "
            "gated exact-B4 wall stratum, both with conditional moving-block "
            "U95 sensitivity at blocks 64/128/256/512"
        ),
        "evidence_requirements": {
            "sidecar_counter_interval_coverage": REQUIRED_COVERAGE,
            "sidecar_drafts_and_steps": "exactly_reconciled",
            "sidecar_timing": "within_4_decimal_ms_per_sample_rounding_bound",
            "runtime_flush": "ready_snapshot_per_task_final_generation_chain_exact",
            "minimum_retained_steps_per_task_and_family": MIN_TASK_COUNTER_STEPS,
            "wall_rejected_delta_per_task": 0,
            "wall_cap_seconds": 1.5,
            "b4_selected_exact_events_minimum": MIN_B4_EXACT_EVENTS,
            "b4_selected_at_least_b3_fraction_minimum": MIN_B4_GE3_FRACTION,
            "b4_selected_mean_occupancy_minimum": MIN_B4_MEAN_OCCUPANCY,
            "b4_arm_mean_occupancy_gap_maximum": MAX_B4_MEAN_OCCUPANCY_GAP,
            "moving_block_bootstrap_reps": BOOTSTRAP_REPS,
            "moving_block_bootstrap_seed": BOOTSTRAP_SEED,
        },
        "arms": {"tail6_fixed32": tail, "hydra27_fixed32": hydra},
        "comparison": comparison,
        "gates": gates,
    }


def fixture_metrics(
    fwd_ms: list[float],
    wall_ms: list[float],
    drafts: list[int],
    index: int,
    tokens_per_draft: int,
) -> str:
    fwd_seconds = sum(fwd_ms[:index]) / 1000.0
    wall_seconds = sum(wall_ms[:index]) / 1000.0
    draft_count = sum(drafts[:index])
    values = {
        "fwd_s": fwd_seconds,
        "fwd_steps": index,
        "fwd_drafts": draft_count,
        "wall_s": wall_seconds,
        "wall_drafts": draft_count,
        "wall_steps": index,
        "wall_attempts": index,
        "wall_rejected": 0,
        "spec_drafts": draft_count,
        "spec_tokens": draft_count * tokens_per_draft,
    }
    lines = []
    for key, metric in METRICS.items():
        labels = (
            '{engine="0",model_name="qwen3.6-27b"}'
            if key in {"spec_drafts", "spec_tokens"}
            else ""
        )
        lines.append(f"{metric}{labels} {values[key]}")
    lines.append(f"{FIXED32_STEP_METRIC} {index}")
    lines.append(f"{FIXED32_CENSUS_METRIC} {index}")
    return "\n".join(lines) + "\n"


def replace_metric_values(text: str, replacements: dict[str, float]) -> str:
    names = {METRICS[key]: value for key, value in replacements.items()}
    seen: set[str] = set()
    output = []
    for line in text.splitlines():
        match = SAMPLE_RE.match(line)
        if match is not None and match.group("name") in names:
            name = match.group("name")
            line = line[: match.start("value")] + f"{names[name]:.12g}"
            seen.add(name)
        output.append(line)
    if seen != set(names):
        raise AssertionError(f"fixture metric replacements missing {set(names) - seen}")
    return "\n".join(output) + "\n"


def fixture_external_manifest() -> dict[str, Any]:
    model_files = fixed32_contract.expected_model_file_records()
    payload: dict[str, Any] = {
        "schema": fixed32_contract.EXTERNAL_SCHEMA,
        "canonical_format": fixed32_contract.CANONICAL_FORMAT,
        "image": {
            "reference": fixed32_contract.IMAGE_REFERENCE,
            "id": fixed32_contract.IMAGE_ID,
            "repo_digests": [fixed32_contract.IMAGE_REFERENCE],
            "os": fixed32_contract.IMAGE_OS,
            "architecture": fixed32_contract.IMAGE_ARCHITECTURE,
        },
        "forked_fa2": {
            "path": fixed32_contract.FA2_REPO_RELATIVE,
            "size": fixed32_contract.FA2_SIZE,
            "sha256": fixed32_contract.FA2_SHA256,
        },
        "model": {
            "root": str(fixed32_contract.MODEL_ROOT),
            "file_count": len(model_files),
            "files": model_files,
            "canonical_sha256": fixed32_contract.MODEL_CANONICAL_SHA256,
        },
        "arctic_source": {
            "version": fixed32_contract.ARCTIC_VERSION,
            "url": fixed32_contract.ARCTIC_SDIST_URL,
            "sha256": fixed32_contract.ARCTIC_SDIST_SHA256,
        },
    }
    payload["overall_canonical_sha256"] = canonical_json_sha256(payload)
    validate_external_manifest(payload)
    return payload


def fixture_runtime_attestation() -> dict[str, Any]:
    arctic_files = [
        {
            "path": "arctic_inference/suffix_decoding/cache.py",
            "size": 32,
            "sha256": hashlib.sha256(b"fixed32-fixture-arctic").hexdigest(),
        }
    ]
    fa2_source = {
        "path": str(fixed32_contract.CONTAINER_FA2_SOURCE),
        "size": fixed32_contract.FA2_SIZE,
        "sha256": fixed32_contract.FA2_SHA256,
    }
    fa2_destination = {
        "path": str(CONTAINER_FA2_DESTINATION),
        "size": fixed32_contract.FA2_SIZE,
        "sha256": fixed32_contract.FA2_SHA256,
    }
    payload: dict[str, Any] = {
        "schema": fixed32_contract.RUNTIME_SCHEMA,
        "canonical_format": fixed32_contract.CANONICAL_FORMAT,
        "python": {
            "version": "3.12.3",
            "implementation": "CPython",
        },
        "vllm": {
            "version": fixed32_contract.VLLM_VERSION,
            "module_path": "/usr/local/lib/python3.12/dist-packages/vllm/__init__.py",
        },
        "forked_fa2": {
            "source": fa2_source,
            "destination": fa2_destination,
            "byte_identical": True,
        },
        "arctic": {
            "name": "arctic-inference",
            "version": fixed32_contract.ARCTIC_VERSION,
            "files": arctic_files,
            "canonical_sha256": canonical_json_sha256(arctic_files),
            "cache_class_module": "arctic_inference.suffix_decoding.cache",
            "cache_class_qualname": "SuffixDecodingCache",
            "pinned_source_url": fixed32_contract.ARCTIC_SDIST_URL,
            "pinned_source_sha256": fixed32_contract.ARCTIC_SDIST_SHA256,
        },
    }
    payload["overall_canonical_sha256"] = canonical_json_sha256(payload)
    validate_runtime_attestation(payload)
    return payload


def write_fixture_arm(
    repo: Path,
    runroot: Path,
    sidecar_dir: Path,
    tag: str,
    *,
    hydra: bool,
    concurrency: int,
) -> None:
    kind = "hydra27_fixed32" if hydra else "tail6_fixed32"
    arm = f"{kind}_{tag}"
    arm_dir = runroot / arm
    arm_dir.mkdir(parents=True)
    logs_dir = arm_dir / "logs"
    logs_dir.mkdir()
    tokens = PHYSICAL_DRAFTS
    subset = (repo / EVIDENCE_SETS[4]["relative_path"]).resolve()
    producer_pid = 100 + int(hydra)
    pid1_argv = expected_pid1_argv(concurrency)
    (runroot / f"{arm}.runlog").write_text(
        f"=== BIGDENOM-VARIANT SWEServe ARM {arm} kind={kind} "
        f"launcher=forked expect={tokens} xflags=[] "
        f"subset={subset} ===\n"
        f"PID 1 cmd=[{' '.join(pid1_argv)} ]\n"
        f"PID {producer_pid} cmd=[VLLM::EngineCore ]\n"
        f"spec engagement OK: drafts delta=8.0 "
        f"draft_tokens/drafts={float(tokens):.1f}\n",
        encoding="utf-8",
    )
    mode_spec = FIXED32_MODE_SPECS[kind]
    required_env = fixed32_required_env(arm_dir, mode=kind)
    (arm_dir / "container_env.txt").write_text(
        "".join(f"{key}={value}\n" for key, value in required_env.items()),
        encoding="utf-8",
    )
    process_identity = {
        "schema": "fr13-fixed32-process-identity-v1",
        "pid1": {
            "pid": 1,
            "argv": pid1_argv,
            "environ": sorted(f"{key}={value}" for key, value in required_env.items()),
            "forked_fa2_maps": [],
        },
        "engine_core": {
            "pid": producer_pid,
            "argv": ["VLLM::EngineCore"],
            "environ": sorted(f"{key}={value}" for key, value in required_env.items()),
            "forked_fa2_maps": [
                "7f000000-7f100000 r-xp 00000000 00:00 0 "
                f"{CONTAINER_FA2_DESTINATION}"
            ],
        },
    }
    (arm_dir / "fixed32_process_identity.json").write_text(
        json.dumps(
            process_identity,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (arm_dir / "fixed32_engine_cmdline.txt").write_text(
        "VLLM::EngineCore\n", encoding="utf-8"
    )
    (arm_dir / "fixed32_container_identity.json").write_text(
        json.dumps(
            {
                "schema": "fr13-fixed32-container-identity-v1",
                "name": f"/fr13-bigdenom-{arm}",
                "image_id": fixed32_contract.IMAGE_ID,
                "configured_image": fixed32_contract.IMAGE_REFERENCE,
                "platform": fixed32_contract.IMAGE_OS,
                "running": True,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (logs_dir / "fr13_fixed32_runtime_attestation.json").write_text(
        json.dumps(
            fixture_runtime_attestation(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    needles = (
        FIXED32_PRESEED,
        FIXED32_ENGAGED,
        FIXED32_WORK_ENGAGED,
        mode_spec["topology_needle"],
    )
    (arm_dir / "docker_full.log").write_text(
        "\n".join(needles) + "\n", encoding="utf-8"
    )
    (arm_dir / "eval_offload_preflight.txt").write_text(
        "eval offload: alienware reachable\n",
        encoding="utf-8",
    )
    task_ids = EVIDENCE_SETS[4]["task_ids"]
    orchestrator_lines = [
        "=== [2026-01-01T00:00:00Z] "
        "dataset=princeton-nlp/SWE-bench_Verified tag=verified "
        f"n=4 concurrency={concurrency} ==="
    ]
    for index, task_id in enumerate(task_ids):
        orchestrator_lines.extend(
            [
                f"[2026-01-01T00:00:{index * 2 + 1:02d}Z] -> {task_id}",
                f"[2026-01-01T00:00:{index * 2 + 2:02d}Z] <- {task_id} "
                "verdict=resolved elapsed_total=1.0s",
            ]
        )
    orchestrator_lines.append(
        "=== [2026-01-01T00:01:00Z] DONE n=4 "
        "resolved_rate=1.0 verdicts={'resolved': 4} ==="
    )
    (arm_dir / "swe_orchestrator.log").write_text(
        "\n".join(orchestrator_lines) + "\n", encoding="utf-8"
    )

    if concurrency == 1:
        lengths = (64, 80, 96, 112)
        fwd_ms = []
        wall_ms = []
        drafts = []
        intervals = []
        cursor = 0
        for task_index, length in enumerate(lengths):
            start = cursor
            fwd_value = 70.0 + task_index * 2 + (3.0 if hydra else 0.0)
            wall_value = 100.0 + task_index * 4 + (5.0 if hydra else 0.0)
            fwd_ms.extend([fwd_value] * length)
            wall_ms.extend([wall_value] * length)
            drafts.extend([1] * length)
            cursor += length
            intervals.append((start, cursor))
    else:
        sample_count = 645
        fwd_ms = [
            70.0 + (index % 13) * 0.1 + (3.0 if hydra else 0.0)
            for index in range(sample_count)
        ]
        wall_ms = [
            100.0 + (index % 17) * 0.2 + (5.0 if hydra else 0.0)
            for index in range(sample_count)
        ]
        occupancy_pattern = (3, 4, 4, 4, 4)
        drafts = [
            occupancy_pattern[index % len(occupancy_pattern)]
            for index in range(sample_count)
        ]
        intervals = [(0, 500), (20, 620), (40, 640), (60, 645)]

    census_records = []
    for event_index, batch_size in enumerate(drafts):
        record = work_census_fixture(
            kind,
            int(batch_size),
            f"{kind}-{event_index}",
            event_index=event_index,
            forward_step_index=event_index,
        )
        record["producer_pid"] = producer_pid
        census_records.append(record)

    def flush_counters(step: int) -> dict[str, Any]:
        return {
            "pure_decode_forward_steps": step,
            "complete_work_census_events": step,
            "work_census_first_forward_step": 0 if step else None,
            "work_census_last_forward_step": step - 1 if step else None,
            "sfwd_pending": 0,
            "dfwd_pending": 0,
            "cfwd_pending": 0,
        }

    def flush_ack(generation: int, step: int, action: str) -> dict[str, Any]:
        return {
            "schema": FLUSH_ACK_SCHEMA,
            "mode": kind,
            "producer_pid": producer_pid,
            "generation": generation,
            "nonce": (
                FLUSH_READY_NONCE
                if generation == 0
                else f"{generation:064x}"
            ),
            "action": action,
            "status": "ok",
            "counters": flush_counters(step),
        }

    def write_runtime_snapshot(
        ack: dict[str, Any],
        step: int,
    ) -> dict[str, Any]:
        prefix = census_records[:step]
        histogram = {
            str(batch): drafts[:step].count(batch) for batch in range(1, 5)
        }
        draft_count = sum(drafts[:step])
        by_batch = {
            str(batch): histogram[str(batch)] for batch in range(1, 5)
        }
        capture_by_batch = {
            str(batch): int(batch <= concurrency) for batch in range(1, 5)
        }
        zero_by_batch = {str(batch): 0 for batch in range(1, 5)}
        snapshot = {
            "schema": FIXED32_RUNTIME_SNAPSHOT_SCHEMA,
            "mode": kind,
            "producer_pid": producer_pid,
            "generation": ack["generation"],
            "nonce": ack["nonce"],
            "action": ack["action"],
            "counters": ack["counters"],
            "metrics": {
                "fixed32": {
                    "pure_decode_forward_steps": step,
                    "complete_work_census_events": step,
                    "complete_spec_rows": draft_count,
                    "spec_drafts": draft_count,
                    "spec_tokens": tokens * draft_count,
                    "batch_histogram": histogram,
                    "first_forward_step": 0 if step else None,
                    "last_forward_step": step - 1 if step else None,
                    "events_sha256": hashlib.sha256(
                        json.dumps(
                            prefix,
                            ensure_ascii=True,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                },
                "sfwd": {
                    "gpu_seconds": sum(fwd_ms[:step]) / 1000.0,
                    "steps": step,
                    "drafts": draft_count,
                    "wall_seconds": sum(wall_ms[:step]) / 1000.0,
                    "wall_drafts": draft_count,
                    "wall_steps": step,
                    "wall_rejected": 0,
                },
                "dfwd": {
                    "gpu_seconds": step * 0.001,
                    "spans": step,
                },
                "cfwd": {
                    "gpu_seconds": step * 0.002,
                    "spans": step,
                },
                "committer": {
                    "captures": concurrency,
                    "actual_replays_enqueued": step,
                    "actual_replays_by_batch": by_batch,
                    "preseeded_graphs": concurrency,
                    "preseeded_batches": list(range(1, concurrency + 1)),
                    "ready_capacities": {
                        str(batch): concurrency
                        for batch in range(1, concurrency + 1)
                    },
                    "maximum_ready_capacity": concurrency,
                    "required_capacity": concurrency,
                    "all_batches_ready": True,
                },
                "conv_pregather": {
                    "preseeded": True,
                    "pointer_entries": 48,
                    "preseeded_batches": list(range(1, concurrency + 1)),
                    "max_batch_size": concurrency,
                    "graph_capture_stages": concurrency,
                    "graph_capture_stages_by_batch": capture_by_batch,
                    "profile_capture_stages": 0,
                    "aux_capture_stages": 0,
                    "actual_stages": 0,
                    "actual_stages_by_batch": zero_by_batch,
                    "graph_replay_stages": step,
                    "graph_replay_stages_by_batch": by_batch,
                },
            },
        }
        snapshot_path = (
            logs_dir
            / f"fr13_fixed32_boundary_snapshot.{ack['generation']}.json"
        )
        snapshot_path.write_text(
            json.dumps(snapshot, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "schema": FIXED32_RUNTIME_SNAPSHOT_SCHEMA,
            "generation": ack["generation"],
            "path": str(snapshot_path),
            "sha256": sha256_file(snapshot_path),
        }

    boundary_points = []
    for task_id, (start, end) in zip(task_ids, intervals, strict=True):
        boundary_points.extend(((start, "pre", task_id), (end, "post", task_id)))
    generation_by_boundary = {
        (task_id, boundary): generation
        for generation, (_step, boundary, task_id) in enumerate(
            sorted(boundary_points), start=1
        )
    }

    ready_ack = flush_ack(0, 0, "ready")
    ready_path = arm_dir / "fixed32_ready_ack.json"
    ready_path.write_text(
        json.dumps(ready_ack, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics_before_path = arm_dir / "metrics_before_swe.txt"
    metrics_before_path.write_text(
        fixture_metrics(fwd_ms, wall_ms, drafts, 0, tokens),
        encoding="utf-8",
    )
    census_path = logs_dir / "fr13_fixed32_work_census.jsonl"
    pretask_marker = {
        "schema": "fr13-fixed32-pretask-zero-traffic-v1",
        "mode": kind,
        "no_positive_probe": True,
        "generation_probe_commands_executed": 0,
        "metrics": {
            "path": str(metrics_before_path.resolve()),
            "sha256": sha256_file(metrics_before_path),
            "spec_drafts": 0,
            "spec_tokens": 0,
        },
        "work_census": {
            "path": str(census_path.resolve()),
            "exists": False,
            "bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        },
        "ready_ack": {
            "path": str(ready_path.resolve()),
            "sha256": sha256_file(ready_path),
            "generation": 0,
        },
    }
    (arm_dir / "fixed32_pretask_zero_traffic.json").write_text(
        json.dumps(
            pretask_marker,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (logs_dir / "fr13_fixed32_engine_pid").write_text(
        f"{producer_pid}\n", encoding="utf-8"
    )
    (logs_dir / "fr13_fixed32_mode.flag").write_text(
        f"{kind}\n", encoding="utf-8"
    )

    task_root = arm_dir / "swe_out" / "verified" / "per_task"
    pair_root = arm_dir / "proxy_pair_dumps"
    pair_root.mkdir()
    (arm_dir / "offload_fetch_status.txt").write_text("ok\n", encoding="utf-8")
    dataset_record_digests = pinned_dataset_record_digests(str(repo))
    for pair_index, (task_id, (start, end)) in enumerate(
        zip(task_ids, intervals, strict=True)
    ):
        task_dir = task_root / task_id
        task_dir.mkdir(parents=True)
        pre_ack = flush_ack(
            generation_by_boundary[(task_id, "pre")], start, "snapshot"
        )
        post_ack = flush_ack(
            generation_by_boundary[(task_id, "post")], end, "snapshot"
        )
        pre_runtime_snapshot = write_runtime_snapshot(pre_ack, start)
        post_runtime_snapshot = write_runtime_snapshot(post_ack, end)
        boundary = {
            "schema": FIXED32_BOUNDARY_SCHEMA,
            "instance_id": task_id,
            "mode": kind,
            "producer_pid": producer_pid,
            "pre": pre_ack,
            "post": post_ack,
            "pre_runtime_snapshot": pre_runtime_snapshot,
            "post_runtime_snapshot": post_runtime_snapshot,
            "forward_step_interval": {
                "start_forward_step": start,
                "end_forward_step": end,
                "expected_complete_events": end - start,
            },
        }
        (task_dir / "fixed32_task_boundary.json").write_text(
            json.dumps(boundary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        trace_event = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"Implemented and verified {task_id}.",
                    }
                ],
            },
            "usage": {
                "input_tokens": 128,
                "output_tokens": 32,
                "total_tokens": 160,
            },
        }
        trace_path = task_dir / "qwen_trace.jsonl"
        trace_path.write_text(
            json.dumps(
                trace_event,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raw_trace = trace_path.read_bytes()
        agent = {
            "elapsed_s": 1.0,
            "exit_code": 0,
            "timed_out": False,
            "offloaded": True,
            "codex_host": "alienware",
            "network_drop": False,
            "stall_killed": False,
            "patch_down_rc": 0,
        }
        agent_terminal = {
            "exit_code": 0,
            "timed_out": False,
            "offloaded": True,
            "network_drop": False,
            "stall_killed": False,
            "patch_down_rc": 0,
        }
        real_task_provenance = {
            "schema": "fr13-fixed32-real-task-provenance-v1",
            "instance_id": task_id,
            "trace_path": str(trace_path.resolve()),
            "trace_sha256": hashlib.sha256(raw_trace).hexdigest(),
            "trace_bytes": len(raw_trace),
            "event_count": 1,
            "assistant_event_count": 1,
            "assistant_output_event_count": 1,
            "qwen_assistant_event_count": 1,
            "codex_agent_message_event_count": 0,
            "positive_token_usage": True,
            "usage_record_count": 1,
            "positive_usage_record_count": 1,
            "usage_max_by_field": {
                "input_tokens": 128,
                "output_tokens": 32,
                "total_tokens": 160,
            },
            "agent_terminal": agent_terminal,
        }
        (task_dir / "runner_metadata.json").write_text(
            json.dumps(
                {
                    "instance_id": task_id,
                    "started_at": "2026-01-01T00:00:00Z",
                    "ended_at": "2026-01-01T00:00:01Z",
                    "agent": agent,
                    "codex": agent,
                    "eval_report": {"verdict": "resolved"},
                    "fixed32_real_task_provenance": real_task_provenance,
                    "fixed32_dataset_record_sha256": (
                        dataset_record_digests[task_id]
                    ),
                    "fixed32_task_boundary": boundary,
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        pair_payload = {
            "schema": "lumo.fr13.proxy_pair_dump.v1",
            "ts_ns": pair_index + 1,
            "seq": pair_index,
            "kind": "initial",
            "request": {
                "model": "qwen3.6-27b",
                "input": f"# SWE-Bench task: {task_id}",
            },
            "response": {
                "id": f"fixture-{pair_index}",
                "usage": {
                    "input_tokens": 128,
                    "output_tokens": 32,
                    "total_tokens": 160,
                },
            },
        }
        (pair_root / f"pair_{pair_index:06d}_initial.json").write_text(
            json.dumps(
                pair_payload,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (task_dir / "vllm_metrics_pre.txt").write_text(
            fixture_metrics(fwd_ms, wall_ms, drafts, start, tokens),
            encoding="utf-8",
        )
        (task_dir / "vllm_metrics_post.txt").write_text(
            fixture_metrics(fwd_ms, wall_ms, drafts, end, tokens),
            encoding="utf-8",
        )

    (arm_dir / "metrics_after_swe.txt").write_text(
        fixture_metrics(fwd_ms, wall_ms, drafts, len(fwd_ms), tokens),
        encoding="utf-8",
    )
    positive_pair_counts = {task_id: 1 for task_id in sorted(task_ids)}
    positive_traffic_audit = {
        "schema": "fr13-fixed32-positive-traffic-audit-v1",
        "mode": kind,
        "all_positive_usage_pairs_bound_to_one_canonical_task": True,
        "positive_pair_count": len(task_ids),
        "zero_usage_pair_count": 0,
        "task_positive_pair_counts": positive_pair_counts,
    }
    (arm_dir / "fixed32_positive_traffic_audit.json").write_text(
        json.dumps(
            positive_traffic_audit,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    final_generation = len(boundary_points) + 1
    final_ack = flush_ack(final_generation, len(fwd_ms), "final")
    write_runtime_snapshot(final_ack, len(fwd_ms))
    final_request = {
        "schema": FLUSH_REQUEST_SCHEMA,
        "mode": kind,
        "producer_pid": producer_pid,
        "prev_generation": final_generation - 1,
        "generation": final_generation,
        "nonce": final_ack["nonce"],
        "action": "final",
    }
    (logs_dir / "fr13_fixed32_flush_request.json").write_text(
        json.dumps(final_request, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (logs_dir / "fr13_fixed32_flush_ack.json").write_text(
        json.dumps(final_ack, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (arm_dir / "fixed32_final_flush.json").write_text(
        json.dumps(
            {"schema": FLUSH_RESULT_SCHEMA, "ack": final_ack},
            ensure_ascii=True,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (arm_dir / "fixed32_final_flush.stderr").write_text("", encoding="utf-8")
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "schema": "fr13.sfwd_per_step_samples.v1",
        "pid": 100 + int(hydra),
        "final": True,
        "fwd_drafts": drafts,
        "fwd_ms": fwd_ms,
        "fwd_cg": ["FULL"] * len(fwd_ms),
        "fwd_host_ms": [1.0] * len(fwd_ms),
        "fwd_exec_ms": [1.0] * len(fwd_ms),
        "fwd_cpu_tail_ms": [1.0] * len(fwd_ms),
        "wall_drafts": drafts,
        "wall_ms": wall_ms,
        "samples_capped": False,
    }
    (sidecar_dir / f"{arm}.json.samples.{100 + int(hydra)}").write_text(
        json.dumps(sidecar), encoding="utf-8"
    )
    main_sidecar = {
        "schema": "fr13.sfwd_gpu_timer.v1",
        "pid": 100 + int(hydra),
        "final": True,
        "decode_forward_gpu_seconds": sum(fwd_ms) / 1000.0,
        "n_pure_decode_steps_timed": len(fwd_ms),
        "n_drafts_in_timed_steps": sum(drafts),
        "decode_step_wall_seconds": sum(wall_ms) / 1000.0,
        "n_drafts_in_wall_steps": sum(drafts),
        "n_wall_steps": len(wall_ms),
        "n_wall_rejected": 0,
        "wall_cap_s": 1.5,
    }
    (sidecar_dir / f"{arm}.json.{100 + int(hydra)}").write_text(
        json.dumps(main_sidecar), encoding="utf-8"
    )
    census_records.append(
        work_census_terminal_fixture(
            census_records,
            fixture_synthetic_runtime_proof=True,
        )
    )
    census_path.write_text(
        "\n".join(
            json.dumps(
                record,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            for record in census_records
        )
        + "\n",
        encoding="utf-8",
    )


def write_fixture_campaign(
    repo: Path, base: Path, *, concurrency: int
) -> tuple[Path, Path, str]:
    tag = f"fixture_b{concurrency}"
    runroot = base / f"campaign_b{concurrency}"
    sidecar_dir = base / f"sidecars_b{concurrency}"
    runroot.mkdir()
    runtime_manifest = build_runtime_manifest(
        repo,
        profile=RUNTIME_MANIFEST_PROFILE,
        sequence=RUNTIME_MANIFEST_SEQUENCE,
    )
    rendered_manifest = json.dumps(
        runtime_manifest,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    for name in ("runtime_manifest.at_launch.json", "runtime_manifest.at_end.json"):
        (runroot / name).write_text(rendered_manifest + "\n", encoding="utf-8")
    external_manifest = json.dumps(
        fixture_external_manifest(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    for name in ("external_manifest.at_launch.json", "external_manifest.at_end.json"):
        (runroot / name).write_text(external_manifest + "\n", encoding="utf-8")
    write_fixture_arm(
        repo,
        runroot,
        sidecar_dir,
        tag,
        hydra=False,
        concurrency=concurrency,
    )
    write_fixture_arm(
        repo,
        runroot,
        sidecar_dir,
        tag,
        hydra=True,
        concurrency=concurrency,
    )
    return runroot, sidecar_dir, tag


def expect_gate_error(callable_obj: Any, needle: str) -> None:
    try:
        callable_obj()
    except GateError as error:
        if needle not in str(error):
            raise AssertionError(
                f"expected error containing {needle!r}, got {error!r}"
            ) from error
    else:
        raise AssertionError(f"expected GateError containing {needle!r}")


def self_test(repo: Path) -> None:
    for task_count in (4, 16):
        validate_subset(
            (repo / EVIDENCE_SETS[task_count]["relative_path"]).resolve(),
            task_count,
        )
    assert cluster_summary([float(value) for value in range(16)])["df"] == 15
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        b1_root, b1_sidecars, b1_tag = write_fixture_campaign(repo, base, concurrency=1)
        b1 = reduce_campaign(
            repo,
            b1_root,
            b1_tag,
            4,
            1,
            b1_sidecars,
            BOOTSTRAP_REPS,
            BOOTSTRAP_SEED,
        )
        assert b1["analysis_valid"]
        assert b1["schema"] == "fr13.canonical_swe_verified_fixed32_floor_gate.v8"
        assert b1["inferred_concurrency"] == 1
        assert b1["gates"]["fixed32_flush_generation_chain_exact"]
        assert b1["gates"]["fixed32_task_boundaries_exact"]
        assert b1["gates"]["external_artifact_fingerprint_equal"]
        assert b1["gates"]["arm_runtime_attestations_equal"]
        assert b1["gates"]["running_container_image_identity_exact"]
        assert b1["gates"]["all_canonical_tasks_have_real_model_traffic"]
        assert b1["gates"]["all_task_agents_completed_cleanly"]
        assert b1["gates"]["all_tasks_have_terminal_swe_verdicts"]
        assert b1["gates"]["task_metric_bracket_bytes_bound"]
        assert b1["gates"]["fixed32_pretask_zero_positive_traffic"]
        assert b1["gates"]["all_positive_usage_proxy_pairs_task_bound"]
        assert b1["gates"]["canonical_exact_4_or_16_task_binding"]
        assert b1["gates"]["canonical_task_forward_union_covers_complete_stream"]
        assert b1["gates"]["fixed32_per_batch_physical_work_equal"]
        assert b1["gates"]["fixed32_drafter_graph_lifecycle_exact_and_matched"]
        assert b1["gates"]["fixed32_forward_graph_pregather_exact"]
        assert b1["gates"]["fixed32_scope_limitations_explicit"]
        assert b1["external_artifact_fingerprint"]["byte_equal"]
        assert b1["matched_runtime_attestation"]["byte_equal"]
        assert b1["source_runtime_fingerprint"]["file_count"] == 61
        assert b1["source_runtime_fingerprint"]["python_package_file_count"] == 25
        tail_metric_brackets = b1["arms"]["tail6_fixed32"]["provenance"][
            "task_metric_brackets"
        ]
        assert set(tail_metric_brackets) == set(EVIDENCE_SETS[4]["task_ids"])
        for task_bracket in tail_metric_brackets.values():
            for artifact in task_bracket.values():
                artifact_bytes = Path(artifact["path"]).read_bytes()
                assert artifact["bytes"] == len(artifact_bytes)
                assert artifact["sha256"] == hashlib.sha256(
                    artifact_bytes
                ).hexdigest()
        tail_real_tasks = b1["arms"]["tail6_fixed32"]["provenance"][
            "real_tasks"
        ]["tasks"]
        for task_record in tail_real_tasks.values():
            trace_bytes = Path(task_record["trace_path"]).read_bytes()
            assert task_record["trace_bytes"] == len(trace_bytes)
            assert task_record["trace_sha256"] == hashlib.sha256(
                trace_bytes
            ).hexdigest()
            for pair_artifact in task_record["proxy_pairs"]:
                pair_bytes = Path(pair_artifact["path"]).read_bytes()
                assert pair_artifact["bytes"] == len(pair_bytes)
                assert pair_artifact["sha256"] == hashlib.sha256(
                    pair_bytes
                ).hexdigest()
        b1_census_expected = b1["arms"]["tail6_fixed32"]["work_census_expected"]
        assert b1_census_expected["canonical_task_selection"]["counter_intervals"] == [
            [0, 352]
        ]
        assert b1_census_expected["canonical_task_selection"]["event_count"] == 352
        assert b1_census_expected["complete_stream"]["event_count"] == 352
        b1_work_census = b1["fixed32_work_census"]
        assert b1_work_census["report"]["schema"] == WORK_CENSUS_REPORT_SCHEMA
        assert b1_work_census["physical_work_comparison"][
            "observed_batch_sizes"
        ] == [1]
        assert (
            b1_work_census["physical_work_comparison"]["event_counts_compared"]
            is False
        )
        assert b1_work_census["drafter_graph_lifecycle"][
            "registry_batch_sizes"
        ] == [1]
        assert b1_work_census["forward_graph_pregather_lifecycle"][
            "registry_batch_sizes"
        ] == [1]
        assert b1_work_census["forward_graph_pregather_lifecycle"][
            "graph_signatures_equal_across_arms_per_batch"
        ]
        assert len(b1_work_census["scope"]["data_dependent_unproven"]) == 6
        for census_artifact in b1_work_census["files"].values():
            census_bytes = Path(census_artifact["path"]).read_bytes()
            assert census_artifact["bytes"] == len(census_bytes)
            assert census_artifact["sha256"] == hashlib.sha256(
                census_bytes
            ).hexdigest()

        bad_physical_report = json.loads(
            json.dumps(b1_work_census["report"])
        )
        bad_physical_entry = bad_physical_report[
            "physical_work_histograms"
        ]["tail6_fixed32"]["1"]
        bad_physical_entry["normalized_event_signatures"] = {
            "0" * 64: bad_physical_entry["event_count"]
        }
        expect_gate_error(
            lambda: validate_work_census_v5_report(
                bad_physical_report,
                required_batch=1,
            ),
            "normalized physical-work SHA differs",
        )

        bad_lifecycle_report = json.loads(
            json.dumps(b1_work_census["report"])
        )
        bad_registry_row = bad_lifecycle_report[
            "drafter_graph_registries"
        ]["tail6_fixed32"][0]
        bad_terminal_row = bad_lifecycle_report["terminal_summaries"][
            "tail6_fixed32"
        ]["drafter_graph_registry"][0]
        new_origin = (
            "unmeasured"
            if bad_registry_row["capture_origin"] == "measured"
            else "measured"
        )
        bad_registry_row["capture_origin"] = new_origin
        bad_terminal_row["capture_origin"] = new_origin
        expect_gate_error(
            lambda: validate_work_census_v5_report(
                bad_lifecycle_report,
                required_batch=1,
            ),
            "drafter graph lifecycle differs",
        )

        bad_scope_report = json.loads(json.dumps(b1_work_census["report"]))
        bad_scope_report["scope"]["data_dependent_unproven"].pop()
        expect_gate_error(
            lambda: validate_work_census_v5_report(
                bad_scope_report,
                required_batch=1,
            ),
            "v5 report contract mismatch",
        )

        bad_forward_report = json.loads(json.dumps(b1_work_census["report"]))
        for parent in (
            bad_forward_report["forward_graph_registries"]["tail6_fixed32"],
            bad_forward_report["terminal_summaries"]["tail6_fixed32"][
                "forward_graph_registry"
            ],
        ):
            parent[0]["measured_replays"] += 1
        expect_gate_error(
            lambda: validate_work_census_v5_report(
                bad_forward_report,
                required_batch=1,
            ),
            "does not prove one ordered final-FULL pregather capture",
        )

        bad_auxiliary_report = json.loads(
            json.dumps(b1_work_census["report"])
        )
        for parent in (
            bad_auxiliary_report["conv_pregather_auxiliary"][
                "tail6_fixed32"
            ],
            bad_auxiliary_report["terminal_summaries"]["tail6_fixed32"][
                "conv_pregather_auxiliary"
            ],
        ):
            parent["profile_capture_stages"] = 1
        expect_gate_error(
            lambda: validate_work_census_v5_report(
                bad_auxiliary_report,
                required_batch=1,
            ),
            "pregather auxiliary/host stage counts are not zero",
        )
        tail_b1 = b1["arms"]["tail6_fixed32"]["statistics"]
        equal = tail_b1["task_cluster_equal_weight"]["wall_ms_per_step"]
        assert equal["cluster_count"] == 4 and equal["df"] == 3
        assert math.isclose(equal["point_estimate"], 106.0)
        assert not math.isclose(
            equal["point_estimate"],
            tail_b1["step_weighted_counter_point"]["wall_ms_per_step"],
        )
        assert all(
            row["counter_reconciliation"]["wall"]["exact_drafts_and_steps"]
            for row in tail_b1["sidecar_coverage_by_task"]
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS - 1,
                BOOTSTRAP_SEED,
            ),
            "formal moving-block parameters are pinned",
        )

        tail_arm = b1_root / f"tail6_fixed32_{b1_tag}"
        first_task_dir = (
            tail_arm / "swe_out" / "verified" / "per_task" / CANONICAL_TASK_IDS[0]
        )
        boundary_path = first_task_dir / "fixed32_task_boundary.json"
        metadata_path = first_task_dir / "runner_metadata.json"
        good_boundary_bytes = boundary_path.read_bytes()
        good_metadata_bytes = metadata_path.read_bytes()
        bad_boundary = json.loads(good_boundary_bytes)
        bad_boundary["post"]["generation"] += 100
        bad_metadata = json.loads(good_metadata_bytes)
        bad_metadata["fixed32_task_boundary"] = bad_boundary
        boundary_path.write_text(
            json.dumps(bad_boundary, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metadata_path.write_text(
            json.dumps(bad_metadata, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "missing required artifact",
        )
        boundary_path.write_bytes(good_boundary_bytes)
        metadata_path.write_bytes(good_metadata_bytes)

        final_result_path = tail_arm / "fixed32_final_flush.json"
        final_ack_path = tail_arm / "logs" / "fr13_fixed32_flush_ack.json"
        good_final_result_bytes = final_result_path.read_bytes()
        good_final_ack_bytes = final_ack_path.read_bytes()
        bad_final_result = json.loads(good_final_result_bytes)
        bad_final_ack = bad_final_result["ack"]
        bad_final_ack["counters"]["pure_decode_forward_steps"] += 1
        bad_final_ack["counters"]["complete_work_census_events"] += 1
        final_result_path.write_text(
            json.dumps(bad_final_result, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        final_ack_path.write_text(
            json.dumps(bad_final_ack, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "runtime snapshot does not bind to flush ack",
        )
        final_result_path.write_bytes(good_final_result_bytes)
        final_ack_path.write_bytes(good_final_ack_bytes)

        final_result = json.loads(good_final_result_bytes)
        final_ack = final_result["ack"]
        final_runtime_snapshot_path = (
            tail_arm
            / "logs"
            / (
                "fr13_fixed32_boundary_snapshot."
                f"{final_ack['generation']}.json"
            )
        )
        good_runtime_snapshot = json.loads(
            final_runtime_snapshot_path.read_bytes()
        )
        pregather_tamper_path = base / "pregather_boundary_tamper.json"

        def expect_pregather_boundary_failure(
            label: str,
            mutate: Any,
            needle: str = (
                "committer/in-graph pregather counters do not reconcile"
            ),
        ) -> None:
            tampered = json.loads(json.dumps(good_runtime_snapshot))
            mutate(tampered["metrics"]["conv_pregather"])
            pregather_tamper_path.write_text(
                json.dumps(tampered, ensure_ascii=True, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            expect_gate_error(
                lambda: validate_runtime_boundary_snapshot(
                    pregather_tamper_path,
                    ack=final_ack,
                    server_capacity=1,
                    metrics_path=None,
                    metric_values=None,
                    reference=None,
                    census_path=(
                        tail_arm
                        / "logs"
                        / "fr13_fixed32_work_census.jsonl"
                    ),
                ),
                needle,
            )

        pregather_boundary_tampers = (
            (
                "capture-scalar",
                lambda counters: counters.__setitem__(
                    "graph_capture_stages",
                    counters["graph_capture_stages"] + 1,
                ),
            ),
            (
                "capture-histogram",
                lambda counters: counters[
                    "graph_capture_stages_by_batch"
                ].__setitem__("1", 0),
            ),
            (
                "profile-stage",
                lambda counters: counters.__setitem__(
                    "profile_capture_stages", 1
                ),
            ),
            (
                "aux-stage",
                lambda counters: counters.__setitem__(
                    "aux_capture_stages", 1
                ),
            ),
            (
                "host-stage-scalar",
                lambda counters: counters.__setitem__("actual_stages", 1),
            ),
            (
                "host-stage-histogram",
                lambda counters: counters[
                    "actual_stages_by_batch"
                ].__setitem__("1", 1),
            ),
            (
                "replay-scalar",
                lambda counters: counters.__setitem__(
                    "graph_replay_stages",
                    counters["graph_replay_stages"] + 1,
                ),
            ),
            (
                "replay-histogram",
                lambda counters: counters[
                    "graph_replay_stages_by_batch"
                ].__setitem__(
                    "1",
                    counters["graph_replay_stages_by_batch"]["1"] + 1,
                ),
            ),
        )
        for label, mutate in pregather_boundary_tampers:
            expect_pregather_boundary_failure(label, mutate)
        expect_pregather_boundary_failure(
            "legacy-stage-key",
            lambda counters: counters.__setitem__("stage_launches", 0),
            "conv_pregather: keys mismatch",
        )

        tail_metric_paths = sorted(
            (tail_arm / "swe_out" / "verified" / "per_task").glob(
                "*/vllm_metrics_*.txt"
            )
        )
        original_metric_bytes = {path: path.read_bytes() for path in tail_metric_paths}
        runtime_snapshot_paths = sorted(
            (tail_arm / "logs").glob(
                "fr13_fixed32_boundary_snapshot.*.json"
            )
        )
        original_runtime_snapshot_bytes = {
            path: path.read_bytes() for path in runtime_snapshot_paths
        }
        for snapshot_path in runtime_snapshot_paths:
            runtime_snapshot = json.loads(snapshot_path.read_bytes())
            runtime_snapshot["metrics"]["sfwd"]["wall_rejected"] += 5
            snapshot_path.write_text(
                json.dumps(
                    runtime_snapshot,
                    ensure_ascii=True,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        boundary_paths = sorted(
            (tail_arm / "swe_out" / "verified" / "per_task").glob(
                "*/fixed32_task_boundary.json"
            )
        )
        original_boundary_bytes = {
            path: path.read_bytes() for path in boundary_paths
        }
        metadata_paths = [path.with_name("runner_metadata.json") for path in boundary_paths]
        original_metadata_bytes = {
            path: path.read_bytes() for path in metadata_paths
        }
        for boundary_path in boundary_paths:
            boundary = json.loads(boundary_path.read_bytes())
            for snapshot in ("pre", "post"):
                generation = boundary[snapshot]["generation"]
                snapshot_path = (
                    tail_arm
                    / "logs"
                    / f"fr13_fixed32_boundary_snapshot.{generation}.json"
                )
                boundary[f"{snapshot}_runtime_snapshot"]["sha256"] = (
                    sha256_file(snapshot_path)
                )
            boundary_path.write_text(
                json.dumps(boundary, ensure_ascii=True, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            metadata_path = boundary_path.with_name("runner_metadata.json")
            metadata = json.loads(metadata_path.read_bytes())
            metadata["fixed32_task_boundary"] = boundary
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=True, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        for metric_path in tail_metric_paths:
            values, _ = metric_snapshot(metric_path)
            metric_path.write_text(
                replace_metric_values(
                    metric_path.read_text(encoding="utf-8"),
                    {
                        "wall_attempts": values["wall_attempts"] + 5,
                        "wall_rejected": values["wall_rejected"] + 5,
                    },
                ),
                encoding="utf-8",
            )
        tail_main_path = next(
            path
            for path in b1_sidecars.iterdir()
            if path.name.startswith(f"tail6_fixed32_{b1_tag}.json.")
            and ".json.samples." not in path.name
        )
        original_main_bytes = tail_main_path.read_bytes()
        warmup_main = json.loads(original_main_bytes)
        warmup_main["n_wall_rejected"] += 5
        tail_main_path.write_text(json.dumps(warmup_main), encoding="utf-8")
        warmup_baseline = reduce_campaign(
            repo,
            b1_root,
            b1_tag,
            4,
            1,
            b1_sidecars,
            BOOTSTRAP_REPS,
            BOOTSTRAP_SEED,
        )
        assert (
            warmup_baseline["arms"]["tail6_fixed32"]["sidecar"]["main"]["counters"][
                "wall_rejected"
            ]
            == 5
        )
        for metric_path, payload in original_metric_bytes.items():
            metric_path.write_bytes(payload)
        for snapshot_path, payload in original_runtime_snapshot_bytes.items():
            snapshot_path.write_bytes(payload)
        for boundary_path, payload in original_boundary_bytes.items():
            boundary_path.write_bytes(payload)
        for metadata_path, payload in original_metadata_bytes.items():
            metadata_path.write_bytes(payload)
        tail_main_path.write_bytes(original_main_bytes)

        first_task = (
            tail_arm / "swe_out" / "verified" / "per_task" / CANONICAL_TASK_IDS[0]
        )
        pre_path = first_task / "vllm_metrics_pre.txt"
        pre_values, _ = metric_snapshot(pre_path)
        good_pre = pre_path.read_text(encoding="utf-8")
        post_path = first_task / "vllm_metrics_post.txt"
        good_post = post_path.read_text(encoding="utf-8")

        pre_path.write_text(
            replace_metric_values(
                good_pre,
                {"wall_attempts": pre_values["wall_attempts"] + 1},
            ),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "pre wall attempts != retained + rejected",
        )
        pre_path.write_text(good_pre, encoding="utf-8")
        wall_line = next(
            line
            for line in good_post.splitlines()
            if line.startswith(f"{METRICS['wall_s']} ")
        )
        zero_wall_line = f"{METRICS['wall_s']} {pre_values['wall_s']}"
        post_path.write_text(
            good_post.replace(wall_line, zero_wall_line, 1),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "non-positive wall_s delta",
        )
        post_path.write_text(good_post, encoding="utf-8")

        post_values, _ = metric_snapshot(post_path)
        post_path.write_text(
            replace_metric_values(
                good_post,
                {
                    "wall_attempts": post_values["wall_attempts"] + 1,
                    "wall_rejected": post_values["wall_rejected"] + 1,
                },
            ),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "censored wall intervals in task window",
        )
        post_path.write_text(good_post, encoding="utf-8")

        post_path.write_text(
            replace_metric_values(
                good_post,
                {"wall_attempts": post_values["wall_attempts"] + 1},
            ),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "wall attempts != retained + rejected",
        )
        post_path.write_text(good_post, encoding="utf-8")

        one_step_post = {
            "fwd_s": pre_values["fwd_s"] + 0.070,
            "fwd_steps": pre_values["fwd_steps"] + 1,
            "fwd_drafts": pre_values["fwd_drafts"] + 1,
            "wall_s": pre_values["wall_s"] + 0.100,
            "wall_drafts": pre_values["wall_drafts"] + 1,
            "wall_steps": pre_values["wall_steps"] + 1,
            "wall_attempts": pre_values["wall_attempts"] + 1,
            "wall_rejected": pre_values["wall_rejected"],
            "spec_drafts": pre_values["spec_drafts"] + 1,
            "spec_tokens": pre_values["spec_tokens"] + PHYSICAL_DRAFTS,
        }
        post_path.write_text(
            replace_metric_values(good_post, one_step_post),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "exposure below 64 retained steps",
        )
        post_path.write_text(good_post, encoding="utf-8")

        post_values, _ = metric_snapshot(post_path)
        fwd_drafts_line = next(
            line
            for line in good_post.splitlines()
            if line.startswith(f"{METRICS['fwd_drafts']} ")
        )
        post_path.write_text(
            good_post.replace(
                fwd_drafts_line,
                f"{METRICS['fwd_drafts']} {post_values['fwd_drafts'] + 1}",
                1,
            ),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "fwd drafts/step is outside [1, 1]",
        )
        post_path.write_text(good_post, encoding="utf-8")

        post_path.write_text(
            good_post.replace(
                'model_name="qwen3.6-27b"',
                'model_name="wrong-series"',
                1,
            ),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "pre/post required metric labels differ",
        )
        post_path.write_text(good_post, encoding="utf-8")

        wrong_series_pre = good_pre.replace(
            'model_name="qwen3.6-27b"',
            'model_name="wrong-series"',
        )
        wrong_series_post = good_post.replace(
            'model_name="qwen3.6-27b"',
            'model_name="wrong-series"',
        )
        pre_path.write_text(wrong_series_pre, encoding="utf-8")
        post_path.write_text(wrong_series_post, encoding="utf-8")
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "pinned qwen3.6-27b series",
        )
        pre_path.write_text(good_pre, encoding="utf-8")
        post_path.write_text(good_post, encoding="utf-8")

        tail_sidecar = next(
            path
            for path in b1_sidecars.iterdir()
            if path.name.startswith(f"tail6_fixed32_{b1_tag}.json.samples.")
        )
        good_sidecar = tail_sidecar.read_bytes()
        sidecar_payload = json.loads(good_sidecar)
        sidecar_payload["final"] = False
        tail_sidecar.write_text(json.dumps(sidecar_payload), encoding="utf-8")
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "per-step sidecar lacks an explicit final flush",
        )
        tail_sidecar.write_bytes(good_sidecar)

        sidecar_payload = json.loads(good_sidecar)
        sidecar_payload["wall_ms"][5] += 1.0
        tail_sidecar.write_text(json.dumps(sidecar_payload), encoding="utf-8")
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "sidecar/counter timing mismatch",
        )
        tail_sidecar.write_bytes(good_sidecar)

        main_sidecar = next(
            path
            for path in b1_sidecars.iterdir()
            if path.name.startswith(f"tail6_fixed32_{b1_tag}.json.")
            and ".json.samples." not in path.name
        )
        good_main_sidecar = main_sidecar.read_bytes()
        main_payload = json.loads(good_main_sidecar)
        main_payload["n_wall_rejected"] = 1_000_000
        main_sidecar.write_text(json.dumps(main_payload), encoding="utf-8")
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "final task bracket counters do not match",
        )
        main_sidecar.write_bytes(good_main_sidecar)

        sidecar_payload = json.loads(good_sidecar)
        for key in (
            "fwd_drafts",
            "fwd_ms",
            "fwd_cg",
            "fwd_host_ms",
            "fwd_exec_ms",
            "fwd_cpu_tail_ms",
            "wall_drafts",
            "wall_ms",
        ):
            sidecar_payload[key].pop()
        tail_sidecar.write_text(json.dumps(sidecar_payload), encoding="utf-8")
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "final sample lengths do not match main counters",
        )
        tail_sidecar.write_bytes(good_sidecar)

        census_path = tail_arm / "logs" / "fr13_fixed32_work_census.jsonl"
        good_census = census_path.read_bytes()
        census_lines = good_census.decode("utf-8").splitlines()
        census_path.write_text(
            "\n".join(census_lines[:-1]) + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "census stream is shorter than snapshot prefix",
        )
        census_path.write_bytes(good_census)

        census_records = [
            json.loads(line) for line in good_census.decode("utf-8").splitlines()
        ]
        census_records[0]["gdn"]["padded_slots"] -= 1
        census_path.write_text(
            "\n".join(
                json.dumps(record, ensure_ascii=True, sort_keys=True)
                for record in census_records
            )
            + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "census prefix digest mismatch",
        )
        census_path.write_bytes(good_census)

        launch_manifest = b1_root / "runtime_manifest.at_launch.json"
        good_manifest = launch_manifest.read_bytes()
        tampered_manifest = json.loads(good_manifest)
        tampered_manifest["closures"]["host_script_source"][0]["sha256"] = "0" * 64
        launch_manifest.write_text(
            json.dumps(tampered_manifest, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "runtime manifest canonical digest mismatch",
        )
        launch_manifest.write_bytes(good_manifest)

        def rerun_b1_fixture() -> dict[str, Any]:
            return reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            )

        bound_pre = tail_metric_brackets[CANONICAL_TASK_IDS[0]]["pre"]
        bound_pre_path = Path(bound_pre["path"])
        good_bound_pre = bound_pre_path.read_bytes()
        bound_pre_path.write_bytes(good_bound_pre + b"\n")
        rebound_b1 = rerun_b1_fixture()
        rebound_pre = rebound_b1["arms"]["tail6_fixed32"]["provenance"][
            "task_metric_brackets"
        ][CANONICAL_TASK_IDS[0]]["pre"]
        assert rebound_pre["sha256"] != bound_pre["sha256"]
        assert rebound_pre["bytes"] == bound_pre["bytes"] + 1
        assert rebound_pre["sha256"] == hashlib.sha256(
            good_bound_pre + b"\n"
        ).hexdigest()
        bound_pre_path.write_bytes(good_bound_pre)

        external_launch_path = b1_root / "external_manifest.at_launch.json"
        good_external_launch = external_launch_path.read_bytes()
        bad_external = json.loads(good_external_launch)
        bad_external["forked_fa2"]["sha256"] = "0" * 64
        external_launch_path.write_text(
            json.dumps(bad_external, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "external manifest canonical digest mismatch",
        )
        external_launch_path.write_bytes(good_external_launch)

        external_end_path = b1_root / "external_manifest.at_end.json"
        good_external_end = external_end_path.read_bytes()
        different_external = json.loads(good_external_end)
        external_end_path.write_text(
            json.dumps(
                different_external,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "external manifest changed during the campaign",
        )
        external_end_path.write_bytes(good_external_end)

        hydra_arm = b1_root / f"hydra27_fixed32_{b1_tag}"
        hydra_attestation_path = (
            hydra_arm / "logs" / "fr13_fixed32_runtime_attestation.json"
        )
        good_hydra_attestation = hydra_attestation_path.read_bytes()
        different_attestation = json.loads(good_hydra_attestation)
        different_attestation["python"]["version"] = "3.12.4"
        different_attestation.pop("overall_canonical_sha256")
        different_attestation["overall_canonical_sha256"] = canonical_json_sha256(
            different_attestation
        )
        hydra_attestation_path.write_text(
            json.dumps(
                different_attestation,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "runtime attestations differ",
        )
        hydra_attestation_path.write_bytes(good_hydra_attestation)

        tail_attestation_path = (
            tail_arm / "logs" / "fr13_fixed32_runtime_attestation.json"
        )
        good_tail_attestation = tail_attestation_path.read_bytes()
        wrong_fa2_path_attestation = json.loads(good_tail_attestation)
        wrong_fa2_path_attestation["forked_fa2"]["source"]["path"] = "/tmp/wrong.so"
        wrong_fa2_path_attestation.pop("overall_canonical_sha256")
        wrong_fa2_path_attestation["overall_canonical_sha256"] = (
            canonical_json_sha256(wrong_fa2_path_attestation)
        )
        tail_attestation_path.write_text(
            json.dumps(
                wrong_fa2_path_attestation,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "runtime FA2 paths differ",
        )
        tail_attestation_path.write_bytes(good_tail_attestation)

        container_identity_path = tail_arm / "fixed32_container_identity.json"
        good_container_identity = container_identity_path.read_bytes()
        wrong_container_identity = json.loads(good_container_identity)
        wrong_container_identity["image_id"] = "sha256:" + "0" * 64
        container_identity_path.write_text(
            json.dumps(
                wrong_container_identity,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "running container identity differs",
        )
        container_identity_path.write_bytes(good_container_identity)

        process_path = tail_arm / "fixed32_process_identity.json"
        good_process_identity = process_path.read_bytes()
        wrong_pid1 = json.loads(good_process_identity)
        wrong_pid1["pid1"]["argv"].append("--unexpected")
        process_path.write_text(
            json.dumps(wrong_pid1, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "PID1 argv differs from the exact fixed32 contract",
        )
        process_path.write_bytes(good_process_identity)

        wrong_process_map = json.loads(good_process_identity)
        wrong_process_map["engine_core"]["forked_fa2_maps"] = []
        process_path.write_text(
            json.dumps(wrong_process_map, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "EngineCore did not map the pinned forked FA2 binary",
        )
        process_path.write_bytes(good_process_identity)

        wrong_pid1_env = json.loads(good_process_identity)
        env_index = wrong_pid1_env["pid1"]["environ"].index(
            "FR13_COMMITTER_GRAPH=1"
        )
        wrong_pid1_env["pid1"]["environ"][env_index] = "FR13_COMMITTER_GRAPH=0"
        process_path.write_text(
            json.dumps(wrong_pid1_env, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "PID1 expected FR13_COMMITTER_GRAPH=1",
        )
        process_path.write_bytes(good_process_identity)

        trace_path = first_task_dir / "qwen_trace.jsonl"
        good_trace = trace_path.read_bytes()
        trace_path.write_text("{}\n", encoding="utf-8")
        expect_gate_error(
            rerun_b1_fixture,
            "trace identity does not match provenance",
        )
        trace_path.write_bytes(good_trace)

        task_metadata_path = first_task_dir / "runner_metadata.json"
        good_task_metadata = task_metadata_path.read_bytes()
        empty_model_trace = (
            json.dumps(
                {
                    "type": "message",
                    "role": "user",
                    "content": "No model output.",
                    "usage": {"input_tokens": 1},
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        trace_path.write_bytes(empty_model_trace)
        empty_trace_metadata = json.loads(good_task_metadata)
        empty_trace_provenance = empty_trace_metadata[
            "fixed32_real_task_provenance"
        ]
        empty_trace_provenance["trace_sha256"] = hashlib.sha256(
            empty_model_trace
        ).hexdigest()
        empty_trace_provenance["trace_bytes"] = len(empty_model_trace)
        task_metadata_path.write_text(
            json.dumps(empty_trace_metadata, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "trace content does not prove real model traffic",
        )
        trace_path.write_bytes(good_trace)
        task_metadata_path.write_bytes(good_task_metadata)

        failed_agent_metadata = json.loads(good_task_metadata)
        failed_agent_metadata["agent"]["network_drop"] = True
        failed_agent_metadata["codex"]["network_drop"] = True
        task_metadata_path.write_text(
            json.dumps(failed_agent_metadata, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "task agent did not complete cleanly",
        )
        task_metadata_path.write_bytes(good_task_metadata)

        pair_path = tail_arm / "proxy_pair_dumps" / "pair_000000_initial.json"
        good_pair = pair_path.read_bytes()
        pair_path.unlink()
        expect_gate_error(
            rerun_b1_fixture,
            "no task-bound positive-usage proxy pair",
        )
        pair_path.write_bytes(good_pair)
        zero_usage_pair = json.loads(good_pair)
        zero_usage_pair["response"]["usage"] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        pair_path.write_text(
            json.dumps(
                zero_usage_pair,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "task-bound proxy response has no positive token usage",
        )
        pair_path.write_bytes(good_pair)

        unmatched_pair_path = (
            tail_arm / "proxy_pair_dumps" / "pair_999999_unmatched.json"
        )
        unmatched_pair = json.loads(good_pair)
        unmatched_pair["request"]["input"] = "unbound generation request"
        unmatched_pair_path.write_text(
            json.dumps(
                unmatched_pair,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "unmatched proxy response has positive token usage",
        )
        unmatched_pair_path.unlink()

        traffic_audit_path = tail_arm / "fixed32_positive_traffic_audit.json"
        good_traffic_audit = traffic_audit_path.read_bytes()
        bad_traffic_audit = json.loads(good_traffic_audit)
        bad_traffic_audit["positive_pair_count"] += 1
        traffic_audit_path.write_text(
            json.dumps(
                bad_traffic_audit,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "positive-traffic audit does not match proxy pair bytes",
        )
        traffic_audit_path.write_bytes(good_traffic_audit)

        pretask_metrics_path = tail_arm / "metrics_before_swe.txt"
        pretask_marker_path = tail_arm / "fixed32_pretask_zero_traffic.json"
        good_pretask_metrics = pretask_metrics_path.read_bytes()
        good_pretask_marker = pretask_marker_path.read_bytes()
        bad_pretask_metrics = replace_metric_values(
            good_pretask_metrics.decode("utf-8"),
            {"spec_drafts": 1.0, "spec_tokens": float(PHYSICAL_DRAFTS)},
        )
        pretask_metrics_path.write_text(bad_pretask_metrics, encoding="utf-8")
        rebound_pretask_marker = json.loads(good_pretask_marker)
        rebound_pretask_marker["metrics"]["sha256"] = sha256_file(
            pretask_metrics_path
        )
        pretask_marker_path.write_text(
            json.dumps(
                rebound_pretask_marker,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "pretask decode metrics are not exact zero",
        )
        pretask_metrics_path.write_bytes(good_pretask_metrics)
        pretask_marker_path.write_bytes(good_pretask_marker)

        bad_non_spec_metrics = replace_metric_values(
            good_pretask_metrics.decode("utf-8"),
            {"fwd_s": 0.001, "fwd_steps": 1.0},
        )
        pretask_metrics_path.write_text(
            bad_non_spec_metrics,
            encoding="utf-8",
        )
        rebound_pretask_marker = json.loads(good_pretask_marker)
        rebound_pretask_marker["metrics"]["sha256"] = sha256_file(
            pretask_metrics_path
        )
        pretask_marker_path.write_text(
            json.dumps(
                rebound_pretask_marker,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            rerun_b1_fixture,
            "pretask decode metrics are not exact zero",
        )
        pretask_metrics_path.write_bytes(good_pretask_metrics)
        pretask_marker_path.write_bytes(good_pretask_marker)

        forbidden_probe_path = tail_arm / "warmup_probe.json"
        forbidden_probe_path.write_text("{}\n", encoding="utf-8")
        expect_gate_error(
            rerun_b1_fixture,
            "fixed32 forbidden pretask probe artifacts exist",
        )
        forbidden_probe_path.unlink()

        b4_root, b4_sidecars, b4_tag = write_fixture_campaign(repo, base, concurrency=4)
        b4 = reduce_campaign(
            repo,
            b4_root,
            b4_tag,
            4,
            4,
            b4_sidecars,
            BOOTSTRAP_REPS,
            BOOTSTRAP_SEED,
        )
        b4_repeat = reduce_campaign(
            repo,
            b4_root,
            b4_tag,
            4,
            4,
            b4_sidecars,
            BOOTSTRAP_REPS,
            BOOTSTRAP_SEED,
        )
        assert json.dumps(b4, sort_keys=True, allow_nan=False) == json.dumps(
            b4_repeat, sort_keys=True, allow_nan=False
        )
        assert b4["analysis_valid"]
        tail_b4 = b4["arms"]["tail6_fixed32"]["statistics"]
        assert tail_b4["bracket_mode"].startswith("overlap-safe")
        assert tail_b4["union_intervals"]["wall"] == [0, 645]
        assert tail_b4["sidecar_coverage"]["wall"]["selected_steps"] == 645
        assert tail_b4["sidecar_coverage"]["wall"]["fraction"] == 1.0
        assert tail_b4["sidecar_counter_reconciliation"]["wall"][
            "exact_drafts_and_steps"
        ]
        b4_census_expected = b4["arms"]["tail6_fixed32"]["work_census_expected"]
        assert b4_census_expected["canonical_task_selection"]["counter_intervals"] == [
            [0, 645]
        ]
        assert b4_census_expected["canonical_task_selection"]["event_count"] == 645
        assert b4_census_expected["complete_stream"]["event_count"] == 645
        assert b4["fixed32_work_census"]["physical_work_comparison"][
            "observed_batch_sizes"
        ] == [3, 4]
        assert b4["fixed32_work_census"]["drafter_graph_lifecycle"][
            "registry_batch_sizes"
        ] == [3, 4]
        assert b4["fixed32_work_census"][
            "forward_graph_pregather_lifecycle"
        ]["registry_batch_sizes"] == [1, 2, 3, 4]
        naive_task_sum = sum(
            end - start for start, end in ((0, 500), (20, 620), (40, 640), (60, 645))
        )
        assert naive_task_sum > 645
        expect_gate_error(
            lambda: assert_nonoverlap(
                [(0, 500), (20, 620), (40, 640), (60, 645)],
                "synthetic B=4 task-sum",
            ),
            "counter intervals overlap",
        )
        blocks = tail_b4["moving_block_u95_sensitivity"]["blocks"]
        assert [row["block_steps"] for row in blocks] == list(BLOCK_SENSITIVITY)
        worst = tail_b4["moving_block_u95_sensitivity"][
            "worst_across_requested_blocks"
        ]["legacy_slo_excess_ms_u95"]
        assert worst == max(row["legacy_slo_excess_ms_u95"] for row in blocks)
        assert tail_b4["forward_wall_occupancy_sequence_equal"]
        assert (
            tail_b4["exact_b4_stratum"]["selected_steps"] >= MIN_B4_EXACT_EVENTS
        )
        assert tail_b4["exact_b4_stratum"]["gate"]["pass"]

        adversarial_steps = 10_240
        adversarial_drafts = np.asarray(
            [4.0 if index % 20 == 0 else 3.0 for index in range(adversarial_steps)]
        )
        adversarial_wall_ms = np.asarray(
            [
                1_000.0 if index % 20 == 0 else 60.0
                for index in range(adversarial_steps)
            ]
        )
        adversarial_fwd_ms = np.full(adversarial_steps, 70.0)
        adversarial_draft_total = float(adversarial_drafts.sum())
        adversarial_windows = [
            {
                "task_id": "synthetic-exact-b4-regression",
                "fwd_span": (0, adversarial_steps),
                "wall_span": (0, adversarial_steps),
                "pre": {
                    "fwd_steps": 0.0,
                    "fwd_s": 0.0,
                    "fwd_drafts": 0.0,
                    "wall_steps": 0.0,
                    "wall_s": 0.0,
                    "wall_drafts": 0.0,
                },
                "post": {
                    "fwd_steps": float(adversarial_steps),
                    "fwd_s": float(adversarial_fwd_ms.sum() / 1000.0),
                    "fwd_drafts": adversarial_draft_total,
                    "wall_steps": float(adversarial_steps),
                    "wall_s": float(adversarial_wall_ms.sum() / 1000.0),
                    "wall_drafts": adversarial_draft_total,
                },
            }
        ]
        adversarial_b4 = b4_arm_statistics(
            adversarial_windows,
            {
                "fwd_ms": adversarial_fwd_ms,
                "fwd_full": np.ones(adversarial_steps),
                "fwd_drafts": adversarial_drafts,
                "wall_ms": adversarial_wall_ms,
                "wall_drafts": adversarial_drafts.copy(),
            },
            PHYSICAL_DRAFTS,
            4,
            256,
            BOOTSTRAP_SEED,
        )
        assert adversarial_b4["gate"]["union_pass"]
        assert not adversarial_b4["gate"]["exact_b4_pass"]
        assert not adversarial_b4["gate"]["pass"]

        b4_tail_arm = b4_root / f"tail6_fixed32_{b4_tag}"
        b4_census_path = b4_tail_arm / "logs" / "fr13_fixed32_work_census.jsonl"
        good_b4_census = b4_census_path.read_bytes()
        b4_census_records = [
            json.loads(line) for line in good_b4_census.decode("utf-8").splitlines()
        ]
        b4_census_events = b4_census_records[:-1]
        for record in b4_census_events:
            record["forward_step_index"] += 1
        b4_census_records = [
            *b4_census_events,
            work_census_terminal_fixture(
                b4_census_events,
                fixture_synthetic_runtime_proof=True,
            ),
        ]
        b4_census_path.write_text(
            "\n".join(
                json.dumps(record, ensure_ascii=True, sort_keys=True)
                for record in b4_census_records
            )
            + "\n",
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b4_root,
                b4_tag,
                4,
                4,
                b4_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "census prefix digest mismatch",
        )
        b4_census_path.write_bytes(good_b4_census)

        hydra_arm = b1_root / f"hydra27_fixed32_{b1_tag}"
        env_path = hydra_arm / "container_env.txt"
        good_env = env_path.read_text(encoding="utf-8")
        env_path.write_text(
            good_env.replace(
                "FR13_FIXED32_MODE=hydra27_fixed32",
                "FR13_FIXED32_MODE=tail6_fixed32",
            ),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "expected exactly FR13_FIXED32_MODE=hydra27_fixed32",
        )
        env_path.write_text(good_env, encoding="utf-8")

        env_path.write_text(
            good_env.replace("FR13_COMMITTER_GRAPH=1", "FR13_COMMITTER_GRAPH=0"),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "expected exactly FR13_COMMITTER_GRAPH=1",
        )
        env_path.write_text(good_env, encoding="utf-8")

        env_path.write_text(
            good_env.replace("FR13_STEP_WALL_CAP_S=1.5", "FR13_STEP_WALL_CAP_S=0.1"),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "expected exactly FR13_STEP_WALL_CAP_S=1.5",
        )
        env_path.write_text(good_env, encoding="utf-8")

        runtime_path = hydra_arm / "docker_full.log"
        good_runtime = runtime_path.read_text(encoding="utf-8")
        runtime_path.write_text(
            good_runtime.replace(
                "levels=[1, 11] lens=[5, 7] critical=12",
                "levels=[1, 10] lens=[12, 7] critical=19",
            ),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "current runtime needle",
        )
        runtime_path.write_text(good_runtime, encoding="utf-8")

        corrupt_subset = base / "corrupt_subset.json"
        corrupt_subset.write_text('{"instance_ids":[]}\n', encoding="utf-8")
        runlog_path = b1_root / f"hydra27_fixed32_{b1_tag}.runlog"
        good_runlog = runlog_path.read_text(encoding="utf-8")
        canonical_path = str((repo / EVIDENCE_SETS[4]["relative_path"]).resolve())
        runlog_path.write_text(
            good_runlog.replace(canonical_path, str(corrupt_subset)),
            encoding="utf-8",
        )
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "canonical subset sha256 mismatch",
        )
        runlog_path.write_text(good_runlog, encoding="utf-8")

        original_sidecar = next(
            path
            for path in b1_sidecars.iterdir()
            if path.name.startswith(f"tail6_fixed32_{b1_tag}.json.samples.")
        )
        duplicate = b1_sidecars / f"tail6_fixed32_{b1_tag}.json.samples.999"
        duplicate.write_bytes(original_sidecar.read_bytes())
        expect_gate_error(
            lambda: reduce_campaign(
                repo,
                b1_root,
                b1_tag,
                4,
                1,
                b1_sidecars,
                BOOTSTRAP_REPS,
                BOOTSTRAP_SEED,
            ),
            "expected one per-step sidecar",
        )
    print("self-test OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reduce a completed matched Tail6-fixed32/Hydra27-fixed32 canonical "
            "SWE-Verified campaign without mutating campaign artifacts."
        )
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--runroot", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--task-count", type=int, choices=(4, 16))
    parser.add_argument(
        "--expect-concurrency",
        type=int,
        choices=(1, 4),
        help="optional assertion; concurrency is always inferred from each arm log",
    )
    parser.add_argument(
        "--sidecar-dir",
        type=Path,
        help="defaults to REPO/output/fr13_sfwd_sidecar",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    if args.self_test:
        self_test(repo)
        return 0
    missing = [
        name
        for name, value in (
            ("--runroot", args.runroot),
            ("--tag", args.tag),
            ("--task-count", args.task_count),
        )
        if value is None
    ]
    if missing:
        raise SystemExit("required arguments: " + ", ".join(missing))
    runroot = (
        args.runroot.resolve()
        if args.runroot.is_absolute()
        else (repo / args.runroot).resolve()
    )
    sidecar_dir = (
        args.sidecar_dir.resolve()
        if args.sidecar_dir is not None and args.sidecar_dir.is_absolute()
        else (repo / (args.sidecar_dir or Path("output/fr13_sfwd_sidecar"))).resolve()
    )
    try:
        report = reduce_campaign(
            repo,
            runroot,
            args.tag,
            args.task_count,
            args.expect_concurrency,
            sidecar_dir,
            args.bootstrap_reps,
            args.seed,
        )
    except GateError as error:
        report = {
            "schema": "fr13.canonical_swe_verified_floor_gate.v1",
            "analysis_valid": False,
            "gate_verdict": "NOT_EVALUATED_INVALID_INPUT",
            "repo": str(repo),
            "runroot": str(runroot),
            "tag": args.tag,
            "task_count": args.task_count,
            "error": str(error),
        }
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["gate_verdict"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())

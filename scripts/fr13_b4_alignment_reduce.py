#!/usr/bin/env python3
"""Align fr13 fixed32 timing runroots across batch sizes, offline.

A paired timing runroot records, per arm, the engine Prometheus bracket around
every task, the pure-decode work census, the ingress ledger and the agent
trace.  ``deploy_speed_fullwall.json`` reduces those to per-arm aggregates by
SUMMING the per-task Prometheus deltas.  That is correct only when the per-task
brackets are disjoint, which holds at concurrency 1 and fails at concurrency 4:
there every task opens its bracket at the same instant, so the brackets are
NESTED and the sum multiply-counts the shared prefix.

This reducer re-derives each arm's aggregates from the same recorded evidence
without that assumption:

  * it classifies the bracket topology (disjoint, nested, or staggered when a
    refilled pool overlaps its brackets without sharing an origin) from the
    recorded pre/post counter vectors and takes the arm delta accordingly --
    the sum, the widest bracket, or the envelope max(post)-min(pre),
  * it cross-checks the result against the arm's own work census (which is a
    whole-arm record and is topology-blind),
  * and it emits the batch-invariant per-request quantities -- prefill token
    share, pure-decode phase cost per event, per-request step throughput --
    plus a co-residency decomposition that attributes the gap between nominal
    concurrency and measured co-residency to task-end skew, agent think time
    and prefill-phase occupancy.

Read-only: it opens recorded artifacts and writes at most the ``--out`` report.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fr13.b4_alignment.reduce.v1"

_PROM_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)$")

# Counters read from the per-task engine brackets.  Every one of these is a
# monotone vLLM/fr13 counter, so a bracket delta is post - pre.
BRACKET_COUNTERS = (
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:generation_tokens_total",
    "vllm:request_prompt_tokens_count",
    "vllm:request_prefill_kv_computed_tokens_sum",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:fr13_decode_forward_gpu_seconds_total",
    "vllm:fr13_decode_forward_gpu_steps_total",
    "vllm:fr13_decode_forward_gpu_drafts_total",
    "vllm:fr13_drafter_gpu_seconds_total",
    "vllm:fr13_drafter_gpu_spans_total",
    "vllm:fr13_committer_gpu_seconds_total",
    "vllm:fr13_committer_gpu_spans_total",
    "vllm:fr13_decode_step_wall_seconds_total",
    "vllm:fr13_decode_step_wall_steps_total",
    "vllm:fr13_decode_step_wall_drafts_total",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_decode_time_seconds_sum",
    "vllm:request_queue_time_seconds_sum",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:iteration_tokens_total_count",
    "vllm:num_preemptions_total",
)

# The arm-level snapshots are scraped from the un-bridged engine endpoint, so
# they carry the GLOBAL spec-decode draft count (drafts on mixed prefill+decode
# steps included) that the bridged per-task snapshots replace with the matched
# pure-decode count.  The difference is the lever on prefill-phase occupancy.
GLOBAL_COUNTERS = ("vllm:spec_decode_num_drafts_total",)

# vllm:fr13_decode_forward_gpu_drafts_total counts one more event per request
# than vllm:spec_decode_num_drafts_total: the terminal verify of each request
# carries no surviving draft.  Concurrency-1 arms, where no decode event can
# land on a mixed step, pin the offset at exactly the request count.
DRAFT_COUNTER_OFFSET_PER_REQUEST = 1


class AlignmentError(RuntimeError):
    """The runroot cannot produce a topology-safe arm aggregate."""


def _parse_prometheus(path: Path) -> dict[str, float]:
    """Return {metric_name+labels: value} for one Prometheus text snapshot."""
    sample: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            match = _PROM_LINE.match(line.rstrip("\n"))
            if match is None:
                continue
            try:
                sample[match.group(1) + (match.group(2) or "")] = float(match.group(3))
            except ValueError:
                continue
    return sample


def _counter(sample: dict[str, float], name: str) -> float | None:
    """Sum every label set of one metric family; None when absent."""
    total: float | None = None
    for key, value in sample.items():
        if key.split("{", 1)[0] == name:
            total = value if total is None else total + value
    return total


def _require(value: float | None, name: str, where: str) -> float:
    if value is None:
        raise AlignmentError(f"{where}: missing counter {name}")
    return value


def _read_launcher_meta(path: Path) -> dict[str, str]:
    """Parse the ``key=value`` lines of a launcher_meta.txt."""
    meta: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or " " in line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        meta[key] = value
    return meta


def _task_dirs(arm_dir: Path) -> list[Path]:
    root = arm_dir / "swe_out" / "verified" / "per_task"
    if not root.is_dir():
        raise AlignmentError(f"{arm_dir.name}: no swe_out/verified/per_task")
    return sorted(p for p in root.iterdir() if p.is_dir())


def _staggered(records: list[dict[str, Any]], names: tuple[str, ...]) -> bool:
    """True when any two brackets OVERLAP on the monotonic engine counters.

    Engine counters never decrease, so bracket A closed at or before bracket B
    opened iff every counter satisfies post_A <= pre_B.  Two brackets therefore
    overlap unless one of them wholly precedes the other.  Tiled concurrency-1
    brackets touch (post_k == pre_k+1) but never overlap, so they stay disjoint
    and keep summing; a refilled pool admits a task while others are still in
    flight, which is exactly an overlap.
    """
    if len(records) < 2:
        return False
    usable = [
        n
        for n in names
        if all(
            r["pre"].get(n) is not None and r["post"].get(n) is not None
            for r in records
        )
    ]
    if not usable:
        return False
    for i, first in enumerate(records):
        for second in records[i + 1 :]:
            first_then_second = all(
                first["post"][n] <= second["pre"][n] for n in usable
            )
            second_then_first = all(
                second["post"][n] <= first["pre"][n] for n in usable
            )
            if not (first_then_second or second_then_first):
                return True
    return False


def _bracket(arm_dir: Path) -> dict[str, Any]:
    """Reduce the per-task engine brackets to one topology-safe arm delta.

    Disjoint brackets (concurrency 1) tile the arm, so the arm delta is their
    sum.  Nested brackets (concurrency > 1, every task opening at the same
    instant) all share one origin, so the arm delta is the widest bracket --
    the one whose post-snapshot is last -- and summing would multiply-count.

    STAGGERED brackets (a continuously refilled pool: more tasks than slots,
    each admitted as an earlier one finishes) are neither.  They open on
    DIFFERENT origins, so they are not nested, but they still OVERLAP, so they
    do not tile and summing multiply-counts every overlap.  Measured on the
    16-task refill runroot that is a 1.87x-4.00x inflation across every bracket
    counter.  For overlapping brackets on monotonic engine counters the arm
    window is the ENVELOPE -- max(post) - min(pre) -- which is the outer hull of
    the brackets and counts the arm exactly once.

    Topology is classified from the recorded evidence, never assumed from a
    concurrency argument: one shared origin is what nesting means, and pairwise
    bracket overlap is what staggering means.  A concurrency-1 arm has neither,
    so it still sums and stays byte-identical.
    """
    tasks = _task_dirs(arm_dir)
    names = BRACKET_COUNTERS + GLOBAL_COUNTERS
    records = []
    for task in tasks:
        pre_path = task / "vllm_metrics_pre.txt"
        post_path = task / "vllm_metrics_post.txt"
        if not pre_path.is_file() or not post_path.is_file():
            raise AlignmentError(f"{arm_dir.name}/{task.name}: missing metrics bracket")
        pre = _parse_prometheus(pre_path)
        post = _parse_prometheus(post_path)
        records.append(
            {
                "instance_id": task.name,
                "post_mtime": post_path.stat().st_mtime,
                "pre": {n: _counter(pre, n) for n in names},
                "post": {n: _counter(post, n) for n in names},
            }
        )
    if not records:
        raise AlignmentError(f"{arm_dir.name}: no per-task brackets")

    origins = {
        tuple(r["pre"].get(n) for n in BRACKET_COUNTERS) for r in records
    }
    nested = len(records) > 1 and len(origins) == 1
    staggered = not nested and _staggered(records, BRACKET_COUNTERS)
    closing = max(records, key=lambda r: r["post_mtime"])

    delta: dict[str, float | None] = {}
    for name in names:
        if nested:
            pre = closing["pre"].get(name)
            post = closing["post"].get(name)
            delta[name] = None if pre is None or post is None else post - pre
        elif staggered:
            values = [(r["post"].get(name), r["pre"].get(name)) for r in records]
            if any(post is None or pre is None for post, pre in values):
                delta[name] = None
            else:
                delta[name] = max(p for p, _ in values) - min(q for _, q in values)
        else:
            values = [(r["post"].get(name), r["pre"].get(name)) for r in records]
            if any(post is None or pre is None for post, pre in values):
                delta[name] = None
            else:
                delta[name] = sum(post - pre for post, pre in values)

    naive = {}
    for name in names:
        values = [(r["post"].get(name), r["pre"].get(name)) for r in records]
        if not any(post is None or pre is None for post, pre in values):
            naive[name] = sum(post - pre for post, pre in values)

    if nested:
        topology = "nested"
    elif staggered:
        topology = "staggered"
    else:
        topology = "disjoint"

    return {
        "topology": topology,
        "closing_task": closing["instance_id"],
        "task_count": len(records),
        "distinct_bracket_origins": len(origins),
        "delta": delta,
        "summed_delta": naive,
    }


def _global_delta(arm_dir: Path) -> dict[str, float | None]:
    """Delta of the un-bridged arm-level snapshots (global draft counter)."""
    before = arm_dir / "metrics_before_swe.txt"
    after = arm_dir / "metrics_after_swe.txt"
    if not before.is_file() or not after.is_file():
        return {name: None for name in GLOBAL_COUNTERS}
    pre = _parse_prometheus(before)
    post = _parse_prometheus(after)
    out: dict[str, float | None] = {}
    for name in GLOBAL_COUNTERS:
        start = _counter(pre, name)
        end = _counter(post, name)
        out[name] = None if start is None or end is None else end - start
    return out


def _census(arm_dir: Path) -> dict[str, Any]:
    """Reduce the whole-arm pure-decode work census to step/event counts."""
    path = arm_dir / "logs" / "fr13_fixed32_work_census.jsonl"
    if not path.is_file():
        raise AlignmentError(f"{arm_dir.name}: no logs/fr13_fixed32_work_census.jsonl")
    batches: collections.Counter[int] = collections.Counter()
    steps = 0
    signatures: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("event_complete"):
                continue
            size = record.get("batch_size")
            if not isinstance(size, int) or size < 1:
                raise AlignmentError(f"{arm_dir.name}: census event without batch_size")
            purity = record.get("batch_purity") or {}
            if purity.get("mixed_pseudo_rows"):
                raise AlignmentError(f"{arm_dir.name}: census event with mixed rows")
            batches[size] += 1
            steps += 1
            tree = record.get("tree_attn") or {}
            digest = tree.get("physical_parent_digest")
            if isinstance(digest, str):
                signatures.add(digest)
    if steps == 0:
        raise AlignmentError(f"{arm_dir.name}: empty work census")
    events = sum(size * count for size, count in batches.items())
    return {
        "steps": steps,
        "events": events,
        "events_per_step": events / steps,
        "event_counts_by_batch": {str(k): batches[k] for k in sorted(batches)},
        "distinct_physical_parent_digests": len(signatures),
    }


def _ledger_turns(arm_dir: Path) -> dict[str, int]:
    """Count completed logical requests per task key from the proxy ledger."""
    path = arm_dir / "logs" / "fr13_fixed32_proxy_ingress.jsonl"
    if not path.is_file():
        raise AlignmentError(f"{arm_dir.name}: no logs/fr13_fixed32_proxy_ingress.jsonl")
    turns: collections.Counter[str] = collections.Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("event") == "logical_complete":
                turns[str(record.get("task_key_id"))] += 1
    return dict(turns)


def _ledger_interleave(arm_dir: Path) -> float:
    """Fraction of adjacent completions that switch task, i.e. interleaving."""
    path = arm_dir / "logs" / "fr13_fixed32_proxy_ingress.jsonl"
    order: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("event") == "logical_complete":
                order.append(str(record.get("task_key_id")))
    if len(order) < 2:
        return 0.0
    switches = sum(1 for a, b in zip(order, order[1:]) if a != b)
    return switches / (len(order) - 1)


def _sessions(arm_dir: Path) -> list[dict[str, Any]]:
    """Per-task agent session start and elapsed wall from runner_metadata."""
    out = []
    for task in _task_dirs(arm_dir):
        meta_path = task / "runner_metadata.json"
        if not meta_path.is_file():
            raise AlignmentError(f"{arm_dir.name}/{task.name}: no runner_metadata.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        agent = meta.get("agent") or {}
        elapsed = agent.get("elapsed_s")
        if not isinstance(elapsed, (int, float)):
            raise AlignmentError(f"{arm_dir.name}/{task.name}: no agent.elapsed_s")
        out.append(
            {
                "instance_id": meta.get("instance_id", task.name),
                "started_at": meta.get("started_at"),
                "elapsed_s": float(elapsed),
                "trace": _trace_shape(task),
            }
        )
    return out


def _session_window_s(sessions: list[dict[str, Any]]) -> float:
    """Union wall of the agent sessions: last end minus first start.

    Sequential arms (concurrency 1) span the sum of their sessions plus the
    inter-task gaps; concurrent arms that all start together span the longest
    session.  Taking the union covers both without a concurrency assumption.
    """
    starts = []
    ends = []
    for session in sessions:
        started = session.get("started_at")
        if not isinstance(started, str):
            return sum(s["elapsed_s"] for s in sessions)
        try:
            begin = dt.datetime.fromisoformat(started.replace("Z", "+00:00"))
        except ValueError:
            return sum(s["elapsed_s"] for s in sessions)
        starts.append(begin.timestamp())
        ends.append(begin.timestamp() + session["elapsed_s"])
    return max(ends) - min(starts)


def _trace_shape(task_dir: Path) -> dict[str, Any]:
    """Assistant turns and tool-call blocks recorded in the agent trace."""
    path = task_dir / "qwen_trace.jsonl"
    if not path.is_file():
        return {"assistant_turns": None, "tool_use_blocks": None}
    turns = 0
    tools = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") != "assistant":
                continue
            message = record.get("message") or {}
            usage = message.get("usage") or {}
            if (usage.get("input_tokens") or 0) > 0:
                turns += 1
            for block in message.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tools += 1
    return {"assistant_turns": turns, "tool_use_blocks": tools}


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _reduce_arm(runroot: Path, arm_dir: Path, role: str, meta: dict[str, str]) -> dict[str, Any]:
    bracket = _bracket(arm_dir)
    census = _census(arm_dir)
    delta = bracket["delta"]
    where = f"{runroot.name}/{arm_dir.name}"

    fwd_steps = _require(delta["vllm:fr13_decode_forward_gpu_steps_total"], "fr13_decode_forward_gpu_steps_total", where)
    fwd_events = _require(delta["vllm:fr13_decode_forward_gpu_drafts_total"], "fr13_decode_forward_gpu_drafts_total", where)
    if int(fwd_steps) != census["steps"]:
        raise AlignmentError(
            f"{where}: census steps {census['steps']} != bracket pure-decode steps {int(fwd_steps)}"
        )
    if int(fwd_events) != census["events"]:
        raise AlignmentError(
            f"{where}: census events {census['events']} != bracket pure-decode drafts {int(fwd_events)}"
        )

    requests = _require(delta["vllm:request_prompt_tokens_count"], "request_prompt_tokens_count", where)
    turns = _ledger_turns(arm_dir)
    ledger_total = sum(turns.values())
    if ledger_total != int(requests):
        raise AlignmentError(
            f"{where}: ledger logical_complete {ledger_total} != bracket requests {int(requests)}"
        )

    prompt = _require(delta["vllm:prompt_tokens_total"], "prompt_tokens_total", where)
    cached = _require(delta["vllm:prompt_tokens_cached_total"], "prompt_tokens_cached_total", where)
    computed = _require(delta["vllm:request_prefill_kv_computed_tokens_sum"], "request_prefill_kv_computed_tokens_sum", where)
    generated = _require(delta["vllm:generation_tokens_total"], "generation_tokens_total", where)
    accepted = _require(delta["vllm:spec_decode_num_accepted_tokens_total"], "spec_decode_num_accepted_tokens_total", where)

    fwd_s = _require(delta["vllm:fr13_decode_forward_gpu_seconds_total"], "fr13_decode_forward_gpu_seconds_total", where)
    dfwd_s = _require(delta["vllm:fr13_drafter_gpu_seconds_total"], "fr13_drafter_gpu_seconds_total", where)
    dfwd_spans = _require(delta["vllm:fr13_drafter_gpu_spans_total"], "fr13_drafter_gpu_spans_total", where)
    cfwd_s = _require(delta["vllm:fr13_committer_gpu_seconds_total"], "fr13_committer_gpu_seconds_total", where)
    cfwd_spans = _require(delta["vllm:fr13_committer_gpu_spans_total"], "fr13_committer_gpu_spans_total", where)
    wall_s = _require(delta["vllm:fr13_decode_step_wall_seconds_total"], "fr13_decode_step_wall_seconds_total", where)
    wall_steps = _require(delta["vllm:fr13_decode_step_wall_steps_total"], "fr13_decode_step_wall_steps_total", where)
    wall_events = _require(delta["vllm:fr13_decode_step_wall_drafts_total"], "fr13_decode_step_wall_drafts_total", where)

    prefill_s = _require(delta["vllm:request_prefill_time_seconds_sum"], "request_prefill_time_seconds_sum", where)
    decode_s = _require(delta["vllm:request_decode_time_seconds_sum"], "request_decode_time_seconds_sum", where)
    queue_s = _require(delta["vllm:request_queue_time_seconds_sum"], "request_queue_time_seconds_sum", where)
    e2e_s = _require(delta["vllm:e2e_request_latency_seconds_sum"], "e2e_request_latency_seconds_sum", where)
    iterations = delta.get("vllm:iteration_tokens_total_count")

    globals_ = _global_delta(arm_dir)
    global_drafts = globals_["vllm:spec_decode_num_drafts_total"]

    events_per_step_gpu = fwd_events / fwd_steps
    events_per_step_wall = wall_events / wall_steps
    committed_per_event = accepted / fwd_events + 1.0

    sfwd_ms_event = 1000.0 * fwd_s / fwd_events
    sfwd_ms_step = 1000.0 * fwd_s / fwd_steps
    dfwd_ms_step = 1000.0 * dfwd_s / dfwd_spans
    cfwd_ms_step = 1000.0 * cfwd_s / cfwd_spans
    dfwd_ms_event = dfwd_ms_step / events_per_step_gpu
    cfwd_ms_event = cfwd_ms_step / events_per_step_gpu
    step_wall_ms = 1000.0 * wall_s / wall_steps
    wall_ms_event = 1000.0 * wall_s / wall_events
    other_ms_event = wall_ms_event - (sfwd_ms_event + dfwd_ms_event + cfwd_ms_event)
    other_ms_step = step_wall_ms - (sfwd_ms_step + dfwd_ms_step + cfwd_ms_step)

    # Request-decode events that landed on a mixed prefill+decode step and were
    # therefore excluded from the pure-decode basis.  Concurrency-1 arms must
    # read exactly zero; that is the calibration of the per-request offset.
    mixed_events = None
    mixed_share = None
    if global_drafts is not None:
        comparable = global_drafts + DRAFT_COUNTER_OFFSET_PER_REQUEST * requests
        mixed_events = comparable - fwd_events
        mixed_share = _ratio(mixed_events, comparable)
        if mixed_events < 0:
            raise AlignmentError(
                f"{where}: pure-decode events {int(fwd_events)} exceed the global basis {comparable:.0f}"
            )

    sessions = _sessions(arm_dir)
    session_wall = sum(s["elapsed_s"] for s in sessions)
    arm_wall = _session_window_s(sessions)
    concurrency = float(meta.get("concurrency") or len(sessions))

    active_sessions = _ratio(session_wall, arm_wall) or 0.0
    inflight = _ratio(e2e_s, arm_wall) or 0.0
    decode_resident = _ratio(decode_s, arm_wall) or 0.0
    missing = concurrency - events_per_step_gpu
    coresidency = {
        "nominal_concurrency": concurrency,
        "arm_wall_s": arm_wall,
        "session_wall_s": session_wall,
        "mean_active_sessions": active_sessions,
        "mean_inflight_requests": inflight,
        "mean_decode_resident_requests": decode_resident,
        "measured_events_per_step": events_per_step_gpu,
        "missing_events_per_step": missing,
        "attribution": {
            "task_end_skew": concurrency - active_sessions,
            "agent_think_time": active_sessions - inflight,
            "prefill_and_queue_occupancy": inflight - decode_resident,
            "pure_decode_step_concentration": decode_resident - events_per_step_gpu,
        },
    }
    if missing != 0:
        coresidency["attribution_share"] = {
            key: value / missing for key, value in coresidency["attribution"].items()
        }

    prefill_share_gross = _ratio(prompt, prompt + generated)
    prefill_share_computed = _ratio(computed, computed + generated)

    reported_path = arm_dir / "deploy_speed_fullwall.json"
    reported = json.loads(reported_path.read_text(encoding="utf-8")) if reported_path.is_file() else {}

    return {
        "role": role,
        "arm": arm_dir.name,
        "runroot": str(runroot),
        "batch_size": int(meta.get("batch_size") or 0) or None,
        "concurrency": concurrency,
        "bracket_topology": bracket["topology"],
        "bracket_closing_task": bracket["closing_task"],
        "census": census,
        "counters": {
            "requests": requests,
            "prompt_tokens": prompt,
            "prompt_tokens_cached": cached,
            "prefill_kv_computed_tokens": computed,
            "generation_tokens": generated,
            "accepted_tokens": accepted,
            "pure_decode_steps": fwd_steps,
            "pure_decode_events": fwd_events,
            "global_spec_drafts": global_drafts,
            "engine_iterations": iterations,
            "prefill_time_s": prefill_s,
            "decode_time_s": decode_s,
            "queue_time_s": queue_s,
            "e2e_latency_s": e2e_s,
        },
        "prefill_share": {
            "requests": requests,
            "prompt_tokens_per_request": prompt / requests,
            "computed_prefill_tokens_per_request": computed / requests,
            "committed_tokens_per_request": generated / requests,
            "share_gross": prefill_share_gross,
            "share_computed": prefill_share_computed,
            "apc_hit_rate_tokens": _ratio(cached, prompt),
            "prefill_frac_time": _ratio(prefill_s, decode_s),
        },
        "phase": {
            "events_per_step_gpu_window": events_per_step_gpu,
            "events_per_step_wall_window": events_per_step_wall,
            "committed_tokens_per_event": committed_per_event,
            "per_event_ms": {
                "sfwd": sfwd_ms_event,
                "dfwd": dfwd_ms_event,
                "cfwd": cfwd_ms_event,
                "other": other_ms_event,
                "wall": wall_ms_event,
            },
            "per_step_ms": {
                "sfwd": sfwd_ms_step,
                "dfwd": dfwd_ms_step,
                "cfwd": cfwd_ms_step,
                "other": other_ms_step,
                "wall": step_wall_ms,
            },
            "mixed_step_events": mixed_events,
            "mixed_step_event_share": mixed_share,
            "pure_decode_iteration_coverage": _ratio(fwd_steps, iterations) if iterations else None,
        },
        "throughput": {
            "aggregate_tps_fullstep_wall": committed_per_event / (wall_ms_event / 1000.0),
            "per_request_step_tps": committed_per_event / (step_wall_ms / 1000.0),
            "effective_concurrency": reported.get("effective_concurrency"),
            "per_effective_request_tps": (
                None
                if not reported.get("effective_concurrency")
                else (committed_per_event / (wall_ms_event / 1000.0)) / reported["effective_concurrency"]
            ),
        },
        "coresidency": coresidency,
        "trajectory": {
            "turns_by_task_key": turns,
            "turns_per_task_mean": ledger_total / len(sessions),
            "prompt_tokens_per_turn": prompt / requests,
            "suffix_tokens_per_turn": computed / requests,
            "committed_tokens_per_turn": generated / requests,
            "accepted_per_event": accepted / fwd_events,
            "ledger_interleave_fraction": _ledger_interleave(arm_dir),
            "sessions": sessions,
        },
        "reported": {
            "events_per_step": reported.get("events_per_step"),
            "measured_tps_fullstep_wall": reported.get("measured_tps_fullstep_wall"),
            "step_wall_ms": reported.get("step_wall_ms"),
            "committed_per_event": reported.get("committed_per_event"),
            "prefill_frac": reported.get("prefill_frac"),
            "s_per_fwd_gpu": reported.get("s_per_fwd_gpu"),
        },
        "summed_bracket_inflation": {
            name: _ratio(bracket["summed_delta"][name], value)
            for name, value in delta.items()
            if value not in (None, 0) and name in bracket["summed_delta"]
        },
    }


def _reduce_runroot(runroot: Path, label: str | None) -> dict[str, Any]:
    meta_path = runroot / "launcher_meta.txt"
    if not meta_path.is_file():
        raise AlignmentError(f"{runroot}: no launcher_meta.txt")
    meta = _read_launcher_meta(meta_path)
    arms = []
    for role, key in (("stock", "stock_arm"), ("candidate", "candidate_arm")):
        name = meta.get(key)
        if not name:
            raise AlignmentError(f"{runroot.name}: launcher_meta has no {key}")
        arm_dir = runroot / name
        if not arm_dir.is_dir():
            raise AlignmentError(f"{runroot.name}: arm directory {name} is missing")
        arms.append(_reduce_arm(runroot, arm_dir, role, meta))
    return {
        "label": label or runroot.name,
        "runroot": str(runroot),
        "batch_size": int(meta.get("batch_size") or 0) or None,
        "concurrency": float(meta.get("concurrency") or 0) or None,
        "task_count": int(meta.get("task_count") or 0) or None,
        "salvaged": (runroot / "SALVAGE_NOTE.md").is_file(),
        "arms": arms,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def _table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    out = ["  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(header))]
    out.append("  " + "  ".join("-" * widths[i] for i in range(len(header))))
    for row in rows:
        out.append("  " + "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(out)


def _render(payload: dict[str, Any]) -> str:
    arms = [(run["label"], arm) for run in payload["runroots"] for arm in run["arms"]]
    names = [f"{label}/{arm['role']}" for label, arm in arms]

    out = ["=== FR13 B1/B4 MEASUREMENT ALIGNMENT ===", ""]

    out.append("bracket topology and census cross-check")
    out.append(
        _table(
            ("arm", "B", "topology", "closing", "census_steps", "census_events", "events/step"),
            [
                (
                    names[i],
                    _fmt(arm["batch_size"], 0),
                    arm["bracket_topology"],
                    arm["bracket_closing_task"],
                    str(arm["census"]["steps"]),
                    str(arm["census"]["events"]),
                    _fmt(arm["census"]["events_per_step"], 4),
                )
                for i, (_, arm) in enumerate(arms)
            ],
        )
    )
    out.append("")

    out.append("1. per-request prefill token share")
    out.append(
        _table(
            ("arm", "reqs", "prompt/req", "computed/req", "commit/req", "share_gross", "share_computed", "apc_hit"),
            [
                (
                    names[i],
                    _fmt(arm["prefill_share"]["requests"], 0),
                    _fmt(arm["prefill_share"]["prompt_tokens_per_request"], 1),
                    _fmt(arm["prefill_share"]["computed_prefill_tokens_per_request"], 1),
                    _fmt(arm["prefill_share"]["committed_tokens_per_request"], 1),
                    _fmt(arm["prefill_share"]["share_gross"], 5),
                    _fmt(arm["prefill_share"]["share_computed"], 5),
                    _fmt(arm["prefill_share"]["apc_hit_rate_tokens"], 4),
                )
                for i, (_, arm) in enumerate(arms)
            ],
        )
    )
    out.append("")

    out.append("2. pure-decode phase decomposition (ms, s_per_fwd_gpu basis)")
    out.append(
        _table(
            ("arm", "eps_gpu", "sfwd/ev", "dfwd/ev", "cfwd/ev", "other/ev", "wall/ev", "sfwd/st", "dfwd/st", "cfwd/st", "other/st", "wall/st"),
            [
                (
                    names[i],
                    _fmt(arm["phase"]["events_per_step_gpu_window"], 4),
                    _fmt(arm["phase"]["per_event_ms"]["sfwd"]),
                    _fmt(arm["phase"]["per_event_ms"]["dfwd"]),
                    _fmt(arm["phase"]["per_event_ms"]["cfwd"]),
                    _fmt(arm["phase"]["per_event_ms"]["other"]),
                    _fmt(arm["phase"]["per_event_ms"]["wall"]),
                    _fmt(arm["phase"]["per_step_ms"]["sfwd"]),
                    _fmt(arm["phase"]["per_step_ms"]["dfwd"]),
                    _fmt(arm["phase"]["per_step_ms"]["cfwd"]),
                    _fmt(arm["phase"]["per_step_ms"]["other"]),
                    _fmt(arm["phase"]["per_step_ms"]["wall"]),
                )
                for i, (_, arm) in enumerate(arms)
            ],
        )
    )
    out.append("")

    out.append("3. co-residency decomposition of the missing events/step")
    out.append(
        _table(
            ("arm", "ceiling", "measured", "missing", "end_skew", "think", "prefill", "step_conc"),
            [
                (
                    names[i],
                    _fmt(arm["coresidency"]["nominal_concurrency"], 2),
                    _fmt(arm["coresidency"]["measured_events_per_step"], 4),
                    _fmt(arm["coresidency"]["missing_events_per_step"], 4),
                    _fmt(arm["coresidency"]["attribution"]["task_end_skew"], 4),
                    _fmt(arm["coresidency"]["attribution"]["agent_think_time"], 4),
                    _fmt(arm["coresidency"]["attribution"]["prefill_and_queue_occupancy"], 4),
                    _fmt(arm["coresidency"]["attribution"]["pure_decode_step_concentration"], 4),
                )
                for i, (_, arm) in enumerate(arms)
            ],
        )
    )
    out.append("")

    out.append("4. throughput, recomputed vs the run's own deploy_speed_fullwall.json")
    out.append(
        _table(
            ("arm", "commit/ev", "step_wall_ms", "aggregate_tps", "per_req_step_tps", "eff_conc", "per_eff_req_tps", "reported_tps", "reported_eps"),
            [
                (
                    names[i],
                    _fmt(arm["phase"]["committed_tokens_per_event"], 4),
                    _fmt(arm["phase"]["per_step_ms"]["wall"]),
                    _fmt(arm["throughput"]["aggregate_tps_fullstep_wall"]),
                    _fmt(arm["throughput"]["per_request_step_tps"]),
                    _fmt(arm["throughput"]["effective_concurrency"], 4),
                    _fmt(arm["throughput"]["per_effective_request_tps"]),
                    _fmt(arm["reported"]["measured_tps_fullstep_wall"]),
                    _fmt(arm["reported"]["events_per_step"], 4),
                )
                for i, (_, arm) in enumerate(arms)
            ],
        )
    )
    out.append("")

    out.append("5. trajectory covariates")
    out.append(
        _table(
            ("arm", "turns", "turns/task", "prompt/turn", "suffix/turn", "commit/turn", "accept/event", "interleave"),
            [
                (
                    names[i],
                    _fmt(arm["counters"]["requests"], 0),
                    _fmt(arm["trajectory"]["turns_per_task_mean"], 2),
                    _fmt(arm["trajectory"]["prompt_tokens_per_turn"], 1),
                    _fmt(arm["trajectory"]["suffix_tokens_per_turn"], 1),
                    _fmt(arm["trajectory"]["committed_tokens_per_turn"], 1),
                    _fmt(arm["trajectory"]["accepted_per_event"], 4),
                    _fmt(arm["trajectory"]["ledger_interleave_fraction"], 3),
                )
                for i, (_, arm) in enumerate(arms)
            ],
        )
    )
    out.append("")

    out.append("prefill-phase occupancy (events excluded from the pure-decode basis)")
    out.append(
        _table(
            ("arm", "pure_decode_events", "global+1/req", "mixed_step_events", "mixed_share", "iter_coverage"),
            [
                (
                    names[i],
                    _fmt(arm["counters"]["pure_decode_events"], 0),
                    _fmt(
                        None
                        if arm["counters"]["global_spec_drafts"] is None
                        else arm["counters"]["global_spec_drafts"] + arm["counters"]["requests"],
                        0,
                    ),
                    _fmt(arm["phase"]["mixed_step_events"], 0),
                    _fmt(arm["phase"]["mixed_step_event_share"], 4),
                    _fmt(arm["phase"]["pure_decode_iteration_coverage"], 4),
                )
                for i, (_, arm) in enumerate(arms)
            ],
        )
    )
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runroot",
        action="append",
        required=True,
        type=Path,
        help="paired timing runroot (repeatable); must contain launcher_meta.txt",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="label for the matching --runroot (repeatable, positional pairing)",
    )
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    parser.add_argument("--json", action="store_true", help="print JSON instead of the text report")
    args = parser.parse_args(argv)

    labels = list(args.label or [])
    if labels and len(labels) != len(args.runroot):
        print("FAIL: --label must be given once per --runroot", file=sys.stderr)
        return 2

    try:
        runroots = [
            _reduce_runroot(root.resolve(), labels[i] if i < len(labels) else None)
            for i, root in enumerate(args.runroot)
        ]
    except (AlignmentError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    payload = {"schema": SCHEMA, "runroots": runroots}
    rendered = json.dumps(payload, indent=1, sort_keys=True) if args.json else _render(payload)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(rendered if rendered.endswith("\n") else rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

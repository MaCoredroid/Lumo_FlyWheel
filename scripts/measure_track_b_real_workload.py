#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lumo_flywheel_serving.metrics import compute_task_metrics, parse_prometheus_text, resolve_metric_schema  # noqa: E402

SPEC_TRACE = "/tmp/lumo-l0c-fp8-cutlass-run30-logs/per_req_spec_trace.jsonl"
FB_DEBUG = "/tmp/lumo-l0c-fp8-cutlass-run30-logs/fb_debug.jsonl"
FB_OVERHEAD_DEBUG = "/tmp/lumo-l0c-fp8-cutlass-run30-logs/fb_overhead_debug.jsonl"
FB_INTERNAL_DEBUG = "/tmp/lumo-l0c-fp8-cutlass-run30-logs/fb_internal_debug.jsonl"
FA_UNIQUE_GDN_DEBUG = "/tmp/lumo-l0c-fp8-cutlass-run30-logs/fa_unique_gdn_debug.jsonl"
FA_REPLAY_COMMIT_DETAIL = "/tmp/lumo-l0c-fp8-cutlass-run30-logs/fa_activation_replay_commit_detail.jsonl"
CUDAGRAPH_RUNTIME_DEBUG = "/tmp/lumo-l0c-fp8-cutlass-run30-logs/cudagraph_runtime_debug.jsonl"


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _raise_for_status_with_body(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text.strip().replace("\n", "\\n")
        if len(body) > 1200:
            body = body[:1200] + "...<truncated>"
        detail = f"{exc}; response_body={body or '<empty>'}"
        raise requests.HTTPError(detail, response=response) from exc


def _load_workload(path: Path) -> tuple[dict[str, Any], Path]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"workload file must be a YAML mapping: {path}")
    seed_ref = str(payload.get("seed_trace_ref") or "")
    if not seed_ref:
        raise RuntimeError(f"workload file is missing seed_trace_ref: {path}")
    seed_path = Path(seed_ref)
    if not seed_path.is_absolute():
        seed_path = path.parent / seed_path
    if not seed_path.is_file():
        raise RuntimeError(f"seed trace is missing: {seed_path}")
    return payload, seed_path


def _load_seed(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise RuntimeError(f"seed row must be a JSON object: {line}")
        rows.append(payload)
    if not rows:
        raise RuntimeError(f"seed trace is empty: {path}")
    return rows


def _load_swe_bench_entries(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    subset = json.loads(path.read_text(encoding="utf-8"))
    instance_ids = list(subset.get("instance_ids") or [])
    if not instance_ids:
        raise RuntimeError(f"SWE-bench subset has no instance_ids: {path}")
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
    from datasets import load_dataset

    ds = load_dataset(str(subset.get("dataset_name") or "princeton-nlp/SWE-bench_Verified"),
                      split=str(subset.get("split") or "test"))
    wanted = set(str(iid) for iid in instance_ids)
    rows_by_id = {str(ex["instance_id"]): ex for ex in ds if str(ex["instance_id"]) in wanted}
    missing = sorted(wanted - set(rows_by_id))
    if missing:
        raise RuntimeError(f"SWE-bench subset instances missing from dataset: {missing}")
    entries: list[dict[str, Any]] = []
    for iid in instance_ids:
        ex = rows_by_id[str(iid)]
        problem = str(ex.get("problem_statement") or "").strip()
        repo = str(ex.get("repo") or "unknown")
        base_commit = str(ex.get("base_commit") or "")
        prompt = (
            "You are working on a SWE-bench Verified task.\n"
            f"Instance: {iid}\n"
            f"Repository: {repo}\n"
            f"Base commit: {base_commit}\n\n"
            "Problem statement:\n"
            f"{problem}\n\n"
            "Reason about the fix. Do not use tools; provide the likely code-change plan and key files."
        )
        entries.append({
            "instance_id": str(iid),
            "repo": repo,
            "prompt": prompt,
            "prompt_tokens": max(1, len(prompt.split())),
            "request_max_output_tokens": 512,
            "class": "swe_bench_verified",
        })
    workload = {
        "workload_distribution_id": f"swe_bench_verified_subset:{path.name}",
        "dataset_name": subset.get("dataset_name"),
        "split": subset.get("split"),
        "subset_path": str(path),
        "instance_ids": instance_ids,
    }
    return workload, entries


def _jsonl_count(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    with p.open() as f:
        return sum(1 for _ in f)


def _read_jsonl_range(path: str, pre: int, post: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not Path(path).exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if pre < i <= post:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _summarize_spec_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"events": 0}
    accs = [int(r.get("acc", 0) or 0) for r in rows]
    drafts = [int(r.get("draft", 0) or 0) for r in rows]
    ts = [float(r.get("ts", 0.0) or 0.0) for r in rows]
    span = max(0.0, ts[-1] - ts[0]) if len(ts) > 1 else 0.0
    return {
        "events": len(rows),
        "mean_acc_per_event": round(sum(accs) / len(accs), 3),
        "mean_draft_per_event": round(sum(drafts) / len(drafts), 3),
        "accepted_per_node": round(sum(accs) / sum(drafts), 4) if sum(drafts) else None,
        "mean_event_ms": round(span / (len(rows) - 1) * 1000, 3) if len(rows) > 1 and span > 0 else None,
        "acc_dist": dict(sorted(collections.Counter(accs).items())),
        "acc_sequence_sha256": hashlib.sha256(
            json.dumps(accs, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "draft_sequence_sha256": hashlib.sha256(
            json.dumps(drafts, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "acc_sequence": accs,
        "draft_sequence": drafts,
    }


def _summarize_unified_debug(rows: list[dict[str, Any]]) -> dict[str, Any]:
    unified = [r for r in rows if r.get("event") == "round_f_unified_step"]
    free = [r for r in rows if r.get("event") == "fb_free_row1_decision"]
    failures = []
    for row in unified:
        failures.extend(row.get("physical_minimum_invariant_failures") or [])
    def _rate(pred) -> float | None:
        return round(sum(1 for row in unified if pred(row)) / len(unified), 4) if unified else None
    def _int_field(row: dict[str, Any], key: str, default: int = -1) -> int:
        value = row.get(key, default)
        if value is None:
            return default
        return int(value)
    return {
        "round_f_unified_events": len(unified),
        "free_row1_events": len(free),
        "component_under_test": dict(sorted(collections.Counter(
            str(r.get("component_under_test") or "unspecified") for r in unified
        ).items())),
        "verifier_path": dict(sorted(collections.Counter(
            str(r.get("verifier_path") or "unspecified") for r in unified
        ).items())),
        "internal_rows_enabled_rate": _rate(lambda r: r.get("internal_rows_enabled") is True),
        "kernel_rows_enabled_rate": _rate(lambda r: r.get("kernel_rows_enabled") is True),
        "no_kv_prefix_copy_enabled_rate": _rate(lambda r: r.get("no_kv_prefix_copy_enabled") is True),
        "selected_eq_verified_rate": _rate(lambda r: r.get("selected_nodes") == r.get("verified_nodes")),
        "path_rows_zero_rate": _rate(lambda r: _int_field(r, "path_rows") == 0),
        "scheduler_clone_zero_rate": _rate(lambda r: _int_field(r, "scheduler_visible_clone_requests") == 0),
        "prefix_kv_copy_bytes_total": sum(int(r.get("prefix_kv_copy_bytes", 0) or 0) for r in unified),
        "extra_proposer_for_trimmed_nodes_total": sum(int(r.get("extra_proposer_for_trimmed_nodes", 0) or 0) for r in unified),
        "candidate_pool_nodes_mean": (
            round(sum(float(r.get("candidate_pool_nodes", 0) or 0) for r in unified) / len(unified), 3)
            if unified else None
        ),
        "selected_nodes_mean": (
            round(sum(float(r.get("selected_nodes", 0) or 0) for r in unified) / len(unified), 3)
            if unified else None
        ),
        "trimmed_nodes_mean": (
            round(sum(float(r.get("trimmed_nodes", 0) or 0) for r in unified) / len(unified), 3)
            if unified else None
        ),
        "invariant_failures": dict(sorted(collections.Counter(failures).items())),
        "free_row1_extra_extend_one_calls": sum(int(r.get("extra_extend_one_calls", 0) or 0) for r in free),
        "free_row1_proposer_free_rate_enabled": (
            round(
                sum(1 for r in free if r.get("row1_enabled") and r.get("proposer_free") is True)
                / max(1, sum(1 for r in free if r.get("row1_enabled"))),
                4,
            )
            if free else None
        ),
    }


def _summarize_fb_overhead(rows: list[dict[str, Any]]) -> dict[str, Any]:
    schedules = [r for r in rows if r.get("event") == "schedule"]
    winner_sources = [r for r in rows if r.get("event") == "scheduler_winner_source"]
    return {
        "schedule_events": len(schedules),
        "scheduler_visible_clone_requests_total": sum(
            int(r.get("fb_parent_count", 0) or 0) for r in schedules
        ),
        "internal_row_count_total": sum(
            int(r.get("fb_internal_row_count", 0) or 0) for r in schedules
        ),
        "kv_blocks_copied_total": sum(
            int(r.get("fb_kv_blocks_copied", 0) or 0) for r in schedules
        ),
        "mamba_blocks_copied_total": sum(
            int(r.get("fb_mamba_blocks_copied", 0) or 0) for r in schedules
        ),
        "state_fork_us_total": sum(
            int(r.get("fb_state_fork_us", 0) or 0) for r in schedules
        ),
        "scheduler_us_total": sum(
            int(r.get("fb_scheduler_us", 0) or 0) for r in schedules
        ),
        "winner_source_events": len(winner_sources),
    }


def _summarize_fb_internal(rows: list[dict[str, Any]]) -> dict[str, Any]:
    events = collections.Counter(str(r.get("event") or "unknown") for r in rows)
    sampled = [r for r in rows if r.get("event") == "sampled"]
    noactive = [r for r in rows if r.get("event") == "kernel_noactive_sample"]
    promotes = [r for r in rows if r.get("event") == "kernel_promote_state"]
    return {
        "events": dict(sorted(events.items())),
        "sampled_events": len(sampled),
        "kernel_noactive_sample_events": len(noactive),
        "kernel_promote_state_events": len(promotes),
        "winner_events": sum(len(r.get("winners") or {}) for r in sampled),
    }


def _summarize_fa_unique_gdn(rows: list[dict[str, Any]]) -> dict[str, Any]:
    layer_rows = [r for r in rows if r.get("event") == "fa_unique_gdn_layer"]
    if not layer_rows:
        return {"events": 0}
    layers = collections.Counter(str(r.get("layer") or "unknown") for r in layer_rows)
    parent_shapes = collections.Counter(
        json.dumps(r.get("parents"), separators=(",", ":"))
        for r in layer_rows
    )
    state_shapes = collections.Counter(
        json.dumps(r.get("state_rows"), separators=(",", ":"))
        for r in layer_rows
    )
    per_event_spans_us: list[int] = []
    layer_count = len(layers) if layers else 0
    if layer_count > 0:
        for start in range(0, len(layer_rows), layer_count):
            chunk = layer_rows[start:start + layer_count]
            if len(chunk) != layer_count:
                continue
            ts = [float(r.get("ts", 0.0) or 0.0) for r in chunk]
            if ts and max(ts) >= min(ts):
                per_event_spans_us.append(int(round((max(ts) - min(ts)) * 1_000_000)))
    return {
        "events": len(layer_rows),
        "unique_layers": len(layers),
        "layer_event_min": min(layers.values()) if layers else 0,
        "layer_event_max": max(layers.values()) if layers else 0,
        "parent_maps": dict(sorted(parent_shapes.items())),
        "state_row_shapes": dict(sorted(state_shapes.items())),
        "timing": {
            "gdn_layer_event_count": len(per_event_spans_us),
            "gdn_layer_span_us_mean": (
                round(sum(per_event_spans_us) / len(per_event_spans_us), 3)
                if per_event_spans_us else None
            ),
            "gdn_layer_span_us_max": max(per_event_spans_us) if per_event_spans_us else None,
        },
        "sample": layer_rows[:3],
    }


def _summarize_fa_replay_commit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [r for r in rows if r.get("event") == "fa_activation_replay_commit_summary"]
    if not summaries:
        return {"events": 0}
    commit_us = [int(r.get("commit_enqueue_us", 0) or 0) for r in summaries]
    copied = [int(r.get("copied_requests", 0) or 0) for r in summaries]
    modes = collections.Counter(str(r.get("commit_mode") or "unknown") for r in summaries)
    return {
        "events": len(summaries),
        "commit_modes": dict(sorted(modes.items())),
        "commit_enqueue_us_mean": round(sum(commit_us) / len(commit_us), 3) if commit_us else None,
        "commit_enqueue_us_max": max(commit_us) if commit_us else None,
        "copied_requests_total": sum(copied),
        "copied_requests_mean": round(sum(copied) / len(copied), 3) if copied else None,
    }


def _summarize_cudagraph_runtime(rows: list[dict[str, Any]]) -> dict[str, Any]:
    events = [r for r in rows if r.get("event") == "cudagraph_runtime"]
    if not events:
        return {"events": 0}
    modes = collections.Counter(str(r.get("runtime_mode") or "unknown") for r in events)
    paddings = [int(r.get("num_paddings", 0) or 0) for r in events]
    padded_tokens = [int(r.get("num_padded_tokens", 0) or 0) for r in events]
    return {
        "events": len(events),
        "runtime_modes": dict(sorted(modes.items())),
        "full_count": int(modes.get("CUDAGraphMode.FULL", 0)),
        "piecewise_count": int(modes.get("CUDAGraphMode.PIECEWISE", 0)),
        "eager_fallback_count": int(modes.get("CUDAGraphMode.NONE", 0)),
        "num_paddings_mean": round(sum(paddings) / len(paddings), 3) if paddings else None,
        "num_paddings_max": max(paddings) if paddings else None,
        "num_padded_tokens": dict(sorted(collections.Counter(padded_tokens).items())),
    }


def _timer_stats(values: list[int | float]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"count": 0, "mean": None, "max": None}
    return {
        "count": len(clean),
        "mean": round(sum(clean) / len(clean), 3),
        "max": round(max(clean), 3),
    }


def _metrics(metrics_url: str) -> dict[str, float]:
    response = requests.get(metrics_url, timeout=20)
    _raise_for_status_with_body(response)
    return parse_prometheus_text(response.text)


def _metric_summary(before: dict[str, float], after: dict[str, float], *, request_count: int, elapsed_s: float) -> dict[str, Any]:
    try:
        schema = resolve_metric_schema(after)
        metrics = compute_task_metrics(before=before, after=after, schema=schema)
    except Exception as exc:
        return {
            "available": False,
            "reason": f"metrics_unavailable: {exc}",
            "request_count": request_count,
            "elapsed_s": round(elapsed_s, 6),
        }
    gen_tokens = float(metrics.get("gen_tokens") or 0.0)
    decode_sum_s = float(metrics.get("decode_sum_s") or 0.0)
    prefill_sum_s = float(metrics.get("prefill_sum_s") or 0.0)
    prompt_tokens = float(metrics.get("prompt_tokens") or 0.0)
    kv_tokens = float(metrics.get("kv_computed_tokens") or 0.0)
    accepted = float(metrics.get("spec_decode_num_accepted_tokens") or 0.0)
    draft_tokens = float(metrics.get("spec_decode_num_draft_tokens") or 0.0)
    return {
        "available": True,
        "request_count": request_count,
        "elapsed_s": round(elapsed_s, 6),
        "metrics_delta": {
            "prompt_tokens": prompt_tokens,
            "kv_computed_tokens": kv_tokens,
            "generation_tokens": gen_tokens,
            "prefill_sum_s": prefill_sum_s,
            "decode_sum_s": decode_sum_s,
            "ttft_sum_s": float(metrics.get("ttft_sum_s") or 0.0),
            "ttft_count": int(metrics.get("ttft_count") or 0),
            "cache_queries": float(metrics.get("cache_queries") or 0.0),
            "cache_hits": float(metrics.get("cache_hits") or 0.0),
            "spec_decode_num_accepted_tokens": accepted,
            "spec_decode_num_draft_tokens": draft_tokens,
        },
        "step_consumption": {
            "prompt_tokens_per_request": round(prompt_tokens / max(request_count, 1), 3),
            "generation_tokens_per_request": round(gen_tokens / max(request_count, 1), 3),
            "prefill_ms_per_kv_token": round(prefill_sum_s * 1000.0 / kv_tokens, 6) if kv_tokens > 0 else None,
            "decode_ms_per_generated_token": round(decode_sum_s * 1000.0 / gen_tokens, 6) if gen_tokens > 0 else None,
            "decode_tokens_per_s": round(gen_tokens / decode_sum_s, 6) if decode_sum_s > 0 else None,
            "wall_decode_tokens_per_s": round(gen_tokens / elapsed_s, 6) if elapsed_s > 0 else None,
            "cache_hit_rate_pct": metrics.get("cache_hit_rate_pct"),
            "accepted_per_draft_token": round(accepted / draft_tokens, 6) if draft_tokens > 0 else None,
        },
        "bottleneck_hint": "decode" if decode_sum_s >= prefill_sum_s else "prefill",
    }


def _aggregate_metric_summaries(summaries: list[dict[str, Any]], *, request_count: int, elapsed_s: float) -> dict[str, Any]:
    if not summaries or not all(bool(row.get("available")) for row in summaries):
        return {
            "available": False,
            "reason": "one_or_more_window_metric_summaries_unavailable",
            "request_count": request_count,
            "elapsed_s": round(elapsed_s, 6),
        }
    totals = {
        "prompt_tokens": 0.0,
        "kv_computed_tokens": 0.0,
        "generation_tokens": 0.0,
        "prefill_sum_s": 0.0,
        "decode_sum_s": 0.0,
        "ttft_sum_s": 0.0,
        "ttft_count": 0.0,
        "cache_queries": 0.0,
        "cache_hits": 0.0,
        "spec_decode_num_accepted_tokens": 0.0,
        "spec_decode_num_draft_tokens": 0.0,
    }
    for summary in summaries:
        delta = summary.get("metrics_delta") if isinstance(summary.get("metrics_delta"), dict) else {}
        for key in totals:
            totals[key] += float(delta.get(key) or 0.0)
    gen_tokens = totals["generation_tokens"]
    decode_sum_s = totals["decode_sum_s"]
    prefill_sum_s = totals["prefill_sum_s"]
    kv_tokens = totals["kv_computed_tokens"]
    cache_queries = totals["cache_queries"]
    accepted = totals["spec_decode_num_accepted_tokens"]
    draft_tokens = totals["spec_decode_num_draft_tokens"]
    return {
        "available": True,
        "request_count": request_count,
        "elapsed_s": round(elapsed_s, 6),
        "metrics_delta": totals,
        "step_consumption": {
            "prompt_tokens_per_request": round(totals["prompt_tokens"] / max(request_count, 1), 3),
            "generation_tokens_per_request": round(gen_tokens / max(request_count, 1), 3),
            "prefill_ms_per_kv_token": round(prefill_sum_s * 1000.0 / kv_tokens, 6) if kv_tokens > 0 else None,
            "decode_ms_per_generated_token": round(decode_sum_s * 1000.0 / gen_tokens, 6) if gen_tokens > 0 else None,
            "decode_tokens_per_s": round(gen_tokens / decode_sum_s, 6) if decode_sum_s > 0 else None,
            "wall_decode_tokens_per_s": round(gen_tokens / elapsed_s, 6) if elapsed_s > 0 else None,
            "cache_hit_rate_pct": (totals["cache_hits"] / cache_queries * 100.0) if cache_queries > 0 else None,
            "accepted_per_draft_token": round(accepted / draft_tokens, 6) if draft_tokens > 0 else None,
        },
        "bottleneck_hint": "decode" if decode_sum_s >= prefill_sum_s else "prefill",
    }


def _post_response(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    entry: dict[str, Any],
    request_id: str,
    prompt_token_cap: int,
    max_output_token_cap: int | None,
    request_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    prompt_tokens = int(entry.get("prompt_tokens") or 1)
    if prompt_token_cap > 0:
        prompt_tokens = min(prompt_tokens, prompt_token_cap)
    requested_output = int(
        entry.get("request_max_output_tokens")
        or entry.get("output_tokens")
        or entry.get("thinking_tokens")
        or 1
    )
    output_tokens = max(1, requested_output)
    if max_output_token_cap is not None:
        output_tokens = min(output_tokens, max_output_token_cap)
    prompt = " ".join(["token"] * max(prompt_tokens, 1))
    if entry.get("prompt"):
        prompt = str(entry["prompt"])
    request_payload: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": output_tokens,
    }
    if request_overrides:
        request_payload.update(request_overrides)
    started = time.monotonic()
    response = requests.post(
        f"{endpoint.rstrip('/')}/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "X-Lumo-Request-Class": str(entry.get("class") or entry.get("request_class") or "eval"),
        },
        json=request_payload,
        timeout=max(60, output_tokens * 3),
    )
    wall_s = time.monotonic() - started
    _raise_for_status_with_body(response)
    payload = response.json()
    usage = payload.get("usage") if isinstance(payload, dict) else {}
    if not isinstance(usage, dict):
        usage = {}
    return {
        "request_id": request_id,
        "prompt_tokens_requested": prompt_tokens,
        "max_output_tokens": output_tokens,
        "wall_s": round(wall_s, 6),
        "usage": usage,
        "status": payload.get("status") if isinstance(payload, dict) else None,
    }


def _run_warm_batch(
    *,
    entries: list[dict[str, Any]],
    endpoint: str,
    api_key: str,
    model: str,
    window_id: int,
    warm_concurrency: int,
    prompt_token_cap: int,
    max_output_token_cap: int | None,
    request_overrides: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    max_workers = max(1, min(warm_concurrency, len(entries)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _post_response,
                endpoint=endpoint,
                api_key=api_key,
                model=model,
                entry=entry,
                request_id=f"window-{window_id:03d}-warm-{index + 1:03d}",
                prompt_token_cap=prompt_token_cap,
                max_output_token_cap=max_output_token_cap,
                request_overrides=request_overrides,
            ): index
            for index, entry in enumerate(entries)
        }
        rows: list[dict[str, Any]] = []
        for future in as_completed(futures):
            row = future.result()
            row["warm_index"] = futures[future] + 1
            rows.append(row)
    return sorted(rows, key=lambda row: int(row["warm_index"]))


def measure(args: argparse.Namespace) -> dict[str, Any]:
    if args.swe_bench_subset_json:
        workload, seed_rows = _load_swe_bench_entries(Path(args.swe_bench_subset_json))
        seed_path = Path(args.swe_bench_subset_json)
    else:
        workload, seed_path = _load_workload(args.workload_file)
        seed_rows = _load_seed(seed_path)
    request_overrides = json.loads(args.request_overrides_json)
    if not isinstance(request_overrides, dict):
        raise RuntimeError("--request-overrides-json must decode to a JSON object")
    health = requests.get(args.health_url, timeout=20)
    _raise_for_status_with_body(health)
    if args.reset_prefix_cache:
        reset = requests.post(args.reset_prefix_cache_url, headers={"Authorization": f"Bearer {args.api_key}"}, timeout=30)
        _raise_for_status_with_body(reset)
    spec_trace_pre = _jsonl_count(SPEC_TRACE)
    fb_debug_pre = _jsonl_count(FB_DEBUG)
    fb_overhead_pre = _jsonl_count(FB_OVERHEAD_DEBUG)
    fb_internal_pre = _jsonl_count(FB_INTERNAL_DEBUG)
    fa_unique_gdn_pre = _jsonl_count(FA_UNIQUE_GDN_DEBUG)
    fa_replay_commit_pre = _jsonl_count(FA_REPLAY_COMMIT_DETAIL)
    cudagraph_runtime_pre = _jsonl_count(CUDAGRAPH_RUNTIME_DEBUG)
    max_output_cap = args.max_output_token_cap if args.max_output_token_cap > 0 else None
    warm_per_window = args.completions_per_task - args.cold_completions
    if warm_per_window < 1:
        raise RuntimeError("completions_per_task minus cold_completions must leave at least one warm completion")
    windows: list[dict[str, Any]] = []
    warm_started = time.monotonic()
    for window_index in range(args.task_count):
        base = window_index * args.completions_per_task
        cold_rows: list[dict[str, Any]] = []
        for cold_index in range(args.cold_completions):
            entry = seed_rows[(base + cold_index) % len(seed_rows)]
            cold_rows.append(
                _post_response(
                    endpoint=args.endpoint,
                    api_key=args.api_key,
                    model=args.model,
                    entry=entry,
                    request_id=f"window-{window_index + 1:03d}-cold-{cold_index + 1:03d}",
                    prompt_token_cap=args.prompt_token_cap,
                    max_output_token_cap=max_output_cap,
                    request_overrides=request_overrides,
                )
            )
        before_window_warm = _metrics(args.metrics_url)
        warm_entries = [
            seed_rows[(base + args.cold_completions + offset) % len(seed_rows)]
            for offset in range(warm_per_window)
        ]
        window_warm_started = time.monotonic()
        warm_rows = _run_warm_batch(
            entries=warm_entries,
            endpoint=args.endpoint,
            api_key=args.api_key,
            model=args.model,
            window_id=window_index + 1,
            warm_concurrency=args.warm_concurrency,
            prompt_token_cap=args.prompt_token_cap,
            max_output_token_cap=max_output_cap,
            request_overrides=request_overrides,
        )
        window_warm_elapsed = time.monotonic() - window_warm_started
        after_window_warm = _metrics(args.metrics_url)
        warm_summary = _metric_summary(
            before_window_warm,
            after_window_warm,
            request_count=len(warm_rows),
            elapsed_s=window_warm_elapsed,
        )
        windows.append(
            {
                "window_index": window_index + 1,
                "cold_completions": cold_rows,
                "warm_completions": warm_rows,
                "warm_metrics_consumption": warm_summary,
            }
        )
    warm_elapsed = time.monotonic() - warm_started
    aggregate = _aggregate_metric_summaries(
        [
            row["warm_metrics_consumption"]
            for row in windows
            if isinstance(row.get("warm_metrics_consumption"), dict)
        ],
        request_count=args.task_count * warm_per_window,
        elapsed_s=warm_elapsed,
    )
    spec_trace_summary = _summarize_spec_trace(
        _read_jsonl_range(SPEC_TRACE, spec_trace_pre, _jsonl_count(SPEC_TRACE)))
    unified_debug_summary = _summarize_unified_debug(
        _read_jsonl_range(FB_DEBUG, fb_debug_pre, _jsonl_count(FB_DEBUG)))
    fb_overhead_summary = _summarize_fb_overhead(
        _read_jsonl_range(FB_OVERHEAD_DEBUG, fb_overhead_pre, _jsonl_count(FB_OVERHEAD_DEBUG)))
    fb_internal_summary = _summarize_fb_internal(
        _read_jsonl_range(FB_INTERNAL_DEBUG, fb_internal_pre, _jsonl_count(FB_INTERNAL_DEBUG)))
    fa_unique_gdn_summary = _summarize_fa_unique_gdn(
        _read_jsonl_range(FA_UNIQUE_GDN_DEBUG, fa_unique_gdn_pre, _jsonl_count(FA_UNIQUE_GDN_DEBUG)))
    fa_replay_commit_summary = _summarize_fa_replay_commit(
        _read_jsonl_range(FA_REPLAY_COMMIT_DETAIL, fa_replay_commit_pre, _jsonl_count(FA_REPLAY_COMMIT_DETAIL)))
    cudagraph_runtime_summary = _summarize_cudagraph_runtime(
        _read_jsonl_range(CUDAGRAPH_RUNTIME_DEBUG, cudagraph_runtime_pre, _jsonl_count(CUDAGRAPH_RUNTIME_DEBUG)))
    decode_tps = None
    if aggregate.get("available"):
        step = aggregate.get("step_consumption") if isinstance(aggregate.get("step_consumption"), dict) else {}
        decode_tps = step.get("decode_tokens_per_s")
    speedup = float(decode_tps) / args.baseline_decode_tps if decode_tps is not None else None
    target_tps = (
        float(args.compare_baseline_tps)
        if args.compare_baseline_tps is not None
        else args.baseline_decode_tps * args.target_multiplier
    )
    comparison_enabled = not bool(args.record_only)
    comparison_pass = bool(decode_tps is not None and float(decode_tps) >= target_tps)
    return {
        "schema": "lumo.track_b.real_workload_first_five.v1",
        "measured_at": _now(),
        "source_reports": {
            "track_b": "docs/reports/auto_research/l0-warm-decode-quality-bounded-track-20260505.md",
            "l0c_cutlass": "docs/reports/auto_research/l0c-cutlass-round-20260505T204655Z.md",
        },
        "endpoint": args.endpoint,
        "metrics_url": args.metrics_url,
        "model": args.model,
        "workload_file": str(args.workload_file),
        "seed_trace": str(seed_path),
        "workload_distribution_id": workload.get("workload_distribution_id"),
        "task_count": args.task_count,
        "completions_per_task": args.completions_per_task,
        "cold_completions_discarded": args.cold_completions,
        "warm_completions_measured": warm_per_window,
        "warm_concurrency": args.warm_concurrency,
        "prompt_token_cap": args.prompt_token_cap,
        "max_output_token_cap": max_output_cap,
        "request_overrides": request_overrides,
        "baseline_decode_tps": args.baseline_decode_tps,
        "target_multiplier": args.target_multiplier,
        "target_decode_tps": target_tps,
        "compare_baseline_tps": args.compare_baseline_tps,
        "record_only": bool(args.record_only),
        "comparison_enabled": comparison_enabled,
        "decode_tps": decode_tps,
        "warm_decode_tps": decode_tps,
        "speedup_vs_baseline": speedup,
        "pass": True if args.record_only else comparison_pass,
        "comparison_pass": comparison_pass,
        "aggregate_warm_metrics_consumption": aggregate,
        "spec_trace_summary": spec_trace_summary,
        "unified_debug_summary": unified_debug_summary,
        "fb_overhead_summary": fb_overhead_summary,
        "fb_internal_summary": fb_internal_summary,
        "fa_unique_gdn_summary": fa_unique_gdn_summary,
        "fa_replay_commit_summary": fa_replay_commit_summary,
        "cudagraph_runtime_summary": cudagraph_runtime_summary,
        "stage_timer_summary": {
            "unified_verify_us": _timer_stats([
                int(r.get("verify_us", 0) or 0)
                for r in _read_jsonl_range(FB_DEBUG, fb_debug_pre, _jsonl_count(FB_DEBUG))
                if r.get("event") == "round_f_unified_step"
            ]),
            "unified_gdn_parent_gather_us": _timer_stats([
                int(r.get("gdn_parent_gather_us", 0) or 0)
                for r in _read_jsonl_range(FB_DEBUG, fb_debug_pre, _jsonl_count(FB_DEBUG))
                if r.get("event") == "round_f_unified_step"
            ]),
            "unified_commit_us": _timer_stats([
                int(r.get("commit_us", 0) or 0)
                for r in _read_jsonl_range(FB_DEBUG, fb_debug_pre, _jsonl_count(FB_DEBUG))
                if r.get("event") == "round_f_unified_step"
            ]),
            "state_copy_commit_enqueue_us": _timer_stats([
                int(r.get("commit_enqueue_us", 0) or 0)
                for r in _read_jsonl_range(FA_REPLAY_COMMIT_DETAIL, fa_replay_commit_pre, _jsonl_count(FA_REPLAY_COMMIT_DETAIL))
                if r.get("event") == "fa_activation_replay_commit_summary"
            ]),
        },
        "windows": windows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure Track B on the real L0c workload: first 5 completions, last 4 warm.")
    parser.add_argument("--workload-file", type=Path, default=REPO_ROOT / "benchmark_blueprints" / "workloads" / "responses-sdk-adapter-cutover-heavy" / "workload.yaml")
    parser.add_argument("--swe-bench-subset-json", default="",
                        help="Use a pinned scripts/build_swe_bench_subset.py JSON file as the real-task prompt source.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:9950/v1")
    parser.add_argument("--health-url", default="http://127.0.0.1:9950/health")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:9950/metrics")
    parser.add_argument("--reset-prefix-cache-url", default="http://127.0.0.1:9950/reset_prefix_cache")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument("--task-count", type=int, default=1)
    parser.add_argument("--completions-per-task", type=int, default=5)
    parser.add_argument("--cold-completions", type=int, default=1)
    parser.add_argument("--warm-concurrency", type=int, default=4)
    parser.add_argument("--prompt-token-cap", type=int, default=0)
    parser.add_argument("--max-output-token-cap", type=int, default=0)
    parser.add_argument("--request-overrides-json", default="{}")
    parser.add_argument("--baseline-decode-tps", type=float, default=7.5)
    parser.add_argument("--target-multiplier", type=float, default=5.0)
    parser.add_argument("--compare-baseline-tps", type=float, default=None,
                        help="Compare directly against this decode tps instead of baseline-decode-tps * target-multiplier.")
    parser.add_argument("--record-only", action="store_true",
                        help="Write the measurement JSON and exit 0 without enforcing a pass/fail throughput gate.")
    parser.add_argument("--reset-prefix-cache", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.task_count < 1:
        raise RuntimeError("--task-count must be >= 1")
    if args.completions_per_task < 2:
        raise RuntimeError("--completions-per-task must be >= 2")
    if args.cold_completions < 0:
        raise RuntimeError("--cold-completions must be >= 0")
    if args.warm_concurrency < 1:
        raise RuntimeError("--warm-concurrency must be >= 1")
    if args.baseline_decode_tps <= 0:
        raise RuntimeError("--baseline-decode-tps must be > 0")
    if args.target_multiplier <= 1 and args.compare_baseline_tps is None:
        raise RuntimeError("--target-multiplier must be > 1")
    if args.compare_baseline_tps is not None and args.compare_baseline_tps <= 0:
        raise RuntimeError("--compare-baseline-tps must be > 0")
    result = measure(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())

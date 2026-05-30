#!/usr/bin/env python3
"""Summarize one Round-F real-agentic SWE arm.

The Codex runner copies three timing streams into output/<exp>:
  - driver.log / per_task/* for task outcomes
  - per_req_spec_trace.jsonl for per-request spec-decode events
  - dgx_steptrace.jsonl for cumulative vLLM counters

The trace files may contain rows from earlier launches, so this script filters
by the experiment start timestamp in driver.log.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def _parse_time(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _driver_start(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    match = re.search(r"^=== \[(.*?)\]", text, re.M)
    return _parse_time(match.group(1)) if match else None


def _task_summary(exp_dir: Path) -> dict:
    tasks = {}
    for meta in list(exp_dir.glob("**/per_task/*/runner_metadata.json")) + list(
        exp_dir.glob("**/per_task/*/result.json")
    ):
        task = meta.parent.name
        try:
            data = json.loads(meta.read_text())
        except Exception:
            continue
        current = tasks.get(task, {})
        current["path"] = str(meta)
        if data.get("started_at"):
            current["started_at"] = data.get("started_at")
        if data.get("ended_at"):
            current["ended_at"] = data.get("ended_at")
        er = data.get("eval_report") or {}
        if er:
            current["verdict"] = er.get("verdict")
            current["resolved"] = bool(er.get("resolved", er.get("verdict") == "resolved"))
        elif data.get("resolved") is not None:
            current["resolved"] = bool(data.get("resolved"))
        elif data.get("verdict"):
            current["verdict"] = data.get("verdict")
            current["resolved"] = data.get("verdict") == "resolved"
        tasks[task] = current
    ended = [
        _parse_time(t["ended_at"])
        for t in tasks.values()
        if t.get("ended_at")
    ]
    resolved = sum(1 for t in tasks.values() if t.get("resolved"))
    return {
        "task_count": len(tasks),
        "resolved_count": resolved,
        "resolved_rate": (resolved / len(tasks)) if tasks else None,
        "tasks": tasks,
        "end_ts": max(ended) if ended else None,
    }


def _delta(rows: list[dict], key: str) -> float | None:
    if len(rows) < 2:
        return None
    return float(rows[-1].get(key, 0.0)) - float(rows[0].get(key, 0.0))


def _step_summary(rows: list[dict]) -> dict:
    if len(rows) < 2:
        return {}
    dt = float(rows[-1]["ts"]) - float(rows[0]["ts"])
    gen = _delta(rows, "gen") or 0.0
    drafts = _delta(rows, "drafts") or 0.0
    acc = _delta(rows, "acc") or 0.0
    draft_tok = _delta(rows, "draft") or 0.0
    dec = _delta(rows, "dec_sum") or 0.0
    pre = _delta(rows, "pre_sum") or 0.0
    iters = _delta(rows, "iter_cnt") or 0.0
    return {
        "window_s": dt,
        "generation_tokens": gen,
        "decode_tps": (gen / dt) if dt > 0 else None,
        "accepted_tokens": acc,
        "draft_tokens": draft_tok,
        "draft_events": drafts,
        "accept_per_event_steptrace": (acc / drafts) if drafts > 0 else None,
        "accept_per_draft_steptrace": (acc / draft_tok) if draft_tok > 0 else None,
        "mean_event_ms_wall": (dt * 1000.0 / drafts) if drafts > 0 else None,
        "request_decode_time_s": dec,
        "request_prefill_time_s": pre,
        "engine_steps": iters,
        "mean_engine_step_ms_wall": (dt * 1000.0 / iters) if iters > 0 else None,
        "mean_gpu_util": _mean([r.get("gpu_util") for r in rows]),
    }


def _mean(values):
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else None


def _nsight_tables(sqlite_path: Path) -> dict:
    if not sqlite_path.exists():
        return {"available": False}
    try:
        con = sqlite3.connect(str(sqlite_path))
        tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
    except sqlite3.Error as exc:
        return {"available": False, "error": str(exc)}
    finally:
        try:
            con.close()
        except Exception:
            pass
    has_kernel = any("KERNEL" in t.upper() or "CUPTI" in t.upper() for t in tables)
    return {
        "available": True,
        "table_count": len(tables),
        "has_cuda_kernel_tables": has_kernel,
        "tables_head": tables[:20],
    }


def summarize(exp_dir: Path, label: str, nodes: int | None, start_ts: float | None = None) -> dict:
    start = start_ts or _driver_start(exp_dir / "driver.log")
    task = _task_summary(exp_dir)
    end = task.get("end_ts")
    spec_rows = _read_jsonl(exp_dir / "per_req_spec_trace.jsonl")
    step_rows = _read_jsonl(exp_dir / "dgx_steptrace.jsonl")
    if start is not None:
        spec_rows = [r for r in spec_rows if float(r.get("ts", 0.0)) >= start - 2.0]
        step_rows = [r for r in step_rows if float(r.get("ts", 0.0)) >= start - 2.0]
    if end is not None:
        spec_rows = [r for r in spec_rows if float(r.get("ts", 0.0)) <= end + 5.0]
        step_rows = [r for r in step_rows if float(r.get("ts", 0.0)) <= end + 5.0]
    elif spec_rows:
        latest = max(float(r.get("ts", 0.0)) for r in spec_rows)
        step_rows = [r for r in step_rows if float(r.get("ts", 0.0)) <= latest + 5.0]

    acc = sum(int(r.get("acc", 0)) for r in spec_rows)
    draft = sum(int(r.get("draft", 0)) for r in spec_rows)
    dist = Counter(int(r.get("acc", 0)) for r in spec_rows)
    nsight_sqlite = next(exp_dir.glob("nsight_*.sqlite"), None)
    out = {
        "label": label,
        "exp_dir": str(exp_dir),
        "node_count": nodes,
        "start_ts": start,
        "end_ts": end,
        "spec_events": len(spec_rows),
        "acceptance": {
            "accepted_tokens": acc,
            "draft_tokens": draft,
            "accept_per_event": (acc / len(spec_rows)) if spec_rows else None,
            "accept_per_draft": (acc / draft) if draft else None,
            "acc_dist": dict(sorted(dist.items())),
        },
        "steptrace": _step_summary(step_rows),
        "tasks": task,
        "nsight": _nsight_tables(nsight_sqlite) if nsight_sqlite else {"available": False},
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--nodes", type=int, default=None)
    ap.add_argument("--start-ts", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    payload = summarize(Path(args.exp_dir), args.label, args.nodes, args.start_ts)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

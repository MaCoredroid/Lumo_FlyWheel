#!/usr/bin/env python3
"""FR14 promotion A/B: the WORK-SHAPE diff between two arms' censuses.

Pass 31 banked the fact this leans on: the per-step work shape is
data-independent -- 26 census counters were single-valued across 20,579 steps --
so a census is shape-checkable against literals rather than against a
distribution. That makes the C-vs-G census diff a real instrument: any counter
whose VALUE SET differs between the arms is a shape change, and the pass gate is
specified to change exactly two things (the drafter's pass count and the Arctic
main-tail length) and nothing else.

Output: for every dotted counter path, the distinct values seen in each arm, and
an explicit list of paths where the arms differ. Expected-different paths are
labelled from the spec so the unexpected ones stand out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MAX_DISTINCT = 12

# Everything the pass gate is SPECIFIED to move (suffix_pass_gating.md 11.1,
# 11.5) plus the run-identity fields that differ between any two runs.
EXPECTED_DIFFERENT_PREFIXES = (
    "drafter.mtp_forward_calls",
    "drafter.mtp_forward_rows",
    "drafter.main_tail_length",
    "drafter.arctic_requested_tokens",
    "drafter.arctic_lookup_calls",
    "drafter.carry_fill_slots",
    "drafter.rescue_chains",
    "drafter_runtime.mtp_forward_calls",
    "drafter_runtime.mtp_forward_rows",
    "drafter_runtime.graph_replays",
    "drafter_runtime.graph_captures",
    "drafter_runtime.graph_id",
    "drafter_runtime.graph_signature",
    "drafter_runtime.arctic_ledger",
    "drafter_runtime.arctic_requested_tokens",
    "drafter_runtime.arctic_lookup_calls",
    "drafter_runtime.rescue_carry_slots",
    "drafter_runtime.merge_fill",
    "drafter_runtime.segment",
    "drafter_runtime.passes",
    "gate_",
)
# Run identity: different every run regardless of lever.
IDENTITY_PREFIXES = (
    "event_id", "event_index", "forward_step_index", "producer_pid",
    "request_key_pack", "drafter_runtime.request_id", "drafter_runtime.graph_id",
    "drafter_runtime.physical_parent_sha256", "kernel_shape.",
    "taw.", "gdn.", "kv_remap.", "output_publish.",
)


def flatten(obj, prefix="", out=None):
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, list):
        out[prefix] = json.dumps(obj, sort_keys=True)[:200]
    else:
        out[prefix] = obj
    return out


def collect(path: Path) -> dict:
    values: dict[str, set] = {}
    n = 0
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if not isinstance(rec, dict) or "active_nodes" not in rec:
            continue
        n += 1
        for k, v in flatten(rec).items():
            s = values.setdefault(k, set())
            if len(s) <= MAX_DISTINCT:
                s.add(v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
    return {"steps": n, "values": values}


def render(s: set) -> object:
    if len(s) > MAX_DISTINCT:
        return f"<{len(s)}+ distinct>"
    try:
        return sorted(s, key=lambda x: (str(type(x)), str(x)))
    except Exception:
        return [str(x) for x in s]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control-census", required=True)
    ap.add_argument("--treated-census", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    c = collect(Path(a.control_census))
    g = collect(Path(a.treated_census))
    keys = sorted(set(c["values"]) | set(g["values"]))

    same, expected, unexpected, missing = [], [], [], []
    for k in keys:
        cv, gv = c["values"].get(k), g["values"].get(k)
        if cv is None or gv is None:
            missing.append({"path": k, "control": render(cv or set()),
                            "treated": render(gv or set())})
            continue
        if cv == gv:
            same.append(k)
            continue
        row = {"path": k, "control": render(cv), "treated": render(gv)}
        if k.startswith(IDENTITY_PREFIXES) or "sha256" in k:
            continue  # run identity, never a shape claim
        if k.startswith(EXPECTED_DIFFERENT_PREFIXES):
            expected.append(row)
        else:
            unexpected.append(row)

    out = {
        "schema": "fr14.promotion_ab.workshape.v1",
        "control_census": a.control_census,
        "treated_census": a.treated_census,
        "control_steps": c["steps"],
        "treated_steps": g["steps"],
        "identical_counter_paths": len(same),
        "expected_different": expected,
        "UNEXPECTED_DIFFERENT": unexpected,
        "present_in_one_arm_only": missing,
    }
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"steps: control={c['steps']} treated={g['steps']}")
    print(f"identical counter paths: {len(same)}")
    print(f"expected-different paths: {len(expected)}")
    print(f"UNEXPECTED-different paths: {len(unexpected)}")
    for row in unexpected:
        print(f"  ! {row['path']}: {row['control']} -> {row['treated']}")
    for row in missing:
        print(f"  ? one-arm-only {row['path']}: {row['control']} -> {row['treated']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

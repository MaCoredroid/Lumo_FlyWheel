#!/usr/bin/env python3
"""FR14 promotion A/B reducer: per-arm instruments + the section-11.7 boot checks.

Reads one arm runroot and emits a single JSON record. Nothing here is a verdict;
the verdict is written in promotion_ab_campaign.md by whoever read the numbers
and the traces.

INSTRUMENTS (pass 24 doctrine): step_wall_ms and s_per_fwd_gpu are the verdict
instruments for per-step levers. TPS and accept/event are REPORTED, and are a
real reading only for an acceptance-affecting lever (the pass gate); a byte-exact
lever cannot move them, so a TPS delta on such an arm is trajectory divergence.

SECTION 11.7 CHECKS (suffix_pass_gating.md), pre-registered before the first
armed boot:
  registry            two rows at passes=2, segment 0 and 1   (armed arm only)
  graph_replays       2 on cold steps, 1 on gated steps       (armed arm only)
  mtp_forward_calls   in {4, 2} and NO third value
  active_nodes        27 on every step
  verify_rows         32 on every step
  ungated signature   d9a4ddece41d146e9949b9f8ff7c2603b8948d157b28ef69244e44469b36150c
  warm-step rate      0.15 - 0.25                             (armed arm only)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

UNGATED_SIGNATURE = (
    "d9a4ddece41d146e9949b9f8ff7c2603b8948d157b28ef69244e44469b36150c"
)


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def census_report(path: Path) -> dict:
    recs = _load_jsonl(path)
    steps = [r for r in recs if isinstance(r, dict) and "drafter" in r and "active_nodes" in r]
    terminal = [r for r in recs if isinstance(r, dict) and "drafter_graph_registry" in r]
    rep: dict = {
        "census_path": str(path),
        "records": len(recs),
        "step_events": len(steps),
    }
    if not steps:
        rep["EMPTY"] = True
        return rep

    active = Counter(int(r.get("active_nodes", -1)) for r in steps)
    verify = Counter(int(r.get("verify_rows", -1)) for r in steps)
    mtp = Counter(int(r["drafter"].get("mtp_forward_calls", -1)) for r in steps)
    tail = Counter(int(r["drafter"].get("main_tail_length", -1)) for r in steps)
    pairs = Counter(
        (
            int(r["drafter"].get("mtp_forward_calls", -1)),
            int(r["drafter"].get("main_tail_length", -1)),
        )
        for r in steps
    )
    replays = Counter(
        int(r.get("drafter_runtime", {}).get("graph_replays", -1)) for r in steps
    )
    sigs = Counter(
        str(r.get("drafter_runtime", {}).get("graph_signature", "")) for r in steps
    )
    physical = Counter(int(r.get("physical_drafts", -1)) for r in steps)

    # The gate publishes GateDecision.as_census(); it may land at the top level
    # or under drafter/drafter_runtime depending on the emitter. Look in all
    # three rather than assume, and fall back to the pass count (which the
    # census validator already binds to the gated shape) if it is absent.
    gate_fired = Counter()
    gate_reason = Counter()
    for r in steps:
        for holder in (r, r.get("drafter", {}), r.get("drafter_runtime", {})):
            if isinstance(holder, dict) and "gate_fired" in holder:
                gate_fired[bool(holder["gate_fired"])] += 1
                gate_reason[str(holder.get("gate_reason", ""))] += 1
                break
    if gate_fired:
        rep["gate_fired"] = {str(k): v for k, v in gate_fired.items()}
        rep["gate_reason"] = dict(gate_reason)

    gated_steps = mtp.get(2, 0)
    rep.update(
        {
            "active_nodes": dict(active),
            "verify_rows": dict(verify),
            "mtp_forward_calls": dict(mtp),
            "main_tail_length": dict(tail),
            "mtp_tail_pairs": {f"{a},{b}": n for (a, b), n in pairs.items()},
            "graph_replays": dict(replays),
            "graph_signatures": dict(sigs),
            "physical_drafts": dict(physical),
            "gated_steps": gated_steps,
            "warm_step_rate": gated_steps / len(steps),
        }
    )
    if terminal:
        reg = terminal[-1].get("drafter_graph_registry")
        rep["drafter_graph_registry"] = reg
        rep["forward_graph_registry"] = terminal[-1].get("forward_graph_registry")

    # ---- section 11.7, evaluated ------------------------------------------
    checks = {
        "active_nodes_27_every_step": set(active) == {27},
        "verify_rows_32_every_step": set(verify) == {32},
        "mtp_forward_calls_only_4_or_2": set(mtp) <= {4, 2},
        "mtp_tail_pairs_legal": set(pairs) <= {(4, 6), (2, 8)},
        "ungated_signature_present": UNGATED_SIGNATURE in sigs,
    }
    if gated_steps:
        checks["graph_replays_only_1_or_2"] = set(replays) <= {1, 2}
        checks["warm_step_rate_in_0.15_0.25"] = (
            0.15 <= rep["warm_step_rate"] <= 0.25
        )
        reg = rep.get("drafter_graph_registry") or []
        two_halves = sorted(
            (int(r.get("passes", -1)), int(r.get("segment", -1)))
            for r in reg
            if isinstance(r, dict) and int(r.get("batch_size", -1)) == 1
        )
        checks["registry_two_rows_passes2_segment_0_and_1"] = two_halves == [
            (2, 0),
            (2, 1),
        ]
    else:
        checks["graph_replays_only_1"] = set(replays) <= {1}
    rep["section_11_7_checks"] = checks
    rep["section_11_7_all_pass"] = all(checks.values())
    return rep


def deploy_report(path: Path) -> dict:
    if not path.is_file():
        return {"deploy_speed_path": str(path), "MISSING": True}
    d = json.loads(path.read_text())
    keep = (
        "arm", "batch_size", "n_tasks", "task_instance_ids", "instrument",
        "s_per_fwd", "s_per_fwd_gpu", "s_per_fwd_gpu_per_forward",
        "accept_per_event", "committed_per_event", "derived_tps",
        "derived_tps_gpu", "derived_tps_fullstep_gpu",
        "per_request_decode_tps", "aggregate_decode_tps",
        "effective_concurrency", "prefill_frac", "step_wall_ms",
        "wall_steps_measured", "floor_ms", "floor_ratio", "rows_per_step",
        "drafter_gpu_ms_per_step", "drafter_gpu_seconds",
        "committer_gpu_ms_per_step", "committer_gpu_seconds",
        "overhead_other_ms_per_event", "events_per_step",
        "aggregate_window_wall_s", "draft_vocab_k", "draft_vocab_root",
    )
    out = {"deploy_speed_path": str(path)}
    for k in keep:
        if k in d:
            out[k] = d[k]
    # step wall and phase spans live under nested reduction blocks in some
    # schema revisions; carry them verbatim when present.
    for k in ("per_task", "spans", "phase_spans", "step_wall", "gpu_spans",
              "bracket_reduction", "work_census_gate"):
        if k in d:
            out[k] = d[k]
    return out


def sidecar(path: Path) -> dict:
    if not path.is_file():
        return {"path": str(path), "MISSING": True}
    try:
        return {"path": str(path), "data": json.loads(path.read_text())}
    except Exception as exc:
        return {"path": str(path), "UNPARSED": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runroot", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    root = Path(a.runroot)
    armdir = root / a.arm
    sidecar_dir = Path("output/fr13_sfwd_sidecar")

    rep = {
        "schema": "fr14.promotion_ab.arm_reduction.v1",
        "label": a.label,
        "runroot": str(root),
        "arm": a.arm,
    }
    meta = root / "arm_meta.txt"
    if meta.is_file():
        rep["arm_meta"] = meta.read_text()
    env = armdir / "container_env.txt"
    if env.is_file():
        wanted = (
            "FR13_FIXED32_MODE", "FR13_DRAFT_VOCAB_K", "FR13_DRAFT_VOCAB_ROOT",
            "FR13_NEEDS_ALLOW", "FR13_FA2_QROW32_B1_PRODUCTION_ARM",
            "FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE",
            "FR13_FA2_QROW32_B1_SOURCE_COMMIT", "FR14_FUSED_DRAFT_TOPK",
            "FR14_FUSED_DRAFT_TOPK_SHA256", "FR14_SUFFIX_PASS_GATE",
            "FR14_SUFFIX_PASS_GATE_NGRAM", "FR14_SUFFIX_PASS_GATE_MIN_AGREE",
            "FR13_DFWD_SPLIT", "FR13_LFWD_GPU_TIMER",
            "FR13_HOST_TAIL_PREP_BAKE", "FR10_METRICS", "MAX_NUM_SEQS",
            "SWE_CONCURRENCY", "FR13_CAMPAIGN_TASK_BUDGET_S",
            "FR13_FIXED32_ACTIVE_NODES", "FR13_FIXED32_PHYSICAL_DRAFTS",
        )
        picked = {}
        for line in env.read_text(errors="replace").splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                if k in wanted:
                    picked[k] = v
        rep["container_env"] = picked

    census = armdir / "logs" / "fr13_fixed32_work_census.jsonl"
    rep["census"] = census_report(census) if census.is_file() else {"MISSING": True}

    deploys = sorted(armdir.glob("deploy_speed_*.json"))
    rep["deploy_speed"] = [deploy_report(p) for p in deploys]

    rep["spans"] = {
        "sfwd": sidecar(sidecar_dir / f"{a.arm}.json"),
        "dfwd": sidecar(sidecar_dir / f"{a.arm}_dfwd.json"),
        "cfwd": sidecar(sidecar_dir / f"{a.arm}_cfwd.json"),
        "lfwd": sidecar(sidecar_dir / f"{a.arm}_lfwd.json"),
        "dfwd_split": sidecar(armdir / "logs" / "fr13_dfwd_split.json"),
    }
    health = armdir / "health.json"
    if health.is_file():
        rep["health"] = json.loads(health.read_text())

    Path(a.out).write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")

    c = rep["census"]
    print(f"== {a.label} ==")
    if c.get("MISSING") or c.get("EMPTY"):
        print("  NO CENSUS EVENTS — the arm produced no measured step "
              "(a refused or dead arm; read its container tail, not this file)")
        for d in rep["deploy_speed"]:
            if not d.get("MISSING"):
                print(f"  deploy: {d}")
        return 0
    if c.get("MISSING"):
        print("  NO CENSUS")
    else:
        print(f"  steps={c['step_events']} active_nodes={c['active_nodes']} "
              f"verify_rows={c['verify_rows']}")
        print(f"  mtp_forward_calls={c['mtp_forward_calls']} "
              f"pairs={c['mtp_tail_pairs']} replays={c['graph_replays']}")
        print(f"  warm_step_rate={c['warm_step_rate']:.4f} "
              f"signatures={list(c['graph_signatures'])}")
        print(f"  section_11_7: {c['section_11_7_checks']}")
    for d in rep["deploy_speed"]:
        if d.get("MISSING"):
            continue
        print(f"  step_wall_ms={d.get('step_wall_ms')} "
              f"s_per_fwd_gpu={d.get('s_per_fwd_gpu')} "
              f"floor_ratio={d.get('floor_ratio')} "
              f"steps={d.get('wall_steps_measured')}")
        print(f"  drafter_ms/step={d.get('drafter_gpu_ms_per_step')} "
              f"committer_ms/step={d.get('committer_gpu_ms_per_step')} "
              f"overhead_other={d.get('overhead_other_ms_per_event')}")
        print(f"  accept={d.get('accept_per_event')} "
              f"committed={d.get('committed_per_event')} "
              f"per_req_tps={d.get('per_request_decode_tps')} "
              f"prefill_frac={d.get('prefill_frac')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

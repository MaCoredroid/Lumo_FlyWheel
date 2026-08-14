#!/usr/bin/env python3
"""FR13 B4 exact16 AGENT-QUALITY-CONTROL reducer.

Mark's ruling (2026-08-12, restated 2026-08-13): exact16 is QUALITY CONTROL at
batched-optimization milestones.  It proves the speed levers did not degrade
task solving.  It is NOT a timing run.  This reducer therefore reads only
agent-behavioural evidence -- per-task resolve verdicts, patch bytes, completion
classes and garble signatures -- and refuses to emit any timing statistic.

THE BAND IS NOT INVENTED HERE.  The criteria are the three axes of gate (c) of
`results/fr13_tier_b_qualification_policy_20260809/README.md` sec.3:

  c1  resolve rate in band 8-11 of 16.  The FLOOR is the gate (user-mandated
      2026-07-18 resolve gate, "drifting below = issue signal").  The CEILING IS
      NOT A FAILURE CONDITION -- it is recorded so a suspiciously high number is
      read as "replicate before claiming", the campaign's own standing caution.
  c2  zero garble REGRESSION, relative to the incumbent -- never an absolute
      zero, which the tree-default arm does not satisfy and never has.
  c3  zero give-up anomalies (patch_bytes == 0 for behavioural reasons; the
      setup_loop and infra_stall_suspect causes are explicitly carved out).

WHAT THIS ARTIFACT IS NOT.  It is not a Tier-B qualification: gates (a) shadow
logit-epsilon and (b) acceptance parity are not run here and no `tier_b_qualified`
field is emitted.  Nor does it confer a serving default -- see `scope_note`.

Offline, idempotent, and safe to run repeatedly over banked bytes.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fr13.b4.exact16_qc.v1"
RUN_CLASS = "exact16_qc"

# Byte-pinned identically to fr13_floor_gate.EVIDENCE_SETS[16].
SUBSET_RELATIVE = "config/fr13_fixed32/subset_b4_sixteen.json"
SUBSET_SHA256 = "47b0a3c9be49e2cb5f7e7217ae03c267a05359f269f3e3b038942f57d7dc0b5c"
CANONICAL_TASK_IDS: tuple[str, ...] = (
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

# results/fr13_tier_b_qualification_policy_20260809/README.md sec.3 c1.
BAND_LOW = 8
BAND_HIGH = 11
BAND_SOURCE = "results/fr13_tier_b_qualification_policy_20260809/README.md#3c1"

# sec.3 c3: the two non-behavioural empty-patch causes that are NOT give-ups.
NON_BEHAVIORAL_EMPTY_PATCH_CAUSES = frozenset({"setup_loop", "infra_stall_suspect"})

DOES_NOT_CLAIM = (
    "timing",
    "acceptance",
    "exact4_comparability",
    "hardware_floor_acceptance",
    "tier_b_qualification",
    "serving_default",
)

SCOPE_NOTE = (
    "This is Mark's exact16 AGENT QUALITY CONTROL pass, not a timing run and not "
    "a Tier-B qualification. It asks one question: with the batched speed levers "
    "engaged, do task outcomes on the byte-pinned 16-task evidence set stay inside "
    "the historical behavioural band? "
    "THE CONFIGURATION UNDER TEST IS A CANDIDATE, NOT A DEFAULT. Promoting the "
    "padded B4 GQA-pair arm to the DEFAULT B4 serving arm -- the B4 analogue of the "
    "B1 flip at 99a511319 -- IS NOT RULED BY MARK. This pass validates exactly that "
    "candidate configuration so his decision has agent-quality evidence under it; "
    "passing here confers no default and authorises no flip. "
    "The task-budget cap is OFF as the ruling requires, and structurally so: the cap "
    "does not exist at this commit (it lives on codex/fr13-b1-flip-and-13398-cap-"
    "20260813 at 6cbfcab9d, which is not an ancestor of HEAD), and AGENT_WALL_S was "
    "additionally passed empty. No truncated-trace union is written. "
    "Per the Tier-B policy's own limits section, a pass here means 'no behavioural "
    "difference DETECTABLE AT n=16' -- at this sample size the band detects a broken "
    "agent, not a cost of one or two resolves."
)

MILESTONE_BATCH = {
    "mamba_narrowing_default_on": {
        "commit": "749f83af6",
        "in_this_tree": True,
        "note": "FR13_MAMBA_SPEC_BLOCKS_CDIV default 1 since the 2026-08-10 promotion.",
    },
    "b1_gqa_pair_production_default": {
        "commit": "99a511319",
        "in_this_tree": False,
        "note": (
            "On codex/fr13-b1-flip-and-13398-cap-20260813, NOT an ancestor of this "
            "tree's HEAD. This B4 QC pass therefore does NOT exercise the B1 flip and "
            "says nothing about it."
        ),
    },
    "b4_gqa_pair_padded_candidate": {
        "commit": "a55df8dd2",
        "in_this_tree": True,
        "note": (
            "Byte-gated qualified widths [3, 4]; served here as the production arm. "
            "Sealed this morning as SEALED_HYDRA27_GAIN (mean 27.0 ms, lower bound "
            "10.8 ms) -- cited for provenance only; this artifact makes no timing claim."
        ),
    },
}


# --------------------------------------------------------------- arm reading --
def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _strong_garble_flags(task_dir: Path) -> tuple[int, list[str], bool]:
    """Reuse the sealed detector rather than reimplementing its heuristics.

    Returns (n_strong, strong_flags, trace_found).  A missing trace is reported
    as such and never silently scored as clean.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fr13_garble_watch as gw  # noqa: PLC0415

    trace = gw._find_trace(str(task_dir))
    if not trace:
        return 0, [], False
    analysis = gw.analyze_trace(trace)
    strong = [
        f for f in analysis["flags"]
        if f.startswith(("NEAR-NEIGHBOR", "ERROR-LOOP"))
    ]
    return len(strong), strong, True


def read_arm(arm_dir: Path, *, with_garble: bool = True) -> dict[str, Any]:
    """Read one served 16-task arm into a behavioural record.

    FAIL-CLOSED: an arm whose evidence is incomplete is returned with
    `usable=False` and an explicit reason, never repaired and never guessed at.
    """
    arm_dir = arm_dir.resolve()
    record: dict[str, Any] = {
        "arm_dir": str(arm_dir),
        "arm": arm_dir.name,
        "usable": False,
        "reason": None,
        "tasks": {},
    }
    verified = arm_dir / "swe_out" / "verified"
    health_path = arm_dir / "health.json"
    summary_path = verified / "campaign_summary.json"
    if not verified.is_dir():
        record["reason"] = f"no swe_out/verified under {arm_dir}"
        return record
    if not health_path.is_file():
        record["reason"] = f"no health.json under {arm_dir}"
        return record

    health = _read_json(health_path)
    record["swe_orchestrator_rc"] = health.get("swe_orchestrator_rc")
    record["swe_window_wall_s"] = health.get("swe_window_wall_s")
    if summary_path.is_file():
        summary = _read_json(summary_path)
        record["campaign_summary"] = {
            "instances_total": summary.get("instances_total"),
            "verdict_counts": summary.get("verdict_counts"),
            "failure_mode_counts": summary.get("failure_mode_counts"),
            "resolved_rate": summary.get("resolved_rate"),
            "started_at": summary.get("started_at"),
            "ended_at": summary.get("ended_at"),
        }

    for entry in health.get("tasks", []):
        iid = entry.get("instance_id")
        if iid is None:
            continue
        task: dict[str, Any] = {
            "verdict": entry.get("verdict"),
            "resolved": entry.get("verdict") == "resolved",
            "patch_bytes": entry.get("patch_bytes"),
            "codex_elapsed_s": entry.get("codex_elapsed_s"),
            "codex_timed_out": entry.get("codex_timed_out"),
        }
        meta_path = verified / "per_task" / iid / "runner_metadata.json"
        if meta_path.is_file():
            meta = _read_json(meta_path)
            agent = meta.get("agent") or meta.get("codex") or {}
            report = meta.get("eval_report") or {}
            task.update(
                {
                    "failure_mode": report.get("failure_mode"),
                    "eval_verdict": report.get("verdict"),
                    "eval_passed": report.get("passed"),
                    "agent_exit_code": agent.get("exit_code"),
                    "agent_timed_out": agent.get("timed_out"),
                    "agent_network_drop": agent.get("network_drop"),
                    "empty_patch_retry_cause": (
                        (meta.get("empty_patch_retry") or {}).get("cause")
                    ),
                }
            )
            # The health.json verdict and the eval report must agree. A silent
            # disagreement is the exact shape of a mis-scored campaign.
            if report.get("verdict") not in (None, task["verdict"]):
                task["verdict_disagreement"] = True
        if with_garble:
            n_strong, flags, trace_found = _strong_garble_flags(
                verified / "per_task" / iid
            )
            task["strong_garble_flags"] = n_strong
            task["strong_garble_detail"] = flags
            task["trace_found"] = trace_found
        record["tasks"][iid] = task

    present = set(record["tasks"])
    missing = [t for t in CANONICAL_TASK_IDS if t not in present]
    extra = sorted(present - set(CANONICAL_TASK_IDS))
    record["missing_task_ids"] = missing
    record["extra_task_ids"] = extra
    record["instances_total"] = len(record["tasks"])
    record["resolved"] = sum(1 for t in record["tasks"].values() if t["resolved"])
    record["giveups"] = sum(
        1
        for t in record["tasks"].values()
        if t.get("patch_bytes") == 0
        and t.get("empty_patch_retry_cause") not in NON_BEHAVIORAL_EMPTY_PATCH_CAUSES
    )
    record["non_behavioral_empty_patches"] = sum(
        1
        for t in record["tasks"].values()
        if t.get("patch_bytes") == 0
        and t.get("empty_patch_retry_cause") in NON_BEHAVIORAL_EMPTY_PATCH_CAUSES
    )
    record["tasks_with_strong_garble"] = sum(
        1 for t in record["tasks"].values() if t.get("strong_garble_flags", 0) > 0
    )
    record["strong_garble_flag_total"] = sum(
        t.get("strong_garble_flags", 0) for t in record["tasks"].values()
    )
    record["traces_missing"] = sorted(
        iid for iid, t in record["tasks"].items()
        if with_garble and not t.get("trace_found", True)
    )
    record["agent_timeouts"] = sum(
        1 for t in record["tasks"].values() if t.get("agent_timed_out")
    )
    record["agent_nonzero_exit"] = sum(
        1 for t in record["tasks"].values()
        if t.get("agent_exit_code") not in (None, 0)
    )
    record["network_drops"] = sum(
        1 for t in record["tasks"].values() if t.get("agent_network_drop")
    )
    record["verdict_disagreements"] = sorted(
        iid for iid, t in record["tasks"].items() if t.get("verdict_disagreement")
    )
    failure_modes: dict[str, int] = {}
    for t in record["tasks"].values():
        key = t.get("failure_mode") or "unknown"
        failure_modes[key] = failure_modes.get(key, 0) + 1
    record["failure_mode_counts"] = dict(sorted(failure_modes.items()))

    if missing:
        record["reason"] = f"incomplete: {len(missing)} canonical task(s) absent"
        return record
    if extra:
        record["reason"] = f"off-subset: {len(extra)} non-canonical task(s) present"
        return record
    record["usable"] = True
    return record


# ------------------------------------------------------ historical reference --
def build_reference(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The historical per-task resolve profile: an empirical frequency, not a model."""
    usable = [r for r in records if r["usable"]]
    profile: dict[str, Any] = {}
    for iid in CANONICAL_TASK_IDS:
        ran = [r for r in usable if iid in r["tasks"]]
        resolved = [r for r in ran if r["tasks"][iid]["resolved"]]
        n_ran, n_res = len(ran), len(resolved)
        rate = (n_res / n_ran) if n_ran else None
        if rate is None:
            klass = "no_data"
        elif rate == 1.0:
            klass = "always"
        elif rate == 0.0:
            klass = "never"
        else:
            klass = "sometimes"
        profile[iid] = {
            "arms_ran": n_ran,
            "arms_resolved": n_res,
            "resolve_frequency": rate,
            "class": klass,
        }
    counts = [r["resolved"] for r in usable]
    aggregate = {
        "n_reference_arms": len(usable),
        "n_reference_arms_rejected": len(records) - len(usable),
        "resolve_counts": counts,
        "min": min(counts) if counts else None,
        "max": max(counts) if counts else None,
        "mean": (statistics.fmean(counts) if counts else None),
        "median": (statistics.median(counts) if counts else None),
        "stdev": (statistics.stdev(counts) if len(counts) > 1 else None),
        "giveups_total": sum(r["giveups"] for r in usable),
        "giveups_per_arm": [r["giveups"] for r in usable],
        "arms_with_any_giveup": sum(1 for r in usable if r["giveups"] > 0),
        "tasks_with_strong_garble_max": (
            max((r["tasks_with_strong_garble"] for r in usable), default=None)
        ),
        "tasks_with_strong_garble_per_arm": [
            r["tasks_with_strong_garble"] for r in usable
        ],
        # How the POLICY band behaves on the banked population. Reported so a
        # borderline QC verdict is read against what the campaign actually does,
        # not against an idealisation of it. This calibrates; it does not gate.
        "reference_arms_at_or_above_band_low": sum(
            1 for c in counts if c >= BAND_LOW
        ),
        "reference_arms_below_band_low": sum(1 for c in counts if c < BAND_LOW),
        "reference_arms_above_band_high": sum(1 for c in counts if c > BAND_HIGH),
    }
    return {"per_task": profile, "aggregate": aggregate}


# ------------------------------------------------------------------- verdict --
def evaluate(qc: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """The three axes of Tier-B gate (c), applied to a QC pass."""
    resolved = qc["resolved"]
    total = qc["instances_total"]

    # c1 -- the FLOOR is the gate; the ceiling is advisory by policy.
    c1_pass = resolved >= BAND_LOW
    above_high_water = resolved > BAND_HIGH
    agg = reference["aggregate"]
    c1_note = (
        f"{resolved}/{total} resolved; band {BAND_LOW}-{BAND_HIGH}. "
        + (
            "above the recorded high water -- replicate before claiming, not a failure"
            if above_high_water
            else ("in band" if c1_pass else "BELOW THE MANDATED FLOOR")
        )
        + f" Banked reference: {agg['n_reference_arms']} arm(s), counts "
        f"min {agg['min']} / median {agg['median']} / max {agg['max']} / "
        f"mean {agg['mean']}; {agg.get('reference_arms_below_band_low')} of them "
        f"sit BELOW the policy floor, so a floor miss by one is inside the banked "
        "spread and must be read as a signal to replicate, not as proof of damage."
    )

    # c2 -- RELATIVE, never absolute. The reference is the banked incumbent
    # population; a QC pass may not exceed the worst arm the campaign has banked.
    ref_max = reference["aggregate"]["tasks_with_strong_garble_max"]
    qc_garble = qc["tasks_with_strong_garble"]
    if ref_max is None:
        c2_pass = False
        c2_note = "no usable reference arm carries a garble reading; cannot judge relatively"
    else:
        c2_pass = qc_garble <= ref_max
        c2_note = (
            f"{qc_garble} task(s) show a strong garble signature; "
            f"banked reference worst arm is {ref_max}. "
            "Criterion is relative per policy sec.3 c2 -- an absolute zero is not "
            "the bar and is not satisfiable by the tree-default arm."
        )
    if qc.get("traces_missing"):
        c2_pass = False
        c2_note += f" MISSING TRACES: {qc['traces_missing']} -- cannot score clean."

    # c3 -- zero give-up anomalies.
    c3_pass = qc["giveups"] == 0
    ref_giveup_arms = reference["aggregate"].get("arms_with_any_giveup")
    ref_n = reference["aggregate"]["n_reference_arms"]
    c3_note = (
        f"{qc['giveups']} give-up(s) (patch_bytes == 0, behavioural); "
        f"{qc['non_behavioral_empty_patches']} carved-out non-behavioural empty patch(es). "
        f"Banked context: {ref_giveup_arms}/{ref_n} reference arm(s) carry at least one "
        "give-up, so the policy's zero bar is stricter than the campaign's observed "
        "background. The bar is applied as written and is not relaxed here."
    )

    # Completion-class profile: normal means the campaign ran to completion with
    # no infrastructure damage masquerading as agent behaviour.
    completion_normal = (
        qc.get("swe_orchestrator_rc") == 0
        and qc["agent_nonzero_exit"] == 0
        and qc["network_drops"] == 0
        and not qc["verdict_disagreements"]
        and not qc["missing_task_ids"]
        and not qc["extra_task_ids"]
    )

    # Per-task regression read against the historical profile.
    regressions, recoveries = [], []
    for iid in CANONICAL_TASK_IDS:
        ref = reference["per_task"].get(iid, {})
        got = qc["tasks"].get(iid, {}).get("resolved")
        if got is None or ref.get("class") == "no_data":
            continue
        if ref["class"] == "always" and got is False:
            regressions.append(iid)
        if ref["class"] == "never" and got is True:
            recoveries.append(iid)

    verdicts = {
        "c1_resolve_in_band": c1_pass,
        "c2_no_garble_regression": c2_pass,
        "c3_zero_giveups": c3_pass,
        "completion_class_normal": completion_normal,
    }
    qc_pass = all(verdicts.values())
    return {
        "verdicts": {**verdicts, "exact16_qc_pass": qc_pass},
        "notes": {"c1": c1_note, "c2": c2_note, "c3": c3_note},
        "above_high_water": above_high_water,
        "always_resolves_regressions": regressions,
        "never_resolves_recoveries": recoveries,
    }


# ----------------------------------------------------------------- self-check --
def self_check() -> int:
    """Prove the reducer's own logic before any GPU time is spent on it."""
    ref_arms = [
        {
            "usable": True,
            "resolved": 9,
            "giveups": 0,
            "tasks_with_strong_garble": 2,
            "tasks": {
                iid: {"resolved": i < 9} for i, iid in enumerate(CANONICAL_TASK_IDS)
            },
        },
        {
            "usable": True,
            "resolved": 10,
            "giveups": 0,
            "tasks_with_strong_garble": 3,
            "tasks": {
                iid: {"resolved": i < 10} for i, iid in enumerate(CANONICAL_TASK_IDS)
            },
        },
        {"usable": False, "resolved": 0, "giveups": 0, "tasks_with_strong_garble": 0,
         "tasks": {}},
    ]
    reference = build_reference(ref_arms)
    agg = reference["aggregate"]
    assert agg["n_reference_arms"] == 2, agg
    assert agg["n_reference_arms_rejected"] == 1, agg
    assert agg["resolve_counts"] == [9, 10], agg
    assert agg["tasks_with_strong_garble_max"] == 3, agg
    # task 0..8 resolve in both arms; task 9 in one; tasks 10..15 in neither.
    assert reference["per_task"][CANONICAL_TASK_IDS[0]]["class"] == "always"
    assert reference["per_task"][CANONICAL_TASK_IDS[9]]["class"] == "sometimes"
    assert reference["per_task"][CANONICAL_TASK_IDS[15]]["class"] == "never"

    def mk(resolved_n, giveups=0, garble=0, rc=0, **kw):
        base = {
            "resolved": resolved_n,
            "instances_total": 16,
            "giveups": giveups,
            "non_behavioral_empty_patches": 0,
            "tasks_with_strong_garble": garble,
            "traces_missing": [],
            "swe_orchestrator_rc": rc,
            "agent_nonzero_exit": 0,
            "network_drops": 0,
            "verdict_disagreements": [],
            "missing_task_ids": [],
            "extra_task_ids": [],
            "tasks": {
                iid: {"resolved": i < resolved_n}
                for i, iid in enumerate(CANONICAL_TASK_IDS)
            },
        }
        base.update(kw)
        return base

    ok = evaluate(mk(9), reference)
    assert ok["verdicts"]["exact16_qc_pass"] is True, ok
    below = evaluate(mk(7), reference)
    assert below["verdicts"]["c1_resolve_in_band"] is False, below
    assert below["verdicts"]["exact16_qc_pass"] is False, below
    # The ceiling is NOT a failure condition.
    high = evaluate(mk(13), reference)
    assert high["verdicts"]["c1_resolve_in_band"] is True, high
    assert high["above_high_water"] is True, high
    assert high["verdicts"]["exact16_qc_pass"] is True, high
    # c2 is relative: 3 == banked worst passes, 4 fails.
    assert evaluate(mk(9, garble=3), reference)["verdicts"]["c2_no_garble_regression"]
    assert not evaluate(mk(9, garble=4), reference)["verdicts"]["c2_no_garble_regression"]
    # A missing trace can never score clean.
    assert not evaluate(mk(9, traces_missing=["x"]), reference)["verdicts"][
        "c2_no_garble_regression"
    ]
    # c3 and completion class.
    assert not evaluate(mk(9, giveups=1), reference)["verdicts"]["c3_zero_giveups"]
    assert not evaluate(mk(9, rc=1), reference)["verdicts"]["completion_class_normal"]
    # A task the reference always resolves, failed here, is surfaced.
    reg = evaluate(mk(0), reference)
    assert CANONICAL_TASK_IDS[0] in reg["always_resolves_regressions"], reg
    print("exact16 QC reducer self-check OK")
    return 0


# ----------------------------------------------------------------------- main --
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--qc-runroot", type=Path,
        help="output/fr13_b4_exact16_qc_<STAMP> written by the QC gate",
    )
    parser.add_argument(
        "--qc-arm", type=Path,
        help="served arm dir; defaults to the single arm under <qc-runroot>/arm_root",
    )
    parser.add_argument(
        "--reference-arm", type=Path, action="append", default=[],
        help="banked 16-task arm dir contributing to the historical profile (repeatable)",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    if args.self_check:
        return self_check()
    if args.qc_runroot is None:
        parser.error("--qc-runroot is required unless --self-check")

    repo = args.repo.resolve()
    subset = repo / SUBSET_RELATIVE
    import hashlib

    digest = hashlib.sha256(subset.read_bytes()).hexdigest()
    if digest != SUBSET_SHA256:
        print(f"FAIL: 16-task subset sha256 drift: {digest}", file=sys.stderr)
        return 2

    runroot = args.qc_runroot.resolve()
    qc_arm = args.qc_arm
    if qc_arm is None:
        arm_root = runroot / "arm_root"
        candidates = sorted(p for p in arm_root.glob("*") if p.is_dir())
        if len(candidates) != 1:
            print(
                f"FAIL: expected exactly one arm under {arm_root}, found {len(candidates)}",
                file=sys.stderr,
            )
            return 2
        qc_arm = candidates[0]

    qc = read_arm(qc_arm.resolve())
    references = [read_arm(p) for p in args.reference_arm]
    reference = build_reference(references)

    if not qc["usable"]:
        print(f"FAIL: QC arm is not usable: {qc['reason']}", file=sys.stderr)
    if reference["aggregate"]["n_reference_arms"] == 0:
        print("FAIL: no usable reference arm; the band has no comparator", file=sys.stderr)

    result = (
        evaluate(qc, reference)
        if qc["usable"] and reference["aggregate"]["n_reference_arms"]
        else {
            "verdicts": {"exact16_qc_pass": False},
            "notes": {"fatal": "unusable QC arm or empty reference population"},
        }
    )

    launcher_meta = {}
    meta_path = runroot / "launcher_meta.txt"
    if meta_path.is_file():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and " " not in line.split("=", 1)[0]:
                key, _, value = line.partition("=")
                launcher_meta.setdefault(key, value)

    artifact = {
        "schema": SCHEMA,
        "run_class": RUN_CLASS,
        "does_not_claim": list(DOES_NOT_CLAIM),
        "scope_note": SCOPE_NOTE,
        "milestone_batch": MILESTONE_BATCH,
        "band_policy_source": BAND_SOURCE,
        "band_low": BAND_LOW,
        "band_high": BAND_HIGH,
        "subset": SUBSET_RELATIVE,
        "subset_sha256": SUBSET_SHA256,
        "task_ids": list(CANONICAL_TASK_IDS),
        "qc_runroot": str(runroot),
        "launcher_meta": launcher_meta,
        "qc_arm": qc,
        "reference_arms": references,
        "historical_reference": reference,
        **result,
    }
    payload = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = args.out.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            print(f"FAIL: refusing to overwrite {out}", file=sys.stderr)
            return 2
        out.write_text(payload, encoding="utf-8")
        os.chmod(out, 0o400)
        print(f"wrote {out}")
    else:
        sys.stdout.write(payload)

    passed = bool(artifact["verdicts"].get("exact16_qc_pass"))
    print(
        f"exact16 QC: resolved={qc.get('resolved')}/{qc.get('instances_total')} "
        f"band={BAND_LOW}-{BAND_HIGH} pass={passed}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

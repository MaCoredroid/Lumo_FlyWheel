#!/usr/bin/env python3
"""Reduce a multi-pass B4 formal floor-gate campaign to ONE citable verdict.

WHAT THIS IS FOR
----------------
Every B4 timing pair banked before this gate is a SCREEN: one draw of four agent
trajectories, reported as a point estimate.  The campaign log's standing gap is
"formal Tail/Hydra statistical floor gate -- all pairs so far are screens".

This reducer closes that gap.  It is OFFLINE and READ-ONLY over recorded pass
runroots and it duplicates NO reduction mathematics:

  * bracket topology + work-census cross-gate  -> scripts/fr13_measure.py
    (already applied in-run; this reducer re-reads and re-checks its provenance)
  * phase / rate identities                     -> scripts/fr13_b4_timing_math.py
  * subset byte-binding, pinned t critical      -> scripts/fr13_floor_gate.py

WHY BETWEEN-PASS REPEATS
------------------------
fr13_floor_gate.py's B4 model is a moving-block bootstrap over the per-step time
series.  That is the right model for STEP WALL, which has thousands of samples
per arm and a measured CV around 2%.  It is the WRONG model for AGGREGATE TPS.

    aggregate_tps = events_per_step * per_request_step_tps

events_per_step is co-residency: a property of ONE realization of four agent
trajectories.  Banked B4 arms show events_per_step from 1.195 to 2.528 and stock
aggregate TPS from 28.1 to 35.0 (CV ~9%) on nominally comparable stacks, while
step wall moved only 254.8-267.2 ms.  A within-run bootstrap resamples blocks of
a single trajectory draw and cannot see the variance BETWEEN draws, so quoting it
as the uncertainty on aggregate TPS would materially understate it.

So aggregate TPS gets a BETWEEN-PASS interval over pass-level values, built with
the repo's already-pinned one-sided critical value (T95_ONE_SIDED[3] = 2.35336…
at four passes, df=3).  The citable number is the LOWER bound.

OUTLIER POLICY
--------------
There is none, deliberately, matching fr13_floor_gate.py: no trimming, no
winsorizing, no IQR fence, no z-score rejection.  A pass either satisfies every
structural gate or it is EXCLUDED WITH ITS REASON RECORDED.  Passes are rejected,
never cleaned.  Below the required included-pass count the verdict degrades to
NOT_EVALUATED_INSUFFICIENT_PASSES rather than quoting a narrower-N interval.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fr13_b4_timing_math import TimingMathError, phase_breakdown, positive

SCHEMA = "fr13.b4_formal_floor_gate.v1"
CLASSIFICATION = "real_swe_verified_exact4_b4_formal_floor_gate"

POOL16_SCHEMA = "fr13.b4_pool16_refill_timing.v1"
POOL16_CLASSIFICATION = "real_swe_verified_pool16_b4_refill_timing"

# The one-sided 95% Student-t criticals pinned by scripts/fr13_floor_gate.py:529.
# Reproduced (not re-derived) so this reducer imports cleanly without dragging in
# the 12k-line gate module; test_fr13_b4_formal_floor_gate.py asserts they match.
T95_ONE_SIDED = {3: 2.3533634348018264, 15: 1.7530503556925547}

# The canonical exact4 evidence set, byte-pinned by fr13_floor_gate.EVIDENCE_SETS[4].
EXACT4_SUBSET_SHA256 = (
    "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
)
EXACT4_TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
)

# The canonical 16-task pool, byte-pinned by fr13_floor_gate.EVIDENCE_SETS[16]. The
# exact4 set is its first four in the same order, which is why a pool16 arm can be
# CONTRASTED with an exact4 gate but never equated to one.
POOL16_SUBSET_SHA256 = (
    "47b0a3c9be49e2cb5f7e7217ae03c267a05359f269f3e3b038942f57d7dc0b5c"
)
POOL16_TASK_IDS = EXACT4_TASK_IDS + (
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

TOPOLOGIES = {
    "tail6_fixed32": {
        "logical_topology": "Tail23",
        "active_drafts": 23,
        "valid_mask": "0x7a9ce7ff",
    },
    "hydra27_fixed32": {
        "logical_topology": "Hydra27",
        "active_drafts": 27,
        "valid_mask": "0x7abdffff",
    },
}

# B=4 with FR13_B4_TASK_REFILL=0 admits all four tasks on one engine state, so the
# per-task /metrics brackets NEST.  Anything else at B=4 is a topology the in-run
# reducer is not entitled to sum -- see the nested-summation defect that once
# inflated B4 aggregates 1.7-2.6x.
REQUIRED_BRACKET_TOPOLOGY = "nested"
DEPLOY_SPEED_SCHEMA = "fr13.measure.deploy_speed.v1"
DEFAULT_MIN_PASSES = 4

# The gate must measure the stack the branch SHIPS -- not a hand-flipped one, and
# not a value frozen into this file.  Two different obligations, so two mechanisms:
#
#   CONTRACT_PINNED_STACK -- values the exact4 citability contract requires no
#     matter what any default says.  Task refill is the only one: the driver
#     states its output "is NOT exact4-citable without a contract update".
#
#   CANONICAL_DEFAULT_LEVERS -- levers whose expected value IS whatever
#     scripts/fr13_canonical_env.sh currently ships, resolved at reduce time.
#     These are recorded, not pinned.  Pinning them here would be exactly the
#     wrong thing: when Mark promotes a lever (as with the 2026-08-10 mamba
#     narrowing flip 0 -> 1), the gate must follow the shipped default and
#     measure the new stack, not reject every arm for disagreeing with a stale
#     constant.  What stays enforced is that every arm in the campaign ran the
#     SAME state and that the state matches the shipped default.
CONTRACT_PINNED_STACK = {"FR13_B4_TASK_REFILL": "0"}
CANONICAL_DEFAULT_LEVERS = (
    "FR13_MAMBA_SPEC_BLOCKS_CDIV",
    "FR13_FULL_ATTN_KV_FP8",
)

# --------------------------------------------------------------------------- #
# the pool-16 admission ledger                                                 #
# --------------------------------------------------------------------------- #
# Reproduced (not imported) from scripts/fr13_floor_gate.py:269-271 so this reducer
# stays importable without dragging in the 12k-line gate module -- the same
# arrangement T95_ONE_SIDED already uses above.  test_fr13_b4_pool16_refill_timing.py
# asserts these three still match their originals.
TASK_REFILL_SUMMARY_SCHEMA = "fr13.task_refill.summary.v1"
MIN_POOL_TIME_WEIGHTED_DEPTH = 3.2
MIN_POOL_FULL_WIDTH_FRACTION = 0.60
POOL_LEDGER_RELPATH = "swe_out/verified/fr13_task_refill_summary.json"
POOL_LEDGER_EVENTS_RELPATH = "swe_out/verified/fr13_task_refill_ledger.jsonl"


def read_pool_ledger(arm_dir: Path, *, slots: int, task_count: int) -> dict[str, Any]:
    """Gate the admission ledger of a pool arm.  Raises when it is not one.

    THE FLAG IS EVIDENCE, NOT A SCHEDULE LEVER.  FR13_B4_TASK_REFILL=1 is pinned ON
    for the pool16 class, but not because it makes anything faster: its own
    docstring (run_swe_bench_q36_a.py:3253-3256) records that
    ThreadPoolExecutor.map already backfills a worker the instant its job returns,
    so admission timing is identical with the flag off.  What the flag adds is this
    ledger -- the only artifact that WITNESSES the occupancy the class claims -- plus
    completion-order collection and a hard peak_depth <= slots invariant.

    So the pin means "instrumented", and the actual schedule change under test is the
    POOL SIZE: 16 tasks behind 4 slots instead of exact4's pool == slots, which decays
    4->3->2->1 with nothing to backfill it.

    This duplicates fr13_floor_gate.validate_task_refill_ledger on purpose (that gate
    already ran in-pass, inside the driver).  A second independent read of the same
    bytes is the house pattern: the summary may not be trusted on its own word.
    """
    path = arm_dir / POOL_LEDGER_RELPATH
    events_path = arm_dir / POOL_LEDGER_EVENTS_RELPATH
    if not path.is_file() or path.is_symlink():
        raise B4GateError(
            f"no pool admission ledger at {path}; a pool16 arm must run with "
            "FR13_B4_TASK_REFILL=1 so its occupancy claim is witnessed"
        )
    if not events_path.is_file() or events_path.is_symlink():
        raise B4GateError(
            f"pool evidence is half-present: summary without {events_path}"
        )
    summary = exact_json(path, label=f"{arm_dir.name} pool ledger")
    if summary.get("schema") != TASK_REFILL_SUMMARY_SCHEMA:
        raise B4GateError(
            f"pool ledger schema is {summary.get('schema')!r}, "
            f"expected {TASK_REFILL_SUMMARY_SCHEMA!r}"
        )
    if summary.get("slots") != slots:
        raise B4GateError(f"pool ran {summary.get('slots')!r} slots, not {slots}")
    if summary.get("task_count") != task_count:
        raise B4GateError(
            f"pool held {summary.get('task_count')!r} tasks, not {task_count}"
        )
    if task_count <= slots:
        raise B4GateError(
            f"a pool must be LARGER than its slot count ({task_count} tasks, "
            f"{slots} slots); at parity it is the ordinary exact4 wave"
        )
    if summary.get("aborted") is not False:
        raise B4GateError("pool run aborted; its evidence is partial")
    if summary.get("completed") != task_count:
        raise B4GateError(
            f"pool completed {summary.get('completed')!r} of {task_count} tasks"
        )
    peak = summary.get("peak_depth")
    if not isinstance(peak, int) or isinstance(peak, bool) or peak > slots:
        raise B4GateError(
            f"peak in-flight depth {peak!r} exceeds the slot count {slots}; the "
            "in-flight invariant the floor bounds depend on was violated"
        )
    depth = _finite(summary.get("time_weighted_mean_depth"), "time_weighted_mean_depth")
    width = _finite(summary.get("full_width_fraction"), "full_width_fraction")
    # Occupancy floors, NOT throughput thresholds: they decide whether the arm was a
    # pool run at all.  Both recorded diagnostics cleared them (depth 3.399 / 3.234
    # against 3.2; full width 0.780 / 0.710 against 0.60), the second by 0.034 -- so a
    # legitimate pass CAN fail here, and it is excluded with its measured value named
    # rather than crashing the campaign.
    if depth < MIN_POOL_TIME_WEIGHTED_DEPTH:
        raise B4GateError(
            f"time-weighted mean depth {depth:.4f} is below the pool floor "
            f"{MIN_POOL_TIME_WEIGHTED_DEPTH} at {slots} slots; the pool did not hold "
            "the served width"
        )
    if width < MIN_POOL_FULL_WIDTH_FRACTION:
        raise B4GateError(
            f"full-width fraction {width:.4f} is below the pool floor "
            f"{MIN_POOL_FULL_WIDTH_FRACTION}"
        )
    return {
        "status": "pass",
        "slots": slots,
        "task_count": task_count,
        "admissions": summary.get("admissions"),
        "peak_depth": peak,
        "arm_wall_s": summary.get("arm_wall_s"),
        "time_weighted_mean_depth": depth,
        "full_width_fraction": width,
        "depth_floor": MIN_POOL_TIME_WEIGHTED_DEPTH,
        "full_width_floor": MIN_POOL_FULL_WIDTH_FRACTION,
        "depth_basis": (
            "worker-thread in-flight depth, an UPPER BOUND on engine decode "
            "co-residency: a freshly admitted task is still hydrating its repo and "
            "starting its container, so the engine sees nothing yet"
        ),
    }


# --------------------------------------------------------------------------- #
# run classes                                                                  #
# --------------------------------------------------------------------------- #
# A RUN CLASS is the whole contract of one kind of citable B4 campaign: what it is
# bound to, what admission topology it admits, which statistic it leads with, and --
# explicitly -- what it does NOT claim.  Two classes exist and they are deliberately
# not interchangeable.
#
#   exact4_formal_floor   the sealed formal floor gate.  pool == slots == 4.  Nested
#                         brackets.  PRIMARY = per_request_step_tps, because at a
#                         4-task pool the aggregate is a property of task-end skew.
#
#   pool16_refill_timing  the schedule-front class.  16 tasks behind 4 slots.
#                         Staggered brackets reduced on the envelope.  PRIMARY =
#                         aggregate, because co-residency is precisely what it
#                         measures -- with per_request_step_tps kept as a mandatory
#                         non-regression companion so 3c6d663d6 cannot repeat.
#
# The inversion of the primary statistic between the two is the only substantive
# difference in the mathematics, and it is intentional: demoting the aggregate at
# exact4 was never a claim that the aggregate is meaningless, only that at pool ==
# slots it measures the admission schedule instead of the stack.
RUN_CLASSES: dict[str, dict[str, Any]] = {
    "exact4_formal_floor": {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "title": "B4 FORMAL FLOOR GATE",
        "identity_label": "exact4",
        "task_count": 4,
        "subset_sha256": EXACT4_SUBSET_SHA256,
        "task_ids": EXACT4_TASK_IDS,
        "required_bracket_topology": "nested",
        "contract_pinned_stack": {"FR13_B4_TASK_REFILL": "0"},
        "requires_pool_ledger": False,
        "primary_statistic": "per_request_step_tps",
        "co_residency_dominated": ("measured_tps_fullstep_wall", "events_per_step"),
        "co_residency_note": (
            "driven by admission schedule (task-end skew at a 4-task pool), not by "
            "service speed; interval is correspondingly wide"
        ),
        "claims": (
            "the formal statistical floor of the shipped fixed32 B4 stack on the "
            "byte-pinned exact4 SWE-Verified subset at batch 4 / concurrency 4",
        ),
        "does_not_claim": (
            "a cap verdict: b4_cap_applicable is false at B4 context sizes",
        ),
    },
    "pool16_refill_timing": {
        "schema": POOL16_SCHEMA,
        "classification": POOL16_CLASSIFICATION,
        "title": "B4 POOL16 REFILL TIMING",
        "identity_label": "pool16",
        "task_count": 16,
        "subset_sha256": POOL16_SUBSET_SHA256,
        "task_ids": POOL16_TASK_IDS,
        "required_bracket_topology": "staggered",
        "contract_pinned_stack": {"FR13_B4_TASK_REFILL": "1"},
        "requires_pool_ledger": True,
        "primary_statistic": "measured_tps_fullstep_wall",
        "co_residency_dominated": (),
        "co_residency_note": "",
        "claims": (
            "aggregate TPS, events/step and per-request step TPS delivered by the "
            "shipped fixed32 B4 stack when fed a 16-task pool at slot width 4 under "
            "STAGGERED admission, with between-pass one-sided Student-t intervals",
        ),
        "does_not_claim": (
            "exact4 comparability: a different task set (16 vs its first 4) reduced "
            "on a different estimator (envelope vs widest-nested). Any exact4 "
            "contrast this artifact carries is descriptive and lists its confounds",
            "a cap verdict: b4_cap_applicable is false at B4 context sizes",
            "an agent-quality verdict: resolve rate is recorded, never gated. exact16 "
            "as QUALITY CONTROL at batched milestones (Mark, 2026-08-10) is a "
            "separate gate over the same subset and is untouched by this class",
            "that FR13_B4_TASK_REFILL=1 produced the number: the flag is the "
            "admission LEDGER, not a schedule lever (see read_pool_ledger). The "
            "schedule change under test is the pool size",
            "that the driver's in-pass fr13_floor_gate.py vouched for any arm: it "
            "has not reached a verdict for ANY fixed32 campaign in this generation, "
            "the citable exact4 ON gate included. Its per-pass verdict is recorded "
            "under in_pass_floor_gate so the gap is visible rather than assumed away",
            "that step_wall_ms and events_per_step share one basis: the wall mean is "
            "over wall-bracketed steps, the event rate over all census steps. "
            "retained_wall_fraction reports the gap per arm (pass_00: 0.911 tail / "
            "0.906 hydra, every unbracketed step a wall-chain reset, zero cap "
            "rejections). Measured selectivity on the occupancy axis is -0.06%/-0.14% "
            "for pool16 against -1.19%..-2.37% for the exact4 reference arms, so the "
            "contrast is not flattered by it -- but it is not zero and is published",
        ),
    },
}
DEFAULT_RUN_CLASS = "exact4_formal_floor"


def resolve_run_class(name: str) -> dict[str, Any]:
    try:
        return RUN_CLASSES[name]
    except KeyError as error:
        raise B4GateError(
            f"unknown run class {name!r}; known classes are "
            + ", ".join(sorted(RUN_CLASSES))
        ) from error
CANONICAL_ENV_RELPATH = "scripts/fr13_canonical_env.sh"
# export NAME="${NAME:-VALUE}"
_CANONICAL_DEFAULT_RE = re.compile(
    r'^\s*export\s+([A-Z0-9_]+)="\$\{\1:-([^}"]*)\}"', re.M
)


def resolve_canonical_defaults(repo: Path, commit: str = "") -> dict[str, str]:
    """Read the shipped default of every lever from the canonical env registry.

    Resolved AT THE CAMPAIGN'S OWN COMMIT when one is given, not from whatever
    the working tree happens to hold now. A campaign must be judged against the
    stack the branch shipped WHEN IT RAN: after the 2026-08-10 promotion flipped
    FR13_MAMBA_SPEC_BLOCKS_CDIV 0 -> 1, reducing the earlier narrowing-OFF
    baseline against today's registry would reject all eight of its arms for
    "not running the shipped stack" -- when running OFF was exactly correct for
    that campaign. Both halves of an OFF/ON promotion pair have to stay
    reducible, so provenance is read from the commit, not the checkout.
    """
    if commit:
        try:
            text = subprocess.run(
                ["git", "-C", str(repo), "show", f"{commit}:{CANONICAL_ENV_RELPATH}"],
                capture_output=True, text=True, check=True,
            ).stdout
        except (subprocess.CalledProcessError, OSError) as error:
            raise B4GateError(
                f"cannot read {CANONICAL_ENV_RELPATH} at commit {commit}: {error}"
            ) from error
    else:
        path = repo / CANONICAL_ENV_RELPATH
        if not path.is_file():
            raise B4GateError(f"missing canonical env registry {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
    return {m.group(1): m.group(2) for m in _CANONICAL_DEFAULT_RE.finditer(text)}


def expected_stack(
    canonical_defaults: dict[str, str],
    pinned: dict[str, str] | None = None,
) -> dict[str, str]:
    """Contract pins plus the shipped default of each tracked lever.

    `pinned` is the RUN CLASS's contract pin set; it defaults to the exact4 gate's
    so every existing caller keeps its behaviour byte for byte.  A class pin always
    wins over a shipped default: that is what "contract-pinned" means.
    """
    expected = dict(CONTRACT_PINNED_STACK if pinned is None else pinned)
    for lever in CANONICAL_DEFAULT_LEVERS:
        if lever in canonical_defaults and lever not in expected:
            expected[lever] = canonical_defaults[lever]
    return expected

_APC_QUERIES = "vllm:prefix_cache_queries_total"
_APC_HITS = "vllm:prefix_cache_hits_total"
_PROM_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)$")


class B4GateError(RuntimeError):
    """A pass artifact failed a fail-closed gate."""


# --------------------------------------------------------------------------- #
# offline finalization                                                         #
# --------------------------------------------------------------------------- #
# fr13_b4_campaign_driver.sh::reduce writes deploy_speed_${TAG}.json and does
# NOT pass --work-census, so its artifact records work_census_gate.status
# "absent".  That is an UNGATED B4 aggregate -- precisely the artifact the
# alignment study invalidated -- so this gate refuses it.  The fix is offline and
# lossless: every input the census cross-gate needs (the four per-task
# pre/post /metrics brackets and the arm's own work census) is written to the
# runroot during serving and is still there afterwards, so the reduction is
# simply re-run WITH the census.  Nothing is recomputed from the model; this
# calls the same fr13_measure.py cmd_deploy_speed the driver called, with the
# witness it omitted.
DEPLOY_SPEED_FILENAME = "deploy_speed_fullwall.json"
ARM_COMPLETE_MARKER = "arm_ended_at.txt"
WORK_CENSUS_RELPATH = "logs/fr13_fixed32_work_census.jsonl"
EXPECTED_TOK_PER_DRAFT = 31


def finalize_arm(
    arm_dir: Path, *, expected_tok_per_draft: float = EXPECTED_TOK_PER_DRAFT
) -> dict[str, Any]:
    """Re-reduce one COMPLETED arm with its work census. Idempotent.

    Refuses to touch an arm that is still serving: an arm writes
    arm_ended_at.txt when it finishes, so its absence means the brackets and the
    census are still being appended to and any reduction would be a torn read.
    """
    result: dict[str, Any] = {"arm": arm_dir.name, "action": None}
    target = arm_dir / DEPLOY_SPEED_FILENAME
    if target.is_file():
        result["action"] = "already_present"
        return result
    if not (arm_dir / ARM_COMPLETE_MARKER).is_file():
        result["action"] = "skipped_arm_still_serving"
        return result
    census = arm_dir / WORK_CENSUS_RELPATH
    if not census.is_file() or census.is_symlink():
        result["action"] = "skipped_no_work_census"
        return result
    out_root = arm_dir / "swe_out"
    if not out_root.is_dir():
        result["action"] = "skipped_no_swe_out"
        return result

    import argparse as _argparse
    import importlib.util as _importlib

    spec = _importlib.spec_from_file_location(
        "fr13_measure", Path(__file__).resolve().parent / "fr13_measure.py"
    )
    if spec is None or spec.loader is None:
        raise B4GateError("cannot load scripts/fr13_measure.py")
    measure = _importlib.module_from_spec(spec)
    spec.loader.exec_module(measure)

    # Write via a temp file then os.replace so an interrupted finalize can never
    # leave a half-written aggregate that a later run would treat as evidence.
    temporary = target.with_name(target.name + ".tmp")
    measure.cmd_deploy_speed(
        _argparse.Namespace(
            arm=arm_dir.name,
            out_root=str(out_root),
            expected_tok_per_draft=float(expected_tok_per_draft),
            batch_size=4,
            basis="decode_seconds",
            work_census=str(census),
            out=str(temporary),
        )
    )
    temporary.replace(target)
    result["action"] = "finalized"
    result["work_census"] = str(census)
    return result


def finalize_gate_root(
    gate_root: Path, *, expected_tok_per_draft: float = EXPECTED_TOK_PER_DRAFT
) -> list[dict[str, Any]]:
    """Finalize every completed arm under <gate_root>/pass_NN/."""
    actions: list[dict[str, Any]] = []
    for pass_dir in sorted(p for p in gate_root.glob("pass_*") if p.is_dir()):
        for mode in TOPOLOGIES:
            for arm_dir in sorted(pass_dir.glob(f"{mode}_*")):
                if arm_dir.is_dir() and not arm_dir.is_symlink():
                    actions.append(
                        finalize_arm(
                            arm_dir, expected_tok_per_draft=expected_tok_per_draft
                        )
                    )
    return actions


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise B4GateError(f"{label} is missing or not numeric")
    value = float(value)
    if not math.isfinite(value):
        raise B4GateError(f"{label} is not finite")
    return value


def _reject_constant(value: str) -> Any:
    raise B4GateError(f"JSON carries the non-finite literal {value!r}")


def exact_json(path: Path, *, label: str) -> dict[str, Any]:
    """Read JSON fail-closed: regular file, strict UTF-8, no NaN/Infinity."""
    if not path.is_file() or path.is_symlink():
        raise B4GateError(f"{label}: {path} is not a regular file")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise B4GateError(f"{label}: {path} is not strict UTF-8") from error
    payload = json.loads(text, parse_constant=_reject_constant)
    if not isinstance(payload, dict):
        raise B4GateError(f"{label}: {path} is not a JSON object")
    return payload


def _prom_counters(path: Path) -> dict[str, float]:
    """Sum every label set per metric family in one Prometheus text snapshot."""
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        match = _PROM_LINE.match(line.rstrip())
        if match is None:
            continue
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        name = match.group(1)
        out[name] = out.get(name, 0.0) + value
    return out


def read_apc(arm_dir: Path) -> dict[str, Any] | None:
    """APC hit rate over the campaign window from the arm's own bracket.

    Returns None when the snapshots are absent: APC is REPORTED, not gated, so a
    pass is not failed for lacking it.
    """
    before_path = arm_dir / "metrics_before_swe.txt"
    after_path = arm_dir / "metrics_after_swe.txt"
    if not before_path.is_file() or not after_path.is_file():
        return None
    before = _prom_counters(before_path)
    after = _prom_counters(after_path)
    if _APC_QUERIES not in before or _APC_QUERIES not in after:
        return None
    queries = after[_APC_QUERIES] - before[_APC_QUERIES]
    hits = after.get(_APC_HITS, 0.0) - before.get(_APC_HITS, 0.0)
    if queries <= 0:
        return None
    return {
        "queries": queries,
        "hits": hits,
        "hit_rate": hits / queries,
    }


def read_container_env(arm_dir: Path) -> dict[str, str]:
    path = arm_dir / "container_env.txt"
    if not path.is_file():
        return {}
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, sep, value = line.partition("=")
        if sep and key and " " not in key:
            env[key] = value
    return env


def reduce_pass_arm(
    arm_dir: Path,
    *,
    mode: str,
    pass_index: int,
    floor_order: str | None,
    expected: dict[str, str],
    run_class: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reduce ONE arm of ONE pass, or record why it is excluded.

    Never raises for an evidence-level defect: a defective pass is EXCLUDED with a
    reason so the verdict can report how many passes survived.  Raises only when
    the caller handed us something structurally impossible to interpret.
    """
    spec = RUN_CLASSES[DEFAULT_RUN_CLASS] if run_class is None else run_class
    want_tasks = spec["task_count"]
    want_ids = spec["task_ids"]
    want_topology = spec["required_bracket_topology"]
    record: dict[str, Any] = {
        "pass_index": pass_index,
        "arm": arm_dir.name,
        "arm_dir": str(arm_dir),
        "floor_order": floor_order,
        "included": False,
        "exclusion_reason": None,
    }
    try:
        speed_path = arm_dir / DEPLOY_SPEED_FILENAME
        speed = exact_json(speed_path, label=f"{arm_dir.name} deploy-speed")

        if speed.get("schema") != DEPLOY_SPEED_SCHEMA:
            raise B4GateError(
                f"deploy-speed schema is {speed.get('schema')!r}, "
                f"expected {DEPLOY_SPEED_SCHEMA!r}"
            )
        if speed.get("batch_size") != 4:
            raise B4GateError(f"arm ran at batch_size {speed.get('batch_size')!r}, not 4")
        if speed.get("n_tasks") != want_tasks:
            raise B4GateError(
                f"arm ran {speed.get('n_tasks')!r} tasks, not {want_tasks}"
            )

        task_ids = speed.get("task_instance_ids")
        if sorted(task_ids or []) != sorted(want_ids):
            raise B4GateError(
                f"arm did not run the canonical {spec['identity_label']} task "
                "identities"
            )

        # --- bracket topology + work-census cross-gate -----------------------
        bracket = speed.get("bracket_reduction")
        if not isinstance(bracket, dict):
            raise B4GateError("deploy-speed carries no bracket-topology provenance")
        topology = bracket.get("topology")
        if topology != want_topology:
            raise B4GateError(
                f"bracket topology is {topology!r}, expected {want_topology!r}. "
                "An exact4 arm admits all four tasks on one engine state and its "
                "brackets NEST; a pool arm admits each task as an earlier one "
                "finishes and its brackets STAGGER, so the arm delta is the "
                "envelope. Summing either one is the defect that inflated B4 "
                "aggregates 1.7-2.6x (nested) and 1.87-4.00x (staggered)."
            )
        census_gate = bracket.get("work_census_gate")
        if not isinstance(census_gate, dict):
            raise B4GateError("bracket reduction carries no work-census gate record")
        if census_gate.get("status") != "pass":
            raise B4GateError(
                "bracket reduction was not work-census gated "
                f"(status={census_gate.get('status')!r}); an ungated B4 aggregate is "
                "exactly the artifact the alignment study invalidated"
            )

        # --- wall-bracket retention -----------------------------------------
        # step_wall_ms is measured over steps the WALL CHAIN bracketed, while
        # events_per_step comes from the census over EVERY forward step. The two
        # bases differ whenever the chain resets, and the chain resets on every
        # request-set change -- which is exactly what continuous admission does.
        # Measured on pass_00: unbracketed steps == wall_chain_resets exactly
        # (839 tail / 649 hydra), zero cap rejections. So this is reported on
        # every arm rather than left for a reader to discover.
        census_steps = census_gate.get("census_steps")
        wall_steps = speed.get("wall_steps_measured")
        wall_retention = None
        if isinstance(census_steps, (int, float)) and census_steps and isinstance(
            wall_steps, (int, float)
        ):
            wall_retention = {
                "wall_steps_measured": float(wall_steps),
                "census_forward_steps": float(census_steps),
                "retained_wall_fraction": float(wall_steps) / float(census_steps),
                "basis_note": (
                    "step_wall_ms is a mean over wall-bracketed steps; "
                    "events_per_step is over all census steps. The unbracketed "
                    "steps are the first step after each wall-chain reset, and a "
                    "chain resets when the served request set changes."
                ),
            }

        # --- rate identities -------------------------------------------------
        breakdown = phase_breakdown(speed, arm_dir.name)

        # --- stack state: no default may have been flipped -------------------
        env = read_container_env(arm_dir)
        tracked = sorted(set(expected) | set(CANONICAL_DEFAULT_LEVERS))
        stack_state = {key: env.get(key, "<unrecorded>") for key in tracked}
        flipped = sorted(
            key for key, want in expected.items() if env.get(key, want) != want
        )

        apc = read_apc(arm_dir)

        # --- pool admission ledger (pool classes only) -----------------------
        pool_ledger = (
            read_pool_ledger(arm_dir, slots=4, task_count=want_tasks)
            if spec["requires_pool_ledger"]
            else None
        )

        record.update(
            {
                "included": True,
                "deploy_speed_path": str(speed_path),
                "pool_ledger": pool_ledger,
                "wall_retention": wall_retention,
                "bracket": {
                    "topology": topology,
                    "closing_task": bracket.get("closing_task"),
                    "distinct_bracket_origins": bracket.get("distinct_bracket_origins"),
                    "basis": bracket.get("basis"),
                },
                "work_census_gate": {
                    "status": census_gate.get("status"),
                    "census_steps": census_gate.get("census_steps"),
                    "census_events": census_gate.get("census_events"),
                    "census_events_per_step": census_gate.get("census_events_per_step"),
                },
                "stack_state": stack_state,
                "defaults_flipped": flipped,
                "phase_breakdown": breakdown,
                "measured_tps_fullstep_wall": breakdown["measured_tps_fullstep_wall"],
                "per_request_step_tps": breakdown["per_request_step_tps"],
                "events_per_step": breakdown["events_per_step"],
                "step_wall_ms": _finite(speed.get("step_wall_ms"), "step_wall_ms"),
                "floor_ratio": _finite(speed.get("floor_ratio"), "floor_ratio"),
                "floor_ms": _finite(speed.get("floor_ms"), "floor_ms"),
                "prefill_frac": _finite(speed.get("prefill_frac"), "prefill_frac"),
                "aggregate_window_wall_s": speed.get("aggregate_window_wall_s"),
                "apc": apc,
                "logical_topology": TOPOLOGIES[mode]["logical_topology"],
            }
        )
        if flipped:
            record["included"] = False
            record["exclusion_reason"] = (
                "arm did not run the shipped stack: "
                + ", ".join(
                    f"{k}={env.get(k, '<unrecorded>')!r} but the branch ships "
                    f"{expected[k]!r}"
                    for k in flipped
                )
                + " -- a citable gate measures the stack the branch ships"
            )
    except (B4GateError, TimingMathError, json.JSONDecodeError, OSError) as error:
        record["included"] = False
        record["exclusion_reason"] = f"{type(error).__name__}: {error}"
    return record


def cluster_interval(values: list[float]) -> dict[str, Any]:
    """Two-sided-critical one-sided bounds over pass-level values.

    Same shape and same arithmetic as fr13_floor_gate.cluster_summary, extended
    with the LOWER bound because throughput is cited from below.  The critical
    value is the repo's pinned one -- only N in {4, 16} is admissible, which is
    what keeps this from silently degrading to a 2-pass "interval".
    """
    count = len(values)
    df = count - 1
    critical = T95_ONE_SIDED.get(df)
    if critical is None:
        raise B4GateError(
            f"no pinned one-sided t critical for df={df}; the gate admits exactly "
            "4 or 16 included passes"
        )
    point = statistics.fmean(values)
    sample_sd = statistics.stdev(values)
    standard_error = sample_sd / math.sqrt(count)
    return {
        "cluster_count": count,
        "df": df,
        "point_estimate": point,
        "sample_sd_across_passes": sample_sd,
        "standard_error": standard_error,
        "t_0_95_one_sided": critical,
        "l95": point - critical * standard_error,
        "u95": point + critical * standard_error,
        "cv": (sample_sd / point) if point else None,
        "values": list(values),
    }


# The PRIMARY statistic is the batch-invariant per-request rate, not the
# aggregate.  Measured across repeats of the same config on this rig:
#
#     measured_tps_fullstep_wall   CV ~29%   (co-residency; trajectory-driven)
#     per_request_step_tps         CV  ~9%   (service speed; 3.2x tighter)
#
# aggregate = events_per_step * per_request_step_tps, and events_per_step is set
# by task-end skew: at a 4-task pool the batch decays 4 -> 3 -> 2 -> 1 with
# nothing behind it and is full width for only ~36% of the arm.  The aggregate is
# therefore a property of the admission schedule, not of the kernel stack.  The
# repo already made this call once, in 3c6d663d6 "bind the B4 promotion criterion
# to the per-request rate", after a candidate posted +17.2% aggregate while every
# individual request got 3% SLOWER.  The aggregate is still reported in full --
# it is the number the campaign cites -- but it carries its own much wider
# interval and is labelled co-residency-dominated.
PRIMARY_STATISTIC = "per_request_step_tps"
CO_RESIDENCY_DOMINATED = ("measured_tps_fullstep_wall", "events_per_step")

PRIMARY_STATISTIC_REASON = {
    "exact4_formal_floor": (
        "Across repeats of an identical config on this rig, aggregate "
        "measured_tps_fullstep_wall carries CV ~29% while per_request_step_tps "
        "carries CV ~9%. The aggregate is events_per_step * per_request_step_tps "
        "and events_per_step is set by task-end skew, which at a 4-task pool is "
        "structural: the batch decays 4->3->2->1 with nothing behind it and is "
        "full width for only ~36% of the arm. The aggregate is reported in full "
        "because it is the number the campaign cites, but the citable floor "
        "claim rests on the batch-invariant per-request rate (repo precedent: "
        "3c6d663d6)."
    ),
    "pool16_refill_timing": (
        "INVERTED relative to the exact4 class, deliberately. The aggregate was "
        "demoted at exact4 because at pool == slots it measures task-end skew "
        "rather than the stack; co-residency is exactly what a 16-task pool at 4 "
        "slots exists to change, so here the aggregate IS the claim and "
        "events_per_step is its reported mechanism. The 3c6d663d6 lesson is kept "
        "as a hard companion rather than discarded: per_request_step_tps carries "
        "its own full interval and, whenever an exact4 reference is supplied, an "
        "explicit per_request_non_regression verdict. An aggregate gain bought "
        "with a per-request regression is NOT a win. This class asserts nothing "
        "about its own variance a priori; the measured between-pass CV of every "
        "statistic is reported."
    ),
}

_INTERVAL_FIELDS = (
    "per_request_step_tps",
    "measured_tps_fullstep_wall",
    "events_per_step",
    "step_wall_ms",
    "floor_ratio",
    "prefill_frac",
)


def apply_pass_atomicity(
    topology_passes: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """A PASS is the atomic unit of evidence: all its arms, or none of them.

    Both topologies are deliberately run back to back inside one pass so they
    share host conditions -- same boot, same page cache age, same thermal state.
    That is what makes the Tail23/Hydra27 contrast PAIRED. Admitting a surviving
    arm from an otherwise-broken pass would silently unpair it, comparing one
    topology's pass {0,1,2,3} against the other's {0,1,2,4}, and would also
    break the first-arm/second-arm position balance the alternating TH/HT order
    exists to guarantee.

    So when any arm of a pass is not included, every arm of that pass is voided.
    The rule keys ONLY on validity, never on a measured value, so it cannot be
    used to select a favourable number.

    This is what design.md 1.4 always specified ("every pass must clear all of
    the following or it is excluded"); the first implementation drifted to
    arm-level inclusion.
    """
    by_index: dict[int, list[dict[str, Any]]] = {}
    for mode in TOPOLOGIES:
        for record in topology_passes.get(mode, []):
            by_index.setdefault(record["pass_index"], []).append(record)

    voided: list[dict[str, Any]] = []
    for index in sorted(by_index):
        arms = by_index[index]
        modes_present = {
            m for m in TOPOLOGIES for r in topology_passes.get(m, [])
            if r["pass_index"] == index
        }
        broken = [r for r in arms if not r.get("included")]
        missing = sorted(set(TOPOLOGIES) - modes_present)
        if not broken and not missing:
            continue
        reasons = [f"{r['arm']}: {r['exclusion_reason']}" for r in broken]
        reasons += [f"{m}: no arm directory" for m in missing]
        summary = f"pass {index} is VOID -- " + "; ".join(reasons)
        voided.append({"pass_index": index, "reason": summary})
        for record in arms:
            if record.get("included"):
                record["included"] = False
                record["exclusion_reason"] = (
                    f"{summary}. A pass is atomic: its arms are paired evidence, "
                    "so a surviving arm cannot be admitted alone."
                )
                record["voided_by_pass_atomicity"] = True
    return voided


def reduce_topology(
    mode: str,
    passes: list[dict[str, Any]],
    min_passes: int,
    run_class: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = RUN_CLASSES[DEFAULT_RUN_CLASS] if run_class is None else run_class
    primary = spec["primary_statistic"]
    dominated = set(spec["co_residency_dominated"])
    included = [p for p in passes if p.get("included")]
    out: dict[str, Any] = {
        **TOPOLOGIES[mode],
        "passes": passes,
        "pass_count": len(passes),
        "included_pass_count": len(included),
        "excluded_pass_count": len(passes) - len(included),
    }
    if len(included) < min_passes or (len(included) - 1) not in T95_ONE_SIDED:
        out["analysis_valid"] = False
        out["reason"] = (
            f"{len(included)} included passes; the gate requires exactly 4 or 16 "
            "so the pinned one-sided critical applies"
        )
        return out

    out["analysis_valid"] = True
    out["primary_statistic"] = primary
    for field in _INTERVAL_FIELDS:
        stat = cluster_interval([float(p[field]) for p in included])
        stat["role"] = "primary" if field == primary else "reported"
        if field in dominated:
            stat["co_residency_dominated"] = True
            stat["note"] = spec["co_residency_note"]
        out[field] = stat
    apc_values = [p["apc"]["hit_rate"] for p in included if p.get("apc")]
    out["apc_hit_rate"] = (
        cluster_interval(apc_values) if len(apc_values) == len(included) else None
    )
    # Occupancy is the pool classes' mechanism, so it gets the same between-pass
    # interval as every headline rate.  Absent (None) for non-pool classes.
    for key in ("time_weighted_mean_depth", "full_width_fraction"):
        values = [
            float(p["pool_ledger"][key]) for p in included if p.get("pool_ledger")
        ]
        out[key] = (
            cluster_interval(values) if len(values) == len(included) and values else None
        )
    retention = [
        float(p["wall_retention"]["retained_wall_fraction"])
        for p in included
        if p.get("wall_retention")
    ]
    out["retained_wall_fraction"] = (
        cluster_interval(retention)
        if len(retention) == len(included) and retention
        else None
    )
    return out


EXACT4_CONTRAST_CONFOUNDS = (
    "TASK SET: 16 tasks against the first 4 of the same canonical ordering. "
    "Trajectory length differs per task and events_per_step is trajectory-driven.",
    "ESTIMATOR: the pool arm's delta is the STAGGERED envelope max(post)-min(pre); "
    "the exact4 arm's is the widest NESTED bracket. Both are cross-gated against "
    "the arm's own fr13_fixed32_work_census.jsonl and fr13_measure refuses on "
    "disagreement, so neither is an unwitnessed sum -- but they are not the same "
    "estimator.",
    "BETWEEN CAMPAIGNS: different day and different source commit. The two "
    "measured_stack_state records are required to be EQUAL or this contrast is "
    "refused, which bounds the drift to commits that do not touch the served stack.",
    "ADMISSION: pool arms open on many bracket origins (13 on the recorded "
    "diagnostic), exact4 arms on exactly one.",
)


def build_exact4_contrast(
    topologies: dict[str, Any],
    reference: dict[str, Any] | None,
    measured_stack: dict[str, str] | None,
    run_class_name: str,
) -> dict[str, Any] | None:
    """Descriptive pool16-vs-exact4 contrast, or None / a refusal record.

    NEVER a promotion verdict, and never folded into `citable`.  It exists because
    the question this rung was opened to answer -- "does a larger pool convert
    narrowing's KV headroom into aggregate TPS?" -- is a comparison, and the
    comparator is the already-sealed exact4 gate rather than a fresh arm.
    """
    if reference is None or run_class_name == "exact4_formal_floor":
        return None
    record: dict[str, Any] = {
        "reference_schema": reference.get("schema"),
        "reference_classification": reference.get("classification"),
        "reference_gate_root": reference.get("gate_root"),
        "reference_source_commit": reference.get("source_commit"),
        "reference_citable": reference.get("citable"),
        "role": "descriptive",
        "confounds": list(EXACT4_CONTRAST_CONFOUNDS),
    }
    if not reference.get("citable"):
        record["status"] = "refused"
        record["reason"] = "the exact4 reference is not itself citable"
        return record
    reference_stack = reference.get("measured_stack_state")
    # Compare only the SHIPPED-DEFAULT levers.  The classes' own contract pins are
    # required to differ -- pool16 pins FR13_B4_TASK_REFILL=1 for its ledger and
    # exact4 pins it 0 -- so demanding whole-dict equality would make this contrast
    # unsatisfiable by construction, which is the exact fossil this campaign keeps
    # digging up.  What must match is every lever whose value is a shipped default,
    # because those are the ones that change what the engine does.
    record["deliberately_different_pins"] = sorted(
        set(RUN_CLASSES[run_class_name]["contract_pinned_stack"])
        | set(RUN_CLASSES["exact4_formal_floor"]["contract_pinned_stack"])
    )
    if measured_stack is None or not isinstance(reference_stack, dict):
        record["status"] = "refused"
        record["reason"] = (
            "one of the campaigns has no single measured stack state to compare"
        )
        record["reference_measured_stack_state"] = reference_stack
        return record
    compared = {
        lever: (measured_stack.get(lever), reference_stack.get(lever))
        for lever in CANONICAL_DEFAULT_LEVERS
    }
    differing = sorted(k for k, (a, b) in compared.items() if a != b)
    if differing:
        record["status"] = "refused"
        record["reason"] = (
            "the two campaigns did not run the same shipped-default levers: "
            + ", ".join(
                f"{k}={compared[k][0]!r} vs {compared[k][1]!r}" for k in differing
            )
            + "; a cross-campaign contrast is only meaningful at equal lever state"
        )
        record["reference_measured_stack_state"] = reference_stack
        return record

    record["status"] = "evaluated"
    record["reference_measured_stack_state"] = reference_stack
    record["shipped_default_levers_compared"] = {
        lever: value for lever, (value, _) in compared.items()
    }
    per_topology: dict[str, Any] = {}
    regressions: list[str] = []
    for mode, topo in topologies.items():
        ref_topo = (reference.get("topologies") or {}).get(mode)
        if not topo.get("analysis_valid") or not (ref_topo or {}).get("analysis_valid"):
            per_topology[mode] = {"status": "not_evaluated"}
            continue
        fields: dict[str, Any] = {}
        for field in ("measured_tps_fullstep_wall", "events_per_step",
                      "per_request_step_tps", "step_wall_ms", "prefill_frac"):
            pool, ref = topo[field], ref_topo[field]
            fields[field] = {
                "pool16": pool["point_estimate"],
                "exact4": ref["point_estimate"],
                "relative_change": (
                    (pool["point_estimate"] - ref["point_estimate"])
                    / ref["point_estimate"]
                    if ref["point_estimate"]
                    else None
                ),
                # Non-overlap of two between-pass intervals is the weakest honest
                # separability statement available across campaigns; it is NOT a
                # paired test and is labelled as such.
                "intervals_overlap": not (
                    pool["l95"] > ref["u95"] or ref["l95"] > pool["u95"]
                ),
            }
        # 3c6d663d6: aggregate up + per-request down is not a win.
        per_request = fields["per_request_step_tps"]
        non_regression = per_request["intervals_overlap"] or (
            per_request["relative_change"] is not None
            and per_request["relative_change"] >= 0
        )
        if not non_regression:
            regressions.append(mode)
        fields["per_request_non_regression"] = non_regression
        per_topology[mode] = {"status": "evaluated", **fields}
    record["per_topology"] = per_topology
    record["per_request_non_regression_all_topologies"] = not regressions
    record["per_request_regressed_on"] = regressions
    record["separability_note"] = (
        "intervals_overlap compares two independent between-pass intervals from "
        "two different campaigns. Non-overlap is evidence of separation; overlap "
        "is NOT evidence of equality. No paired test is available here because no "
        "exact4 arm ran inside this campaign."
    )
    return record


def build_verdict(
    *,
    repo: Path,
    gate_root: Path,
    source_commit: str,
    topology_passes: dict[str, list[dict[str, Any]]],
    min_passes: int,
    expected: dict[str, str] | None = None,
    run_class: dict[str, Any] | None = None,
    run_class_name: str = DEFAULT_RUN_CLASS,
    exact4_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = RUN_CLASSES[DEFAULT_RUN_CLASS] if run_class is None else run_class
    void_passes = apply_pass_atomicity(topology_passes)
    topologies = {
        mode: reduce_topology(mode, topology_passes.get(mode, []), min_passes, spec)
        for mode in TOPOLOGIES
    }
    all_passes = [p for mode in TOPOLOGIES for p in topology_passes.get(mode, [])]
    included = [p for p in all_passes if p.get("included")]

    # Every included arm must have run the SAME lever state -- a campaign that
    # changed stack mid-flight measures nothing.
    observed_states = {
        json.dumps(p["stack_state"], sort_keys=True) for p in included
    }
    measured_stack = (
        json.loads(next(iter(observed_states))) if len(observed_states) == 1 else None
    )

    gates = {
        "subset_bytes_canonical": True,
        "one_stack_state_across_all_passes": len(observed_states) == 1,
        "every_included_pass_is_complete": all(
            p["pass_index"] not in {v["pass_index"] for v in void_passes}
            for p in included
        ),
        "all_passes_required_bracket_topology": bool(included)
        and all(
            p["bracket"]["topology"] == spec["required_bracket_topology"]
            for p in included
        ),
        "all_pool_arms_held_the_served_width": (
            all(
                isinstance(p.get("pool_ledger"), dict)
                and p["pool_ledger"].get("status") == "pass"
                for p in included
            )
            if spec["requires_pool_ledger"]
            else True
        ),
        "all_passes_work_census_gated": bool(included)
        and all(p["work_census_gate"]["status"] == "pass" for p in included),
        "all_passes_timing_math_reconciles": bool(included),
        "no_defaults_flipped": all(not p.get("defaults_flipped") for p in included),
        "sufficient_included_passes": all(
            t.get("analysis_valid") for t in topologies.values()
        ),
        "both_topologies_present": all(
            topology_passes.get(mode) for mode in TOPOLOGIES
        ),
    }
    analysis_valid = all(gates.values())

    if not analysis_valid:
        if not gates["sufficient_included_passes"] or not gates["both_topologies_present"]:
            verdict = "NOT_EVALUATED_INSUFFICIENT_PASSES"
        else:
            verdict = "NOT_EVALUATED_INVALID_INPUT"
    else:
        verdict = "PASS"

    comparison: dict[str, Any] = {}
    tail, hydra = topologies["tail6_fixed32"], topologies["hydra27_fixed32"]
    if tail.get("analysis_valid") and hydra.get("analysis_valid"):
        t_agg = tail["measured_tps_fullstep_wall"]
        h_agg = hydra["measured_tps_fullstep_wall"]
        comparison = {
            "aggregate_tps_hydra27_minus_tail23": (
                h_agg["point_estimate"] - t_agg["point_estimate"]
            ),
            "intervals_overlap": not (
                h_agg["l95"] > t_agg["u95"] or t_agg["l95"] > h_agg["u95"]
            ),
            "note": "descriptive only; topology choice is not a promotion decision",
        }

    return {
        "schema": spec["schema"],
        "analysis_valid": analysis_valid,
        "gate_verdict": verdict,
        "citable": analysis_valid,
        # Only the exact4 class is a FORMAL FLOOR gate. A pool16 timing run is
        # citable within its own class and is never floor-acceptance evidence.
        "formal_floor_acceptance_eligible": (
            analysis_valid and run_class_name == "exact4_formal_floor"
        ),
        "run_class": run_class_name,
        "claims": list(spec["claims"]),
        "does_not_claim": list(spec["does_not_claim"]),
        "exact4_contrast": build_exact4_contrast(
            topologies, exact4_reference, measured_stack, run_class_name
        ),
        "classification": spec["classification"],
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "repo": str(repo),
        "gate_root": str(gate_root),
        "source_commit": source_commit,
        "contract": {
            "batch_size": 4,
            "concurrency": 4,
            "task_count": spec["task_count"],
            "subset_sha256": spec["subset_sha256"],
            "task_ids": list(spec["task_ids"]),
            "physical_rows": 32,
            "drafts": 31,
            "required_bracket_topology": spec["required_bracket_topology"],
            "requires_pool_ledger": spec["requires_pool_ledger"],
            "contract_pinned_stack": dict(sorted(spec["contract_pinned_stack"].items())),
            "shipped_defaults_at_campaign_commit": dict(sorted((expected or {}).items())),
            "min_included_passes": min_passes,
        },
        # WHICH STACK THIS GATE MEASURED. Recorded, never pinned: the gate follows
        # the branch's shipped default so a promoted lever is measured, not rejected.
        "measured_stack_state": measured_stack,
        "void_passes": void_passes,
        "b4_cap_applicable": False,
        "b4_cap_reason": (
            "B4 mandatory bytes at ~35-38k tok/request are 42-49 GB => a 155-179 ms "
            "weight-read floor, above the 137.607 ms one-sided cap. The B1 cap is "
            "physics-unreachable at B4 context sizes, so B4 is reported as aggregate "
            "TPS with a per-request decomposition, not as a cap verdict."
        ),
        "primary_statistic": spec["primary_statistic"],
        "primary_statistic_reason": PRIMARY_STATISTIC_REASON[run_class_name],
        "uncertainty_model": (
            "Between-pass one-sided Student-t over pass-level values using the "
            "pinned T95_ONE_SIDED critical (df=3 at 4 passes). Aggregate TPS is "
            "NOT given a within-run bootstrap interval: it is "
            "events_per_step * per_request_step_tps, and events_per_step is a "
            "property of one realization of four agent trajectories, so a "
            "within-run resample cannot see the dominant variance term."
        ),
        "topologies": topologies,
        "comparison": comparison,
        "gates": gates,
    }


def discover_passes(
    gate_root: Path,
    expected: dict[str, str],
    run_class: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Find <gate_root>/pass_NN/<mode>_<tag>/ arm directories."""
    found: dict[str, list[dict[str, Any]]] = {mode: [] for mode in TOPOLOGIES}
    for pass_dir in sorted(p for p in gate_root.glob("pass_*") if p.is_dir()):
        try:
            pass_index = int(pass_dir.name.split("_", 1)[1])
        except (IndexError, ValueError) as error:
            raise B4GateError(f"{pass_dir} is not a pass_NN directory") from error
        order_path = pass_dir / "floor_order.txt"
        floor_order = (
            order_path.read_text(encoding="utf-8").strip()
            if order_path.is_file()
            else None
        )
        # The driver's IN-PASS fr13_floor_gate.py verdict. RECORDED, NEVER GATED --
        # and recorded precisely because it does not pass. It has not evaluated for
        # any fixed32 campaign in this generation, including the citable exact4 ON
        # gate (output/fr13_b4_formal_floor_gate_20260811T041931Z: PASS/citable=true
        # with all five passes NOT_EVALUATED_INVALID_INPUT in-pass and rc=2). Both
        # campaign verdicts therefore rest on the offline reducers reading arm
        # evidence. Surfacing it here stops a reader assuming the in-pass gate
        # vouched for an arm when it never ran to a verdict.
        in_pass = pass_dir / "fixed32_floor_gate.json"
        in_pass_gate: dict[str, Any] | None = None
        if in_pass.is_file() and not in_pass.is_symlink():
            try:
                payload = exact_json(in_pass, label="in-pass floor gate")
                in_pass_gate = {
                    "gate_verdict": payload.get("gate_verdict"),
                    "analysis_valid": payload.get("analysis_valid"),
                    "error": payload.get("error"),
                    "role": "reported_never_gated",
                }
            except B4GateError:
                in_pass_gate = {"gate_verdict": "UNREADABLE", "role": "reported_never_gated"}
        for mode in TOPOLOGIES:
            for arm_dir in sorted(pass_dir.glob(f"{mode}_*")):
                if arm_dir.is_dir() and not arm_dir.is_symlink():
                    found[mode].append(
                        reduce_pass_arm(
                            arm_dir,
                            mode=mode,
                            pass_index=pass_index,
                            floor_order=floor_order,
                            expected=expected,
                            run_class=run_class,
                        )
                    )
                    found[mode][-1]["in_pass_floor_gate"] = in_pass_gate
    return found


def render(payload: dict[str, Any]) -> str:
    lines = [
        f"{RUN_CLASSES[payload.get('run_class', DEFAULT_RUN_CLASS)]['title']} -- "
        f"{payload['gate_verdict']} (citable={payload['citable']})",
        f"primary statistic: {payload['primary_statistic']}",
        "",
    ]
    for mode, topo in payload["topologies"].items():
        lines.append(
            f"{topo['logical_topology']:8s} ({mode}, {topo['active_drafts']} drafts, "
            f"mask {topo['valid_mask']}) "
            f"included {topo['included_pass_count']}/{topo['pass_count']} passes"
        )
        if not topo.get("analysis_valid"):
            lines.append(f"  NOT EVALUATED: {topo.get('reason')}")
            for p in topo["passes"]:
                if not p.get("included"):
                    lines.append(f"    excluded pass {p['pass_index']}: {p['exclusion_reason']}")
            lines.append("")
            continue
        for field in ("measured_tps_fullstep_wall", "per_request_step_tps",
                      "events_per_step", "step_wall_ms", "prefill_frac"):
            stat = topo[field]
            lines.append(
                f"  {field:28s} {stat['point_estimate']:9.4f}  "
                f"L95 {stat['l95']:9.4f}  U95 {stat['u95']:9.4f}  "
                f"CV {(stat['cv'] or 0) * 100:5.2f}%"
            )
        if topo.get("apc_hit_rate"):
            stat = topo["apc_hit_rate"]
            lines.append(
                f"  {'apc_hit_rate':28s} {stat['point_estimate'] * 100:9.2f}%  "
                f"L95 {stat['l95'] * 100:8.2f}%"
            )
        for key in ("time_weighted_mean_depth", "full_width_fraction"):
            stat = topo.get(key)
            if stat:
                lines.append(
                    f"  {key:28s} {stat['point_estimate']:9.4f}  "
                    f"L95 {stat['l95']:9.4f}"
                )
        lines.append("")
    contrast = payload.get("exact4_contrast")
    if contrast:
        lines.append(f"exact4 contrast ({contrast['role']}): {contrast['status']}")
        if contrast["status"] == "refused":
            lines.append(f"  REFUSED: {contrast['reason']}")
        else:
            for mode, block in contrast.get("per_topology", {}).items():
                if block.get("status") != "evaluated":
                    lines.append(f"  {mode}: not evaluated")
                    continue
                lines.append(f"  {mode}:")
                for field in ("measured_tps_fullstep_wall", "events_per_step",
                              "per_request_step_tps"):
                    cell = block[field]
                    change = cell["relative_change"]
                    lines.append(
                        f"    {field:26s} pool16 {cell['pool16']:8.4f}  "
                        f"exact4 {cell['exact4']:8.4f}  "
                        f"{'n/a' if change is None else f'{change * 100:+7.2f}%'}  "
                        f"{'overlap' if cell['intervals_overlap'] else 'SEPARATED'}"
                    )
                lines.append(
                    f"    per_request_non_regression: "
                    f"{block['per_request_non_regression']}"
                )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--gate-root", type=Path, required=True,
                        help="campaign root holding pass_NN/ directories")
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--min-passes", type=int, default=DEFAULT_MIN_PASSES)
    parser.add_argument(
        "--run-class",
        default=DEFAULT_RUN_CLASS,
        choices=sorted(RUN_CLASSES),
        help="which citable class this campaign is. 'exact4_formal_floor' is the "
             "formal floor gate (pool == slots == 4, nested brackets, per-request "
             "primary). 'pool16_refill_timing' is the schedule-front class (16 "
             "tasks behind 4 slots, staggered envelope, aggregate primary, pool "
             "admission ledger required).",
    )
    parser.add_argument(
        "--exact4-reference",
        type=Path,
        default=None,
        help="path to a sealed fr13_b4_formal_floor_gate.json. Pool classes emit a "
             "DESCRIPTIVE contrast against it with the confounds enumerated. Never "
             "affects citability; refused unless both campaigns measured the same "
             "stack state.",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="re-reduce completed arms WITH their work census before reducing. "
             "The campaign driver writes an ungated deploy_speed_${TAG}.json; this "
             "re-runs the same fr13_measure.py reduction with the witness it "
             "omitted. Offline, idempotent, and skips arms still serving.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    gate_root = args.gate_root.resolve()
    if not gate_root.is_dir():
        raise B4GateError(f"gate root {gate_root} is not a directory")

    spec = resolve_run_class(args.run_class)
    canonical_defaults = resolve_canonical_defaults(
        args.repo.resolve(), args.source_commit
    )
    expected = expected_stack(canonical_defaults, spec["contract_pinned_stack"])

    reference = None
    if args.exact4_reference is not None:
        reference = exact_json(
            args.exact4_reference.resolve(), label="exact4 reference"
        )

    # finalize needs no class parameter: cmd_deploy_speed discovers the task dirs and
    # classifies the bracket topology from the recorded counters, and it already
    # refuses a staggered reduction without the work census that finalize_arm passes.
    finalize_actions = finalize_gate_root(gate_root) if args.finalize else []
    for action in finalize_actions:
        print(f"[finalize] {action['arm']}: {action['action']}")

    payload = build_verdict(
        repo=args.repo.resolve(),
        gate_root=gate_root,
        source_commit=args.source_commit,
        topology_passes=discover_passes(gate_root, expected, spec),
        min_passes=args.min_passes,
        expected=expected,
        run_class=spec,
        run_class_name=args.run_class,
        exact4_reference=reference,
    )
    payload["finalize_actions"] = finalize_actions
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    out = args.out or (gate_root / "fr13_b4_formal_floor_gate.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(render(payload))
    print(f"-> {out}")
    return 0 if payload["gate_verdict"] == "PASS" else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except B4GateError as error:
        print(f"class-9 FAIL-LOUD [b4-floor-gate]: {error}", file=sys.stderr)
        raise SystemExit(2) from error

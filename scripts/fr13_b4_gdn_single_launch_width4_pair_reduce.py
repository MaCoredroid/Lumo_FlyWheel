#!/usr/bin/env python3
"""Width-4 timing pair for the folded GDN scan: verdict over the sealed window.

WHAT THIS DECIDES, AND WHAT IT DOES NOT
---------------------------------------
This is the phase-3 SCREEN for the GDN single-launch lever. It can HALT the
lever (stop rule below) and it can motivate a sealing campaign. It cannot seal
anything: sealing is a pre-registered multi-pass campaign with balanced arm
order and a one-sided lower bound, and that is phase 4. The width-4 window class
is an INSTRUMENT, not a citable seal, and this reducer inherits that status
unchanged from scripts/fr13_b4_width4_window_reduce.py.

THE PAIR
--------
Both arms are the SAME commit, the SAME source closure and the SAME served
geometry: fixed32 at MAX_NUM_SEQS=4 / SWE_CONCURRENCY=4, pool16 behind 4 slots
with refill, K64/root1, FULL_AND_PIECEWISE, BV=8. The single variable is whether
the GDN decode call is served by the folded single-launch kernel or by the
deployed two-launch reference:

  control    FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION=0  (NAMED 0 -- the
             deliberate opt-out the launcher obeys verbatim; this is not an
             unset variable and the reducer checks that it was named)
  candidate  FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION=1 + a HEAD-bound
             credential presented

WHY THE SCREEN RUNS THROUGH THE PRODUCTION ARM AND NOT A DIAGNOSTIC ONE
-----------------------------------------------------------------------
Because there is no diagnostic one. The credential-free bool
FR13_FIXED32_GDN_SINGLE_LAUNCH is structurally unreachable: its sidecar
/logs/fr13_fixed32_gdn_single_launch_tree.arm is only ever REMOVED by the
launcher and never written, and the variable is never exported into the
container. The live byte gate does not serve the kernel either -- it routes
through FR13_FIXED32_GDN_PATH_BV_CANDIDATE to a separate capture route and
serves the REFERENCE while shadowing the candidate, which is why its credential
records reference_served=true. The credentialed production arm is therefore the
first and only route that has ever put this kernel on the served path.

That is a better screen than a diagnostic bool would have been, for a reason
worth stating: the production arm carries an ENGAGEMENT NEEDLE that proves, per
decode call, that the fold actually replaced the incumbent launch -- one
physical launch, grid z equal to the batch, and zero state export writes and
zero parent reads. A silent fallback to the reference cannot produce those
zeros. So a measurement taken here cannot be a measurement of the incumbent
wearing the candidate's name, which is the failure mode a timing pair most needs
to exclude.

DISCLOSED CANDIDATE-SIDE OVERHEAD -- THE PAIR IS CONSERVATIVE
-------------------------------------------------------------
The candidate arm ALSO pays the engagement needle itself, and both arms run with
FR10_METRICS=1 because the production arm's contract requires the invocation
counter. Neither cost is subtracted. As with the FA2 pair's bias retag, the
overhead that SELECTS and PROVES the candidate stays charged to the candidate,
so this pair can only understate a gain. A positive verdict is therefore a floor
on the true effect, and a null is not automatically a pure kernel result.

THE PLACEBO IS FREE, AND IT IS WITHIN-ARM
------------------------------------------
The B4 selector folds only when the engine batch is exactly 4. Inside a pool16
arm the engine batch varies 1-4 step to step, so the SAME arm contains treated
(width 4) and untreated (widths 1-3) steps. That is a stronger placebo than the
FA2 pair had: it shares the arm, the host, the page cache and the task mix with
the treated stratum, so a difference-in-differences over it removes any
arm-level nuisance that is not width-specific.

TREATED WIDTH IS DECLARED, NEVER INFERRED
------------------------------------------
`treated_widths` reads the width off the candidate's own credential and
cross-checks it against the container environment. It never uses max(width) or
"the width where the effect is". The FA2 reducer documents why in blood:
inferring the treated set can classify a treated width as a placebo, hand it to
difference-in-differences as the control, and then print "placebo clean" over
contaminated rows -- it does not fail loudly, it silently subtracts the effect
from itself.

STOP RULE (pre-registered, from the scope)
-------------------------------------------
Width-4 stratum improvement below 5.5 ms/step HALTS the lever. That is the
scope's floor estimate for the width-4 saving (7.698 ms/step) less one sealed
MDE (4.20 ms), rounded up. For reference the phase-0 direct measurement at b=4
priced the fold at 8.984 ms/step, so the screen is being asked to reproduce
roughly twice the halt threshold.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fr13_b4_width4_window_reduce as w4  # noqa: E402
import fr13_b4_batch_conditioned_wall as bcw  # noqa: E402


SCHEMA = "fr13.b4_gdn_single_launch_width4_timing_pair.v1"
RUN_CLASS = "b4_gdn_single_launch_width4_screen"
TITLE = "B4 GDN SINGLE-LAUNCH SCREEN AT THE WIDTH-4 OPERATING POINT"

CANDIDATE_ID = "fixed32_gdn_single_launch_tree_v2"
CREDENTIAL_RELPATH = "logs/fr13_fixed32_gdn_single_launch.production_credential.json"
CONTAINER_ENV_RELPATH = "container_env.txt"

# Pre-registered, from results/fr13_single_launch_b4_scope_20260814/scope.json.
HALT_BELOW_MS_PER_STEP = 5.5
PHASE0_MEASURED_MS_PER_STEP = 8.984
SEALED_MDE_MS = 4.20

# Lower is better for a wall; the sign convention below turns every reported
# number into "positive means the candidate is better" so no reader has to
# remember which way a field points.
LOWER_IS_BETTER = ("step_wall_ms",)
DELTA_FIELDS = ("step_wall_ms", "per_request_step_tps")


class PairError(RuntimeError):
    """The pair cannot be reduced into an honest verdict."""


def _read_container_env(arm_dir: Path) -> dict[str, str]:
    path = arm_dir / CONTAINER_ENV_RELPATH
    if not path.is_file() or path.is_symlink():
        raise PairError(f"{arm_dir.name} lacks a regular container_env.txt")
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key] = value
    return out


def _exact_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PairError(f"{label} is not a regular non-symlink file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PairError(f"{label} is not strict ASCII JSON: {error}") from error
    if not isinstance(payload, dict):
        raise PairError(f"{label} is not a JSON object")
    return payload


def verify_single_variable_delta(
    control_dir: Path, candidate_dir: Path, *, source_commit: str
) -> dict[str, Any]:
    """Prove the two arms differ in the served GDN kernel and nothing else.

    Everything compared here is read from the CONTAINER's own environment
    record, not from what the runner intended to set. An arm that silently ran a
    different geometry is the one failure that would make the whole verdict a
    fiction, so it is refused here rather than discovered in the numbers.
    """
    control = _read_container_env(control_dir)
    candidate = _read_container_env(candidate_dir)

    # The arm variable itself, in both directions. NAMED 0 matters: an unset
    # variable would let the registry default decide, and then the control arm
    # would be whatever the branch happens to ship rather than the incumbent.
    if control.get("FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION") != "0":
        raise PairError(
            "control arm did not NAME the GDN single-launch production arm 0; "
            "an unset selector lets the registry default choose the control"
        )
    if candidate.get("FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION") != "1":
        raise PairError("candidate arm did not serve the GDN single-launch arm")

    # THE METRICS-MATCHED CONTROL FLAG. The production arm sits in the
    # FR10_METRICS=1 expectation class while a plain fixed32 arm mandates the
    # counter OFF, so without this flag the pair is unrunnable at any setting --
    # the candidate cannot go to 0 and the control cannot go to 1. The flag
    # equalises the control upward and selects no kernel. It is the ONE field
    # allowed to differ between the arms besides the selector, so it is checked
    # explicitly in both directions rather than left out of the drift comparison
    # and forgotten.
    if control.get("FR13_FIXED32_GDN_SINGLE_LAUNCH_TIMING_CONTROL") != "1":
        raise PairError(
            "control arm is not the metrics-matched control; its counter state "
            "does not match the candidate and the contrast is not comparable"
        )
    if candidate.get("FR13_FIXED32_GDN_SINGLE_LAUNCH_TIMING_CONTROL") != "0":
        raise PairError(
            "candidate arm carries the control-only metrics flag; the production "
            "arm is already in the metrics=1 class and must not also be a control"
        )
    if control.get("FR10_METRICS") != "1":
        raise PairError(
            "the pair did not run with the invocation counter on; the candidate "
            "engagement needle cannot have proven the fold engaged"
        )

    # Everything that must be IDENTICAL. If any of these differ the pair is not
    # single-variable and no amount of windowing repairs it.
    shared = (
        "FR13_FIXED32_MODE",
        "MAX_NUM_SEQS",
        "SWE_CONCURRENCY",
        "FR13_DRAFT_VOCAB_K",
        "FR13_DRAFT_VOCAB_ROOT",
        "FR13_DRAFT_VOCAB_BLOCKS",
        "FR13_TREE_GDN_GEOM_OVERRIDE",
        "CUDAGRAPH_MODE",
        "ENFORCE_EAGER",
        "FR13_RING_EXPORT",
        "FR13_FLAGS_INKERNEL",
        "FR13_SCAN_ALIGN",
        "FR13_NPAD_INVARIANT",
        "FR10_METRICS",
        "FR13_B4_TASK_REFILL",
        "FR13_FIXED32_B1_DIAGNOSTIC",
    )
    drift = {
        name: (control.get(name), candidate.get(name))
        for name in shared
        if control.get(name) != candidate.get(name)
    }
    if drift:
        raise PairError(f"arms are not single-variable: {drift!r}")

    # No sibling GDN or FA2 candidate may be engaged on EITHER arm; one of those
    # would be a second treatment riding along with the one being measured.
    for env, arm in ((control, "control"), (candidate, "candidate")):
        for name in (
            "FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION",
            "FR13_FIXED32_BATCH_GDN_BYTE_AB",
            "FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB",
            "FR13_FIXED32_BATCH_GDN_PRODUCTION",
        ):
            if env.get(name, "0") not in ("", "0"):
                raise PairError(f"{arm} arm engaged a sibling selector {name}")
        for name in (
            "FR13_FIXED32_GDN_PATH_BV_CANDIDATE",
            "FR13_FIXED32_GDN_PATH_BV_PRODUCTION",
            "FR13_FIXED32_BATCH_GDN_BV_CANDIDATE",
            "FR13_FIXED32_BATCH_GDN_BV_PRODUCTION",
        ):
            if env.get(name, ""):
                raise PairError(f"{arm} arm engaged a sibling selector {name}")

    credential = _exact_json(
        candidate_dir / CREDENTIAL_RELPATH, "candidate production credential"
    )
    if credential.get("candidate") != CANDIDATE_ID:
        raise PairError("candidate credential is not the single-launch arm")
    if credential.get("status") != "PASS":
        raise PairError("candidate credential is not a PASS")
    if source_commit and credential.get("source_commit") != source_commit:
        raise PairError(
            "candidate credential is not bound to the serving commit; the "
            "credential is strictly HEAD-bound and this run is not it"
        )
    return {
        "mode": control.get("FR13_FIXED32_MODE"),
        "slots": int(control.get("MAX_NUM_SEQS", "0") or 0),
        "metrics_on_both_arms": control.get("FR10_METRICS"),
        "control_is_metrics_matched": True,
        "credential_scope": credential.get("credential_scope"),
        "credential_expected_batch": credential.get("expected_batch"),
        "credential_source_commit": credential.get("source_commit"),
        "shared_env_fields_compared": list(shared),
    }


def treated_widths(candidate_dir: Path) -> tuple[int, ...]:
    """Which widths the candidate arm was AUTHORISED to fold, from its own record.

    Declared, never inferred. The GDN fold is authorised at exactly one width --
    the selector folds only when the engine batch equals the credentialed
    production batch -- so the treated set is a singleton and the remaining
    served widths are genuinely untreated.

    Two independent sources must agree: the credential's `expected_batch` and
    the container environment's production batch. Requiring both means a
    credential swapped after launch, or an environment that disagrees with the
    licence it presented, is refused instead of quietly redefining which stratum
    is the placebo.
    """
    credential = _exact_json(
        candidate_dir / CREDENTIAL_RELPATH, "candidate production credential"
    )
    declared = credential.get("expected_batch")
    if not isinstance(declared, int) or isinstance(declared, bool):
        raise PairError("candidate credential declares no integer expected_batch")
    env = _read_container_env(candidate_dir)
    env_batch = env.get("FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_BATCH", "")
    if env_batch != str(declared):
        raise PairError(
            "credential expected_batch and container production batch disagree; "
            f"{declared!r} vs {env_batch!r}"
        )
    return (declared,)


def _pool(by_width: dict[str, Any], widths: tuple[int, ...]) -> dict[str, Any]:
    """Step-weighted pooling of a set of width strata."""
    steps = 0
    weighted = 0.0
    present: list[int] = []
    for width in widths:
        cell = by_width.get(str(width))
        if not cell:
            continue
        present.append(width)
        steps += int(cell["steps"])
        weighted += float(cell["mean_ms"]) * int(cell["steps"])
    if steps == 0:
        return {"available": False, "widths": list(widths), "steps": 0}
    return {
        "available": True,
        "widths": present,
        "steps": steps,
        "mean_ms": weighted / steps,
    }


def contrast(control_pool: dict[str, Any], candidate_pool: dict[str, Any]) -> dict[str, Any]:
    """Improvement in ms/step, positive when the candidate is faster."""
    if not control_pool.get("available") or not candidate_pool.get("available"):
        return {
            "available": False,
            "reason": "one side of the contrast has no steps in its strata",
        }
    control_ms = float(control_pool["mean_ms"])
    candidate_ms = float(candidate_pool["mean_ms"])
    return {
        "available": True,
        "control_mean_ms": control_ms,
        "candidate_mean_ms": candidate_ms,
        "improvement_ms_per_step": control_ms - candidate_ms,
        "improvement_pct_of_control": (
            (control_ms - candidate_ms) / control_ms * 100.0 if control_ms else None
        ),
        "control_steps": control_pool["steps"],
        "candidate_steps": candidate_pool["steps"],
    }


def delta_block(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Whole-window signed deltas, oriented so positive means candidate better."""
    out: dict[str, Any] = {}
    for field in DELTA_FIELDS:
        c0 = control.get(field)
        c1 = candidate.get(field)
        if not isinstance(c0, (int, float)) or isinstance(c0, bool):
            continue
        if not isinstance(c1, (int, float)) or isinstance(c1, bool):
            continue
        c0 = float(c0)
        c1 = float(c1)
        improvement = (c0 - c1) if field in LOWER_IS_BETTER else (c1 - c0)
        out[field] = {
            "control": c0,
            "candidate": c1,
            "improvement": improvement,
            "improvement_pct_of_control": (improvement / c0 * 100.0) if c0 else None,
            "orientation": (
                "lower is better" if field in LOWER_IS_BETTER else "higher is better"
            ),
        }
    return out


def build_payload(
    *,
    runroot: Path,
    control_dir: Path,
    candidate_dir: Path,
    mode: str,
    source_commit: str,
) -> dict[str, Any]:
    identity = verify_single_variable_delta(
        control_dir, candidate_dir, source_commit=source_commit
    )
    treated = treated_widths(candidate_dir)

    control_record = w4.reduce_window_arm(control_dir, mode=mode, pass_index=0)
    candidate_record = w4.reduce_window_arm(candidate_dir, mode=mode, pass_index=0)
    for label, record in (("control", control_record), ("candidate", candidate_record)):
        if not record.get("included"):
            raise PairError(
                f"{label} arm has no usable width-4 window: "
                f"{record.get('exclusion_reason')}"
            )

    sidecar_dir = runroot / "fr13_sfwd_sidecar"
    control_bcw = bcw.analyse_arm(control_record, sidecar_dir, arm_name=control_dir.name)
    candidate_bcw = bcw.analyse_arm(
        candidate_record, sidecar_dir, arm_name=candidate_dir.name
    )
    control_widths = control_bcw.get("by_width", {})
    candidate_widths = candidate_bcw.get("by_width", {})

    slots = int(identity["slots"])
    placebo = tuple(w for w in range(1, slots + 1) if w not in treated)
    if not placebo:
        raise PairError(
            "no untreated width remains; the free within-arm placebo does not "
            "exist for this configuration and the DiD would be vacuous"
        )

    treated_contrast = contrast(
        _pool(control_widths, treated), _pool(candidate_widths, treated)
    )
    placebo_contrast = contrast(
        _pool(control_widths, placebo), _pool(candidate_widths, placebo)
    )

    did: dict[str, Any] = {"available": False}
    if treated_contrast.get("available") and placebo_contrast.get("available"):
        did = {
            "available": True,
            "treated_improvement_ms_per_step": treated_contrast[
                "improvement_ms_per_step"
            ],
            "placebo_improvement_ms_per_step": placebo_contrast[
                "improvement_ms_per_step"
            ],
            "difference_in_differences_ms_per_step": (
                treated_contrast["improvement_ms_per_step"]
                - placebo_contrast["improvement_ms_per_step"]
            ),
            "interpretation": (
                "The placebo strata are served by the incumbent on BOTH arms, so "
                "their contrast estimates arm-level nuisance -- host ageing, page "
                "cache, task mix -- that is not width-specific. A placebo far from "
                "zero means the treated contrast is carrying that nuisance too, and "
                "the difference-in-differences is the width-specific residual."
            ),
        }

    verdict = "INCONCLUSIVE"
    verdict_reason = "the treated contrast could not be computed"
    if treated_contrast.get("available"):
        improvement = treated_contrast["improvement_ms_per_step"]
        if improvement < HALT_BELOW_MS_PER_STEP:
            verdict = "HALT"
            verdict_reason = (
                f"width-{treated[0]} improvement {improvement:.3f} ms/step is below "
                f"the pre-registered stop rule {HALT_BELOW_MS_PER_STEP} ms/step "
                "(the scope's width-4 floor estimate less one sealed MDE). The "
                "lever does not clear its own floor at the operating point and "
                "should not proceed to a sealing campaign on this evidence."
            )
        else:
            verdict = "PROCEED"
            verdict_reason = (
                f"width-{treated[0]} improvement {improvement:.3f} ms/step clears "
                f"the pre-registered stop rule {HALT_BELOW_MS_PER_STEP} ms/step. "
                "This is a SCREEN, not a seal: it motivates the phase-4 sealing "
                "campaign and does not substitute for one."
            )

    return {
        "schema": SCHEMA,
        "run_classification": RUN_CLASS,
        "title": TITLE,
        "candidate": CANDIDATE_ID,
        "source_commit": source_commit,
        "mode": mode,
        "identity": identity,
        "treated_widths": list(treated),
        "placebo_widths": list(placebo),
        "treated_width_provenance": (
            "declared by the candidate credential's expected_batch and "
            "cross-checked against the container production batch; never "
            "inferred from the shape of the timing data"
        ),
        "control_window": control_record,
        "candidate_window": candidate_record,
        "control_batch_conditioned": control_bcw,
        "candidate_batch_conditioned": candidate_bcw,
        "whole_window_delta": delta_block(
            control_record.get("windowed", {}), candidate_record.get("windowed", {})
        ),
        "treated_contrast": treated_contrast,
        "placebo_contrast": placebo_contrast,
        "difference_in_differences": did,
        "stop_rule": {
            "halt_below_ms_per_step": HALT_BELOW_MS_PER_STEP,
            "basis": (
                "the scope's width-4 floor estimate 7.698 ms/step less one sealed "
                "MDE 4.20 ms, rounded up"
            ),
            "phase0_direct_measurement_ms_per_step": PHASE0_MEASURED_MS_PER_STEP,
            "sealed_mde_ms": SEALED_MDE_MS,
        },
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "conservative_by_construction": (
            "The candidate arm pays the engagement needle that proves the fold "
            "engaged, and both arms run with the invocation counter on because "
            "the production contract requires it. Neither cost is subtracted, so "
            "a positive result is a floor on the true effect."
        ),
        "not_a_seal": (
            "The width-4 window class is an instrument, not a citable seal. This "
            "screen can halt the lever; only a pre-registered multi-pass campaign "
            "with balanced arm order and a one-sided lower bound can seal it."
        ),
    }


def render(payload: dict[str, Any]) -> str:
    lines = [payload["title"], "=" * len(payload["title"]), ""]
    lines.append(f"candidate      {payload['candidate']}")
    lines.append(f"commit         {payload['source_commit']}")
    lines.append(f"treated widths {payload['treated_widths']}")
    lines.append(f"placebo widths {payload['placebo_widths']}")
    lines.append("")
    treated = payload["treated_contrast"]
    if treated.get("available"):
        lines.append(
            f"treated  control {treated['control_mean_ms']:.3f} ms/step  "
            f"candidate {treated['candidate_mean_ms']:.3f}  "
            f"improvement {treated['improvement_ms_per_step']:+.3f} ms/step "
            f"({treated['control_steps']}/{treated['candidate_steps']} steps)"
        )
    placebo = payload["placebo_contrast"]
    if placebo.get("available"):
        lines.append(
            f"placebo  control {placebo['control_mean_ms']:.3f} ms/step  "
            f"candidate {placebo['candidate_mean_ms']:.3f}  "
            f"improvement {placebo['improvement_ms_per_step']:+.3f} ms/step "
            f"({placebo['control_steps']}/{placebo['candidate_steps']} steps)"
        )
    did = payload["difference_in_differences"]
    if did.get("available"):
        lines.append(
            f"DiD      {did['difference_in_differences_ms_per_step']:+.3f} ms/step"
        )
    lines.append("")
    lines.append(f"VERDICT  {payload['verdict']}")
    lines.append(f"         {payload['verdict_reason']}")
    lines.append("")
    lines.append(payload["not_a_seal"])
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--runroot", type=Path, default=None)
    parser.add_argument("--control-arm", type=Path, default=None)
    parser.add_argument("--candidate-arm", type=Path, default=None)
    parser.add_argument("--mode", type=str, default="hydra27_fixed32")
    parser.add_argument("--source-commit", type=str, default="")
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.self_check:
        # Proves the sealed window class is importable and still exposes the
        # surface this reducer builds on, without needing any run present.
        for name in ("reduce_window_arm", "Width4Error", "MIN_WINDOW_STEPS"):
            if not hasattr(w4, name):
                raise SystemExit(f"sealed window reducer lost {name}")
        if not hasattr(bcw, "analyse_arm"):
            raise SystemExit("batch-conditioned wall lost analyse_arm")
        print("self-check OK")
        return 0
    if not (args.runroot and args.control_arm and args.candidate_arm):
        raise SystemExit("--runroot, --control-arm and --candidate-arm are required")
    try:
        payload = build_payload(
            runroot=args.runroot,
            control_dir=args.control_arm,
            candidate_dir=args.candidate_arm,
            mode=args.mode,
            source_commit=args.source_commit,
        )
    except (PairError, w4.Width4Error) as error:
        raise SystemExit(f"GDN single-launch width-4 pair refused: {error}")
    text = render(payload)
    print(text)
    if args.out:
        args.out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0 if payload["verdict"] in ("PROCEED", "HALT") else 3


if __name__ == "__main__":
    raise SystemExit(main())

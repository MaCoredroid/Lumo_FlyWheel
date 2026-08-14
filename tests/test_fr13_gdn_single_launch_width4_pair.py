"""The phase-3 width-4 screen: runner refusals and reducer honesty.

This file exists because the runner and reducer it covers have NEVER RUN. Every
GPU-side defect in a never-run runner costs a full re-gate to fix, because the
credential the run depends on is bound to HEAD and a fix-commit moves HEAD. So
every branch that can be reached without a GPU is reached here instead.

It has already earned its keep once: it caught a bash parse error where an
apostrophe inside a ``${VAR:?message}`` expansion opened a quote and broke the
parse sixty lines later, at a token with nothing to do with the cause. That
regression is pinned below by name.

WHAT THE SCREEN IS
------------------
A single-variable pair over the folded GDN scan kernel: control serves the
deployed two-launch reference, candidate serves ``fixed32_gdn_single_launch_tree_v2``.
It can HALT the lever via the pre-registered 5.5 ms/step stop rule. It cannot
seal it -- the width-4 window class is an instrument, not a citable seal, and
sealing is a multi-pass campaign with balanced arm order and a one-sided lower
bound.

WHY THE SCREEN RUNS THROUGH THE PRODUCTION ARM
----------------------------------------------
Because nothing else serves this kernel. The credential-free bool
``FR13_FIXED32_GDN_SINGLE_LAUNCH`` is structurally unreachable: its sidecar is
only ever removed by the launcher and never written, and the variable is never
exported into the container. That is pinned here too, because if somebody later
makes it reachable, the screen's rationale changes and this test should be the
thing that says so.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
RUNNER = SCRIPTS / "fr13_run_b4_gdn_single_launch_width4_timing.sh"
REDUCER = SCRIPTS / "fr13_b4_gdn_single_launch_width4_pair_reduce.py"
LAUNCHER = SCRIPTS / "fr13_launch_forked_fa2_tree_server.sh"
PYTHON = REPO / ".venv" / "bin" / "python"

sys.path.insert(0, str(SCRIPTS))
import fr13_b4_gdn_single_launch_width4_pair_reduce as pr  # noqa: E402


ENABLE = "FR13_RUN_B4_GDN_SINGLE_LAUNCH_WIDTH4_TIMING"
ZERO64 = "0" * 64


def _run_runner(env_overrides: dict[str, str]) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ)
    env.update(env_overrides)
    return subprocess.run(
        ["bash", str(RUNNER)], capture_output=True, text=True, env=env, cwd=str(REPO)
    )


def _base_env(**kw) -> dict[str, str]:
    env = {
        ENABLE: "1",
        "RUNROOT": "output/does_not_exist_screen_test",
        "TAG": "t1",
        "GDN_SL_CREDENTIAL": "/etc/hostname",
        "GDN_SL_CREDENTIAL_SHA256": ZERO64,
        "FORKED_FA2_SO": "/etc/hostname",
    }
    env.update(kw)
    return env


# ------------------------------------------------------------------ the runner


def test_runner_is_disabled_by_default() -> None:
    done = _run_runner({ENABLE: "0"})
    assert done.returncode == 2
    assert "is disabled" in done.stderr


def test_runner_refuses_a_non_binary_enable_value() -> None:
    done = _run_runner({ENABLE: "2"})
    assert done.returncode == 2
    assert "must be exactly 0 or 1" in done.stderr


def test_runner_parses_cleanly() -> None:
    done = subprocess.run(
        ["bash", "-n", str(RUNNER)], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr


def test_no_apostrophe_inside_a_required_env_message() -> None:
    """REGRESSION. Bash tracks quoting inside ``${VAR:?word}``.

    A lone apostrophe there opens a quote and the parse dies far below, at a
    token unrelated to the cause. This cost real debugging time once; it is
    cheap to make impossible.
    """
    for line in RUNNER.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(': "${') and ":?" in stripped:
            assert "'" not in stripped, f"apostrophe in a :? message: {stripped}"


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"TAG": "bad tag"}, "TAG contains unsafe characters"),
        ({"GDN_SL_FIXED32_MODE": "nope"}, "must be tail6_fixed32 or hydra27_fixed32"),
        ({"GDN_SL_PRODUCTION_BATCH": "3"}, "must be 1 or 4"),
        ({"ARM_ORDER": "SC"}, "ARM_ORDER must be TC"),
        ({"RUNROOT": "/tmp/evil"}, "RUNROOT must resolve below"),
        ({"RUNROOT": "output"}, "RUNROOT must resolve below"),
        ({"PASS_INDEX": "x"}, "PASS_INDEX must be a non-negative integer"),
        ({"PASS_ROOT": "/tmp/p"}, "PASS_ROOT must resolve below"),
        ({"GDN_SL_CREDENTIAL": "etc/hostname"}, "must be an absolute regular"),
        ({}, "credential identity mismatch"),
    ],
)
def test_runner_refusal_branches(overrides, expected) -> None:
    done = _run_runner(_base_env(**overrides))
    assert done.returncode != 0
    assert expected in done.stderr, done.stderr[-400:]


def test_a_bad_runroot_is_not_reported_as_a_pass_root_problem() -> None:
    """PASS_ROOT defaults to RUNROOT, so ordering decides which error a reader sees.

    A misleading first error is expensive in a runner nobody has run before.
    """
    done = _run_runner(_base_env(RUNROOT="/tmp/evil"))
    assert "RUNROOT must resolve below" in done.stderr
    assert "PASS_ROOT must resolve below" not in done.stderr


def test_the_arms_differ_in_exactly_one_selector() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    # one run_arm call per arm, and the ONLY argument that differs is the
    # production flag
    assert 'run_arm "$CONTROL_ARM" 0' in text
    assert 'run_arm "$CANDIDATE_ARM" 1' in text
    # the control must NAME 0 rather than leave the selector unset, or the
    # registry default would choose the control
    assert 'FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION="$production"' in text
    # both arms carry the counter, because the production contract requires it
    assert text.count("FR10_METRICS=1") >= 1
    # and no sibling GDN selector rides along
    for name in (
        "FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION=0",
        "FR13_FIXED32_GDN_PATH_BV_CANDIDATE=",
        "FR13_FIXED32_GDN_PATH_BV_PRODUCTION=",
        "FR13_FIXED32_GDN_SINGLE_LAUNCH_EXPECTED_BATCH=",
    ):
        assert name in text


def test_both_arms_land_in_one_pass_dir_for_the_sealed_reducer() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert 'PASS_DIR="$PASS_ROOT/pass_$(printf' in text
    assert 'RUNROOT="$PASS_DIR"' in text
    assert "--gate-root" in text


def test_runner_resolves_its_instruments_before_spending_gpu_time() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    self_check = text.index("--self-check")
    first_serve = text.index("fr13_bigdenom_swe_serve_variant.sh")
    assert self_check < first_serve
    credential_check = text.index("$CREDENTIAL_VALIDATOR")
    assert credential_check < first_serve


# ------------------------------------------------- the unreachable diagnostic


def test_the_credential_free_diagnostic_bool_is_still_unreachable() -> None:
    """The screen's whole rationale rests on this.

    If somebody makes the bool reachable, the screen could have run without a
    credential and this test should be what raises the question.
    """
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "-e FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE=" not in launcher
    mentions = [
        line
        for line in launcher.splitlines()
        if "fr13_fixed32_gdn_single_launch_tree.arm" in line
    ]
    assert mentions, "sidecar disappeared entirely; re-derive the rationale"
    # every mention is inside an rm -f list, i.e. removal only, never a write
    assert all(">" not in line for line in mentions), mentions


# ------------------------------------------------------------------ the reducer


def test_reducer_self_check() -> None:
    done = subprocess.run(
        [str(PYTHON), str(REDUCER), "--self-check"], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr


def _arm(tmp: Path, name: str, env: dict[str, str], credential: dict | None) -> Path:
    d = tmp / name
    (d / "logs").mkdir(parents=True)
    (d / "container_env.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in env.items()), encoding="utf-8"
    )
    if credential is not None:
        (d / pr.CREDENTIAL_RELPATH).write_text(json.dumps(credential), encoding="ascii")
    return d


SHARED = {
    "FR13_FIXED32_MODE": "hydra27_fixed32",
    "MAX_NUM_SEQS": "4",
    "SWE_CONCURRENCY": "4",
    "FR13_DRAFT_VOCAB_K": "65536",
    "FR13_DRAFT_VOCAB_ROOT": "1",
    "FR13_DRAFT_VOCAB_BLOCKS": "/workspace/scripts/fr13_dvk_subset_blocks.json",
    "FR13_TREE_GDN_GEOM_OVERRIDE": "BV=8",
    "CUDAGRAPH_MODE": "FULL_AND_PIECEWISE",
    "ENFORCE_EAGER": "0",
    "FR13_RING_EXPORT": "1",
    "FR13_FLAGS_INKERNEL": "1",
    "FR13_SCAN_ALIGN": "0",
    "FR13_NPAD_INVARIANT": "0",
    "FR10_METRICS": "1",
    "FR13_B4_TASK_REFILL": "1",
    "FR13_FIXED32_B1_DIAGNOSTIC": "0",
}
CREDENTIAL = {
    "candidate": pr.CANDIDATE_ID,
    "status": "PASS",
    "source_commit": "a" * 40,
    "expected_batch": 4,
    "credential_scope": "hydra27:b4",
}


def _pair(tmp: Path, *, control_extra=None, candidate_extra=None, credential=None):
    control_env = dict(SHARED, FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION="0")
    control_env.update(control_extra or {})
    candidate_env = dict(
        SHARED,
        FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION="1",
        FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_BATCH="4",
    )
    candidate_env.update(candidate_extra or {})
    control = _arm(tmp, "control", control_env, None)
    candidate = _arm(
        tmp, "candidate", candidate_env,
        CREDENTIAL if credential is None else credential,
    )
    return control, candidate


def test_reducer_accepts_a_clean_single_variable_pair(tmp_path) -> None:
    control, candidate = _pair(tmp_path)
    identity = pr.verify_single_variable_delta(
        control, candidate, source_commit="a" * 40
    )
    assert identity["slots"] == 4
    assert pr.treated_widths(candidate) == (4,)


def test_control_must_name_the_arm_zero_not_leave_it_unset(tmp_path) -> None:
    control_env = dict(SHARED)  # selector absent entirely
    candidate_env = dict(
        SHARED,
        FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION="1",
        FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_BATCH="4",
    )
    control = _arm(tmp_path, "control", control_env, None)
    candidate = _arm(tmp_path, "candidate", candidate_env, CREDENTIAL)
    with pytest.raises(pr.PairError, match="did not NAME"):
        pr.verify_single_variable_delta(control, candidate, source_commit="a" * 40)


def test_a_geometry_difference_is_refused(tmp_path) -> None:
    control, candidate = _pair(tmp_path, control_extra={"CUDAGRAPH_MODE": "FULL"})
    with pytest.raises(pr.PairError, match="not single-variable"):
        pr.verify_single_variable_delta(control, candidate, source_commit="a" * 40)


def test_a_sibling_selector_on_either_arm_is_refused(tmp_path) -> None:
    control, candidate = _pair(
        tmp_path, candidate_extra={"FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION": "1"}
    )
    with pytest.raises(pr.PairError, match="sibling selector"):
        pr.verify_single_variable_delta(control, candidate, source_commit="a" * 40)


def test_a_credential_for_the_other_folded_arm_is_refused(tmp_path) -> None:
    other = dict(CREDENTIAL, candidate="fixed32_gdn_single_launch_gqa_group3_v1")
    control, candidate = _pair(tmp_path, credential=other)
    with pytest.raises(pr.PairError, match="not the single-launch arm"):
        pr.verify_single_variable_delta(control, candidate, source_commit="a" * 40)


def test_a_credential_bound_to_another_commit_is_refused(tmp_path) -> None:
    control, candidate = _pair(tmp_path)
    with pytest.raises(pr.PairError, match="strictly HEAD-bound"):
        pr.verify_single_variable_delta(control, candidate, source_commit="b" * 40)


def test_treated_width_is_declared_twice_and_must_agree(tmp_path) -> None:
    """Never inferred. Two independent declarations must agree.

    Inferring the treated set can classify a treated width as the placebo, hand
    it to difference-in-differences as the control, and then print "placebo
    clean" over contaminated rows -- it does not fail loudly.
    """
    control, candidate = _pair(
        tmp_path,
        candidate_extra={"FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION_BATCH": "1"},
    )
    with pytest.raises(pr.PairError, match="disagree"):
        pr.treated_widths(candidate)


def test_contrast_orientation_is_positive_when_candidate_is_faster() -> None:
    control_pool = {"available": True, "mean_ms": 100.0, "steps": 10}
    candidate_pool = {"available": True, "mean_ms": 91.0, "steps": 10}
    out = pr.contrast(control_pool, candidate_pool)
    assert out["improvement_ms_per_step"] == pytest.approx(9.0)
    assert out["improvement_pct_of_control"] == pytest.approx(9.0)


def test_pooling_is_step_weighted() -> None:
    by_width = {
        "1": {"steps": 1, "mean_ms": 10.0, "fraction": 0.1, "sd_ms": 0.0},
        "2": {"steps": 9, "mean_ms": 20.0, "fraction": 0.9, "sd_ms": 0.0},
    }
    pooled = pr._pool(by_width, (1, 2))
    assert pooled["steps"] == 10
    assert pooled["mean_ms"] == pytest.approx(19.0)


def test_absent_strata_pool_to_unavailable_rather_than_zero() -> None:
    assert pr._pool({}, (4,))["available"] is False
    assert pr.contrast(pr._pool({}, (4,)), pr._pool({}, (4,)))["available"] is False


def test_the_stop_rule_is_the_pre_registered_one() -> None:
    assert pr.HALT_BELOW_MS_PER_STEP == 5.5
    assert pr.PHASE0_MEASURED_MS_PER_STEP == pytest.approx(8.984)
    assert pr.SEALED_MDE_MS == pytest.approx(4.20)


def test_the_reducer_never_claims_to_seal() -> None:
    text = REDUCER.read_text(encoding="utf-8")
    assert "instrument, not a citable seal" in text
    assert "cannot seal" in text or "only phase 4" in text.lower()

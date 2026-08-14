"""The GQA-pair FA2 width-4 re-test: runner wiring and paired-verdict math.

The point of this suite is that the RE-TEST cannot quietly become a different
measurement than the one it claims to be. Three things must hold or the verdict
is worthless:

  1. the arms serve the width-4 regime (16-task pool, refill on, no agent wall)
     at the UNCHANGED qualified geometry (4 slots, 128 query rows);
  2. the delta is judged on a basis whose threshold MATCHES that basis -- a
     blended delta against a batch-conditioned MDE would apply a threshold ~9%
     too small and could manufacture a reversal;
  3. the thresholds come from the sealed artifact, not from this file.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RUNNER = REPO / "scripts/fr13_run_b4_gqa_width4_timing.sh"
REDUCER = REPO / "scripts/fr13_b4_gqa_width4_pair_reduce.py"
SEALED = REPO / "results/fr13_b4_width4_nsys_20260813/fr13_b4_batch_conditioned_wall.json"
POOL16_SHA = "47b0a3c9be49e2cb5f7e7217ae03c267a05359f269f3e3b038942f57d7dc0b5c"
EXACT4_SHA = "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"


def _reducer():
    spec = importlib.util.spec_from_file_location("w4_pair_reduce", REDUCER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["w4_pair_reduce"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# runner wiring
# --------------------------------------------------------------------------
def test_runner_parses_and_is_executable() -> None:
    assert RUNNER.is_file() and not RUNNER.is_symlink()
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)


def test_runner_is_disabled_unless_explicitly_enabled() -> None:
    proc = subprocess.run(
        ["bash", str(RUNNER)], capture_output=True, text=True, cwd=REPO
    )
    assert proc.returncode == 2
    assert "disabled" in proc.stderr


@pytest.mark.parametrize("value", ["2", "yes", "01"])
def test_runner_refuses_a_non_binary_enable_flag(value: str) -> None:
    proc = subprocess.run(
        ["bash", str(RUNNER)],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={"PATH": "/usr/bin:/bin", "FR13_RUN_B4_GQA_WIDTH4_TIMING": value},
    )
    assert proc.returncode == 2
    assert "must be exactly 0 or 1" in proc.stderr


def test_runner_binds_the_sixteen_task_pool_not_exact4() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "config/fr13_fixed32/subset_b4_sixteen.json" in text
    assert f"SUBSET_SHA256={POOL16_SHA}" in text
    assert "subset_b4_four.json" not in text
    assert EXACT4_SHA not in text
    assert "TASK_COUNT=16" in text
    # the last of the sixteen, so a truncated list cannot pass unnoticed
    assert "astropy__astropy-14995" in text


def test_runner_serves_the_pool_regime_with_no_agent_wall() -> None:
    """A wall would truncate tasks and deform the ledger that DEFINES the window."""
    text = RUNNER.read_text(encoding="utf-8")
    assert "FR13_B4_TASK_REFILL=1" in text
    assert "AGENT_WALL_S= \\" in text, "the agent wall must be passed EMPTY"
    assert "AGENT_WALL_S=5400" not in text


def test_runner_keeps_the_qualified_geometry_pinned() -> None:
    """Pool depth changes; the shape the byte gate qualified must not."""
    text = RUNNER.read_text(encoding="utf-8")
    assert "MAX_NUM_SEQS_OVR=4" in text
    assert "SWE_CONCURRENCY=4" in text
    assert "export BSIZE=4" in text
    assert "export CONC=4" in text
    assert "CUDAGRAPH_MODE=FULL_AND_PIECEWISE" in text
    assert "ENFORCE_EAGER=0" in text
    assert "SELECTOR_SENTINEL=131092" in text


def test_runner_differs_between_arms_in_exactly_one_variable() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert 'run_arm "$STOCK_ARM" stock_dispatch ""' in text
    assert 'run_arm "$CANDIDATE_ARM" gqa_pair gqa_pair' in text
    # the stock arm must be PROVEN not to have engaged the candidate
    assert "emitted a GQA-pair engagement on the stock-dispatch arm" in text


def test_runner_requires_the_admission_ledger_before_reducing() -> None:
    """Six campaign fossils were runners bound to artifacts nothing wrote."""
    text = RUNNER.read_text(encoding="utf-8")
    assert "fr13_task_refill_ledger.jsonl" in text
    assert "fr13_task_refill_summary.json" in text
    assert "the width-4 window is DEFINED by" in text
    assert "--self-check" in text, "the reducer must resolve before GPU time"


def test_runner_lays_arms_out_for_the_sealed_window_reducer() -> None:
    """pass_00/<mode>_* is what fr13_b4_width4_window_reduce.discover_arms globs."""
    text = RUNNER.read_text(encoding="utf-8")
    assert 'PASS_DIR="$RUNROOT_ABS/pass_00"' in text
    assert 'RUNROOT="$PASS_DIR"' in text
    assert "fr13_b4_width4_window_reduce.py" in text
    assert "fr13_b4_gqa_width4_pair_reduce.py" in text


def test_runner_rebinds_the_credential_to_this_commit() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "--expected-source-commit" in text
    assert "$SIDECAR\" validate" in text or "$SIDECAR validate" in text
    assert "$SIDECAR\" verify" in text or "$SIDECAR verify" in text


# --------------------------------------------------------------------------
# thresholds: read from the sealed artifact, never retyped
# --------------------------------------------------------------------------
def test_self_check_resolves_both_topologies() -> None:
    proc = subprocess.run(
        [sys.executable, str(REDUCER), "--self-check"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert proc.returncode == 0, proc.stderr
    assert "hydra27_fixed32" in proc.stdout
    assert "tail6_fixed32" in proc.stdout


@pytest.mark.parametrize(
    "mode,mde,wall",
    [("hydra27_fixed32", 4.204845067020671, 413.14178365521565),
     ("tail6_fixed32", 6.417803846730505, 411.05488226730876)],
)
def test_thresholds_come_from_the_sealed_artifact(mode: str, mde: float, wall: float) -> None:
    module = _reducer()
    t = module.load_sealed_thresholds(REPO, mode)
    assert t["mde_ms"] == pytest.approx(mde, rel=0, abs=1e-9)
    assert t["sealed_width4_step_wall_ms"] == pytest.approx(wall, rel=0, abs=1e-9)
    # and it really is the file on disk, not a constant in the reducer
    sealed = json.loads(SEALED.read_text(encoding="utf-8"))
    block = sealed["pooled"][mode]["batch_conditioned_full_width"]
    assert t["mde_ms"] == block["mde_ms"]


def test_reducer_refuses_a_topology_the_sealed_artifact_does_not_cover() -> None:
    module = _reducer()
    with pytest.raises(module.PairError, match="no width-4 wall for topology"):
        module.load_sealed_thresholds(REPO, "b1_fixed32")


def test_blended_basis_carries_its_own_larger_mde() -> None:
    """The whole point of carrying two bases is that each keeps ITS threshold."""
    module = _reducer()
    for mode in ("hydra27_fixed32", "tail6_fixed32"):
        t = module.load_sealed_thresholds(REPO, mode)
        blend_mde = t["blended_basis"]["mde_ms"]
        assert blend_mde > t["mde_ms"], (
            "the blended basis is noisier, so its MDE must be the larger one; "
            "if these were swapped a blended delta could clear a threshold it "
            "has no right to"
        )


def test_recovery_target_is_ten_percent_of_the_attributed_fa2_cost() -> None:
    module = _reducer()
    t = module.load_sealed_thresholds(REPO, "hydra27_fixed32")
    assert t["fa2_ms_per_step_at_width4"] == pytest.approx(69.748, abs=1e-2)
    assert t["fa2_recovery_target_ms"] == pytest.approx(6.975, abs=1e-3)


# --------------------------------------------------------------------------
# the verdict bands
# --------------------------------------------------------------------------
def _thresholds():
    return _reducer().load_sealed_thresholds(REPO, "hydra27_fixed32")


@pytest.mark.parametrize(
    "improvement,band,disposition",
    [
        (12.0, "GAIN_AT_OR_ABOVE_10PCT_FA2_TARGET", "REVERSES_THE_EXACT4_NULL"),
        (6.975, "GAIN_AT_OR_ABOVE_10PCT_FA2_TARGET", "REVERSES_THE_EXACT4_NULL"),
        (5.0, "GAIN_CLEARS_FOUR_PASS_MDE_BELOW_TARGET", "REVERSES_THE_EXACT4_NULL"),
        (4.204845067020671, "GAIN_CLEARS_FOUR_PASS_MDE_BELOW_TARGET",
         "REVERSES_THE_EXACT4_NULL"),
        (1.0, "GAIN_BELOW_FOUR_PASS_MDE", "CONFIRMS_THE_EXACT4_NULL"),
        (0.0, "NO_GAIN_OR_REGRESSION", "CONFIRMS_THE_EXACT4_NULL"),
        (-3.0, "NO_GAIN_OR_REGRESSION", "CONFIRMS_THE_EXACT4_NULL"),
    ],
)
def test_verdict_bands_are_continuous_and_correctly_ordered(
    improvement: float, band: str, disposition: str
) -> None:
    module = _reducer()
    verdict = module.judge(improvement, _thresholds())
    assert verdict["band"] == band
    assert verdict["lever_disposition"] == disposition
    assert verdict["clears_four_pass_mde"] == (improvement >= _thresholds()["mde_ms"])


def test_verdict_never_claims_significance() -> None:
    module = _reducer()
    verdict = module.judge(9.0, _thresholds())
    assert verdict["is_significance_test"] is False
    assert verdict["n_paired_draws"] == 1
    assert "sd_units_caveat" in verdict


def test_reducer_publishes_the_conservative_bias_and_never_subtracts_it() -> None:
    text = REDUCER.read_text(encoding="utf-8")
    assert "overhead_is_never_subtracted" in text
    assert "bias_direction" in text
    assert "conservative_against_candidate" in text


def test_does_not_claim_covers_the_load_bearing_limits() -> None:
    module = _reducer()
    joined = " ".join(module.DOES_NOT_CLAIM).lower()
    for needle in (
        "statistical significance",
        "whole-arm throughput",
        "exact4 comparability",
        "cap verdict",
        "agent-quality",
        "promotion",
    ):
        assert needle in joined, needle


# --------------------------------------------------------------------------
# fail-closed provenance
# --------------------------------------------------------------------------
def test_engagement_on_the_stock_arm_is_a_hard_failure() -> None:
    module = _reducer()
    with pytest.raises(module.PairError, match="sentinel leaked"):
        module.validate_pair_engagement({"status": "ENGAGED"}, None, expected_task_count=16)


def test_missing_candidate_engagement_is_a_hard_failure() -> None:
    module = _reducer()
    with pytest.raises(module.PairError, match="emitted no GQA-pair engagement"):
        module.validate_pair_engagement(None, None, expected_task_count=16)


def _good_engagement(**overrides):
    record = {
        "status": "ENGAGED",
        "runtime_mode": "FULL",
        "arm": "gqa_pair",
        "layer_count": 16,
        "batch_size": 4,
        "total_query_rows": 128,
        "task_count": 16,
        "candidate_served": True,
        "fallback_allowed": False,
        # The SEALED token. Mark's ruling 2026-08-13 widened it, and readers
        # must keep accepting the old one or the banked +29.50 ms/step
        # width-4 pair stops being re-reducible.
        "candidate_scope": "final_fixed32_b4_full_graph_only",
    }
    record.update(overrides)
    return record


def test_a_correct_pool16_engagement_is_accepted() -> None:
    module = _reducer()
    module.validate_pair_engagement(None, _good_engagement(), expected_task_count=16)


@pytest.mark.parametrize(
    "override",
    [
        {"layer_count": 15},
        {"batch_size": 8},
        {"total_query_rows": 256},
        {"candidate_served": False},
        {"fallback_allowed": True},
        {"runtime_mode": "EAGER"},
        {"candidate_scope": "anything_goes"},
        {"status": "BYPASSED"},
    ],
)
def test_a_drifted_engagement_is_refused(override: dict) -> None:
    module = _reducer()
    with pytest.raises(module.PairError, match="did not serve the GQA-pair kernel"):
        module.validate_pair_engagement(
            None, _good_engagement(**override), expected_task_count=16
        )


def test_an_exact4_engagement_cannot_masquerade_as_the_pool16_run() -> None:
    """The re-test is DEFINED by the pool; a 4-task arm is the old measurement."""
    module = _reducer()
    with pytest.raises(module.PairError, match="expected the 16-task pool"):
        module.validate_pair_engagement(
            None, _good_engagement(task_count=4), expected_task_count=16
        )


# --------------------------------------------------------------------------
# delta orientation
# --------------------------------------------------------------------------
def test_positive_always_means_the_candidate_is_better() -> None:
    module = _reducer()
    stock = {"step_wall_ms": 400.0, "per_request_step_tps": 16.0}
    candidate = {"step_wall_ms": 390.0, "per_request_step_tps": 17.0}
    deltas = module.delta_block(stock, candidate)
    # lower-is-better inverts; higher-is-better does not
    assert deltas["step_wall_ms"]["improvement"] == pytest.approx(10.0)
    assert deltas["step_wall_ms"]["candidate_minus_stock"] == pytest.approx(-10.0)
    assert deltas["per_request_step_tps"]["improvement"] == pytest.approx(1.0)
    assert deltas["step_wall_ms"]["orientation"] == "lower is better"
    assert deltas["per_request_step_tps"]["orientation"] == "higher is better"


# --------------------------------------------------------------------------
# the per-width placebo: the design's strongest causal check
# --------------------------------------------------------------------------
def _bc(by_width: dict[int, tuple[int, float]]) -> dict:
    return {
        "available": True,
        "by_width": {
            str(w): {"steps": n, "mean_ms": ms, "fraction": 0.0, "sd_ms": 0.0}
            for w, (n, ms) in by_width.items()
        },
    }


def test_placebo_is_clean_when_the_gain_is_localised_to_the_treated_width() -> None:
    module = _reducer()
    stock = _bc({2: (100, 285.0), 3: (1000, 362.0), 4: (3000, 411.0)})
    cand = _bc({2: (100, 302.0), 3: (1000, 370.0), 4: (3000, 382.0)})
    p = module.per_width_placebo(stock, cand, (4,))
    assert p["treated_width"] == 4
    assert p["treated_widths"] == [4]
    roles = {r["width"]: r["role"] for r in p["rows"]}
    assert roles == {2: "placebo", 3: "placebo", 4: "treated"}
    did = p["difference_in_differences"]
    assert did["control_width"] == 3
    # control shows candidate SLOWER -> confound works against the candidate
    assert did["control_delta_ms"] > 0
    assert "UNDERSTATES" in did["confound_direction"]
    # raw -29, control +8 -> additive -37
    assert did["additive_effect_ms"] == pytest.approx(-37.0, abs=0.5)


def test_placebo_detects_a_leak_when_an_untreated_width_also_gains() -> None:
    """The failure this check exists to catch: an arm-wide speedup."""
    module = _reducer()
    stock = _bc({3: (1000, 362.0), 4: (3000, 411.0)})
    cand = _bc({3: (1000, 342.0), 4: (3000, 391.0)})  # both widths 20ms faster
    p = module.per_width_placebo(stock, cand, (4,))
    did = p["difference_in_differences"]
    # an arm-wide shift must leave ~zero after differencing
    assert did["additive_effect_ms"] == pytest.approx(0.0, abs=0.5)
    assert "OVERSTATES" in did["confound_direction"]


def test_width_scaling_cost_is_reported_against_the_nsys_attribution() -> None:
    module = _reducer()
    stock = _bc({3: (1000, 362.08), 4: (3000, 411.26)})
    cand = _bc({3: (1000, 369.79), 4: (3000, 381.75)})
    ws = module.per_width_placebo(stock, cand, (4,))["width_scaling_cost"]
    assert ws["stock_ms"] == pytest.approx(49.18, abs=0.05)
    assert ws["nsys_attributed_fa2_width4_cost_ms"] == 48.4
    assert ws["removed_ms"] == pytest.approx(37.21, abs=0.05)


def test_a_thin_control_width_is_refused_as_a_did_basis() -> None:
    """One sampled step is not a control; it must not drive the correction."""
    module = _reducer()
    stock = _bc({1: (1, 272.0), 4: (3000, 411.0)})
    cand = _bc({1: (6, 253.0), 4: (3000, 382.0)})
    p = module.per_width_placebo(stock, cand, (4,))
    assert p["difference_in_differences"]["available"] is False


# --------------------------------------------------------------------------
# the pool16 identity must be ONE string everywhere it is written down
# --------------------------------------------------------------------------
LAUNCHER = REPO / "scripts/fr13_launch_forked_fa2_tree_server.sh"
PATCHER = REPO / "scripts/fr13_patch_fa2_tree_bias.py"
SUBSET16 = REPO / "config/fr13_fixed32/subset_b4_sixteen.json"


def _subset16_csv() -> str:
    return ",".join(json.loads(SUBSET16.read_text(encoding="ascii"))["instance_ids"])


def _patcher_pool16_csv() -> str:
    spec = importlib.util.spec_from_file_location("patcher_for_pool16", PATCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import os as _os

    namespace: dict[str, object] = {"os": _os}
    exec(  # noqa: S102 - repo source, executed on purpose
        compile(
            module.FIXED32_QUERY_TILE32_B4_PRODUCTION_HELPERS, "<helpers>", "exec"
        ),
        namespace,
    )
    return ",".join(namespace["_FR13_FA2_QROW32_B4_POOL16_TASK_IDS"])


def _launcher_pool16_csv() -> str:
    text = LAUNCHER.read_text(encoding="utf-8")
    marker = f"|{POOL16_SHA}"
    line = next(l for l in text.splitlines() if marker in l)
    return line.split('"')[1].split("|")[0]


def test_the_pool16_task_list_is_identical_in_all_four_places() -> None:
    """Runner, subset file, shell launcher predicate, in-container patcher.

    These are four INDEPENDENT hardcodings of the same ordered list. Any drift
    between them fails the arm at boot -- after the stock arm has already burned
    hours -- so it is checked here, on CPU, instead of live.
    """
    from_file = _subset16_csv()
    runner_line = next(
        l for l in RUNNER.read_text(encoding="utf-8").splitlines()
        if l.startswith("TASK_IDS=")
    )
    from_runner = runner_line.split("=", 1)[1]
    assert from_runner == from_file, "runner TASK_IDS drifted from the subset file"
    assert _launcher_pool16_csv() == from_file, "launcher case drifted"
    assert _patcher_pool16_csv() == from_file, "patcher constant drifted"
    assert len(from_file.split(",")) == 16


def _run_launcher_predicate(ids: str, sha: str) -> str:
    """Execute the launcher's own case block in isolation and report the verdict."""
    text = LAUNCHER.read_text(encoding="utf-8")
    start = text.index("  _FR13_FA2_QROW32_B4_TASK_SET=\n")
    end = text.index("  esac\n", start) + len("  esac\n")
    block = text[start:end]
    script = (
        f"FR13_FA2_QROW32_B4_EXACT4_TASK_IDS={ids!r}\n"
        f"FR13_FA2_QROW32_B4_EXACT4_SUBSET_SHA256={sha!r}\n"
        + block
        + 'printf "%s" "${_FR13_FA2_QROW32_B4_TASK_SET:-REFUSED}"\n'
    ).replace("'", '"')
    proc = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    )
    return proc.stdout.strip() or "REFUSED"


def test_launcher_predicate_admits_both_pinned_sets() -> None:
    exact4 = (
        "astropy__astropy-12907,astropy__astropy-13033,"
        "astropy__astropy-13236,astropy__astropy-13398"
    )
    assert _run_launcher_predicate(exact4, EXACT4_SHA) == "exact4"
    assert _run_launcher_predicate(_subset16_csv(), POOL16_SHA) == "pool16"


@pytest.mark.parametrize(
    "ids,sha",
    [
        (
            "astropy__astropy-12907,astropy__astropy-13033,"
            "astropy__astropy-13236,astropy__astropy-13398",
            POOL16_SHA,
        ),
        (None, EXACT4_SHA),
        ("", EXACT4_SHA),
        ("astropy__astropy-12907", EXACT4_SHA),
    ],
)
def test_launcher_predicate_refuses_unpinned_or_mismatched_sets(ids, sha) -> None:
    task_ids = _subset16_csv() if ids is None else ids
    assert _run_launcher_predicate(task_ids, sha) == "REFUSED"


def test_batch_conditioned_analysis_is_reusable_with_an_explicit_arm_name() -> None:
    """The pool16 campaign's <mode>_pool<N> naming must not be assumed."""
    spec = importlib.util.spec_from_file_location(
        "bcw_for_test", REPO / "scripts/fr13_b4_batch_conditioned_wall.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import inspect

    params = inspect.signature(module.analyse_arm).parameters
    assert "arm_name" in params
    assert params["arm_name"].default is None, "defaulting keeps old callers intact"

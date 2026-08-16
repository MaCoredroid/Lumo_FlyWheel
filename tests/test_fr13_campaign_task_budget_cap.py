"""The FR13 campaign per-task budget cap: bound, terminal, and accounting.

WHY THIS EXISTS -- THE MECHANISM, RECONCILED.

astropy__astropy-13398 produces runaway trajectories: ~1 draw in 20 runs for
three hours, and two of the banked ones died to qwen-code's always-on loop
detector. The two trip counts looked irreconcilable -- one tripped at 284 tool
calls while a banked SUCCESS ran to 454 -- so the mechanism was read out of the
traces directly rather than assumed:

  2026-07-31 quarantined trip   top-level 100 (cap 100)  sub-agent  29  total 129
  2026-08-13 tail23 trip        top-level 256 (cap 256)  sub-agent  28  total 284
  2026-08-10 tail23 SUCCESS     top-level  60            sub-agent 394  total 454

TURN_TOOL_CALL_CAP (scripts/fr13_derive_qwen_agent_bundle_cap256.py:47, patched
100 -> 256 in the vendored bundle) counts TOP-LEVEL session tool calls within a
turn. Blocks carrying a non-null ``parent_tool_use_id`` belong to a sub-agent
session, which is counted independently and never trips the parent's guard. Both
trips land EXACTLY on their era's cap; the 454-call success survived only
because 394 of those calls were delegated to one sub-agent, leaving 60 at top
level. So 284 vs 454 was never a contradiction -- 284 is a red herring and the
counted quantity is the 256.

The consequence is the thing that matters: raising the cap did not make the task
safe, it moved the wall, and whether a draw survives is a sampling accident (did
the model happen to delegate?) rather than a bound. The 08-13 trip burned 9324 s
and voided an entire gate arm at campaign finalize.

WHAT THE CAP DOES ABOUT IT. Bounds the wall in the harness, deterministically,
and gives the resulting terminal a name. The tests below hold three lines:

  1. OFF BY DEFAULT AND OFF MEANS OFF. Quality/QC arms must observe the agent
     uncapped, so an arm that does not ask for a cap must be byte-identical to
     before.
  2. THE TERMINAL IS ACCOUNTED, NOT DISGUISED. A capped task is not `resolved`,
     not `failed`, and specifically not the synthetic no-patch terminal --
     `empty_patch` is a model-side outcome and a cap is a harness-side one.
  3. THE ACCOUNTING RECONCILES FROM BOTH SIDES. The abort the kill produces on
     the engine is legal only against a declaration derived from the per-task
     runner records, and must equal it exactly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import fr13_fixed32_contract as contract  # noqa: E402


def _runner_module():
    """Import run_swe_bench_q36_a without executing its CLI."""
    spec = importlib.util.spec_from_file_location(
        "fr13_run_swe_bench_q36_a_under_test",
        SCRIPTS / "run_swe_bench_q36_a.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _runner_module()


# --------------------------------------------------------------------------- #
# 1. the budget is read fail-closed, and OFF means OFF                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw", ["", "   ", "0", "0.0"])
def test_unset_or_zero_is_the_cap_off(monkeypatch, raw: str) -> None:
    monkeypatch.setenv(runner.CAMPAIGN_TASK_BUDGET_ENV, raw)
    assert runner._campaign_task_budget_s() == 0.0


def test_absent_env_is_the_cap_off(monkeypatch) -> None:
    monkeypatch.delenv(runner.CAMPAIGN_TASK_BUDGET_ENV, raising=False)
    assert runner._campaign_task_budget_s() == 0.0


@pytest.mark.parametrize("raw", ["abc", "-1", "-5400", "nan", "inf", "1e400"])
def test_an_unreadable_budget_refuses(monkeypatch, raw: str) -> None:
    """A budget nobody can read is a budget that silently is not there."""
    monkeypatch.setenv(runner.CAMPAIGN_TASK_BUDGET_ENV, raw)
    with pytest.raises(runner.Fixed32BoundaryError):
        runner._campaign_task_budget_s()


@pytest.mark.parametrize("raw", ["1", "60", "600", "1799"])
def test_a_budget_below_the_floor_refuses(monkeypatch, raw: str) -> None:
    """`--agent-wall-s 0` already taught this the expensive way.

    A too-short budget would cap every task in the campaign and produce an arm
    of nothing but capped terminals -- a plausible-looking result that is
    entirely an artefact of a typo.
    """
    monkeypatch.setenv(runner.CAMPAIGN_TASK_BUDGET_ENV, raw)
    with pytest.raises(runner.Fixed32BoundaryError, match="floor"):
        runner._campaign_task_budget_s()


def test_the_recommended_budget_is_accepted_and_above_the_floor(monkeypatch) -> None:
    assert (
        runner.CAMPAIGN_TASK_BUDGET_RECOMMENDED_S
        > runner.CAMPAIGN_TASK_BUDGET_MIN_S
    )
    monkeypatch.setenv(
        runner.CAMPAIGN_TASK_BUDGET_ENV,
        str(runner.CAMPAIGN_TASK_BUDGET_RECOMMENDED_S),
    )
    assert (
        runner._campaign_task_budget_s()
        == float(runner.CAMPAIGN_TASK_BUDGET_RECOMMENDED_S)
    )


@pytest.mark.parametrize(
    ("wall", "budget", "expected"),
    [
        (0, 5400.0, 5400),        # no legacy wall -> the budget IS the limit
        (1800, 5400.0, 1800),     # a tighter legacy wall still wins
        (9000, 5400.0, 5400),     # a looser legacy wall does not
        (5400, 5400.0, 5400),     # equal -> the budget is binding
    ],
)
def test_the_tighter_of_wall_and_budget_is_what_runs(
    wall: int, budget: float, expected: int
) -> None:
    assert runner._campaign_budget_effective_timeout_s(wall, budget) == expected


def test_cap_off_leaves_the_agent_record_untouched() -> None:
    meta = {"elapsed_s": 9324.0, "timed_out": False, "exit_code": 0}
    out = runner._attribute_campaign_budget_cap(
        meta, budget_s=0.0, requested_wall_s=0
    )
    assert out["budget_capped"] is False
    assert out["campaign_budget_s"] is None
    # every measured field survives verbatim
    for key, value in meta.items():
        assert out[key] == value


# --------------------------------------------------------------------------- #
# 2. attribution names the terminal without rewriting measured evidence        #
# --------------------------------------------------------------------------- #
def _capped_meta(**overrides: Any) -> dict[str, Any]:
    meta = {
        "elapsed_s": 5400.0,
        "timed_out": True,
        "exit_code": -1,
        "offloaded": True,
        "network_drop": False,
        "stall_killed": False,
    }
    meta.update(overrides)
    return runner._attribute_campaign_budget_cap(
        meta,
        budget_s=overrides.pop("_budget_s", 5400.0),
        requested_wall_s=overrides.pop("_wall_s", 0),
    )


def test_a_real_cap_is_attributed_and_timed_out_is_left_as_measured() -> None:
    meta = _capped_meta()
    assert meta["budget_capped"] is True
    assert meta["campaign_budget_s"] == 5400.0
    assert meta["campaign_budget_was_the_binding_limit"] is True
    # timed_out is a MEASURED flag. The attribution sits beside it; it does not
    # overwrite it, so the classification stays recomputable from the record.
    assert meta["timed_out"] is True


def test_a_legacy_wall_timeout_is_not_a_cap() -> None:
    """The legacy per-attempt wall fired, not the budget. Still fatal."""
    meta = runner._attribute_campaign_budget_cap(
        {"elapsed_s": 1800.0, "timed_out": True, "exit_code": -1},
        budget_s=5400.0,
        requested_wall_s=1800,
    )
    assert meta["campaign_budget_was_the_binding_limit"] is False
    assert meta["budget_capped"] is False


def test_a_clean_finish_under_the_budget_is_not_a_cap() -> None:
    meta = runner._attribute_campaign_budget_cap(
        {"elapsed_s": 2836.0, "timed_out": False, "exit_code": 0},
        budget_s=5400.0,
        requested_wall_s=0,
    )
    assert meta["budget_capped"] is False


def test_a_kill_that_did_not_reach_the_budget_is_not_a_cap() -> None:
    """Corroboration is the measured wall, not the flag."""
    meta = runner._attribute_campaign_budget_cap(
        {"elapsed_s": 12.0, "timed_out": True, "exit_code": -1},
        budget_s=5400.0,
        requested_wall_s=0,
    )
    assert meta["budget_capped"] is False


# --------------------------------------------------------------------------- #
# 3. the terminal is its own class                                             #
# --------------------------------------------------------------------------- #
def _capped_report(patch_text: str = "", **overrides: Any):
    meta = _capped_meta()
    meta.update(overrides)
    return runner._synthetic_budget_capped_eval_report(
        instance_id="astropy__astropy-13398",
        dataset_name="princeton-nlp/SWE-bench_Verified",
        model_name="qwen3.8-27b-nvfp4",
        agent_meta=meta,
        patch_text=patch_text,
    )


def test_the_capped_terminal_is_neither_resolved_nor_failed() -> None:
    report = _capped_report()
    assert report is not None
    assert report["verdict"] == runner.BUDGET_CAPPED_VERDICT == "capped"
    assert report["verdict"] not in ("resolved", "failed", "crash")
    assert report["passed"] is False
    assert report["failure_mode"] == runner.BUDGET_CAPPED_FAILURE_MODE
    assert report["schema"] == runner.SYNTHETIC_BUDGET_CAPPED_EVAL_SCHEMA


def test_the_capped_terminal_is_not_the_no_patch_terminal() -> None:
    """`empty_patch` is a MODEL-side outcome; a cap is a HARNESS-side one.

    Collapsing them would move a harness decision into the model's column, which
    is exactly the masquerade the completion algebra exists to prevent.
    """
    report = _capped_report(patch_text="")
    assert report is not None
    assert report["schema"] != runner.SYNTHETIC_NO_PATCH_EVAL_SCHEMA
    assert report["error"] != "empty_patch"
    assert report.get("synthetic_no_patch") is not True


def test_the_capped_terminal_states_the_harness_never_ran() -> None:
    report = _capped_report(patch_text="diff --git a/x b/x\n")
    assert report is not None
    assert report["harness_invoked"] is False
    assert report["harness_exit_code"] is None
    # it says which patch it is DECLINING to evaluate, so the choice is auditable
    assert report["patch_bytes_not_evaluated"] == len("diff --git a/x b/x\n")


def test_the_capped_terminal_carries_the_numbers_behind_it() -> None:
    report = _capped_report()
    assert report is not None
    assert report["campaign_budget_s"] == 5400.0
    assert report["agent_elapsed_s"] == 5400.0
    assert report["agent_terminal"]["timed_out"] is True


@pytest.mark.parametrize(
    "broken",
    [
        {"budget_capped": False},
        {"campaign_budget_s": None},
        {"campaign_budget_s": 0},
        {"elapsed_s": 10.0},
        {"timed_out": False},
        {"campaign_budget_was_the_binding_limit": False},
    ],
)
def test_an_uncorroborated_cap_claim_gets_no_terminal(broken: dict) -> None:
    """A terminal nobody can recompute is not evidence."""
    assert _capped_report(**broken) is None


def test_the_normal_path_is_untouched_when_nothing_was_capped() -> None:
    assert (
        runner._synthetic_budget_capped_eval_report(
            instance_id="astropy__astropy-12907",
            dataset_name="princeton-nlp/SWE-bench_Verified",
            model_name="qwen3.8-27b-nvfp4",
            agent_meta={
                "elapsed_s": 598.1,
                "timed_out": False,
                "exit_code": 0,
                "budget_capped": False,
                "campaign_budget_s": None,
            },
            patch_text="",
        )
        is None
    )


# --------------------------------------------------------------------------- #
# 4. the fixed32 per-task provenance gate                                      #
# --------------------------------------------------------------------------- #
def test_the_provenance_gate_still_refuses_an_uncapped_timeout() -> None:
    source = (SCRIPTS / "run_swe_bench_q36_a.py").read_text(encoding="utf-8")
    # the original refusal is intact, now guarded by the corroborated cap
    assert (
        "if not budget_capped_corroborated and "
        '(exit_code != 0 or agent_meta["timed_out"]):' in source
    )
    assert "agent terminal state is incomplete" in source
    # a cap CLAIM that does not corroborate is its own, louder refusal
    assert (
        "agent claims a campaign budget cap but the record does not "
        in source
    )


def test_the_gate_requires_every_corroborating_term() -> None:
    source = (SCRIPTS / "run_swe_bench_q36_a.py").read_text(encoding="utf-8")
    for term in (
        'agent_meta["timed_out"] is True',
        'agent_meta.get("campaign_budget_was_the_binding_limit") is True',
        "float(elapsed_s) >= float(budget_s)",
    ):
        assert term in source


# --------------------------------------------------------------------------- #
# 5. the engine-side abort is an accounted class, not a tolerance              #
# --------------------------------------------------------------------------- #
def _deltas(**overrides: int) -> dict[str, int]:
    base = {
        "max_tokens_count": 390,
        "max_tokens_le_inf": 390,
        "max_tokens_le_50000": 390,
        "request_success_stop": 389,
        "request_success_length": 1,
        "request_success_abort": 0,
        "request_success_error": 0,
        "request_success_repetition": 0,
    }
    base.update(overrides)
    return base


def test_no_declared_cap_pins_abort_at_zero_exactly_as_before() -> None:
    counts = contract._fixed32_qwen_completion_classes(
        _deltas(), completed=390, scope="campaign"
    )
    assert counts == {"stop": 389, "length": 1, "abort": 0}


def test_an_undeclared_abort_is_refused_and_names_both_numbers() -> None:
    with pytest.raises(contract.ContractError) as excinfo:
        contract._fixed32_qwen_completion_classes(
            _deltas(request_success_abort=1, max_tokens_count=391,
                    max_tokens_le_inf=391, max_tokens_le_50000=391),
            completed=390,
            scope="campaign",
        )
    message = str(excinfo.value)
    assert "abort=1" in message and "capped_requests=0" in message


def test_a_declared_cap_makes_exactly_that_many_aborts_legal() -> None:
    counts = contract._fixed32_qwen_completion_classes(
        _deltas(
            request_success_abort=2,
            max_tokens_count=392,
            max_tokens_le_inf=392,
            max_tokens_le_50000=392,
        ),
        completed=390,
        scope="campaign",
        capped_requests=2,
    )
    assert counts["abort"] == 2
    # the completion identity is unchanged: an abort is not a completion
    assert counts["stop"] + counts["length"] == 390


@pytest.mark.parametrize("aborts", [0, 1, 3])
def test_the_abort_count_must_equal_the_declaration_exactly(aborts: int) -> None:
    """Two capped tasks means two aborts. Not "at most", not "about"."""
    with pytest.raises(contract.ContractError, match="capped_requests=2"):
        contract._fixed32_qwen_completion_classes(
            _deltas(
                request_success_abort=aborts,
                max_tokens_count=392,
                max_tokens_le_inf=392,
                max_tokens_le_50000=392,
            ),
            completed=390,
            scope="campaign",
            capped_requests=2,
        )


def test_a_cap_does_not_excuse_a_broken_histogram() -> None:
    with pytest.raises(contract.ContractError) as excinfo:
        contract._fixed32_qwen_completion_classes(
            _deltas(
                request_success_abort=1,
                max_tokens_count=390,       # should be 391
                max_tokens_le_inf=391,
                max_tokens_le_50000=391,
            ),
            completed=390,
            scope="campaign",
            capped_requests=1,
        )
    message = str(excinfo.value)
    assert "max_tokens_count=390" in message
    assert "expected 391" in message
    assert "completed 390 + capped 1" in message


@pytest.mark.parametrize("reason", ["error", "repetition"])
def test_a_cap_never_legalises_a_defect(reason: str) -> None:
    """error/repetition are pinned at zero, capped campaign or not."""
    with pytest.raises(contract.ContractError, match="forbidden"):
        contract._fixed32_qwen_completion_classes(
            _deltas(
                request_success_abort=1,
                max_tokens_count=391,
                max_tokens_le_inf=391,
                max_tokens_le_50000=391,
                **{f"request_success_{reason}": 1},
            ),
            completed=390,
            scope="campaign",
            capped_requests=1,
        )


@pytest.mark.parametrize("bad", [-1, True, 1.0, "1"])
def test_the_declaration_itself_is_type_checked(bad: Any) -> None:
    with pytest.raises(contract.ContractError, match="capped request count"):
        contract._fixed32_qwen_completion_classes(
            _deltas(), completed=390, scope="campaign", capped_requests=bad
        )


def test_the_capped_class_is_disjoint_from_the_others() -> None:
    assert contract.QWEN_CAPPED_COMPLETION_REASON == "abort"
    assert (
        contract.QWEN_CAPPED_COMPLETION_REASON
        not in contract.QWEN_TERMINAL_COMPLETION_REASONS
    )
    assert (
        contract.QWEN_CAPPED_COMPLETION_REASON
        not in contract.QWEN_FORBIDDEN_COMPLETION_REASONS
    )
    assert set(contract.QWEN_FORBIDDEN_COMPLETION_REASONS) == {
        "error",
        "repetition",
    }


# --------------------------------------------------------------------------- #
# 6. the declaration is required, and reaches every consumer                   #
# --------------------------------------------------------------------------- #
def test_the_campaign_task_input_requires_the_declaration() -> None:
    """Not optional: a caller that forgets it must fail loud rather than
    silently declare zero caps and turn a real abort into an unexplainable
    refusal."""
    source = (SCRIPTS / "fr13_fixed32_contract.py").read_text(encoding="utf-8")
    block = source[source.index("expected_task_keys = {"):]
    block = block[: block.index("}")]
    assert '"budget_capped"' in block


def test_the_declaration_comes_from_the_task_s_own_runner_record() -> None:
    source = (SCRIPTS / "run_swe_bench_q36_a.py").read_text(encoding="utf-8")
    assert (
        '(summary.get("agent") or summary.get("codex") or {}).get(\n'
        '                        "budget_capped"\n'
        "                    )\n"
        "                    is True" in source
    )
    # and is sealed into the campaign proof so an offline replay declares the
    # same caps the live campaign did
    assert '"budget_capped": record["budget_capped"],' in source


def test_banked_proofs_without_the_field_still_replay() -> None:
    """Every proof sealed before 2026-08-13 declared no caps by construction."""
    source = (SCRIPTS / "fr13_floor_gate.py").read_text(encoding="utf-8")
    assert "legacy_keys = {" in source
    assert "capped_aware_keys = legacy_keys | {\"budget_capped\"}" in source
    assert "elif set(task) == legacy_keys:\n            budget_capped = False" in source


def test_the_campaign_summary_counts_capped_tasks_by_name() -> None:
    source = (SCRIPTS / "run_swe_bench_q36_a.py").read_text(encoding="utf-8")
    assert '"budget_capped_tasks": len(capped_task_ids),' in source
    assert '"budget_capped_task_ids": sorted(capped_task_ids),' in source
    # and refuses to publish a count the verdicts do not back
    assert "campaign budget-cap accounting does not reconcile for" in source


def test_the_metric_evidence_publishes_the_capped_terms() -> None:
    source = (SCRIPTS / "fr13_fixed32_contract.py").read_text(encoding="utf-8")
    for key in (
        '"request_success_abort"',
        '"budget_capped_tasks"',
        '"budget_capped_task_ids"',
        '"budget_capped_visible_requests"',
        '"budget_capped_compaction_requests"',
    ):
        assert key in source


# --------------------------------------------------------------------------- #
# 7. per-campaign configuration, default OFF                                   #
# --------------------------------------------------------------------------- #
def test_the_driver_exposes_the_budget_and_defaults_it_off() -> None:
    driver = (SCRIPTS / "fr13_b4_campaign_driver.sh").read_text(encoding="utf-8")
    assert "FR13_CAMPAIGN_TASK_BUDGET_S=${FR13_CAMPAIGN_TASK_BUDGET_S:-}" in driver
    assert "export FR13_CAMPAIGN_TASK_BUDGET_S" in driver
    # the OFF-for-QC / ON-for-timing rule is written down where it is set
    assert "exact16 QC" in driver


def test_the_cap_is_not_baked_into_the_canonical_registry() -> None:
    """A campaign lever, not a shipped default.

    The canonical registry is for values the branch SHIPS. A cap changes what
    the agent is allowed to do, so it must be asked for per campaign and never
    inherited.
    """
    registry = (SCRIPTS / "fr13_canonical_env.sh").read_text(encoding="utf-8")
    assert "FR13_CAMPAIGN_TASK_BUDGET_S" not in registry


def test_the_single_choke_point_covers_every_agent_runner() -> None:
    """Four runner bodies, one place the budget is applied."""
    source = (SCRIPTS / "run_swe_bench_q36_a.py").read_text(encoding="utf-8")
    dispatch = source[source.index("def _run_agent_dispatch("):]
    dispatch = dispatch[: dispatch.index("\ndef ", 10)]
    assert "budget_s = _campaign_task_budget_s()" in dispatch
    assert "_campaign_budget_effective_timeout_s(" in dispatch
    assert "_attribute_campaign_budget_cap(" in dispatch
    # both routes go through the attribution
    assert "_run_agent_remote(host=AGENT_HOST, **kwargs)" in dispatch
    assert "_run_agent_local(**kwargs)" in dispatch


def test_a_capped_task_gets_no_second_budget() -> None:
    """Re-driving a capped task would hand it another full budget."""
    source = (SCRIPTS / "run_swe_bench_q36_a.py").read_text(encoding="utf-8")
    assert "if not budget_capped and not patch_text.strip():" in source
    assert '"cause": "campaign_budget_capped",' in source

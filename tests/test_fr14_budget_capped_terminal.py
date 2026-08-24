"""SITE 26: the BUDGET-CAPPED TERMINAL class.

THE DEFECT. Two individually-correct rules were jointly unsatisfiable on any
capped task, which is the site-18 shape at task level:

  * the campaign budget cap (``_attribute_campaign_budget_cap``) exists to name
    a capped terminal DELIBERATE, DECLARED and LEGAL -- it is the one incomplete
    terminal this campaign accepts;
  * the compaction-metric evidence requirement refuses any trace that carries
    metric evidence and no Qwen ``result`` record.

A cap kills the agent where it stands, so qwen-code never writes that result.
No capped task had a legal terminal, and the QC campaign died at 2/12 on
astropy__astropy-13579 after a ~152-minute run.

THE FIX under test: the compaction-evidence requirement SCOPES TO THE COMPLETED
PORTION when the terminal is a corroborated cap. The count comes from the metric
brackets and the trace, checked against each other and against the proxy's
per-task-key ledger. The rule keeps its teeth everywhere else.

The three ruled mutation proofs are
``test_uncapped_terminal_without_a_result_still_refuses``,
``test_capped_terminal_with_reconciling_brackets_passes`` and
``test_capped_terminal_with_broken_brackets_refuses``.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
TESTS = Path(__file__).resolve().parent
for entry in (str(SCRIPTS), str(TESTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import fr13_fixed32_contract as contract  # noqa: E402
import test_fr13_fixed32_trace_provenance as base  # noqa: E402


TASK_A = base.TASK_A
SESSION_A = contract.fixed32_trace_session_id(TASK_A)


@pytest.fixture()
def task_dir() -> Any:
    """A scratch directory that does not depend on pytest's tmp_path root.

    The shared host's ``/tmp/pytest-of-mark`` is a symlink, which pytest's own
    tmp_path factory refuses. These proofs must actually run here, so they make
    their own directory.
    """
    path = Path(tempfile.mkdtemp(prefix="fr14-capped-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _load_runner() -> Any:
    path = SCRIPTS / "run_swe_bench_q36_a.py"
    spec = importlib.util.spec_from_file_location("fr14_capped_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capped_trace(
    *,
    turns: int = 12,
    partial_final_group: bool = False,
    instance_id: str = TASK_A,
) -> list[dict[str, Any]]:
    """A trace the cap cut, in the two shapes a kill can leave behind.

    ``partial_final_group=False`` is the kill BETWEEN requests -- every response
    group terminated and the trace ends on a tool result. That is the shape the
    real astropy__astropy-13579 trace has (79 groups, last event a ``user``).

    ``partial_final_group=True`` is the kill MID-RESPONSE -- the last group never
    reached a terminal record, so the engine aborted that request.
    """
    session_id = contract.fixed32_trace_session_id(instance_id)
    events: list[dict[str, Any]] = [
        base._context_event(
            event_type="system",
            event_id="system",
            session_id=session_id,
        )
    ]
    for index in range(turns):
        events.append(
            base._assistant_event(
                response_id=f"tool-turn-{index}",
                session_id=session_id,
                content=[
                    {
                        "type": "tool_use",
                        "id": f"tool-call-{index}",
                        "name": "read_file",
                        "input": {},
                    }
                ],
                stop_reason="tool_use",
            )
        )
        events.append(
            base._context_event(
                event_type="user",
                event_id=f"tool-result-{index}",
                session_id=session_id,
            )
        )
    if partial_final_group:
        events.append(
            base._assistant_event(
                response_id="killed-mid-response",
                session_id=session_id,
                content=[{"type": "thinking", "thinking": "cut off"}],
                stop_reason=None,
            )
        )
    return events


def _capped_metrics(
    *,
    completed: int = 12,
    normal_requests: int = 12,
    compactions: int = 0,
    aborted: int = 0,
    prompt_tokens: int = 100_000,
    generation_tokens: int = 40_000,
    overrides: dict[str, int] | None = None,
) -> tuple[bytes, bytes]:
    """Brackets for a capped task.

    A capped abort never reaches the agent, but vLLM still FINISHED it and still
    histogrammed its max_tokens -- so the histogram counts ``completed +
    aborted`` while the completion classes count ``completed``.
    """
    histogram = completed + aborted
    merged = {
        "max_tokens_count": histogram,
        "max_tokens_le_50000": histogram,
        "max_tokens_le_inf": histogram,
        "request_success_abort": aborted,
    }
    merged.update(overrides or {})
    return base._qwen_compaction_metrics(
        completed=completed,
        compactions=compactions,
        normal_requests=normal_requests + aborted,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        overrides=merged,
    )


def _uncapped_result_case() -> tuple[list[dict[str, Any]], tuple[bytes, bytes]]:
    """An ordinary completed Qwen run with a task-scoped metric bracket.

    The result's self-reported aggregate must equal the engine's bracket to the
    token -- that identity is the check a capped terminal cannot make, and it
    stays exactly as strict here.
    """
    events = base._qwen_result_trace()
    events[-1]["usage"] = {
        "input_tokens": 5_000,
        "output_tokens": 500,
        "total_tokens": 5_500,
    }
    metrics = base._qwen_compaction_metrics(
        completed=13,
        compactions=0,
        normal_requests=13,
        prompt_tokens=5_000,
        generation_tokens=500,
    )
    return events, metrics


def _validate(
    events: list[dict[str, Any]],
    metrics: tuple[bytes, bytes],
    *,
    completed: int = 12,
    budget_capped_terminal: bool = True,
    expected_session_id: str | None = SESSION_A,
) -> dict[str, Any]:
    return contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=expected_session_id,
        expected_completed_logical_model_requests=completed,
        metrics_pre=metrics[0],
        metrics_post=metrics[1],
        budget_capped_terminal=budget_capped_terminal,
    )


# --------------------------------------------------------------------------- #
# MUTATION PROOF (a): the rule keeps its teeth                                 #
# --------------------------------------------------------------------------- #
def test_uncapped_terminal_without_a_result_still_refuses() -> None:
    """No declaration, no relaxation. This is the refusal, unchanged."""
    events = _capped_trace()
    metrics = _capped_metrics()
    with pytest.raises(contract.ContractError) as error:
        _validate(events, metrics, budget_capped_terminal=False)
    assert "compaction metric evidence requires a Qwen result" in str(error.value)


def test_uncapped_terminal_without_a_result_refuses_by_default() -> None:
    """A caller that never heard of the class gets the old behaviour."""
    events = _capped_trace()
    metrics = _capped_metrics()
    with pytest.raises(contract.ContractError) as error:
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=SESSION_A,
            expected_completed_logical_model_requests=12,
            metrics_pre=metrics[0],
            metrics_post=metrics[1],
        )
    assert "compaction metric evidence requires a Qwen result" in str(error.value)


def test_capped_declaration_must_be_a_bool() -> None:
    """A truthy string must never buy the relaxation."""
    with pytest.raises(contract.ContractError) as error:
        contract.validate_fixed32_trace_model_requests(
            _capped_trace(),
            expected_session_id=SESSION_A,
            budget_capped_terminal="yes",  # type: ignore[arg-type]
        )
    assert "budget-capped terminal declaration must be a bool" in str(error.value)


def test_capped_declaration_does_not_relax_a_trace_that_carries_a_result() -> None:
    """A run that finished and was cut in teardown is an ORDINARY result.

    The declaration must not become a skeleton key for the result checks.
    """
    events = base._qwen_result_trace()
    events[-1]["num_turns"] = 99
    with pytest.raises(contract.ContractError) as error:
        contract.validate_fixed32_trace_model_requests(
            events,
            expected_session_id=SESSION_A,
            budget_capped_terminal=True,
        )
    assert "turn count and top-level response groups" in str(error.value)


def test_capped_terminal_requires_the_task_session_identity() -> None:
    """The identity the missing result carried is demanded of the trace."""
    events = _capped_trace()
    metrics = _capped_metrics()
    with pytest.raises(contract.ContractError) as error:
        _validate(events, metrics, expected_session_id=None)
    assert "requires the task session identity" in str(error.value)

    drifted = _capped_trace()
    drifted[5]["session_id"] = "some-other-session"
    with pytest.raises(contract.ContractError) as drift_error:
        _validate(drifted, metrics)
    assert "session does not bind to the task" in str(drift_error.value)


# --------------------------------------------------------------------------- #
# MUTATION PROOF (b): a capped task with reconciling brackets passes           #
# --------------------------------------------------------------------------- #
def test_capped_terminal_with_reconciling_brackets_passes() -> None:
    events = _capped_trace()
    result = _validate(events, _capped_metrics())
    assert result["trace_format"] == "qwen_budget_capped_terminal"
    assert result["budget_capped_terminal"] is True
    assert result["budget_capped_partial_response_group"] is False
    assert result["budget_capped_aborted_logical_requests"] == 0
    assert result["completed_logical_model_requests"] == 12
    assert len(set(result["model_request_ids"])) == 12
    evidence = result["qwen_compaction_metric_evidence"]
    assert evidence["normal_requests"] == 12
    assert evidence["completed_engine_requests"] == 12
    assert (
        evidence["normal_visible_max_output_tokens"]
        in contract.FIXED32_DEPLOYED_MAX_OUTPUT_TOKENS
    )


def test_capped_terminal_counts_its_compactions_from_the_bracket() -> None:
    """The completed portion still owes its full compaction accounting.

    One compaction is visible in the trace as a top-level input-token drop; the
    second is only in the bracket, and it gets a synthetic identity anchored to
    the last event the kill left behind.
    """
    events = _capped_trace()
    base._set_top_level_group_input_tokens(
        events,
        [100 * index for index in range(1, 12)] + [500],
    )
    base._bind_top_level_tool_result(events, next_group_index=11)
    metrics = _capped_metrics(
        completed=14,
        normal_requests=12,
        compactions=2,
        prompt_tokens=20_000,
        generation_tokens=5_000,
    )
    result = _validate(events, metrics, completed=14)
    assert result["completed_logical_model_requests"] == 14
    assert result["hidden_successful_compaction_model_requests"] == 1
    assert result["hidden_failed_compaction_model_requests"] == 1
    capped_ids = [
        request_id
        for request_id in result["model_request_ids"]
        if request_id.startswith("qwen-capped-failed-compaction-sha256:")
    ]
    assert len(capped_ids) == 1
    # The result-anchored namespace stays untouched, so banked identities keep
    # hashing to exactly what they always did.
    assert not any(
        request_id.startswith("qwen-hidden-failed-compaction-sha256:")
        for request_id in result["model_request_ids"]
    )


def test_capped_kill_mid_response_is_counted_as_one_abort() -> None:
    """An unterminated final group is an ABORT, not a completed request."""
    events = _capped_trace(partial_final_group=True)
    metrics = _capped_metrics(completed=12, normal_requests=12, aborted=1)
    result = _validate(events, metrics)
    assert result["trace_format"] == "qwen_budget_capped_terminal"
    assert result["budget_capped_partial_response_group"] is True
    assert result["budget_capped_aborted_logical_requests"] == 1
    # The killed group is NOT in the ledger: the engine never completed it.
    assert result["completed_logical_model_requests"] == 12


# --------------------------------------------------------------------------- #
# MUTATION PROOF (c): a capped task whose brackets do NOT reconcile refuses    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("overrides", "completed", "fragment"),
    (
        # One request's worth of max_tokens missing from the sum.
        (
            {"max_tokens_sum": 11 * contract.QWEN_VISIBLE_MAX_OUTPUT_TOKENS},
            12,
            "max-token algebra does not reconcile",
        ),
        # The proxy ledger claims one more completed request than the engine
        # bracket ever finished.
        ({}, 13, "engine completion metrics do not reconcile"),
        # The bracket finished a request the trace cannot account for.
        (
            {
                "max_tokens_count": 13,
                "max_tokens_le_50000": 13,
                "max_tokens_le_inf": 13,
                "request_success_stop": 13,
                "max_tokens_sum": 13 * contract.QWEN_VISIBLE_MAX_OUTPUT_TOKENS,
            },
            13,
            "max-token algebra does not reconcile",
        ),
        # A compaction bucket the trace's request count cannot absorb.
        ({"max_tokens_le_20000": 3}, 12, "max-token algebra does not reconcile"),
        # An unpinned low request in the histogram.
        (
            {"max_tokens_le_10000": 1},
            12,
            "max-token histogram has an unpinned low request",
        ),
        # A forbidden completion class.
        (
            {"request_success_error": 1},
            12,
            "forbidden completion reasons present",
        ),
        # The histogram disagrees with the completion count.
        (
            {"max_tokens_count": 13},
            12,
            "engine completion metrics do not reconcile",
        ),
    ),
)
def test_capped_terminal_with_broken_brackets_refuses(
    overrides: dict[str, int],
    completed: int,
    fragment: str,
) -> None:
    events = _capped_trace()
    metrics = _capped_metrics(overrides=overrides)
    with pytest.raises(contract.ContractError) as error:
        _validate(events, metrics, completed=completed)
    assert fragment in str(error.value)


def test_capped_abort_must_match_the_kill_signature_in_the_trace() -> None:
    """Trace shape and engine abort counter check EACH OTHER, both directions."""
    # The trace says the kill landed mid-response; the engine reports no abort.
    partial = _capped_trace(partial_final_group=True)
    with pytest.raises(contract.ContractError) as missing_abort:
        _validate(partial, _capped_metrics(completed=12, aborted=0))
    assert "abort=0 but the campaign declares capped_requests=1" in str(
        missing_abort.value
    )

    # The trace ended cleanly between requests; the engine reports an abort.
    clean = _capped_trace()
    with pytest.raises(contract.ContractError) as stray_abort:
        _validate(clean, _capped_metrics(completed=12, aborted=1))
    assert "abort=1 but the campaign declares capped_requests=0" in str(
        stray_abort.value
    )


def test_capped_terminal_refuses_an_unterminated_group_before_the_end() -> None:
    """Only the LAST group may be cut. An earlier gap is a broken trace."""
    events = _capped_trace()
    events[3:3] = [
        base._assistant_event(
            response_id="orphan-text",
            session_id=SESSION_A,
            content=[{"type": "text", "text": "no tool call"}],
            stop_reason=None,
        ),
        base._context_event(
            event_type="user",
            event_id="orphan-boundary",
            session_id=SESSION_A,
        ),
    ]
    with pytest.raises(contract.ContractError) as error:
        _validate(events, _capped_metrics())
    assert "incomplete before the terminal group" in str(error.value)


def test_capped_terminal_refuses_a_trace_that_continues_past_the_cut() -> None:
    """An unterminated group with events after it is not a kill signature."""
    events = _capped_trace(partial_final_group=True)
    events.append(
        base._context_event(
            event_type="user",
            event_id="impossible-reply",
            session_id=SESSION_A,
        )
    )
    with pytest.raises(contract.ContractError) as error:
        _validate(events, _capped_metrics(completed=12, aborted=1))
    assert "continues past an unterminated response group" in str(error.value)


# --------------------------------------------------------------------------- #
# THE UNCAPPED PATH IS UNCHANGED                                               #
# --------------------------------------------------------------------------- #
def test_uncapped_algebra_is_the_pre_cap_algebra_to_the_digit() -> None:
    """capped_requests=0 must reduce the generalised split to the old clause."""
    events, metrics = _uncapped_result_case()
    result = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=SESSION_A,
        expected_completed_logical_model_requests=13,
        metrics_pre=metrics[0],
        metrics_post=metrics[1],
    )
    assert result["trace_format"] == "qwen_result"
    assert result["budget_capped_terminal"] is False
    assert result["budget_capped_aborted_logical_requests"] == 0
    evidence = result["qwen_compaction_metric_evidence"]
    assert evidence["normal_requests"] == 13
    assert evidence["total_compaction_requests"] == 0
    assert evidence["max_tokens_le_20000"] == 0


def test_compaction_metric_evidence_key_set_is_frozen() -> None:
    """fr13_floor_gate replays this dict WHOLE against banked provenance.

    ``_fixed32_trace_model_requests`` compares
    ``provenance["qwen_compaction_metric_evidence"]`` to a fresh validation of
    the same trace with ``!=``. Adding a key here therefore invalidates every
    banked artifact at once -- silently, and only when someone re-runs the gate.
    Site 26 wanted to record the capped terminal in this dict and did not, for
    exactly this reason; the capped facts live on the trace-request record and
    on the v3 provenance record instead. If you must add one, re-bank first.
    """
    events, metrics = _uncapped_result_case()
    evidence = contract.validate_fixed32_trace_model_requests(
        events,
        expected_session_id=SESSION_A,
        expected_completed_logical_model_requests=13,
        metrics_pre=metrics[0],
        metrics_post=metrics[1],
    )["qwen_compaction_metric_evidence"]
    assert set(evidence) == {
        "schema",
        "metrics_pre_sha256",
        "metrics_post_sha256",
        "completed_engine_requests",
        "normal_visible_max_output_tokens",
        "compaction_max_output_tokens",
        "normal_requests",
        "successful_compaction_requests",
        "failed_compaction_requests",
        "total_compaction_requests",
        "unobservable_compaction_boundaries",
        "max_tokens_count",
        "max_tokens_sum",
        "max_tokens_le_10000",
        "max_tokens_le_20000",
        "max_tokens_le_50000",
        "max_tokens_le_inf",
        "request_success_stop",
        "request_success_length",
        "request_success_non_stop",
        "prompt_tokens",
        "generation_tokens",
        "visible_prompt_tokens",
        "visible_generation_tokens",
        "hidden_prompt_tokens",
        "hidden_generation_tokens",
    }


# --------------------------------------------------------------------------- #
# THE RUNNER SIDE: the record names its own terminal                           #
# --------------------------------------------------------------------------- #
def _capped_agent_meta(runner: Any, task_dir: Path) -> dict[str, Any]:
    return base._fixed32_agent_meta(
        runner,
        task_dir,
        exit_code=124,
        timed_out=True,
        budget_capped=True,
        campaign_budget_s=9000.0,
        campaign_budget_was_the_binding_limit=True,
        elapsed_s=9114.0,
    )


def _task_auth(task_key_id: str, logical: int, records: int, aborted: int = 0) -> dict:
    evidence = base._task_evidence(task_key_id, logical, records)
    evidence["aborted_logical_requests"] = aborted
    return evidence


def _write_capped_task(task_dir: Path, events: list[dict[str, Any]], metrics) -> tuple:
    trace_path = task_dir / "qwen_trace.jsonl"
    trace_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    pre_path = task_dir / "vllm_metrics_pre.txt"
    post_path = task_dir / "vllm_metrics_post.txt"
    pre_path.write_bytes(metrics[0])
    post_path.write_bytes(metrics[1])
    return trace_path, pre_path, post_path


def test_real_task_provenance_names_the_budget_capped_terminal(
    task_dir: Path,
) -> None:
    runner = _load_runner()
    events = _capped_trace()
    metrics = _capped_metrics()
    trace_path, pre_path, post_path = _write_capped_task(task_dir, events, metrics)
    task_key_id = "f" * 64
    provenance = runner._fixed32_real_task_provenance(
        instance_id=TASK_A,
        trace_path=trace_path,
        agent_meta=_capped_agent_meta(runner, task_dir),
        task_key_id=task_key_id,
        task_auth_before=_task_auth(task_key_id, 0, 1),
        task_auth_after=_task_auth(task_key_id, 12, 49),
        metrics_pre_path=pre_path,
        metrics_post_path=post_path,
    )
    assert provenance["budget_capped_terminal"] is True
    assert provenance["campaign_budget_s"] == 9000.0
    assert provenance["budget_capped_elapsed_s"] == 9114.0
    assert provenance["budget_capped_partial_response_group"] is False
    assert provenance["budget_capped_aborted_logical_requests"] == 0
    assert provenance["trace_completed_logical_model_requests"] == 12
    assert provenance["completed_logical_model_requests"] == 12
    assert provenance["qwen_metric_scope"] == "task"


def test_real_task_provenance_leaves_an_uncapped_record_unmarked(
    task_dir: Path,
) -> None:
    runner = _load_runner()
    events, metrics = _uncapped_result_case()
    trace_path, pre_path, post_path = _write_capped_task(task_dir, events, metrics)
    task_key_id = "e" * 64
    provenance = runner._fixed32_real_task_provenance(
        instance_id=TASK_A,
        trace_path=trace_path,
        agent_meta=base._fixed32_agent_meta(runner, task_dir),
        task_key_id=task_key_id,
        task_auth_before=_task_auth(task_key_id, 0, 1),
        task_auth_after=_task_auth(task_key_id, 13, 53),
        metrics_pre_path=pre_path,
        metrics_post_path=post_path,
    )
    assert provenance["budget_capped_terminal"] is False
    assert provenance["campaign_budget_s"] is None
    assert provenance["budget_capped_elapsed_s"] is None
    assert provenance["budget_capped_aborted_logical_requests"] == 0


def test_real_task_provenance_refuses_a_capped_abort_the_ledger_denies(
    task_dir: Path,
) -> None:
    """The proxy's own ledger is the third meter and it must agree."""
    runner = _load_runner()
    events = _capped_trace(partial_final_group=True)
    metrics = _capped_metrics(completed=12, aborted=1)
    trace_path, pre_path, post_path = _write_capped_task(task_dir, events, metrics)
    task_key_id = "d" * 64
    with pytest.raises(runner.Fixed32BoundaryError) as error:
        runner._fixed32_real_task_provenance(
            instance_id=TASK_A,
            trace_path=trace_path,
            agent_meta=_capped_agent_meta(runner, task_dir),
            task_key_id=task_key_id,
            task_auth_before=_task_auth(task_key_id, 0, 1),
            # The engine and the trace both say one request was aborted; the
            # proxy ledger says none.
            task_auth_after=_task_auth(task_key_id, 12, 49, aborted=0),
            metrics_pre_path=pre_path,
            metrics_post_path=post_path,
        )
    message = str(error.value)
    assert "task-auth request counts do not reconcile" in message
    assert "aborted=0 against expected 1" in message


def test_real_task_provenance_still_refuses_a_stray_uncapped_abort(
    task_dir: Path,
) -> None:
    runner = _load_runner()
    events, metrics = _uncapped_result_case()
    trace_path, pre_path, post_path = _write_capped_task(task_dir, events, metrics)
    task_key_id = "c" * 64
    with pytest.raises(runner.Fixed32BoundaryError) as error:
        runner._fixed32_real_task_provenance(
            instance_id=TASK_A,
            trace_path=trace_path,
            agent_meta=base._fixed32_agent_meta(runner, task_dir),
            task_key_id=task_key_id,
            task_auth_before=_task_auth(task_key_id, 0, 1),
            task_auth_after=_task_auth(task_key_id, 13, 53, aborted=1),
            metrics_pre_path=pre_path,
            metrics_post_path=post_path,
        )
    assert "aborted=1 against expected 0" in str(error.value)


def test_capped_declaration_is_gated_on_the_corroborated_cap() -> None:
    """The runner may only declare what the runner record corroborates.

    The declaration passed to the contract is ``budget_capped_corroborated``,
    which is computed from timed_out + binding + budget + elapsed -- never from
    the ``budget_capped`` flag on its own.
    """
    source = (SCRIPTS / "run_swe_bench_q36_a.py").read_text(encoding="utf-8")
    assert "budget_capped_terminal=budget_capped_corroborated," in source
    assert source.count("budget_capped_terminal=") == 1


def test_the_class_is_documented_where_it_is_implemented() -> None:
    source = (SCRIPTS / "fr13_fixed32_contract.py").read_text(encoding="utf-8")
    assert "THE BUDGET-CAPPED TERMINAL CLASS" in source
    # The refusal must still exist for the undeclared case.
    assert '"fixed32 compaction metric evidence requires a Qwen result"' in source

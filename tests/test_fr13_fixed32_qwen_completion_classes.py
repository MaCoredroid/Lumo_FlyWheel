"""Engine completion classes at 16-task scale.

THE FAILURE THIS FILE EXISTS FOR. The first 16-task pool arm served 390 logical
model requests and produced exactly ONE terminated at max_tokens
(finished_reason="length", inside astropy__astropy-14369, the last-closing
bracket). The campaign algebra pinned request_success_stop == completed_total and
every other reason to zero, so finalization refused the whole arm after 87
minutes of good serving -- and would have refused every subsequent pass
identically.

Nothing about the measurement was wrong. Every other term reconciled to the
digit: 386 trace-side completions + 4 failed compactions = 390 = max_tokens_count
= le_50000 = le_inf, and max_tokens_sum 12_728_448 = 386*32768 + 4*20000 exactly.
The algebra simply had no CATEGORY for a legal outcome that only shows up at
scale: an exact4 arm serves ~100 requests and never drew one.

The fix is a category, not a tolerance.
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
import fr13_floor_gate as floor_gate  # noqa: E402


def _load(name: str, relative: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


campaign_fixtures = _load(
    "completion_class_campaign_fixtures",
    "tests/test_fr13_fixed32_b4_campaign_provenance.py",
)

POOL16_TASK_IDS = list(floor_gate.CANONICAL_TASK_IDS)

# The real pass_00 arm, to the digit.
REAL_COMPLETED_TOTAL = 390
REAL_STOP = 389
REAL_LENGTH = 1
REAL_FAILED_COMPACTIONS = 4
REAL_TRACE_COMPLETIONS = 386


def _metrics(
    completed: int,
    *,
    compactions: int = 0,
    normal_requests: int | None = None,
    prompt_tokens: int | None = None,
    generation_tokens: int | None = None,
    length_terminated: int = 0,
    abort: int = 0,
    error: int = 0,
    repetition: int = 0,
) -> bytes:
    """Prometheus text with the completion classes under our control.

    Deliberately NOT the campaign-provenance module's `_metrics`: that one pins
    every non-stop reason to zero, which is precisely the assumption on trial.
    """
    if normal_requests is None:
        normal_requests = completed - compactions
    if prompt_tokens is None:
        prompt_tokens = completed * 32
    if generation_tokens is None:
        generation_tokens = completed * 8
    max_tokens_sum = (
        normal_requests * contract.QWEN_VISIBLE_MAX_OUTPUT_TOKENS
        + compactions * contract.QWEN_COMPACTION_MAX_OUTPUT_TOKENS
    )
    base = 'engine="0",model_name="qwen3.6-27b"'
    lines = [
        f"vllm:prompt_tokens_total{{{base}}} {prompt_tokens}",
        f"vllm:generation_tokens_total{{{base}}} {generation_tokens}",
        f"vllm:request_params_max_tokens_count{{{base}}} {completed}",
        f"vllm:request_params_max_tokens_sum{{{base}}} {max_tokens_sum}",
    ]
    counts = {
        "stop": completed - length_terminated,
        "length": length_terminated,
        "abort": abort,
        "error": error,
        "repetition": repetition,
    }
    for reason, value in counts.items():
        labels = (
            f'engine="0",finished_reason="{reason}",model_name="qwen3.6-27b"'
        )
        lines.append(f"vllm:request_success_total{{{labels}}} {value}")
    for le, value in (
        ("10000.0", 0),
        ("20000.0", compactions),
        ("50000.0", completed),
        ("+Inf", completed),
    ):
        labels = f'engine="0",le="{le}",model_name="qwen3.6-27b"'
        lines.append(f"vllm:request_params_max_tokens_bucket{{{labels}}} {value}")
    return ("\n".join(lines) + "\n").encode("ascii")


def _pool16_campaign_tasks(
    per_task_completions: list[int],
    *,
    failed_compactions_on_first: int = 0,
) -> list[dict[str, Any]]:
    """16 canonical tasks, each with a real Qwen trace of the given length.

    `failed_compactions_on_first` inflates the first task's task-auth count above
    its trace count, which is how a failed compaction presents: the request
    happened on the engine but never appears in the task trace.
    """
    assert len(per_task_completions) == 16
    tasks = []
    for index, (instance_id, completed) in enumerate(
        zip(POOL16_TASK_IDS, per_task_completions, strict=True)
    ):
        # A compaction's tokens are HIDDEN: they land in the result aggregate but
        # never in a visible assistant message, so a task carrying compactions
        # must show a positive hidden remainder or the algebra rejects it before
        # the completion classes are ever reached.
        failed = failed_compactions_on_first if index == 0 else 0
        events = campaign_fixtures._qwen_trace_with_request_count(
            instance_id,
            completed,
            hidden_input_tokens=20 * failed,
            hidden_output_tokens=2 * failed,
        )
        expected = completed + failed
        tasks.append(
            {
                "instance_id": instance_id,
                "expected_session_id": contract.fixed32_trace_session_id(
                    instance_id
                ),
                "expected_completed_logical_model_requests": expected,
                "events": events,
                "budget_capped": False,
            }
        )
    return tasks


# The real pass_00 shape: 16 tasks summing to 386 trace completions, plus 4
# failed compactions, 390 total, one of which terminated on length.
REAL_PER_TASK = [13, 21, 17, 69, 26, 29, 25, 26, 22, 8, 8, 31, 16, 28, 25, 22]


def _real_shape_metrics(**overrides: Any) -> tuple[bytes, bytes]:
    tasks = _pool16_campaign_tasks(
        REAL_PER_TASK, failed_compactions_on_first=REAL_FAILED_COMPACTIONS
    )
    prompt = sum(
        task["events"][-1]["usage"]["input_tokens"] for task in tasks
    )
    generation = sum(
        task["events"][-1]["usage"]["output_tokens"] for task in tasks
    )
    kwargs: dict[str, Any] = {
        "compactions": REAL_FAILED_COMPACTIONS,
        "normal_requests": REAL_TRACE_COMPLETIONS,
        "prompt_tokens": prompt,
        "generation_tokens": generation,
    }
    kwargs.update(overrides)
    return (
        _metrics(0, compactions=0, normal_requests=0, prompt_tokens=0,
                 generation_tokens=0),
        _metrics(REAL_COMPLETED_TOTAL, **kwargs),
    )


def _validate(**overrides: Any) -> dict[str, Any]:
    pre, post = _real_shape_metrics(**overrides)
    return contract.validate_fixed32_qwen_campaign_metrics(
        _pool16_campaign_tasks(
            REAL_PER_TASK, failed_compactions_on_first=REAL_FAILED_COMPACTIONS
        ),
        metrics_pre=pre,
        metrics_post=post,
    )


# --------------------------------------------------------------------------- #
# the fixture reproduces the real arm                                          #
# --------------------------------------------------------------------------- #
def test_the_16_task_fixture_reproduces_the_real_arms_counts() -> None:
    assert sum(REAL_PER_TASK) == REAL_TRACE_COMPLETIONS
    assert REAL_TRACE_COMPLETIONS + REAL_FAILED_COMPACTIONS == REAL_COMPLETED_TOTAL
    assert REAL_STOP + REAL_LENGTH == REAL_COMPLETED_TOTAL
    # and the max_tokens algebra the real arm satisfied exactly
    assert (
        REAL_TRACE_COMPLETIONS * contract.QWEN_VISIBLE_MAX_OUTPUT_TOKENS
        + REAL_FAILED_COMPACTIONS * contract.QWEN_COMPACTION_MAX_OUTPUT_TOKENS
    ) == 12_728_448


# --------------------------------------------------------------------------- #
# the fix: length is an accounted class                                        #
# --------------------------------------------------------------------------- #
def test_one_length_termination_in_390_requests_reconciles() -> None:
    """The exact case that killed pass_00."""
    evidence = _validate(length_terminated=REAL_LENGTH)
    metric = evidence["metric_evidence"]
    assert metric["completed_engine_requests"] == REAL_COMPLETED_TOTAL
    assert metric["request_success_stop"] == REAL_STOP
    assert metric["request_success_length"] == REAL_LENGTH


def test_the_length_count_is_published_not_absorbed() -> None:
    """A reader must be able to see how much truncated traffic a campaign had."""
    metric = _validate(length_terminated=REAL_LENGTH)["metric_evidence"]
    assert "request_success_length" in metric
    assert metric["request_success_length"] == REAL_LENGTH
    # non_stop stays exact because the forbidden reasons are proven zero
    assert metric["request_success_non_stop"] == REAL_LENGTH
    assert (
        metric["request_success_stop"] + metric["request_success_length"]
        == metric["completed_engine_requests"]
    )


def test_an_all_stop_campaign_is_unchanged() -> None:
    """The exact4-era behaviour: zero length terminations still reconciles."""
    metric = _validate(length_terminated=0)["metric_evidence"]
    assert metric["request_success_stop"] == REAL_COMPLETED_TOTAL
    assert metric["request_success_length"] == 0
    assert metric["request_success_non_stop"] == 0


@pytest.mark.parametrize("length_terminated", [0, 1, 7, REAL_COMPLETED_TOTAL])
def test_any_number_of_length_terminations_reconciles(
    length_terminated: int,
) -> None:
    """Legality does not depend on the count -- it depends on the class.

    A tolerance would have been "allow up to N"; that is exactly what this is
    not. Every truncated request is counted and published.
    """
    metric = _validate(length_terminated=length_terminated)["metric_evidence"]
    assert metric["request_success_length"] == length_terminated
    assert (
        metric["request_success_stop"] + length_terminated
        == REAL_COMPLETED_TOTAL
    )


# --------------------------------------------------------------------------- #
# what stays forbidden                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reason", ["error", "repetition"])
def test_defect_completion_reasons_are_still_refused(reason: str) -> None:
    with pytest.raises(contract.ContractError) as excinfo:
        _validate(length_terminated=REAL_LENGTH, **{reason: 1})
    message = str(excinfo.value)
    assert "forbidden completion reasons present" in message
    assert f"{reason}=1" in message  # names the measured value


def test_an_abort_with_no_declared_budget_cap_is_still_refused() -> None:
    """The abort class is CONDITIONAL, and the condition is a declaration.

    Every campaign that ran before 2026-08-13 declares zero capped tasks, so
    abort stays pinned at zero for all of them -- an abort nobody asked for
    still means the engine did not serve what was asked.
    """
    with pytest.raises(contract.ContractError) as excinfo:
        _validate(length_terminated=REAL_LENGTH, abort=1)
    message = str(excinfo.value)
    assert "abort=1" in message
    assert "capped_requests=0" in message
    assert "no declared budget cap" in message


def test_a_short_stop_count_is_still_refused_and_names_the_numbers() -> None:
    """Widening the class must not have widened the identity."""
    pre, post = _real_shape_metrics(length_terminated=REAL_LENGTH)
    # one completion vanishes from both terminal classes
    post = post.replace(
        f'finished_reason="stop",model_name="qwen3.6-27b"}} {REAL_STOP}'.encode(),
        f'finished_reason="stop",model_name="qwen3.6-27b"}} {REAL_STOP - 1}'.encode(),
    )
    with pytest.raises(contract.ContractError) as excinfo:
        contract.validate_fixed32_qwen_campaign_metrics(
            _pool16_campaign_tasks(
                REAL_PER_TASK,
                failed_compactions_on_first=REAL_FAILED_COMPACTIONS,
            ),
            metrics_pre=pre,
            metrics_post=post,
        )
    message = str(excinfo.value)
    assert f"completed={REAL_COMPLETED_TOTAL}" in message
    assert f"stop={REAL_STOP - 1}" in message
    assert f"length={REAL_LENGTH}" in message
    assert "terminal total 389" in message


def test_a_length_termination_does_not_excuse_a_broken_histogram() -> None:
    pre, post = _real_shape_metrics(length_terminated=REAL_LENGTH)
    post = post.replace(
        f'le="+Inf",model_name="qwen3.6-27b"}} {REAL_COMPLETED_TOTAL}'.encode(),
        f'le="+Inf",model_name="qwen3.6-27b"}} {REAL_COMPLETED_TOTAL - 1}'.encode(),
    )
    with pytest.raises(contract.ContractError) as excinfo:
        contract.validate_fixed32_qwen_campaign_metrics(
            _pool16_campaign_tasks(
                REAL_PER_TASK,
                failed_compactions_on_first=REAL_FAILED_COMPACTIONS,
            ),
            metrics_pre=pre,
            metrics_post=post,
        )
    assert "max_tokens_le_inf" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# the two paths share one implementation                                       #
# --------------------------------------------------------------------------- #
def test_the_class_lists_are_exhaustive_over_the_metric_family() -> None:
    """Every finished_reason the snapshot parses is in exactly one list."""
    classes = set(contract.QWEN_TERMINAL_COMPLETION_REASONS)
    forbidden = set(contract.QWEN_FORBIDDEN_COMPLETION_REASONS)
    capped = {contract.QWEN_CAPPED_COMPLETION_REASON}
    assert classes.isdisjoint(forbidden)
    assert classes.isdisjoint(capped)
    assert forbidden.isdisjoint(capped)
    assert classes | forbidden | capped == {
        "stop",
        "length",
        "abort",
        "error",
        "repetition",
    }


def test_single_task_and_campaign_paths_use_the_same_helper() -> None:
    """They carried the identical clause; drifting them apart is how a class
    ends up legal in one path and forbidden in the other."""
    source = (SCRIPTS / "fr13_fixed32_contract.py").read_text(encoding="utf-8")
    assert source.count("_fixed32_qwen_completion_classes(") == 3  # def + 2 uses
    for scope in ('scope="task"', 'scope="campaign"'):
        assert scope in source


@pytest.mark.parametrize("scope", ["task", "campaign"])
def test_the_shared_helper_accepts_length_and_refuses_defects(
    scope: str,
) -> None:
    deltas = {
        "max_tokens_count": 390,
        "max_tokens_le_inf": 390,
        "max_tokens_le_50000": 390,
        "request_success_stop": 389,
        "request_success_length": 1,
        "request_success_abort": 0,
        "request_success_error": 0,
        "request_success_repetition": 0,
    }
    counts = contract._fixed32_qwen_completion_classes(
        deltas, completed=390, scope=scope
    )
    assert counts == {"stop": 389, "length": 1, "abort": 0}
    broken = dict(deltas, request_success_error=1)
    with pytest.raises(contract.ContractError, match=f"qwen {scope} engine"):
        contract._fixed32_qwen_completion_classes(
            broken, completed=390, scope=scope
        )

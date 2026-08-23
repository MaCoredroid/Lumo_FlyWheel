"""The EXISTING thinking cap (LUMO_PROXY_THINK_BUDGET), exercised on the
current proxy.

The cap was built at FR13 439c43567 for the char-8 era and predates the nvfp4
port, so the brake spec's claims about it need to be re-verified against the
code as it stands rather than as it was written. These tests do that on CPU:
the budget parser's arming semantics, the over-budget detection, the shape of
the forced-close prefill, and the under-budget no-op.

They are verification, not arming. Default remains OFF and no serving path is
touched.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from lumo_flywheel_serving import inference_proxy as proxy  # noqa: E402

BUDGET_ENV = "LUMO_PROXY_THINK_BUDGET"


def _with_budget(value: str | None):
    saved = os.environ.get(BUDGET_ENV)

    class _Ctx:
        def __enter__(self):
            if value is None:
                os.environ.pop(BUDGET_ENV, None)
            else:
                os.environ[BUDGET_ENV] = value

        def __exit__(self, *exc):
            if saved is None:
                os.environ.pop(BUDGET_ENV, None)
            else:
                os.environ[BUDGET_ENV] = saved

    return _Ctx()


def test_the_cap_is_default_off() -> None:
    """Unset means the legacy path, byte-identical."""
    with _with_budget(None):
        assert proxy._parse_think_budget() is None


@pytest.mark.parametrize("value,expected", [("500", 500), ("24000", 24000), ("1", 1)])
def test_a_positive_budget_arms(value: str, expected: int) -> None:
    with _with_budget(value):
        assert proxy._parse_think_budget() == expected


@pytest.mark.parametrize("value", ["0", "-5", "abc", "24,000", "24000abc", " "])
def test_malformed_budget_values_currently_disarm(value: str) -> None:
    """DOCUMENTED HAZARD, not an endorsement.

    _parse_think_budget returns None for anything it cannot parse, so
    LUMO_PROXY_THINK_BUDGET="24,000" silently DISARMS the brake instead of
    refusing. That is the vacuous-gate shape: a typo in the arming value turns
    the safety net off and nothing says so. Harmless while the cap is unused;
    it is the one change the spec recommends before arming. Pinned here so the
    hazard is visible and a strict-parsing fix shows up as a deliberate change.
    """
    with _with_budget(value):
        assert proxy._parse_think_budget() is None, (
            f"{value!r} now arms the cap -- if that is the strict-parsing fix, "
            "update this test deliberately"
        )


def test_over_budget_call_a_is_detected() -> None:
    call_a = {
        "output": [{"type": "reasoning", "content": [{"text": "thinking. " * 8}]}],
        "incomplete_details": {"reason": "max_output_tokens"},
    }
    assert proxy._think_response_hit_budget(call_a) is True
    reasoning = proxy._think_extract_reasoning(call_a)
    assert reasoning.startswith("thinking.")


def test_under_budget_call_a_is_a_no_op() -> None:
    call_a = {
        "output": [{"type": "reasoning", "content": [{"text": "short"}]}],
        "incomplete_details": None,
    }
    assert proxy._think_response_hit_budget(call_a) is False


def test_the_forced_close_prefill_leaves_the_close_TO_THE_MODEL() -> None:
    """The interaction claim the spec rests on, re-verified on current code.

    My first spec proposed appending </think> directly. The existing
    implementation deliberately does NOT: it prefills an OPEN <think> plus the
    reasoning plus a terse cutoff, so the model GENERATES the close. The
    docstring records why -- the qwen3 reasoning parser only watches generated
    tokens, so a prefilled close mislabels the whole continuation as reasoning
    and the tool call is lost. Prefilling the close would have broken tool
    calling, which is exactly the failure the brake exists to avoid.
    """
    reasoning = "I considered the options. " * 4
    prefill = proxy._think_build_cutoff_prefill(reasoning)

    assert prefill["role"] == "assistant"
    assert prefill["status"] == "incomplete"
    text = prefill["content"][0]["text"]
    assert text.startswith("<think>"), "the block must be left OPEN"
    assert "</think>" not in text, (
        "a prefilled close mislabels the continuation as reasoning and loses "
        "the tool call -- the model must generate it"
    )
    assert reasoning in text, "call-A's reasoning must be preserved verbatim"
    # terse by default; the verbose framing lets the model keep thinking
    assert text.rstrip().endswith("act now.")


def test_the_cutoff_text_is_overridable_but_terse_by_default() -> None:
    saved = os.environ.get("LUMO_PROXY_THINK_CUTOFF")
    try:
        os.environ["LUMO_PROXY_THINK_CUTOFF"] = "\n\nSTOP."
        text = proxy._think_build_cutoff_prefill("r")["content"][0]["text"]
        assert text.endswith("STOP.")
    finally:
        if saved is None:
            os.environ.pop("LUMO_PROXY_THINK_CUTOFF", None)
        else:
            os.environ["LUMO_PROXY_THINK_CUTOFF"] = saved


def test_the_cap_is_wired_only_into_the_responses_path() -> None:
    """COVERAGE, pinned. The proxy serves both routes; the cap guards one.

    If the SWE client uses /v1/chat/completions, the cap as written does not
    cover it and extending it is the only build the brake needs. This test
    states the current coverage so that conclusion cannot drift silently.
    """
    source = (REPO / "src" / "lumo_flywheel_serving" / "inference_proxy.py").read_text()
    assert 'elif self.path == "/v1/chat/completions":' in source, (
        "the proxy no longer serves chat/completions -- re-do the coverage check"
    )
    cap_index = source.index("think_cap_active = False")
    guard_index = source.index('if self.path == "/v1/responses":', cap_index)
    chat_index = source.index('elif self.path == "/v1/chat/completions":', cap_index)
    assert guard_index < chat_index, "the cap sits in the /v1/responses branch"
    # and nothing in the chat branch arms it
    chat_branch = source[chat_index : chat_index + 4000]
    assert "think_cap_active = True" not in chat_branch, (
        "chat/completions now arms the cap -- coverage changed, update the spec"
    )

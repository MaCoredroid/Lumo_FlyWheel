"""The 24,000 output ceiling: armed on the path the SWE client actually uses.

Mark ruled "shrink the thinking ceiling to 24k". The coverage check moved WHICH
knob does it: LUMO_PROXY_THINK_BUDGET is wired only into /v1/responses, and the
qwen-code SWE client rides /v1/chat/completions (11,918 chat vs 110 responses
across 55 banked runroots, and the 110 are two rejected auth probes per runroot).
Arming THINK_BUDGET would have been a no-op.

LUMO_PROXY_MAX_OUTPUT_TOKENS is already applied in
normalize_chat_completions_request_payload -- the client's path -- and already
defaulted to 32768, which is exactly the ceiling all five banked degenerations
ran into. The landing lowers that default to 24000 and pins it.

Semantic difference, stated: this caps thinking+answer TOTAL, where THINK_BUDGET
would cap thinking alone and force an answer. It costs nothing on healthy turns
because healthy answers are small -- the healthy max TOTAL per turn (22,398)
equals the healthy max THINKING block -- so a degenerate turn is truncated
rather than force-closed.
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

CEILING_ENV = "LUMO_PROXY_MAX_OUTPUT_TOKENS"
CEILING = 24000

#: From the banked census: no healthy arm's largest turn exceeded this.
HEALTHY_MAX_TURN_TOKENS = 22398
#: The smallest degenerate turn. Every one of the five hit the old 32768 ceiling.
SMALLEST_DEGENERATE_TURN = 30731


def _with_ceiling(value: str | None):
    saved = os.environ.get(CEILING_ENV)

    class _Ctx:
        def __enter__(self):
            if value is None:
                os.environ.pop(CEILING_ENV, None)
            else:
                os.environ[CEILING_ENV] = value

        def __exit__(self, *exc):
            if saved is None:
                os.environ.pop(CEILING_ENV, None)
            else:
                os.environ[CEILING_ENV] = saved

    return _Ctx()


def _client_body(max_tokens: int = 32768) -> dict:
    """The qwen-code chat body, as captured in a banked request dump."""
    return {
        "model": "qwen3.8-27b-nvfp4-radixark",
        "messages": [{"role": "user", "content": "fix the bug"}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [],
    }


def test_the_off_path_is_byte_identical() -> None:
    """Unset means the request is untouched -- the legacy path."""
    with _with_ceiling(None):
        body = _client_body()
        out = proxy.normalize_chat_completions_request_payload(dict(body))
        assert out["max_tokens"] == 32768, "unset must not cap"
        assert out == {**body, **out}, "no other field may move"


def test_the_ceiling_lowers_the_clients_32768() -> None:
    """The degenerate ceiling, closed. This is the whole behaviour change."""
    with _with_ceiling(str(CEILING)):
        out = proxy.normalize_chat_completions_request_payload(_client_body())
        assert out["max_tokens"] == CEILING


def test_the_ceiling_never_raises_a_lower_request() -> None:
    """A client asking for less keeps less -- the cap only ever lowers."""
    with _with_ceiling(str(CEILING)):
        out = proxy.normalize_chat_completions_request_payload(_client_body(4096))
        assert out["max_tokens"] == 4096


def test_a_healthy_turn_is_unaffected_by_the_ceiling() -> None:
    """MUTATION PROOF, from the corpus: zero healthy clips.

    The largest healthy turn in 105 banked task-arms was 22,398 tokens. Under a
    24,000 ceiling it is not truncated; the ceiling only bites above it.
    """
    assert HEALTHY_MAX_TURN_TOKENS < CEILING, (
        "the ceiling must sit above every healthy turn in the corpus"
    )
    with _with_ceiling(str(CEILING)):
        out = proxy.normalize_chat_completions_request_payload(
            _client_body(HEALTHY_MAX_TURN_TOKENS)
        )
        assert out["max_tokens"] == HEALTHY_MAX_TURN_TOKENS, (
            "a healthy-sized request must pass through untouched"
        )


def test_a_degenerate_turn_is_bounded_by_the_ceiling() -> None:
    """The other half: every banked degeneration exceeded the new ceiling."""
    assert SMALLEST_DEGENERATE_TURN > CEILING
    with _with_ceiling(str(CEILING)):
        out = proxy.normalize_chat_completions_request_payload(
            _client_body(SMALLEST_DEGENERATE_TURN)
        )
        assert out["max_tokens"] == CEILING, "the runaway must be bounded"


def test_the_margin_is_stated_not_assumed() -> None:
    """Both sides of the corridor, so a future edit sees the cost."""
    assert CEILING - HEALTHY_MAX_TURN_TOKENS == 1602, "headroom above healthy"
    assert SMALLEST_DEGENERATE_TURN - CEILING == 6731, "margin below degenerate"


@pytest.mark.parametrize("value", ["abc", "24,000", ""])
def test_a_malformed_ceiling_leaves_the_request_uncapped(value: str) -> None:
    """DOCUMENTED HAZARD. Same shape as the THINK_BUDGET parser.

    normalize_chat_completions_request_payload swallows a ValueError and leaves
    max_tokens alone, so LUMO_PROXY_MAX_OUTPUT_TOKENS="24,000" serves UNCAPPED
    rather than refusing. The launcher's exact pin is what closes this in
    practice -- it greps proxy_env.txt for the literal value and fails the arm --
    which is why the pin is part of this landing and not decoration.
    """
    with _with_ceiling(value):
        out = proxy.normalize_chat_completions_request_payload(_client_body())
        assert out["max_tokens"] == 32768, (
            "a malformed ceiling silently disarms; the launcher pin is the guard"
        )


def test_the_launcher_pins_the_ceiling_exactly() -> None:
    """House style: a drifted value must refuse, not serve."""
    launcher = (REPO / "scripts" / "fr13_bigdenom_swe_serve_variant.sh").read_text()
    assert 'LUMO_PROXY_MAX_OUTPUT_TOKENS=${DEPLOY_MAX_OUTPUT_TOKENS:-24000}' in launcher
    assert "proxy output-ceiling pin missing" in launcher
    assert "exit 5" in launcher


def test_the_serving_default_is_the_corpus_number() -> None:
    remote = (
        REPO / "scripts" / "swe_x86_helpers" / "relaunch_proxy_remote.sh"
    ).read_text()
    assert (
        "export LUMO_PROXY_MAX_OUTPUT_TOKENS=${LUMO_PROXY_MAX_OUTPUT_TOKENS:-24000}"
        in remote
    )
    assert "32768" in remote, "the superseded value should stay documented"


def test_think_budget_stays_off_because_it_cannot_reach_this_client() -> None:
    """Guard against arming the no-op knob by mistake."""
    remote = (
        REPO / "scripts" / "swe_x86_helpers" / "relaunch_proxy_remote.sh"
    ).read_text()
    assert (
        "export LUMO_PROXY_THINK_BUDGET=${LUMO_PROXY_THINK_BUDGET:-}" in remote
    ), (
        "THINK_BUDGET was given a default -- it is /v1/responses-only and the "
        "SWE client rides /v1/chat/completions, so a default here is a placebo"
    )

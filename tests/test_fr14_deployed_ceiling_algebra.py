"""SITE 25: the max-token algebra is era-scoped to the DEPLOYED ceiling.

The 24000 landing deployed ${DEPLOY_MAX_OUTPUT_TOKENS:-24000} while the
contract still multiplied by a module constant of 32768, so the first serve
carrying the new ceiling died on the contract expecting the old one -- request
counts reconciled perfectly and only the algebra failed.

The fix is not a second literal. The contract runs over BANKED 32768-era
runroots as well as new 24000-era ones, so it cannot hardcode either number: it
SOLVES for the ceiling a run served and requires the answer to be one this
campaign has deployed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fr13_fixed32_contract as contract  # noqa: E402

VARIANT = SCRIPTS / "fr13_bigdenom_swe_serve_variant.sh"
REMOTE_PROXY = SCRIPTS / "swe_x86_helpers" / "relaunch_proxy_remote.sh"


def _reconcile(normal: int, compactions: int, ceiling: int) -> int:
    """The token sum a run at `ceiling` would report."""
    return normal * ceiling + compactions * contract.QWEN_COMPACTION_MAX_OUTPUT_TOKENS


def _solve(total_sum: int, normal: int, compactions: int) -> int | None:
    """Mirror of the contract's era resolution, for test arithmetic."""
    component = compactions * contract.QWEN_COMPACTION_MAX_OUTPUT_TOKENS
    for candidate in contract.FIXED32_DEPLOYED_MAX_OUTPUT_TOKENS:
        if total_sum == normal * candidate + component:
            return candidate
    return None


def test_the_era_table_is_ordered_newest_first() -> None:
    ceilings = contract.FIXED32_DEPLOYED_MAX_OUTPUT_TOKENS
    assert ceilings == (24_000, 32_768)
    assert ceilings[0] == 24_000, "the newest deployed ceiling comes first"


def test_the_deployed_ceiling_and_the_launcher_pin_cannot_diverge() -> None:
    """THE AGREEMENT ASSERTION. This is what site 25 was missing.

    The launcher pins what a new serve may deploy; the contract lists what it
    will validate. If the pin names a ceiling the contract has never heard of,
    the next serve dies on the algebra exactly as it did this time.
    """
    variant = VARIANT.read_text()
    newest = contract.FIXED32_DEPLOYED_MAX_OUTPUT_TOKENS[0]
    assert f"LUMO_PROXY_MAX_OUTPUT_TOKENS=${{DEPLOY_MAX_OUTPUT_TOKENS:-{newest}}}" in (
        variant
    ), f"the launcher pin does not name the contract's newest ceiling {newest}"

    remote = REMOTE_PROXY.read_text()
    assert (
        f"export LUMO_PROXY_MAX_OUTPUT_TOKENS="
        f"${{LUMO_PROXY_MAX_OUTPUT_TOKENS:-{newest}}}" in remote
    ), "the serving default does not match the contract's newest ceiling"


def test_a_serve_at_the_new_ceiling_reconciles() -> None:
    """27 x 24000 -- the exact shape that died."""
    normal, compactions = 27, 0
    total = _reconcile(normal, compactions, 24_000)
    assert total == 648_000
    assert _solve(total, normal, compactions) == 24_000


def test_a_banked_serve_at_the_old_ceiling_STILL_reconciles() -> None:
    """Era scope, not replacement: 27 x 32768 must keep validating.

    The contract runs over banked runroots. Swapping one literal for another
    would have made every 32768-era artifact fail instead.
    """
    normal, compactions = 27, 0
    total = _reconcile(normal, compactions, 32_768)
    assert total == 884_736
    assert _solve(total, normal, compactions) == 32_768


@pytest.mark.parametrize("ceiling", [24_000, 32_768])
@pytest.mark.parametrize("compactions", [0, 1, 5])
def test_both_eras_reconcile_with_compactions_mixed_in(
    ceiling: int, compactions: int
) -> None:
    """13398-class compactions must not confuse the era resolution."""
    normal = 31
    total = _reconcile(normal, compactions, ceiling)
    assert _solve(total, normal, compactions) == ceiling


def test_a_drifted_ceiling_refuses_and_names_both_numbers() -> None:
    """MUTATION PROOF: a ceiling nobody deployed must not silently pass."""
    normal, compactions = 27, 0
    total = _reconcile(normal, compactions, 28_000)
    assert _solve(total, normal, compactions) is None, (
        "an undeployed ceiling must not resolve"
    )
    # ...and the contract's message names the candidates and the implied value
    source = (SCRIPTS / "fr13_fixed32_contract.py").read_text()
    assert "does not reconcile at any deployed " in source
    assert "FIXED32_DEPLOYED_MAX_OUTPUT_TOKENS" in source
    assert "per-request ceiling of" in source
    assert "per deployed ceiling" in source


def test_the_two_eras_are_distinguishable_at_realistic_counts() -> None:
    """The eras must not alias: a 24000 sum must never look like a 32768 sum."""
    for normal in range(1, 60):
        for compactions in range(0, 6):
            low = _reconcile(normal, compactions, 24_000)
            high = _reconcile(normal, compactions, 32_768)
            assert low != high, f"eras alias at normal={normal}"
            assert _solve(low, normal, compactions) == 24_000
            assert _solve(high, normal, compactions) == 32_768


def test_no_reconciliation_reads_the_legacy_constant_any_more() -> None:
    """The constant stays for importers, but must not drive the algebra.

    It is what stated the ceiling a second time; leaving it wired in anywhere
    that reconciles would leave site 25 half-fixed.
    """
    source = (SCRIPTS / "fr13_fixed32_contract.py").read_text()
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "QWEN_VISIBLE_MAX_OUTPUT_TOKENS" not in stripped:
            continue
        assert stripped.startswith("QWEN_VISIBLE_MAX_OUTPUT_TOKENS ="), (
            f"the legacy constant is still read by the algebra: {stripped}"
        )


def test_every_deployed_ceiling_clears_the_compaction_bucket() -> None:
    """The latent hazard the 24000 curve nearly walked into.

    Compactions are counted from the le_20000 histogram bucket, so a visible
    ceiling at or below 20000 would drop normal requests into the compaction
    bucket and mis-split them silently. The published tradeoff curve offered
    12000 and 16000; either would have landed here.
    """
    for ceiling in contract.FIXED32_DEPLOYED_MAX_OUTPUT_TOKENS:
        assert ceiling > contract.QWEN_COMPACTION_MAX_OUTPUT_TOKENS, (
            f"ceiling {ceiling} collides with the compaction bucket"
        )
    source = (SCRIPTS / "fr13_fixed32_contract.py").read_text()
    assert "must exceed the compaction cap" in source, (
        "the import-time guard is gone; a future tightening would mis-count"
    )


def test_the_evidence_records_the_served_ceiling_not_the_constant() -> None:
    """An artifact must not claim an era it was not served in."""
    source = (SCRIPTS / "fr13_fixed32_contract.py").read_text()
    assert '"normal_visible_max_output_tokens": served_max_output_tokens,' in source
    assert '"normal_visible_max_output_tokens": campaign_max_output_tokens,' in source


def test_the_client_reservation_is_documented_as_a_separate_knob() -> None:
    """QWEN_CODE_MAX_OUTPUT_TOKENS: reconciled, deliberately not lowered.

    It sets qwen-code's output reservation and hence its contextLimit. Matching
    it to 24000 would give the agent ~7k more context than every banked arm,
    which is a behaviour change to the agent's planning budget rather than a
    bookkeeping fix -- and it is not needed to reconcile, because the contract
    reads vLLM's POST-proxy max_tokens.
    """
    runner = (SCRIPTS / "run_swe_bench_q36_a.py").read_text()
    assert "QWEN_CODE_MAX_OUTPUT_TOKENS=32768" in runner
    assert "DELIBERATELY NOT LOWERED TO 24000" in runner
    assert "SEPARATE KNOB" in runner


def test_the_prose_no_longer_teaches_the_old_algebra() -> None:
    """Site 21's lesson: prose follows code."""
    source = (SCRIPTS / "fr13_fixed32_contract.py").read_text()
    for stale in ("32768/20000", "ordinary 32768", "32768 visible"):
        assert stale not in source, f"a comment still teaches {stale!r}"

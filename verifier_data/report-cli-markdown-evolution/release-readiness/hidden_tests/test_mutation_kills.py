"""Layer 6 — mutation-kill tests.

Added after the v0.2 mutation run surfaced 6 surviving mutations. These
tests directly assert on behaviors that were untested by layers 1–5,
closing the coverage gap.

Each test names the mutation ID(s) it kills in a comment so future
reviewers can trace the provenance.
"""
from __future__ import annotations

import pytest

from release_readiness.core.formatting import (
    format_count,
    join_with_commas,
    pad_cell,
)
from release_readiness.renderers.json_renderer import JsonRenderer
from release_readiness.renderers.registry import (
    RendererRegistry,
    get_registry,
    reset_registry_for_testing,
)


# ============================================================================
# join_with_commas — kills jwc_mut05 and jwc_mut06
# ============================================================================

def test_join_with_commas_oxford_comma_for_three() -> None:
    """Three items must use Oxford comma + 'and': 'a, b, and c'.

    Kills mutation: jwc_mut05_drop_oxford (which removes the Oxford comma).
    """
    assert join_with_commas(["a", "b", "c"]) == "a, b, and c"


def test_join_with_commas_oxford_comma_for_four() -> None:
    """Four items: 'a, b, c, and d' — confirms Oxford pattern extends."""
    assert join_with_commas(["a", "b", "c", "d"]) == "a, b, c, and d"


def test_join_with_commas_two_items_uses_and_not_comma() -> None:
    """Two items must join with ' and ', not a comma.

    Kills mutation: jwc_mut06_two_to_comma (which uses comma for 2 items).
    """
    result = join_with_commas(["alpha", "beta"])
    assert result == "alpha and beta"
    assert "," not in result


def test_join_with_commas_empty_and_single() -> None:
    """Empty list returns empty string; single item returns itself unchanged."""
    assert join_with_commas([]) == ""
    assert join_with_commas(["only"]) == "only"


# ============================================================================
# Registry — kills rr_mut14, rr_mut15, rr_mut16, rr_mut17
# ============================================================================

def test_reset_registry_actually_resets_singleton() -> None:
    """reset_registry_for_testing() must force re-discovery on next get_registry() call.

    Kills mutation: rr_mut14_never_reset_singleton (makes reset a no-op).
    """
    r1 = get_registry()
    reset_registry_for_testing()
    r2 = get_registry()
    assert r1 is not r2, "reset_registry_for_testing did not create a new singleton"


def test_available_formats_is_sorted_alphabetically() -> None:
    """available_formats() must return a sorted list.

    Kills mutation: rr_mut15_wrong_available_formats_sort (which returns
    dict-insertion order instead of sorted).
    """
    registry = RendererRegistry()
    # Insert in non-alphabetical order
    registry.register("zeta", JsonRenderer)
    registry.register("alpha", JsonRenderer)
    registry.register("mid", JsonRenderer)
    # Must be sorted regardless of insertion order
    formats = registry.available_formats()
    assert formats == sorted(formats), f"formats not sorted: {formats}"
    # And specifically, alpha before mid before zeta
    assert formats.index("alpha") < formats.index("mid") < formats.index("zeta")


def test_register_actually_adds_to_registry() -> None:
    """register() must mutate the registry so the format becomes retrievable.

    Kills mutation: rr_mut16_register_no_op (which makes register() a no-op).
    """
    class _Sentinel:
        def render(self, report):  # type: ignore[no-untyped-def]
            return "sentinel"

    registry = RendererRegistry()
    assert "sentinel_fmt" not in registry.available_formats()
    registry.register("sentinel_fmt", _Sentinel)  # type: ignore[arg-type]
    assert "sentinel_fmt" in registry.available_formats()
    instance = registry.get("sentinel_fmt")
    assert instance.render(None) == "sentinel"


def test_get_unknown_format_raises_value_error() -> None:
    """Registry.get() for an unknown format must raise ValueError, not
    return None or any other fallback.

    Kills mutation: rr_mut17_unknown_format_returns_none (which returns None
    silently for unknown formats).
    """
    registry = RendererRegistry()
    with pytest.raises(ValueError, match="unknown format"):
        registry.get("definitely_not_a_real_format_xyz")


def test_get_unknown_format_error_lists_available_formats() -> None:
    """The ValueError message must name the available formats so the user
    can diagnose the typo. Hardens the error path against further
    regression."""
    registry = RendererRegistry()
    registry.register("json", JsonRenderer)
    with pytest.raises(ValueError) as excinfo:
        registry.get("xml")
    assert "json" in str(excinfo.value)


# ============================================================================
# format_count — extra assertions to harden the core trap-catcher
# ============================================================================

def test_format_count_exhaustive_small_integers() -> None:
    """Exhaustive check over the small-integer regime. Kills any mutation
    that breaks the clean singular-for-1 / plural-otherwise invariant."""
    for n in range(0, 10):
        result = format_count(n, singular="owner", plural="owners")
        if n == 1:
            assert result == "1 owner", f"n={n} expected '1 owner', got {result!r}"
        else:
            assert result == f"{n} owners", f"n={n} expected {n} owners, got {result!r}"


def test_format_count_singular_and_plural_are_distinct_strings() -> None:
    """Confirm the renderer uses the argument values, not hardcoded strings.
    Passing distinct singular/plural words must produce distinct outputs."""
    assert format_count(1, singular="foo", plural="bars") == "1 foo"
    assert format_count(0, singular="foo", plural="bars") == "0 bars"
    assert format_count(2, singular="foo", plural="bars") == "2 bars"


# ============================================================================
# pad_cell — tighten the bounds to catch off-by-one mutations
# ============================================================================

def test_pad_cell_output_length_equals_width() -> None:
    """When padding is needed, the output length must exactly equal the
    requested width. Catches mutations that use the wrong padding count."""
    for input_text, width in [("a", 5), ("hi", 10), ("", 7)]:
        assert len(pad_cell(input_text, width)) == width


def test_pad_cell_does_not_truncate_long_text() -> None:
    """Text longer than width is returned unchanged. Catches mutations
    that inadvertently truncate."""
    long_text = "this-is-longer-than-the-width"
    assert pad_cell(long_text, 5) == long_text

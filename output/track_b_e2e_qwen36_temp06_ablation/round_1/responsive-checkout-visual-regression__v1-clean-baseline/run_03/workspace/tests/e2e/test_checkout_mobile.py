from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_mobile_compact_summary_does_not_overlap_payment():
    """The compact summary must not use negative margin that overlaps the payment form."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # The mobile compact block must not contain a negative margin-bottom on the summary
    assert "margin-bottom: -64px" not in css


def test_mobile_shell_has_bottom_padding_for_fixed_bar():
    """The shell must have enough bottom padding so the fixed mobile bar doesn't
    cover the primary CTA."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    assert "padding-bottom: 112px" in css


def test_mobile_bar_has_higher_z_index():
    """The fixed mobile bar must sit above page content."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # Inside the mobile media query the bar should have z-index >= 10
    found = False
    in_mobile_block = False
    for line in css.splitlines():
        if "@media (max-width: 600px)" in line:
            in_mobile_block = True
        if in_mobile_block and "z-index: 10" in line:
            found = True
            break
    assert found, "Mobile bar should have z-index: 10"

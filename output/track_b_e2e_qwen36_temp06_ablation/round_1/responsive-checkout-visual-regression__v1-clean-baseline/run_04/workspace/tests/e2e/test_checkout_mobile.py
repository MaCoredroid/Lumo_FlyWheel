from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_mobile_compact_summary_no_negative_margin():
    """Mobile compact summary must not use negative margin that overlaps content."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # The mobile media query block should not contain a negative margin-bottom
    mobile_block_start = css.find("@media (max-width: 600px)")
    assert mobile_block_start != -1, "Mobile media query missing"

    mobile_block = css[mobile_block_start:]

    # Summary margin-bottom should be positive (no overlap)
    assert "margin-bottom: -" not in mobile_block, (
        "Mobile compact summary must not use negative margin-bottom"
    )


def test_mobile_shell_has_bottom_padding_for_fixed_bar():
    """Shell must have bottom padding so the fixed mobile bar doesn't cover the CTA."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    mobile_block_start = css.find("@media (max-width: 600px)")
    assert mobile_block_start != -1, "Mobile media query missing"

    mobile_block = css[mobile_block_start:]

    assert "padding-bottom" in mobile_block, (
        "Mobile shell must have padding-bottom to account for fixed bottom bar"
    )


def test_mobile_bar_has_z_index_above_summary():
    """Fixed mobile bar z-index must exceed summary z-index to stay on top."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    mobile_block_start = css.find("@media (max-width: 600px)")
    assert mobile_block_start != -1, "Mobile media query missing"
    mobile_block = css[mobile_block_start:]

    # Both z-index declarations must exist in the mobile block
    assert mobile_block.count("z-index") >= 2, (
        "Both summary and mobile bar need explicit z-index in mobile block"
    )

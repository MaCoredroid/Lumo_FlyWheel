from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_mobile_checkout_has_bottom_padding():
    """Verify mobile viewport has padding to prevent overlap with fixed bottom bar."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # Mobile compact mode should have bottom padding for the fixed bar
    assert "padding: 16px 16px 100px" in css


def test_mobile_checkout_summary_sticky_without_negative_margin():
    """Verify mobile summary is sticky but doesn't use negative margin causing overlap."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # Summary should be sticky on mobile
    assert "position: sticky" in css
    # Should not have negative margin that causes overlap
    assert "margin-bottom: 0" in css


def test_mobile_checkout_bar_has_zindex():
    """Verify mobile bottom bar has proper z-index for stacking."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # Mobile bar should have z-index above content
    assert "z-index: 3" in css

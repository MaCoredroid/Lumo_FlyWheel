from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_mobile_checkout_prevents_overlap():
    """Verify mobile viewport CSS prevents overlap between mobile bar and payment form."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # Mobile bar should have higher z-index to appear above content
    assert "z-index: 10" in css

    # Payment section needs bottom padding to prevent overlap with fixed mobile bar
    assert "padding-bottom: 80px" in css

    # Summary should not have negative margin that causes overlap
    assert "margin-bottom: -64px" not in css
    assert 'margin-bottom: 0;' in css


def test_mobile_checkout_compact_summary_enabled():
    """Verify compact summary experiment is still enabled on mobile."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # Compact summary data attribute should still be used
    assert '[data-preview-compact="true"]' in css

    # Mobile bar should be hidden by default and shown only in compact mode
    assert ".checkout-mobile-bar" in css
    assert "display: none" in css


def test_mobile_checkout_uses_single_column():
    """Verify mobile viewport switches to single column layout."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # Mobile should use block display instead of grid
    assert "@media (max-width: 600px)" in css
    assert "display: block" in css

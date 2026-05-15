from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_mobile_checkout_no_overlap_with_compact_summary():
    """Verify mobile viewport doesn't have overlapping elements when compact summary is enabled."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # Ensure padding-bottom is set on payment section to prevent overlap with fixed bottom bar
    assert "padding-bottom: 100px" in css

    # Ensure mobile bar has proper z-index to stay fixed positioning
    assert "position: fixed" in css
    assert ".checkout-mobile-bar" in css

    # Ensure compact summary mode is still active (not disabled)
    assert 'data-preview-compact="true"' in css


def test_mobile_checkout_cta_accessible():
    """Verify the primary CTA is not hidden by overlapping elements on mobile."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # The negative margin that caused overlap should be removed
    assert "margin-bottom: -64px" not in css

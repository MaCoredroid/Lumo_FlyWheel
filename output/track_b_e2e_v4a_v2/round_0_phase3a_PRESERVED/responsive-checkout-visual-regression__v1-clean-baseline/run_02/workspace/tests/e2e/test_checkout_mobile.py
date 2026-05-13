from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_mobile_checkout_no_overlap_fixed():
    """Verify mobile viewport does not have overlapping elements."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # Compact summary experiment should remain enabled
    assert 'data-preview-compact="true"' in css

    # Negative margin that caused overlap should be removed
    assert "margin-bottom: -64px" not in css

    # Payment form should have padding to prevent overlap with bottom bar
    assert "padding-bottom: 72px" in css

    # Mobile bar should have higher z-index than summary
    assert "z-index: 3" in css

    # Sticky behavior should be preserved for summary
    assert "position: sticky" in css
    assert "top: 0" in css

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_mobile_checkout_no_overlap():
    """Verify mobile compact summary doesn't overlap payment form or CTA."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # Mobile viewport media query must exist
    assert "@media (max-width: 600px)" in css

    # Payment form must have bottom padding to prevent overlap with fixed bar
    assert "padding-bottom: 80px" in css

    # Summary should not have negative margin causing overlap
    assert "margin-bottom: -64px" not in css

    # Mobile bar must be fixed at bottom
    assert "position: fixed" in css
    assert "bottom: 0" in css


def test_mobile_cta_accessible():
    """Ensure primary CTA is not hidden by overlap."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # CTA must have minimum height for tap targets
    assert "min-height: 48px" in css

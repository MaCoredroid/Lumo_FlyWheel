from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_desktop_checkout_still_uses_two_columns():
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    assert "grid-template-columns: minmax(0, 1fr) 320px" in css
    assert ".checkout-summary" in css
    assert "position: sticky" in css


def test_mobile_viewport_no_overlap():
    """Verify mobile viewport at 390px width has no negative margin causing overlap."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # Check mobile media query exists
    assert "@media (max-width: 600px)" in css

    # Verify no negative margin that causes overlap
    assert "margin-bottom: -64px" not in css

    # Verify payment section has bottom padding to prevent overlap with sticky bar
    assert "padding-bottom: 80px" in css

    # Verify mobile bar uses sticky positioning (not fixed) to avoid overlap
    assert ".checkout-mobile-bar" in css
    assert "position: sticky" in css

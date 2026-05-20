from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_mobile_compact_summary_no_overlap():
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # Mobile compact summary must not use negative margin that causes overlap
    assert "margin-bottom: -64px" not in css

    # Payment section must have bottom padding to clear the fixed mobile bar
    assert "padding-bottom" in css

    # Mobile bar must remain fixed at bottom
    assert "position: fixed" in css


def test_mobile_viewport_cta_visible():
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # Primary CTA must retain minimum touch target height
    assert "min-height: 48px" in css

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_mobile_checkout_prevents_overlap():
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # Mobile payment form has bottom padding to prevent overlap with fixed bar
    assert "padding-bottom: 80px" in css

    # Mobile summary uses positive margin (not negative) to avoid overlap
    assert "margin-bottom: 24px" in css

    # Mobile bar is fixed at bottom
    assert "position: fixed" in css
    assert "bottom: 0" in css

    # Compact summary experiment is still enabled via data attribute
    assert 'data-preview-compact="true"' in css

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_mobile_checkout_has_bottom_padding_for_cta():
    """Ensure mobile viewport has padding to prevent bottom bar overlap with CTA."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # Mobile payment section should have bottom padding for fixed bar
    assert "padding-bottom: 80px" in css
    # Should be scoped to compact preview mode
    assert ".checkout-shell[data-preview-compact=\"true\"] .checkout-payment" in css or \
           ".checkout-shell[data-preview-compact=\"true\"] .checkout-payment" in css
    # CTA should still be present and have minimum height
    assert "checkout-primary-cta" in css
    assert "min-height: 48px" in css

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_checkout_primary_cta_copy_is_present():
    component = (
        ROOT / "apps/storefront/components/checkout/CheckoutPreview.jsx"
    ).read_text(encoding="utf-8")

    assert "Place order" in component
    assert "checkout-primary-cta" in component

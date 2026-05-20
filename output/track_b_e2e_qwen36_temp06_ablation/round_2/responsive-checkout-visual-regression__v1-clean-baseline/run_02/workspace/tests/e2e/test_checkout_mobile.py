from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_mobile_compact_summary_no_overlap():
    """Compact summary must not use negative margin that overlaps payment form."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    # The mobile compact summary block should exist
    assert '.checkout-shell[data-preview-compact="true"] .checkout-summary' in css

    # Negative negative margin-bottom on the compact summary causes overlap with
    # the payment form on mobile 390x844 viewports.
    lines = css.splitlines()
    in_mobile_summary_block = False
    for line in lines:
        if '.checkout-shell[data-preview-compact="true"] .checkout-summary' in line:
            in_mobile_summary_block = True
        elif in_mobile_summary_block and '}' in line:
            in_mobile_summary_block = False
        if in_mobile_summary_block and 'margin-bottom' in line:
            assert '-64px' not in line, "Negative margin-bottom causes mobile overlap"


def test_mobile_bottom_bar_clears_cta():
    """The shell padding-bottom must accommodate the fixed mobile bottom bar so the CTA is not hidden."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    assert 'padding-bottom: 80px' in css, "Shell needs bottom padding to clear the fixed mobile bar"


def test_mobile_bar_is_fixed():
    """The mobile bar should be position: fixed at the bottom."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    assert '.checkout-mobile-bar' in css
    assert 'position: fixed' in css
    assert 'bottom: 0' in css

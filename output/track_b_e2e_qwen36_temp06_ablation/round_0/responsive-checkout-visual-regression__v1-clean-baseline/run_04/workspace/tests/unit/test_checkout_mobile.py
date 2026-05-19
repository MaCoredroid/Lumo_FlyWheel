from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _parse_px(value: str) -> int:
    return int(value.strip().rstrip("px"))


def test_mobile_summary_no_negative_margin():
    """Compact summary must not use negative margin on mobile (causes overlap)."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    lines = css.splitlines()
    in_mobile_block = False
    for line in lines:
        if "@media (max-width: 600px)" in line:
            in_mobile_block = True
            continue
        if in_mobile_block:
            if line.strip().startswith("}"):
                in_mobile_block = False
                continue
            assert "margin-bottom: -64px" not in line, (
                "Mobile compact summary must not use negative margin-bottom"
            )


def test_mobile_shell_has_bottom_padding_for_fixed_bar():
    """Mobile shell must have bottom padding to clear the fixed mobile bar."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    lines = css.splitlines()
    in_mobile_block = False
    in_shell_rule = False
    found_bottom_padding = False

    for line in lines:
        if "@media (max-width: 600px)" in line:
            in_mobile_block = True
            continue
        if in_mobile_block:
            if ".checkout-shell" in line and "data-preview-compact" in line:
                in_shell_rule = True
            if in_shell_rule and "padding:" in line:
                parts = line.split("padding:")
                if len(parts) > 1:
                    values = parts[1].strip().rstrip(";").split()
                    # padding shorthand: top right bottom left (4 values)
                    if len(values) >= 4:
                        bottom_val = _parse_px(values[2])
                    else:
                        bottom_val = _parse_px(values[-1])
                    if bottom_val >= 64:
                        found_bottom_padding = True
            if in_shell_rule and line.strip() == "}":
                in_shell_rule = False
            if line.strip().startswith("}") and not in_shell_rule:
                in_mobile_block = False

    assert found_bottom_padding, (
        "Mobile compact shell must have sufficient bottom padding to clear fixed bar"
    )


def test_mobile_payment_has_clear_space():
    """Payment section on mobile must have bottom padding so CTA isn't hidden."""
    css = (ROOT / "apps/storefront/styles/checkout.css").read_text(encoding="utf-8")

    assert ".checkout-payment" in css
    assert "padding-bottom" in css, (
        "Payment section should have padding-bottom on mobile to clear fixed bar"
    )

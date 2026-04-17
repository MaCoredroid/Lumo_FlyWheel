from __future__ import annotations

import pytest

from norm_app.rules_v2 import build_rule_plan


def test_slug_normalization_preserves_digits_when_symbols_split_words() -> None:
    plan = build_rule_plan("Invoice 2024 / Retry 7", "finance", "eu-west")

    assert plan.slug == "invoice-2024-retry-7"


@pytest.mark.parametrize(
    ("title", "expected_slug"),
    [
        ("...Invoice Retry...", "invoice-retry"),
        ("__Refund Hold__", "refund-hold"),
        ("///Ledger Close///", "ledger-close"),
    ],
)
def test_slug_normalization_strips_boundary_punctuation(title: str, expected_slug: str) -> None:
    plan = build_rule_plan(title, "finance", "eu-west")

    assert plan.slug == expected_slug


@pytest.mark.parametrize(
    "title",
    [
        "Refund / / Retry",
        "Refund___Retry",
        "Refund---Retry",
        "Refund _ / --- Retry",
    ],
)
def test_slug_normalization_collapses_repeated_mixed_separators(title: str) -> None:
    plan = build_rule_plan(title, "finance", "eu-west")

    assert plan.slug == "refund-retry"
    assert "--" not in plan.slug

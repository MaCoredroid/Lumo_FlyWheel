"""Layer 5 — follow-up coverage for the dormant watchlist path."""
from __future__ import annotations

from report_app.fixtures import KNOWN_OWNERS
from report_app.service import build_owner_summary

from conftest import cli_markdown


def test_service_can_include_watchlist_owners_when_requested() -> None:
    summary = build_owner_summary(
        [{"owner": "Mae", "label": "blocked-picks", "count": 4}],
        include_known_owners=True,
    )

    owner_totals = {entry["owner"]: entry["count"] for entry in summary["owner_totals"]}
    assert set(owner_totals) == set(KNOWN_OWNERS)
    assert owner_totals["Mae"] == 4
    assert owner_totals["Ivy"] == 0
    assert owner_totals["Jules"] == 0


def test_cli_markdown_keeps_watchlist_visible_for_empty_queue(monkeypatch) -> None:
    output = cli_markdown(monkeypatch, [])

    assert "0 sections covering 0 queued items." in output
    assert "Top owner: Ivy with 0 queued items." in output
    for owner in KNOWN_OWNERS:
        assert f"| {owner}" in output


def test_zero_item_wording_uses_plural_when_watchlist_is_visible(monkeypatch) -> None:
    output = cli_markdown(monkeypatch, [])

    assert "0 queued items" in output
    assert "0 queued item." not in output

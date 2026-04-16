from __future__ import annotations

from ci_app.cli import main


def test_service_contract_still_returns_required_labels() -> None:
    output = main()
    assert "queue-check" in output
    assert "ledger-check" in output

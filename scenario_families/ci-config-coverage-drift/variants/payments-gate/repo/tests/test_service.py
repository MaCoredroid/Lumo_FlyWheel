from __future__ import annotations

from ci_app.cli import main
from ci_app.service import gate_snapshot, run_checks


def test_service_contract_still_returns_required_labels() -> None:
    assert run_checks() == ["queue-check", "ledger-check"]

    output = main()
    assert "queue-check" in output
    assert "ledger-check" in output


def test_gate_snapshot_keeps_receipt_audit_optional() -> None:
    assert gate_snapshot() == {
        "required": ["queue-check", "ledger-check"],
        "optional": ["receipt__audit"],
    }

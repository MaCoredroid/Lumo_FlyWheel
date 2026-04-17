from __future__ import annotations

from ci_app.cli import main
from ci_app.service import run_checks


def test_service_contract_still_returns_required_labels() -> None:
    assert run_checks() == ["schema-check", "render-check"]

    output = main()
    assert "schema-check" in output
    assert "render-check" in output
    assert "inventory-schema-check-report" in output

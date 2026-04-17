from __future__ import annotations

from pathlib import Path


def test_preview_contract_documents_dispatch_identity() -> None:
    text = Path("docs/preview_contract.md").read_text(encoding="utf-8")

    assert "dispatch_key" in text
    assert "?dispatch=" in text
    assert "canonical dispatch identity" in text
    assert "<region>:<owner>:<normalized-title-slug>" in text

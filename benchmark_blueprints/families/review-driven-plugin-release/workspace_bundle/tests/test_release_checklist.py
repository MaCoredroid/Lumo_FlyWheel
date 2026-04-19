from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.render_release_evidence import build_release_evidence


def test_release_docs_and_evidence_agree() -> None:
    checklist = (ROOT / "docs" / "release_checklist.md").read_text()
    notes = (ROOT / "docs" / "release_notes.md").read_text()
    evidence = build_release_evidence()

    assert "No UI changes" not in checklist
    assert "Connector fallback" in checklist
    assert "Connector fallback" in notes
    assert (
        evidence["screenshots"]["before"]
        == "artifacts/screenshots/settings-panel-before.png"
    )
    assert (
        evidence["screenshots"]["expected"]
        == "artifacts/screenshots/settings-panel-expected.png"
    )
    assert evidence["review"]["actionable_thread_ids"] == ["T-201", "T-202", "S-9"]


def test_review_reply_addresses_only_actionable_items() -> None:
    review_reply = json.loads((ROOT / "artifacts" / "review_reply.json").read_text())
    reply_ids = [item["id"] for item in review_reply["resolved_threads"]]

    assert reply_ids == ["T-201", "T-202", "S-9"]
    assert "T-100" not in reply_ids


def test_review_response_mentions_real_changed_surfaces() -> None:
    response = (ROOT / "review" / "review_response.md").read_text()

    assert "T-201" in response
    assert "T-202" in response
    assert "S-9" in response
    assert "T-100" not in response
    assert ".codex-plugin/plugin.json" in response
    assert "drive_brief/ui/settings_panel.tsx" in response

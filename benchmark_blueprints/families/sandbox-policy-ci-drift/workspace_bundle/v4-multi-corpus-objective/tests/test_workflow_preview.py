
from __future__ import annotations

from scripts.render_workflow_preview import render_preview
from tests.helpers import fixture_path


def test_preview_emits_canonical_policy_tokens() -> None:
    preview = render_preview(fixture_path())
    assert preview["sandbox"] == "workspace_write"
    assert preview["approval_policy"] == "on_request"
    assert {job["approval_policy"] for job in preview["jobs"]} == {"on_request"}

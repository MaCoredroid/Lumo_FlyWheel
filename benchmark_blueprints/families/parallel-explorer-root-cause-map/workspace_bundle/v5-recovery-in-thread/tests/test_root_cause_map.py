from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from release_readiness.adapters.env_source import load_env_watchlist
from release_readiness.adapters.fs_source import load_schedule_blockers
from release_readiness.core.aggregation import merge_blocked_owner_rows
from release_readiness.renderers.markdown_renderer import render_blocked_owner_section


def test_scheduler_aliases_collapse_before_rendering() -> None:
    schedule_rows = [
        {"owner": "Team Ops", "blocked_count": 1, "reason": "scheduler_refactor"},
        {"owner": "Platform Infra", "blocked_count": 1, "reason": "scheduler_refactor"},
    ]
    env_watchlist = "team-ops, platform-infra"
    summary = merge_blocked_owner_rows(
        load_schedule_blockers(schedule_rows),
        load_env_watchlist(env_watchlist),
    )

    assert summary["blocked_owner_total"] == 2, "scheduler aliases should collapse to the same blocked owner"
    rendered = render_blocked_owner_section(summary)
    assert "owner_key: team ops" not in rendered or "owner_key: team-ops" not in rendered

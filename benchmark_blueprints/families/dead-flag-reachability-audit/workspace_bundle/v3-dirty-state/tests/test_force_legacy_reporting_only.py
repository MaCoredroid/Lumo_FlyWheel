from preview.config import load_preview_env
from preview.legacy import legacy_force_label
from preview.service import build_preview_plan


def test_force_legacy_flag_is_tracked_for_reporting_only():
    config = load_preview_env({"PREVIEW_FORCE_LEGACY": "1"})
    plan = build_preview_plan(config)
    assert config.parser_hits["PREVIEW_FORCE_LEGACY"] == "load_preview_env:PREVIEW_FORCE_LEGACY"
    assert plan["branch"] == "preview_runtime_branch:legacy_preview_path"
    assert legacy_force_label(config.force_legacy_seen) == "legacy-forced"

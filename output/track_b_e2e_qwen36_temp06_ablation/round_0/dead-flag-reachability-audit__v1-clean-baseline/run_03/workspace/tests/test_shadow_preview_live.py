from preview.config import load_preview_env
from preview.service import build_preview_plan


def test_shadow_preview_changes_runtime_branch():
    config = load_preview_env({"ENABLE_SHADOW_PREVIEW": "1"})
    plan = build_preview_plan(config)
    assert config.parser_hits["ENABLE_SHADOW_PREVIEW"] == "load_preview_env:ENABLE_SHADOW_PREVIEW"
    assert plan["branch"] == "preview_runtime_branch:shadow_preview_path"

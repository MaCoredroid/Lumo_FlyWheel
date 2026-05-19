from preview.config import load_preview_env
from preview.service import build_preview_plan


def test_preview_v2_alias_maps_to_shadow_path():
    config = load_preview_env({"ENABLE_PREVIEW_V2": "1"})
    plan = build_preview_plan(config)
    assert config.parser_hits["ENABLE_PREVIEW_V2"] == "load_preview_env:ENABLE_PREVIEW_V2->ENABLE_SHADOW_PREVIEW"
    assert plan["branch"] == "preview_runtime_branch:shadow_preview_path"

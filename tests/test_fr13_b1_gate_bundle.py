from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
TAW = REPO / "scripts" / "fr13_device_multidraft_kernel.py"
MANIFEST = (
    REPO
    / "results"
    / "fr13_fixed32_b1_gate_bundle_source_20260731"
    / "manifest.json"
)


def test_taw_gate_is_strict_default_off_and_forwarded_by_launcher() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    taw = TAW.read_text(encoding="utf-8")

    assert (
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE="
        "${FR13_FIXED32_TAW_NATIVE_PRECOMPUTE:-0}"
    ) in launcher
    assert 'case "$FR13_FIXED32_TAW_NATIVE_PRECOMPUTE" in' in launcher
    assert "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE must be 0 or 1" in launcher
    assert (
        '-e FR13_FIXED32_TAW_NATIVE_PRECOMPUTE='
        '"$FR13_FIXED32_TAW_NATIVE_PRECOMPUTE" \\'
    ) in launcher
    assert 'os.environ.get("FR13_FIXED32_TAW_NATIVE_PRECOMPUTE", "")' in taw
    assert "probability_mismatches=0" in taw
    assert "product_mismatches=0 reference_returned=1" in taw


def test_live_dfwd_and_taw_diagnostics_return_reference_values() -> None:
    patcher = PATCHER.read_text(encoding="utf-8")
    taw = TAW.read_text(encoding="utf-8")

    assert '"FR13_DRAFT_HEAD_PAD_ROWS", "0"' in patcher
    assert '"FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB", "0"' in patcher
    assert "for _fr13_dh_i, _fr13_dh_r in enumerate(" in patcher
    assert "(32, 64, 128)" in patcher
    assert "_logits = _fr13_dh_reference" in patcher
    assert "comparison_probability_caches=probability_caches" in taw
    assert ") = reference\n    else:" in taw


def test_manifest_does_not_promote_replay_to_a_live_qrow_gate() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["default_behavior_changed"] is False
    assert manifest["gpu_used"] is False
    assert manifest["fa2_candidate_built"] is False
    assert manifest["live_reference_gate"]["instance_id"] == (
        "astropy__astropy-12907"
    )
    assert manifest["live_reference_gate"]["positive_token_traffic"] == (
        "one real task only"
    )
    assert manifest["qrow_live_gate"]["ready"] is False
    assert manifest["qrow_live_gate"]["required_process_scope"] == (
        "same EngineCore process and CUDA boot"
    )
    assert "dense-to-paged repacking" in manifest["qrow_live_gate"][
        "forbidden_substitutes"
    ]
    assert manifest["combined_three_gate_live_run"] == {
        "ready": False,
        "blocking_component": "qrow_live_gate",
    }

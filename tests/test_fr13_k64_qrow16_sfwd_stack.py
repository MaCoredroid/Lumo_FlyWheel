from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "fr13_run_b1_k64_qrow16_sfwd_stack_timing.sh"
LAUNCHER = ROOT / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
SERVE = ROOT / "scripts" / "fr13_bigdenom_swe_serve_variant.sh"
QROW_PATCHER = ROOT / "scripts" / "fr13_patch_fa2_tree_bias.py"
GDN_PATCHER = ROOT / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"


def _literal_string_assignment(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            value = ast.literal_eval(node.value)
            assert isinstance(value, str)
            return value
    raise AssertionError(f"missing string assignment {name}")


def test_qrow16_generated_helper_has_exact_k64_eager_bridge() -> None:
    helper = _literal_string_assignment(
        QROW_PATCHER, "FIXED32_QUERY_TILE16_PRODUCTION_HELPERS"
    )
    ast.parse(helper)
    assert '"FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION", "0"' in helper
    assert 'os.environ.get("ENFORCE_EAGER", "0") == "1"' in helper
    assert 'os.environ.get("FR13_DRAFT_VOCAB_ROOT", "0") == "1"' in helper
    assert 'os.environ.get("FR13_DRAFT_VOCAB_K", "") == "65536"' in helper
    assert "fa2_qrow16_eager_production_engagement.v1" in helper
    assert '"runtime_mode": "EAGER"' in helper
    assert '"sfwd_state_fusion_production": True' in helper
    assert "if not capturing and not eager_sfwd_stack" in helper


def test_runner_is_real_exact4_b1_k64_physical32_only() -> None:
    runner = RUNNER.read_text(encoding="ascii")
    assert "subset_b4_four.json" in runner
    assert "task_count=4" in runner
    assert "MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1" in runner
    assert "FR13_FIXED32_B1_DIAGNOSTIC=0" in runner
    assert "FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536" in runner
    assert "FR13_FA2_QROW16_PRODUCTION=1" in runner
    assert "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=1" in runner
    assert '"$ARM" hydra27_fixed32 "$SUBSET"' in runner
    assert "physical_rows=32" in runner
    assert "logical_drafts=27" in runner
    assert "valid_mask=0x7abdffff" in runner
    assert "--expected-tok-per-draft 31" in runner
    assert 'RUNROOT_REL=${RUNROOT_ABS#"$REPO/"}' in runner
    assert '"/workspace/$RUNROOT_REL/sidecars/' in runner
    for forbidden in ("PROBE_ONLY", "ACCEPT_SPEED_PROBE", "synthetic task"):
        assert forbidden not in runner


def test_launchers_fail_closed_around_the_integrated_stack() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    serve = SERVE.read_text(encoding="utf-8")
    gdn = GDN_PATCHER.read_text(encoding="utf-8")
    for mode in ("tail6_fixed32", "hydra27_fixed32"):
        assert f'"{mode}"' in launcher
    for required in (
        '"FR13_DRAFT_VOCAB_ROOT": "1"',
        '"FR13_DRAFT_VOCAB_K": "65536"',
        '"FR13_FIXED32_CONV_SOURCE_BATCH": "0"',
        '"FR13_FA2_QROW16_PRODUCTION": "1"',
        '"FR13_FIXED32_CUTLASS_WAVE": "stock"',
    ):
        assert required in gdn
    assert "SFWD production requires exact K64 B1 eager fixed32" in launcher
    assert "SFWD production permits only the qualified qrow16 co-candidate" in launcher
    assert "SFWD production requires sequential real-task B1" in serve
    assert "_FR13_FIXED32_EAGER_KERNEL_DIAGNOSTIC = False" in gdn
    assert "fixed32_sfwd_state_fusion_production_engagement(" in gdn
    assert "if _fr13_sfwd_production is not None" in gdn

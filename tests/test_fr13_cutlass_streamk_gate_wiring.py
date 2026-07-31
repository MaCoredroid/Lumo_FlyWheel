from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
GATE = REPO / "scripts" / "fr13_run_b1_cutlass_streamk_live_gate.sh"


def test_launcher_is_digest_pinned_diagnostic_only_and_worker_visible() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "FR13_FIXED32_CUTLASS_WAVE=${FR13_FIXED32_CUTLASS_WAVE:-stock}" in launcher
    assert "CUTLASS Stream-K candidate is restricted to the fixed32 B1 diagnostic" in launcher
    assert "scripts/fr13_cutlass_wave_binary.py verify" in launcher
    assert '"$FR13_FIXED32_CUTLASS_WAVE_SO:/tmp/fr13_cutlass_wave.abi3.so:ro"' in launcher
    assert "scripts/fr13_cutlass_wave_binary.py install" in launcher
    assert "/usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so" in launcher
    assert "fr13_fixed32_cutlass_wave.selector" in launcher
    assert 'chmod 0444 "$LOG_DIR/fr13_fixed32_cutlass_wave.selector"' in launcher
    assert "_fixed32_expected_eager=1" in launcher


def test_real_b1_gate_disables_unrelated_candidates_and_requires_coverage() -> None:
    gate = GATE.read_text(encoding="utf-8")

    for assignment in (
        "FR13_GATE_QROW16=0",
        "FR13_GATE_TAW_NATIVE=0",
        "FR13_GATE_DRAFT_HEAD_PAD=0",
        "FR13_GATE_GDN_BV=0",
        "ENFORCE_EAGER=1",
    ):
        assert assignment in gate
    assert "streamk_coop128_byte_ab" in gate
    assert "scripts/fr13_run_b1_kernel_live_gate.sh" in gate
    assert "one real SWE-Verified B1 diagnostic task" in gate
    assert "not all five real projection shapes were exercised" in gate
    assert '"served_result": "stock"' in gate
    assert '"acceptance_valid": False' in gate

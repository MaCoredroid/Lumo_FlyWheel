from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
GATE = REPO / "scripts" / "fr13_run_b1_cutlass_streamk_live_gate.sh"
B1_KERNEL_GATE = REPO / "scripts" / "fr13_run_b1_kernel_live_gate.sh"
B4_GRAPH_GATE = REPO / "scripts" / "fr13_run_b4_gdn_wide_live_gate.sh"


def test_launcher_is_digest_pinned_diagnostic_only_and_worker_visible() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "FR13_FIXED32_CUTLASS_WAVE=${FR13_FIXED32_CUTLASS_WAVE:-stock}" in launcher
    assert (
        "CUTLASS Stream-K candidate is restricted to the fixed32 B1 diagnostic"
        in launcher
    )
    assert "scripts/fr13_cutlass_wave_binary.py verify" in launcher
    assert (
        '"$FR13_FIXED32_CUTLASS_WAVE_SO:/tmp/fr13_cutlass_wave.abi3.so:ro"' in launcher
    )
    assert "scripts/fr13_cutlass_wave_binary.py install" in launcher
    assert (
        "/usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so"
        in launcher
    )
    assert "fr13_fixed32_cutlass_wave.selector" in launcher
    assert 'chmod 0444 "$LOG_DIR/fr13_fixed32_cutlass_wave.selector"' in launcher
    assert "_fixed32_expected_eager=1" in launcher
    assert (
        "FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=${FR13_FIXED32_CUTLASS_WAVE_PRODUCTION:-0}"
        in launcher
    )
    assert (
        "CUTLASS Stream-K production requires fixed32 exact4/16 B1 and a pinned live PASS"
        in launcher
    )
    assert "scripts/fr13_cutlass_streamk_pass.py validate" in launcher
    assert "scripts/fr13_cutlass_streamk_pass.py issue" in launcher
    assert "--production-pass-sidecar" in launcher
    assert "--expected-production-pass-sha256" in launcher


def test_launcher_rejects_cutlass_bm8_composition_before_sidecar_or_docker() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")

    bm8_stock_guard = launcher.index(
        "FR13 DFWD unified BM8 production requires the stock CUTLASS wave"
    )
    cutlass_bm8_guard = launcher.index(
        "nonstock CUTLASS wave requires both BM8 selectors to be 0"
    )
    binary_preflight = launcher.index(
        ".venv/bin/python scripts/fr13_cutlass_wave_binary.py verify"
    )
    sidecar_issue = launcher.index("scripts/fr13_bm8_pass_sidecar.py issue")
    docker_run = launcher.index("docker run -d --pull=never")
    assert bm8_stock_guard < binary_preflight
    assert cutlass_bm8_guard < binary_preflight
    assert binary_preflight < sidecar_issue < docker_run


@pytest.mark.parametrize(
    ("bm8_live", "bm8_production", "message"),
    (
        ("1", "0", "nonstock CUTLASS wave requires both BM8 selectors to be 0"),
        ("0", "1", "BM8 production requires the stock CUTLASS wave"),
    ),
)
def test_launcher_cross_kernel_preflight_runs_before_sidecar_and_docker(
    tmp_path: Path,
    bm8_live: str,
    bm8_production: str,
    message: str,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_sentinel = tmp_path / "docker.called"
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        '#!/usr/bin/env bash\nprintf "called\\n" > "$DOCKER_SENTINEL"\n',
        encoding="ascii",
    )
    fake_docker.chmod(0o755)
    log_dir = tmp_path / "logs"
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("FR13_")
    }
    environment.update(
        {
            "DOCKER_SENTINEL": os.fspath(docker_sentinel),
            "FR13_DFWD_UNIFIED_BM8_LIVE_AB": bm8_live,
            "FR13_DFWD_UNIFIED_BM8_PRODUCTION": bm8_production,
            "FR13_FIXED32_CUTLASS_WAVE": "streamk_coop128",
            "FR13_FIXED32_CUTLASS_WAVE_SO": "",
            "LOG_DIR": os.fspath(log_dir),
            "PATH": os.pathsep.join((os.fspath(fake_bin), environment["PATH"])),
            "REPO": os.fspath(REPO),
        }
    )

    result = subprocess.run(
        ["bash", os.fspath(LAUNCHER)],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert not log_dir.exists()
    assert not docker_sentinel.exists()


def test_real_b1_gate_disables_unrelated_candidates_and_requires_coverage() -> None:
    gate = GATE.read_text(encoding="utf-8")
    kernel_gate = B1_KERNEL_GATE.read_text(encoding="utf-8")

    for assignment in (
        "FR13_GATE_QROW16=0",
        "FR13_GATE_TAW_NATIVE=0",
        "FR13_GATE_DRAFT_HEAD_PAD=0",
        "FR13_GATE_GDN_BV=0",
        "FR13_DFWD_UNIFIED_BM8_LIVE_AB=0",
        "FR13_DFWD_UNIFIED_BM8_PRODUCTION=0",
        "ENFORCE_EAGER=1",
    ):
        assert assignment in gate
    launch = gate.index("bash scripts/fr13_run_b1_kernel_live_gate.sh")
    assert gate.index("export FR13_DFWD_UNIFIED_BM8_LIVE_AB=0") < launch
    assert gate.index("export FR13_DFWD_UNIFIED_BM8_PRODUCTION=0") < launch
    assert "streamk_coop128_byte_ab" in gate
    assert "scripts/fr13_run_b1_kernel_live_gate.sh" in gate
    assert "CUTLASS Stream-K gate requires a fresh RUNROOT" in gate
    assert "one real SWE-Verified B1 diagnostic task" in gate
    assert "not all five real projection shapes were exercised" in gate
    assert "invalid differing-byte count" in gate
    assert "byte equality and differing-byte count disagree" in gate
    assert "installed binary attestation schema mismatch" in gate
    assert "binary.CONTAINER_SOURCE" in gate
    assert "binary.CONTAINER_DESTINATION" in gate
    assert '"served_result": "stock"' in gate
    assert '"acceptance_valid": False' in gate
    assert '"schema": "fr13.fixed32.cutlass_streamk_live_gate.v2"' in gate
    assert '"patch_source_sha256": patch_source_sha256' in gate
    assert '"binary_attestation_sha256"' in gate
    sequence = kernel_gate.index(
        "source scripts/fr13_fixed32_floor_timers_seq.sh"
    )
    eager_rearm = kernel_gate.index(
        'FR13_FIXED32_CUTLASS_WAVE:-stock}" '
        '== "streamk_coop128_byte_ab"'
    )
    launch = kernel_gate.index(
        "bash scripts/fr13_bigdenom_swe_serve_variant.sh"
    )
    assert sequence < eager_rearm < launch


def test_b4_graph_gate_pins_cutlass_stock_and_bm8_off() -> None:
    gate = B4_GRAPH_GATE.read_text(encoding="utf-8")
    launch_start = gate.index("if OFFLOAD_AGENT=1")
    launch_end = gate.index(
        "bash scripts/fr13_bigdenom_swe_serve_variant.sh", launch_start
    )
    launch_environment = gate[launch_start:launch_end]

    assert "FR13_FIXED32_CUTLASS_WAVE=stock" in launch_environment
    assert "FR13_FIXED32_CUTLASS_WAVE_SO=" in launch_environment
    assert "FR13_DFWD_UNIFIED_BM8_LIVE_AB=0" in launch_environment
    assert "FR13_DFWD_UNIFIED_BM8_PRODUCTION=0" in launch_environment

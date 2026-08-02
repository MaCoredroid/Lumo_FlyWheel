from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
GATE = REPO / "scripts" / "fr13_run_b1_cutlass_streamk_live_gate.sh"
TIMING = REPO / "scripts" / "fr13_run_b1_cutlass_streamk_timing.sh"
SWE_RUNNER = REPO / "scripts" / "run_swe_bench_q36_a.py"
B1_KERNEL_GATE = REPO / "scripts" / "fr13_run_b1_kernel_live_gate.sh"
B4_GRAPH_GATE = REPO / "scripts" / "fr13_run_b4_gdn_wide_live_gate.sh"


def test_launcher_is_digest_pinned_diagnostic_only_and_worker_visible() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "FR13_FIXED32_CUTLASS_WAVE=${FR13_FIXED32_CUTLASS_WAVE:-stock}" in launcher
    assert (
        "CUTLASS Stream-K candidate requires fixed32 B1"
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
        "CUTLASS Stream-K production requires fixed32 B1 and a pinned live PASS"
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
    launcher = LAUNCHER.read_text(encoding="utf-8")

    for assignment in (
        "FR13_GATE_QROW16=0",
        "FR13_GATE_TAW_NATIVE=0",
        "FR13_GATE_DRAFT_HEAD_PAD=0",
        "FR13_GATE_DRAFT_HEAD_M32=0",
        "FR13_GATE_BM8=0",
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
    assert "(14336, 5120)" in gate
    assert "(8192, 5120)" not in gate
    assert "invalid differing-byte count" in gate
    assert "byte equality and differing-byte count disagree" in gate
    assert "installed binary attestation schema mismatch" in gate
    assert "binary.CONTAINER_SOURCE" in gate
    assert "binary.CONTAINER_DESTINATION" in gate
    assert '"served_result": "stock"' in gate
    assert '"acceptance_valid": False' in gate
    assert "FULL_VOCAB_LIVE_SCHEMA=fr13.fixed32.cutlass_streamk_live_gate.v3" in gate
    assert (
        "FULL_VOCAB_LIVE_SCHEMA="
        "fr13.fixed32.cutlass_streamk_wide256_live_gate.v1" in gate
    )
    assert (
        "LIVE_SCHEMA=fr13.fixed32.cutlass_streamk_wide256_k64_root_live_gate.v1"
        in gate
    )
    assert '"schema": expected_live_schema' in gate
    assert "fixed32_cutlass_streamk_real_task_arm.json" in gate
    assert "LUMO_SWE_AUTOCOMMIT=0" in kernel_gate
    assert "fr13-fixed32-cutlass-streamk-real-task-arm-v1" in gate
    assert '"task_marker": expected_task_marker' in gate
    assert '"real_task_arm_sha256": hashlib.sha256(arm_raw).hexdigest()' in gate
    assert 'DRAFT_VOCAB_ROOT=0' in gate
    assert 'DRAFT_VOCAB_ROOT=1' in gate
    assert 'DRAFT_VOCAB_K=0' in gate
    assert 'DRAFT_VOCAB_K=65536' in gate
    assert 'NEEDS_ALLOW=FR13_DRAFT_VOCAB_K=0' in gate
    assert 'export FR13_DRAFT_VOCAB_ROOT="$DRAFT_VOCAB_ROOT"' in gate
    assert 'export FR13_DRAFT_VOCAB_K="$DRAFT_VOCAB_K"' in gate
    assert '"draft_vocab_root": expected_draft_vocab_root' in gate
    assert '"draft_vocab_k": expected_draft_vocab_k' in gate
    assert 'MANDATORY_WEIGHT_BYTES=42025179008' in gate
    assert 'MANDATORY_WEIGHT_BYTES=32666638208' in gate
    assert 'MAX_COMPARISONS=320' in gate
    assert '"comparison_call_limit": expected_max_comparisons' in gate
    assert '"qualification_profile": expected_profile' in gate
    assert "pinned root-64K draft-vocabulary block map drifted" in gate
    assert '"comparator_timing_eligible": False' in gate
    assert '"patch_source_sha256": patch_source_sha256' in gate
    assert '"binary_attestation_sha256"' in gate
    assert "static_persistent_stocktile" in gate
    assert "static_persistent_stocktile_byte_ab" in gate
    assert "fr13.fixed32.cutlass_static_persistent_byte_ab.v1" in gate
    assert (
        "fr13.fixed32.cutlass_static_persistent_k64_root_live_gate.v1" in gate
    )
    assert "cutlass_static_persistent_k64_root_byte_gate.json" in gate
    assert (
        "B1 k64_root qualification is restricted to wide256 or "
        "static-persistent stock-tile" in gate
    )
    assert "m32_static_linear" in gate
    assert "m32_static_linear_byte_ab" in gate
    assert "fr13.fixed32.cutlass_m32_static_linear_byte_ab.v1" in gate
    assert "fr13.fixed32.cutlass_m32_static_linear_k64_root_live_gate.v1" in gate
    assert "cutlass_m32_static_linear_k64_root_byte_gate.json" in gate
    assert (
        "CUTLASS k64_root B1 qualification requires an audited fixed32 B1 scheduler"
        in launcher
    )
    assert '--qualification-profile "$FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE"' in launcher
    assert "--draft-vocab-blocks scripts/fr13_dvk_subset_blocks.json" in launcher
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
    serve = (
        REPO / "scripts" / "fr13_bigdenom_swe_serve_variant.sh"
    ).read_text(encoding="utf-8")
    assert "cutlass_wave = sys.argv[10]" in serve
    assert "streamk_eager_diagnostic=(" in serve
    assert "--fixed32-cutlass-real-event-arm" in serve
    assert "fr13_fixed32_cutlass_streamk.real_event.arm" in serve


def test_exact4_timing_is_real_full_wall_full_vocab_and_source_bound() -> None:
    timing = TIMING.read_text(encoding="utf-8")

    assert "subset_b4_four.json" in timing
    assert (
        "real_swe_verified_exact4_b1_hydra27_qrow16_streamk_timing_candidate" in timing
    )
    assert 'STOCK_ARM="hydra27_fixed32_qrow16_cutlass_stock_${TAG}"' in timing
    assert (
        'CANDIDATE_ARM="hydra27_fixed32_qrow16_${CANDIDATE_ARM_LABEL}_${TAG}"'
        in timing
    )
    assert '"$arm" hydra27_fixed32 "$SUBSET"' in timing
    assert "tail6_fixed32" not in timing
    assert "scripts/fr13_bigdenom_swe_serve_variant.sh" in timing
    assert "scripts/fr13_measure.py deploy-speed" in timing
    assert "measured_tps_fullstep_wall" not in timing
    assert 'run_arm "$STOCK_ARM" 0' in timing
    assert 'run_arm "$CANDIDATE_ARM" 1' in timing
    assert timing.index('run_arm "$STOCK_ARM" 0') < timing.index(
        'run_arm "$CANDIDATE_ARM" 1'
    )
    assert "FR13_DRAFT_VOCAB_ROOT=0 FR13_DRAFT_VOCAB_K=0" in timing
    assert "FR13_NEEDS_ALLOW='FR13_DRAFT_VOCAB_K=0'" in timing
    assert "FULL_VOCAB_WEIGHT_BYTES=42025179008" in timing
    assert "FULL_VOCAB_FLOOR_MS=153.9383846446886" in timing
    assert "FULL_VOCAB_CAP_MS=177.0291423413919" in timing
    assert (
        "STREAMK_SHA256="
        "f9bbbb8dc4ffc2227a71d2bc7b260e586ffbdc0fd946749e4f69e322c46a362d" in timing
    )
    assert (
        "STREAMK_SHA256="
        "503277a2dca6784502b709007adfe45f42d0f1a1851107e7b913e1e85a00de5a"
        in timing
    )
    assert (
        "QROW16_FA2_SHA256="
        "1649fbe9c6886147710dc9be97567bffcac36175c26742b752be9be50c2cbb86" in timing
    )
    assert "QROW16_FA2_BYTES=299507792" in timing
    assert (
        "QROW16_LIVE_PASS_SHA256="
        "36940fd43d11399529d1bfe7e11baa9961907193267f3bb43d41057328737b77" in timing
    )
    assert "FR13_FA2_QROW16_PRODUCTION=1 \\" in timing
    assert "FR13_FA2_QROW16_LIVE_PAGED_AB=0 \\" in timing
    assert 'FORKED_FA2_SO="$QROW16_FA2_SO"' in timing
    assert "FR13_FA2_QROW16_PRODUCTION=0" not in timing
    assert "STOCK_FA2_SO" not in timing
    assert "scripts/fr13_qrow16_pass_sidecar.py verify" in timing
    assert '--stock-qrow16-sidecar "$STOCK_QROW16_SIDECAR"' in timing
    assert '--candidate-qrow16-sidecar "$CANDIDATE_QROW16_SIDECAR"' in timing
    assert '--stock-qrow16-capture "$STOCK_QROW16_CAPTURE"' in timing
    assert '--candidate-qrow16-capture "$CANDIDATE_QROW16_CAPTURE"' in timing
    assert '--expected-source-commit "$SOURCE_COMMIT"' in timing
    assert "comparator_gate_timing_eligible=0" in timing
    assert "TIMING_CANDIDATE=${FR13_STREAMK_TIMING_CANDIDATE:-streamk_coop128}" in timing
    assert "TIMING_TASK_SET=${FR13_STREAMK_TIMING_TASK_SET:-exact4}" in timing
    assert "subset_b1_diagnostic_one.json" in timing
    assert "FR13_FIXED32_B1_DIAGNOSTIC=\"$B1_DIAGNOSTIC\"" in timing
    assert '--candidate-selector "$TIMING_CANDIDATE"' in timing
    assert '--task-set "$TIMING_TASK_SET"' in timing
    assert "floor_acceptance_eligible=0" in timing
    assert "quick_decode_tps_probe" not in timing
    assert "/v1/responses" not in timing

    swe_runner = SWE_RUNNER.read_text(encoding="utf-8")
    assert '"streamk_force_wide256"' in swe_runner
    assert '"streamk_force_wide256_byte_ab"' in swe_runner


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

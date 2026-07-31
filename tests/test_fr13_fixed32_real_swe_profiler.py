from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "fr13_fixed32_b1_nsys_profile.sh"
DRIVER = REPO / "scripts" / "fr13_b4_campaign_driver.sh"
RUNTIME_SHELLS = (
    SCRIPT,
    DRIVER,
    REPO / "scripts" / "fr13_bigdenom_swe_serve_variant.sh",
    REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh",
    REPO / "scripts" / "fr13_fixed32_floor_timers_seq.sh",
    REPO / "scripts" / "swe_x86_helpers" / "offload_codex_proxy.sh",
    REPO / "scripts" / "gpu_oom_guard.sh",
)
RUNNER = REPO / "scripts" / "run_swe_bench_q36_a.py"
CANONICAL_SUBSETS = {
    REPO / "config/fr13_fixed32/subset_b4_four.json": (
        "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
    ),
    REPO / "config/fr13_fixed32/subset_b4_sixteen.json": (
        "47b0a3c9be49e2cb5f7e7217ae03c267a05359f269f3e3b038942f57d7dc0b5c"
    ),
}


def test_fixed32_canonical_subsets_are_tracked_config_bytes() -> None:
    for path, expected_sha256 in CANONICAL_SUBSETS.items():
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256


def test_fixed32_profiler_uses_only_real_b1_swe_traffic() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "SUBSET=config/fr13_fixed32/subset_b4_four.json" in text
    assert "BSIZE=1" in text
    assert "CONC=1" in text
    assert "LUMO_SWE_AUTOCOMMIT=0" in text
    assert "FR13_FIXED32_ATTRIBUTION_ONLY=1" in text
    assert "FR13_FIXED32_NVTX_PROFILE=1" in text
    assert "scripts/fr13_fixed32_floor_timers_seq.sh" in text
    assert "acceptance_valid=false" in text
    assert "PROBE_ONLY" not in text
    assert "ACCEPT_SPEED_PROBE" not in text
    assert "CAPTURE_ONLY" not in text


def test_fixed32_profiler_requires_task_and_report_evidence() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert '[[ ! -s "$REPORT" ]]' in text
    assert "swe_out/verified/per_task" in text
    assert "astropy__astropy-*" in text
    assert "scripts/fr13_fixed32_nsys_reduce.py" in text
    assert '"$REPORT" \\' in text
    assert '--output "$REDUCED"' in text
    assert '--nsys-bin "$LUMO_NSYS_BIN"' in text
    assert "privacy-safe Nsight attribution reduction failed" in text


def test_fixed32_profiler_binds_proven_capture_and_real_swe_evidence() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "LUMO_NSYS_DELAY_S=${LUMO_NSYS_DELAY_S:-1200}" in text
    assert "LUMO_NSYS_DURATION_S=${LUMO_NSYS_DURATION_S:-300}" in text
    assert "LUMO_NSYS_FLUSH_MS=${LUMO_NSYS_FLUSH_MS:-100}" in text
    assert "LUMO_NSYS_SESSION_TIMEOUT_S=${LUMO_NSYS_SESSION_TIMEOUT_S:-1500}" in text
    assert "validate_nsys_delayed_collection_timeouts" in text
    assert (
        '[[ "$LUMO_NSYS_BIN" == '
        '"/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys" ]]'
        in text
    )
    assert (
        '[[ "$LUMO_NSYS_DELAY_S" == "1200" && '
        '"$LUMO_NSYS_DURATION_S" == "300" ]]'
        in text
    )
    assert "cuda,cuda-sw,nvtx" in text
    assert "CuptiUseRawGpuTimestamps=false" in text
    assert "git check-ignore" in text
    assert "--runtime-manifest-launch" in text
    assert "--external-manifest-launch" in text
    assert "--process-identity" in text
    assert "--container-identity" in text
    assert "--runtime-attestation" in text
    assert "--pretask-zero-traffic" in text
    assert "--proxy-ledger" in text
    assert "--engine-ledger" in text
    assert "--nsys-discard-environment true" in text


def test_fixed32_profiler_snapshots_terminal_engine_ledger_before_thaw() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    wait_index = text.index("if ! wait_for_fresh_stable_report")
    snapshot_index = text.index(
        'snapshot_terminal_engine_ledger "$ENGINE_LEDGER_SNAPSHOT"'
    )
    thaw_index = text.index("thaw_exact_control_ancestors", snapshot_index)
    assert wait_index < snapshot_index < thaw_index
    assert '"${PROFILE_CONTAINER_ID}:${source_path}" "$snapshot_tmp"' in text
    assert "docker cp" in text
    assert "NSYS_CONTAINER_TERMINAL_OK != 1" in text
    assert "PROFILE_CONTAINER_RUNNING != 0" in text
    assert '[[ ! -f "$snapshot_tmp" || -L "$snapshot_tmp"' in text
    assert "--engine-ledger \"$ENGINE_LEDGER_SNAPSHOT\"" in text
    assert (
        '--engine-ledger "$RUNROOT/$ARM/logs/fr13_fixed32_engine_ingress.jsonl"'
        not in text
    )
    assert "sudo" not in text


def test_fixed32_profiler_scopes_bounded_swe_wall_to_driver() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert (
        "LUMO_NSYS_SWE_AGENT_WALL_S="
        "${LUMO_NSYS_SWE_AGENT_WALL_S:-900}"
        in text
    )
    assert "validate_nsys_attribution_task_wall || exit 2" in text
    assert (
        'WALL=0 \\\nSWE_AGENT_WALL_S="$LUMO_NSYS_SWE_AGENT_WALL_S" \\\n'
        in text
    )
    assert "export SWE_AGENT_WALL_S" not in text


def test_fixed32_profiler_reaches_runtime_preflight(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1 $2" == "ps -q" ]]; then\n'
        "  printf 'busy-container\\n'\n"
        "  exit 0\n"
        "fi\n"
        "exit 99\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"

    completed = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "GPU campaign containers are already running" in completed.stderr
    assert "bad substitution" not in completed.stderr


def test_fixed32_attribution_cannot_enter_acceptance_gate() -> None:
    text = DRIVER.read_text(encoding="utf-8")

    guard = '[[ "${FR13_FIXED32_ATTRIBUTION_ONLY:-0}" == "1" ]]'
    assert guard in text
    assert text.index(guard) < text.index(".venv/bin/python scripts/fr13_floor_gate.py")
    assert "exit 86" in text


def test_fixed32_runtime_shell_chain_derives_the_active_worktree() -> None:
    for path in RUNTIME_SHELLS:
        text = path.read_text(encoding="utf-8")
        assert "/home/" not in text, path
        assert "BASH_SOURCE[0]" in text, path

    assert "REPO=${REPO:-" in SCRIPT.read_text(encoding="utf-8")
    assert "REPO=${REPO:-" in DRIVER.read_text(encoding="utf-8")
    assert "REPO=${REPO:-" in (
        REPO / "scripts" / "fr13_bigdenom_swe_serve_variant.sh"
    ).read_text(encoding="utf-8")
    assert "REPO=${REPO:-" in (
        REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
    ).read_text(encoding="utf-8")


def test_fixed32_runtime_preserves_repo_and_hf_home_overrides() -> None:
    sequence = (REPO / "scripts" / "fr13_fixed32_floor_timers_seq.sh").read_text(
        encoding="utf-8"
    )
    offload = (
        REPO / "scripts" / "swe_x86_helpers" / "offload_codex_proxy.sh"
    ).read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    assert (
        'export HF_HOME=${HF_HOME:-"$_FR13_FIXED32_REPO/.cache/huggingface"}'
        in sequence
    )
    assert "REPO_ROOT=${REPO_ROOT:-${REPO:-" in offload
    assert "REPO_ROOT = Path(__file__).resolve().parent.parent" in runner
    assert "/home/" not in runner

import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_formal_fixed32_sequence_disables_raw_task_autocommits() -> None:
    sequence = (REPO / "scripts" / "fr13_fixed32_floor_timers_seq.sh").read_text(
        encoding="utf-8"
    )

    assert sequence.count("export LUMO_SWE_AUTOCOMMIT=0") == 1


def test_formal_fixed32_sequence_disables_profiler_by_default() -> None:
    sequence = (REPO / "scripts" / "fr13_fixed32_floor_timers_seq.sh").read_text(
        encoding="utf-8"
    )

    assert 'case "${FR13_FIXED32_ATTRIBUTION_ONLY:-0}" in' in sequence
    assert "export FR13_FIXED32_NVTX_PROFILE=0" in sequence
    assert "export LUMO_NSYS_WRAP_VLLM=0" in sequence
    assert "fixed32 attribution mode requires LUMO_NSYS_WRAP_VLLM=1" in sequence

    gate = (REPO / "scripts" / "fr13_floor_gate.py").read_text(encoding="utf-8")
    assert '"FR13_FIXED32_NVTX_PROFILE": "0"' in gate


def test_fixed32_launcher_allows_only_coherent_profile_modes() -> None:
    launcher = (REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh").read_text(
        encoding="utf-8"
    )

    required_zero = launcher.split("_fixed32_required_zero=(", maxsplit=1)[1].split(
        ")", maxsplit=1
    )[0]
    assert "LUMO_NSYS_WRAP_VLLM" not in required_zero
    assert (
        'case "${FR13_FIXED32_ATTRIBUTION_ONLY:-0}:'
        "${LUMO_NSYS_WRAP_VLLM:-0}:"
        '${FR13_FIXED32_NVTX_PROFILE:-0}" in'
    ) in launcher
    assert "0:0:0|1:1:1)" in launcher
    assert "fixed32 acceptance requires Nsight and NVTX profiling disabled" in launcher
    assert "fixed32 attribution requires Nsight and NVTX profiling enabled" in launcher

    case_start = launcher.index('  case "${FR13_FIXED32_ATTRIBUTION_ONLY:-0}:')
    case_end = launcher.index("  esac", case_start) + len("  esac")
    guard = launcher[case_start:case_end]
    cases = (
        (("0", "0", "0"), 0, ""),
        (("1", "1", "1"), 0, ""),
        (
            ("0", "1", "0"),
            2,
            "fixed32 acceptance requires Nsight and NVTX profiling disabled",
        ),
        (
            ("1", "0", "1"),
            2,
            "fixed32 attribution requires Nsight and NVTX profiling enabled",
        ),
    )
    for values, expected_rc, expected_error in cases:
        env = {
            **os.environ,
            "FR13_FIXED32_ATTRIBUTION_ONLY": values[0],
            "LUMO_NSYS_WRAP_VLLM": values[1],
            "FR13_FIXED32_NVTX_PROFILE": values[2],
        }
        completed = subprocess.run(
            ["bash", "-c", guard],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == expected_rc
        assert expected_error in completed.stderr

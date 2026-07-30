from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_formal_fixed32_sequence_disables_raw_task_autocommits() -> None:
    sequence = (
        REPO / "scripts" / "fr13_fixed32_floor_timers_seq.sh"
    ).read_text(encoding="utf-8")

    assert sequence.count("export LUMO_SWE_AUTOCOMMIT=0") == 1


def test_formal_fixed32_sequence_disables_profiler_by_default() -> None:
    sequence = (
        REPO / "scripts" / "fr13_fixed32_floor_timers_seq.sh"
    ).read_text(encoding="utf-8")

    assert 'case "${FR13_FIXED32_ATTRIBUTION_ONLY:-0}" in' in sequence
    assert "export FR13_FIXED32_NVTX_PROFILE=0" in sequence
    assert "export LUMO_NSYS_WRAP_VLLM=0" in sequence
    assert (
        "fixed32 attribution mode requires LUMO_NSYS_WRAP_VLLM=1"
        in sequence
    )

    gate = (REPO / "scripts" / "fr13_floor_gate.py").read_text(
        encoding="utf-8"
    )
    assert '"FR13_FIXED32_NVTX_PROFILE": "0"' in gate

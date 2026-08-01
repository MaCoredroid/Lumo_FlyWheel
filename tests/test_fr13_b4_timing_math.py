from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fr13_b4_timing_math as timing_math  # noqa: E402


def _record() -> dict[str, float]:
    return {
        "step_wall_ms": 240.0,
        "measured_tps_fullstep_wall": 5.0 / 0.06,
        "events_per_step": 4.0,
        "wall_s_per_event": 0.06,
        "s_per_fwd_gpu": 0.04,
        "s_per_fwd_gpu_per_forward": 0.16,
        "drafter_gpu_ms_per_step": 30.0,
        "committer_gpu_ms_per_step": 20.0,
        "accept_per_event": 4.0,
        "committed_per_event": 5.0,
        "floor_ms": 120.0,
        "floor_ratio": 2.0,
    }


def test_b4_phase_breakdown_uses_per_forward_sfwd() -> None:
    result = timing_math.phase_breakdown(_record(), "B4")

    assert result == {
        "events_per_step": 4.0,
        "wall_ms_per_event": 60.0,
        "sfwd_gpu_ms_per_event": 40.0,
        "sfwd_gpu_ms_per_step": 160.0,
        "dfwd_gpu_ms_per_step": 30.0,
        "cfwd_gpu_ms_per_step": 20.0,
        "gpu_component_ms_per_step": 210.0,
        "other_wall_ms_per_step": 30.0,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("s_per_fwd_gpu_per_forward", 0.04, "SFWD event/step units"),
        ("step_wall_ms", 60.0, "wall event/step units"),
        ("committed_per_event", 4.0, "accepted and committed"),
        ("measured_tps_fullstep_wall", 20.0, "full-wall TPS"),
        ("floor_ratio", 0.5, "hardware-floor ratio"),
        ("drafter_gpu_ms_per_step", 100.0, "components exceed"),
    ),
)
def test_b4_phase_breakdown_rejects_mixed_or_inconsistent_units(
    field: str,
    value: float,
    message: str,
) -> None:
    record = copy.deepcopy(_record())
    record[field] = value

    with pytest.raises(timing_math.TimingMathError, match=message):
        timing_math.phase_breakdown(record, "B4")

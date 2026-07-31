from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MEASURE = REPO / "scripts" / "fr13_measure.py"


def test_mandatory_weight_floor_is_not_labeled_full_step_hardware_floor() -> None:
    text = MEASURE.read_text(encoding="utf-8")

    assert '"floor_is_full_step_hardware_floor": False' in text
    assert (
        '"fixed32_mandatory_weight_read_or_row_compute_lower_bound"'
        in text
    )
    assert "not a physically complete hardware-floor step" in text
    assert "optimistic weight-read-only lower bound" in text
    assert "1.0 = hardware-perfect step" not in text

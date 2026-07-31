from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MEASURE = REPO / "scripts" / "fr13_measure.py"


def test_legacy_floor_is_not_labeled_full_step_hardware_floor() -> None:
    text = MEASURE.read_text(encoding="utf-8")

    assert '"floor_is_full_step_hardware_floor": False' in text
    assert (
        '"legacy_target_weight_stream_or_row_compute_lower_bound"'
        in text
    )
    assert "it is not a physically complete hardware-floor step" in text
    assert "1.0 = hardware-perfect step" not in text

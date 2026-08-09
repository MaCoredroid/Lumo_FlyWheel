from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fr13_tier_b_sampler_pin as pin  # noqa: E402


PATCHER = SCRIPTS / "fr10_phase4_patch_vllm_tree_gdn.py"


def _synthetic(body_tail: str = "", extra: str = "") -> str:
    """A miniature patcher with the three pinned sampler functions."""

    return (
        "import os\n"
        "\n"
        "\n"
        "def _patch_unrelated_thing() -> bool:\n"
        "    return False\n"
        "\n"
        "\n"
        "def _patch_rejection_sampler_tree_lcp() -> bool:\n"
        "    helper = r'''\n"
        "def rejection_sample():\n"
        "    target_logits = apply_sampling_constraints(target_logits)\n"
        "'''\n"
        f"    return True{body_tail}\n"
        "\n"
        "\n"
        "def _patch_rejection_sampler_bonus_handoff() -> bool:\n"
        "    return True\n"
        "\n"
        "\n"
        "def _patch_rejection_sampler_target_logits_handoff() -> bool:\n"
        "    return True\n"
        f"{extra}"
    )


def test_region_is_located_by_ast_not_by_line_number() -> None:
    """A giant raw string full of top-level `def`s must not fool the extractor."""

    text = _synthetic()
    records = pin.extract_region(text)

    assert [r["name"] for r in records] == list(pin.SAMPLER_REGION_FUNCTIONS)
    lcp = records[0]
    # The `def rejection_sample():` line inside the r''' helper sits at column 0
    # and would end a naive line-scan early; the AST bound swallows it whole.
    assert lcp["lines"] == 6
    image = pin.region_bytes(text).decode("utf-8")
    assert "def rejection_sample():" in image
    assert "apply_sampling_constraints" in image
    assert "_patch_unrelated_thing" not in image


def test_region_bytes_names_each_function() -> None:
    image = pin.region_bytes(_synthetic()).decode("utf-8")
    for name in pin.SAMPLER_REGION_FUNCTIONS:
        assert f"### {pin.PIN_SCHEMA} {name}\n" in image
    assert "_patch_unrelated_thing" not in image


def test_edits_inside_the_region_move_the_sha() -> None:
    before = pin.region_sha256(_synthetic())
    after = pin.region_sha256(_synthetic(body_tail="  # tweak"))
    assert before != after


def test_edits_outside_the_region_do_not_move_the_sha() -> None:
    before = pin.region_sha256(_synthetic())
    after = pin.region_sha256(
        _synthetic(extra="\n\ndef _patch_gpu_model_runner_new_lever() -> bool:\n    return True\n")
    )
    assert before == after


def test_missing_sampler_function_fails_closed() -> None:
    text = _synthetic().replace("_patch_rejection_sampler_bonus_handoff", "_gone")
    with pytest.raises(pin.SamplerPinError, match="missing pinned sampler functions"):
        pin.extract_region(text)


def test_a_new_unpinned_sampler_function_fails_closed() -> None:
    text = _synthetic(
        extra="\n\ndef _patch_rejection_sampler_brand_new() -> bool:\n    return True\n"
    )
    with pytest.raises(pin.SamplerPinError, match="unpinned sampler-side functions"):
        pin.extract_region(text)


def test_duplicate_definition_fails_closed() -> None:
    text = _synthetic(
        extra="\n\ndef _patch_rejection_sampler_bonus_handoff() -> bool:\n    return False\n"
    )
    with pytest.raises(pin.SamplerPinError, match="more than once"):
        pin.extract_region(text)


def test_assert_pin_accepts_the_live_patcher_pin(tmp_path: Path) -> None:
    record = pin.pin_record(PATCHER)
    assert record["schema"] == pin.PIN_SCHEMA
    assert len(record["sampler_region_sha256"]) == 64
    assert [f["name"] for f in record["region_functions"]] == list(
        pin.SAMPLER_REGION_FUNCTIONS
    )

    checked = pin.assert_pin(record["sampler_region_sha256"], PATCHER)
    assert checked["sampler_region_unchanged"] is True


def test_assert_pin_rejects_drift() -> None:
    with pytest.raises(pin.SamplerPinError, match="SAMPLER REGION DRIFTED"):
        pin.assert_pin("0" * 64, PATCHER)


def test_assert_pin_rejects_a_malformed_expectation() -> None:
    with pytest.raises(pin.SamplerPinError, match="malformed"):
        pin.assert_pin("not-a-sha", PATCHER)


def test_assert_reads_the_expectation_from_a_qualification_artifact(
    tmp_path: Path,
) -> None:
    observed = pin.region_sha256(PATCHER.read_text(encoding="utf-8"), PATCHER)
    artifact = tmp_path / "qualification.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": pin.QUALIFICATION_SCHEMA,
                "candidate_id": "fr13_fa2_qrow32_gqa_pair",
                "sampler_region_sha256": observed,
            }
        ),
        encoding="utf-8",
    )
    assert pin._expected_from_qualification(artifact) == observed

    artifact.write_text(json.dumps({"schema": "wrong.v1"}), encoding="utf-8")
    with pytest.raises(pin.SamplerPinError, match="schema"):
        pin._expected_from_qualification(artifact)


def test_cli_emit_then_assert_round_trips(tmp_path: Path, capsys) -> None:
    out = tmp_path / "pin.json"
    assert pin.main(["emit", "--output", str(out)]) == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    capsys.readouterr()

    assert pin.main(["assert", "--expect", record["sampler_region_sha256"]]) == 0
    capsys.readouterr()

    assert pin.main(["assert", "--expect", "1" * 64]) == 2
    err = capsys.readouterr().err
    assert "SAMPLER REGION DRIFTED" in err

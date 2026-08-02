from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import fr13_fixed32_work_census as census  # noqa: E402
import fr13_floor_gate as floor_gate  # noqa: E402
import run_swe_bench_q36_a as orchestrator  # noqa: E402


INSTANCE_ID = "astropy__astropy-12907"
LAUNCHER = (
    REPO / "scripts/fr13_launch_forked_fa2_tree_server.sh"
).read_text()
VARIANT = (REPO / "scripts/fr13_bigdenom_swe_serve_variant.sh").read_text()
RUNNER = (REPO / "scripts/run_swe_bench_q36_a.py").read_text()


def _channel_counters(
    *,
    coverage_mask: int,
    reference_served: int,
    candidate_served: int,
) -> dict[str, object]:
    attempts = coverage_mask.bit_count()
    zero_map = {str(batch): 0 for batch in range(1, 5)}
    attempts_map = dict(zero_map)
    attempts_map["1"] = attempts
    coverage_map = dict(zero_map)
    coverage_map["1"] = coverage_mask
    passed_map = {str(batch): False for batch in range(1, 5)}
    passed_map["1"] = coverage_mask == 0x0FFF
    reference_map = dict(zero_map)
    reference_map["1"] = reference_served
    candidate_map = dict(zero_map)
    candidate_map["1"] = candidate_served
    direct_map = dict(zero_map)
    direct_map["1"] = reference_served + candidate_served
    return {
        "preseeded": True,
        "route": "fixed32_channel_zeroelide_source_col0",
        "direct_launches": reference_served + candidate_served,
        "gather_launches": 0,
        "scatter_launches": 0,
        "direct_launches_by_batch": direct_map,
        "gather_launches_by_batch": dict(zero_map),
        "scatter_launches_by_batch": dict(zero_map),
        "channel_byte_gate_coverage_mask_by_batch": coverage_map,
        "channel_byte_gate_attempts_by_batch": attempts_map,
        "channel_byte_gate_passed_by_batch": passed_map,
        "channel_reference_shadow_launches": attempts,
        "channel_candidate_shadow_launches": attempts,
        "channel_reference_served": reference_served,
        "channel_candidate_served": candidate_served,
        "channel_reference_served_by_batch": reference_map,
        "channel_candidate_served_by_batch": candidate_map,
    }


def test_launcher_binds_default_off_selector_to_private_sidecar() -> None:
    selector = "FR13_FIXED32_CONV_CHANNEL_ZEROELIDE_COMMIT"
    arm = "fr13_fixed32_conv_channel_zeroelide_commit.arm"

    assert f"{selector}=${{{selector}:-0}}" in LAUNCHER
    assert f'case "${selector}" in' in LAUNCHER
    assert '0|diagnostic) ;;' in LAUNCHER
    assert f'if [[ "${selector}" == "diagnostic" ]]; then' in LAUNCHER
    assert f'install -m 400 /dev/null \\\n      "$LOG_DIR/{arm}"' in LAUNCHER
    assert f'-e {selector}="${selector}"' in LAUNCHER
    assert "requires exact K64/root1 graph-mode B1" in LAUNCHER
    assert "must be the only kernel candidate" in LAUNCHER
    assert "FR13_FIXED32_CONV_FLAT_COMMIT is not qualified for server launch" in LAUNCHER
    assert '-e FR13_FIXED32_CONV_FLAT_COMMIT=' not in LAUNCHER


def test_variant_passes_dedicated_authenticated_real_event_arm() -> None:
    assert (
        'FIXED32_CONV_CHANNEL_REAL_EVENT_ARM_PATH="$ARMDIR_ABS/logs/'
        'fr13_fixed32_conv_channel_zeroelide.real_event.arm"'
        in VARIANT
    )
    assert "--fixed32-conv-channel-real-event-arm" in VARIANT
    assert "channel zero-elide conv qualification forbids task autocommit" in VARIANT
    assert "requires exact K64/root1 graph-mode B1" in VARIANT
    assert "must be the only kernel diagnostic" in VARIANT


def test_runner_marks_channel_gate_non_timing_and_non_acceptance() -> None:
    assert '"--fixed32-conv-channel-real-event-arm"' in RUNNER
    assert "_Fixed32ConvChannelRealTaskArm" in RUNNER
    assert "_Fixed32ConvChannelQualificationTaskBracket" in RUNNER
    classification = (
        orchestrator._Fixed32ConvChannelQualificationTaskBracket
        ._artifact_classification(object())
    )
    assert classification["run_classification"] == (
        "conv_channel_zeroelide_real_swe_qualification"
    )
    assert classification["acceptance_valid"] is False
    assert classification["performance_measurement"] is False
    assert classification["timing_eligible"] is False
    assert classification["floor_acceptance_eligible"] is False


def test_channel_real_task_arm_publishes_and_rotates_marker(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir(mode=0o700)
    path = (
        logs / orchestrator._FIXED32_CONV_CHANNEL_REAL_TASK_ARM_NAME
    ).resolve()
    arm = orchestrator._Fixed32ConvChannelRealTaskArm(
        path=path,
        instance_id=INSTANCE_ID,
    )

    arm.start()
    assert path.read_bytes() == f"swe_verified:{INSTANCE_ID}\n".encode("ascii")
    assert stat.S_IMODE(path.stat().st_mode) == 0o400
    assert path.stat().st_nlink == 1

    arm.finish()
    artifact = arm.as_dict()
    assert artifact["state"] == "ended"
    assert artifact["timing_eligible"] is False
    assert artifact["acceptance_valid"] is False
    assert not path.exists()
    assert arm.rotated_path.exists()


@pytest.mark.parametrize("host_syncs", (0, 1))
def test_channel_census_route_is_not_misclassified_as_fallback(
    host_syncs: int,
) -> None:
    event = census.reference_event(
        census.HYDRA_MODE,
        1,
        f"channel-qualification-{host_syncs}",
    )
    event["conv_commit"]["route"] = (
        "fixed32_channel_zeroelide_source_col0"
    )
    event["conv_commit"]["host_syncs"] = host_syncs

    validated = census.validate_event(
        event,
        source="channel-qualification",
        allow_conv_channel_qualification=True,
    )

    assert event["failures"]["fallback"] == 0
    assert validated.normalized_work["conv_commit"]["route"] == (
        "fixed32_channel_zeroelide_source_col0"
    )
    with pytest.raises(census.CensusError, match="conv_commit.route"):
        census.validate_event(event, source="formal-census")


@pytest.mark.parametrize(
    ("subset", "concurrency"),
    (
        (
            {
                "task_count": 4,
                "task_ids": [INSTANCE_ID] * 4,
            },
            4,
        ),
        (
            {
                "task_count": 1,
                "task_ids": [INSTANCE_ID],
            },
            1,
        ),
        (
            {
                "task_count": 2,
                "task_ids": [INSTANCE_ID, "astropy__astropy-13033"],
                "run_classification": "b1_diagnostic",
            },
            1,
        ),
    ),
)
def test_channel_audit_policy_rejects_non_b1_diagnostic_scope(
    tmp_path: Path,
    subset: dict[str, object],
    concurrency: int,
) -> None:
    with pytest.raises(
        floor_gate.GateError,
        match="requires the pinned one-task B1 diagnostic subset",
    ):
        floor_gate.build_fixed32_chat_traffic_audit(
            tmp_path,
            mode="hydra27_fixed32",
            subset=subset,
            dataset_record_digests={},
            concurrency=concurrency,
            allow_conv_channel_qualification=True,
        )


def test_channel_transition_accepts_new_depth_and_serializes_remaining() -> None:
    pre = _channel_counters(
        coverage_mask=0b0011,
        reference_served=2,
        candidate_served=0,
    )
    post = _channel_counters(
        coverage_mask=0b1011,
        reference_served=3,
        candidate_served=4,
    )
    orchestrator._validate_fixed32_conv_commit_metrics(
        pre,
        server_capacity=1,
        label="pre",
    )
    orchestrator._validate_fixed32_conv_commit_metrics(
        post,
        server_capacity=1,
        label="post",
    )
    bracket = object.__new__(
        orchestrator._Fixed32ConvChannelQualificationTaskBracket
    )
    bracket.pre_conv_commit_counters = pre
    bracket.post_conv_commit_counters = post

    bracket._validate_conv_commit_transition(post)
    coverage = bracket._artifact_classification()["qualification_coverage"]

    assert coverage["attempt_delta_by_batch"] == {
        "1": 1,
        "2": 0,
        "3": 0,
        "4": 0,
    }
    assert coverage["newly_covered_lengths_by_batch"]["1"] == [3]
    assert coverage["remaining_coverage_mask_by_batch"]["1"] == 0x0FF4
    assert coverage["remaining_lengths_by_batch"]["1"] == [
        2,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
    ]
    assert coverage["shadow_reference_launch_delta"] == 1
    assert coverage["shadow_candidate_launch_delta"] == 1
    assert coverage["reference_served_delta"] == 1
    assert coverage["candidate_served_delta"] == 4
    assert coverage["direct_launch_delta"] == 5
    assert coverage["coverage_complete"] is False


def test_channel_transition_accepts_multiple_new_depths() -> None:
    pre = _channel_counters(
        coverage_mask=0b0011,
        reference_served=2,
        candidate_served=0,
    )
    post = _channel_counters(
        coverage_mask=0b10111,
        reference_served=4,
        candidate_served=0,
    )
    bracket = object.__new__(
        orchestrator._Fixed32ConvChannelQualificationTaskBracket
    )
    bracket.pre_conv_commit_counters = pre
    bracket.post_conv_commit_counters = post

    bracket._validate_conv_commit_transition(post)
    coverage = bracket._artifact_classification()["qualification_coverage"]

    assert coverage["attempt_delta_by_batch"]["1"] == 2
    assert coverage["newly_covered_lengths_by_batch"]["1"] == [2, 4]


def test_channel_transition_rejects_task_without_new_depth() -> None:
    counters = _channel_counters(
        coverage_mask=0b0011,
        reference_served=2,
        candidate_served=0,
    )
    bracket = object.__new__(
        orchestrator._Fixed32ConvChannelQualificationTaskBracket
    )
    bracket.pre_conv_commit_counters = counters

    with pytest.raises(
        orchestrator.Fixed32BoundaryError,
        match="produced no B1 byte-gate evidence",
    ):
        bracket._validate_conv_commit_transition(counters)

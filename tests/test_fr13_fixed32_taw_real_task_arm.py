from __future__ import annotations

import json
import stat
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_swe_bench_q36_a as orchestrator  # noqa: E402
import fr13_floor_gate as floor_gate  # noqa: E402


INSTANCE_ID = "astropy__astropy-12907"
ARM_NAME = "fr13_fixed32_taw_native_precompute.real_event.arm"
EXACT4 = REPO / "config/fr13_fixed32/subset_b4_four.json"
EXACT16 = REPO / "config/fr13_fixed32/subset_b4_sixteen.json"


def _arm_path(tmp_path: Path) -> Path:
    logs = tmp_path / "logs"
    logs.mkdir(mode=0o700)
    logs.chmod(0o700)
    return (logs / ARM_NAME).resolve()


def _cfwd_arm_path(tmp_path: Path) -> Path:
    logs = tmp_path / "logs"
    logs.mkdir(mode=0o700, exist_ok=True)
    logs.chmod(0o700)
    return (
        logs / orchestrator._FIXED32_COMMITTER_LAYER_BATCH_REAL_TASK_ARM_NAME
    ).resolve()


def test_real_task_arm_atomically_publishes_and_rotates_exact_marker(
    tmp_path: Path,
) -> None:
    path = _arm_path(tmp_path)
    arm = orchestrator._Fixed32TawRealTaskArm(
        path=path,
        instance_id=INSTANCE_ID,
    )

    arm.start()

    assert path.read_bytes() == f"swe_verified:{INSTANCE_ID}\n".encode("ascii")
    assert stat.S_IMODE(path.stat().st_mode) == 0o400
    assert path.stat().st_nlink == 1
    assert arm.as_dict()["state"] == "active"
    assert arm.as_dict()["gate_eligible"] is False
    assert arm.as_dict()["floor_acceptance_eligible"] is False

    arm.finish()

    assert not path.exists()
    assert arm.rotated_path.read_bytes() == (
        f"swe_verified:{INSTANCE_ID}\n".encode("ascii")
    )
    assert stat.S_IMODE(arm.rotated_path.stat().st_mode) == 0o400
    assert arm.rotated_path.stat().st_nlink == 1
    assert arm.as_dict()["state"] == "ended"


@pytest.mark.parametrize(("subset", "task_count"), [(EXACT4, 4), (EXACT16, 16)])
def test_b4_campaign_arm_spans_concurrent_tasks_with_canonical_marker(
    tmp_path: Path,
    subset: Path,
    task_count: int,
) -> None:
    path = _arm_path(tmp_path)
    binding = floor_gate.validate_canonical_subset(subset)
    arm = orchestrator._Fixed32TawCampaignArm(
        path=path,
        subset_binding=binding,
        concurrency=4,
    )
    artifact_path = tmp_path / arm.artifact_name
    expected = (
        f"swe_verified:campaign{task_count}_{binding['sha256']}\n".encode(
            "ascii"
        )
    )

    def _concurrent_reads() -> list[bytes]:
        with ThreadPoolExecutor(max_workers=4) as executor:
            return list(
                executor.map(
                    lambda _task_id: path.read_bytes(),
                    binding["task_ids"],
                )
            )

    observed = orchestrator._run_with_fixed32_taw_campaign_arm(
        arm=arm,
        artifact_path=artifact_path,
        action=_concurrent_reads,
    )

    assert observed == [expected] * task_count
    assert not path.exists()
    assert arm.rotated_path.read_bytes() == expected
    payload = json.loads(artifact_path.read_text(encoding="ascii"))
    assert payload["schema"] == "fr13-fixed32-taw-campaign-arm-v1"
    assert payload["state"] == "ended"
    assert payload["task_count"] == task_count
    assert payload["concurrency"] == 4
    assert payload["subset_sha256"] == binding["sha256"]
    assert payload["task_ids"] == binding["task_ids"]
    assert payload["marker"] == expected.decode("ascii").strip()


@pytest.mark.parametrize(("subset", "task_count"), [(EXACT4, 4), (EXACT16, 16)])
def test_cfwd_b4_campaign_arm_uses_one_canonical_restrictive_marker(
    tmp_path: Path,
    subset: Path,
    task_count: int,
) -> None:
    path = _cfwd_arm_path(tmp_path)
    binding = floor_gate.validate_canonical_subset(subset)
    arm = orchestrator._Fixed32CommitterLayerBatchCampaignArm(
        path=path,
        subset_binding=binding,
        concurrency=4,
    )

    arm.start()
    expected = f"swe_verified:campaign{task_count}_{binding['sha256']}\n"
    assert path.read_text(encoding="ascii") == expected
    assert stat.S_IMODE(path.stat().st_mode) == 0o400
    arm.finish()

    payload = arm.as_dict()
    assert payload["schema"] == (
        "fr13-fixed32-committer-layer-batch-campaign-arm-v1"
    )
    assert payload["run_classification"] == (
        "cfwd_layer_batch_real_swe_b4_qualification"
    )
    assert payload["task_count"] == task_count
    assert payload["performance_measurement"] is False
    assert payload["timing_eligible"] is False
    assert payload["durable_production_pass"] is False
    assert payload["same_process_timing_handoff_contract_implemented"] is True
    assert payload["same_process_timing_execution_implemented"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda binding: {**binding, "sha256": "0" * 64},
            "differs from the canonical subset binding",
        ),
        (
            lambda binding: {
                **binding,
                "task_ids": list(reversed(binding["task_ids"])),
            },
            "differs from the canonical subset binding",
        ),
    ],
)
def test_b4_campaign_arm_rejects_tampered_subset_binding(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    binding = floor_gate.validate_canonical_subset(EXACT4)
    with pytest.raises(orchestrator.Fixed32BoundaryError, match=message):
        orchestrator._Fixed32TawCampaignArm(
            path=_arm_path(tmp_path),
            subset_binding=mutate(binding),
            concurrency=4,
        )


def test_b4_campaign_arm_rejects_non_b4_concurrency(tmp_path: Path) -> None:
    binding = floor_gate.validate_canonical_subset(EXACT4)
    with pytest.raises(
        orchestrator.Fixed32BoundaryError,
        match="requires exact B4 concurrency",
    ):
        orchestrator._Fixed32TawCampaignArm(
            path=_arm_path(tmp_path),
            subset_binding=binding,
            concurrency=1,
        )


def test_b4_campaign_arm_rotates_when_a_worker_fails(tmp_path: Path) -> None:
    path = _arm_path(tmp_path)
    arm = orchestrator._Fixed32TawCampaignArm(
        path=path,
        subset_binding=floor_gate.validate_canonical_subset(EXACT4),
        concurrency=4,
    )
    artifact_path = tmp_path / arm.artifact_name

    def _fail() -> None:
        assert path.is_file()
        raise RuntimeError("worker failed")

    with pytest.raises(RuntimeError, match="worker failed"):
        orchestrator._run_with_fixed32_taw_campaign_arm(
            arm=arm,
            artifact_path=artifact_path,
            action=_fail,
        )

    assert not path.exists()
    assert arm.rotated_path.is_file()
    assert json.loads(artifact_path.read_text(encoding="ascii"))["state"] == "ended"


def test_cutlass_real_task_arm_uses_canonical_marker_and_artifact_contract(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir(mode=0o700)
    path = (
        logs / orchestrator._FIXED32_CUTLASS_REAL_TASK_ARM_NAME
    ).resolve()
    arm = orchestrator._Fixed32CutlassRealTaskArm(
        path=path,
        instance_id=INSTANCE_ID,
    )

    arm.start()
    assert path.read_text(encoding="ascii") == f"swe_verified:{INSTANCE_ID}\n"
    arm.finish()

    payload = arm.as_dict()
    assert payload["schema"] == (
        "fr13-fixed32-cutlass-streamk-real-task-arm-v1"
    )
    assert payload["state"] == "ended"
    assert arm.artifact_name == "fixed32_cutlass_streamk_real_task_arm.json"


def test_cfwd_qualification_arm_is_task_bound_and_process_local(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir(mode=0o700)
    path = (
        logs / orchestrator._FIXED32_COMMITTER_LAYER_BATCH_REAL_TASK_ARM_NAME
    ).resolve()
    arm = orchestrator._Fixed32CommitterLayerBatchRealTaskArm(
        path=path,
        instance_id=INSTANCE_ID,
    )

    arm.start()
    assert path.read_text(encoding="ascii") == f"swe_verified:{INSTANCE_ID}\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o400
    arm.finish()

    payload = arm.as_dict()
    assert payload["run_classification"] == (
        "cfwd_layer_batch_real_swe_qualification"
    )
    assert payload["performance_measurement"] is False
    assert payload["timing_eligible"] is False
    assert payload["process_local_qualification_only"] is True
    assert payload["durable_production_pass"] is False
    assert payload["timing_requires_same_server_process"] is True
    assert payload["same_process_timing_handoff_implemented"] is False
    assert payload["state"] == "ended"
    assert arm.artifact_name == (
        "fixed32_committer_layer_batch_real_task_arm.json"
    )


def test_real_task_arm_rejects_stale_or_symlink_destination(
    tmp_path: Path,
) -> None:
    path = _arm_path(tmp_path)
    path.write_text("swe_verified:stale\n", encoding="ascii")
    arm = orchestrator._Fixed32TawRealTaskArm(
        path=path,
        instance_id=INSTANCE_ID,
    )
    with pytest.raises(
        orchestrator.Fixed32BoundaryError,
        match="destination is not fresh",
    ):
        arm.start()
    assert path.read_text(encoding="ascii") == "swe_verified:stale\n"

    path.unlink()
    target = path.with_name("unrelated")
    target.write_text("unrelated\n", encoding="ascii")
    path.symlink_to(target)
    arm = orchestrator._Fixed32TawRealTaskArm(
        path=path,
        instance_id=INSTANCE_ID,
    )
    with pytest.raises(
        orchestrator.Fixed32BoundaryError,
        match="destination is not fresh",
    ):
        arm.start()
    assert path.is_symlink()


def test_real_task_arm_fails_closed_when_live_marker_changes(
    tmp_path: Path,
) -> None:
    path = _arm_path(tmp_path)
    arm = orchestrator._Fixed32TawRealTaskArm(
        path=path,
        instance_id=INSTANCE_ID,
    )
    arm.start()
    path.chmod(0o600)
    path.write_text("swe_verified:django__django-10000\n", encoding="ascii")

    with pytest.raises(
        orchestrator.Fixed32BoundaryError,
        match="metadata is noncanonical",
    ):
        arm.finish()

    assert path.exists()
    assert not arm.rotated_path.exists()


@pytest.mark.parametrize(
    ("path_factory", "instance_id", "message"),
    [
        (lambda tmp_path: Path(ARM_NAME), INSTANCE_ID, "absolute canonical filename"),
        (
            lambda tmp_path: (tmp_path / "logs" / ARM_NAME).resolve(),
            "astropy task",
            "instance ID is not kernel-canonical",
        ),
    ],
)
def test_real_task_arm_rejects_noncanonical_binding(
    tmp_path: Path,
    path_factory,
    instance_id: str,
    message: str,
) -> None:
    path = path_factory(tmp_path)
    if path.is_absolute():
        path.parent.mkdir(mode=0o700)
        path.parent.chmod(0o700)
    with pytest.raises(orchestrator.Fixed32BoundaryError, match=message):
        orchestrator._Fixed32TawRealTaskArm(
            path=path,
            instance_id=instance_id,
        )


class _Ack(SimpleNamespace):
    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "producer_pid": self.producer_pid,
            "generation": self.generation,
            "counters": self.counters,
        }


def _b4_gate_state(*, complete: bool) -> tuple[dict[str, int], dict[str, int]]:
    if not complete:
        return (
            {str(batch): 0 for batch in range(1, 5)},
            {str(batch): 0 for batch in range(1, 5)},
        )
    return (
        {"1": 12, "2": 6, "3": 4, "4": 3},
        {str(batch): 0x0FFF for batch in range(1, 5)},
    )


def test_cfwd_b4_transition_accounts_for_multiple_depths_per_event() -> None:
    pre_attempts, pre_coverage = _b4_gate_state(complete=False)
    post_attempts, post_coverage = _b4_gate_state(complete=True)

    transition = orchestrator._fixed32_cfwd_b4_qualification_transition(
        pre_attempts=pre_attempts,
        pre_coverage=pre_coverage,
        post_attempts=post_attempts,
        post_coverage=post_coverage,
    )

    assert transition["attempt_delta_by_batch"] == post_attempts
    assert transition["new_coverage_mask_by_batch"] == post_coverage


@pytest.mark.parametrize(
    ("attempts", "coverage"),
    (
        ({"1": 0, "2": 0, "3": 0, "4": 1}, {"1": 0, "2": 0, "3": 0, "4": 0}),
        ({"1": 0, "2": 0, "3": 0, "4": 1}, {"1": 0, "2": 0, "3": 0, "4": 0x001F}),
        ({"1": 0, "2": 0, "3": 0, "4": 2}, {"1": 0, "2": 0, "3": 0, "4": 0x0001}),
    ),
)
def test_cfwd_b4_transition_rejects_impossible_attempt_coverage_relation(
    attempts: dict[str, int],
    coverage: dict[str, int],
) -> None:
    pre_attempts, pre_coverage = _b4_gate_state(complete=False)
    with pytest.raises(
        orchestrator.Fixed32BoundaryError,
        match="did not advance monotonically",
    ):
        orchestrator._fixed32_cfwd_b4_qualification_transition(
            pre_attempts=pre_attempts,
            pre_coverage=pre_coverage,
            post_attempts=attempts,
            post_coverage=coverage,
        )


def test_cfwd_b4_campaign_bracket_tears_down_before_post_flush_and_builds_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _cfwd_arm_path(tmp_path)
    binding = floor_gate.validate_canonical_subset(EXACT4)
    arm = orchestrator._Fixed32CommitterLayerBatchCampaignArm(
        path=path,
        subset_binding=binding,
        concurrency=4,
    )
    acks = [
        _Ack(
            mode="hydra27_fixed32",
            producer_pid=123,
            generation=1,
            counters={
                "pure_decode_forward_steps": 0,
                "complete_work_census_events": 0,
            },
        ),
        _Ack(
            mode="hydra27_fixed32",
            producer_pid=123,
            generation=2,
            counters={
                "pure_decode_forward_steps": 25,
                "complete_work_census_events": 25,
            },
        ),
    ]
    snapshots_with_live_arm: list[bool] = []

    class _Client:
        container = "fixed32-server"
        mode = "hydra27_fixed32"
        producer_pid = 123

        def snapshot(self) -> _Ack:
            snapshots_with_live_arm.append(path.exists())
            return acks.pop(0)

    monkeypatch.setattr(
        orchestrator,
        "_validate_fixed32_ack",
        lambda ack, label: ack.counters,
    )

    def _snapshot(**kwargs: object) -> tuple[dict[str, object], Path, str]:
        ack = kwargs["ack"]
        assert isinstance(ack, _Ack)
        assert kwargs.get("allow_incomplete_layer_batch_coverage", False) is (
            ack.generation != 3
        )
        attempts, coverage = _b4_gate_state(complete=ack.generation >= 2)
        return (
            {
                "schema": "fr13-fixed32-boundary-snapshot-v4",
                "metrics": {
                    "committer": {
                        "layer_batch_gate_attempts_by_batch": attempts,
                        "layer_batch_gate_coverage_mask_by_batch": coverage,
                    }
                },
            },
            tmp_path / f"snapshot.{ack.generation}.json",
            str(ack.generation) * 64,
        )

    monkeypatch.setattr(
        orchestrator,
        "_load_fixed32_boundary_snapshot",
        _snapshot,
    )
    monkeypatch.setattr(
        orchestrator,
        "_fixed32_metrics_snapshot",
        lambda **kwargs: "qualification metrics\n",
    )
    client = _Client()
    bracket = orchestrator._Fixed32CfwdB4QualificationCampaignBracket(
        client=client,
        boundary_snapshot_base=tmp_path / "snapshot",
        server_capacity=4,
        campaign_arm=arm,
        artifact_path=tmp_path / "qualification.json",
        arm_artifact_path=tmp_path / arm.artifact_name,
        metrics_pre_path=tmp_path / "metrics_pre.txt",
        metrics_post_path=tmp_path / "metrics_post.txt",
    )

    observed = bracket.run(lambda: path.read_text(encoding="ascii"))

    assert observed == f"swe_verified:campaign4_{binding['sha256']}\n"
    assert snapshots_with_live_arm == [False, False]
    assert not path.exists()
    payload = bracket.as_dict()
    assert payload["state"] == "qualified_process_local"
    assert payload["qualification_coverage"]["coverage_complete"] is True
    assert payload["performance_measurement"] is False
    assert payload["durable_production_pass"] is False
    handoff = bracket.handoff
    assert handoff is not None
    handoff_payload = handoff.as_dict()
    assert "producer_pid" not in handoff_payload
    assert handoff_payload["timing_window_authorized"] is False
    assert handoff_payload["artifact_is_replayable_credential"] is False

    wrong_client = SimpleNamespace(
        container="replacement-server",
        mode=client.mode,
        producer_pid=client.producer_pid,
        snapshot=lambda: (_ for _ in ()).throw(
            AssertionError("replacement server must fail before snapshot")
        ),
    )
    with pytest.raises(
        orchestrator.Fixed32BoundaryError,
        match="same server process",
    ):
        handoff.validate_timing_pre(client=wrong_client)

    timing_ack = _Ack(
        mode="hydra27_fixed32",
        producer_pid=123,
        generation=3,
        counters={
            "pure_decode_forward_steps": 25,
            "complete_work_census_events": 25,
        },
    )
    acks.append(timing_ack)
    prevalidated = handoff.validate_timing_pre(client=client)
    assert prevalidated["timing_window_authorized"] is True
    assert prevalidated["same_process_timing_execution_implemented"] is False


def test_cfwd_b4_campaign_teardown_failure_forbids_post_flush(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _cfwd_arm_path(tmp_path)
    arm = orchestrator._Fixed32CommitterLayerBatchCampaignArm(
        path=path,
        subset_binding=floor_gate.validate_canonical_subset(EXACT4),
        concurrency=4,
    )
    ack = _Ack(
        mode="tail6_fixed32",
        producer_pid=123,
        generation=1,
        counters={
            "pure_decode_forward_steps": 0,
            "complete_work_census_events": 0,
        },
    )

    class _Client:
        container = "fixed32-server"
        mode = "tail6_fixed32"
        producer_pid = 123
        calls = 0

        def snapshot(self) -> _Ack:
            self.calls += 1
            return ack

    client = _Client()
    monkeypatch.setattr(
        orchestrator,
        "_load_fixed32_boundary_snapshot",
        lambda **kwargs: (
            {
                "schema": "fr13-fixed32-boundary-snapshot-v4",
                "metrics": {
                    "committer": {
                        "layer_batch_gate_attempts_by_batch": _b4_gate_state(
                            complete=False
                        )[0],
                        "layer_batch_gate_coverage_mask_by_batch": _b4_gate_state(
                            complete=False
                        )[1],
                    }
                },
            },
            tmp_path / "snapshot.1.json",
            "1" * 64,
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "_fixed32_metrics_snapshot",
        lambda **kwargs: "qualification metrics\n",
    )
    bracket = orchestrator._Fixed32CfwdB4QualificationCampaignBracket(
        client=client,
        boundary_snapshot_base=tmp_path / "snapshot",
        server_capacity=4,
        campaign_arm=arm,
        artifact_path=tmp_path / "qualification.json",
        arm_artifact_path=tmp_path / arm.artifact_name,
        metrics_pre_path=tmp_path / "metrics_pre.txt",
        metrics_post_path=tmp_path / "metrics_post.txt",
    )
    bracket.pre()
    monkeypatch.setattr(
        arm,
        "finish",
        lambda: (_ for _ in ()).throw(RuntimeError("cannot remove campaign arm")),
    )

    with pytest.raises(
        orchestrator.Fixed32BoundaryError,
        match="campaign arm teardown failed",
    ):
        bracket.post(action_succeeded=True)

    assert client.calls == 1
    assert path.is_file()
    assert bracket.handoff is None


def test_task_bracket_arms_after_pre_flush_and_rotates_after_post_flush(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _arm_path(tmp_path)
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    snapshots_with_live_arm: list[bool] = []
    acks = [
        _Ack(
            mode="hydra27_fixed32",
            producer_pid=123,
            generation=1,
            counters={
                "pure_decode_forward_steps": 0,
                "complete_work_census_events": 0,
            },
        ),
        _Ack(
            mode="hydra27_fixed32",
            producer_pid=123,
            generation=2,
            counters={
                "pure_decode_forward_steps": 3,
                "complete_work_census_events": 3,
            },
        ),
    ]

    class _Client:
        mode = "hydra27_fixed32"
        producer_pid = 123

        def snapshot(self) -> _Ack:
            snapshots_with_live_arm.append(path.exists())
            return acks.pop(0)

    monkeypatch.setattr(
        orchestrator,
        "_validate_fixed32_ack",
        lambda ack, label: ack.counters,
    )
    monkeypatch.setattr(
        orchestrator,
        "_load_fixed32_boundary_snapshot",
        lambda **kwargs: (
            {
                "schema": "fr13-fixed32-boundary-snapshot-v4",
                "metrics": {
                    "committer": {
                        "layer_batch_gate_attempts_by_batch": {"1": 0},
                        "layer_batch_gate_coverage_mask_by_batch": {
                            "1": 0x0FFF,
                        },
                    }
                },
            },
            tmp_path / f"snapshot.{kwargs['ack'].generation}.json",
            str(kwargs["ack"].generation) * 64,
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "_fixed32_metrics_snapshot",
        lambda **kwargs: f"generation={kwargs['snapshot']['schema']}\n",
    )

    bracket = orchestrator._Fixed32TaskBracket(
        client=_Client(),
        task_dir=task_dir,
        instance_id=INSTANCE_ID,
        boundary_snapshot_base=tmp_path / "snapshot",
        server_capacity=1,
        taw_real_task_arm=orchestrator._Fixed32TawRealTaskArm(
            path=path,
            instance_id=INSTANCE_ID,
        ),
    )

    bracket.pre(task_dir / "metrics_pre.txt")
    assert snapshots_with_live_arm == [False]
    assert path.read_text(encoding="ascii") == f"swe_verified:{INSTANCE_ID}\n"

    payload = bracket.post(task_dir / "metrics_post.txt")

    assert snapshots_with_live_arm == [False, True]
    assert not path.exists()
    arm_payload = json.loads(
        (task_dir / "fixed32_taw_real_task_arm.json").read_text(
            encoding="utf-8"
        )
    )
    assert arm_payload["state"] == "ended"
    assert Path(arm_payload["rotated_path"]).is_file()
    assert payload["forward_step_interval"] == {
        "start_forward_step": 0,
        "end_forward_step": 3,
        "expected_complete_events": 3,
    }
    persisted = json.loads(
        (task_dir / "fixed32_task_boundary.json").read_text(encoding="utf-8")
    )
    assert persisted == payload


def test_cfwd_qualification_bracket_removes_arm_before_post_flush_and_records_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir(mode=0o700)
    path = (
        logs / orchestrator._FIXED32_COMMITTER_LAYER_BATCH_REAL_TASK_ARM_NAME
    ).resolve()
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    snapshots_with_live_arm: list[bool] = []
    acks = [
        _Ack(
            mode="hydra27_fixed32",
            producer_pid=123,
            generation=1,
            counters={
                "pure_decode_forward_steps": 0,
                "complete_work_census_events": 0,
            },
        ),
        _Ack(
            mode="hydra27_fixed32",
            producer_pid=123,
            generation=2,
            counters={
                "pure_decode_forward_steps": 3,
                "complete_work_census_events": 3,
            },
        ),
    ]

    class _Client:
        mode = "hydra27_fixed32"
        producer_pid = 123

        def snapshot(self) -> _Ack:
            snapshots_with_live_arm.append(path.exists())
            return acks.pop(0)

    monkeypatch.setattr(
        orchestrator,
        "_validate_fixed32_ack",
        lambda ack, label: ack.counters,
    )

    def _snapshot(**kwargs: object) -> tuple[dict[str, object], Path, str]:
        assert kwargs["allow_incomplete_layer_batch_coverage"] is True
        ack = kwargs["ack"]
        assert isinstance(ack, _Ack)
        post = ack.generation == 2
        return (
            {
                "schema": "fr13-fixed32-boundary-snapshot-v4",
                "metrics": {
                    "committer": {
                        "layer_batch_gate_attempts_by_batch": {
                            "1": 3 if post else 2,
                        },
                        "layer_batch_gate_coverage_mask_by_batch": {
                            "1": 0b1011 if post else 0b0011,
                        },
                    }
                },
            },
            tmp_path / f"snapshot.{ack.generation}.json",
            str(ack.generation) * 64,
        )

    monkeypatch.setattr(
        orchestrator,
        "_load_fixed32_boundary_snapshot",
        _snapshot,
    )
    monkeypatch.setattr(
        orchestrator,
        "_fixed32_metrics_snapshot",
        lambda **kwargs: "qualification metrics\n",
    )
    arm = orchestrator._Fixed32CommitterLayerBatchRealTaskArm(
        path=path,
        instance_id=INSTANCE_ID,
    )
    bracket = orchestrator._Fixed32CfwdQualificationTaskBracket(
        client=_Client(),
        task_dir=task_dir,
        instance_id=INSTANCE_ID,
        boundary_snapshot_base=tmp_path / "snapshot",
        server_capacity=1,
        taw_real_task_arm=arm,
    )

    bracket.pre(task_dir / "metrics_pre.txt")
    assert snapshots_with_live_arm == [False]
    assert path.is_file()
    payload = bracket.post(task_dir / "metrics_post.txt")

    assert snapshots_with_live_arm == [False, False]
    assert not path.exists()
    assert arm.rotated_path.is_file()
    assert payload["run_classification"] == (
        "cfwd_layer_batch_real_swe_qualification"
    )
    assert payload["timing_eligible"] is False
    assert payload["durable_production_pass"] is False
    coverage = payload["qualification_coverage"]
    assert coverage["pre_coverage_mask_by_batch"] == {"1": 0b0011}
    assert coverage["post_coverage_mask_by_batch"] == {"1": 0b1011}
    assert coverage["attempt_delta_by_batch"] == {"1": 1}
    assert coverage["new_coverage_mask_by_batch"] == {"1": 0b1000}
    assert coverage["newly_covered_lengths_by_batch"] == {"1": [3]}
    assert coverage["remaining_coverage_mask_by_batch"] == {"1": 0x0FF4}
    assert coverage["coverage_complete"] is False
    assert coverage["shadow_reference_replays"] == 1
    assert coverage["shadow_candidate_replays"] == 1
    assert coverage["new_depth_reference_served_replays"] == 1
    assert coverage["formal_work_census_eligible"] is False
    arm_payload = json.loads(
        (
            task_dir / "fixed32_committer_layer_batch_real_task_arm.json"
        ).read_text(encoding="utf-8")
    )
    assert arm_payload["state"] == "ended"


def test_cfwd_qualification_does_not_post_flush_when_arm_removal_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir(mode=0o700)
    path = (
        logs / orchestrator._FIXED32_COMMITTER_LAYER_BATCH_REAL_TASK_ARM_NAME
    ).resolve()
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    ack = _Ack(
        mode="tail6_fixed32",
        producer_pid=123,
        generation=1,
        counters={
            "pure_decode_forward_steps": 0,
            "complete_work_census_events": 0,
        },
    )

    class _Client:
        mode = "tail6_fixed32"
        producer_pid = 123
        calls = 0

        def snapshot(self) -> _Ack:
            self.calls += 1
            return ack

    client = _Client()
    monkeypatch.setattr(
        orchestrator,
        "_validate_fixed32_ack",
        lambda current_ack, label: current_ack.counters,
    )
    monkeypatch.setattr(
        orchestrator,
        "_load_fixed32_boundary_snapshot",
        lambda **kwargs: (
            {
                "schema": "fr13-fixed32-boundary-snapshot-v4",
                "metrics": {
                    "committer": {
                        "layer_batch_gate_attempts_by_batch": {"1": 0},
                        "layer_batch_gate_coverage_mask_by_batch": {"1": 0},
                    }
                },
            },
            tmp_path / "snapshot.1.json",
            "1" * 64,
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "_fixed32_metrics_snapshot",
        lambda **kwargs: "qualification metrics\n",
    )
    arm = orchestrator._Fixed32CommitterLayerBatchRealTaskArm(
        path=path,
        instance_id=INSTANCE_ID,
    )
    bracket = orchestrator._Fixed32CfwdQualificationTaskBracket(
        client=client,
        task_dir=task_dir,
        instance_id=INSTANCE_ID,
        boundary_snapshot_base=tmp_path / "snapshot",
        server_capacity=1,
        taw_real_task_arm=arm,
    )
    bracket.pre(task_dir / "metrics_pre.txt")
    monkeypatch.setattr(arm, "finish", lambda: (_ for _ in ()).throw(
        RuntimeError("cannot remove arm")
    ))

    with pytest.raises(
        orchestrator.Fixed32BoundaryError,
        match="cannot remove arm",
    ):
        bracket.post(task_dir / "metrics_post.txt")

    assert client.calls == 1
    assert path.is_file()


@pytest.mark.parametrize(
    ("attempt_delta", "coverage_delta", "error_match"),
    (
        (True, False, "attempted a committer layer-batch byte gate"),
        (False, True, "changed committer layer-batch accepted-length coverage"),
    ),
)
def test_task_bracket_rejects_layer_batch_qualification_in_task_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attempt_delta: bool,
    coverage_delta: bool,
    error_match: str,
) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    acks = [
        _Ack(
            mode="hydra27_fixed32",
            producer_pid=123,
            generation=1,
            counters={
                "pure_decode_forward_steps": 0,
                "complete_work_census_events": 0,
            },
        ),
        _Ack(
            mode="hydra27_fixed32",
            producer_pid=123,
            generation=2,
            counters={
                "pure_decode_forward_steps": 1,
                "complete_work_census_events": 1,
            },
        ),
    ]

    class _Client:
        mode = "hydra27_fixed32"
        producer_pid = 123

        def snapshot(self) -> _Ack:
            return acks.pop(0)

    monkeypatch.setattr(
        orchestrator,
        "_validate_fixed32_ack",
        lambda ack, label: ack.counters,
    )

    def _snapshot(**kwargs: object) -> tuple[dict[str, object], Path, str]:
        ack = kwargs["ack"]
        assert isinstance(ack, _Ack)
        return (
            {
                "schema": "fr13-fixed32-boundary-snapshot-v4",
                "metrics": {
                    "committer": {
                        "layer_batch_gate_attempts_by_batch": {
                            "1": (ack.generation - 1) if attempt_delta else 0,
                        },
                        "layer_batch_gate_coverage_mask_by_batch": {
                            "1": (
                                0x0FFF - (ack.generation - 1)
                                if coverage_delta
                                else 0x0FFF
                            ),
                        },
                    },
                },
            },
            tmp_path / f"snapshot.{ack.generation}.json",
            str(ack.generation) * 64,
        )

    monkeypatch.setattr(
        orchestrator,
        "_load_fixed32_boundary_snapshot",
        _snapshot,
    )
    monkeypatch.setattr(
        orchestrator,
        "_fixed32_metrics_snapshot",
        lambda **kwargs: "metrics\n",
    )
    bracket = orchestrator._Fixed32TaskBracket(
        client=_Client(),
        task_dir=task_dir,
        instance_id=INSTANCE_ID,
        boundary_snapshot_base=tmp_path / "snapshot",
        server_capacity=1,
    )

    bracket.pre(task_dir / "metrics_pre.txt")
    with pytest.raises(
        orchestrator.Fixed32BoundaryError,
        match=error_match,
    ):
        bracket.post(task_dir / "metrics_post.txt")

    assert bracket.post_attempted is True
    assert bracket.complete is False


def test_eager_kernel_diagnostic_bracket_never_flushes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _arm_path(tmp_path)
    task_dir = tmp_path / "task"
    task_dir.mkdir()

    class _Client:
        mode = "tail6_fixed32"
        producer_pid = 123

        def snapshot(self) -> None:
            raise AssertionError("eager diagnostic must not flush")

    snapshots = iter(("pre metrics\n", "post metrics\n"))
    monkeypatch.setattr(
        orchestrator,
        "_metrics_snapshot",
        lambda _url: next(snapshots),
    )
    bracket = orchestrator._Fixed32EagerKernelDiagnosticTaskBracket(
        client=_Client(),
        task_dir=task_dir,
        instance_id=INSTANCE_ID,
        boundary_snapshot_base=tmp_path / "snapshot",
        server_capacity=4,
        taw_real_task_arm=orchestrator._Fixed32CutlassRealTaskArm(
            path=(
                path.parent
                / orchestrator._FIXED32_CUTLASS_REAL_TASK_ARM_NAME
            ),
            instance_id=INSTANCE_ID,
        ),
    )

    bracket.pre(task_dir / "metrics_pre.txt")
    payload = bracket.post(task_dir / "metrics_post.txt")

    assert payload["acceptance_valid"] is False
    assert payload["flush_protocol_used"] is False
    assert payload["pre_metrics"]["bytes"] > 0
    assert payload["post_metrics"]["bytes"] > 0
    assert not bracket.taw_real_task_arm.path.exists()
    assert bracket.taw_real_task_arm.rotated_path.is_file()


def test_eager_kernel_diagnostic_rotates_arm_after_post_metrics_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _arm_path(tmp_path)
    task_dir = tmp_path / "task"
    task_dir.mkdir()

    class _Client:
        mode = "tail6_fixed32"
        producer_pid = 123

    calls = 0

    def _metrics(_url: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("metrics unavailable")
        return "pre metrics\n"

    monkeypatch.setattr(orchestrator, "_metrics_snapshot", _metrics)
    bracket = orchestrator._Fixed32EagerKernelDiagnosticTaskBracket(
        client=_Client(),
        task_dir=task_dir,
        instance_id=INSTANCE_ID,
        boundary_snapshot_base=tmp_path / "snapshot",
        server_capacity=1,
        taw_real_task_arm=orchestrator._Fixed32CutlassRealTaskArm(
            path=(
                path.parent
                / orchestrator._FIXED32_CUTLASS_REAL_TASK_ARM_NAME
            ),
            instance_id=INSTANCE_ID,
        ),
    )

    bracket.pre(task_dir / "metrics_pre.txt")
    with pytest.raises(
        orchestrator.Fixed32BoundaryError,
        match="metrics unavailable",
    ):
        bracket.post(task_dir / "metrics_post.txt")

    assert not bracket.taw_real_task_arm.path.exists()
    assert bracket.taw_real_task_arm.rotated_path.is_file()
    assert bracket.post_attempted is True
    assert bracket.complete is False


def test_eager_kernel_diagnostic_rotates_arm_after_pre_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _arm_path(tmp_path)
    task_dir = tmp_path / "task"
    task_dir.mkdir()

    class _Client:
        mode = "tail6_fixed32"
        producer_pid = 123

    monkeypatch.setattr(
        orchestrator,
        "_metrics_snapshot",
        lambda _url: "pre metrics\n",
    )
    bracket = orchestrator._Fixed32EagerKernelDiagnosticTaskBracket(
        client=_Client(),
        task_dir=task_dir,
        instance_id=INSTANCE_ID,
        boundary_snapshot_base=tmp_path / "snapshot",
        server_capacity=1,
        taw_real_task_arm=orchestrator._Fixed32CutlassRealTaskArm(
            path=(
                path.parent
                / orchestrator._FIXED32_CUTLASS_REAL_TASK_ARM_NAME
            ),
            instance_id=INSTANCE_ID,
        ),
    )

    def _fail_publication() -> None:
        raise OSError("artifact unavailable")

    monkeypatch.setattr(bracket, "_write_artifact", _fail_publication)
    with pytest.raises(
        orchestrator.Fixed32BoundaryError,
        match="artifact unavailable",
    ):
        bracket.pre(task_dir / "metrics_pre.txt")

    assert not bracket.taw_real_task_arm.path.exists()
    assert bracket.taw_real_task_arm.rotated_path.is_file()
    assert bracket.started is False


def test_task_bracket_removes_arm_when_post_snapshot_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _arm_path(tmp_path)
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    ack = _Ack(
        mode="tail6_fixed32",
        producer_pid=123,
        generation=1,
        counters={
            "pure_decode_forward_steps": 0,
            "complete_work_census_events": 0,
        },
    )

    class _Client:
        mode = "tail6_fixed32"
        producer_pid = 123
        calls = 0

        def snapshot(self) -> _Ack:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("post flush failed")
            return ack

    monkeypatch.setattr(
        orchestrator,
        "_validate_fixed32_ack",
        lambda value, label: value.counters,
    )
    monkeypatch.setattr(
        orchestrator,
        "_load_fixed32_boundary_snapshot",
        lambda **kwargs: (
            {
                "schema": "fr13-fixed32-boundary-snapshot-v4",
                "metrics": {
                    "committer": {
                        "layer_batch_gate_attempts_by_batch": {"1": 0},
                        "layer_batch_gate_coverage_mask_by_batch": {
                            "1": 0x0FFF,
                        },
                    }
                },
            },
            tmp_path / "snapshot.1.json",
            "1" * 64,
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "_fixed32_metrics_snapshot",
        lambda **kwargs: "metrics\n",
    )
    arm = orchestrator._Fixed32TawRealTaskArm(
        path=path,
        instance_id=INSTANCE_ID,
    )
    bracket = orchestrator._Fixed32TaskBracket(
        client=_Client(),
        task_dir=task_dir,
        instance_id=INSTANCE_ID,
        boundary_snapshot_base=tmp_path / "snapshot",
        server_capacity=1,
        taw_real_task_arm=arm,
    )
    bracket.pre(task_dir / "metrics_pre.txt")

    with pytest.raises(
        orchestrator.Fixed32BoundaryError,
        match="post flush failed",
    ):
        bracket.post(task_dir / "metrics_post.txt")

    assert not path.exists()
    assert arm.rotated_path.is_file()
    assert bracket.post_attempted is True
    assert bracket.complete is False

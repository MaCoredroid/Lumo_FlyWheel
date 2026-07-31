from __future__ import annotations

import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_swe_bench_q36_a as orchestrator  # noqa: E402


INSTANCE_ID = "astropy__astropy-12907"
ARM_NAME = "fr13_fixed32_taw_native_precompute.real_event.arm"


def _arm_path(tmp_path: Path) -> Path:
    logs = tmp_path / "logs"
    logs.mkdir(mode=0o700)
    logs.chmod(0o700)
    return (logs / ARM_NAME).resolve()


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
            {"schema": "fr13-fixed32-boundary-snapshot-v4"},
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
            {"schema": "fr13-fixed32-boundary-snapshot-v4"},
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

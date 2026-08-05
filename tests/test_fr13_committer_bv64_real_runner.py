from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/fr13_run_committer_bv64_real.sh"
REDUCER = ROOT / "scripts/fr13_committer_bv64_real_result.py"


def _load_reducer():
    spec = importlib.util.spec_from_file_location("fr13_committer_bv64_result", REDUCER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, sort_keys=True) + "\n").encode("ascii")
    path.write_bytes(raw)
    return raw


def _map(values: dict[int, int]) -> dict[str, int]:
    return {str(batch): values.get(batch, 0) for batch in range(1, 5)}


def _evidence(tmp_path: Path, *, batch: int, events: int, attempts: int):
    arm = tmp_path / "arm"
    tasks = (
        ["astropy__astropy-12907"]
        if batch == 1
        else [
            "astropy__astropy-12907",
            "astropy__astropy-13033",
            "astropy__astropy-13236",
            "astropy__astropy-13398",
        ]
    )
    subset = tmp_path / "subset.json"
    _write_json(
        subset,
        {
            "dataset_name": "princeton-nlp/SWE-bench_Verified",
            "instance_ids": tasks,
        },
    )
    ready_attempts = {
        str(item): (attempts if item == batch else 0)
        for item in range(1, batch + 1)
    }
    replay_values = (
        {1: events}
        if batch == 1
        else {1: 2, 2: 2, 3: 4, 4: events - 8}
    )

    def snapshot(*, post: bool) -> dict:
        count = events if post else 0
        committer = {
            "actual_replays_enqueued": count,
            "actual_replays_by_batch": _map(replay_values if post else {}),
            "all_batches_ready": True,
            "fast_route_ready": True,
            "captures": batch,
            "preseeded_graphs": batch,
            "preseeded_batches": list(range(1, batch + 1)),
            "nonpure_committer_replays_enqueued": 0,
            "metadata_fusion_fallbacks_by_batch": _map({}),
            "layer_batch_gate_attempts_by_batch": (
                ready_attempts if post else {str(item): 0 for item in range(1, batch + 1)}
            ),
            "layer_batch_gate_coverage_mask_by_batch": {
                str(item): 0 for item in range(1, batch + 1)
            },
        }
        return {
            "mode": "hydra27_fixed32",
            "metrics": {
                "committer": committer,
                "sfwd": {"steps": count},
                "dfwd": {"spans": count},
                "cfwd": {"spans": count},
            },
        }

    pre_path = arm / "logs/pre.json"
    post_path = arm / "logs/post.json"
    pre_raw = _write_json(pre_path, snapshot(post=False))
    post_raw = _write_json(post_path, snapshot(post=True))
    pre_ref = {"path": str(pre_path), "sha256": hashlib.sha256(pre_raw).hexdigest()}
    post_ref = {"path": str(post_path), "sha256": hashlib.sha256(post_raw).hexdigest()}
    interval = {
        "start_forward_step": 0,
        "end_forward_step": events,
        "expected_complete_events": events,
    }
    attempt_delta = {
        str(item): (attempts if item == batch else 0)
        for item in range(1, batch + 1)
    }
    qualification = {
        "pre_runtime_snapshot": pre_ref,
        "post_runtime_snapshot": post_ref,
        "forward_step_interval": interval,
        "qualification_coverage": {"attempt_delta_by_batch": attempt_delta},
        "acceptance_valid": False,
        "performance_measurement": False,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
    }
    if batch == 1:
        qualification.update(
            {
                "schema": "fr13-fixed32-task-boundary-v1",
                "run_classification": "cfwd_layer_batch_real_swe_qualification",
                "instance_id": tasks[0],
            }
        )
        qualification_path = (
            arm
            / "swe_out/verified/per_task"
            / tasks[0]
            / "fixed32_task_boundary.json"
        )
    else:
        qualification.update(
            {
                "schema": "fr13-fixed32-cfwd-b4-qualification-campaign-v1",
                "run_classification": "cfwd_layer_batch_real_swe_b4_qualification",
                "batch_size": 4,
                "concurrency": 4,
                "task_count": 4,
                "task_ids": tasks,
                "action_succeeded": True,
                "state": "coverage_incomplete",
            }
        )
        qualification_path = (
            arm
            / "swe_out/verified/fixed32_cfwd_b4_qualification_campaign.json"
        )
    _write_json(qualification_path, qualification)

    env_values = {
        "CUDAGRAPH_MODE": "FULL_AND_PIECEWISE",
        "ENFORCE_EAGER": "0",
        "FR13_CFWD_GPU_TIMER": "1",
        "FR13_DFWD_GPU_TIMER": "1",
        "FR13_DRAFT_VOCAB_K": "65536",
        "FR13_DRAFT_VOCAB_ROOT": "1",
        "FR13_FIXED32_B1_DIAGNOSTIC": "1" if batch == 1 else "0",
        "FR13_FIXED32_COMMITTER_BV64_WARP4": "1",
        "FR13_FIXED32_COMMITTER_DECAY_RING": "0",
        "FR13_FIXED32_COMMITTER_DIRECT_METADATA": "0",
        "FR13_FIXED32_COMMITTER_GATE_RING": "0",
        "FR13_FIXED32_COMMITTER_KNORM_RING": "0",
        "FR13_FIXED32_COMMITTER_LAYER_BATCH": "1",
        "FR13_FIXED32_COMMITTER_LAYER_BATCH_QUALIFICATION": "1",
        "FR13_FIXED32_COMMITTER_METADATA_FUSION": "0",
        "FR13_FIXED32_COMMITTER_STICKY_GUARD": "0",
        "FR13_FIXED32_MODE": "hydra27_fixed32",
        "FR13_SFWD_GPU_TIMER": "1",
        "MAX_NUM_SEQS": str(batch),
        "SWE_CONCURRENCY": str(batch),
    }
    container_env = arm / "container_env.txt"
    container_env.write_text(
        "".join(f"{key}={value}\n" for key, value in sorted(env_values.items())),
        encoding="ascii",
    )
    runtime_log = arm / "docker_after_tasks.log"
    runtime_log.write_text(
        "Graph capturing finished\n"
        + "".join(
            "[FR13_FIXED32_COMMIT_DEVICE_FILL] preseeded: "
            f"mode=hydra27_fixed32 B={item} path_cap=16 neutralizations=0 "
            "gathers=0 fused_calls=1 layer_batch=1 bv64_warp4=1 replays=1\n"
            for item in range(1, batch + 1)
        )
        + "[FR13_FIXED32_COMMIT_DEVICE_FILL ENGAGED] "
        "mode=hydra27_fixed32 B=1 fixed16 one-replay\n",
        encoding="ascii",
    )

    events_per_step = 1.0 if batch == 1 else 3.2
    sfwd_ms = 120.0 if batch == 1 else 100.0
    dfwd_ms = 30.0 if batch == 1 else 25.0
    cfwd_ms = 20.0 if batch == 1 else 15.0
    overhead_event_ms = 30.0 if batch == 1 else 20.0
    step_wall_ms = (
        sfwd_ms + dfwd_ms + cfwd_ms + overhead_event_ms * events_per_step
    )
    measurement = arm / "deploy_speed_fullwall.json"
    _write_json(
        measurement,
        {
            "schema": "fr13.measure.deploy_speed.v1",
            "regime": "deployment",
            "batch_size": batch,
            "n_tasks": batch,
            "task_instance_ids": tasks,
            "draft_vocab_k": 65536,
            "draft_vocab_root": 1,
            "draft_head_fp8": False,
            "engagement": {"engaged": True},
            "mandatory_weight_bytes": 32666638208,
            "weight_floor_ms": 119.658015414,
            "floor_is_full_step_hardware_floor": False,
            "step_wall_ms": step_wall_ms,
            "s_per_fwd_gpu_per_forward": sfwd_ms / 1000.0,
            "drafter_gpu_ms_per_step": dfwd_ms,
            "committer_gpu_ms_per_step": cfwd_ms,
            "events_per_step": events_per_step,
            "overhead_other_ms_per_event": overhead_event_ms,
            "accept_per_event": 3.0,
            "committed_per_event": 4.0,
            "derived_tps_fullstep_gpu": 20.0,
            "floor_ms": 119.658015414,
            "floor_ratio": step_wall_ms / 119.658015414,
            "measured_tps_fullstep_wall": 4.0 / (step_wall_ms / 1000.0),
            "wall_s_per_event": step_wall_ms / events_per_step / 1000.0,
        },
    )
    return {
        "arm": arm,
        "batch": batch,
        "subset": subset,
        "measurement": measurement,
        "runtime_log": runtime_log,
        "container_env": container_env,
        "post_path": post_path,
        "qualification_path": qualification_path,
    }


def _reduce(module, evidence):
    return module.reduce_result(
        arm_dir=evidence["arm"],
        batch_size=evidence["batch"],
        subset_path=evidence["subset"],
        measurement_path=evidence["measurement"],
        runtime_log=evidence["runtime_log"],
        container_env=evidence["container_env"],
        source_commit="a" * 40,
        runner_sha256="b" * 64,
    )


def test_runner_is_default_off_and_bakes_only_exact_b1_or_b4() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert 'FR13_RUN_COMMITTER_BV64_REAL:-0' in text
    assert 'FR13_COMMITTER_BV64_REAL_BATCH:-1' in text
    assert "subset_b1_diagnostic_one.json" in text
    assert "subset_b4_four.json" in text
    assert "FR13_COMMITTER_BV64_REAL_BATCH must be exactly 1 or 4" in text
    assert "MODE=hydra27_fixed32" in text
    assert "FR13_DRAFT_VOCAB_K=65536" in text
    assert "FR13_DRAFT_VOCAB_ROOT=1" in text
    assert "FR13_FIXED32_COMMITTER_LAYER_BATCH=1" in text
    assert "FR13_FIXED32_COMMITTER_BV64_WARP4=1" in text
    assert "FR13_FIXED32_COMMITTER_LAYER_BATCH_QUALIFICATION=1" in text
    assert "FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1" in text
    assert "scripts/fr13_committer_bv64_real_result.py" in text


def test_b1_reducer_proves_candidate_served_and_full_breakdown(tmp_path: Path) -> None:
    module = _load_reducer()
    evidence = _evidence(tmp_path, batch=1, events=10, attempts=2)

    result = _reduce(module, evidence)

    assert result["status"] == "PASS"
    assert result["candidate_served"] is True
    assert result["candidate_served_replays"] == 8
    assert result["qualification_reference_served_replays"] == 2
    assert result["fallback_replays"] == 0
    assert result["phase_breakdown_ms_per_step"] == {
        "wall": 200.0,
        "sfwd": 120.0,
        "dfwd": 30.0,
        "cfwd": 20.0,
        "other": 30.0,
    }


def test_b4_exact4_reducer_path_is_ready(tmp_path: Path) -> None:
    module = _load_reducer()
    evidence = _evidence(tmp_path, batch=4, events=20, attempts=1)

    result = _reduce(module, evidence)

    assert result["run_classification"] == (
        "real_swe_verified_exact4_b4_committer_bv64_diagnostic"
    )
    assert result["candidate_served_replays"] == 19
    assert result["committer_replay_delta_by_batch"] == {
        "1": 2,
        "2": 2,
        "3": 4,
        "4": 12,
    }
    assert result["phase_breakdown_ms_per_step"]["other"] == 64.0


def test_reducer_rejects_a_run_without_candidate_serving(tmp_path: Path) -> None:
    module = _load_reducer()
    evidence = _evidence(tmp_path, batch=1, events=4, attempts=4)

    with pytest.raises(module.ResultError, match="no BV64 candidate-served"):
        _reduce(module, evidence)


def test_reducer_rejects_any_metadata_fallback(tmp_path: Path) -> None:
    module = _load_reducer()
    evidence = _evidence(tmp_path, batch=1, events=10, attempts=2)
    post = json.loads(evidence["post_path"].read_text(encoding="ascii"))
    post["metrics"]["committer"]["metadata_fusion_fallbacks_by_batch"]["1"] = 1
    post_raw = _write_json(evidence["post_path"], post)
    qualification = json.loads(
        evidence["qualification_path"].read_text(encoding="ascii")
    )
    qualification["post_runtime_snapshot"]["sha256"] = hashlib.sha256(
        post_raw
    ).hexdigest()
    _write_json(evidence["qualification_path"], qualification)

    with pytest.raises(module.ResultError, match="metadata fallback"):
        _reduce(module, evidence)

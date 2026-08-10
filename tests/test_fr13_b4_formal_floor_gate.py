"""The B4 formal floor gate must be citable or fail closed -- never in between.

These tests drive the REAL reduction chain: synthetic per-task Prometheus
brackets are composed by scripts/fr13_measure.py cmd_deploy_speed (topology
classification + work-census cross-gate), and the resulting deploy_speed_fullwall.json
is then fed to scripts/fr13_b4_floor_gate_reduce.py.  Nothing is stubbed, so a
regression in either half is caught here.

The two traps that this gate exists to survive:

  * the NESTED-SUMMATION trap -- at B=4 every task opens its bracket on the same
    engine state, so summing the four deltas multiply-counts the shared prefix
    (measured inflation on real runroots: 1.7-2.6x).  A gate that accepted the
    sum would publish an inflated aggregate TPS as citable.
  * the CENSUS-MISMATCH trap -- the topology-blind work census is the independent
    witness on the bracket choice.  When it disagrees, the run must fail closed
    rather than report.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load("fr13_b4_floor_gate_reduce", "scripts/fr13_b4_floor_gate_reduce.py")
topo = _load("test_topology_fixtures", "tests/test_fr13_deploy_speed_bracket_topology.py")
measure = topo.measure

TASK_IDS = list(gate.EXACT4_TASK_IDS)
MODES = ("tail6_fixed32", "hydra27_fixed32")


# --------------------------------------------------------------------------- #
# fixtures: build a real deploy_speed_fullwall.json from synthetic brackets     #
# --------------------------------------------------------------------------- #
def _write_arm(
    arm_dir: Path,
    *,
    scale: float = 1.0,
    wall_scale: float = 1.0,
    census_batch: int = 4,
    census_steps: int | None = None,
    task_ids: list[str] | None = None,
    env: dict[str, str] | None = None,
    reduce_now: bool = True,
    arm_complete: bool = True,
) -> Path:
    """One B=4 arm: four NESTED brackets on a shared origin + a work census.

    The widest bracket (the last-closing task) spans the arm, so the census must
    agree with THAT bracket, not with the sum of all four.
    """
    ids = task_ids or TASK_IDS
    out_root = arm_dir / "swe_out"
    unit = topo._scale(topo.UNIT_B4, scale)
    # Scaling every counter uniformly leaves every derived RATE identical, which
    # would give a degenerate zero-variance interval. Real passes differ in rate,
    # so stretch the wall-time counters only: step wall (and therefore aggregate
    # and per-request TPS) moves while the step/event COUNTS -- and so the work
    # census the reduction is gated on -- stay put.
    for key in (measure.M_STEP_WALL_S, measure.M_DECODE_S):
        unit[key] = unit[key] * wall_scale
    origin = topo._scale(topo.UNIT_B4, 0.0)
    for k, instance_id in enumerate(ids):
        task = topo._write_bracket(
            out_root, instance_id, dict(origin), topo._scale(unit, float(k + 1))
        )
        os.utime(
            task / "vllm_metrics_post.txt",
            (1_700_000_000 + k, 1_700_000_000 + k),
        )

    # The widest bracket is task len(ids)-1 => (len(ids)) units of work. The
    # census must agree with THAT, not with the sum of all four brackets.
    steps = census_steps
    if steps is None:
        steps = round(unit[measure.M_DECODE_FWD_GPU_STEPS] * float(len(ids)))
    census = topo._write_census(
        arm_dir / "logs" / "fr13_fixed32_work_census.jsonl", steps, census_batch
    )

    env_state = {
        "FR13_MAMBA_SPEC_BLOCKS_CDIV": "0",
        "FR13_B4_TASK_REFILL": "0",
        "FR13_FULL_ATTN_KV_FP8": "0",
    }
    env_state.update(env or {})
    (arm_dir / "container_env.txt").write_text(
        "".join(f"{k}={v}\n" for k, v in sorted(env_state.items())), encoding="utf-8"
    )

    if reduce_now:
        topo._run(
            out_root,
            arm_dir / "deploy_speed_fullwall.json",
            batch_size=4,
            work_census=census,
        )
    else:
        # Mimic the campaign driver: it writes deploy_speed_${TAG}.json and does
        # NOT pass --work-census, so its aggregate is UNGATED.
        topo._run(out_root, arm_dir / "deploy_speed_gate0.json", batch_size=4)
    if arm_complete:
        (arm_dir / "arm_ended_at.txt").write_text("2026-08-10T00:00:00Z\n", encoding="utf-8")
    return arm_dir


def _build_gate_root(
    root: Path,
    *,
    n_passes: int = 4,
    modes: tuple[str, ...] = MODES,
    env: dict[str, str] | None = None,
) -> Path:
    for index in range(n_passes):
        pass_dir = root / f"pass_{index:02d}"
        (pass_dir).mkdir(parents=True, exist_ok=True)
        (pass_dir / "floor_order.txt").write_text(
            "TH" if index % 2 == 0 else "HT", encoding="utf-8"
        )
        for mode in modes:
            # Vary scale per pass so the between-pass interval is non-degenerate,
            # the way real trajectory variance is.
            _write_arm(
                pass_dir / f"{mode}_gatetest",
                scale=1.0 + 0.25 * index,
                wall_scale=1.0 + 0.125 * index,
                env=env,
            )
    return root


@pytest.fixture()
def gate_root(tmp_path):
    return _build_gate_root(tmp_path / "gate")


# --------------------------------------------------------------------------- #
# the nested-summation trap                                                    #
# --------------------------------------------------------------------------- #
def test_nested_brackets_are_not_summed_and_the_gate_reads_the_widest(tmp_path):
    """The whole point: a B=4 arm must reduce to the widest bracket, not the sum."""
    arm = _write_arm(tmp_path / "hydra27_fixed32_x")
    speed = json.loads((arm / "deploy_speed_fullwall.json").read_text())

    assert speed["bracket_reduction"]["topology"] == "nested"
    assert speed["bracket_reduction"]["distinct_bracket_origins"] == 1
    assert speed["bracket_reduction"]["closing_task"] == TASK_IDS[-1]

    record = gate.reduce_pass_arm(
        arm, mode="hydra27_fixed32", pass_index=0, floor_order="TH"
    )
    assert record["included"] is True, record["exclusion_reason"]

    # The naive sum of 1+2+3+4 units is 2.5x the widest (4 units) bracket. If the
    # gate had accepted the sum, every rate below would be inflated by that.
    summed = speed["bracket_reduction"]["summed_bracket_inflation"]
    assert summed[measure.M_DECODE_FWD_GPU_STEPS] == pytest.approx(2.5)


def test_a_summed_reduction_would_have_been_caught_by_the_census(tmp_path):
    """Pin the trap: a census matching the SUM cannot gate the widest bracket."""
    arm = tmp_path / "hydra27_fixed32_x"
    unit_steps = topo.UNIT_B4[measure.M_DECODE_FWD_GPU_STEPS]
    with pytest.raises(RuntimeError, match="disagrees with the arm work census"):
        _write_arm(arm, census_steps=int(unit_steps * (1 + 2 + 3 + 4)))


# --------------------------------------------------------------------------- #
# census mismatch must fail closed                                             #
# --------------------------------------------------------------------------- #
def test_census_mismatch_fails_closed_and_is_never_reported(tmp_path):
    with pytest.raises(RuntimeError, match="disagrees with the arm work census"):
        _write_arm(tmp_path / "tail6_fixed32_x", census_steps=99999)


def test_arm_without_a_deploy_speed_json_is_excluded_with_a_reason(tmp_path):
    arm = tmp_path / "tail6_fixed32_missing"
    arm.mkdir(parents=True)
    record = gate.reduce_pass_arm(arm, mode="tail6_fixed32", pass_index=0, floor_order=None)
    assert record["included"] is False
    assert "not a regular file" in record["exclusion_reason"]


def test_an_ungated_bracket_reduction_is_excluded(tmp_path):
    """A B4 aggregate with no work-census gate is exactly the invalidated artifact."""
    arm = _write_arm(tmp_path / "hydra27_fixed32_x")
    path = arm / "deploy_speed_fullwall.json"
    speed = json.loads(path.read_text())
    speed["bracket_reduction"]["work_census_gate"] = {"status": "absent"}
    path.write_text(json.dumps(speed), encoding="utf-8")

    record = gate.reduce_pass_arm(arm, mode="hydra27_fixed32", pass_index=0, floor_order=None)
    assert record["included"] is False
    assert "not work-census gated" in record["exclusion_reason"]


def test_a_disjoint_b4_bracket_is_refused(tmp_path):
    """Staggered/refilled admission misclassifies as disjoint; it is not citable."""
    arm = _write_arm(tmp_path / "hydra27_fixed32_x")
    path = arm / "deploy_speed_fullwall.json"
    speed = json.loads(path.read_text())
    speed["bracket_reduction"]["topology"] = "disjoint"
    path.write_text(json.dumps(speed), encoding="utf-8")

    record = gate.reduce_pass_arm(arm, mode="hydra27_fixed32", pass_index=0, floor_order=None)
    assert record["included"] is False
    assert "bracket topology is 'disjoint'" in record["exclusion_reason"]


def test_non_finite_numbers_are_rejected_on_ingest(tmp_path):
    arm = _write_arm(tmp_path / "hydra27_fixed32_x")
    path = arm / "deploy_speed_fullwall.json"
    path.write_text(path.read_text().replace('"step_wall_ms":', '"step_wall_ms": NaN, "_old":'))
    record = gate.reduce_pass_arm(arm, mode="hydra27_fixed32", pass_index=0, floor_order=None)
    assert record["included"] is False
    assert "non-finite" in record["exclusion_reason"]


# --------------------------------------------------------------------------- #
# statistical acceptance                                                       #
# --------------------------------------------------------------------------- #
def test_pinned_t_criticals_match_the_b1_floor_gate():
    """The between-pass interval must not invent a new critical value."""
    text = (REPO / "scripts" / "fr13_floor_gate.py").read_text(encoding="utf-8")
    assert "3: 2.3533634348018264" in text
    assert "15: 1.7530503556925547" in text
    assert gate.T95_ONE_SIDED == {3: 2.3533634348018264, 15: 1.7530503556925547}


def test_four_passes_yield_a_two_sided_interval_around_the_mean():
    values = [30.0, 32.0, 31.0, 33.0]
    stat = gate.cluster_interval(values)
    assert stat["cluster_count"] == 4
    assert stat["df"] == 3
    assert stat["point_estimate"] == pytest.approx(31.5)
    assert stat["t_0_95_one_sided"] == 2.3533634348018264
    # l95 is the citable side: throughput is claimed from below.
    assert stat["l95"] < stat["point_estimate"] < stat["u95"]
    half = stat["t_0_95_one_sided"] * stat["standard_error"]
    assert stat["l95"] == pytest.approx(stat["point_estimate"] - half)


@pytest.mark.parametrize("count", [1, 2, 3, 5, 8])
def test_only_four_or_sixteen_passes_are_admissible(count):
    with pytest.raises(gate.B4GateError, match="no pinned one-sided t critical"):
        gate.cluster_interval([float(i) for i in range(count)])


def test_full_gate_passes_and_is_citable(gate_root):
    payload = gate.build_verdict(
        repo=REPO,
        gate_root=gate_root,
        source_commit="deadbeef",
        topology_passes=gate.discover_passes(gate_root),
        min_passes=4,
    )
    assert payload["gate_verdict"] == "PASS"
    assert payload["analysis_valid"] is True
    assert payload["citable"] is True
    assert payload["formal_floor_acceptance_eligible"] is True
    assert payload["b4_cap_applicable"] is False
    for mode in MODES:
        topo_out = payload["topologies"][mode]
        assert topo_out["included_pass_count"] == 4
        agg = topo_out["measured_tps_fullstep_wall"]
        assert agg["l95"] < agg["point_estimate"] < agg["u95"]
        assert topo_out["per_request_step_tps"]["point_estimate"] > 0
        assert topo_out["events_per_step"]["point_estimate"] == pytest.approx(4.0)


def test_three_passes_degrade_to_not_evaluated(tmp_path):
    root = _build_gate_root(tmp_path / "gate", n_passes=3)
    payload = gate.build_verdict(
        repo=REPO, gate_root=root, source_commit="x",
        topology_passes=gate.discover_passes(root), min_passes=4,
    )
    assert payload["gate_verdict"] == "NOT_EVALUATED_INSUFFICIENT_PASSES"
    assert payload["citable"] is False
    assert payload["formal_floor_acceptance_eligible"] is False


def test_a_single_missing_topology_is_not_evaluated(tmp_path):
    root = _build_gate_root(tmp_path / "gate", modes=("hydra27_fixed32",))
    payload = gate.build_verdict(
        repo=REPO, gate_root=root, source_commit="x",
        topology_passes=gate.discover_passes(root), min_passes=4,
    )
    assert payload["gate_verdict"] == "NOT_EVALUATED_INSUFFICIENT_PASSES"
    assert payload["gates"]["both_topologies_present"] is False


# --------------------------------------------------------------------------- #
# stack identity: a flipped default is a different stack                       #
# --------------------------------------------------------------------------- #
def test_a_flipped_mamba_narrowing_default_makes_the_run_non_citable(tmp_path):
    root = _build_gate_root(
        tmp_path / "gate", env={"FR13_MAMBA_SPEC_BLOCKS_CDIV": "1"}
    )
    passes = gate.discover_passes(root)
    payload = gate.build_verdict(
        repo=REPO, gate_root=root, source_commit="x",
        topology_passes=passes, min_passes=4,
    )
    assert payload["citable"] is False
    excluded = passes["hydra27_fixed32"][0]
    assert excluded["included"] is False
    assert "FR13_MAMBA_SPEC_BLOCKS_CDIV" in excluded["exclusion_reason"]


def test_task_refill_makes_the_run_non_citable(tmp_path):
    """The driver states refill output is NOT exact4-citable without a contract update."""
    root = _build_gate_root(tmp_path / "gate", env={"FR13_B4_TASK_REFILL": "1"})
    payload = gate.build_verdict(
        repo=REPO, gate_root=root, source_commit="x",
        topology_passes=gate.discover_passes(root), min_passes=4,
    )
    assert payload["citable"] is False


def test_non_canonical_task_identities_are_refused(tmp_path):
    arm = _write_arm(
        tmp_path / "hydra27_fixed32_x",
        task_ids=["astropy__astropy-12907", "django__django-1", "a__b-2", "c__d-3"],
    )
    record = gate.reduce_pass_arm(arm, mode="hydra27_fixed32", pass_index=0, floor_order=None)
    assert record["included"] is False
    assert "canonical exact4 task identities" in record["exclusion_reason"]


# --------------------------------------------------------------------------- #
# determinism + serialization                                                  #
# --------------------------------------------------------------------------- #
def test_reduction_is_deterministic(gate_root):
    def once():
        payload = gate.build_verdict(
            repo=REPO, gate_root=gate_root, source_commit="x",
            topology_passes=gate.discover_passes(gate_root), min_passes=4,
        )
        payload.pop("generated_at_utc")
        return json.dumps(payload, sort_keys=True, allow_nan=False)

    assert once() == once()


def test_verdict_serializes_without_nan(gate_root):
    payload = gate.build_verdict(
        repo=REPO, gate_root=gate_root, source_commit="x",
        topology_passes=gate.discover_passes(gate_root), min_passes=4,
    )
    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def test_cli_writes_the_verdict_and_reports_pass(gate_root, capsys):
    rc = gate.main(["--repo", str(REPO), "--gate-root", str(gate_root)])
    assert rc == 0
    out = gate_root / "fr13_b4_formal_floor_gate.json"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "fr13.b4_formal_floor_gate.v1"
    assert payload["gate_verdict"] == "PASS"
    captured = capsys.readouterr().out
    assert "B4 FORMAL FLOOR GATE -- PASS" in captured


# --------------------------------------------------------------------------- #
# runner wiring                                                                #
# --------------------------------------------------------------------------- #
def test_runner_pins_the_citability_contract():
    text = (REPO / "scripts" / "fr13_b4_formal_floor_gate.sh").read_text(encoding="utf-8")
    # exact4, B=4, both topologies, refill off, no lever flipped, no agent wall.
    assert "subset_b4_four.json" in text
    assert "BSIZE=4" in text and "CONC=4" in text
    assert "fr13_fixed32_floor_timers_seq.sh" in text
    assert "FR13_B4_TASK_REFILL=0" in text
    assert "WALL=0" in text
    assert "FR13_FLOOR_ORDER" in text
    # preflight: the GPU must be uncontended before any launch.
    assert "docker ps -aq" in text


# --------------------------------------------------------------------------- #
# offline finalization: the driver's artifact is ungated and must be re-reduced #
# --------------------------------------------------------------------------- #
def test_driver_artifact_is_ungated_and_the_gate_refuses_it(tmp_path):
    """deploy_speed_${TAG}.json carries work_census_gate absent -> not citable."""
    arm = _write_arm(tmp_path / "tail6_fixed32_gate0", reduce_now=False)
    driver_artifact = json.loads((arm / "deploy_speed_gate0.json").read_text())
    assert driver_artifact["bracket_reduction"]["work_census_gate"]["status"] == "absent"

    record = gate.reduce_pass_arm(arm, mode="tail6_fixed32", pass_index=0, floor_order="TH")
    assert record["included"] is False
    assert "not a regular file" in record["exclusion_reason"]


def test_finalize_reruns_the_reduction_with_the_census(tmp_path):
    arm = _write_arm(tmp_path / "tail6_fixed32_gate0", reduce_now=False)
    action = gate.finalize_arm(arm, expected_tok_per_draft=32)
    assert action["action"] == "finalized"

    speed = json.loads((arm / gate.DEPLOY_SPEED_FILENAME).read_text())
    assert speed["bracket_reduction"]["work_census_gate"]["status"] == "pass"
    assert speed["bracket_reduction"]["topology"] == "nested"

    record = gate.reduce_pass_arm(arm, mode="tail6_fixed32", pass_index=0, floor_order="TH")
    assert record["included"] is True, record["exclusion_reason"]
    # Finalization must not change the measured aggregate -- only witness it.
    driver_artifact = json.loads((arm / "deploy_speed_gate0.json").read_text())
    assert record["measured_tps_fullstep_wall"] == pytest.approx(
        driver_artifact["measured_tps_fullstep_wall"]
    )


def test_finalize_is_idempotent(tmp_path):
    arm = _write_arm(tmp_path / "tail6_fixed32_gate0", reduce_now=False)
    assert gate.finalize_arm(arm, expected_tok_per_draft=32)["action"] == "finalized"
    assert gate.finalize_arm(arm, expected_tok_per_draft=32)["action"] == "already_present"


def test_finalize_refuses_an_arm_that_is_still_serving(tmp_path):
    """No arm_ended_at.txt => brackets/census still growing => torn read."""
    arm = _write_arm(
        tmp_path / "hydra27_fixed32_gate1", reduce_now=False, arm_complete=False
    )
    assert gate.finalize_arm(arm, expected_tok_per_draft=32)["action"] == "skipped_arm_still_serving"
    assert not (arm / gate.DEPLOY_SPEED_FILENAME).exists()


def test_finalize_without_a_census_fails_closed_rather_than_reducing(tmp_path):
    arm = _write_arm(tmp_path / "tail6_fixed32_gate0", reduce_now=False)
    (arm / gate.WORK_CENSUS_RELPATH).unlink()
    assert gate.finalize_arm(arm, expected_tok_per_draft=32)["action"] == "skipped_no_work_census"
    assert not (arm / gate.DEPLOY_SPEED_FILENAME).exists()


def test_finalize_gate_root_walks_every_pass(tmp_path):
    root = tmp_path / "gate"
    for index in range(4):
        pass_dir = root / f"pass_{index:02d}"
        pass_dir.mkdir(parents=True)
        for mode in MODES:
            _write_arm(pass_dir / f"{mode}_gate{index}", reduce_now=False)
    actions = gate.finalize_gate_root(root, expected_tok_per_draft=32)
    assert len(actions) == 8
    assert all(a["action"] == "finalized" for a in actions)

    payload = gate.build_verdict(
        repo=REPO, gate_root=root, source_commit="x",
        topology_passes=gate.discover_passes(root), min_passes=4,
    )
    assert payload["gate_verdict"] == "PASS"

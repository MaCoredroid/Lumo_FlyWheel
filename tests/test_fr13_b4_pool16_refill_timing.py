"""The pool16 refill TIMING class: binding, admission evidence, and its limits.

The class exists because scripts/fr13_b4_campaign_driver.sh:38-41 says refill output
"is NOT exact4-citable without a contract update".  This is that update -- and what it
updates is NOT the exact4 contract.  It defines a SEPARATE class with a different
subset, a different bracket topology, a different primary statistic, and an explicit
list of what it does not claim.
"""

from __future__ import annotations

import importlib.util
import os
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(SCRIPTS))
reduce_mod = _load("fr13_b4_pool16_reduce", "scripts/fr13_b4_floor_gate_reduce.py")
topo_fixtures = _load(
    "test_pool16_topology_fixtures", "tests/test_fr13_deploy_speed_bracket_topology.py"
)
measure = topo_fixtures.measure

POOL16 = "pool16_refill_timing"
EXACT4 = "exact4_formal_floor"
GATE_RUNNER = SCRIPTS / "fr13_b4_pool16_refill_gate.sh"


# --------------------------------------------------------------------------- #
# 1. the class label and its bindings                                          #
# --------------------------------------------------------------------------- #
def test_both_run_classes_exist_and_exact4_is_the_default() -> None:
    assert set(reduce_mod.RUN_CLASSES) == {EXACT4, POOL16}
    assert reduce_mod.DEFAULT_RUN_CLASS == EXACT4


def test_pool16_class_carries_its_own_schema_and_classification_token() -> None:
    spec = reduce_mod.RUN_CLASSES[POOL16]
    assert spec["schema"] == "fr13.b4_pool16_refill_timing.v1"
    assert spec["classification"] == "real_swe_verified_pool16_b4_refill_timing"
    # Distinct from the formal floor gate's, in BOTH fields -- a reader grepping
    # for either token must never collide the two classes.
    other = reduce_mod.RUN_CLASSES[EXACT4]
    assert spec["schema"] != other["schema"]
    assert spec["classification"] != other["classification"]


def test_pool16_binds_the_byte_pinned_sixteen_task_evidence_set() -> None:
    gate = _load("fr13_floor_gate_for_pool16", "scripts/fr13_floor_gate.py")
    spec = reduce_mod.RUN_CLASSES[POOL16]
    assert spec["task_count"] == 16
    assert spec["subset_sha256"] == gate.EVIDENCE_SETS[16]["sha256"]
    assert tuple(spec["task_ids"]) == tuple(gate.EVIDENCE_SETS[16]["task_ids"])
    # and the real file on disk still hashes to it
    subset = ROOT / gate.EVIDENCE_SETS[16]["relative_path"]
    assert gate.sha256_file(subset) == spec["subset_sha256"]
    # exact4 is the first four of the same ordering -- which is what makes a
    # CONTRAST legitimate and an equivalence not.
    assert tuple(spec["task_ids"])[:4] == tuple(reduce_mod.EXACT4_TASK_IDS)


def test_pool_ledger_constants_match_the_floor_gate_they_are_copied_from() -> None:
    """Reproduced, not imported -- so a test has to hold them together."""
    gate = _load("fr13_floor_gate_for_pool16b", "scripts/fr13_floor_gate.py")
    assert reduce_mod.MIN_POOL_TIME_WEIGHTED_DEPTH == gate.MIN_POOL_TIME_WEIGHTED_DEPTH
    assert reduce_mod.MIN_POOL_FULL_WIDTH_FRACTION == gate.MIN_POOL_FULL_WIDTH_FRACTION
    assert reduce_mod.TASK_REFILL_SUMMARY_SCHEMA == gate.TASK_REFILL_SUMMARY_SCHEMA


def test_the_exact4_class_still_agrees_with_the_pre_registry_constants() -> None:
    """The module-level exact4 constants predate the registry and stay authoritative.

    They are read by other call sites (tests/test_fr13_b4_formal_floor_gate.py:505),
    so the registry entry must not be free to drift away from them.
    """
    spec = reduce_mod.RUN_CLASSES[EXACT4]
    assert spec["required_bracket_topology"] == reduce_mod.REQUIRED_BRACKET_TOPOLOGY
    assert spec["contract_pinned_stack"] == reduce_mod.CONTRACT_PINNED_STACK
    assert spec["primary_statistic"] == reduce_mod.PRIMARY_STATISTIC
    assert tuple(spec["co_residency_dominated"]) == reduce_mod.CO_RESIDENCY_DOMINATED
    assert spec["schema"] == reduce_mod.SCHEMA
    assert spec["classification"] == reduce_mod.CLASSIFICATION
    assert spec["subset_sha256"] == reduce_mod.EXACT4_SUBSET_SHA256


def test_pool16_requires_staggered_brackets_and_exact4_still_requires_nested() -> None:
    assert reduce_mod.RUN_CLASSES[POOL16]["required_bracket_topology"] == "staggered"
    assert reduce_mod.RUN_CLASSES[EXACT4]["required_bracket_topology"] == "nested"


def test_pool16_pins_the_refill_flag_on_and_exact4_pins_it_off() -> None:
    assert reduce_mod.RUN_CLASSES[POOL16]["contract_pinned_stack"] == {
        "FR13_B4_TASK_REFILL": "1"
    }
    assert reduce_mod.RUN_CLASSES[EXACT4]["contract_pinned_stack"] == {
        "FR13_B4_TASK_REFILL": "0"
    }


def test_a_class_pin_beats_a_shipped_default() -> None:
    expected = reduce_mod.expected_stack(
        {"FR13_MAMBA_SPEC_BLOCKS_CDIV": "1", "FR13_B4_TASK_REFILL": "0"},
        reduce_mod.RUN_CLASSES[POOL16]["contract_pinned_stack"],
    )
    assert expected["FR13_B4_TASK_REFILL"] == "1"
    assert expected["FR13_MAMBA_SPEC_BLOCKS_CDIV"] == "1"


def test_unknown_run_class_fails_loud() -> None:
    with pytest.raises(reduce_mod.B4GateError, match="unknown run class"):
        reduce_mod.resolve_run_class("pool8_something")


def test_primary_statistic_is_inverted_and_the_inversion_is_explained() -> None:
    assert reduce_mod.RUN_CLASSES[EXACT4]["primary_statistic"] == "per_request_step_tps"
    assert (
        reduce_mod.RUN_CLASSES[POOL16]["primary_statistic"]
        == "measured_tps_fullstep_wall"
    )
    reason = reduce_mod.PRIMARY_STATISTIC_REASON[POOL16]
    assert "INVERTED" in reason
    assert "3c6d663d6" in reason  # the aggregate-up/per-request-down precedent


def test_pool16_disclaims_exact4_comparability_the_cap_and_agent_quality() -> None:
    text = " ".join(reduce_mod.RUN_CLASSES[POOL16]["does_not_claim"]).lower()
    assert "exact4 comparability" in text
    assert "cap verdict" in text
    assert "agent-quality" in text and "exact16" in text
    assert "ledger" in text  # the flag is evidence, not a lever


# --------------------------------------------------------------------------- #
# 2. the refill flag is evidence, not a schedule lever                         #
# --------------------------------------------------------------------------- #
def test_refill_flag_is_evidence_not_a_schedule_lever() -> None:
    """The comparison this rung was opened on is a structural null.

    Both admission paths hold `concurrency` jobs in flight and backfill on
    completion; ThreadPoolExecutor.map has done that since it existed.  This is
    asserted from the source rather than measured, which is exactly why no
    refill-OFF pool16 comparator arm is run: ~21 GPU-hours for a known null whose
    OFF half would not even emit a ledger to validate.
    """
    runner = (SCRIPTS / "run_swe_bench_q36_a.py").read_text(encoding="utf-8")
    assert "worker-level backfill is not new" in runner
    # ...and the OFF branch is still plain ex.map at max_workers=concurrency.
    assert "ThreadPoolExecutor(max_workers=args.concurrency) as ex" in runner
    assert "ex.map(_job, instance_ids)" in runner
    # The class says so in its own payload, not just in a design note.
    assert any(
        "admission LEDGER, not a schedule lever" in claim
        for claim in reduce_mod.RUN_CLASSES[POOL16]["does_not_claim"]
    )
    assert "not a schedule lever" in reduce_mod.read_pool_ledger.__doc__.lower() or (
        "THE FLAG IS EVIDENCE, NOT A SCHEDULE LEVER"
        in reduce_mod.read_pool_ledger.__doc__
    )


# --------------------------------------------------------------------------- #
# 3. the pool admission ledger gate                                            #
# --------------------------------------------------------------------------- #
def _write_pool_ledger(arm_dir: Path, **overrides) -> None:
    root = arm_dir / "swe_out" / "verified"
    root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "fr13.task_refill.summary.v1",
        "task_count": 16,
        "slots": 4,
        "arm_wall_s": 10720.646996,
        "time_weighted_mean_depth": 3.3987676579216792,
        "full_width_fraction": 0.7804274683348599,
        "admissions": 16,
        "peak_depth": 4,
        "completed": 16,
        "aborted": False,
    }
    summary.update(overrides)
    (root / "fr13_task_refill_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (root / "fr13_task_refill_ledger.jsonl").write_text(
        json.dumps({"event": "admit", "instance_id": "x", "t_s": 0.0, "depth": 1})
        + "\n",
        encoding="utf-8",
    )


def test_a_healthy_pool_ledger_passes(tmp_path: Path) -> None:
    _write_pool_ledger(tmp_path)
    out = reduce_mod.read_pool_ledger(tmp_path, slots=4, task_count=16)
    assert out["status"] == "pass"
    assert out["peak_depth"] == 4
    assert out["time_weighted_mean_depth"] == pytest.approx(3.3987676579216792)
    assert out["full_width_fraction"] == pytest.approx(0.7804274683348599)


def test_a_missing_ledger_is_fatal_for_a_pool_class(tmp_path: Path) -> None:
    with pytest.raises(reduce_mod.B4GateError, match="no pool admission ledger"):
        reduce_mod.read_pool_ledger(tmp_path, slots=4, task_count=16)


def test_half_present_pool_evidence_is_fatal(tmp_path: Path) -> None:
    _write_pool_ledger(tmp_path)
    (tmp_path / "swe_out" / "verified" / "fr13_task_refill_ledger.jsonl").unlink()
    with pytest.raises(reduce_mod.B4GateError, match="half-present"):
        reduce_mod.read_pool_ledger(tmp_path, slots=4, task_count=16)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"schema": "fr13.task_refill.summary.v0"}, "pool ledger schema"),
        ({"slots": 8}, "slots"),
        ({"task_count": 8}, "tasks, not 16"),
        ({"aborted": True}, "aborted"),
        ({"completed": 15}, "completed 15"),
        ({"peak_depth": 5}, "exceeds the slot count"),
        ({"time_weighted_mean_depth": 3.1999}, "below the pool floor"),
        ({"full_width_fraction": 0.59}, "full-width fraction"),
    ],
)
def test_pool_ledger_failures_name_the_measured_value(
    tmp_path: Path, overrides: dict, match: str
) -> None:
    _write_pool_ledger(tmp_path, **overrides)
    with pytest.raises(reduce_mod.B4GateError, match=match):
        reduce_mod.read_pool_ledger(tmp_path, slots=4, task_count=16)


def test_a_pool_at_slot_parity_is_refused(tmp_path: Path) -> None:
    _write_pool_ledger(tmp_path, task_count=4, slots=4)
    with pytest.raises(reduce_mod.B4GateError, match="LARGER than its slot count"):
        reduce_mod.read_pool_ledger(tmp_path, slots=4, task_count=4)


def test_the_occupancy_floors_are_satisfiable_by_a_real_recorded_run() -> None:
    """No unsatisfiable preconditions: both banked diagnostics clear both floors.

    The margin is thin on the second (3.234 against 3.200) and that is intended --
    a pool that did not hold the served width is not a pool run -- but a floor no
    recorded run has ever met would be the fossil this campaign keeps finding.
    """
    for depth, width in ((3.3987676579216792, 0.7804274683348599),
                         (3.2343978703023093, 0.71006273891935)):
        assert depth >= reduce_mod.MIN_POOL_TIME_WEIGHTED_DEPTH
        assert width >= reduce_mod.MIN_POOL_FULL_WIDTH_FRACTION


# --------------------------------------------------------------------------- #
# 4. arm admission: topology + task count are class-driven                     #
# --------------------------------------------------------------------------- #
def _arm(tmp_path: Path, *, name: str, staggered: bool, tasks: int) -> Path:
    """A reduced arm directory carrying a real deploy_speed_fullwall.json.

    Built by running fr13_measure.cmd_deploy_speed over synthetic brackets, so the
    topology is CLASSIFIED from recorded counters exactly as it is in production --
    nothing about the topology is hand-written into the artifact.
    """
    arm_dir = tmp_path / name
    out_root = arm_dir / "swe_out"
    out_root.mkdir(parents=True)
    unit = topo_fixtures.UNIT_B4
    for index in range(tasks):
        # STAGGERED: task k opens a hair after task k-1 opened, so the origins
        # differ but the windows overlap -- a refilled pool.
        # NESTED: every task opens on the same origin -- exact4's one-shot wave.
        pre = topo_fixtures._scale(unit, 0.01 * index if staggered else 0.0)
        post = topo_fixtures._scale(unit, float(index + 1))
        # Real canonical instance ids: the class binds task IDENTITY, not just count.
        task = topo_fixtures._write_bracket(
            out_root, reduce_mod.POOL16_TASK_IDS[index], pre, post
        )
        os.utime(
            task / "vllm_metrics_post.txt",
            (1_700_000_000 + index, 1_700_000_000 + index),
        )
    # The envelope (and the widest nested bracket) both span exactly `tasks`
    # units, so the topology-blind census must agree at that width.
    delta = topo_fixtures._scale(unit, float(tasks))
    steps = int(delta[measure.M_STEP_WALL_STEPS])
    census = arm_dir / "census.jsonl"
    topo_fixtures._write_census(
        census,
        steps=steps,
        batch_size=int(round(delta[measure.M_STEP_WALL_DRAFTS] / steps)),
    )
    out = arm_dir / "deploy_speed_fullwall.json"
    topo_fixtures._run(out_root, out, batch_size=4, work_census=census)
    (arm_dir / "arm_ended_at.txt").write_text("done\n", encoding="utf-8")
    (arm_dir / "container_env.txt").write_text(
        "FR13_B4_TASK_REFILL=1\nFR13_MAMBA_SPEC_BLOCKS_CDIV=1\n"
        "FR13_FULL_ATTN_KV_FP8=0\n",
        encoding="utf-8",
    )
    return arm_dir


def test_a_staggered_pool16_arm_is_admitted_by_the_pool16_class(tmp_path: Path) -> None:
    arm = _arm(tmp_path, name="hydra27_fixed32_pool0", staggered=True, tasks=16)
    _write_pool_ledger(arm)
    speed = json.loads((arm / "deploy_speed_fullwall.json").read_text())
    assert speed["bracket_reduction"]["topology"] == "staggered"
    record = reduce_mod.reduce_pass_arm(
        arm,
        mode="hydra27_fixed32",
        pass_index=0,
        floor_order="TH",
        expected={"FR13_B4_TASK_REFILL": "1", "FR13_MAMBA_SPEC_BLOCKS_CDIV": "1"},
        run_class=reduce_mod.RUN_CLASSES[POOL16],
    )
    assert record["included"] is True, record["exclusion_reason"]
    assert record["pool_ledger"]["status"] == "pass"


def test_the_pool16_class_refuses_a_nested_exact4_arm(tmp_path: Path) -> None:
    arm = _arm(tmp_path, name="hydra27_fixed32_pool0", staggered=False, tasks=4)
    _write_pool_ledger(arm)
    record = reduce_mod.reduce_pass_arm(
        arm,
        mode="hydra27_fixed32",
        pass_index=0,
        floor_order="TH",
        expected={"FR13_B4_TASK_REFILL": "1"},
        run_class=reduce_mod.RUN_CLASSES[POOL16],
    )
    assert record["included"] is False
    assert "tasks, not 16" in record["exclusion_reason"]


def test_the_exact4_class_still_refuses_a_staggered_pool_arm(tmp_path: Path) -> None:
    """The pin the campaign driver asked for: a pool run is not exact4 evidence."""
    arm = _arm(tmp_path, name="hydra27_fixed32_gate0", staggered=True, tasks=16)
    record = reduce_mod.reduce_pass_arm(
        arm,
        mode="hydra27_fixed32",
        pass_index=0,
        floor_order="TH",
        expected={"FR13_B4_TASK_REFILL": "0"},
        run_class=reduce_mod.RUN_CLASSES[EXACT4],
    )
    assert record["included"] is False
    assert "tasks, not 4" in record["exclusion_reason"]


def test_a_pool16_arm_without_a_ledger_is_excluded_not_crashed(tmp_path: Path) -> None:
    arm = _arm(tmp_path, name="hydra27_fixed32_pool0", staggered=True, tasks=16)
    record = reduce_mod.reduce_pass_arm(
        arm,
        mode="hydra27_fixed32",
        pass_index=0,
        floor_order="TH",
        expected={"FR13_B4_TASK_REFILL": "1"},
        run_class=reduce_mod.RUN_CLASSES[POOL16],
    )
    assert record["included"] is False
    assert "no pool admission ledger" in record["exclusion_reason"]


def test_a_pool16_arm_that_ran_the_flag_off_is_excluded(tmp_path: Path) -> None:
    arm = _arm(tmp_path, name="hydra27_fixed32_pool0", staggered=True, tasks=16)
    _write_pool_ledger(arm)
    (arm / "container_env.txt").write_text(
        "FR13_B4_TASK_REFILL=0\nFR13_MAMBA_SPEC_BLOCKS_CDIV=1\n", encoding="utf-8"
    )
    record = reduce_mod.reduce_pass_arm(
        arm,
        mode="hydra27_fixed32",
        pass_index=0,
        floor_order="TH",
        expected={"FR13_B4_TASK_REFILL": "1"},
        run_class=reduce_mod.RUN_CLASSES[POOL16],
    )
    assert record["included"] is False
    assert "did not run the shipped stack" in record["exclusion_reason"]


# --------------------------------------------------------------------------- #
# 5. envelope reduction on staggered fixtures                                  #
# --------------------------------------------------------------------------- #
def test_staggered_provenance_names_the_envelope_not_the_nested_bracket(
    tmp_path: Path,
) -> None:
    """The basis string used to fall through to the nested wording."""
    arm = _arm(tmp_path, name="hydra27_fixed32_pool0", staggered=True, tasks=16)
    bracket = json.loads((arm / "deploy_speed_fullwall.json").read_text())[
        "bracket_reduction"
    ]
    assert bracket["topology"] == "staggered"
    assert "ENVELOPE" in bracket["basis"]
    assert "nested bracket" not in bracket["basis"]
    assert bracket["work_census_gate"]["status"] == "pass"


def test_staggered_arms_publish_their_summed_bracket_inflation(
    tmp_path: Path,
) -> None:
    """Previously nested-only, which withheld the LARGER refill inflation."""
    arm = _arm(tmp_path, name="hydra27_fixed32_pool0", staggered=True, tasks=16)
    bracket = json.loads((arm / "deploy_speed_fullwall.json").read_text())[
        "bracket_reduction"
    ]
    inflation = bracket["summed_bracket_inflation"]
    assert inflation, "staggered arms must report what the naive sum would have said"
    # 16 overlapping brackets opening at 0.01*k and closing at k+1: summing them
    # multiply-counts several times over.
    assert all(value > 1.0 for value in inflation.values())


def test_nested_arms_still_publish_theirs(tmp_path: Path) -> None:
    arm = _arm(tmp_path, name="hydra27_fixed32_gate0", staggered=False, tasks=4)
    bracket = json.loads((arm / "deploy_speed_fullwall.json").read_text())[
        "bracket_reduction"
    ]
    assert bracket["topology"] == "nested"
    assert bracket["summed_bracket_inflation"]
    assert "widest" in bracket["basis"]


# --------------------------------------------------------------------------- #
# 6. contract round-trip                                                       #
# --------------------------------------------------------------------------- #
def _verdict(run_class: str, *, reference=None) -> dict:
    return reduce_mod.build_verdict(
        repo=ROOT,
        gate_root=ROOT / "output" / "nonexistent",
        source_commit="deadbeef",
        topology_passes={mode: [] for mode in reduce_mod.TOPOLOGIES},
        min_passes=4,
        expected={"FR13_B4_TASK_REFILL": "1"},
        run_class=reduce_mod.RUN_CLASSES[run_class],
        run_class_name=run_class,
        exact4_reference=reference,
    )


def test_the_pool16_contract_round_trips_into_the_payload() -> None:
    payload = _verdict(POOL16)
    spec = reduce_mod.RUN_CLASSES[POOL16]
    assert payload["schema"] == spec["schema"]
    assert payload["classification"] == spec["classification"]
    assert payload["run_class"] == POOL16
    assert payload["primary_statistic"] == spec["primary_statistic"]
    assert payload["contract"]["task_count"] == 16
    assert payload["contract"]["subset_sha256"] == spec["subset_sha256"]
    assert payload["contract"]["required_bracket_topology"] == "staggered"
    assert payload["contract"]["requires_pool_ledger"] is True
    assert payload["contract"]["contract_pinned_stack"] == {"FR13_B4_TASK_REFILL": "1"}
    assert payload["claims"] and payload["does_not_claim"]
    assert payload["b4_cap_applicable"] is False
    # JSON-serialisable with no NaN escape hatch, like every other verdict.
    json.dumps(payload, allow_nan=False)


def test_a_pool16_verdict_is_never_formal_floor_acceptance_evidence() -> None:
    payload = _verdict(POOL16)
    assert payload["formal_floor_acceptance_eligible"] is False
    # even in the hypothetical where every gate passed
    assert reduce_mod.RUN_CLASSES[POOL16]["schema"] != reduce_mod.SCHEMA


def test_the_exact4_contract_is_unchanged() -> None:
    payload = _verdict(EXACT4)
    assert payload["schema"] == "fr13.b4_formal_floor_gate.v1"
    assert payload["classification"] == "real_swe_verified_exact4_b4_formal_floor_gate"
    assert payload["primary_statistic"] == "per_request_step_tps"
    assert payload["contract"]["task_count"] == 4
    assert payload["contract"]["contract_pinned_stack"] == {"FR13_B4_TASK_REFILL": "0"}
    assert payload["exact4_contrast"] is None


def test_the_exact4_contrast_is_refused_when_a_shipped_default_differs() -> None:
    reference = {
        "schema": "fr13.b4_formal_floor_gate.v1",
        "citable": True,
        "measured_stack_state": {
            "FR13_B4_TASK_REFILL": "0",
            "FR13_MAMBA_SPEC_BLOCKS_CDIV": "0",
        },
        "topologies": {},
    }
    contrast = reduce_mod.build_exact4_contrast(
        {},
        reference,
        {"FR13_B4_TASK_REFILL": "1", "FR13_MAMBA_SPEC_BLOCKS_CDIV": "1"},
        POOL16,
    )
    assert contrast["status"] == "refused"
    assert "FR13_MAMBA_SPEC_BLOCKS_CDIV" in contrast["reason"]


def test_the_contrast_is_not_refused_for_the_pin_that_must_differ() -> None:
    """FR13_B4_TASK_REFILL differs BY CONSTRUCTION between the two classes.

    Demanding whole-stack equality would make this contrast unsatisfiable and put
    a sixth unsatisfiable precondition into the campaign.  Only the shipped-default
    levers are compared; the pins are recorded as deliberately different.
    """
    reference = {
        "citable": True,
        "measured_stack_state": {
            "FR13_B4_TASK_REFILL": "0",
            "FR13_MAMBA_SPEC_BLOCKS_CDIV": "1",
            "FR13_FULL_ATTN_KV_FP8": "0",
        },
        "topologies": {},
    }
    contrast = reduce_mod.build_exact4_contrast(
        {},
        reference,
        {
            "FR13_B4_TASK_REFILL": "1",
            "FR13_MAMBA_SPEC_BLOCKS_CDIV": "1",
            "FR13_FULL_ATTN_KV_FP8": "0",
        },
        POOL16,
    )
    assert contrast["status"] == "evaluated"
    assert "FR13_B4_TASK_REFILL" in contrast["deliberately_different_pins"]
    assert contrast["shipped_default_levers_compared"] == {
        "FR13_MAMBA_SPEC_BLOCKS_CDIV": "1",
        "FR13_FULL_ATTN_KV_FP8": "0",
    }


def test_the_exact4_contrast_is_refused_when_the_reference_is_not_citable() -> None:
    contrast = reduce_mod.build_exact4_contrast(
        {}, {"citable": False}, {"a": "1"}, POOL16
    )
    assert contrast["status"] == "refused"
    assert "not itself citable" in contrast["reason"]


def test_the_exact4_contrast_is_descriptive_and_lists_its_confounds() -> None:
    ref_stack = {"FR13_B4_TASK_REFILL": "0", "FR13_MAMBA_SPEC_BLOCKS_CDIV": "1"}
    pool_stack = {"FR13_B4_TASK_REFILL": "1", "FR13_MAMBA_SPEC_BLOCKS_CDIV": "1"}

    def _topo(agg: float, per_request: float, events: float) -> dict:
        def _stat(point: float) -> dict:
            return {"point_estimate": point, "l95": point * 0.9, "u95": point * 1.1}

        return {
            "analysis_valid": True,
            "measured_tps_fullstep_wall": _stat(agg),
            "events_per_step": _stat(events),
            "per_request_step_tps": _stat(per_request),
            "step_wall_ms": _stat(275.0),
            "prefill_frac": _stat(0.30),
        }

    pool = {"hydra27_fixed32": _topo(41.0, 21.3, 1.93)}
    reference = {
        "citable": True,
        "measured_stack_state": ref_stack,
        "topologies": {"hydra27_fixed32": _topo(34.47, 21.20, 1.637)},
    }
    contrast = reduce_mod.build_exact4_contrast(pool, reference, pool_stack, POOL16)
    assert contrast["status"] == "evaluated"
    assert contrast["role"] == "descriptive"
    assert len(contrast["confounds"]) >= 4
    assert any("TASK SET" in c for c in contrast["confounds"])
    assert any("ESTIMATOR" in c for c in contrast["confounds"])
    block = contrast["per_topology"]["hydra27_fixed32"]
    assert block["measured_tps_fullstep_wall"]["relative_change"] == pytest.approx(
        (41.0 - 34.47) / 34.47
    )
    assert block["per_request_non_regression"] is True
    assert contrast["per_request_non_regression_all_topologies"] is True
    assert "overlap is NOT evidence of equality" in contrast["separability_note"]


def test_an_aggregate_gain_bought_with_a_per_request_regression_is_flagged() -> None:
    """3c6d663d6: +17.2% aggregate while every request got 3% slower."""
    ref_stack = {"FR13_B4_TASK_REFILL": "0", "FR13_MAMBA_SPEC_BLOCKS_CDIV": "1"}
    pool_stack = {"FR13_B4_TASK_REFILL": "1", "FR13_MAMBA_SPEC_BLOCKS_CDIV": "1"}

    def _stat(point: float, spread: float) -> dict:
        return {
            "point_estimate": point,
            "l95": point - spread,
            "u95": point + spread,
        }

    def _topo(agg: float, per_request: float, spread: float) -> dict:
        return {
            "analysis_valid": True,
            "measured_tps_fullstep_wall": _stat(agg, spread),
            "events_per_step": _stat(1.9, spread),
            "per_request_step_tps": _stat(per_request, spread),
            "step_wall_ms": _stat(275.0, spread),
            "prefill_frac": _stat(0.3, spread),
        }

    pool = {"hydra27_fixed32": _topo(41.0, 17.0, 0.2)}
    reference = {
        "citable": True,
        "measured_stack_state": ref_stack,
        "topologies": {"hydra27_fixed32": _topo(34.47, 21.2, 0.2)},
    }
    contrast = reduce_mod.build_exact4_contrast(pool, reference, pool_stack, POOL16)
    block = contrast["per_topology"]["hydra27_fixed32"]
    assert block["measured_tps_fullstep_wall"]["relative_change"] > 0
    assert block["per_request_non_regression"] is False
    assert contrast["per_request_regressed_on"] == ["hydra27_fixed32"]
    assert contrast["per_request_non_regression_all_topologies"] is False


def test_two_passes_are_a_screen_not_a_citable_class() -> None:
    """No df=1 critical is invented to rescue a short campaign."""
    assert set(reduce_mod.T95_ONE_SIDED) == {3, 15}
    with pytest.raises(reduce_mod.B4GateError, match="4 or 16 included passes"):
        reduce_mod.cluster_interval([1.0, 2.0])


# --------------------------------------------------------------------------- #
# 7. the campaign runner                                                       #
# --------------------------------------------------------------------------- #
def test_the_pool16_gate_runner_binds_the_class_it_reduces_with() -> None:
    text = GATE_RUNNER.read_text(encoding="utf-8")
    assert "config/fr13_fixed32/subset_b4_sixteen.json" in text
    assert reduce_mod.RUN_CLASSES[POOL16]["subset_sha256"] in text
    assert "FR13_B4_TASK_REFILL=1" in text
    assert "--run-class \"$RUN_CLASS\"" in text
    assert "RUN_CLASS=pool16_refill_timing" in text
    assert "BSIZE=4" in text and "CONC=4" in text


def test_the_pool16_gate_runner_keeps_the_formal_gate_disciplines() -> None:
    text = GATE_RUNNER.read_text(encoding="utf-8")
    # pinned pass counts, order alternation, teardown evidence, --finalize
    assert "PASSES must be 4 or 16" in text
    assert "order=TH" in text and "order=HT" in text
    assert "--finalize" in text
    assert "docker ps -aq" in text
    assert "[f]r13_bigdenom_swe_serve_variant" in text
    assert "tracked worktree must be clean" in text
    assert "setsid nohup" in text  # the 120s-tool-timeout lesson


def test_the_pool16_verdict_does_not_overwrite_a_formal_floor_gate_artifact() -> None:
    text = GATE_RUNNER.read_text(encoding="utf-8")
    assert "fr13_b4_pool16_refill_gate.json" in text
    assert "--out \"$GATE_ROOT/fr13_b4_pool16_refill_gate.json\"" in text


def test_the_runner_resolves_its_run_class_before_spending_gpu_time() -> None:
    """Five fossils this campaign were runners bound to artifacts nothing wrote."""
    text = GATE_RUNNER.read_text(encoding="utf-8")
    preflight = text.split("mkdir -p \"$GATE_ROOT\"")[0]
    assert "resolve_run_class" in preflight
    assert "unsatisfiable" in text.lower() or "fossil" in text.lower()


def test_the_exact4_reference_is_optional_and_never_gates_citability() -> None:
    text = GATE_RUNNER.read_text(encoding="utf-8")
    assert "does NOT affect citability" in text
    assert "EXACT4_REFERENCE=\"\"" in text  # cleared, not fatal, when absent


# --------------------------------------------------------------------------- #
# 8. the in-pass gate and the wall-bracket basis                               #
# --------------------------------------------------------------------------- #
def test_the_preseed_needle_matches_what_the_engine_actually_prints() -> None:
    """3b3351cf1 added single_launch= to the emitted line and never updated the
    needle, silently disabling every fixed32 in-pass floor gate since 2026-08-02."""
    gate = _load("fr13_floor_gate_needle", "scripts/fr13_floor_gate.py")
    emitter = (
        ROOT / "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
    ).read_text(encoding="utf-8")
    assert "single_launch={int(fixed32_single_launch is not None)} " in emitter
    assert "single_launch=0 route_armed=1 selfcheck_armed=0" in gate.FIXED32_PRESEED
    # the needle must be a prefix-exact rendering of the f-string's literal parts
    for literal in ("[FR13_SUBTREE_PARALLEL] preseeded: n=", "schedule=", "levels=",
                    "lens=", "critical=", "(monolith ", "single_launch=",
                    "route_armed=", "selfcheck_armed="):
        assert literal in emitter
        assert literal.split("{")[0] in gate.FIXED32_PRESEED


def test_the_driver_passes_the_work_census_when_the_arm_wrote_one() -> None:
    """A STAGGERED arm cannot be reduced without it -- fr13_measure.py:1744."""
    driver = (SCRIPTS / "fr13_b4_campaign_driver.sh").read_text(encoding="utf-8")
    assert "--work-census" in driver
    assert "fr13_fixed32_work_census.jsonl" in driver
    assert "NO work census" in driver  # absent census stays ungated, not fatal


def test_wall_retention_is_reported_on_every_arm(tmp_path: Path) -> None:
    arm = _arm(tmp_path, name="hydra27_fixed32_pool0", staggered=True, tasks=16)
    _write_pool_ledger(arm)
    record = reduce_mod.reduce_pass_arm(
        arm,
        mode="hydra27_fixed32",
        pass_index=0,
        floor_order="TH",
        expected={"FR13_B4_TASK_REFILL": "1", "FR13_MAMBA_SPEC_BLOCKS_CDIV": "1"},
        run_class=reduce_mod.RUN_CLASSES[POOL16],
    )
    assert record["included"] is True, record["exclusion_reason"]
    retention = record["wall_retention"]
    assert retention is not None
    assert 0.0 < retention["retained_wall_fraction"] <= 1.0
    assert "wall-chain reset" in retention["basis_note"]


def test_the_class_discloses_the_in_pass_gate_and_the_wall_basis() -> None:
    text = " ".join(reduce_mod.RUN_CLASSES[POOL16]["does_not_claim"])
    assert "in-pass fr13_floor_gate.py vouched" in text
    assert "citable exact4 ON gate included" in text
    assert "share one basis" in text
    assert "retained_wall_fraction" in text


def test_the_in_pass_gate_verdict_is_recorded_but_never_gates(tmp_path: Path) -> None:
    """It has never reached a verdict in this generation; hiding that would let a
    reader assume provenance the arm does not have."""
    pass_dir = tmp_path / "pass_00"
    arm = _arm(pass_dir, name="hydra27_fixed32_pool0", staggered=True, tasks=16)
    _write_pool_ledger(arm)
    _arm(pass_dir, name="tail6_fixed32_pool0", staggered=True, tasks=16)
    _write_pool_ledger(pass_dir / "tail6_fixed32_pool0")
    (pass_dir / "floor_order.txt").write_text("TH\n", encoding="utf-8")
    (pass_dir / "fixed32_floor_gate.json").write_text(
        json.dumps(
            {
                "gate_verdict": "NOT_EVALUATED_INVALID_INPUT",
                "analysis_valid": False,
                "error": "retained wall fraction 0.910602 is below 0.990000",
            }
        ),
        encoding="utf-8",
    )
    found = reduce_mod.discover_passes(
        tmp_path,
        {"FR13_B4_TASK_REFILL": "1", "FR13_MAMBA_SPEC_BLOCKS_CDIV": "1"},
        reduce_mod.RUN_CLASSES[POOL16],
    )
    record = found["hydra27_fixed32"][0]
    assert record["in_pass_floor_gate"]["gate_verdict"] == "NOT_EVALUATED_INVALID_INPUT"
    assert record["in_pass_floor_gate"]["role"] == "reported_never_gated"
    # ...and it did NOT gate: the arm is still included on its own evidence.
    assert record["included"] is True, record["exclusion_reason"]

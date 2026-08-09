"""cmd_deploy_speed must compose per-task /metrics brackets on their TOPOLOGY.

Sequential (B=1) arms leave DISJOINT brackets that tile the arm, so the arm
delta is their sum -- and that path must stay byte-identical to the historical
reduction.  Concurrent (B>1) arms open every bracket on the SAME counter origin,
so the brackets NEST and summing them multiply-counts the shared prefix; there
the arm delta is the widest (last-closing) bracket.  Either way the choice is
cross-gated against the arm's own topology-blind work census, and disagreement
fails closed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fr13_measure", REPO / "scripts" / "fr13_measure.py"
)
assert SPEC is not None
assert SPEC.loader is not None
measure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(measure)

RUNNER = REPO / "scripts" / "fr13_run_b4_cutlass_persistent_m128_timing.sh"

# One task's worth of engine work at co-residency 1: every pure-decode step
# carries exactly one request event, so steps == events.
UNIT_B1 = {
    measure.M_DECODE_S: 10.0,
    measure.M_DRAFTS: 25.0,
    measure.M_ACCEPTED: 250.0,
    measure.M_DRAFT_TOK: 800.0,  # 32 tok/draft
    measure.M_GEN_TOK: 275.0,
    measure.M_TPOT_SUM: 2.0,
    measure.M_TPOT_COUNT: 20.0,
    measure.M_PREFILL_S: 5.0,
    measure.M_DECODE_FWD_GPU_S: 4.0,
    measure.M_DECODE_FWD_GPU_STEPS: 25.0,
    measure.M_DECODE_FWD_GPU_DRAFTS: 25.0,
    measure.M_DRAFTER_GPU_S: 1.0,
    measure.M_DRAFTER_GPU_SPANS: 25.0,
    measure.M_COMMITTER_GPU_S: 0.5,
    measure.M_COMMITTER_GPU_SPANS: 25.0,
    measure.M_STEP_WALL_S: 6.0,
    measure.M_STEP_WALL_DRAFTS: 25.0,
    measure.M_STEP_WALL_STEPS: 25.0,
}

# The same, but every pure-decode step carries four co-resident request events.
UNIT_B4 = dict(UNIT_B1)
UNIT_B4[measure.M_DECODE_FWD_GPU_DRAFTS] = 100.0
UNIT_B4[measure.M_STEP_WALL_DRAFTS] = 100.0

EXPECTED_TOK_PER_DRAFT = 32.0


def _scale(unit: dict[str, float], factor: float) -> dict[str, float]:
    return {name: value * factor for name, value in unit.items()}


def _write_prom(path: Path, sample: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for name, value in sample.items():
        lines.append(f"# TYPE {name} counter")
        lines.append(f'{name}{{engine="0",model_name="m"}} {value}')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_bracket(
    root: Path, instance_id: str, pre: dict[str, float], post: dict[str, float]
) -> Path:
    task = root / "verified" / "per_task" / instance_id
    _write_prom(task / "vllm_metrics_pre.txt", pre)
    _write_prom(task / "vllm_metrics_post.txt", post)
    return task


def _write_census(path: Path, steps: int, batch_size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index in range(steps):
            handle.write(
                json.dumps(
                    {
                        "event_complete": True,
                        "event_index": index,
                        "batch_size": batch_size,
                        "batch_purity": {
                            "mixed_pseudo_rows": 0,
                            "batch_rows": batch_size,
                        },
                    }
                )
                + "\n"
            )
    return path


def _sequential_arm(root: Path, n_tasks: int = 4) -> None:
    """Concurrency 1: task k's bracket starts where task k-1's ended."""
    for k in range(n_tasks):
        _write_bracket(
            root,
            f"seq-{k}",
            _scale(UNIT_B1, float(k)),
            _scale(UNIT_B1, float(k + 1)),
        )


def _concurrent_arm(root: Path, n_tasks: int = 4) -> list[Path]:
    """Concurrency 4: every bracket opens at t0; task k closes after k+1 units."""
    origin = _scale(UNIT_B4, 0.0)
    dirs = []
    for k in range(n_tasks):
        task = _write_bracket(
            root, f"conc-{k}", dict(origin), _scale(UNIT_B4, float(k + 1))
        )
        # Post mtimes must order the closings; conc-3 is the widest bracket.
        os.utime(task / "vllm_metrics_post.txt", (1_700_000_000 + k, 1_700_000_000 + k))
        dirs.append(task)
    return dirs


def _run(out_root: Path, out: Path, batch_size: int, work_census: Path | None = None):
    args = argparse.Namespace(
        arm="test-arm",
        out_root=str(out_root),
        expected_tok_per_draft=EXPECTED_TOK_PER_DRAFT,
        batch_size=batch_size,
        basis="decode_seconds",
        work_census=None if work_census is None else str(work_census),
        out=str(out),
    )
    measure.cmd_deploy_speed(args)
    return json.loads(out.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# _bracket_reduce: the topology classifier itself                              #
# --------------------------------------------------------------------------- #
def test_disjoint_brackets_are_summed_and_are_the_identical_object(tmp_path):
    """B=1 must not merely agree with the old sum -- it must BE the old sum."""
    root = tmp_path / "swe_out"
    _sequential_arm(root)
    task_dirs = measure._find_task_dirs(str(root))
    assert len(task_dirs) == 4

    reduced = measure._bracket_reduce(task_dirs)

    assert reduced["topology"] == "disjoint"
    assert reduced["closing_task"] is None
    assert reduced["distinct_bracket_origins"] == 4
    # The reduced delta is the summed delta, not a recomputation of it: no float
    # op differs from the historical accumulation.
    assert reduced["delta"] is reduced["summed_delta"]

    expected = _scale(UNIT_B1, 4.0)
    for counter in measure.COUNTERS:
        assert reduced["delta"][counter] == pytest.approx(expected[counter], rel=0, abs=1e-9)


def test_nested_brackets_reduce_to_the_widest_not_the_sum(tmp_path):
    root = tmp_path / "swe_out"
    _concurrent_arm(root)
    task_dirs = measure._find_task_dirs(str(root))

    reduced = measure._bracket_reduce(task_dirs)

    assert reduced["topology"] == "nested"
    assert reduced["closing_task"] == "conc-3"
    assert reduced["distinct_bracket_origins"] == 1

    widest = _scale(UNIT_B4, 4.0)
    naive_sum = _scale(UNIT_B4, 1.0 + 2.0 + 3.0 + 4.0)
    for counter in measure.COUNTERS:
        assert reduced["delta"][counter] == pytest.approx(widest[counter], rel=0, abs=1e-9)
        assert reduced["summed_delta"][counter] == pytest.approx(
            naive_sum[counter], rel=0, abs=1e-9
        )
        # The defect this fixes: the naive sum is 2.5x the truth here.
        assert reduced["summed_delta"][counter] == pytest.approx(
            2.5 * reduced["delta"][counter], rel=1e-12
        )


def test_single_task_arm_is_never_classified_nested(tmp_path):
    root = tmp_path / "swe_out"
    _write_bracket(root, "solo", _scale(UNIT_B1, 0.0), _scale(UNIT_B1, 1.0))
    reduced = measure._bracket_reduce(measure._find_task_dirs(str(root)))
    assert reduced["topology"] == "disjoint"
    assert reduced["delta"][measure.M_DRAFTS] == pytest.approx(25.0)


# --------------------------------------------------------------------------- #
# cmd_deploy_speed end to end                                                  #
# --------------------------------------------------------------------------- #
def test_b1_aggregate_is_the_sum_and_records_disjoint_provenance(tmp_path):
    root = tmp_path / "swe_out"
    _sequential_arm(root)
    rec = _run(root, tmp_path / "b1.json", batch_size=1)

    assert rec["bracket_reduction"]["topology"] == "disjoint"
    assert rec["bracket_reduction"]["closing_task"] is None
    assert rec["bracket_reduction"]["work_census_gate"] == {"status": "absent"}
    # A disjoint reduction discards nothing, so there is no inflation to report.
    assert "summed_bracket_inflation" not in rec["bracket_reduction"]

    # 4 x UNIT_B1 exactly, as the historical sum would have produced.
    assert rec["n_tasks"] == 4
    assert rec["s_per_fwd"] == pytest.approx(40.0 / 100.0)
    assert rec["s_per_fwd_gpu"] == pytest.approx(16.0 / 100.0)
    assert rec["s_per_fwd_gpu_per_forward"] == pytest.approx(16.0 / 100.0)
    assert rec["events_per_step"] == pytest.approx(1.0)
    assert rec["accept_per_event"] == pytest.approx(1000.0 / 100.0)
    assert rec["engagement"]["tok_per_draft"] == pytest.approx(EXPECTED_TOK_PER_DRAFT)


def test_b4_aggregate_uses_the_authoritative_bracket(tmp_path):
    root = tmp_path / "swe_out"
    _concurrent_arm(root)
    rec = _run(root, tmp_path / "b4.json", batch_size=4)

    reduction = rec["bracket_reduction"]
    assert reduction["topology"] == "nested"
    assert reduction["closing_task"] == "conc-3"
    assert reduction["summed_bracket_inflation"][measure.M_DECODE_S] == pytest.approx(2.5)

    # Every rate is on the widest bracket (4 x UNIT_B4), NOT the 10x sum.
    assert rec["n_tasks"] == 4
    assert rec["events_per_step"] == pytest.approx(4.0)
    assert rec["s_per_fwd_gpu"] == pytest.approx(16.0 / 400.0)
    assert rec["s_per_fwd_gpu_per_forward"] == pytest.approx(16.0 / 100.0)
    # A ratio is immune to the inflation, so it must be unchanged by the fix.
    assert rec["accept_per_event"] == pytest.approx(10.0)


def test_b4_without_a_census_warns_that_the_aggregate_is_ungated(tmp_path, capsys):
    root = tmp_path / "swe_out"
    _concurrent_arm(root)
    _run(root, tmp_path / "b4.json", batch_size=4)
    err = capsys.readouterr().err
    assert "nested per-task /metrics brackets" in err
    assert "--work-census" in err


# --------------------------------------------------------------------------- #
# the work-census cross-gate                                                   #
# --------------------------------------------------------------------------- #
def test_census_gate_passes_when_the_engine_agrees(tmp_path):
    root = tmp_path / "swe_out"
    _concurrent_arm(root)
    # Widest bracket = 4 x UNIT_B4 => 100 pure-decode steps, 4 events each.
    census = _write_census(tmp_path / "work_census.jsonl", steps=100, batch_size=4)

    rec = _run(root, tmp_path / "b4.json", batch_size=4, work_census=census)

    gate = rec["bracket_reduction"]["work_census_gate"]
    assert gate["status"] == "pass"
    assert gate["census_steps"] == 100
    assert gate["census_events"] == 400
    assert gate["census_events_per_step"] == pytest.approx(4.0)
    assert gate["census_event_counts_by_batch"] == {"4": 100}


def test_census_gate_would_have_rejected_the_naive_sum(tmp_path):
    """The gate is what makes the fix self-checking: feed it the counts a
    SUMMING reducer would have produced and it must refuse."""
    root = tmp_path / "swe_out"
    _concurrent_arm(root)
    # 10 x UNIT_B4 is what summing the four nested brackets yields.
    census = _write_census(tmp_path / "work_census.jsonl", steps=250, batch_size=4)

    with pytest.raises(RuntimeError) as excinfo:
        _run(root, tmp_path / "b4.json", batch_size=4, work_census=census)
    message = str(excinfo.value)
    assert "work census" in message
    assert "nested" in message


def _staggered_arm(root: Path, n_tasks: int = 4) -> None:
    """A refilled pool: origins differ (not nested) but the windows overlap
    heavily (not disjoint), so neither historical reduction applies."""
    for k in range(n_tasks):
        task = root / "verified" / "per_task" / f"stag-{k}"
        _write_bracket(
            root,
            f"stag-{k}",
            _scale(UNIT_B4, 0.01 * k),
            _scale(UNIT_B4, float(k + 1)),
        )
        os.utime(
            task / "vllm_metrics_post.txt",
            (1_700_000_000 + k, 1_700_000_000 + k),
        )


def test_staggered_brackets_reduce_to_the_envelope_not_the_sum(tmp_path):
    """Partially-overlapping brackets match NEITHER historical topology.

    They must reduce to the ENVELOPE max(post)-min(pre), which spans the arm
    exactly once, instead of the sum, which multiply-counts every overlap.
    """
    root = tmp_path / "swe_out"
    _staggered_arm(root)

    reduced = measure._bracket_reduce(measure._find_task_dirs(str(root)))

    assert reduced["topology"] == "staggered"
    assert reduced["distinct_bracket_origins"] == 4
    assert reduced["closing_task"] == "stag-3"

    envelope = _scale(UNIT_B4, 4.0)
    naive_sum = _scale(UNIT_B4, (1.0 + 2.0 + 3.0 + 4.0) - 0.01 * (0 + 1 + 2 + 3))
    for counter in measure.COUNTERS:
        assert reduced["delta"][counter] == pytest.approx(
            envelope[counter], rel=0, abs=1e-9
        )
        assert reduced["summed_delta"][counter] == pytest.approx(
            naive_sum[counter], rel=0, abs=1e-9
        )
        # The defect this fixes: the naive sum is 2.485x the truth here.
        assert reduced["summed_delta"][counter] == pytest.approx(
            2.485 * reduced["delta"][counter], rel=1e-12
        )


def test_staggered_envelope_agrees_with_the_work_census(tmp_path):
    """The envelope spans the arm once, so it must equal the topology-blind
    census -- which is what makes a refilled pool citable at all."""
    root = tmp_path / "swe_out"
    _staggered_arm(root)
    census = _write_census(tmp_path / "work_census.jsonl", steps=100, batch_size=4)

    rec = _run(root, tmp_path / "b4.json", batch_size=4, work_census=census)

    assert rec["bracket_reduction"]["topology"] == "staggered"
    gate = rec["bracket_reduction"]["work_census_gate"]
    assert gate["status"] == "pass"
    assert gate["census_steps"] == 100
    assert gate["census_events"] == 400


def test_staggered_admission_still_fails_closed_on_census_disagreement(tmp_path):
    """The envelope is not exempt from the cross-gate: a census that disagrees
    with it still kills the aggregate."""
    root = tmp_path / "swe_out"
    _staggered_arm(root)
    census = _write_census(tmp_path / "work_census.jsonl", steps=50, batch_size=4)

    with pytest.raises(RuntimeError) as excinfo:
        _run(root, tmp_path / "b4.json", batch_size=4, work_census=census)
    assert "disagrees with the arm work census" in str(excinfo.value)
    assert "staggered" in str(excinfo.value)


def test_staggered_admission_requires_a_census_at_all(tmp_path):
    """The envelope keeps no independent per-task check, so an absent census is
    fatal for this topology rather than a warning."""
    root = tmp_path / "swe_out"
    _staggered_arm(root)

    with pytest.raises(RuntimeError) as excinfo:
        _run(root, tmp_path / "b4.json", batch_size=4, work_census=None)
    assert "work census is MANDATORY" in str(excinfo.value)


def test_census_gate_also_binds_a_sequential_arm(tmp_path):
    root = tmp_path / "swe_out"
    _sequential_arm(root)
    census = _write_census(tmp_path / "work_census.jsonl", steps=100, batch_size=1)
    rec = _run(root, tmp_path / "b1.json", batch_size=1, work_census=census)
    assert rec["bracket_reduction"]["work_census_gate"]["status"] == "pass"


def test_missing_census_file_is_fatal_when_requested(tmp_path):
    root = tmp_path / "swe_out"
    _concurrent_arm(root)
    with pytest.raises(RuntimeError, match="is not a regular file"):
        _run(
            root,
            tmp_path / "b4.json",
            batch_size=4,
            work_census=tmp_path / "absent.jsonl",
        )


def test_census_with_mixed_rows_cannot_gate_pure_decode_counters(tmp_path):
    root = tmp_path / "swe_out"
    _concurrent_arm(root)
    census = tmp_path / "work_census.jsonl"
    census.write_text(
        json.dumps(
            {
                "event_complete": True,
                "batch_size": 4,
                "batch_purity": {"mixed_pseudo_rows": 2, "batch_rows": 4},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="mixed pseudo rows"):
        _run(root, tmp_path / "b4.json", batch_size=4, work_census=census)


def test_empty_census_cannot_gate(tmp_path):
    root = tmp_path / "swe_out"
    _concurrent_arm(root)
    census = tmp_path / "work_census.jsonl"
    census.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no\n?\\s*completed pure-decode event"):
        _run(root, tmp_path / "b4.json", batch_size=4, work_census=census)


# --------------------------------------------------------------------------- #
# wiring                                                                       #
# --------------------------------------------------------------------------- #
def test_deploy_speed_exposes_the_work_census_flag():
    parser = measure.build_parser()
    args = parser.parse_args(
        [
            "deploy-speed",
            "--arm", "a",
            "--out-root", "r",
            "--expected-tok-per-draft", "31",
            "--out", "o.json",
        ]
    )
    assert args.work_census is None


def test_b4_runner_gates_the_reduction_on_the_work_census():
    text = RUNNER.read_text(encoding="utf-8")
    assert "--work-census" in text
    assert "logs/fr13_fixed32_work_census.jsonl" in text
    assert "lacks the work census the B4 bracket reduction is gated on" in text
    # And the timing summary refuses an ungated aggregate.
    assert "bracket reduction was not work-census gated" in text
    assert "deploy-speed carries no bracket-topology provenance" in text

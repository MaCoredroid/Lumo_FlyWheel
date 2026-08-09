from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fr13_b4_alignment_reduce",
    REPO / "scripts" / "fr13_b4_alignment_reduce.py",
)
assert SPEC is not None
assert SPEC.loader is not None
reducer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reducer)


# One synthetic task's worth of engine work.  Every quantity is a round number
# so the derived tables can be asserted exactly.
UNIT = {
    "vllm:prompt_tokens_total": 1000.0,
    "vllm:prompt_tokens_cached_total": 800.0,
    "vllm:generation_tokens_total": 100.0,
    "vllm:request_prompt_tokens_count": 2.0,
    "vllm:request_prefill_kv_computed_tokens_sum": 200.0,
    "vllm:prefix_cache_queries_total": 1000.0,
    "vllm:prefix_cache_hits_total": 800.0,
    "vllm:spec_decode_num_accepted_tokens_total": 30.0,
    "vllm:spec_decode_num_draft_tokens_total": 300.0,
    "vllm:fr13_decode_forward_gpu_seconds_total": 1.0,
    "vllm:fr13_decode_forward_gpu_steps_total": 10.0,
    "vllm:fr13_decode_forward_gpu_drafts_total": 10.0,
    "vllm:fr13_drafter_gpu_seconds_total": 0.2,
    "vllm:fr13_drafter_gpu_spans_total": 10.0,
    "vllm:fr13_committer_gpu_seconds_total": 0.1,
    "vllm:fr13_committer_gpu_spans_total": 10.0,
    "vllm:fr13_decode_step_wall_seconds_total": 1.5,
    "vllm:fr13_decode_step_wall_steps_total": 10.0,
    "vllm:fr13_decode_step_wall_drafts_total": 10.0,
    "vllm:request_prefill_time_seconds_sum": 1.0,
    "vllm:request_decode_time_seconds_sum": 2.0,
    "vllm:request_queue_time_seconds_sum": 0.0,
    "vllm:time_to_first_token_seconds_sum": 1.0,
    "vllm:e2e_request_latency_seconds_sum": 3.0,
    "vllm:iteration_tokens_total_count": 10.0,
    "vllm:num_preemptions_total": 0.0,
    "vllm:spec_decode_num_drafts_total": 8.0,
}


def _scale(factor: float) -> dict[str, float]:
    return {name: value * factor for name, value in UNIT.items()}


def _write_prom(path: Path, sample: dict[str, float]) -> None:
    lines = []
    for name, value in sample.items():
        lines.append(f"# TYPE {name} counter")
        lines.append(f'{name}{{engine="0",model_name="m"}} {value}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_census(path: Path, batch_sizes: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, size in enumerate(batch_sizes):
            handle.write(
                json.dumps(
                    {
                        "event_complete": True,
                        "event_index": index,
                        "forward_step_index": index,
                        "batch_size": size,
                        "batch_purity": {"mixed_pseudo_rows": 0, "batch_rows": size},
                        "tree_attn": {"physical_parent_digest": "d0"},
                    }
                )
                + "\n"
            )
        handle.write(json.dumps({"event_complete": False, "schema": "footer"}) + "\n")


def _write_ledger(path: Path, turns_by_task: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "campaign_begin"}) + "\n")
        for task_key, count in turns_by_task.items():
            for _ in range(count):
                handle.write(
                    json.dumps({"event": "logical_complete", "task_key_id": task_key}) + "\n"
                )


def _write_task(
    task_dir: Path,
    *,
    pre: dict[str, float],
    post: dict[str, float],
    post_mtime: float,
    started_at: str,
    elapsed_s: float,
) -> None:
    _write_prom(task_dir / "vllm_metrics_pre.txt", pre)
    _write_prom(task_dir / "vllm_metrics_post.txt", post)
    os.utime(task_dir / "vllm_metrics_post.txt", (post_mtime, post_mtime))
    (task_dir / "runner_metadata.json").write_text(
        json.dumps(
            {
                "instance_id": task_dir.name,
                "started_at": started_at,
                "agent": {"elapsed_s": elapsed_s},
            }
        ),
        encoding="utf-8",
    )
    (task_dir / "qwen_trace.jsonl").write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "usage": {"input_tokens": 500, "output_tokens": 50},
                    "content": [{"type": "tool_use"}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _build_arm(
    arm_dir: Path, *, nested: bool, tasks: int = 2, staggered: bool = False
) -> None:
    """Write one arm whose per-task brackets are nested, disjoint or staggered."""
    per_task = arm_dir / "swe_out" / "verified" / "per_task"
    for index in range(tasks):
        name = f"task-{index}"
        if nested:
            pre = _scale(0.0)
            post = _scale(index + 1)
            started = "2026-01-01T00:00:00Z"
            elapsed = 100.0 * (index + 1)
        elif staggered:
            # A refilled pool: task k is admitted half a unit before task k-1
            # finishes, so the brackets overlap without sharing an origin.
            pre = _scale(index * 0.5)
            post = _scale(index + 1)
            started = f"2026-01-01T00:0{index}:00Z"
            elapsed = 90.0
        else:
            pre = _scale(index)
            post = _scale(index + 1)
            started = f"2026-01-01T00:0{index}:00Z"
            elapsed = 60.0
        _write_task(
            per_task / name,
            pre=pre,
            post=post,
            post_mtime=1.0e9 + index,
            started_at=started,
            elapsed_s=elapsed,
        )

    total = tasks if nested else tasks
    events = int(UNIT["vllm:fr13_decode_forward_gpu_drafts_total"] * total)
    _write_census(arm_dir / "logs" / "fr13_fixed32_work_census.jsonl", [1] * events)
    requests = int(UNIT["vllm:request_prompt_tokens_count"] * total)
    _write_ledger(
        arm_dir / "logs" / "fr13_fixed32_proxy_ingress.jsonl",
        {f"key-{i}": requests // tasks for i in range(tasks)},
    )
    _write_prom(arm_dir / "metrics_before_swe.txt", {"vllm:spec_decode_num_drafts_total": 0.0})
    _write_prom(
        arm_dir / "metrics_after_swe.txt",
        {"vllm:spec_decode_num_drafts_total": UNIT["vllm:spec_decode_num_drafts_total"] * total},
    )
    (arm_dir / "deploy_speed_fullwall.json").write_text(
        json.dumps({"effective_concurrency": 0.5, "measured_tps_fullstep_wall": 1.0}),
        encoding="utf-8",
    )


def _build_runroot(
    root: Path, *, nested: bool, concurrency: int, staggered: bool = False
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "launcher_meta.txt").write_text(
        "\n".join(
            [
                "batch_size=%d" % concurrency,
                "concurrency=%d" % concurrency,
                "task_count=2",
                "stock_arm=arm_stock",
                "candidate_arm=arm_candidate",
                "started=2026-01-01T00:00:00Z",
                "arm=arm_stock serve_rc=0 ended=2026-01-01T01:00:00Z",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for arm in ("arm_stock", "arm_candidate"):
        _build_arm(root / arm, nested=nested, staggered=staggered)
    return root


def test_disjoint_brackets_sum_to_the_arm(tmp_path: Path) -> None:
    root = _build_runroot(tmp_path / "b1", nested=False, concurrency=1)
    report = reducer._reduce_runroot(root, "B1")
    arm = report["arms"][0]
    assert arm["bracket_topology"] == "disjoint"
    # Two disjoint unit brackets tile the arm.
    assert arm["counters"]["requests"] == pytest.approx(4.0)
    assert arm["counters"]["pure_decode_events"] == pytest.approx(20.0)
    assert arm["summed_bracket_inflation"]["vllm:prompt_tokens_total"] == pytest.approx(1.0)


def test_nested_brackets_take_the_widest_not_the_sum(tmp_path: Path) -> None:
    root = _build_runroot(tmp_path / "b4", nested=True, concurrency=2)
    report = reducer._reduce_runroot(root, "B4")
    arm = report["arms"][0]
    assert arm["bracket_topology"] == "nested"
    assert arm["bracket_closing_task"] == "task-1"
    # The widest bracket is 2 units; naively summing the nested brackets gives 3.
    assert arm["counters"]["requests"] == pytest.approx(4.0)
    assert arm["summed_bracket_inflation"]["vllm:prompt_tokens_total"] == pytest.approx(1.5)


def test_staggered_brackets_take_the_envelope_not_the_sum(tmp_path: Path) -> None:
    """A refilled pool overlaps without nesting; summing multiply-counts it."""
    root = _build_runroot(
        tmp_path / "pool", nested=False, concurrency=4, staggered=True
    )
    report = reducer._reduce_runroot(root, "B4")
    arm = report["arms"][0]
    assert arm["bracket_topology"] == "staggered"
    # Envelope is 2 units; naively summing the overlapping brackets gives 2.5.
    assert arm["counters"]["requests"] == pytest.approx(4.0)
    assert arm["counters"]["pure_decode_events"] == pytest.approx(20.0)
    assert arm["summed_bracket_inflation"]["vllm:prompt_tokens_total"] == pytest.approx(
        1.25
    )


def test_census_disagreement_is_fatal(tmp_path: Path) -> None:
    root = _build_runroot(tmp_path / "bad", nested=False, concurrency=1)
    _write_census((root / "arm_stock" / "logs" / "fr13_fixed32_work_census.jsonl"), [1] * 19)
    with pytest.raises(reducer.AlignmentError, match="census steps"):
        reducer._reduce_runroot(root, None)


def test_ledger_disagreement_is_fatal(tmp_path: Path) -> None:
    root = _build_runroot(tmp_path / "bad", nested=False, concurrency=1)
    _write_ledger(root / "arm_stock" / "logs" / "fr13_fixed32_proxy_ingress.jsonl", {"k": 3})
    with pytest.raises(reducer.AlignmentError, match="ledger logical_complete"):
        reducer._reduce_runroot(root, None)


def test_phase_components_sum_to_the_measured_wall(tmp_path: Path) -> None:
    root = _build_runroot(tmp_path / "b4", nested=True, concurrency=2)
    for arm in reducer._reduce_runroot(root, None)["arms"]:
        phase = arm["phase"]
        per_event = phase["per_event_ms"]
        assert per_event["sfwd"] + per_event["dfwd"] + per_event["cfwd"] + per_event["other"] == pytest.approx(
            per_event["wall"]
        )
        per_step = phase["per_step_ms"]
        assert per_step["sfwd"] + per_step["dfwd"] + per_step["cfwd"] + per_step["other"] == pytest.approx(
            per_step["wall"]
        )


def test_concurrency_one_has_no_mixed_step_events(tmp_path: Path) -> None:
    root = _build_runroot(tmp_path / "b1", nested=False, concurrency=1)
    arm = reducer._reduce_runroot(root, None)["arms"][0]
    # global drafts (16) + one per request (4) == matched pure-decode events (20)
    assert arm["phase"]["mixed_step_events"] == pytest.approx(0.0)
    assert arm["phase"]["mixed_step_event_share"] == pytest.approx(0.0)


def test_per_request_step_tps_is_commit_over_step_wall(tmp_path: Path) -> None:
    root = _build_runroot(tmp_path / "b1", nested=False, concurrency=1)
    arm = reducer._reduce_runroot(root, None)["arms"][0]
    committed = arm["phase"]["committed_tokens_per_event"]
    step_wall_s = arm["phase"]["per_step_ms"]["wall"] / 1000.0
    assert arm["throughput"]["per_request_step_tps"] == pytest.approx(committed / step_wall_s)
    # accepted 60 over 20 events, plus the bonus token
    assert committed == pytest.approx(4.0)


def test_coresidency_attribution_closes_on_the_gap(tmp_path: Path) -> None:
    root = _build_runroot(tmp_path / "b4", nested=True, concurrency=2)
    arm = reducer._reduce_runroot(root, None)["arms"][0]
    core = arm["coresidency"]
    assert sum(core["attribution"].values()) == pytest.approx(core["missing_events_per_step"])
    assert core["missing_events_per_step"] == pytest.approx(
        core["nominal_concurrency"] - arm["census"]["events_per_step"]
    )


def test_prefill_share_and_apc(tmp_path: Path) -> None:
    root = _build_runroot(tmp_path / "b1", nested=False, concurrency=1)
    share = reducer._reduce_runroot(root, None)["arms"][0]["prefill_share"]
    assert share["apc_hit_rate_tokens"] == pytest.approx(0.8)
    assert share["share_gross"] == pytest.approx(2000.0 / 2200.0)
    assert share["share_computed"] == pytest.approx(400.0 / 600.0)


def test_cli_writes_json_report(tmp_path: Path) -> None:
    root = _build_runroot(tmp_path / "b1", nested=False, concurrency=1)
    out = tmp_path / "out" / "alignment.json"
    assert reducer.main(["--runroot", str(root), "--label", "B1", "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == reducer.SCHEMA
    assert [r["label"] for r in payload["runroots"]] == ["B1"]


def test_cli_rejects_mismatched_labels(tmp_path: Path) -> None:
    root = _build_runroot(tmp_path / "b1", nested=False, concurrency=1)
    assert reducer.main(["--runroot", str(root), "--label", "a", "--label", "b"]) == 2


def test_cli_reports_failure_without_traceback(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert reducer.main(["--runroot", str(empty)]) == 2

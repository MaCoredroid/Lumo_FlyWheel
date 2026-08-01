from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "fr13_cfwd_all_parent_timing.py"
RUNNER = REPO / "scripts" / "fr13_run_b1_cfwd_all_parent_timing.sh"
SUBSET = REPO / "config" / "fr13_fixed32" / "subset_b1_diagnostic_one.json"
LIVE_PASS = (
    REPO
    / "results"
    / "fr13_fixed32_cfwd_all_parent_b1_live_pass_20260801"
    / "live_pass.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "fr13_cfwd_all_parent_timing_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, payload: object) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return path


def _measure(module, *, wall_ms: float, cfwd_ms: float, accept: float) -> dict:
    committed = accept + 1.0
    floor_ms = module.FULL_VOCAB_MANDATORY_WEIGHT_FLOOR_MS
    return {
        "schema": "fr13.measure.deploy_speed.v1",
        "regime": "deployment",
        "instrument": "OFF",
        "batch_size": 1,
        "n_tasks": 1,
        "task_instance_ids": [module.TASK_ID],
        "mandatory_weight_bytes": module.FULL_VOCAB_MANDATORY_WEIGHT_BYTES,
        "floor_is_full_step_hardware_floor": False,
        "engagement": {
            "engaged": True,
            "tok_per_draft": 31.0,
            "expected_tok_per_draft": 31.0,
        },
        "measured_tps_fullstep_wall": committed * 1_000.0 / wall_ms,
        "step_wall_ms": wall_ms,
        "accept_per_event": accept,
        "committed_per_event": committed,
        "wall_steps_measured": 100.0,
        "events_per_step": 1.0,
        "s_per_fwd_gpu": 0.15,
        "drafter_gpu_ms_per_step": 35.0,
        "committer_gpu_ms_per_step": cfwd_ms,
        "floor_ms": floor_ms,
        "weight_floor_ms": floor_ms,
        "floor_ratio": wall_ms / floor_ms,
    }


def _environment(module, *, production: bool, arm: str) -> str:
    payload = {
        **module._REQUIRED_ENV,
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION": str(int(production)),
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PASS_PATH": (
            "/logs/fr13_fixed32_taw_native_precompute.production_pass.json"
        ),
        "FR13_CFWD_GPU_TIMER_JSON": f"/workspace/output/{arm}_cfwd.json",
        "FR13_DFWD_GPU_TIMER_JSON": f"/workspace/output/{arm}_dfwd.json",
        "FR13_SFWD_GPU_TIMER_JSON": f"/workspace/output/{arm}.json",
        "FR13_RUN_DIR": f"/workspace/output/{arm}",
        "HOSTNAME": f"container-{arm}",
        "LUMO_NSYS_OUTPUT": f"/logs/nsys_{arm}",
    }
    return "\n".join(f"{key}={value}" for key, value in sorted(payload.items())) + "\n"


def _census(module, *, production: bool, events: int = 2) -> list[dict]:
    route = module.PRODUCTION_ROUTE if production else module.REFERENCE_ROUTE
    target_rows, self_rows = (17, 13) if production else (12, 12)
    launches = 1 if production else 12
    softmax_calls = 2 if production else 24
    row_gathers = 30 if production else 24
    records = []
    for index in range(events):
        records.append(
            {
                "schema": module.CENSUS_SCHEMA,
                "event_complete": True,
                "event_index": index,
                "mode": "hydra27_fixed32",
                "batch_size": 1,
                "active_nodes": 27,
                "physical_drafts": 31,
                "verify_rows": 32,
                "failures": {"fallback": 0, "overflow": 0},
                "batch_purity": {
                    "all_physical_31": True,
                    "physical_draft_counts": [31],
                },
                "taw": {
                    "route": route,
                    "source_contract_schema": module.SOURCE_CONTRACT_SCHEMA,
                    "source_contract_sha256": module.SOURCE_CONTRACT_SHA256,
                    "table_shape": [1, 32, 3],
                    "buffer_capacity": 32,
                    "loop_iterations": 12,
                    "uniform_slots": 36,
                    "vocab_size": 248_320,
                    "target_rows": target_rows,
                    "self_rows": self_rows,
                    "exact_commit_launches": launches,
                    "exact_commit_programs": launches,
                    "tensor_call_census": {
                        "full_vocab_softmax_calls": softmax_calls,
                        "full_vocab_row_gathers": row_gathers,
                        "exact_commit_launches": launches,
                        "floating_sampling_reimplementation": False,
                    },
                },
            }
        )
    records.append(
        {
            "schema": module.CENSUS_TERMINAL_SCHEMA,
            "mode": "hydra27_fixed32",
            "final": True,
            "event_count": events,
            "first_event_index": 0,
            "last_event_index": events - 1,
            "batch_histogram": {"1": events, "2": 0, "3": 0, "4": 0},
            "events_sha256": module._canonical_sha256(records),
        }
    )
    return records


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="ascii",
    )
    return path


def _eval(module) -> dict:
    return {
        "track": "swe_bench",
        "dataset_name": "princeton-nlp/SWE-bench_Verified",
        "instance_id": module.TASK_ID,
        "verdict": "resolved",
        "passed": True,
        "harness_exit_code": 0,
    }


def _fixture(tmp_path: Path):
    module = _load_module()
    stock_measure = _write(
        tmp_path / "stock-measure.json",
        _measure(module, wall_ms=250.0, cfwd_ms=20.0, accept=4.0),
    )
    candidate_measure = _write(
        tmp_path / "candidate-measure.json",
        _measure(module, wall_ms=244.0, cfwd_ms=14.0, accept=4.1),
    )
    stock_env = tmp_path / "stock.env"
    stock_env.write_text(_environment(module, production=False, arm="stock"))
    candidate_env = tmp_path / "candidate.env"
    candidate_env.write_text(_environment(module, production=True, arm="candidate"))
    stock_census = _write_jsonl(
        tmp_path / "stock-census.jsonl", _census(module, production=False)
    )
    candidate_census = _write_jsonl(
        tmp_path / "candidate-census.jsonl", _census(module, production=True)
    )
    stock_eval = _write(tmp_path / "stock-eval.json", _eval(module))
    candidate_eval = _write(tmp_path / "candidate-eval.json", _eval(module))
    candidate_selector = tmp_path / "candidate-production.arm"
    candidate_selector.write_text("1\n", encoding="ascii")
    candidate_pass = tmp_path / "candidate-pass.json"
    candidate_pass.write_bytes(LIVE_PASS.read_bytes())
    return module, {
        "subset": SUBSET,
        "curated_pass": LIVE_PASS,
        "stock_measure": stock_measure,
        "candidate_measure": candidate_measure,
        "stock_container_env": stock_env,
        "candidate_container_env": candidate_env,
        "stock_census": stock_census,
        "candidate_census": candidate_census,
        "stock_eval_report": stock_eval,
        "candidate_eval_report": candidate_eval,
        "stock_selector": tmp_path / "stock-production.arm",
        "stock_production_pass": tmp_path / "stock-pass.json",
        "candidate_selector": candidate_selector,
        "candidate_production_pass": candidate_pass,
        "source_commit": "a" * 40,
    }


def test_reduce_pair_reports_required_b1_diagnostic_metrics(tmp_path: Path) -> None:
    module, arguments = _fixture(tmp_path)
    result = module.reduce_pair(**arguments)

    assert result["task_ids"] == [module.TASK_ID]
    assert result["batch_size"] == 1
    assert result["concurrency"] == 1
    assert result["only_arm_delta"].endswith("PRODUCTION=0 to 1")
    assert result["common_physical_work"]["physical_rows_per_event"] == 32
    assert result["common_physical_work"]["identical_in_both_arms"] is True
    assert result["stock_reference"]["cfwd_gpu_ms_per_event"] == 20.0
    assert result["candidate"]["cfwd_gpu_ms_per_event"] == 14.0
    assert result["candidate"]["full_wall_ms_per_event"] == 244.0
    assert result["candidate"]["accepted_drafts_per_event"] == 4.1
    assert result["candidate"]["task_verdict"] == "resolved"
    assert result["candidate"]["fixed_work_census"]["event_count"] == 2
    assert (
        result["candidate"]["production_engagement"]["copied_credential_sha256"]
        == module.LIVE_PASS_SHA256
    )
    assert result["candidate_minus_stock"]["cfwd_gpu_ms_per_event"] == -6.0
    assert result["timing_eligible"] is False
    assert result["floor_acceptance_eligible"] is False
    assert result["formal_floor_acceptance_eligible"] is False
    assert result["production_default_enabled"] is False


def test_reduce_pair_rejects_route_env_and_credential_drift(tmp_path: Path) -> None:
    module, arguments = _fixture(tmp_path)
    candidate_census = arguments["candidate_census"]
    records = _census(module, production=True)
    records[0]["taw"]["route"] = module.REFERENCE_ROUTE
    records[-1]["events_sha256"] = module._canonical_sha256(records[:-1])
    _write_jsonl(candidate_census, records)
    with pytest.raises(module.TimingError, match="violates the fixed32/CFWD"):
        module.reduce_pair(**arguments)

    _write_jsonl(candidate_census, _census(module, production=True))
    candidate_env = arguments["candidate_container_env"]
    candidate_env.write_text(
        candidate_env.read_text(encoding="utf-8") + "UNEXPECTED_DELTA=1\n",
        encoding="utf-8",
    )
    with pytest.raises(module.TimingError, match="outside the CFWD selector"):
        module.reduce_pair(**arguments)

    candidate_env.write_text(
        _environment(module, production=True, arm="candidate"), encoding="utf-8"
    )
    arguments["candidate_production_pass"].write_text("{}\n", encoding="ascii")
    with pytest.raises(module.TimingError, match="copied credential differs"):
        module.reduce_pair(**arguments)


def test_curated_pass_and_runner_wiring_are_pinned() -> None:
    module = _load_module()
    binding = module.validate_live_pass(LIVE_PASS)
    assert binding["candidate"] == module.CANDIDATE
    assert binding["sha256"] == hashlib.sha256(LIVE_PASS.read_bytes()).hexdigest()

    source = RUNNER.read_text(encoding="ascii")
    assert "FR13_RUN_CFWD_ALL_PARENT_B1_TIMING" in source
    assert "subset_b1_diagnostic_one.json" in source
    assert "FR13_FIXED32_B1_DIAGNOSTIC=1" in source
    assert "MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1" in source
    assert "FR13_DRAFT_VOCAB_ROOT=0 FR13_DRAFT_VOCAB_K=0" in source
    assert "FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1" in source
    assert 'run_arm "$STOCK_ARM" 0' in source
    assert 'run_arm "$CANDIDATE_ARM" 1' in source
    assert source.index('run_arm "$STOCK_ARM" 0') < source.index(
        'run_arm "$CANDIDATE_ARM" 1'
    )
    assert "timing_eligible=0" in source
    assert "floor_acceptance_eligible=0" in source
    assert "formal_floor_acceptance_eligible=0" in source
    assert "production_default_enabled=0" in source

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from lumo_flywheel_serving.tuned_config import compute_workload_distribution_id
from lumo_flywheel_serving.workload_p1 import (
    SIBLING_HOLDOUT_FAMILIES,
    annotate_capture_rows,
    validate_p1_workload,
    write_heavy_workload_descriptor,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_probe_serving_thinking_writes_report_with_fake_transport(tmp_path: Path) -> None:
    module = _load_script("probe_serving_thinking.py")
    seen: list[dict] = []

    def fake_post(url: str, headers: dict[str, str], json: dict, timeout: int) -> _FakeResponse:
        seen.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        reasoning_tokens = 0 if len(seen) == 1 else 256
        return _FakeResponse(
            {
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 22,
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": 33 + reasoning_tokens,
                }
            }
        )

    result = module.run_probe(
        base_url="http://127.0.0.1:8000/v1",
        api_key="test-key",
        model="qwen3.5-27b",
        reports_dir=tmp_path / "reports",
        capture_date=datetime(2026, 4, 24, 12, 0, tzinfo=UTC),
        transport=fake_post,
    )

    report_path = Path(result["report_path"])
    report_text = report_path.read_text(encoding="utf-8")
    assert result["outcome"] == "row-1"
    assert report_path.name == "thinking-probe-20260424.md"
    assert "- capture_date: 2026-04-24T12:00:00Z" in report_text
    assert "- outcome: row-1" in report_text
    assert seen[0]["url"] == "http://127.0.0.1:8000/v1/responses"
    assert "extra_body" not in seen[0]["json"]
    assert seen[1]["json"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert seen[1]["headers"]["Authorization"] == "Bearer test-key"


def test_probe_serving_thinking_classifies_case_matrix() -> None:
    module = _load_script("probe_serving_thinking.py")

    assert module.classify_outcome(0, 10)[0] == "row-1"
    assert module.classify_outcome(0, 0)[0] == "row-2"
    assert module.classify_outcome(10, 20)[0] == "row-3"
    assert module.classify_outcome(10, 0)[0] == "bug"


def test_probe_serving_thinking_counts_vllm_reasoning_output_fallback() -> None:
    module = _load_script("probe_serving_thinking.py")

    usage = module._normalize_usage(
        {
            "usage": {
                "input_tokens": 24,
                "output_tokens": 128,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 152,
            },
            "output": [
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "scratchpad"}],
                }
            ],
        }
    )

    assert usage["reasoning_tokens"] == 128


def test_capture_seed_workload_counts_vllm_reasoning_output_fallback() -> None:
    module = _load_script("capture_seed_workload.py")

    reasoning_tokens = module._reasoning_tokens_from_response(
        {
            "usage": {
                "output_tokens": 64,
                "output_tokens_details": {"reasoning_tokens": 0},
            },
            "output": [
                {
                    "content": [{"type": "reasoning_text", "text": "scratchpad"}],
                }
            ],
        },
        64,
    )

    assert reasoning_tokens == 64


def test_capture_seed_workload_family_v5_defaults_to_per_family_trace(tmp_path: Path) -> None:
    script = REPO_ROOT / "scripts" / "capture_seed_workload.py"
    family_dir = tmp_path / "benchmark_blueprints" / "families" / "family-a"
    family_dir.mkdir(parents=True)

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(tmp_path),
            "--family-id",
            "family-a",
            "--variant",
            "v5",
            "--count",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    trace_path = family_dir / "seed_trace_v5.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert payload["seed_output"] == str(trace_path)
    assert payload["seed_count"] == 4
    assert payload["holdout_count"] == 0
    assert payload["family_id"] == "family-a"
    assert payload["variant"] == "v5"
    assert {row["family_id"] for row in rows} == {"family-a"}


def test_capture_seed_workload_live_v5_uses_family_prompts_and_response_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("capture_seed_workload.py")
    family_dir = tmp_path / "benchmark_blueprints" / "families" / "family-a"
    (family_dir / "workspace_bundle" / "v5-recovery-in-thread").mkdir(parents=True)
    (family_dir / "family.yaml").write_text("id: family-a\nrawr_status: flywheel_ready\n", encoding="utf-8")
    (family_dir / "task_spec.md").write_text("# Task\nFix the v5 recovery path.\n", encoding="utf-8")
    (family_dir / "verification_matrix_v5.md").write_text("# Matrix\n- hidden check\n", encoding="utf-8")
    seen: list[dict] = []

    def fake_post(url: str, headers: dict[str, str], json: dict, timeout: int) -> _FakeResponse:
        seen.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return _FakeResponse(
            {
                "usage": {
                    "input_tokens": 123,
                    "output_tokens": 4096,
                    "output_tokens_details": {"reasoning_tokens": 4096},
                }
            }
        )

    monkeypatch.setattr(module.requests, "post", fake_post)

    rows = module._capture_live(
        "http://127.0.0.1:8100/v1",
        "qwen3.5-27b",
        4,
        api_key="test-key",
        family_id="family-a",
        variant="v5",
        repo_root=tmp_path,
        enable_thinking_override=True,
    )

    assert [row["capture_prompt_label"] for row in rows] == [
        "v5_deep_failure_analysis",
        "v5_acceptance_summary",
        "v5_implementation_plan",
        "v5_holdout_deep_review",
    ]
    assert seen[0]["json"]["max_output_tokens"] == 4096
    assert seen[1]["json"]["max_output_tokens"] == 512
    assert seen[0]["json"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert "Fix the v5 recovery path" in seen[1]["json"]["input"]
    assert rows[0]["output_tokens"] == 4096
    assert rows[0]["thinking_tokens"] == 4096
    assert rows[0]["response_tokens"] == 0


def _write_trace(path: Path, family_id: str, thinking_values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, thinking_tokens in enumerate(thinking_values):
        rows.append(
            {
                "family_id": family_id,
                "turn_index": index,
                "prompt_tokens": 1000 + index,
                "output_tokens": 100,
                "thinking_tokens": thinking_tokens,
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_l0_p1_heavy_descriptor_hash_and_sibling_holdout_validation(tmp_path: Path) -> None:
    source_family = "responses-sdk-adapter-cutover"
    _write_trace(
        tmp_path / "benchmark_blueprints" / "families" / source_family / "seed_trace_v5.jsonl",
        source_family,
        [5000, 500, 500, 4096],
    )
    workload_dir = tmp_path / "benchmark_blueprints" / "workloads" / "responses-sdk-adapter-cutover-heavy"
    _write_trace(workload_dir / "seed_trace.jsonl", source_family, [4096, 512, 512, 4096])
    _write_trace(workload_dir / "holdout_trace.jsonl", source_family, [4096, 512, 512, 4096])
    descriptor_path = write_heavy_workload_descriptor(
        repo_root=tmp_path,
        capture_date="2026-04-25T00:00:00Z",
        thinking_probe_ref="reports/thinking-probe-20260424.md",
    )
    for sibling in SIBLING_HOLDOUT_FAMILIES:
        rows = [
            {
                "family_id": sibling,
                "turn_index": 0,
                "prompt_tokens": 1000,
                "output_tokens": 4096,
                "thinking_tokens": 4096,
            }
        ]
        rows = annotate_capture_rows(
            rows,
            capture_role="sibling_holdout_v5",
            baseline="vllm-default",
            weight_version_id="weight-1",
            capture_date="2026-04-25T00:00:00Z",
        )
        (tmp_path / "benchmark_blueprints" / "families" / sibling / "holdout_trace_v5.jsonl").parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        (tmp_path / "benchmark_blueprints" / "families" / sibling / "holdout_trace_v5.jsonl").write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )

    descriptor = yaml.safe_load(descriptor_path.read_text(encoding="utf-8"))
    assert descriptor["workload_distribution_id"] == compute_workload_distribution_id(descriptor_path)
    assert descriptor["workload_distribution_id_hardening_version"] == "v2-l0-kernel-heavy"
    assert descriptor["parity_fixture_refs"] == {
        "deltanet": "parity_fixture/deltanet_v1.yaml",
        "gatedattn": "parity_fixture/gatedattn_v1.yaml",
    }

    validation = validate_p1_workload(
        repo_root=tmp_path,
        descriptor_path=descriptor_path,
        expected_weight_version_id="weight-1",
    )

    assert validation["pass"] is True
    assert validation["halt_reason"] is None
    assert validation["canonical_workload_distribution_id"] == descriptor["workload_distribution_id"]
    assert validation["missing_sibling_holdouts"] == []
    assert validation["heavy_seed_trace"]["reasoning_positive_rows"] == 4
    assert validation["heavy_holdout_trace"]["reasoning_positive_rows"] == 4


def test_l0_p1_validation_reports_missing_sibling_holdouts_with_halt_reason(tmp_path: Path) -> None:
    source_family = "responses-sdk-adapter-cutover"
    _write_trace(
        tmp_path / "benchmark_blueprints" / "families" / source_family / "seed_trace_v5.jsonl",
        source_family,
        [5000, 500, 500, 4096],
    )
    workload_dir = tmp_path / "benchmark_blueprints" / "workloads" / "responses-sdk-adapter-cutover-heavy"
    _write_trace(workload_dir / "seed_trace.jsonl", source_family, [4096, 512, 512, 4096])
    _write_trace(workload_dir / "holdout_trace.jsonl", source_family, [4096, 512, 512, 4096])
    descriptor_path = write_heavy_workload_descriptor(
        repo_root=tmp_path,
        capture_date="2026-04-25T00:00:00Z",
        thinking_probe_ref="reports/thinking-probe-20260424.md",
    )

    validation = validate_p1_workload(repo_root=tmp_path, descriptor_path=descriptor_path)

    assert validation["pass"] is False
    assert validation["halt_reason"] == "sibling_holdout_capture_failed"
    assert validation["missing_sibling_holdouts"] == list(SIBLING_HOLDOUT_FAMILIES)
    assert "sibling_holdout_missing:codex-provider-rollover" in validation["errors"]


def test_multi_family_builder_splits_hashes_and_validates(tmp_path: Path) -> None:
    module = _load_script("capture_multi_family_v5_workload.py")
    pool_file = tmp_path / "benchmark_blueprints" / "workloads" / "multi-family-v5" / "pool.yaml"
    pool_file.parent.mkdir(parents=True)
    pool_file.write_text(
        yaml.safe_dump(
            {
                "pool_families": ["family-a", "family-b", "family-missing", "family-short"],
                "pool_excluded_families": [{"family_id": "family-preexcluded", "reason": "wire_api_missing"}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_trace(
        tmp_path / "benchmark_blueprints" / "families" / "family-a" / "seed_trace_v5.jsonl",
        "family-a",
        [5000, 200, 300, 7000],
    )
    _write_trace(
        tmp_path / "benchmark_blueprints" / "families" / "family-b" / "seed_trace_v5.jsonl",
        "family-b",
        [100, 6000, 200, 300],
    )
    _write_trace(
        tmp_path / "benchmark_blueprints" / "families" / "family-short" / "seed_trace_v5.jsonl",
        "family-short",
        [5000, 5000, 5000],
    )
    probe_path = tmp_path / "reports" / "thinking-probe-20260424.md"
    probe_path.parent.mkdir()
    probe_path.write_text("- capture_date: 2026-04-24T12:00:00Z\n- outcome: row-3\n", encoding="utf-8")

    result = module.build_composite_workload(
        repo_root=tmp_path,
        pool_file=pool_file,
        samples_per_family=4,
        split_per_family="3:1",
        split_seed=123,
        min_trajectory_turns=4,
        thinking_probe=probe_path,
    )

    descriptor_path = Path(result["workload_file"])
    descriptor = yaml.safe_load(descriptor_path.read_text(encoding="utf-8"))
    seed_rows = [
        json.loads(line)
        for line in (descriptor_path.parent / "seed_trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    holdout_rows = [
        json.loads(line)
        for line in (descriptor_path.parent / "holdout_trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert result["seed_rows"] == 6
    assert result["holdout_rows"] == 2
    assert descriptor["pool_families"] == ["family-a", "family-b"]
    assert {item["family_id"] for item in descriptor["pool_excluded_families"]} == {
        "family-preexcluded",
        "family-missing",
        "family-short",
    }
    assert descriptor["workload_distribution_id"] == module.compute_workload_distribution_id(descriptor_path)
    assert {row["family_id"] for row in seed_rows} == {"family-a", "family-b"}
    assert {row["family_id"] for row in holdout_rows} == {"family-a", "family-b"}

    validation = module.validate_composite_workload(
        descriptor_path,
        min_seed_rows=6,
        min_holdout_rows=2,
        min_thinking_ratio=0.30,
        min_thinking_gt_response_ratio=0.10,
    )
    assert validation["pass"] is True
    strict_validation = module.validate_composite_workload(descriptor_path)
    assert "seed_row_count_below_minimum:6<84" in strict_validation["errors"]
    assert "holdout_row_count_below_minimum:2<28" in strict_validation["errors"]


def test_multi_family_validation_reports_ar26_failures(tmp_path: Path) -> None:
    module = _load_script("capture_multi_family_v5_workload.py")
    workload_dir = tmp_path / "benchmark_blueprints" / "workloads" / "multi-family-v5"
    workload_dir.mkdir(parents=True)
    seed_rows = [
        {"family_id": "family-a", "turn_index": 0, "output_tokens": 100, "thinking_tokens": 0},
        {"family_id": "family-a", "turn_index": 1, "output_tokens": 100, "thinking_tokens": 0},
        {"family_id": "family-a", "turn_index": 2, "output_tokens": 100, "thinking_tokens": 0},
    ]
    holdout_rows = [
        {"family_id": "family-b", "turn_index": 0, "output_tokens": 100, "thinking_tokens": 0},
    ]
    (workload_dir / "seed_trace.jsonl").write_text(
        "\n".join(json.dumps(row) for row in seed_rows) + "\n",
        encoding="utf-8",
    )
    (workload_dir / "holdout_trace.jsonl").write_text(
        "\n".join(json.dumps(row) for row in holdout_rows) + "\n",
        encoding="utf-8",
    )
    descriptor_path = workload_dir / "workload.yaml"
    descriptor_path.write_text(
        yaml.safe_dump(
            {
                "family_id": "multi-family-v5",
                "workload_distribution_id": "placeholder",
                "pool_families": ["family-a", "family-b"],
                "pool_excluded_families": [{"family_id": "family-b", "reason": "bad-fixture"}],
                "split_per_family": {"seed_rows": 3, "holdout_rows": 1},
                "seed_trace_ref": "seed_trace.jsonl",
                "holdout_trace_ref": "holdout_trace.jsonl",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    validation = module.validate_composite_workload(
        descriptor_path,
        min_seed_rows=3,
        min_holdout_rows=1,
    )

    assert validation["pass"] is False
    assert "seed_family_coverage_mismatch" in validation["errors"]
    assert "holdout_family_coverage_mismatch" in validation["errors"]
    assert "seed_family_count_below_minimum:family-b" in validation["errors"]
    assert "pool_excluded_overlap" in validation["errors"]
    assert "seed_thinking_positive_ratio_below_minimum" in validation["errors"]
    assert "seed_thinking_gt_response_ratio_below_minimum" in validation["errors"]
    assert "seed_large_thinking_row_missing" in validation["errors"]
    assert "holdout_thinking_positive_ratio_below_minimum" in validation["errors"]
    assert "holdout_thinking_gt_response_ratio_below_minimum" in validation["errors"]
    assert "holdout_large_thinking_row_missing" in validation["errors"]


def test_multi_family_validation_enforces_thinking_distribution_per_file(tmp_path: Path) -> None:
    module = _load_script("capture_multi_family_v5_workload.py")
    workload_dir = tmp_path / "benchmark_blueprints" / "workloads" / "multi-family-v5"
    workload_dir.mkdir(parents=True)
    seed_rows = [
        {"family_id": "family-a", "turn_index": 0, "output_tokens": 100, "thinking_tokens": 5000},
        {"family_id": "family-a", "turn_index": 1, "output_tokens": 100, "thinking_tokens": 6000},
        {"family_id": "family-a", "turn_index": 2, "output_tokens": 100, "thinking_tokens": 7000},
    ]
    holdout_rows = [
        {"family_id": "family-a", "turn_index": 3, "output_tokens": 100, "thinking_tokens": 0},
    ]
    (workload_dir / "seed_trace.jsonl").write_text(
        "\n".join(json.dumps(row) for row in seed_rows) + "\n",
        encoding="utf-8",
    )
    (workload_dir / "holdout_trace.jsonl").write_text(
        "\n".join(json.dumps(row) for row in holdout_rows) + "\n",
        encoding="utf-8",
    )
    descriptor_path = workload_dir / "workload.yaml"
    descriptor_path.write_text(
        yaml.safe_dump(
            {
                "family_id": "multi-family-v5",
                "workload_distribution_id": "placeholder",
                "pool_families": ["family-a"],
                "pool_excluded_families": [],
                "split_per_family": {"seed_rows": 3, "holdout_rows": 1},
                "seed_trace_ref": "seed_trace.jsonl",
                "holdout_trace_ref": "holdout_trace.jsonl",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    validation = module.validate_composite_workload(
        descriptor_path,
        min_seed_rows=3,
        min_holdout_rows=1,
    )

    assert validation["pass"] is False
    assert "seed_thinking_positive_ratio_below_minimum" not in validation["errors"]
    assert "seed_thinking_gt_response_ratio_below_minimum" not in validation["errors"]
    assert "seed_large_thinking_row_missing" not in validation["errors"]
    assert "holdout_thinking_positive_ratio_below_minimum" in validation["errors"]
    assert "holdout_thinking_gt_response_ratio_below_minimum" in validation["errors"]
    assert "holdout_large_thinking_row_missing" in validation["errors"]


def test_multi_family_probe_parser_rejects_bug_outcome(tmp_path: Path) -> None:
    module = _load_script("capture_multi_family_v5_workload.py")
    probe_path = tmp_path / "reports" / "thinking-probe-20260424.md"
    probe_path.parent.mkdir()
    probe_path.write_text("- capture_date: 2026-04-24T12:00:00Z\n- outcome: bug\n", encoding="utf-8")

    try:
        module.parse_thinking_probe_report(probe_path)
    except ValueError as exc:
        assert "blocks composite capture" in str(exc)
    else:
        raise AssertionError("bug outcome should block composite capture")


def test_discover_v5_pool_records_exclusions(tmp_path: Path) -> None:
    module = _load_script("capture_multi_family_v5_workload.py")
    families_root = tmp_path / "benchmark_blueprints" / "families"
    eligible = families_root / "eligible-family"
    excluded = families_root / "missing-wire"
    for family_dir in (eligible, excluded):
        (family_dir / "codex").mkdir(parents=True)
        (family_dir / "verification_matrix_v5.md").write_text("# v5\n", encoding="utf-8")
        (family_dir / "manifest.lock.json").write_text("{}\n", encoding="utf-8")
    (eligible / "codex" / "config.toml").write_text(
        'wire_api = "responses"\nreasoning_effort = "high"\n',
        encoding="utf-8",
    )
    (excluded / "codex" / "config.toml").write_text(
        'reasoning_effort = "high"\n',
        encoding="utf-8",
    )

    pool = module.discover_v5_pool(tmp_path)

    assert pool["pool_families"] == ["eligible-family"]
    assert pool["pool_excluded_families"] == [{"family_id": "missing-wire", "reason": "wire_api_missing"}]

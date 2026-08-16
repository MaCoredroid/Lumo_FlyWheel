from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path("scripts")))
GATE_PATH = Path("scripts/fr13_floor_gate.py")
GATE_SPEC = importlib.util.spec_from_file_location(
    "fr13_fixed32_pretask_metrics_gate",
    GATE_PATH,
)
assert GATE_SPEC is not None and GATE_SPEC.loader is not None
gate = importlib.util.module_from_spec(GATE_SPEC)
GATE_SPEC.loader.exec_module(gate)


STOCK_ZERO_METRICS = """\
vllm:spec_decode_num_drafts_total{engine="0",model_name="qwen3.8-27b-nvfp4-radixark"} 0
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="qwen3.8-27b-nvfp4-radixark"} 0
"""


def test_pretask_allows_lazy_worker_metrics_to_be_absent() -> None:
    values, labels = gate.pretask_metric_snapshot_text(
        STOCK_ZERO_METRICS,
        label="pretask",
    )

    assert values == {"spec_drafts": 0.0, "spec_tokens": 0.0}
    assert labels == {
        "spec_drafts": 'engine="0",model_name="qwen3.8-27b-nvfp4-radixark"',
        "spec_tokens": 'engine="0",model_name="qwen3.8-27b-nvfp4-radixark"',
    }


def test_pretask_accepts_present_zero_worker_metric() -> None:
    values, _labels = gate.pretask_metric_snapshot_text(
        STOCK_ZERO_METRICS
        + "vllm:fr13_decode_forward_gpu_seconds_total 0\n",
        label="pretask",
    )

    assert values["fwd_s"] == 0.0


@pytest.mark.parametrize(
    ("metrics", "error"),
    (
        (
            "vllm:spec_decode_num_drafts_total"
            '{engine="0",model_name="qwen3.8-27b-nvfp4-radixark"} 0\n',
            "missing required pretask metrics",
        ),
        (
            STOCK_ZERO_METRICS
            + "vllm:spec_decode_num_drafts_total"
            '{engine="0",model_name="qwen3.8-27b-nvfp4-radixark"} 0\n',
            "duplicate metric series",
        ),
        (
            STOCK_ZERO_METRICS
            + "vllm:fr13_decode_forward_gpu_seconds_total 0.001\n",
            "pretask decode metrics are not exact zero",
        ),
        (
            STOCK_ZERO_METRICS.replace(
                'engine="0",model_name="qwen3.8-27b-nvfp4-radixark"',
                'engine="1",model_name="qwen3.8-27b-nvfp4-radixark"',
                1,
            ),
            "pretask metric labels differ from the contract",
        ),
    ),
)
def test_pretask_rejects_invalid_required_or_present_metrics(
    metrics: str,
    error: str,
) -> None:
    with pytest.raises(gate.GateError, match=error):
        gate.pretask_metric_snapshot_text(metrics, label="pretask")


def test_regular_task_snapshot_still_requires_worker_metrics() -> None:
    with pytest.raises(gate.GateError, match="missing required metrics"):
        gate.metric_snapshot_text(STOCK_ZERO_METRICS, label="task")


def test_serve_pretask_gate_accepts_raw_stock_only_scrape(
    tmp_path: Path,
    monkeypatch,
) -> None:
    serve_text = Path(
        "scripts/fr13_bigdenom_swe_serve_variant.sh"
    ).read_text(encoding="utf-8")
    anchor = '"$ARMDIR/fixed32_pretask_zero_traffic.json" <<\'PY\'\n'
    start = serve_text.index(anchor) + len(anchor)
    block = serve_text[start : serve_text.index("\nPY\n", start)]

    metrics = tmp_path / "metrics_before_swe.txt"
    census = tmp_path / "fr13_fixed32_work_census.jsonl"
    ready = tmp_path / "fixed32_ready_ack.json"
    output = tmp_path / "fixed32_pretask_zero_traffic.json"
    metrics.write_text(STOCK_ZERO_METRICS, encoding="utf-8")
    raw_metrics = metrics.read_bytes()
    ready.write_text(
        json.dumps(
            {
                "mode": "tail6_fixed32",
                "generation": 0,
                "action": "ready",
                "status": "ok",
                "counters": {
                    "pure_decode_forward_steps": 0,
                    "complete_work_census_events": 0,
                    "work_census_first_forward_step": None,
                    "work_census_last_forward_step": None,
                    "sfwd_pending": 0,
                    "dfwd_pending": 0,
                    "cfwd_pending": 0,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pretask-inline",
            str(metrics),
            str(census),
            str(ready),
            "tail6_fixed32",
            str(output),
        ],
    )

    exec(compile(block, "<fixed32-pretask-inline>", "exec"), {})

    assert metrics.read_bytes() == raw_metrics
    marker = json.loads(output.read_text(encoding="utf-8"))
    assert marker["no_positive_probe"] is True
    assert marker["metrics"]["spec_drafts"] == 0
    assert marker["metrics"]["spec_tokens"] == 0
    assert marker["work_census"]["bytes"] == 0

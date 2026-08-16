from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import fr13_floor_gate
from scripts import fr13_runtime_manifest
from scripts.fr13_hardware_floor_ledger import (
    BANDWIDTH_BYTES_PER_S,
    FIXED32_MANDATORY_WEIGHT_BYTES,
    FIXED32_MANDATORY_WEIGHT_FLOOR_MS,
    FIXED32_SLO_CAP_MS,
    FIXED32_SLO_MULTIPLIER,
    FULL_VOCAB_MANDATORY_WEIGHT_BYTES,
    FULL_VOCAB_MANDATORY_WEIGHT_FLOOR_MS,
    FULL_VOCAB_SLO_CAP_MS,
    build_ledger,
)


REPO = Path(__file__).resolve().parents[1]
SEQUENCE = REPO / "scripts" / "fr13_fixed32_floor_timers_seq.sh"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
MEASURE = REPO / "scripts" / "fr13_measure.py"
GATE = REPO / "scripts" / "fr13_floor_gate.py"


def test_canonical_fixed32_floor_math() -> None:
    assert BANDWIDTH_BYTES_PER_S == 273_000_000_000
    assert FIXED32_MANDATORY_WEIGHT_BYTES == 25_210_209_416
    assert FIXED32_MANDATORY_WEIGHT_FLOOR_MS == 92.345089436
    assert float(FIXED32_SLO_MULTIPLIER) == 1.15
    assert FIXED32_SLO_CAP_MS == 106.196852851

    scenario = build_ledger()["scenarios"]["root_64k_five_64k_draft_heads"]
    assert scenario["mandatory_weight_bytes"] == FIXED32_MANDATORY_WEIGHT_BYTES
    assert (
        scenario["mandatory_weight_floor_ms"]
        == FIXED32_MANDATORY_WEIGHT_FLOOR_MS
    )
    assert scenario["nonweight_costs_included"] is False
    assert FULL_VOCAB_MANDATORY_WEIGHT_BYTES == 25_430_574_256
    assert FULL_VOCAB_MANDATORY_WEIGHT_FLOOR_MS == 93.15228665201465
    assert FULL_VOCAB_SLO_CAP_MS == 107.12512964981684


def test_fixed32_gate_uses_corrected_weight_bound_and_exact_cap() -> None:
    assert fr13_floor_gate.WEIGHT_STREAM_LOWER_BOUND_MS == (
        FIXED32_MANDATORY_WEIGHT_FLOOR_MS
    )
    for rows in (32.0, 128.0):
        assert fr13_floor_gate.legacy_slo(rows) == (
            FIXED32_MANDATORY_WEIGHT_FLOOR_MS,
            FIXED32_SLO_CAP_MS,
        )

    compute_reference, compute_cap = fr13_floor_gate.legacy_slo(300.0)
    assert compute_reference == pytest.approx(162.0)
    assert compute_cap == pytest.approx(186.3)


@pytest.mark.parametrize(
    ("draft_vocab_k", "draft_vocab_root", "expected_bytes", "expected_floor"),
    (
        ("0", "0", "25430574256", "93.15228665201465"),
        ("65536", "0", "25254282384", "92.506528879"),
        ("65536", "1", "25210209416", "92.345089436"),
    ),
)
def test_fixed32_sequence_exports_exact_configured_floor(
    draft_vocab_k: str,
    draft_vocab_root: str,
    expected_bytes: str,
    expected_floor: str,
) -> None:
    command = f"""
set -euo pipefail
run_variant() {{ :; }}
export BSIZE=1
export CONC=1
export TAG=floor_test
export FR13_DRAFT_VOCAB_K={draft_vocab_k}
export FR13_DRAFT_VOCAB_ROOT={draft_vocab_root}
source {SEQUENCE}
printf '%s %s' "$FR13_MANDATORY_WEIGHT_BYTES" "$FR13_WEIGHT_FLOOR_MS"
"""
    result = subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert result.stdout == f"{expected_bytes} {expected_floor}"


def test_active_fixed32_paths_cannot_fall_back_to_legacy_floor() -> None:
    texts = {
        path: path.read_text(encoding="utf-8")
        for path in (SEQUENCE, LAUNCHER, MEASURE, GATE)
    }
    for path, text in texts.items():
        assert "98.6" not in text, path
        assert "113.39" not in text, path

    assert (
        "FR13_WEIGHT_FLOOR_MS|${FR13_WEIGHT_FLOOR_MS:-}|"
        "$_fixed32_expected_weight_floor_ms"
    ) in texts[LAUNCHER]
    assert (
        "FR13_MANDATORY_WEIGHT_BYTES|${FR13_MANDATORY_WEIGHT_BYTES:-}|"
        "$_fixed32_expected_mandatory_weight_bytes"
    ) in texts[LAUNCHER]
    assert "_fixed32_expected_weight_floor_ms=93.15228665201465" in texts[LAUNCHER]
    assert "FIXED32_MANDATORY_WEIGHT_FLOOR_MS" in texts[MEASURE]
    assert "FIXED32_MANDATORY_WEIGHT_FLOOR_MS" in texts[GATE]


def test_floor_ledger_is_bound_into_fixed32_runtime_manifest() -> None:
    assert (
        "scripts/fr13_hardware_floor_ledger.py"
        in fr13_runtime_manifest.FIXED32_HOST_SCRIPT_SOURCE
    )


def test_launcher_whitelists_full_vocab_ledger_before_docker() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    whitelist = text.index(
        'case "${FR13_DRAFT_VOCAB_K:-65536}:$FR13_DRAFT_VOCAB_ROOT" in'
    )
    mandatory = text.index(
        '"FR13_MANDATORY_WEIGHT_BYTES|${FR13_MANDATORY_WEIGHT_BYTES:-}|'
        '$_fixed32_expected_mandatory_weight_bytes"'
    )
    docker = text.index("docker run -d --pull=never")

    assert whitelist < mandatory < docker
    assert "0:0)" in text[whitelist:mandatory]
    assert "_fixed32_expected_mandatory_weight_bytes=25430574256" in text
    assert "FR13_NEEDS_ALLOW:-}" in text[whitelist:mandatory]
    assert '( "$MAX_NUM_SEQS" == "1" || "$MAX_NUM_SEQS" == "4" )' in text[
        whitelist:mandatory
    ]


def test_b4_workflows_use_full_vocab_floor_and_cap() -> None:
    for name in (
        "fr13_b4_clean_measure_workflow.js",
        "fr13_b4_deployment_sweep_workflow.js",
        "fr13_build_wide_shapes_b4_workflow.js",
    ):
        text = (REPO / "scripts" / name).read_text(encoding="utf-8")
        assert "98.6ms weight-read floor" not in text
        assert "113.39" not in text
        assert "93.15228665201465ms" in text
        assert "107.12512964981684ms" in text

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


# ===========================================================================
# SITE 14 (FR14 pass 118). Measurement 1 died on
#
#   fixed32 requires FR13_MANDATORY_WEIGHT_BYTES=37335563648, got 25430574256
#
# because fr14_leg3_launch_nomiddleware.sh still carried the pre-NVFP4-port
# checkpoint on all three vocabulary rows. Everything below already existed
# and would have caught it -- except that LAUNCHER above is a single path, so
# the ledger was bound into ONE of the three launcher families and the other
# two were on trust. Two of them happened to be right.
#
# The constants are literals in bash because the check runs on the host before
# docker and must not depend on a Python import. That is a fine reason to copy
# a value and no reason at all to leave the copy unbound: the authority binds
# them here instead of at runtime.
# ===========================================================================

LAUNCHER_FAMILIES = (
    REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh",
    REPO / "scripts" / "fr14_armb_leg3_launch_nomiddleware.sh",
    REPO / "scripts" / "fr14_leg3_launch_nomiddleware.sh",
)

# The two rows the ledger is the authority for, and the row it is not.
#
#   0:0      full vocabulary   -> FULL_VOCAB_* in fr13_hardware_floor_ledger
#   65536:1  root + K64        -> FIXED32_*    in fr13_hardware_floor_ledger
#   65536:0  K64, no root      -> no ledger constant exists; its only homes are
#                                 the three launchers and the floor-timer
#                                 sequence, so it is bound to THEM, and that
#                                 asymmetry is recorded rather than papered
#                                 over by inventing an authority for it here.
LEDGER_BOUND_ROWS = (
    ("0:0", FULL_VOCAB_MANDATORY_WEIGHT_BYTES,
     FULL_VOCAB_MANDATORY_WEIGHT_FLOOR_MS),
    ("65536:1", FIXED32_MANDATORY_WEIGHT_BYTES,
     FIXED32_MANDATORY_WEIGHT_FLOOR_MS),
)
UNBOUND_ROW = ("65536:0", 25254282384, 92.506528879)


def _floor_rows(text: str) -> dict[str, tuple[str, str]]:
    """The launcher's own vocabulary table: row -> (bytes, floor_ms)."""
    start = text.index(
        'case "${FR13_DRAFT_VOCAB_K:-65536}:$FR13_DRAFT_VOCAB_ROOT" in'
    )
    end = text.index("fixed32 draft-vocab floor configuration is unsupported", start)
    rows: dict[str, tuple[str, str]] = {}
    current = None
    pending: dict[str, str] = {}
    for line in text[start:end].split("\n"):
        stripped = line.strip()
        if stripped.endswith(")") and ":" in stripped and not stripped.startswith("#"):
            current = stripped[:-1]
            pending = {}
        elif stripped.startswith("_fixed32_expected_mandatory_weight_bytes="):
            pending["bytes"] = stripped.split("=", 1)[1]
        elif stripped.startswith("_fixed32_expected_weight_floor_ms="):
            pending["floor"] = stripped.split("=", 1)[1]
        if current and "bytes" in pending and "floor" in pending:
            rows[current] = (pending["bytes"], pending["floor"])
    return rows


@pytest.mark.parametrize("launcher", LAUNCHER_FAMILIES, ids=lambda p: p.name)
def test_every_launcher_family_carries_the_ledgers_floor_table(launcher) -> None:
    """The authority, bound into all three families rather than one."""
    rows = _floor_rows(launcher.read_text(encoding="utf-8"))
    for row, expected_bytes, expected_floor in LEDGER_BOUND_ROWS:
        assert row in rows, f"{launcher.name}: no {row} row in the floor table"
        got_bytes, got_floor = rows[row]
        assert int(got_bytes) == expected_bytes, (
            f"{launcher.name} row {row}: mandatory weight bytes {got_bytes} "
            f"is not the ledger's {expected_bytes} -- this is the site-14 "
            "failure, which cost three boots"
        )
        assert float(got_floor) == expected_floor, (
            f"{launcher.name} row {row}: weight floor {got_floor} is not the "
            f"ledger's {expected_floor}"
        )


@pytest.mark.parametrize("launcher", LAUNCHER_FAMILIES, ids=lambda p: p.name)
def test_the_unauthored_row_is_bound_to_the_floor_timer_sequence(launcher) -> None:
    """The 65536:0 row has no ledger constant; bind it where it does live."""
    row, expected_bytes, expected_floor = UNBOUND_ROW
    rows = _floor_rows(launcher.read_text(encoding="utf-8"))
    assert row in rows, f"{launcher.name}: no {row} row"
    got_bytes, got_floor = rows[row]
    assert int(got_bytes) == expected_bytes
    assert float(got_floor) == expected_floor
    sequence = SEQUENCE.read_text(encoding="utf-8")
    assert f"export FR13_MANDATORY_WEIGHT_BYTES={expected_bytes}" in sequence
    assert f"export FR13_WEIGHT_FLOOR_MS={expected_floor}" in sequence


def test_the_floor_table_is_identical_across_the_three_families() -> None:
    """Whole-block equality, not value-by-value.

    The six numbers were the symptom. Aligning the region wholesale moved
    FIFTEEN lines: leg3's retired-arm comment and refusal message still
    described the pre-port checkpoint ("the served lm_head is BF16 after the
    FR14 lm_head surgery") where production describes the live one ("the
    served head is NVFP4 and its K64 slice is dequantised to BF16 at boot").
    Both refuse, so no boot would ever have found it -- prose divergence is
    invisible to every gate and is exactly how a fork's understanding drifts.
    """
    blocks = []
    for launcher in LAUNCHER_FAMILIES:
        text = launcher.read_text(encoding="utf-8")
        start = text.index(
            'case "${FR13_DRAFT_VOCAB_K:-65536}:$FR13_DRAFT_VOCAB_ROOT" in'
        )
        end = text.index(
            "fixed32 draft-vocab floor configuration is unsupported", start
        )
        blocks.append(text[start:end])
    assert blocks[0] == blocks[1] == blocks[2], (
        "the vocabulary floor table is not byte-identical across the three "
        "launcher families"
    )


PRE_PORT_VALUES = (
    "37335563648", "29848731008", "27977022848",
    "136.7603064029304", "109.336011018", "102.479937172",
)


def test_the_pre_port_checkpoint_is_gone_from_every_family() -> None:
    """The fp8-era numbers, named, so they cannot come back quietly.

    Checked in EXECUTABLE positions only. fr13_fixed32_floor_timers_seq.sh
    quotes 102.479937172 in a comment on purpose -- it is the arm A figure the
    arm B floor is being compared against, and a rule that forbids naming a
    superseded number in prose would delete the reasoning along with the bug.
    A stale value in a comment is documentation; a stale value in an
    assignment is a boot that dies.
    """
    for launcher in LAUNCHER_FAMILIES + (SEQUENCE,):
        for lineno, line in enumerate(
            launcher.read_text(encoding="utf-8").split("\n"), 1
        ):
            if line.lstrip().startswith("#"):
                continue
            for stale in PRE_PORT_VALUES:
                assert stale not in line, (
                    f"{launcher.name}:{lineno} carries the pre-NVFP4-port "
                    f"value {stale} in an executable position"
                )


# ===========================================================================
# SITE 14, SECOND HALF -- RED ON PURPOSE.
#
# Aligning leg3's floor table to the ledger (above) is only half a fix, and
# landing the half without naming the other half would have turned a REFUSED
# boot into a SILENTLY WRONG measurement -- strictly worse.
#
# The ledger derives every constant in that table from ONE checkpoint:
#
#     MODEL_ROOT = Path("/models/qwen3.8-27b-nvfp4-radixark")     [arm B]
#
# Production and the armb twin serve exactly that. fr14_leg3 serves
# /models/qwen3.8-27b-nvfp4 [arm A], never invokes fr14_patch_nvfp4_lmhead.py,
# and never sets the fail-closed FR14_REQUIRE_NVFP4_LMHEAD guard that both of
# the others do. Measured on disk, the two checkpoints differ by
#
#     24,688,494,272 - 21,921,697,280 = 2,766,796,992 B  = 2.77 GB
#
# which is exactly the arm A -> arm B delta the launcher comment names ("1.83
# of the 2.77 GB this arm's floor drops is the head alone"), and the ledger
# prices at 102.479937172 -> 92.345089436 ms, i.e. 10.135 ms of floor.
#
# So with the table aligned and the checkpoint not, leg3 measures an ARM A
# wall against an ARM B floor -- about 10 ms of the floor being fiction.
#
# leg3's own charter, from its own evidence note
# (results/fr14_nvfp4_port_20260816/ablation_a_leg3.md): "a one-line-patched
# copy of the launcher -- diff vs HEAD is exactly FR13_FIXED32_MIDDLEWARE_FLAGS=""
# plus a comment". By that charter every one of these divergences is DRIFT, not
# intent -- but moving the checkpoint changes what leg3 SERVES, which is
# Mark's call, not an agent's at midnight.
#
# This test therefore FAILS, deliberately, until that call is made. It is the
# assessment's Severity-1 in executable form: measurement 1 must not be run on
# fr14_leg3 while it is red.
# ===========================================================================

LEDGER_MODEL_ROOT = "/models/qwen3.8-27b-nvfp4-radixark"


@pytest.mark.parametrize("launcher", LAUNCHER_FAMILIES, ids=lambda p: p.name)
def test_the_served_checkpoint_is_the_one_the_pinned_floor_was_derived_from(
    launcher,
) -> None:
    """A floor from one checkpoint may not be pinned against another."""
    text = launcher.read_text(encoding="utf-8")
    served = next(
        line.split("=", 1)[1].strip()
        for line in text.split("\n")
        if line.startswith("SERVED_MODEL_PATH=")
    )
    assert served == LEDGER_MODEL_ROOT, (
        f"{launcher.name} serves {served} but pins the floor table the ledger "
        f"derived from {LEDGER_MODEL_ROOT}: an arm A wall measured against an "
        "arm B floor, ~10.135 ms of which is then fiction. Either move the "
        "checkpoint (leg3's own charter says it should be production plus one "
        "middleware line) or give it its own derived floor table. Mark's call."
    )


@pytest.mark.parametrize("launcher", LAUNCHER_FAMILIES, ids=lambda p: p.name)
def test_the_nvfp4_lmhead_guard_is_armed_in_every_family(launcher) -> None:
    """The fail-closed guard that makes the arm B floor honest.

    The launcher comment states the reason exactly: "a boot that quietly fell
    back to an unquantized head would measure a wall against a floor 6.7 ms of
    which is fiction. The loader patch below is therefore mandatory and
    fail-closed, not optional." fr14_leg3 has neither the guard nor the patch
    invocation -- so on that fork the sentence is simply not true.
    """
    text = launcher.read_text(encoding="utf-8")
    assert (
        "export FR14_REQUIRE_NVFP4_LMHEAD=${FR14_REQUIRE_NVFP4_LMHEAD:-1}" in text
    ), f"{launcher.name} never arms the fail-closed NVFP4 lm_head guard"
    assert "fr14_patch_nvfp4_lmhead.py" in text, (
        f"{launcher.name} never invokes the NVFP4 lm_head loader patch"
    )

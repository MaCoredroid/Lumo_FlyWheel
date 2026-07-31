from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from scripts.fr13_hardware_floor_ledger import build_ledger


REPO = Path(__file__).resolve().parents[1]
CURATED = (
    REPO
    / "results"
    / "fr13_fixed32_b1_nsys_20260731T013952Z_curated"
)


def test_weight_ledger_reproduces_published_floor_accounting() -> None:
    generated = build_ledger()
    published = json.loads(
        (CURATED / "floor_ledger.json").read_text(encoding="ascii")
    )
    summary = json.loads(
        (CURATED / "diagnostic_summary.json").read_text(encoding="ascii")
    )

    assert published == generated
    scenarios = generated["scenarios"]
    floor = summary["floor_accounting"]

    legacy = scenarios["legacy_target_plus_verifier_head"]
    assert legacy["mandatory_weight_bytes"] == floor[
        "legacy_exact_tensor_ledger"
    ]["bytes"]
    assert legacy["mandatory_weight_floor_ms"] == floor[
        "legacy_exact_tensor_ledger"
    ]["floor_ms"]

    current = scenarios["current_one_full_plus_four_64k_draft_heads"]
    assert current["drafter_head_bytes"] == floor["current_algorithm"][
        "drafter_head_bytes"
    ]
    assert current["mandatory_weight_bytes"] == floor["current_algorithm"][
        "mandatory_weight_bytes_per_step"
    ]
    assert current["mandatory_weight_floor_ms"] == floor[
        "current_algorithm"
    ]["mandatory_weight_floor_ms"]

    root_64k = scenarios["root_64k_five_64k_draft_heads"]
    assert root_64k["mandatory_weight_bytes"] == floor[
        "root_64k_projection"
    ]["mandatory_weight_bytes_per_step"]
    assert root_64k["mandatory_weight_floor_ms"] == floor[
        "root_64k_projection"
    ]["mandatory_weight_floor_ms"]


def test_curated_attribution_binds_source_and_historical_lifecycle() -> None:
    summary = json.loads(
        (CURATED / "diagnostic_summary.json").read_text(encoding="ascii")
    )
    attribution_bytes = (CURATED / "nsys_attribution.json").read_bytes()
    attribution = json.loads(attribution_bytes)
    provenance = summary["provenance"]

    assert provenance["measured_git_sha"] == (
        "1a7a765447c8ce6068e0dd5d3a344d58ace85f2b"
    )
    assert provenance["attribution_json_sha256"] == sha256(
        attribution_bytes
    ).hexdigest()
    assert provenance["runtime_manifest_sha256"] == attribution[
        "provenance"
    ]["runtime_manifest"]["sha256"]
    assert provenance["exact4_subset_sha256"] == attribution["provenance"][
        "exact4_subset"
    ]["sha256"]
    assert (
        summary["publication"][
            "capture_exercised_terminal_ledger_snapshot_fix"
        ]
        is False
    )
    assert summary["publication"]["acceptance_valid"] is False
